"""
on_the_wild_evaluation/musiq_evaluator.py
==========================================
MUSIQ (Multi-Scale Image Quality Transformer) - No-Reference IQA

MUSIQ is a state-of-the-art no-reference image quality assessment model
that uses a multi-scale vision transformer to predict quality scores
without needing a reference image.

Key properties:
  - No reference image needed (ideal for in-the-wild)
  - Higher score = better quality (typically 0-100 scale)
  - Handles variable resolution inputs (no resizing needed)
  - Trained on multiple IQA datasets (KonIQ-10k, PaQ-2-PiQ, SPAQ)

Reference:
  Ke, J., et al. (2021).
  "MUSIQ: Multi-scale Image Quality Transformer."
  ICCV 2021.

Pretrained Model: MUSIQ (timm / HuggingFace)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class MUSIQEvaluator:
    """
    MUSIQ-based image quality evaluator.
    
    Higher MUSIQ scores indicate better perceptual quality.
    Typical ranges (when trained on KonIQ-10k):
      - High quality: 70-100
      - Moderate quality: 40-70
      - Low quality: 0-40
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: torch device string
        """
        self.device = device
        self._model = None
        self._backend = "none"
        self._load_backend()
        
        # Accumulated results
        self._scores: List[float] = []
    
    def _load_backend(self):
        """Load MUSIQ computation backend."""
        # Try pyiqa (comprehensive IQA library with MUSIQ)
        try:
            import pyiqa
            self._model = pyiqa.create_metric("musiq", device=self.device)
            self._backend = "pyiqa"
            print("[MUSIQEvaluator] Using pyiqa MUSIQ implementation.")
            return
        except Exception as e:
            print(f"[MUSIQEvaluator] pyiqa MUSIQ unavailable ({e}).")
        
        # Try pyiqa with alternative MUSIQ variant
        try:
            import pyiqa
            self._model = pyiqa.create_metric("musiq-koniq", device=self.device)
            self._backend = "pyiqa-koniq"
            print("[MUSIQEvaluator] Using pyiqa MUSIQ-KonIQ implementation.")
            return
        except Exception as e:
            print(f"[MUSIQEvaluator] pyiqa MUSIQ-KonIQ unavailable ({e}).")
        
        # Try timm-based MUSIQ (if available as a model)
        try:
            import timm
            # MUSIQ may not be in standard timm, but worth checking
            if "musiq" in timm.list_models(pretrained=True):
                self._model = timm.create_model("musiq", pretrained=True).to(self.device).eval()
                self._backend = "timm"
                print("[MUSIQEvaluator] Using timm MUSIQ implementation.")
                return
        except Exception:
            pass
        
        # Fallback: Use CLIP-IQA as proxy (also no-reference, transformer-based)
        try:
            import pyiqa
            self._model = pyiqa.create_metric("clipiqa", device=self.device)
            self._backend = "clipiqa"
            print("[MUSIQEvaluator] MUSIQ unavailable, using CLIP-IQA as proxy.")
            return
        except Exception as e:
            print(f"[MUSIQEvaluator] CLIP-IQA unavailable ({e}).")
        
        # Final fallback: ViT-based quality proxy
        print("[MUSIQEvaluator] No IQA backend found. Using ViT embedding variance as proxy.")
        self._backend = "vit_proxy"
        self._load_vit_proxy()
    
    def _load_vit_proxy(self):
        """Load ViT as a quality proxy."""
        try:
            import timm
            self._model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            ).to(self.device).eval()
            print("[MUSIQEvaluator] Using ViT embedding quality proxy.")
        except Exception as e:
            print(f"[MUSIQEvaluator] ViT proxy failed ({e}).")
            self._model = None
    
    def _compute_vit_quality_proxy(self, img: torch.Tensor) -> float:
        """
        Compute quality proxy using ViT embedding statistics.
        
        Hypothesis: High-quality images have more structured/consistent
        embeddings, while low-quality images have higher entropy/variance.
        """
        if self._model is None:
            return 50.0  # Neutral score
        
        import torchvision.transforms as T
        
        # Normalize and resize
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img_resized = F.interpolate(
            img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )
        img_norm = normalize(img_resized.squeeze(0)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self._model.forward_features(img_norm)  # (1, N+1, D)
            # Use patch token statistics
            patch_tokens = features[:, 1:, :]  # (1, N, D)
            
            # Quality proxy: inverse of token variance (more uniform = higher quality)
            token_var = patch_tokens.var(dim=1).mean().item()
            # Also consider CLS token norm (well-formed images have consistent norms)
            cls_norm = features[:, 0, :].norm().item()
            
            # Combine into score [0, 100]
            # Lower variance and higher CLS norm = better quality
            score = 100 / (1 + token_var * 10) * min(cls_norm / 10, 1.5)
            score = np.clip(score, 0, 100)
        
        return float(score)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
    ) -> List[float]:
        """
        Evaluate MUSIQ for a batch of images.
        
        Args:
            images: (B, 3, H, W) float32 [0, 1]
        
        Returns:
            List of MUSIQ scores (higher = better)
        """
        B = images.shape[0]
        scores = []
        
        if self._backend in ["pyiqa", "pyiqa-koniq", "clipiqa"]:
            # pyiqa handles batches
            with torch.no_grad():
                batch_scores = self._model(images.to(self.device))
                if batch_scores.dim() == 0:
                    scores = [float(batch_scores.item())] * B
                else:
                    scores = batch_scores.cpu().tolist()
                    if len(scores) != B:
                        # Some pyiqa metrics return single value
                        scores = [scores[0]] * B if len(scores) == 1 else scores
        
        elif self._backend == "vit_proxy":
            for i in range(B):
                score = self._compute_vit_quality_proxy(images[i])
                scores.append(score)
        
        else:
            # No backend available
            scores = [50.0] * B  # Neutral scores
        
        self._scores.extend(scores)
        return scores
    
    def evaluate_single(self, image: torch.Tensor) -> float:
        """Evaluate a single image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.evaluate_batch(image)[0]
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all evaluated images."""
        if not self._scores:
            return {}
        
        arr = np.array(self._scores)
        return {
            "musiq_mean": float(np.mean(arr)),
            "musiq_std": float(np.std(arr)),
            "musiq_median": float(np.median(arr)),
            "musiq_min": float(np.min(arr)),
            "musiq_max": float(np.max(arr)),
            "n_samples": len(self._scores),
            "backend": self._backend,
        }
    
    def reset(self):
        """Clear accumulated results."""
        self._scores = []
    
    @property
    def all_scores(self) -> List[float]:
        """Return all MUSIQ scores."""
        return self._scores
