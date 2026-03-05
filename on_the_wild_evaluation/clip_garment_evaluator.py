"""
on_the_wild_evaluation/clip_garment_evaluator.py
=================================================
CLIP Garment Score for In-the-Wild Try-On Evaluation

Measures how well the garment appearance in the try-on result matches
the original garment image using CLIP embeddings.

Metrics computed:
  - Garment-TryOn Similarity: Cosine similarity between garment and try-on CLIP embeddings
  - Garment Region Similarity: CLIP similarity focused on the garment region
  - Cross-Modal Score: Text-guided garment assessment (optional)

Pretrained Model: CLIP ViT-B/32 (OpenAI)

This metric is reference-free in the sense that it doesn't need a
ground-truth try-on image, only the input garment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder


class CLIPGarmentEvaluator:
    """
    CLIP-based garment fidelity evaluator.
    
    Measures how well the garment's visual characteristics (texture, color,
    pattern) are preserved in the try-on result.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: torch device string
        """
        self.device = device
        self._encoder = _GarmentEncoder(device=device)
        
        # Accumulated results
        self._cosine_sims: List[float] = []
        self._l2_distances: List[float] = []
    
    def _encode(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Encode images to CLIP embeddings.
        
        Args:
            imgs: (B, 3, H, W) float32 [0, 1]
        
        Returns:
            (B, D) normalized embeddings
        """
        embs = self._encoder(imgs)  # (B, D)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return embs / norms
    
    def evaluate_batch(
        self,
        garment_images: torch.Tensor,
        tryon_images: torch.Tensor,
    ) -> Dict[str, List[float]]:
        """
        Evaluate garment preservation for a batch.
        
        Args:
            garment_images: (B, 3, H, W) original garment images
            tryon_images: (B, 3, H, W) generated try-on results
        
        Returns:
            Dict with per-image metrics:
                "cosine_sim": Cosine similarity (higher = better, max 1.0)
                "l2_distance": L2 distance in embedding space (lower = better)
                "clip_score": Scaled cosine similarity as a score [0, 100]
        """
        garment_embs = self._encode(garment_images)  # (B, D)
        tryon_embs = self._encode(tryon_images)      # (B, D)
        
        B = garment_embs.shape[0]
        results = {
            "cosine_sim": [],
            "l2_distance": [],
            "clip_score": [],
        }
        
        for i in range(B):
            # Cosine similarity
            cos_sim = np.dot(garment_embs[i], tryon_embs[i])
            results["cosine_sim"].append(float(cos_sim))
            self._cosine_sims.append(float(cos_sim))
            
            # L2 distance
            l2_dist = np.linalg.norm(garment_embs[i] - tryon_embs[i])
            results["l2_distance"].append(float(l2_dist))
            self._l2_distances.append(float(l2_dist))
            
            # CLIP Score: scale cosine similarity to [0, 100]
            # Typical cosine similarities range from ~0.5 to ~0.95 for similar images
            clip_score = max(0, (cos_sim + 1) / 2 * 100)  # map [-1, 1] to [0, 100]
            results["clip_score"].append(float(clip_score))
        
        return results
    
    def evaluate_single(
        self,
        garment_image: torch.Tensor,
        tryon_image: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate a single image pair."""
        if garment_image.dim() == 3:
            garment_image = garment_image.unsqueeze(0)
        if tryon_image.dim() == 3:
            tryon_image = tryon_image.unsqueeze(0)
        
        result = self.evaluate_batch(garment_image, tryon_image)
        return {k: v[0] for k, v in result.items()}
    
    def evaluate_batch_with_text(
        self,
        tryon_images: torch.Tensor,
        text_descriptions: List[str],
    ) -> List[float]:
        """
        Evaluate try-on images against text descriptions of the garment.
        
        Requires CLIP model with text encoder (openai-clip package).
        
        Args:
            tryon_images: (B, 3, H, W) generated try-on results
            text_descriptions: List of garment descriptions
        
        Returns:
            List of text-image similarity scores
        """
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Encode text
            text_tokens = clip.tokenize(text_descriptions).to(self.device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Encode images
            # Preprocess expects PIL images, so we convert
            from PIL import Image
            import torchvision.transforms.functional as TF
            
            pil_images = [TF.to_pil_image(img.cpu()) for img in tryon_images]
            image_inputs = torch.stack([preprocess(img) for img in pil_images]).to(self.device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarities = (image_features * text_features).sum(dim=-1)
            return similarities.cpu().tolist()
            
        except ImportError:
            print("[CLIPGarmentEvaluator] Text evaluation requires 'openai-clip' package")
            return [0.0] * len(tryon_images)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all evaluated images."""
        if not self._cosine_sims:
            return {}
        
        return {
            "cosine_sim_mean": float(np.mean(self._cosine_sims)),
            "cosine_sim_std": float(np.std(self._cosine_sims)),
            "l2_distance_mean": float(np.mean(self._l2_distances)),
            "l2_distance_std": float(np.std(self._l2_distances)),
            "clip_score_mean": float(np.mean([(c + 1) / 2 * 100 for c in self._cosine_sims])),
            "n_samples": len(self._cosine_sims),
        }
    
    def reset(self):
        """Clear accumulated results."""
        self._cosine_sims = []
        self._l2_distances = []
