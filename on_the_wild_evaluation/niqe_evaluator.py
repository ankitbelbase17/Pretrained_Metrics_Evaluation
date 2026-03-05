"""
on_the_wild_evaluation/niqe_evaluator.py
=========================================
NIQE (Natural Image Quality Evaluator) - No-Reference Image Quality

NIQE is a completely blind/no-reference image quality metric that
measures how "natural" an image looks based on statistical regularities
of natural scenes.

Key properties:
  - No reference image needed (ideal for in-the-wild)
  - Lower score = better quality (more natural)
  - Based on multivariate Gaussian model of natural scene statistics
  - Trained on pristine natural images

Reference:
  Mittal, A., Soundararajan, R., & Bovik, A. C. (2013).
  "Making a 'Completely Blind' Image Quality Analyzer."
  IEEE Signal Processing Letters.

Pretrained Model: Uses pre-computed natural scene statistics (NSS)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class NIQEEvaluator:
    """
    NIQE-based natural image quality evaluator.
    
    Lower NIQE scores indicate more natural-looking images.
    Typical ranges:
      - High quality natural photos: 2-4
      - Moderate quality: 4-6
      - Low quality / artifacts: 6+
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: torch device string (NIQE is CPU-based but accepts GPU tensors)
        """
        self.device = device
        self._niqe_fn = None
        self._backend = "none"
        self._load_backend()
        
        # Accumulated results
        self._scores: List[float] = []
    
    def _load_backend(self):
        """Load NIQE computation backend."""
        # Try pyiqa (comprehensive IQA library)
        try:
            import pyiqa
            self._niqe_fn = pyiqa.create_metric("niqe", device=self.device)
            self._backend = "pyiqa"
            print("[NIQEEvaluator] Using pyiqa NIQE implementation.")
            return
        except Exception as e:
            print(f"[NIQEEvaluator] pyiqa unavailable ({e}).")
        
        # Try skvideo (has NIQE implementation)
        try:
            from skvideo.measure import niqe as skvideo_niqe
            self._niqe_fn = skvideo_niqe
            self._backend = "skvideo"
            print("[NIQEEvaluator] Using skvideo NIQE implementation.")
            return
        except Exception as e:
            print(f"[NIQEEvaluator] skvideo unavailable ({e}).")
        
        # Try image-quality (another IQA package)
        try:
            from image_quality.brisque import BRISQUE  # Similar blind IQA
            # NIQE not directly available, fall through
            raise ImportError("NIQE not in image_quality")
        except Exception:
            pass
        
        # Fallback: simplified NIQE approximation
        print("[NIQEEvaluator] No NIQE backend found. Using simplified NSS approximation.")
        self._backend = "approx"
    
    def _compute_niqe_approx(self, img: np.ndarray) -> float:
        """
        Simplified NIQE approximation using basic natural scene statistics.
        
        This is a rough approximation; install pyiqa for accurate NIQE.
        """
        from scipy import ndimage
        from scipy.special import gamma
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img
        
        gray = gray.astype(np.float64)
        
        # Local mean and variance (MSCN normalization)
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        mu = ndimage.convolve(gray, kernel, mode='nearest')
        mu_sq = ndimage.convolve(gray ** 2, kernel, mode='nearest')
        sigma = np.sqrt(np.maximum(mu_sq - mu ** 2, 0)) + 1e-7
        
        # MSCN coefficients
        mscn = (gray - mu) / sigma
        
        # Fit generalized Gaussian distribution (GGD) to MSCN
        mscn_flat = mscn.flatten()
        mscn_flat = mscn_flat[np.isfinite(mscn_flat)]
        
        if len(mscn_flat) < 100:
            return 10.0  # Default poor score for very small images
        
        # Estimate GGD shape parameter (simplified)
        mean_abs = np.mean(np.abs(mscn_flat))
        variance = np.var(mscn_flat)
        
        if variance < 1e-10:
            return 10.0
        
        rho = variance / (mean_abs ** 2 + 1e-10)
        
        # Natural images typically have rho close to certain values
        # Deviation from this indicates unnaturalness
        natural_rho = 1.5  # Approximate for Laplacian-like distributions
        niqe_approx = np.abs(rho - natural_rho) * 5 + 3  # Scale to typical NIQE range
        
        return float(np.clip(niqe_approx, 1, 15))
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
    ) -> List[float]:
        """
        Evaluate NIQE for a batch of images.
        
        Args:
            images: (B, 3, H, W) float32 [0, 1]
        
        Returns:
            List of NIQE scores (lower = better)
        """
        B = images.shape[0]
        scores = []
        
        for i in range(B):
            img = images[i]
            
            if self._backend == "pyiqa":
                # pyiqa expects (B, C, H, W) in [0, 1]
                score = self._niqe_fn(img.unsqueeze(0).to(self.device))
                score = float(score.item())
            
            elif self._backend == "skvideo":
                # skvideo expects (H, W, C) uint8 numpy array
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                score = float(self._niqe_fn(img_np))
            
            else:  # approx
                img_np = img.permute(1, 2, 0).cpu().numpy()
                score = self._compute_niqe_approx(img_np)
            
            scores.append(score)
            self._scores.append(score)
        
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
            "niqe_mean": float(np.mean(arr)),
            "niqe_std": float(np.std(arr)),
            "niqe_median": float(np.median(arr)),
            "niqe_min": float(np.min(arr)),
            "niqe_max": float(np.max(arr)),
            "n_samples": len(self._scores),
            "backend": self._backend,
        }
    
    def reset(self):
        """Clear accumulated results."""
        self._scores = []
    
    @property
    def all_scores(self) -> List[float]:
        """Return all NIQE scores."""
        return self._scores
