"""
on_the_wild_evaluation/jepa_evaluator.py
=========================================
JEPA Embedding Prediction Error in Log Scale

Measures the internal consistency and naturalness of the try-on image
using JEPA-style self-supervised embeddings.

Input: tryon_image ONLY (no reference needed)

The JEPA evaluator computes:
  1. Self-consistency: How well different patches of the image predict each other
  2. Naturalness: Whether the embedding lies in the manifold of natural images
  3. Coherence: Whether the image parts form a coherent whole

Output: log₁₀(EPE) — Embedding Prediction Error in log scale
        Lower values indicate more internally consistent/natural images.

Log scale is used because:
  - EPE values can span several orders of magnitude
  - Makes visualization and comparison more interpretable
  - Better captures perceptual differences (Weber-Fechner law)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.jepa_metrics import JEPAMetrics


class JEPAEvaluator:
    """
    JEPA-based embedding consistency evaluator.
    
    Input: tryon_image ONLY
    
    Computes Embedding Prediction Error (EPE) in log scale:
        log_EPE = log₁₀(self_prediction_error)
    
    Lower log_EPE indicates the try-on result is more internally
    consistent and lies in the manifold of natural images.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: torch device string
        """
        self.device = device
        self._metric = JEPAMetrics(device=device)
        self._epe_values: List[float] = []
        self._log_epe_values: List[float] = []
    
    def evaluate_batch(
        self,
        tryon_images: torch.Tensor,
    ) -> Dict[str, List[float]]:
        """
        Evaluate embedding prediction error for a batch of try-on images.
        
        Args:
            tryon_images: (B, 3, H, W) float32 [0, 1] - generated try-on results
        
        Returns:
            Dict with:
                "epe": List of raw EPE values
                "log_epe": List of log₁₀(EPE) values
        """
        # Compute self-consistency EPE (using only tryon images)
        epe_list = self._metric.compute_self_consistency_batch(tryon_images)
        
        # Convert to log scale (add small epsilon to avoid log(0))
        eps = 1e-10
        log_epe_list = [math.log10(max(e, eps)) for e in epe_list]
        
        self._epe_values.extend(epe_list)
        self._log_epe_values.extend(log_epe_list)
        
        # Accumulate embeddings for trace computation
        self._metric.update_embeddings(tryon_images)
        
        return {
            "epe": epe_list,
            "log_epe": log_epe_list,
        }
    
    def evaluate_single(
        self,
        tryon_image: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate a single try-on image."""
        if tryon_image.dim() == 3:
            tryon_image = tryon_image.unsqueeze(0)
        
        result = self.evaluate_batch(tryon_image)
        return {
            "epe": result["epe"][0],
            "log_epe": result["log_epe"][0],
        }
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all evaluated images.
        
        Returns:
            Dict with mean/std for EPE, log_EPE, and embedding trace
        """
        if not self._epe_values:
            return {}
        
        epe_arr = np.array(self._epe_values)
        log_epe_arr = np.array(self._log_epe_values)
        
        summary = {
            "epe_mean": float(np.mean(epe_arr)),
            "epe_std": float(np.std(epe_arr)),
            "epe_median": float(np.median(epe_arr)),
            "log_epe_mean": float(np.mean(log_epe_arr)),
            "log_epe_std": float(np.std(log_epe_arr)),
            "log_epe_median": float(np.median(log_epe_arr)),
            "embedding_trace": self._metric.compute_embedding_trace(),
            "n_samples": len(self._epe_values),
        }
        return summary
    
    def reset(self):
        """Clear accumulated results."""
        self._epe_values = []
        self._log_epe_values = []
        self._metric.reset()
    
    @property
    def all_log_epe(self) -> List[float]:
        """Return all log₁₀(EPE) values."""
        return self._log_epe_values
    
    @property
    def all_epe(self) -> List[float]:
        """Return all raw EPE values."""
        return self._epe_values
