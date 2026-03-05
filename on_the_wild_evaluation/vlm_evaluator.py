"""
on_the_wild_evaluation/vlm_evaluator.py
========================================
VLM-based Multi-Dimensional Plausibility Score for In-the-Wild Try-On.

Wraps the VLMScoreMetric from metrics/vlm_score.py with additional
utilities for in-the-wild evaluation scenarios.

Input: person_image + cloth_image + tryon_image (ALL THREE)

The VLM analyzes the try-on result in context of what the input person
and garment looked like to assess plausibility.

Sub-scores (continuous 0-1 scale, higher = better):
  S1: Garment Fidelity (texture, colour, pattern preservation vs cloth_image)
  S2: Geometric Naturalness (fabric draping, folding)
  S3: Identity & Body Preservation (face, pose, proportions vs person_image)
  S4: Scene Coherence (lighting, shadows, background integration)

Final Score: VLM_score = 0.30*S1 + 0.25*S2 + 0.25*S3 + 0.20*S4

All scores are normalized to [0, 1] continuous range.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.vlm_score import VLMScoreMetric


class VLMEvaluator:
    """
    VLM-based evaluation for in-the-wild virtual try-on.
    
    This evaluator does NOT require ground truth images.
    It assesses the plausibility and quality of try-on results
    using vision-language model understanding.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            device: torch device string
            weights: Optional custom weights for sub-scores
                     Default: {"s1": 0.30, "s2": 0.25, "s3": 0.25, "s4": 0.20}
        """
        self.device = device
        self.weights = weights or {"s1": 0.30, "s2": 0.25, "s3": 0.25, "s4": 0.20}
        self._metric = VLMScoreMetric(device=device)
        self._results: List[Dict[str, float]] = []
    
    def evaluate_batch(
        self,
        tryon_images: torch.Tensor,
        person_images: Optional[torch.Tensor] = None,
        cloth_images: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, float]]:
        """
        Evaluate a batch of try-on results using ALL THREE inputs.
        
        Args:
            tryon_images: (B, 3, H, W) float32 [0, 1] - generated try-on images
            person_images: (B, 3, H, W) - original person images (for identity check)
            cloth_images: (B, 3, H, W) - original cloth images (for fidelity check)
        
        Returns:
            List of dicts with keys: s1, s2, s3, s4, vlm_score
        """
        # Pass all three to the metric for comprehensive evaluation
        results = self._metric.compute_batch(
            tryon_images, 
            person_images=person_images,
            cloth_images=cloth_images,
        )
        # Normalize scores from [1, 10] to [0, 1] continuous range
        normalized_results = []
        for r in results:
            norm_r = {
                "s1": (r["s1"] - 1.0) / 9.0,  # Map [1, 10] -> [0, 1]
                "s2": (r["s2"] - 1.0) / 9.0,
                "s3": (r["s3"] - 1.0) / 9.0,
                "s4": (r["s4"] - 1.0) / 9.0,
                "vlm_score": (r["vlm_score"] - 1.0) / 9.0,
            }
            normalized_results.append(norm_r)
        self._results.extend(normalized_results)
        return normalized_results
    
    def evaluate_single(
        self,
        tryon_image: torch.Tensor,
        person_image: Optional[torch.Tensor] = None,
        cloth_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single try-on image using all three inputs.
        
        Args:
            tryon_image: (3, H, W) - generated try-on result
            person_image: (3, H, W) - original person image
            cloth_image: (3, H, W) - original cloth image
        """
        if tryon_image.dim() == 3:
            tryon_image = tryon_image.unsqueeze(0)
        if person_image is not None and person_image.dim() == 3:
            person_image = person_image.unsqueeze(0)
        if cloth_image is not None and cloth_image.dim() == 3:
            cloth_image = cloth_image.unsqueeze(0)
        return self.evaluate_batch(tryon_image, person_image, cloth_image)[0]
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all evaluated images.
        
        Returns:
            Dict with mean/std for each sub-score and overall VLM score
        """
        if not self._results:
            return {}
        
        arr = {k: np.array([r[k] for r in self._results]) 
               for k in ["s1", "s2", "s3", "s4", "vlm_score"]}
        
        summary = {}
        for k, v in arr.items():
            summary[f"{k}_mean"] = float(np.mean(v))
            summary[f"{k}_std"] = float(np.std(v))
        
        summary["n_samples"] = len(self._results)
        return summary
    
    def reset(self):
        """Clear accumulated results."""
        self._results = []
    
    @property
    def all_results(self) -> List[Dict[str, float]]:
        """Return all accumulated per-image results."""
        return self._results
