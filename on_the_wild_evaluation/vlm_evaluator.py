"""
on_the_wild_evaluation/vlm_evaluator.py
========================================
VLM-based Plausibility Score for In-the-Wild Try-On.

Wraps the VLMScoreMetric from metrics/vlm_score.py with additional
utilities for in-the-wild evaluation scenarios.

Input: person_image + cloth_image + tryon_image (ALL THREE)

The VLM analyzes the try-on result to assess realism and integration
based on 7 key aspects:

  1. Photorealism - Fabric texture, wrinkles, shading naturalness
  2. Lighting Consistency - Matching scene lighting, shadows, highlights
  3. Color/Intensity Matching - Exposure and brightness consistency
  4. Seamless Blending - No cut-and-paste artifacts, halos, sharp edges
  5. Body Alignment - Clothing follows body pose and geometry
  6. Occlusion Handling - Correct interaction with arms, hair, objects
  7. Global Scene Consistency - Result looks captured in same photograph

Output:
  vlm_score : float ∈ [0, 1]  (continuous, higher = better)
  reason    : str             (brief explanation of score)

Score Interpretation:
  1.0       : Perfect photorealistic try-on
  0.8-0.99  : Very realistic with minor imperfections
  0.6-0.79  : Generally believable but noticeable artifacts
  0.4-0.59  : Clearly synthetic in some areas
  0.2-0.39  : Strong artifacts, incorrect lighting
  0.0-0.19  : Completely unrealistic
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
    
    Output is a continuous score between 0 and 1.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        **kwargs,  # Accept legacy kwargs for backward compatibility
    ):
        """
        Args:
            device: torch device string
        """
        self.device = device
        self._metric = VLMScoreMetric(device=device)
        self._results: List[Dict[str, any]] = []
    
    def evaluate_batch(
        self,
        tryon_images: torch.Tensor,
        person_images: Optional[torch.Tensor] = None,
        cloth_images: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, any]]:
        """
        Evaluate a batch of try-on results.
        
        Args:
            tryon_images: (B, 3, H, W) float32 [0, 1] - generated try-on images
            person_images: (B, 3, H, W) - original person images (context)
            cloth_images: (B, 3, H, W) - original cloth images (context)
        
        Returns:
            List of dicts with keys:
                vlm_score: float ∈ [0, 1] - continuous plausibility score
                reason: str - brief explanation of the score
        """
        # Pass all three to the metric for comprehensive evaluation
        results = self._metric.compute_batch(
            tryon_images, 
            person_images=person_images,
            cloth_images=cloth_images,
        )
        self._results.extend(results)
        return results
    
    def evaluate_single(
        self,
        tryon_image: torch.Tensor,
        person_image: Optional[torch.Tensor] = None,
        cloth_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Evaluate a single try-on image.
        
        Args:
            tryon_image: (3, H, W) - generated try-on result
            person_image: (3, H, W) - original person image (context)
            cloth_image: (3, H, W) - original cloth image (context)
        
        Returns:
            Dict with:
                vlm_score: float ∈ [0, 1] - continuous plausibility score
                reason: str - brief explanation of the score
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
            Dict with mean/std for VLM score
        """
        if not self._results:
            return {}
        
        scores = np.array([r["vlm_score"] for r in self._results])
        
        summary = {
            "vlm_score_mean": float(np.mean(scores)),
            "vlm_score_std": float(np.std(scores)),
            "vlm_score_min": float(np.min(scores)),
            "vlm_score_max": float(np.max(scores)),
            "vlm_score_median": float(np.median(scores)),
            "n_samples": len(self._results),
        }
        return summary
    
    def reset(self):
        """Clear accumulated results."""
        self._results = []
    
    @property
    def all_results(self) -> List[Dict[str, any]]:
        """Return all accumulated per-image results."""
        return self._results
    
    @property
    def all_scores(self) -> List[float]:
        """Return all VLM scores as a list."""
        return [r["vlm_score"] for r in self._results]
