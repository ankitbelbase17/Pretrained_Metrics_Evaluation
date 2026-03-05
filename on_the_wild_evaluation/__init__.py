"""
on_the_wild_evaluation/__init__.py
===================================
In-the-Wild Virtual Try-On Evaluation Suite

This module provides reference-free and perceptual quality metrics
suitable for evaluating virtual try-on results on unconstrained,
in-the-wild images where ground truth is unavailable.

Triplet Structure:
  (person_image, cloth_image, tryon_image)

Metric → Input Mapping:
───────────────────────
  - VLM Score           : person + cloth + tryon (ALL THREE)
  - Pose Consistency    : person + tryon (pose preservation)
  - NIQE                : tryon ONLY (no-reference quality)
  - MUSIQ               : tryon ONLY (no-reference quality)
  - JEPA                : tryon ONLY (self-consistency)
  - CLIP Garment        : cloth + tryon (garment fidelity)

All metrics are designed for:
  - No ground-truth required (suitable for in-the-wild)
  - Batch processing
  - GPU acceleration where available
"""

from .vlm_evaluator import VLMEvaluator
from .jepa_evaluator import JEPAEvaluator
from .pose_evaluator import PoseConsistencyEvaluator
from .clip_garment_evaluator import CLIPGarmentEvaluator
from .niqe_evaluator import NIQEEvaluator
from .musiq_evaluator import MUSIQEvaluator
from .dataloaders import TripletDataLoader, TripletDataset, collate_triplets
from .run_evaluation import run_wild_evaluation, WildEvaluationSuite

__all__ = [
    "VLMEvaluator",
    "JEPAEvaluator",
    "PoseConsistencyEvaluator",
    "CLIPGarmentEvaluator",
    "NIQEEvaluator",
    "MUSIQEvaluator",
    "TripletDataLoader",
    "TripletDataset",
    "collate_triplets",
    "run_wild_evaluation",
    "WildEvaluationSuite",
]
