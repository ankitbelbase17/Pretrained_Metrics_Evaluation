"""
on_the_wild_evaluation/pose_evaluator.py
=========================================
Pose Consistency Metric for In-the-Wild Try-On Evaluation

Measures how well the person's pose is preserved in the try-on result
compared to the original input image. This is critical for in-the-wild
scenarios where poses are diverse and unconstrained.

Metrics computed:
  - Keypoint MSE: Mean squared error of normalized keypoints
  - Keypoint PCK@0.1: Percentage of Correct Keypoints within 10% of bbox
  - Joint Angle Consistency: Preservation of limb angles
  - Pose Vector Cosine Similarity: Overall pose structure similarity

Pretrained Model: ViTPose-B (mmpose) or HRNet-W32 (fallback)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrained_metrics.metrics.m1_pose import (
    _KeypointExtractor, _normalise_pose, _joint_angle, TRIPLET_IDX
)


class PoseConsistencyEvaluator:
    """
    Evaluates pose preservation between input person and try-on result.
    
    Key insight: A good try-on should NOT change the person's pose.
    Arms, legs, torso orientation should remain identical.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: torch device string
        """
        self.device = device
        self._kp_extractor = _KeypointExtractor(device=device)
        
        # Accumulated results
        self._kp_mse: List[float] = []
        self._pck_scores: List[float] = []
        self._angle_errors: List[float] = []
        self._cosine_sims: List[float] = []
    
    def _extract_pose_features(
        self, 
        imgs: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pose features from images.
        
        Returns:
            kps_raw: (B, 17, 2) raw keypoints
            kps_norm: (B, 17, 2) normalized keypoints
            angles: (B, 8) joint angles
        """
        kps_raw = self._kp_extractor(imgs)  # (B, 17, 2)
        B = kps_raw.shape[0]
        
        kps_norm_list = []
        angles_list = []
        
        for i in range(B):
            kp_norm, valid = _normalise_pose(kps_raw[i:i+1])
            kp_norm = kp_norm[0] if valid[0] else np.zeros((17, 2))
            kps_norm_list.append(kp_norm)
            
            # Compute joint angles
            angles = []
            for ia, ib, ic in TRIPLET_IDX:
                ang = _joint_angle(kp_norm[ia], kp_norm[ib], kp_norm[ic])
                angles.append(ang if not math.isnan(ang) else 0.0)
            angles_list.append(angles)
        
        return kps_raw, np.array(kps_norm_list), np.array(angles_list)
    
    def evaluate_batch(
        self,
        person_images: torch.Tensor,
        tryon_images: torch.Tensor,
        pck_threshold: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Evaluate pose consistency for a batch.
        
        Args:
            person_images: (B, 3, H, W) input person images
            tryon_images: (B, 3, H, W) generated try-on results
            pck_threshold: Threshold for PCK metric (default 0.1 = 10% of bbox)
        
        Returns:
            Dict with per-image metrics:
                "kp_mse": Keypoint MSE
                "pck": Percentage of Correct Keypoints
                "angle_error": Mean joint angle error (degrees)
                "cosine_sim": Pose vector cosine similarity
        """
        # Extract poses
        _, person_kps, person_angles = self._extract_pose_features(person_images)
        _, tryon_kps, tryon_angles = self._extract_pose_features(tryon_images)
        
        B = person_kps.shape[0]
        results = {
            "kp_mse": [],
            "pck": [],
            "angle_error": [],
            "cosine_sim": [],
        }
        
        for i in range(B):
            # Keypoint MSE
            mse = np.mean((person_kps[i] - tryon_kps[i]) ** 2)
            results["kp_mse"].append(float(mse))
            self._kp_mse.append(float(mse))
            
            # PCK (Percentage of Correct Keypoints)
            # Threshold relative to normalized coordinates [0, 1]
            distances = np.linalg.norm(person_kps[i] - tryon_kps[i], axis=1)
            pck = np.mean(distances < pck_threshold) * 100  # percentage
            results["pck"].append(float(pck))
            self._pck_scores.append(float(pck))
            
            # Joint angle error (in degrees)
            angle_diff = np.abs(person_angles[i] - tryon_angles[i])
            angle_error = np.mean(np.degrees(angle_diff))
            results["angle_error"].append(float(angle_error))
            self._angle_errors.append(float(angle_error))
            
            # Cosine similarity of flattened pose vectors
            p_vec = person_kps[i].flatten()
            t_vec = tryon_kps[i].flatten()
            p_norm = np.linalg.norm(p_vec) + 1e-8
            t_norm = np.linalg.norm(t_vec) + 1e-8
            cos_sim = np.dot(p_vec, t_vec) / (p_norm * t_norm)
            results["cosine_sim"].append(float(cos_sim))
            self._cosine_sims.append(float(cos_sim))
        
        return results
    
    def evaluate_single(
        self,
        person_image: torch.Tensor,
        tryon_image: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate a single image pair."""
        if person_image.dim() == 3:
            person_image = person_image.unsqueeze(0)
        if tryon_image.dim() == 3:
            tryon_image = tryon_image.unsqueeze(0)
        
        result = self.evaluate_batch(person_image, tryon_image)
        return {k: v[0] for k, v in result.items()}
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all evaluated images.
        """
        if not self._kp_mse:
            return {}
        
        return {
            "kp_mse_mean": float(np.mean(self._kp_mse)),
            "kp_mse_std": float(np.std(self._kp_mse)),
            "pck_mean": float(np.mean(self._pck_scores)),
            "pck_std": float(np.std(self._pck_scores)),
            "angle_error_mean": float(np.mean(self._angle_errors)),
            "angle_error_std": float(np.std(self._angle_errors)),
            "cosine_sim_mean": float(np.mean(self._cosine_sims)),
            "cosine_sim_std": float(np.std(self._cosine_sims)),
            "n_samples": len(self._kp_mse),
        }
    
    def reset(self):
        """Clear accumulated results."""
        self._kp_mse = []
        self._pck_scores = []
        self._angle_errors = []
        self._cosine_sims = []
