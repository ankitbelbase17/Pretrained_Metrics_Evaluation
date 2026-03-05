"""
metrics/pose_error.py
======================
Pose-error metric for virtual try-on evaluation.

Computes per-image L2 pose distance between predicted and ground-truth
images by extracting 2-D keypoints (via HRNet / stub) and measuring the
mean normalised Euclidean error across the 17 COCO joints.

Interface
---------
    metric = PoseErrorMetric(device="cuda")
    errors = metric.compute_batch(pred, gt)   # list[float], len = B
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# Re-use the keypoint extractor and normalisation from the pretrained_metrics
# module so there is a single source of truth for pose estimation.
try:
    from pretrained_metrics.metrics.m1_pose import (
        _KeypointExtractor,
        _normalise_pose,
    )
    _HAS_KEYPOINT = True
except ImportError:
    _HAS_KEYPOINT = False


class PoseErrorMetric:
    """
    Per-image pose error between *pred* and *gt* images.

    Steps:
      1. Extract 17-joint COCO keypoints from both images.
      2. Normalise each skeleton (translate to hip centre, scale by torso).
      3. Compute mean per-joint L2 distance.

    Returns one scalar per image (lower is better).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        if _HAS_KEYPOINT:
            self.extractor = _KeypointExtractor(device=device)
        else:
            self.extractor = None
            print("[PoseErrorMetric] KeypointExtractor unavailable; "
                  "returning NaN for all samples.")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_batch(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> List[float]:
        """
        Parameters
        ----------
        pred, gt : (B, 3, H, W) float32 tensors in [0, 1]

        Returns
        -------
        List of B floats — normalised mean-per-joint L2 error.
        """
        B = pred.shape[0]
        if self.extractor is None:
            return [float("nan")] * B

        kps_pred = self.extractor(pred)   # (B, 17, 2)
        kps_gt   = self.extractor(gt)     # (B, 17, 2)

        kps_pred_n, valid_pred = _normalise_pose(kps_pred)
        kps_gt_n,   valid_gt   = _normalise_pose(kps_gt)

        errors: List[float] = []
        for i in range(B):
            if not valid_pred[i] or not valid_gt[i]:
                errors.append(float("nan"))
                continue
            diff = kps_pred_n[i] - kps_gt_n[i]          # (17, 2)
            per_joint = np.linalg.norm(diff, axis=-1)    # (17,)
            errors.append(float(np.mean(per_joint)))

        return errors