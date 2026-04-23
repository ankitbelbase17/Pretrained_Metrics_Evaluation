"""
metrics/m1_pose.py
==================
Metric 1 — Pose Diversity & Pose Articulation Complexity
----------------------------------------------------------

1A. Pose Diversity (D_pose)
    log det(Cov(v_i) + ε·I)
    where v_i ∈ R^34 is the flattened, normalised pose vector for image i.

1B. Pose Articulation Complexity (C_artic)
    Sum of joint-angle variances across a predefined set of limb triplets.

Pretrained model
-----------------
Keypoint R-CNN (torchvision), COCO-17 keypoints.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns
--------
dict with keys:
    "pose_diversity"           float  (dataset-level, call compute() at end)
    "pose_artic_complexity"    float  (dataset-level)
    "per_image_artic"          list[float]   (per batch / accumulated)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# COCO 17-joint skeleton
# ─────────────────────────────────────────────────────────────────────────────
# Joint index → name
COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
J2I = {n: i for i, n in enumerate(COCO_JOINTS)}

# Limb triplets (a, b, c): b is the vertex joint (angle measured at b)
LIMB_TRIPLETS = [
    ("left_shoulder",  "left_elbow",   "left_wrist"),    # left elbow
    ("right_shoulder", "right_elbow",  "right_wrist"),   # right elbow
    ("left_hip",       "left_knee",    "left_ankle"),    # left knee
    ("right_hip",      "right_knee",   "right_ankle"),   # right knee
    ("left_elbow",     "left_shoulder","left_hip"),      # left shoulder tilt
    ("right_elbow",    "right_shoulder","right_hip"),    # right shoulder tilt
    ("left_shoulder",  "left_hip",     "left_knee"),     # left torso-hip
    ("right_shoulder", "right_hip",    "right_knee"),    # right torso-hip
]
TRIPLET_IDX = [(J2I[a], J2I[b], J2I[c]) for a, b, c in LIMB_TRIPLETS]

# SPIN model neck proxy: midpoint of left/right shoulder
IDX_L_SHOULDER = J2I["left_shoulder"]
IDX_R_SHOULDER = J2I["right_shoulder"]
IDX_L_HIP      = J2I["left_hip"]
IDX_R_HIP      = J2I["right_hip"]


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained keypoint extractor
# ─────────────────────────────────────────────────────────────────────────────

class _KeypointExtractor:
    """COCO-17 keypoint extractor using torchvision KeypointRCNN."""

    INPUT_SIZE = (256, 192)   # H×W for most top-down pose models

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "keypointrcnn"
        self._krcnn = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        try:
            import torchvision

            weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            self._krcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                weights=weights
            )
            self._krcnn = self._krcnn.to(self.device).eval()
            self._backend = "keypointrcnn"
            print("[PoseMetric] Using KeypointRCNN for keypoint extraction.")
        except Exception as e2:
            raise RuntimeError(
                "[PoseMetric] KeypointRCNN backend unavailable. "
                "Ensure torchvision detection/keypoint dependencies are installed."
            ) from e2

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns : (B, 17, 2) numpy array of (x, y) pixel coordinates
        """
        dets = self._krcnn([im.to(self.device) for im in imgs])
        all_kps: List[np.ndarray] = []
        for det in dets:
            kps = det.get("keypoints")
            scores = det.get("scores")
            if kps is None or kps.numel() == 0:
                all_kps.append(np.zeros((17, 2), dtype=np.float32))
                continue
            best = int(torch.argmax(scores).item()) if scores is not None and scores.numel() else 0
            kp = kps[best, :, :2].detach().cpu().numpy().astype(np.float32)
            if kp.shape != (17, 2):
                all_kps.append(np.zeros((17, 2), dtype=np.float32))
            else:
                all_kps.append(kp)
        return np.stack(all_kps, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Normalise pose
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_pose(kps: np.ndarray) -> np.ndarray:
    """
    kps : (B, 17, 2)
    Returns normalised (B, 17, 2) and a validity mask (B,) bool.
    Translation: subtract hip centre.
    Scale:       divide by torso length (neck→hip).
    """
    B = kps.shape[0]
    out   = kps.copy()
    valid = np.ones(B, dtype=bool)

    for i in range(B):
        p = kps[i]
        hip    = (p[IDX_L_HIP]      + p[IDX_R_HIP])      / 2.0
        neck   = (p[IDX_L_SHOULDER] + p[IDX_R_SHOULDER]) / 2.0
        torso  = np.linalg.norm(neck - hip)

        if torso < 1e-6:
            valid[i] = False
            continue

        out[i] = (p - hip) / torso

    return out, valid


# ─────────────────────────────────────────────────────────────────────────────
# Joint-angle computation
# ─────────────────────────────────────────────────────────────────────────────

def _joint_angle(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    """Angle at joint b (vertex), in radians."""
    va = pa - pb
    vc = pc - pb
    n_a = np.linalg.norm(va)
    n_c = np.linalg.norm(vc)
    if n_a < 1e-8 or n_c < 1e-8:
        return float("nan")
    cos_t = np.dot(va, vc) / (n_a * n_c)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.arccos(cos_t))


# ─────────────────────────────────────────────────────────────────────────────
# PoseMetrics class
# ─────────────────────────────────────────────────────────────────────────────

class PoseMetrics:
    """
    Accumulates per-image pose features, then computes:
      - D_pose  = log det(Cov(v_i) + ε·I)
      - C_artic = Σ_limbs Var(θ_limb)
    """

    def __init__(self, device: str = "cpu", eps: float = 1e-6):
        self.extractor = _KeypointExtractor(device=device)
        self.eps = eps
        self._pose_vecs: List[np.ndarray]     = []   # (34,) vectors
        self._all_angles: Dict[int, List[float]] = {t: [] for t in range(len(TRIPLET_IDX))}
        self._per_image_artic: List[float]    = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """
        person_imgs : (B, 3, H, W)  float32  [0,1]
        Call once per batch.
        """
        kps_raw  = self.extractor(person_imgs)      # (B,17,2) numpy
        kps_norm, valid = _normalise_pose(kps_raw)  # (B,17,2) normalised

        for i in range(kps_raw.shape[0]):
            if not valid[i]:
                continue

            pn = kps_norm[i]   # (17, 2)

            # ── Pose vector ─────────────────────────────────────────────
            # Use raw keypoints for log-det diversity (no pre-normalisation).
            self._pose_vecs.append(kps_raw[i].flatten())   # (34,)

            # ── Joint angles ─────────────────────────────────────────────
            img_angles = []
            for t_idx, (ia, ib, ic) in enumerate(TRIPLET_IDX):
                ang = _joint_angle(pn[ia], pn[ib], pn[ic])
                if not math.isnan(ang):
                    self._all_angles[t_idx].append(ang)
                    img_angles.append(ang)

            # Per-image articulation = std of all valid angles
            if img_angles:
                self._per_image_artic.append(float(np.std(img_angles)))
            else:
                self._per_image_artic.append(float("nan"))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        """
          Returns dict with formulas:

             1) Pose Diversity
                 D_pose = log det( Cov(v) + eps * I )
                 where v in R^34 is flattened normalized keypoints.

             2) Pose Articulation Complexity
                 C_artic = sum_limb Var(theta_limb)
                 where theta_limb is the joint angle at each predefined triplet.

             3) Mean per-image articulation
                 mean_i std_j(theta_{i,j}) over valid angles in each image.
        """
        if len(self._pose_vecs) < 2:
            return {
                "pose_diversity": float("nan"),
                "pose_artic_complexity": float("nan"),
                "pose_artic_mean_per_image": float("nan"),
            }

        # 1A — Diversity
        V   = np.stack(self._pose_vecs, axis=0).astype(np.float64)    # (N, D)
        D   = V.shape[1]
        mu  = V.mean(axis=0, keepdims=True)
        Vc  = V - mu
        cov = (Vc.T @ Vc) / max(len(V) - 1, 1)    # (D, D)
        reg = cov + self.eps * np.eye(D)
        sign, log_det = np.linalg.slogdet(reg)
        d_pose = float(log_det) if sign > 0 else float("nan")
        d_pose_norm = (d_pose / D) if (not math.isnan(d_pose) and D > 0) else float("nan")

        # 1B — Complexity (Absolute magnitude per sample, NOT dataset variance)
        c_artic = 0.0
        valid_limbs = 0
        for t_idx in range(len(TRIPLET_IDX)):
            angles = self._all_angles[t_idx]
            if len(angles) > 0:
                # Average angle (how bent the limb is)
                c_artic += float(np.mean(np.abs(np.array(angles) - math.pi))) # Deviation from straight (pi)
                valid_limbs += 1
        
        if valid_limbs > 0:
            c_artic = c_artic / valid_limbs

        artic_per_image = [v for v in self._per_image_artic if not math.isnan(v)]

        return {
            "pose_diversity":            d_pose,
            "pose_diversity_logdet_raw": d_pose,
            "pose_diversity_logdet_normalized": d_pose_norm,
            "pose_artic_complexity":     c_artic,
            "pose_artic_mean_per_image": float(np.mean(artic_per_image)) if artic_per_image else float("nan"),
        }

    def reset(self):
        self._pose_vecs.clear()
        for k in self._all_angles:
            self._all_angles[k].clear()
        self._per_image_artic.clear()
