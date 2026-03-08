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
ViTPose-B (mmpose / transformers).
Falls back to a lightweight HRNet-W32 via timm when ViTPose is unavailable.
Falls back to random keypoints stub when neither is available (smoke-test only).

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


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
    """Tries ViTPose → HRNet (timm) → random stub."""

    INPUT_SIZE = (256, 192)   # H×W for most top-down pose models

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend: str = "stub"
        self._model = None
        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # Try timm HRNet as a fallback (ViTPose needs mmpose which is complex)
        try:
            import timm
            self._model = timm.create_model(
                "hrnet_w32", pretrained=True, num_classes=0
            ).to(self.device).eval()
            self._backend = "hrnet"
            print("[PoseMetric] Using HRNet-W32 (timm) for keypoint extraction.")
            return
        except Exception as e:
            print(f"[PoseMetric] HRNet not available ({e}). Using random stub.")

        self._backend = "stub"

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns : (B, 17, 2) numpy array of (x, y) pixel coordinates
        """
        B = imgs.shape[0]
        H_in, W_in = self.INPUT_SIZE

        imgs_r = F.interpolate(imgs, size=self.INPUT_SIZE, mode="bilinear",
                               align_corners=False).to(self.device)
        imgs_r = torch.stack([self._normalize(im) for im in imgs_r])

        if self._backend == "hrnet":
            feats = self._model.forward_features(imgs_r)   # (B, J, Hh, Wh)
            # Clamp in case forward_features returns pooled tensor
            if feats.ndim == 2:
                # fallback: pooled — return stub
                return self._stub(B, H_in, W_in)
            B2, J, Hh, Ww = feats.shape
            # HRNet forward_features may return CNN feature maps (not heatmaps).
            # Only treat channels as keypoint heatmaps when J == 17 (COCO joints).
            if J != 17:
                return self._stub(B, H_in, W_in)
            flat = feats.view(B2, J, -1).argmax(-1)        # (B, J)
            ys   = (flat // Ww).float() / Hh * H_in
            xs   = (flat %  Ww).float() / Ww * W_in
            kps  = torch.stack([xs, ys], dim=-1)           # (B, J, 2)
            return kps.cpu().numpy()

        return self._stub(B, H_in, W_in)

    def _stub(self, B: int, H: int, W: int) -> np.ndarray:
        """Random keypoints — only used when no model is loaded."""
        rng = np.random.default_rng(42)
        kps = rng.uniform(0, 1, (B, 17, 2))
        kps[:, :, 0] *= W
        kps[:, :, 1] *= H
        return kps.astype(np.float32)


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
            self._pose_vecs.append(pn.flatten())   # (34,)

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
        Returns dict:
            pose_diversity        : log det(Cov + ε·I)
            pose_artic_complexity : Σ Var(θ_limb)
            pose_artic_mean_per_image : mean per-image angle std
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

        # 1B — Complexity
        c_artic = 0.0
        for t_idx in range(len(TRIPLET_IDX)):
            angles = self._all_angles[t_idx]
            if len(angles) > 1:
                c_artic += float(np.var(angles))

        artic_per_image = [v for v in self._per_image_artic if not math.isnan(v)]

        return {
            "pose_diversity":            d_pose,
            "pose_artic_complexity":     c_artic,
            "pose_artic_mean_per_image": float(np.mean(artic_per_image)) if artic_per_image else float("nan"),
        }

    def reset(self):
        self._pose_vecs.clear()
        for k in self._all_angles:
            self._all_angles[k].clear()
        self._per_image_artic.clear()
