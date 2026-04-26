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
OpenMMLab MMPose ViTPose + person detector (primary),
HuggingFace ViTPose fallback. COCO-17 keypoints.

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
import torchvision.transforms.functional as TF


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
    """COCO-17 keypoint extractor using MMPose ViTPose (fallback: HF ViTPose)."""

    INPUT_SIZE = (256, 192)   # H×W for most top-down pose models

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "none"
        self._mmpose_inferencer = None
        self._model = None
        self._processor = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        try:
            from mmpose.apis import MMPoseInferencer

            self._mmpose_inferencer = MMPoseInferencer(
                pose2d="vitpose-b",
                det_model="rtmdet_m",
                det_cat_ids=[0],
                device=self.device,
            )
            self._backend = "mmpose_vitpose"
            print("[PoseMetric] Using OpenMMLab MMPose ViTPose + detector.")
            return
        except Exception as e:
            print(f"[PoseMetric] MMPose ViTPose unavailable ({e}); trying HF ViTPose fallback.")

        try:
            from transformers import AutoProcessor, VitPoseForPoseEstimation

            self._processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
            self._model = VitPoseForPoseEstimation.from_pretrained(
                "usyd-community/vitpose-base-simple",
                use_safetensors=True,
            ).to(self.device).eval()
            self._backend = "vitpose_hf_fallback"
            print("[PoseMetric] Using HF ViTPose fallback for keypoint extraction.")
            return
        except Exception as e2:
            raise RuntimeError(
                "[PoseMetric] No ViTPose backend available. "
                "Install MMPose(+mmdet) or transformers ViTPose dependencies."
            ) from e2

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns : (B, 17, 2) numpy array of (x, y) pixel coordinates
        """
        if imgs.ndim != 4:
            raise RuntimeError(f"[PoseMetric] Expected 4D tensor, got {tuple(imgs.shape)}")
        if imgs.shape[1] != 3 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        if imgs.shape[1] != 3:
            raise RuntimeError(f"[PoseMetric] Expected C=3, got {tuple(imgs.shape)}")

        if self._backend == "mmpose_vitpose":
            return self._extract_with_mmpose(imgs)
        return self._extract_with_hf(imgs)

    def _extract_with_mmpose(self, imgs: torch.Tensor) -> np.ndarray:
        b = imgs.shape[0]
        all_kps = np.zeros((b, 17, 2), dtype=np.float32)
        for i in range(b):
            img = imgs[i].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            try:
                res = next(self._mmpose_inferencer(img, return_vis=False))
                preds = res.get("predictions", [])
                persons = preds[0] if preds and len(preds) > 0 else []
                if not persons:
                    continue

                def _person_score(p):
                    sc = p.get("keypoint_scores", None)
                    if sc is None:
                        return 0.0
                    arr = np.asarray(sc, dtype=np.float32).reshape(-1)
                    return float(np.mean(arr)) if arr.size else 0.0

                best = max(persons, key=_person_score)
                kps = np.asarray(best.get("keypoints", []), dtype=np.float32)
                if kps.ndim == 2 and kps.shape[1] >= 2:
                    n = min(17, kps.shape[0])
                    all_kps[i, :n, :] = kps[:n, :2]
            except Exception:
                continue
        return all_kps

    def _extract_with_hf(self, imgs: torch.Tensor) -> np.ndarray:
        b, _c, h, w = imgs.shape
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]
        boxes = [np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32) for _ in range(b)]

        inputs = self._processor(images=pils, boxes=boxes, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        pose_results = self._processor.post_process_pose_estimation(
            outputs, boxes=boxes, threshold=0.1
        )

        all_kps = np.zeros((b, 17, 2), dtype=np.float32)
        for i in range(b):
            if i >= len(pose_results) or not pose_results[i]:
                continue
            person = pose_results[i][0]
            keypoints = person.get("keypoints", [])
            labels = person.get("labels", [])
            for kp, label in zip(keypoints, labels):
                idx = int(label.item() if torch.is_tensor(label) else label)
                if 0 <= idx < 17:
                    x = float(kp[0].item() if torch.is_tensor(kp[0]) else kp[0])
                    y = float(kp[1].item() if torch.is_tensor(kp[1]) else kp[1])
                    all_kps[i, idx, 0] = x
                    all_kps[i, idx, 1] = y
        return all_kps

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
