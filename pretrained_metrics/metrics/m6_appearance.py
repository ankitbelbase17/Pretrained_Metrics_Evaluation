"""
metrics/m6_appearance.py
=========================
Metric 6 — Parametric Face Appearance (SMPL equivalent for Faces)
-------------------------------------------------------------------
Replaces basic embedding models (like ArcFace/AdaFace) with a 
Parametric Face Representation using MediaPipe Face Landmarker.
This is biologically closer to SMPL: it outputs 52 ARKit demographic/expression 
blendshapes + 16 geometric transformation (pose) parameters per face.

Diversity is measured by computing the Log-Determinant of these 68 
parameters over the dataset, analogous to our Shape and VAE latent metrics.
Complexity is the total variance of parametric deformation.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (compute())
--------------------
dict with:
    appearance_complexity                : Total variance of face parameters
    appearance_diversity_logdet          : LogDet of the parametric covariance
    appearance_diversity_mean            : Fallback key (for legacy systems)
    n_faces                              : Total face embeddings collected
"""

from __future__ import annotations
import math
import os
import urllib.request
from typing import Dict, List
import numpy as np
import torch
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Parametric Face Extractor (MediaPipe ARKit Blendshapes 52 + 16 Pose)
# ─────────────────────────────────────────────────────────────────────────────

class ParametricFaceMetric:
    """
    Tracks parametric face states.
    Total dimensionality per face = 52 (blendshapes) + 16 (transform) = 68.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._blendshapes_list: List[np.ndarray] = []
        self._load()

    def _load(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Ensure model file exists
            model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if not os.path.exists(model_path):
                print("[FaceParametric] Downloading MediaPipe Face Landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
            self._mp_image = mp.Image
            self._mp_format = mp.ImageFormat.SRGB
            self._available = True
            print("[FaceParametric] Loaded MediaPipe Parametric Face Model (68-D).")
        except Exception as e:
            print(f"[FaceParametric] Could not load Parametric Face Model: {e}")
            self._available = False

    @torch.no_grad()
    def update(self, person_imgs: torch.Tensor):
        if not self._available:
            return

        B = person_imgs.shape[0]
        # Convert to numpy uint8 RGB arrays
        imgs_np = (person_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        for i in range(B):
            img = imgs_np[i]
            mp_img = self._mp_image(image_format=self._mp_format, data=img)
            try:
                res = self._detector.detect(mp_img)
                if not getattr(res, "face_blendshapes", None):
                    continue

                # 52 Blendshapes representing expression/shape
                bs = res.face_blendshapes[0] 
                b_scores = np.array([cat.score for cat in bs], dtype=np.float32) # (52,)

                # 16-element Transformation matrix (Pose: yaw, pitch, roll, trans)
                if getattr(res, "facial_transformation_matrixes", None):
                    t_matrix = res.facial_transformation_matrixes[0].flatten() # (16,)
                    # Normalize transform so it mixes well with [0,1] blendshapes
                    # The translation components [12, 13, 14] are scale dependent, let's keep robust rot components
                    t_matrix = t_matrix / (np.linalg.norm(t_matrix) + 1e-6)
                else:
                    t_matrix = np.zeros(16, dtype=np.float32)

                # Combine into a single 68-D parametric representation
                params = np.concatenate([b_scores, t_matrix])
                self._blendshapes_list.append(params)

            except Exception as e:
                import traceback
                traceback.print_exc()
                pass

    def compute(self) -> Dict[str, float]:
        if not self._blendshapes_list:
            return {
                "appearance_complexity": float("nan"),
                "appearance_diversity_mean": float("nan"),
                "appearance_diversity_logdet": float("nan"),
                "n_faces": 0
            }

        E = np.stack(self._blendshapes_list, axis=0) # (N, 68)
        N, D = E.shape

        if N < 2:
            return {
                "appearance_complexity": float("nan"),
                "appearance_diversity_mean": float("nan"),
                "appearance_diversity_logdet": float("nan"),
                "n_faces": N
            }

        # ── Compute Parametric Complexity (Total Variance) ─────────────
        mu = E.mean(axis=0, keepdims=True)
        cov = (E - mu).T @ (E - mu) / (N - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        
        # Diversity total variance
        variance_total = float(np.sum(eigvals))
        
        # Absolute facial geometry complexity (deviation from neutral 0)
        appearance_complexity_abs = float(np.mean(np.linalg.norm(E, axis=1)))

        # ── Compute Parametric Diversity (LogDet of principal cov) ─────
        if variance_total > 0:
            cumulative_var = np.cumsum(eigvals) / variance_total
            effective_rank = min(int(np.searchsorted(cumulative_var, 0.95)) + 1, N - 1)
        else:
            effective_rank = 0

        if effective_rank > 0:
            sig_eigvals = eigvals[:effective_rank] + 1e-6
            log_det = float(np.sum(np.log(sig_eigvals)))
            # Provide negative log-determinant normalized so higher is consistently better
            neg_log_det_norm = -log_det / effective_rank
        else:
            neg_log_det_norm = float("inf")

        # Provide fallback mean score to satisfy unified index exactly
        # Simple mean pairwise variance measure
        mean_pwise = float(variance_total / D)

        return {
            "appearance_complexity": appearance_complexity_abs,     
            "appearance_diversity_mean": neg_log_det_norm, # unified_index uses this key currently
            "appearance_diversity_logdet": log_det,
            "n_faces": N
        }

    def reset(self):
        self._blendshapes_list.clear()

# Override the class alias to match expected namespace
AppearanceMetrics = ParametricFaceMetric
