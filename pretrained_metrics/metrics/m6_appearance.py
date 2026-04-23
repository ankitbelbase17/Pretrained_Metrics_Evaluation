"""
metrics/m6_appearance.py
=========================
Metric 6 — Parametric Face Appearance (SMPL equivalent for Faces)
-------------------------------------------------------------------
Replaces basic embedding models (like ArcFace/AdaFace) with a 
Parametric Face Representation using MediaPipe Face Landmarker.

This version strictly isolates the exact face using a robust 
RetinaFace detector (via insightface) if available, falling back to 
a manual demographic top-center crop to guarantee MediaPipe can 
succeed even on full-body high-res studio datasets.
"""

from __future__ import annotations
import math
import os
import io
import contextlib
import warnings
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
        self._face_app = None
        self._load()

    def _load(self):
        # 1. Load RetinaFace (Insightface) Bounding Box Detector
        try:
            import insightface
            from insightface.app import FaceAnalysis
            quiet_out = io.StringIO()
            warnings.filterwarnings("ignore")
            with contextlib.redirect_stdout(quiet_out), contextlib.redirect_stderr(quiet_out):
                # Use CPU execution strictly for the detector to avoid GPU memory splits
                self._face_app = FaceAnalysis(providers=["CPUExecutionProvider"], allowed_modules=['detection'])
                self._face_app.prepare(ctx_id=-1, det_size=(320, 320))
            print("[FaceParametric] Loaded InsightFace RetinaFace for explicit face isolation.")
        except Exception as e:
            print(f"[FaceParametric] InsightFace unavailable for dynamic crops. Defaulting to standard studio-crop. ({e})")
            self._face_app = None

        # 2. Load MediaPipe Parametric Extractor
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
                num_faces=1,
                min_face_detection_confidence=0.1,
                min_face_presence_confidence=0.1,
                min_tracking_confidence=0.1
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
            self._mp_image = mp.Image
            self._mp_format = mp.ImageFormat.SRGB
            self._available = True
            print("[FaceParametric] Loaded MediaPipe Parametric Model (68-D).")
        except Exception as e:
            print(f"[FaceParametric] Could not load Parametric Face Model: {e}")
            self._available = False

    @torch.no_grad()
    def update(self, person_imgs: torch.Tensor):
        if not self._available:
            return

        B = person_imgs.shape[0]
        # Base numpy arrays [B, H, W, 3] uint8
        base_imgs_np = (person_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        for i in range(B):
            img_full = base_imgs_np[i]
            H, W, _ = img_full.shape
            
            crop_success = False
            img_crop = img_full

            # Try to dynamically crop the exact face using RetinaFace
            if self._face_app is not None:
                # InsightFace requires BGR for detection
                bgr_img = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
                faces = self._face_app.get(bgr_img)
                if len(faces) > 0:
                    # Get the largest face
                    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                    bbox = faces[0].bbox.astype(int)
                    # Add 20% margin to ensure jaw/forehead are captured perfectly for MediaPipe
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    margin_x = int(bw * 0.20)
                    margin_y = int(bh * 0.20)
                    
                    x1 = max(0, bbox[0] - margin_x)
                    y1 = max(0, bbox[1] - margin_y)
                    x2 = min(W, bbox[2] + margin_x)
                    y2 = min(H, bbox[3] + margin_y)
                    
                    img_crop = img_full[y1:y2, x1:x2]
                    crop_success = True

            # If RetinaFace failed or is uninstalled, use the manual studio center-top crop
            if not crop_success:
                crop_h = max(1, int(H * 0.35))
                crop_w0 = max(0, int(W * 0.20))
                crop_w1 = min(W, int(W * 0.80))
                img_crop = img_full[:crop_h, crop_w0:crop_w1]

            # Enforce C-contiguous memory for MediaPipe C++ bindings
            img_crop_cmem = np.ascontiguousarray(img_crop)
            mp_img = self._mp_image(image_format=self._mp_format, data=img_crop_cmem)
            
            try:
                res = self._detector.detect(mp_img)
                if not getattr(res, "face_blendshapes", None):
                    continue

                bs = res.face_blendshapes[0] 
                b_scores = np.array([cat.score for cat in bs], dtype=np.float32)

                if getattr(res, "facial_transformation_matrixes", None):
                    t_matrix = res.facial_transformation_matrixes[0].flatten()
                    t_matrix = t_matrix / (np.linalg.norm(t_matrix) + 1e-6)
                else:
                    t_matrix = np.zeros(16, dtype=np.float32)

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

        mu = E.mean(axis=0, keepdims=True)
        cov = (E - mu).T @ (E - mu) / (N - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        
        # Diversity total variance
        variance_total = float(np.sum(eigvals))
        
        # Absolute facial geometry complexity (deviation from neutral 0)
        appearance_complexity_abs = float(np.mean(np.linalg.norm(E, axis=1)))

        if variance_total > 0:
            cumulative_var = np.cumsum(eigvals) / variance_total
            effective_rank = min(int(np.searchsorted(cumulative_var, 0.95)) + 1, N - 1)
        else:
            effective_rank = 0

        if effective_rank > 0:
            sig_eigvals = eigvals[:effective_rank] + 1e-6
            log_det = float(np.sum(np.log(sig_eigvals)))
            neg_log_det_norm = -log_det / effective_rank
        else:
            neg_log_det_norm = float("inf")

        mean_pwise = float(variance_total / D)

        return {
            "appearance_complexity": appearance_complexity_abs,     
            "appearance_diversity_mean": neg_log_det_norm,
            "appearance_diversity_logdet": log_det,
            "n_faces": N
        }

    def reset(self):
        self._blendshapes_list.clear()

AppearanceMetrics = ParametricFaceMetric
