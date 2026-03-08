"""
metrics/m6_appearance.py
=========================
Metric 6 — Appearance / Ethnicity-Proxy Diversity
---------------------------------------------------
We do NOT classify ethnicity directly (avoids bias).
Instead we measure embedding diversity of face regions using ArcFace.

    D_face = (2 / (N(N-1))) * Σ_{i<j} (1 - cos_sim(f_i, f_j))
           = mean pairwise cosine distance

Pretrained model
-----------------
ArcFace via InsightFace (insightface package).
Falls back to CLIP ViT-B/32 face-region encoder when InsightFace unavailable.
Falls back to random 512-D embeddings (smoke-test stub).

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Implementation note
--------------------
We crop the upper-third of the person image as a face proxy
when a face detector is unavailable (avoids dependency on RetinaFace).

Returns (compute())
--------------------
dict with:
    appearance_diversity_mean    : mean pairwise cosine distance  (D_face)
    appearance_diversity_std     : std of pairwise cosine distances
    n_faces                      : total face embeddings collected
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Face extractor
# ─────────────────────────────────────────────────────────────────────────────

class _FaceEmbedder:
    """
    Returns (B, 512) face embeddings.
    Backend priority: ArcFace (insightface) → CLIP → stub.
    """
    EMBED_DIM = 512

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "stub"
        self._model   = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # Try InsightFace ArcFace
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # Use GPU if available (onnxruntime-gpu), fall back to CPU
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
            except ImportError:
                available = []
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                print("[AppearanceMetric] WARNING: onnxruntime-gpu not installed! "
                      "InsightFace will run on CPU (very slow). "
                      "Install with: pip install onnxruntime-gpu")
            self._app = FaceAnalysis(providers=providers)
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._backend = "arcface"
            print(f"[AppearanceMetric] Using InsightFace ArcFace ({providers[0]}).")
            return
        except Exception as e:
            print(f"[AppearanceMetric] InsightFace unavailable ({e}).")

        # Try openai/clip (package name: openai-clip)
        try:
            import clip as _oa_clip
            if not hasattr(_oa_clip, "load"):
                raise ImportError("'clip' package installed is not openai/clip "
                                  "(missing 'load'). Try: pip install openai-clip")
            self._clip_model, self._clip_preprocess = _oa_clip.load(
                "ViT-B/32", device=self.device
            )
            self._clip_model.eval()
            self._backend = "clip"
            self.EMBED_DIM = 512
            print("[AppearanceMetric] Using CLIP ViT-B/32 (openai) as face proxy.")
            return
        except Exception as e:
            print(f"[AppearanceMetric] openai/clip unavailable ({e}).")

        # Try open_clip_torch (pip install open_clip_torch) — different import name,
        # unaffected by numpy ABI issues that break the transformers-based HF CLIP.
        try:
            import open_clip
            self._oc_model, _, self._oc_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._oc_model = self._oc_model.to(self.device).eval()
            self._backend = "open_clip"
            self.EMBED_DIM = 512
            print("[AppearanceMetric] Using open_clip ViT-B/32 as face proxy.")
            return
        except Exception as e:
            print(f"[AppearanceMetric] open_clip unavailable ({e}). Using random stub.")

        self._backend = "stub"

    # --------------------------------------------------------------------- #
    def _crop_face_region(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Crop upper ~30% of image as face proxy (H/3 rows from top)."""
        H = img_tensor.shape[1]
        return img_tensor[:, : max(H // 3, 1), :]

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns (B, D) numpy float32.
        """
        B = imgs.shape[0]

        if self._backend == "arcface":
            return self._arcface_embeddings(imgs)

        if self._backend in ("clip", "open_clip"):
            return self._clip_embeddings(imgs)

        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (B, self.EMBED_DIM)).astype(np.float32)

    def _clip_embeddings(self, imgs: torch.Tensor) -> np.ndarray:
        """Shared encoder for both openai/clip and open_clip backends."""
        face_crops = torch.stack(
            [self._crop_face_region(imgs[i]) for i in range(imgs.shape[0])]
        )
        pils = [TF.to_pil_image(fc.clamp(0, 1).cpu()) for fc in face_crops]
        if self._backend == "open_clip":
            import open_clip
            inp = torch.stack([self._oc_preprocess(p) for p in pils]).to(self.device)
            emb = self._oc_model.encode_image(inp)
        else:
            inp = torch.stack([self._clip_preprocess(p) for p in pils]).to(self.device)
            emb = self._clip_model.encode_image(inp)
        emb = F.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()

    def _arcface_embeddings(self, imgs: torch.Tensor) -> np.ndarray:
        import cv2
        results = []
        for i in range(imgs.shape[0]):
            rgb = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = self._app.get(bgr)
            if faces:
                emb = faces[0].normed_embedding
            else:
                emb = np.zeros(512, dtype=np.float32)
            results.append(emb)
        return np.stack(results, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# AppearanceMetrics
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceMetrics:

    def __init__(self, device: str = "cpu"):
        self._embedder = _FaceEmbedder(device)
        self._embeddings: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        embs = self._embedder(person_imgs)          # (B, D)
        for e in embs:
            self._embeddings.append(e)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        N = len(self._embeddings)
        if N < 2:
            return {
                "appearance_diversity_mean": float("nan"),
                "appearance_diversity_std":  float("nan"),
                "n_faces":                   float(N),
            }

        E   = np.stack(self._embeddings, axis=0)          # (N, D)
        # L2-normalise
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        E_n   = E / norms                                  # (N, D)

        # Pairwise cosine similarity matrix
        C = E_n @ E_n.T                                    # (N, N)
        # Extract upper triangle (i < j)
        triu = C[np.triu_indices(N, k=1)]                  # (N*(N-1)/2,)
        cos_dist = 1.0 - triu                              # cosine distance

        return {
            "appearance_diversity_mean": float(cos_dist.mean()),
            "appearance_diversity_std":  float(cos_dist.std()),
            "n_faces":                   float(N),
        }

    def reset(self):
        self._embeddings.clear()
