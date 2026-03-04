"""
metrics/m5_body_shape.py
=========================
Metric 5 — Body Shape Diversity
---------------------------------
Uses SMPL shape coefficients β ∈ R^10 extracted from HMR2.0 or a proxy.

D_shape = log det(Cov(β) + ε·I)

Pretrained model
-----------------
HMR2.0 (4D-Humans) — direct successor to SPIN from the same research lineage.
Weights are downloaded automatically from HuggingFace on first use.
  pip install git+https://github.com/shubham-goel/4D-Humans.git

Backend priority:
  Level 1 — HMR2.0:       Genuine β ∈ R^10 SMPL shape coefficients.
             Requires:  pip install git+https://github.com/shubham-goel/4D-Humans.git
             Weights:   auto-downloaded from HuggingFace (shubham-goel/4D-Humans)
                        on first use — no manual checkpoint needed.
  Level 2 — ViT-B/16 proxy:  10-D projection of ViT CLS embedding.
             NOT equivalent to SMPL β, but captures body-shape variety.
             Requires:  pip install timm
  Level 3 — Random stub (smoke-test only).

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (compute())
--------------------
dict with:
    shape_diversity_logdet   : log det(Cov(β) + ε·I)
    shape_variance_total     : sum of eigenvalues of Cov(β)  (total variance)
    shape_dims               : number of shape coefficients (10)
    backend                  : "hmr2", "vit_proxy", or "stub"
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Shape extractor — backend priority: HMR2.0 → ViT proxy → stub
# ─────────────────────────────────────────────────────────────────────────────

class _ShapeExtractor:
    SHAPE_DIM = 10

    _NORMALIZE = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._backend = "stub"

        # HMR2.0
        self._hmr2_model = None
        self._hmr2_cfg   = None

        # ViT proxy
        self._vit      = None
        self._vit_proj: Optional[nn.Linear] = None

        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        if self._try_hmr2():
            return
        if self._try_vit():
            return
        print("[BodyShapeMetric] All backends failed. Using random stub.")
        self._backend = "stub"

    # ------------------------------------------------------------------ #
    def _try_hmr2(self) -> bool:
        try:
            from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT

            # download_models() fetches the checkpoint if not already cached.
            # It returns None — the actual path is exposed via DEFAULT_CHECKPOINT.
            download_models()
            self._hmr2_model, self._hmr2_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            self._hmr2_model = self._hmr2_model.to(self.device).eval()

            self._backend = "hmr2"
            print("[BodyShapeMetric] HMR2.0 loaded ✓  "
                  "(weights auto-downloaded from HuggingFace).")
            return True

        except ImportError:
            print("[BodyShapeMetric] HMR2.0 not installed. "
                  "Run: pip install git+https://github.com/shubham-goel/4D-Humans.git  "
                  "Falling back to ViT proxy.")
            return False

        except Exception as e:
            print(f"[BodyShapeMetric] HMR2.0 initialisation failed: {e}. "
                  "Falling back to ViT proxy.")
            self._hmr2_model = None
            return False

    # ------------------------------------------------------------------ #
    def _try_vit(self) -> bool:
        try:
            import timm
            self._vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            ).to(self.device).eval()
            torch.manual_seed(0)
            self._vit_proj = nn.Linear(768, self.SHAPE_DIM, bias=False).to(self.device)
            nn.init.orthogonal_(self._vit_proj.weight)
            self._vit_proj.eval()
            self._backend = "vit_proxy"
            print("[BodyShapeMetric] Using ViT-B/16 proxy for body shape (no HMR2.0).")
            return True
        except Exception as e:
            print(f"[BodyShapeMetric] ViT unavailable ({e}).")
            return False

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns (B, 10) numpy float32 — SMPL β or proxy.
        """
        B = imgs.shape[0]

        if self._backend == "hmr2":
            return self._hmr2_forward(imgs)

        if self._backend == "vit_proxy":
            x = TF.resize(imgs, [224, 224]).to(self.device)
            x = torch.stack([self._NORMALIZE(im) for im in x])
            feats = self._vit(x)              # (B, 768)
            betas = self._vit_proj(feats)     # (B, 10)
            return betas.cpu().numpy()

        # Stub
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (B, self.SHAPE_DIM)).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _hmr2_forward(self, imgs: torch.Tensor) -> np.ndarray:
        """
        HMR2.0 expects images at 256×192 (W×H), normalised with ImageNet stats.
        Output dict contains 'pred_smpl_params' → 'betas' : (B, 10).
        """
        # HMR2.0 canonical input size: 256 wide × 192 tall
        x = TF.resize(imgs, [192, 256]).to(self.device)
        x = torch.stack([self._NORMALIZE(im) for im in x])

        batch = {"img": x}
        out   = self._hmr2_model(batch)

        # 'pred_smpl_params' is always present; betas shape is (B, 10)
        betas = out["pred_smpl_params"]["betas"]   # (B, 10)
        return betas.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# BodyShapeMetrics  (public API — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class BodyShapeMetrics:

    def __init__(self, device: str = "cpu", eps: float = 1e-6):
        self._extractor = _ShapeExtractor(device)
        self.eps = eps
        self._betas: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        betas = self._extractor(person_imgs)    # (B, 10)
        for b in betas:
            self._betas.append(b)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if len(self._betas) < 2:
            return {k: float("nan") for k in [
                "shape_diversity_logdet", "shape_variance_total",
                "shape_dims", "backend",
            ]}

        B_mat = np.stack(self._betas, axis=0)       # (N, 10)
        mu    = B_mat.mean(axis=0, keepdims=True)
        Bc    = B_mat - mu
        cov   = (Bc.T @ Bc) / max(len(B_mat) - 1, 1)   # (10, 10)
        reg   = cov + self.eps * np.eye(10)

        sign, log_det = np.linalg.slogdet(reg)
        d_shape   = float(log_det) if sign > 0 else float("nan")
        total_var = float(np.linalg.eigvalsh(reg).sum())

        return {
            "shape_diversity_logdet": d_shape,
            "shape_variance_total":   total_var,
            "shape_dims":             float(self._extractor.SHAPE_DIM),
            "backend":                self._extractor._backend,
        }

    def reset(self):
        self._betas.clear()