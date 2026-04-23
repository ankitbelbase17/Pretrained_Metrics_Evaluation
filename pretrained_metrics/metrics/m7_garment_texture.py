"""
metrics/m7_garment_texture.py
==============================
Metric 7 — Garment Texture Diversity
--------------------------------------
Uses CLIP image embeddings of garment crops to measure texture/style spread.

    D_garment = log det(Cov(g_i) + ε·I)

where g_i ∈ R^512 is the CLIP embedding of the i-th garment image.

Pretrained model
-----------------
open_clip ViT-B/32.
Falls back to ViT-B/16 (timm) proxy if open_clip is absent.

Input
------
cloth_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]
(The CLOTH / garment tensor from the dataloader — NOT the person image.)

Returns (compute())
--------------------
dict with:
    garment_diversity_logdet  : log det(Cov(g) + ε·I)
    garment_variance_total    : sum of embedding eigenvalues
    garment_embed_dim         : embedding dimensionality used
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
# CLIP / ViT garment encoder
# ─────────────────────────────────────────────────────────────────────────────

class _GarmentEncoder:
    """
    Returns (B, D) garment embeddings.
    Backend priority: open_clip → ViT (timm) → stub.
    """

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._backend = "stub"
        self.embed_dim = 512
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # 1. Try FashionCLIP (elite semantic fashion mappings)
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._hf_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device).eval()
            self._hf_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            self._backend = "fashion_clip"
            self.embed_dim = 512
            print("[GarmentMetric] Using FashionCLIP (patrickjohncyh/fashion-clip) for absolute domain accuracy.")
            return
        except Exception as e:
            print(f"[GarmentMetric] FashionCLIP unavailable ({e}). Falling back to OpenCLIP.")

        # 2. Try open_clip_torch as fallback
        try:
            import open_clip
            self._oc_model, _, self._oc_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._oc_model = self._oc_model.to(self.device).eval()
            self._backend = "open_clip"
            self.embed_dim = 512
            print("[GarmentMetric] Using open_clip ViT-B/32 for garment embeddings.")
            return
        except Exception as e:
            print(f"[GarmentMetric] open_clip unavailable ({e}).")

        # Try ViT (timm)
        try:
            import timm
            self._vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            ).to(self.device).eval()
            self._norm = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            self._backend = "vit"
            self.embed_dim = 768
            print("[GarmentMetric] Using ViT-B/16 (timm) as CLIP proxy.")
            return
        except Exception as e:
            raise RuntimeError(
                "[GarmentMetric] No valid garment encoder available. "
                "Install open_clip or timm."
            ) from e

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, cloth_imgs: torch.Tensor) -> np.ndarray:
        """cloth_imgs : (B, 3, H, W)  float32  [0,1] → (B, D) np.ndarray"""
        B = cloth_imgs.shape[0]

        if self._backend == "fashion_clip":
            return self._fashion_clip_embed(cloth_imgs)

        if self._backend == "open_clip":
            return self._open_clip_embed(cloth_imgs)

        if self._backend == "vit":
            return self._vit_embed(cloth_imgs)

        raise RuntimeError("[GarmentMetric] No valid garment encoder available.")

    def _fashion_clip_embed(self, imgs: torch.Tensor) -> np.ndarray:
        from PIL import Image
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]
        inputs = self._hf_processor(images=pils, return_tensors="pt").to(self.device)
        emb = self._hf_model.get_image_features(**inputs)
        # Some FashionCLIP checkpoints/wrappers can return a model output object
        # instead of a raw tensor; handle both safely.
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif hasattr(emb, "last_hidden_state"):
            emb = emb.last_hidden_state[:, 0]
        emb = F.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()

    def _open_clip_embed(self, imgs: torch.Tensor) -> np.ndarray:
        from PIL import Image
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]
        inp  = torch.stack([self._oc_preprocess(p) for p in pils]).to(self.device)
        emb  = self._oc_model.encode_image(inp)
        emb  = F.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()

    def _vit_embed(self, imgs: torch.Tensor) -> np.ndarray:
        x = T.functional.resize(imgs, [224, 224]).to(self.device)
        x = torch.stack([self._norm(im) for im in x])
        emb = self._vit(x)          # (B, 768)
        emb = F.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# GarmentTextureMetrics
# ─────────────────────────────────────────────────────────────────────────────

class GarmentTextureMetrics:

    def __init__(self, device: str = "cpu", eps: float = 1e-6,
                 n_components: int = 128):
        """
        n_components : number of PCA components to keep before computing
                       log-det.  Avoids the rank-deficiency collapse that
                       occurs when D (e.g. 512 or 768) >> effective rank of
                       L2-normalised embeddings.  Only the top-k singular
                       values that carry real signal are included; the
                       discarded near-zero dimensions no longer contribute
                       large negative log terms.
        eps          : small absolute floor added to each retained eigenvalue
                       for numerical safety.
        """
        self._encoder    = _GarmentEncoder(device)
        self.eps         = eps
        self.n_components = n_components
        self._embeddings: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def update(self, cloth_imgs: torch.Tensor):
        """cloth_imgs : (B, 3, H, W)  float32  [0,1]"""
        embs = self._encoder(cloth_imgs)    # (B, D)
        for e in embs:
            self._embeddings.append(e)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        D = self._encoder.embed_dim
        N = len(self._embeddings)
        if N < 2:
            return {
                "garment_diversity_neg_logdet":      float("nan"),
                "garment_diversity_neg_logdet_normalized":  float("nan"),
                "garment_variance_total":        float("nan"),
                "garment_embed_dim":             float(D),
                "garment_effective_rank":         float("nan"),
            }

        E  = np.stack(self._embeddings, axis=0)        # (N, D)
        mu = E.mean(axis=0, keepdims=True)
        Ec = E - mu                                     # centred (N, D)

        # ── PCA via thin SVD ──────────────────────────────────────────────────
        k_max = min(N - 1, D, self.n_components)
        _, S, _ = np.linalg.svd(Ec, full_matrices=False)
        S = S[:k_max]
        eigvals = (S ** 2) / max(N - 1, 1)

        # ── Principal Components Selection (95% Variance) ────────────────────
        # L2-normalised embeddings lie on a low-dimensional manifold.
        # Ensure we only use the principal components that explain 95% of the variance
        # to compute log-determinant safely.
        total_variance = float(eigvals.sum())
        if total_variance > 0:
            cumulative_var = np.cumsum(eigvals) / total_variance
            # Find how many components needed to reach 95% variance
            effective_rank = int(np.searchsorted(cumulative_var, 0.95)) + 1
        else:
            effective_rank = 0

        sig_eigvals = eigvals[:effective_rank]

        if effective_rank == 0:
            return {
                "garment_diversity_neg_logdet":      float("inf"),
                "garment_diversity_neg_logdet_normalized":  float("inf"),
                "garment_variance_total":        float(eigvals.sum()),
                "garment_embed_dim":             float(k_max),
                "garment_effective_rank":         0.0,
            }

        # Negative log-det over significant eigenvalues only
        reg_eigvals = sig_eigvals + self.eps
        neg_log_det = -float(np.sum(np.log(reg_eigvals)))
        total_var   = float(eigvals.sum())

        # Normalised negative log-det (per effective dimension)
        neg_log_det_norm = neg_log_det / effective_rank

        return {
            "garment_diversity_neg_logdet":      neg_log_det,
            "garment_diversity_neg_logdet_normalized":  neg_log_det_norm,
            "garment_variance_total":        total_var,
            "garment_embed_dim":             float(effective_rank),
            "garment_effective_rank":         float(effective_rank),
        }

    def reset(self):
        self._embeddings.clear()
