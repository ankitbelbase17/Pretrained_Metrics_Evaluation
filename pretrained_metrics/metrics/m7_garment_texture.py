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
CLIP ViT-B/32 (openai/clip-vit-base-patch32) via the `clip` package or
transformers CLIPModel.  Falls back to ViT-B/16 (timm) if CLIP is absent.

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
    Backend priority: openai/clip → transformers CLIPModel → ViT (timm) → stub.
    """

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._backend = "stub"
        self.embed_dim = 512
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # Try openai/clip
        try:
            import clip as openai_clip
            self._clip, self._preprocess = openai_clip.load(
                "ViT-B/32", device=self.device
            )
            self._clip.eval()
            self._backend = "openai_clip"
            self.embed_dim = 512
            print("[GarmentMetric] Using CLIP ViT-B/32 for garment embeddings.")
            return
        except Exception as e:
            print(f"[GarmentMetric] openai/clip unavailable ({e}).")

        # Try HuggingFace CLIP
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._hf_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device).eval()
            self._hf_proc  = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._backend  = "hf_clip"
            self.embed_dim = 512
            print("[GarmentMetric] Using HuggingFace CLIP for garment embeddings.")
            return
        except Exception as e:
            print(f"[GarmentMetric] HuggingFace CLIP unavailable ({e}).")

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
            print(f"[GarmentMetric] ViT unavailable ({e}). Using stub.")

        self._backend = "stub"

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, cloth_imgs: torch.Tensor) -> np.ndarray:
        """cloth_imgs : (B, 3, H, W)  float32  [0,1] → (B, D) np.ndarray"""
        B = cloth_imgs.shape[0]

        if self._backend == "openai_clip":
            return self._openai_clip_embed(cloth_imgs)

        if self._backend == "hf_clip":
            return self._hf_clip_embed(cloth_imgs)

        if self._backend == "vit":
            return self._vit_embed(cloth_imgs)

        return np.random.default_rng(42).normal(0, 1, (B, self.embed_dim)).astype(np.float32)

    def _openai_clip_embed(self, imgs: torch.Tensor) -> np.ndarray:
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]
        inp  = torch.stack([self._preprocess(p) for p in pils]).to(self.device)
        emb  = self._clip.encode_image(inp)
        emb  = F.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()

    def _hf_clip_embed(self, imgs: torch.Tensor) -> np.ndarray:
        from PIL import Image
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]
        inputs = self._hf_proc(images=pils, return_tensors="pt", padding=True).to(self.device)
        emb = self._hf_model.get_image_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        emb = F.normalize(emb.float(), dim=-1)
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

    def __init__(self, device: str = "cpu", eps: float = 1e-6):
        self._encoder   = _GarmentEncoder(device)
        self.eps        = eps
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
        if len(self._embeddings) < 2:
            return {
                "garment_diversity_logdet": float("nan"),
                "garment_variance_total":   float("nan"),
                "garment_embed_dim":        float(D),
            }

        E  = np.stack(self._embeddings, axis=0)        # (N, D)
        mu = E.mean(axis=0, keepdims=True)
        Ec = E - mu
        cov = (Ec.T @ Ec) / max(len(E) - 1, 1)        # (D, D)

        # Regularise
        reg_cov = cov + self.eps * np.eye(D)

        sign, log_det = np.linalg.slogdet(reg_cov)
        d_garment = float(log_det) if sign > 0 else float("nan")

        eigvals = np.linalg.eigvalsh(reg_cov)
        total_var = float(eigvals.sum())

        return {
            "garment_diversity_logdet": d_garment,
            "garment_variance_total":   total_var,
            "garment_embed_dim":        float(D),
        }

    def reset(self):
        self._embeddings.clear()
