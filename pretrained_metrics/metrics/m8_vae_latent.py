"""
pretrained_metrics/metrics/m8_vae_latent.py
============================================
Metric 8 — VAE Latent Space Diversity
--------------------------------------
Uses Stable Diffusion VAE encoder to extract image-level latent codes.

    D_vae = log det(Cov(z_i) + ε·I)

where z_i ∈ R^D is the flattened VAE latent for image i.

VAE embeddings capture:
  - Global composition and layout
  - Colour palette and distribution
  - Texture statistics
  - Overall visual "gestalt"

Pretrained Model
-----------------
Stable Diffusion VAE (stabilityai/sd-vae-ft-mse).
Falls back to CompVis/stable-diffusion-v1-4 VAE.
Falls back to a simple autoencoder proxy (for smoke tests).

Input
------
imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns
--------
dict with:
    vae_diversity_logdet  : log det(Cov(z) + ε·I)
    vae_variance_total    : sum of embedding eigenvalues
    vae_embed_dim         : embedding dimensionality
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# VAE Encoder Backend
# ─────────────────────────────────────────────────────────────────────────────

class _VAEEncoder:
    """
    Extracts VAE latent embeddings from images.
    
    Backend priority:
      1. diffusers AutoencoderKL (stabilityai/sd-vae-ft-mse)
      2. diffusers from full SD model (CompVis/stable-diffusion-v1-4)
      3. Simple conv stub for smoke tests
    """
    
    DEFAULT_EMBED_DIM = 4 * 64 * 48  # 4 channels × 64 × 48 spatial

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "stub"
        self._model = None
        self.embed_dim = self.DEFAULT_EMBED_DIM
        self._load()

    def _load(self):
        # Try diffusers standalone VAE (sd-vae-ft-mse)
        try:
            from diffusers import AutoencoderKL
            self._model = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float32,
            ).to(self.device).eval()
            self._backend = "sd_vae_mse"
            print("[VAEMetric] Using stabilityai/sd-vae-ft-mse VAE encoder.")
            return
        except Exception as e:
            print(f"[VAEMetric] sd-vae-ft-mse unavailable ({e}).")

        # Try diffusers VAE from full SD model
        try:
            from diffusers import AutoencoderKL
            self._model = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                torch_dtype=torch.float32,
            ).to(self.device).eval()
            self._backend = "sd_v14_vae"
            print("[VAEMetric] Using CompVis/stable-diffusion-v1-4 VAE encoder.")
            return
        except Exception as e:
            print(f"[VAEMetric] SD v1.4 VAE unavailable ({e}).")

        # Try runwayml SD v1.5
        try:
            from diffusers import AutoencoderKL
            self._model = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="vae",
                torch_dtype=torch.float32,
            ).to(self.device).eval()
            self._backend = "sd_v15_vae"
            print("[VAEMetric] Using runwayml/stable-diffusion-v1-5 VAE encoder.")
            return
        except Exception as e:
            print(f"[VAEMetric] SD v1.5 VAE unavailable ({e}).")

        # Stub fallback
        print("[VAEMetric] No VAE backend available. Using stub (random).")
        self._backend = "stub"
        self.embed_dim = 512  # Smaller for stub

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Extract VAE latent embeddings.
        
        Args:
            imgs: (B, 3, H, W) float32 tensor in [0, 1]
            
        Returns:
            (B, D) numpy array of flattened latents
        """
        B = imgs.shape[0]

        if self._backend == "stub":
            return np.random.randn(B, self.embed_dim).astype(np.float32)

        # Normalize to [-1, 1] for SD VAE
        imgs_norm = imgs * 2.0 - 1.0
        imgs_norm = imgs_norm.to(self.device)

        # Resize to multiple of 8 if needed
        H, W = imgs_norm.shape[2], imgs_norm.shape[3]
        new_H = (H // 8) * 8
        new_W = (W // 8) * 8
        if (new_H, new_W) != (H, W):
            imgs_norm = F.interpolate(
                imgs_norm, size=(new_H, new_W), 
                mode="bilinear", align_corners=False
            )

        # Encode
        latent_dist = self._model.encode(imgs_norm).latent_dist
        latents = latent_dist.mean  # (B, 4, H/8, W/8)
        
        # Flatten spatial dimensions
        embs = latents.view(B, -1).cpu().numpy()
        self.embed_dim = embs.shape[1]
        
        return embs.astype(np.float32)

    @property
    def backend_name(self) -> str:
        return self._backend


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation (dataset-level)
# ─────────────────────────────────────────────────────────────────────────────

class VAELatentMetric:
    """
    Computes VAE latent space diversity across a dataset.
    
    Diversity is measured via:
      - Log determinant of embedding covariance (higher = more diverse)
      - Total variance (sum of eigenvalues)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._encoder = _VAEEncoder(device)
        self._embeddings: List[np.ndarray] = []

    def update(self, imgs: torch.Tensor):
        """
        Process a batch of images.
        
        Args:
            imgs: (B, 3, H, W) float32 [0, 1]
        """
        embs = self._encoder(imgs)
        self._embeddings.append(embs)

    def compute(self) -> Dict[str, float]:
        """
        Compute dataset-level VAE latent diversity metrics.
        """
        if not self._embeddings:
            return {
                "vae_diversity_logdet": 0.0,
                "vae_variance_total": 0.0,
                "vae_embed_dim": self._encoder.embed_dim,
                "n_samples": 0,
            }

        E = np.concatenate(self._embeddings, axis=0)  # (N, D)
        N, D = E.shape

        # Covariance matrix
        E_centered = E - E.mean(axis=0, keepdims=True)
        cov = (E_centered.T @ E_centered) / max(N - 1, 1)

        # Regularize
        eps = 1e-6
        cov += eps * np.eye(D)

        # Log determinant
        sign, logdet = np.linalg.slogdet(cov)
        logdet = logdet if sign > 0 else -np.inf

        # Total variance (trace)
        total_var = np.trace(cov)

        return {
            "vae_diversity_logdet": float(logdet),
            "vae_variance_total": float(total_var),
            "vae_embed_dim": D,
            "n_samples": N,
            "backend": self._encoder.backend_name,
        }

    def reset(self):
        self._embeddings = []
