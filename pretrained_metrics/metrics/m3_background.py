"""
metrics/m3_background.py
=========================
Metric 3 — Background Complexity
----------------------------------

3A. Background Texture Entropy
    H_bg = -Σ p_j log p_j  (histogram entropy over 256-bin grayscale)

3B. Background Object Density
    C_obj = E[n_i]  where n_i = # objects detected in the background region

Pretrained models
------------------
- DeepLabV3-ResNet101 (torchvision) → person/background segmentation
- DETR (facebook/detr-resnet-50)    → object detection for density
  Falls back to connected-components counting when DETR is unavailable.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (via compute())
------------------------
dict with:
    bg_entropy_mean          : mean texture entropy across dataset
    bg_entropy_var           : variance of texture entropy
    bg_object_density_mean   : mean #objects in background per image
    bg_complexity_3A         : bg_entropy_mean  (alias)
    bg_complexity_3B         : bg_object_density_mean  (alias)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# DeepLabV3 person-masker
# ─────────────────────────────────────────────────────────────────────────────

class _PersonSegmenter:
    """Returns a binary person mask (B, H, W) from DeepLabV3."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._load()

    def _load(self):
        try:
            import torchvision.models.segmentation as s
            self._model = s.deeplabv3_resnet101(
                weights=s.DeepLabV3_ResNet101_Weights.DEFAULT
            ).to(self.device).eval()
            print("[BackgroundMetric] DeepLabV3 loaded for person segmentation.")
        except Exception as e:
            print(f"[BackgroundMetric] DeepLabV3 unavailable ({e}). "
                  "Using brightness-threshold person proxy.")
            self._model = None

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Returns (B, H, W) bool tensor — True = person pixel."""
        B, C, H, W = imgs.shape
        if self._model is not None:
            norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            x    = torch.stack([norm(im) for im in imgs]).to(self.device)
            out  = self._model(x)["out"]          # (B, 21, H, W)
            pred = out.argmax(1)                   # (B, H, W)
            return (pred == 15).cpu()              # class 15 = person

        # Proxy: centre of the image ≈ person
        mask = torch.zeros(B, H, W, dtype=torch.bool)
        ch, cw = H // 4, W // 4
        mask[:, ch: H - ch, cw: W - cw] = True
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Texture entropy
# ─────────────────────────────────────────────────────────────────────────────

def _texture_entropy(bg_rgb: torch.Tensor, person_mask: torch.Tensor) -> float:
    """
    bg_rgb      : (3, H, W) float [0,1]
    person_mask : (H, W) bool — True = person (exclude)
    Returns scalar entropy of background grayscale histogram.
    """
    # Grayscale
    gray = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]  # (H,W)
    bg_pixels = gray[~person_mask].numpy()

    if bg_pixels.size == 0:
        return float("nan")

    bg_pixels = np.clip(bg_pixels, 0.0, 1.0)
    counts, _ = np.histogram(bg_pixels, bins=256, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return float("nan")
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


# ─────────────────────────────────────────────────────────────────────────────
# Object detector
# ─────────────────────────────────────────────────────────────────────────────

class _ObjectDetector:
    """DETR → number of objects in background region."""

    CONF_THRESHOLD = 0.5

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._model   = None
        self._feature = None
        self._backend = "stub"
        self._load()

    def _load(self):
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self._feature = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50"
            )
            self._model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50"
            ).to(self.device).eval()
            self._backend = "detr"
            print("[BackgroundMetric] DETR loaded for object density.")
        except Exception as e:
            print(f"[BackgroundMetric] DETR unavailable ({e}). "
                  "Using connected-components object proxy.")
            self._backend = "components"

    @torch.no_grad()
    def count_objects(
        self, imgs: torch.Tensor, person_masks: torch.Tensor
    ) -> List[int]:
        """
        imgs         : (B, 3, H, W) float [0,1]
        person_masks : (B, H, W  )  bool
        Returns List[int] — number of background objects per image.
        """
        B = imgs.shape[0]

        if self._backend == "detr":
            import torchvision.transforms.functional as TF

            # Batch person-masking on CPU
            imgs_masked = imgs.clone()
            for i in range(B):
                imgs_masked[i, :, person_masks[i]] = 0.0

            # PIL conversion (all at once)
            pils = []
            for i in range(B):
                pil = TF.to_pil_image(imgs_masked[i].clamp(0, 1).cpu()).convert("RGB")
                if pil.width < 32 or pil.height < 32:
                    pil = pil.resize((224, 224))
                pils.append(pil)

            # Batched DETR forward (single pass)
            inputs = self._feature(
                images=pils,
                return_tensors="pt",
                input_data_format="channels_last",
            ).to(self.device)
            outs = self._model(**inputs)

            # Per-image confidence thresholding (cheap CPU indexing)
            counts = []
            for i in range(B):
                probs = outs.logits.softmax(-1)[i, :, :-1]
                conf  = probs.max(-1).values
                n_obj = int((conf > self.CONF_THRESHOLD).sum().item())
                counts.append(n_obj)
            return counts

        # Proxy: connected components of high-gradient background pixels
        return self._component_count(imgs, person_masks)

    def _component_count(
        self, imgs: torch.Tensor, person_masks: torch.Tensor
    ) -> List[int]:
        """Count connected edge-components in background as object proxy."""
        try:
            from skimage.measure import label as sk_label
        except ImportError:
            return [0] * imgs.shape[0]

        counts = []
        for i in range(imgs.shape[0]):
            gray = (0.299 * imgs[i, 0] + 0.587 * imgs[i, 1]
                    + 0.114 * imgs[i, 2]).numpy()
            bg   = ~person_masks[i].numpy()
            gray_bg = gray * bg

            # Sobel gradient magnitude
            from scipy.ndimage import sobel
            gx = sobel(gray_bg, axis=0)
            gy = sobel(gray_bg, axis=1)
            mag = np.sqrt(gx ** 2 + gy ** 2)

            thr = np.percentile(mag[bg], 75) if bg.any() else 0.0
            edge_map = (mag > thr) & bg

            labeled  = sk_label(edge_map)
            n_comp   = int(labeled.max())
            counts.append(n_comp)
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# BackgroundMetrics
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundMetrics:
    """Accumulates background texture entropy and object density."""

    def __init__(self, device: str = "cpu"):
        self._segmenter = _PersonSegmenter(device)
        self._detector  = _ObjectDetector(device)
        self._entropies: List[float] = []
        self._obj_counts: List[int]  = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        person_masks = self._segmenter(person_imgs)      # (B,H,W) bool
        obj_counts   = self._detector.count_objects(person_imgs, person_masks)

        for i in range(person_imgs.shape[0]):
            ent = _texture_entropy(person_imgs[i], person_masks[i])
            self._entropies.append(ent)

        self._obj_counts.extend(obj_counts)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        ent = np.array([v for v in self._entropies if not math.isnan(v)])
        obj = np.array(self._obj_counts, dtype=float)

        return {
            "bg_entropy_mean":        float(ent.mean()) if len(ent) else float("nan"),
            "bg_entropy_var":         float(ent.var())  if len(ent) else float("nan"),
            "bg_object_density_mean": float(obj.mean()) if len(obj) else float("nan"),
            "bg_complexity_3A":       float(ent.mean()) if len(ent) else float("nan"),
            "bg_complexity_3B":       float(obj.mean()) if len(obj) else float("nan"),
        }

    def reset(self):
        self._entropies.clear()
        self._obj_counts.clear()
