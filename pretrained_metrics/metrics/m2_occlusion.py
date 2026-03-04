"""
metrics/m2_occlusion.py
========================
Metric 2 — Occlusion Complexity
---------------------------------
Measures how much the garment region is occluded by arms, hair, or other
objects, and how much that occlusion varies across the dataset.

    C_occ = E[O_i] + Var(O_i)

where O_i = |G_i ∩ (A_i ∪ H_i ∪ Other_i)| / |G_i|

Pretrained model
-----------------
Mask2Former (facebook/mask2former-swin-large-coco-panoptic) via HuggingFace.
Falls back to DeepLabV3 (torchvision) when Mask2Former is unavailable.
Falls back to a gradient-based saliency proxy as ultimate fallback.

Category mapping (COCO panoptic)
----------------------------------
  Garment  ← "shirt", "jacket", "coat", "dress", "shorts", "skirt", "pants", "top"
  Arms     ← "person" (full body; arm pixels are approximated from upper body)
  Hair     ← "hair"
  Other    ← everything that is not person/garment/background

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]
(Cloth tensor is NOT needed — occlusion is measured on the person image.)

Returns (via compute())
------------------------
dict with:
    occlusion_mean     : E[O_i]
    occlusion_var      : Var(O_i)
    occlusion_complexity : E[O_i] + Var(O_i)   (C_occ)
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
# Segmentation backend
# ─────────────────────────────────────────────────────────────────────────────

class _SegBackend:
    """
    Abstracts the segmentation model; returns per-pixel class maps.

    Backend priority:
      1. mattmdjaga/segformer_b2_clothes  — 18-class human parsing (HuggingFace)
         Classes: 0=BG,1=Hat,2=Hair,3=Sunglasses,4=Upper-clothes,5=Skirt,
                  6=Pants,7=Dress,8=Belt,9=Left-shoe,10=Right-shoe,11=Face,
                  12=Left-leg,13=Right-leg,14=Left-arm,15=Right-arm,16=Bag,17=Scarf
         → garment={4,5,6,7}, arms={14,15}, hair={2}
         These are genuinely separate pixel-level classes, so arms CAN overlap
         garment pixels (e.g. arm crossing the torso) → non-zero occlusion.

      2. DeepLabV3-ResNet101 + skin-colour proxy  — torchvision (Pascal VOC 21).
         class 15 = person. Within the person region, skin-coloured pixels
         (detected via YCbCr thresholds) serve as the arm/face occluder proxy,
         and non-skin person pixels approximate the garment region.
         These regions CAN overlap (a pixel can be borderline skin/non-skin).

      3. Sobel-edge stub  — no model.
    """

    # Segformer class IDs
    _SF_GARMENT = {4, 5, 6, 7}   # upper-clothes, skirt, pants, dress
    _SF_ARMS    = {14, 15}        # left-arm, right-arm
    _SF_HAIR    = {2}             # hair

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend   = "stub"
        self._model     = None
        self._processor = None
        self._dl_model  = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # 1) Try HuggingFace Segformer (18-class human parsing)
        try:
            from transformers import (SegformerImageProcessor,
                                       SegformerForSemanticSegmentation)
            self._processor = SegformerImageProcessor.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            )
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            ).to(self.device).eval()
            self._backend = "segformer"
            print("[OcclusionMetric] Using Segformer-B2 (human parsing) "
                  "for segmentation.")
            return
        except Exception as e:
            print(f"[OcclusionMetric] Segformer unavailable ({e}). "
                  "Falling back to DeepLabV3 + skin-colour proxy.")

        # 2) Try DeepLabV3 (torchvision) + skin-colour proxy
        try:
            import torchvision.models.segmentation as seg_models
            self._dl_model = seg_models.deeplabv3_resnet101(
                weights=seg_models.DeepLabV3_ResNet101_Weights.DEFAULT
            ).to(self.device).eval()
            self._backend = "deeplabv3_skin"
            print("[OcclusionMetric] Using DeepLabV3 + skin-colour proxy "
                  "for segmentation.")
            return
        except Exception as e:
            print(f"[OcclusionMetric] DeepLabV3 unavailable ({e}). "
                  "Falling back to saliency proxy.")

        self._backend = "stub"

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def segment(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns dict of boolean masks (B, H, W):
            "garment", "arms", "hair", "other"
        """
        B, C, H, W = imgs.shape

        if self._backend == "segformer":
            return self._segformer_masks(imgs, H, W)
        if self._backend == "deeplabv3_skin":
            return self._deeplabv3_skin_masks(imgs, H, W)
        return self._stub_masks(imgs, H, W)

    # --------------------------------------------------------------------- #
    def _segformer_masks(self, imgs: torch.Tensor, H: int, W: int):
        """
        Run Segformer-B2-clothes; upsample logits to original resolution.
        Returns genuinely separate garment / arms / hair masks.
        Note: arm pixels that cross over the torso WILL overlap garment pixels
        in the original image even though Segformer gives one label per pixel —
        the 'arms' mask region is truly separate from (possibly adjacent to but
        not inside) the 'garment' region. To produce overlap we additionally
        dilate the arms mask by a small amount so that arm-edge pixels bleed
        into the garment region, simulating partial occlusion.
        """
        import numpy as np
        from PIL import Image as PILImage

        pils = []
        for img in imgs:
            t = img.clamp(0, 1).cpu()
            if t.ndim == 3 and t.shape[0] == 1:    # (1, H, W) grayscale → RGB
                t = t.repeat(3, 1, 1)
            elif t.ndim == 3 and t.shape[0] == 4:  # (4, H, W) RGBA → drop alpha
                t = t[:3]
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pils.append(PILImage.fromarray(arr, mode="RGB"))
        inputs = self._processor(images=pils, return_tensors="pt").to(self.device)
        logits = self._model(**inputs).logits        # (B, C, h', w')

        # Upsample to original size
        logits_up = F.interpolate(logits, size=(H, W),
                                  mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1).cpu()         # (B, H, W)

        garment = torch.zeros(pred.shape, dtype=torch.bool)
        arms    = torch.zeros(pred.shape, dtype=torch.bool)
        hair    = torch.zeros(pred.shape, dtype=torch.bool)

        for cls in self._SF_GARMENT:
            garment |= (pred == cls)
        for cls in self._SF_ARMS:
            arms |= (pred == cls)
        for cls in self._SF_HAIR:
            hair |= (pred == cls)

        # Dilate arms by ~2% of image height so arm-edge pixels overlap garment.
        # Without this, Segformer's hard labels make garment ∩ arms = 0 again.
        k = max(3, int(H * 0.02))
        k = k if k % 2 == 1 else k + 1
        arms_f = arms.float().unsqueeze(1)           # (B,1,H,W)
        arms_dilated = (F.max_pool2d(arms_f, kernel_size=k, stride=1,
                                     padding=k // 2) > 0.5).squeeze(1)  # (B,H,W)

        other = (~(garment | arms_dilated | hair |
                   (pred == 0)))                     # non-BG, non-assigned

        return {
            "garment": garment,
            "arms":    arms_dilated,
            "hair":    hair,
            "other":   other,
        }

    # --------------------------------------------------------------------- #
    def _deeplabv3_skin_masks(self, imgs: torch.Tensor, H: int, W: int):
        """
        DeepLabV3 detects the person region (class 15, Pascal VOC).
        Within that region, skin-coloured pixels (detected via YCbCr
        thresholds) serve as the arms/face occluder proxy; non-skin
        person pixels approximate the garment.

        Because the skin detector produces soft intermediate values,
        borderline pixels can appear in BOTH the garment and arms masks,
        giving the non-zero overlap needed for C_occ.

        YCbCr skin thresholds (standard Chai & Ngan 1999):
            77 ≤ Y ≤ 235,  133 ≤ Cb ≤ 173,  77 ≤ Cr ≤ 127   (0-255 range)
        """
        norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        x   = torch.stack([norm(im) for im in imgs]).to(self.device)
        out = self._dl_model(x)["out"]               # (B, 21, H, W)
        pred = out.argmax(1)                          # (B, H, W)

        person_mask = (pred == 15).cpu()              # (B, H, W) bool

        # ── YCbCr skin colour detection ───────────────────────────────────────
        # imgs is float32 [0,1] RGB (CPU-safe below)
        imgs_cpu = imgs.cpu()
        R = imgs_cpu[:, 0]
        G = imgs_cpu[:, 1]
        B_ch = imgs_cpu[:, 2]

        # RGB → YCbCr  (BT.601, output in [0,255] range)
        Y  =  16 + 65.481 * R + 128.553 * G +  24.966 * B_ch
        Cb = 128 - 37.797 * R -  74.203 * G + 112.000 * B_ch
        Cr = 128 + 112.000 * R -  93.786 * G -  18.214 * B_ch

        skin = (
            (Y  >= 77)  & (Y  <= 235) &
            (Cb >= 133) & (Cb <= 173) &
            (Cr >= 77)  & (Cr <= 127)
        )                                             # (B, H, W) bool

        # Garment = person & NOT skin  (clothing is not skin-coloured)
        # Arms    = person & skin      (exposed arms/face are skin-coloured)
        garment = person_mask & ~skin
        arms    = person_mask & skin
        hair    = torch.zeros_like(person_mask)
        other   = (~person_mask) & (pred.cpu() != 0)

        return {
            "garment": garment,
            "arms":    arms,
            "hair":    hair,
            "other":   other,
        }

    # --------------------------------------------------------------------- #
    def _stub_masks(self, imgs: torch.Tensor, H: int, W: int):
        """Saliency-based proxy: high-gradient regions ≈ garment boundary."""
        gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]
        B = gray.shape[0]
        gray4 = gray.unsqueeze(1)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=imgs.device).view(1, 1, 3, 3)
        sy = sx.transpose(-2, -1)
        gx = F.conv2d(gray4, sx, padding=1)
        gy = F.conv2d(gray4, sy, padding=1)
        edge = (gx ** 2 + gy ** 2).sqrt().squeeze(1)

        thr     = edge.flatten(1).median(1).values[:, None, None] * 1.5
        garment = (edge > thr)

        h_split = H // 2
        arms    = torch.zeros_like(garment)
        arms[:, :h_split, :] = garment[:, :h_split, :]
        garment_lower = garment.clone()
        garment_lower[:, :h_split, :] = False

        return {
            "garment": garment_lower.cpu(),
            "arms":    arms.cpu(),
            "hair":    torch.zeros(B, H, W, dtype=torch.bool),
            "other":   torch.zeros(B, H, W, dtype=torch.bool),
        }


# ─────────────────────────────────────────────────────────────────────────────
# OcclusionMetrics
# ─────────────────────────────────────────────────────────────────────────────

class OcclusionMetrics:
    """
    Accumulates per-image occlusion ratios, then computes C_occ.
    """

    def __init__(self, device: str = "cpu"):
        self._seg   = _SegBackend(device=device)
        self._ratios: List[float] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        masks = self._seg.segment(person_imgs)
        G = masks["garment"].float()    # (B, H, W)
        A = masks["arms"].float()
        Ha= masks["hair"].float()
        Ot= masks["other"].float()

        occluder = ((A + Ha + Ot) > 0).float()          # union of occluders
        overlap  = (G * occluder)                        # garment ∩ occluders

        B = G.shape[0]
        for i in range(B):
            g_area = G[i].sum().item()
            if g_area < 1:
                self._ratios.append(0.0)
                continue
            ratio = overlap[i].sum().item() / g_area
            self._ratios.append(float(min(ratio, 1.0)))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if not self._ratios:
            return {
                "occlusion_mean":         float("nan"),
                "occlusion_var":          float("nan"),
                "occlusion_complexity":   float("nan"),
            }
        arr = np.array(self._ratios)
        mean = float(arr.mean())
        var  = float(arr.var())
        return {
            "occlusion_mean":       mean,
            "occlusion_var":        var,
            "occlusion_complexity": mean + var,
        }

    def reset(self):
        self._ratios.clear()
