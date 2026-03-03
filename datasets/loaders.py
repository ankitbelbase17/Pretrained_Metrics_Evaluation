"""
datasets/loaders.py
--------------------
10 concrete dataset loaders for Virtual Try-On evaluation.
Each loader extends BaseTryOnDataset and implements _load_samples().

Datasets:
  1.  VITON           – classic paired try-on benchmark (256×192)
  2.  VITON-HD        – high-res VITON variant (1024×768)
  3.  DressCode        – multi-category garment benchmark
  4.  MPV              – multi-pose virtual try-on
  5.  DeepFashion-TryOn – fashion compatibility pairs
  6.  ACGPN            – ACGPN dataset split
  7.  CP-VTON          – cloth-person VTON original split
  8.  HR-VTON          – high-resolution VTON
  9.  LaDI-VTON        – latent diffusion tryon dataset
  10. OVNet            – outfit-virtual-net dataset

HOW TO ADAPT
------------
Each loader reads the ACTUAL folder structure of the respective dataset.
Point `root` to the dataset root directory when instantiating the loader.
The docstring of each class describes the expected folder layout.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from .base_dataset import BaseTryOnDataset
from .anish_loaders import (
    AnishDressCodeDataset,
    AnishVITONHDDataset,
    AnishLAIONDataset,
    AnishStreetTryOnDataset,
    AnishCurvTONDataset
)

# Standalone dataloaders (created independently in dataloaders/ package)
try:
    from dataloaders.dresscode_dataloader import Dresscode as StandaloneDresscode
    from dataloaders.vitonhd_dataloader import VITONHDDataset as StandaloneVITONHD
    from dataloaders.street_tryon_dataloader import GeneralTryOnDataset as StandaloneStreetTryOn
    from dataloaders.laion_rvs_fashion_dataloader import LAIONRVSFashionDataset as StandaloneLAION
    _HAS_STANDALONE = True
except ImportError:
    _HAS_STANDALONE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. VITON
# ─────────────────────────────────────────────────────────────────────────────
class VITONDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/          <- person images (*.jpg)
            cloth/          <- garment images (*.jpg)
            image-parse/    <- segmentation masks (*.png)
          test_pairs.txt    <- "person_name.jpg cloth_name.jpg" per line
    The ground-truth is the person image itself (paired benchmark).
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                person_path = self.root / self.split / "image" / person_name
                cloth_path  = self.root / self.split / "cloth" / cloth_name
                gt_path     = person_path  # paired – person IS the GT
                mask_path   = self.root / self.split / "image-parse" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=person_path,
                    cloth_path=cloth_path,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. VITON-HD
# ─────────────────────────────────────────────────────────────────────────────
class VITONHDDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/          <- person images (*.jpg)
            cloth/          <- garment flat images (*.jpg)
            agnostic-mask/  <- binary body mask (*.png)
          test_pairs.txt
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                person_path = self.root / self.split / "image" / person_name
                cloth_path  = self.root / self.split / "cloth" / cloth_name
                gt_path     = person_path
                mask_path   = self.root / self.split / "agnostic-mask" / person_name.replace(".jpg", "_mask.png")
                samples.append(dict(
                    id=person_name,
                    person_path=person_path,
                    cloth_path=cloth_path,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 3. DressCode
# ─────────────────────────────────────────────────────────────────────────────
class DressCodeDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          upper_body/  (or lower_body / dresses)
            images/        <- person + GT images (id_0.jpg = person, id_1.jpg = GT)
            clothes/       <- flat garment images (id_1.jpg)
            skeletons/     <- pose skeletons (id_0.json) [optional]
            label_maps/    <- segmentation maps (id_0.png) [optional]
          <category>_pairs_<split>.txt
    `category` defaults to 'upper_body'.
    """

    def __init__(self, root, split="test", category="upper_body", **kwargs):
        self.category = category
        super().__init__(root, split, **kwargs)

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.category}_pairs_{self.split}.txt"
        if not pairs_file.exists():
            # Try alternate naming
            pairs_file = self.root / f"{self.split}_pairs.txt"

        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        cat_dir = self.root / self.category
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_id, cloth_id = parts[0], parts[1]
                person_path = cat_dir / "images"  / f"{person_id}_0.jpg"
                gt_path     = cat_dir / "images"  / f"{person_id}_1.jpg"
                cloth_path  = cat_dir / "clothes" / f"{cloth_id}_1.jpg"
                mask_path   = cat_dir / "label_maps" / f"{person_id}_0.png"
                samples.append(dict(
                    id=person_id,
                    person_path=person_path,
                    cloth_path=cloth_path,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. MPV (Multi-Pose Virtual Try-On)
# ─────────────────────────────────────────────────────────────────────────────
class MPVDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          MPV_dataset/
            image/         <- person images
            cloth/         <- garment images
            pose/          <- DensePose or OpenPose JSONs [optional]
          all_poseA_poseB_clothes.txt   <- "person cloth gt" TSV
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / "all_poseA_poseB_clothes.txt"
        if not pairs_file.exists():
            # Fallback
            possible = list(self.root.glob("*.txt"))
            if possible:
                pairs_file = possible[0]
            else:
                raise FileNotFoundError(f"No pairs file found in {self.root}")

        dataset_dir = self.root / "MPV_dataset"
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                person_name = parts[0]
                cloth_name  = parts[1]
                gt_name     = parts[2] if len(parts) > 2 else parts[0]

                samples.append(dict(
                    id=person_name,
                    person_path=dataset_dir / "image" / person_name,
                    cloth_path=dataset_dir / "cloth"  / cloth_name,
                    gt_path=dataset_dir / "image"     / gt_name,
                    mask_path=None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5. DeepFashion-TryOn
# ─────────────────────────────────────────────────────────────────────────────
class DeepFashionTryOnDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/       <- person images
            cloth/       <- garment images
            gt/          <- ground-truth try-on results
            mask/        <- body/cloth masks [optional]
          test_pairs.txt   "person_img   cloth_img   gt_img"
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name = parts[0]
                cloth_name  = parts[1]
                gt_name     = parts[2] if len(parts) > 2 else parts[0]

                mask_path = base / "mask" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "gt"    / gt_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACGPN
# ─────────────────────────────────────────────────────────────────────────────
class ACGPNDataset(BaseTryOnDataset):
    """
    Expected structure (mirrors VITON naming kept by ACGPN paper):
        <root>/
          test/
            image/
            cloth/
            cloth-mask/   <- white/black garment mask
            image-parse/
          test_pairs.txt
    GT = person image (paired).
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                mask_path = base / "image-parse" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "image" / person_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 7. CP-VTON
# ─────────────────────────────────────────────────────────────────────────────
class CPVTONDataset(BaseTryOnDataset):
    """
    Expected structure (identical to VITON; re-defined separately for clarity):
        <root>/
          test/
            image/
            cloth/
            cloth-mask/
          test_pairs.txt
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "image" / person_name,
                    mask_path=None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 8. HR-VTON
# ─────────────────────────────────────────────────────────────────────────────
class HRVTONDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/            <- person images (1024×768)
            cloth/
            cloth_mask/
            agnostic/         <- agnostic-v3.2 person images
            agnostic_mask/
            image_parse_v3/   <- ATR or CIHP parsing maps (*.png)
          test_pairs.txt
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                mask_path = base / "image_parse_v3" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "image" / person_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 9. LaDI-VTON
# ─────────────────────────────────────────────────────────────────────────────
class LaDIVTONDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            images/       <- person images
            clothes/      <- garment images
            gt/           <- ground-truth results (if available)
            masks/        <- segmentation masks
          test_pairs.txt   "person_name  cloth_name"
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                gt_path   = base / "gt"   / person_name
                if not gt_path.exists():
                    gt_path = base / "images" / person_name  # fallback
                mask_path = base / "masks" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "images"  / person_name,
                    cloth_path=base  / "clothes" / cloth_name,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 10. OVNet
# ─────────────────────────────────────────────────────────────────────────────
class OVNetDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            person/     <- wearing reference (person image)
            outfit/     <- target outfit / garment images
            gt/         <- ground-truth composed results
            mask/       <- body parsing mask
          test_pairs.txt  "person_name outfit_name gt_name"
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name = parts[0]
                outfit_name = parts[1]
                gt_name     = parts[2] if len(parts) > 2 else parts[0]

                mask_path = base / "mask" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "person" / person_name,
                    cloth_path=base  / "outfit" / outfit_name,
                    gt_path=base     / "gt"     / gt_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 11. StreetTryOn
# ─────────────────────────────────────────────────────────────────────────────
class StreetTryOnDataset(VITONHDDataset):
    """
    Street-style try-on dataset. Mirrors VITON-HD structure.
    """
    pass

# ─────────────────────────────────────────────────────────────────────────────
# 12. CurvTON (Custom Dataset)
# ─────────────────────────────────────────────────────────────────────────────
class CurvTONDataset(VITONHDDataset):
    """
    Custom CurvTON dataset. Mirrors VITON-HD structure.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "viton":              VITONDataset,
    "vton":               VITONDataset,
    "viton_hd":           VITONHDDataset,
    "vitonhd":            VITONHDDataset,
    "dresscode":          DressCodeDataset,
    "mpv":                MPVDataset,
    "deepfashion_tryon":  DeepFashionTryOnDataset,
    "deepfashion":        DeepFashionTryOnDataset,
    "acgpn":              ACGPNDataset,
    "cp_vton":            CPVTONDataset,
    "hr_vton":            HRVTONDataset,
    "ladi_vton":          LaDIVTONDataset,
    "ovnet":              OVNetDataset,

    "street_tryon":       StreetTryOnDataset,
    "curvton":            CurvTONDataset,

    # Specialized "Anish" Dedicated Loaders
    "dresscode_anish":    AnishDressCodeDataset,
    "vitonhd_anish":      AnishVITONHDDataset,
    "laion_anish":        AnishLAIONDataset,
    "laion":              AnishLAIONDataset,
    "laion_fashion":      AnishLAIONDataset,
    "street_tryon_anish": AnishStreetTryOnDataset,
    "curvton_anish":      AnishCurvTONDataset,
}

# Register standalone dataloaders if available
if _HAS_STANDALONE:
    DATASET_REGISTRY.update({
        "dresscode_standalone":    StandaloneDresscode,
        "vitonhd_standalone":      StandaloneVITONHD,
        "street_tryon_standalone": StandaloneStreetTryOn,
        "laion_standalone":        StandaloneLAION,
    })

# Names that use non-standard constructors (not BaseTryOnDataset)
_STANDALONE_NAMES = {
    "dresscode_standalone", "vitonhd_standalone",
    "street_tryon_standalone", "laion_standalone",
}
_LAION_NAMES = {
    "laion", "laion_anish", "laion_fashion", "laion_standalone",
}


def get_dataset(name: str, root: str, **kwargs):
    """
    Factory function; returns the appropriate dataset by name.

    Handles three kinds of constructors:
      1. BaseTryOnDataset subclasses   → (root=, split=, img_size=, ...)
      2. LAION streaming loaders       → (split=, limit=, img_size=, ...)
      3. Standalone dataloaders from dataloaders/ package → custom signatures
    """
    name = name.lower().replace("-", "_")
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )

    # ── LAION (no root needed) ────────────────────────────────────────────
    if name in _LAION_NAMES:
        return DATASET_REGISTRY[name](**kwargs)

    # ── Standalone dataloaders (custom constructors) ──────────────────────
    if name in _STANDALONE_NAMES:
        cls = DATASET_REGISTRY[name]
        if name == "dresscode_standalone":
            return cls(root_dir=root)
        elif name == "vitonhd_standalone":
            return cls(
                data_root_path=root,
                output_dir=kwargs.get("output_dir", "output"),
                eval_pair=kwargs.get("eval_pair", False),
            )
        elif name == "street_tryon_standalone":
            config = kwargs.get("config", {})
            split = kwargs.get("split", "test")
            return cls(dataroot=root, config=config, split=split)
        else:
            return cls(**kwargs)

    # ── Standard BaseTryOnDataset subclasses ──────────────────────────────
    return DATASET_REGISTRY[name](root=root, **kwargs)
