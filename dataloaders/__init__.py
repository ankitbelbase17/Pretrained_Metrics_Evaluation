"""
dataloaders/__init__.py
========================
Package init for the standalone dataloaders.

Provides:
  - Direct access to dataset classes and collate functions
  - Adapter wrappers that translate dataloader output dicts to the canonical
    format expected by pretrained_metrics/, EDA/, and evaluate.py:

        { "person": Tensor, "cloth": Tensor, "gt": Tensor,
          "mask": Tensor, "meta": list[dict] }

  - A `DATALOADER_REGISTRY` mapping names → (Dataset class, collate_fn)
  - `get_dataloader_adapted()` factory that returns a DataLoader producing
    batches in the canonical format
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

# ── Concrete dataset classes ─────────────────────────────────────────────────
from dataloaders.dresscode_dataloader import (
    Dresscode,
    custom_collate_fn as dresscode_collate_fn,
)
from dataloaders.vitonhd_dataloader import (
    VITONHDDataset,
    custom_collate_fn as vitonhd_collate_fn,
)
from dataloaders.street_tryon_dataloader import GeneralTryOnDataset
from dataloaders.laion_rvs_fashion_dataloader import LAIONRVSFashionDataset


# ─────────────────────────────────────────────────────────────────────────────
# Canonical-format adapter collate
# ─────────────────────────────────────────────────────────────────────────────

def _renorm_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] (mean=0.5, std=0.5 normalisation) back to [0, 1]."""
    return tensor * 0.5 + 0.5


def canonical_collate_dresscode(batch: List[Dict]) -> Dict:
    """
    Collate Dresscode samples into the canonical format.
    Converts key names and re-normalises images from [-1,1] → [0,1].
    """
    person_images = []
    cloth_images = []
    masks = []
    metas = []

    for item in batch:
        person_images.append(item["person_image"])
        cloth_images.append(item["cloth_image"])
        masks.append(item["mask"].unsqueeze(0) if item["mask"].dim() == 2 else item["mask"])
        metas.append({
            "id": item["filename"],
            "dataset": "dresscode",
            "caption": item.get("caption", ""),
        })

    person_t = _renorm_to_01(torch.stack(person_images))
    cloth_t = _renorm_to_01(torch.stack(cloth_images))

    return {
        "person": person_t,
        "cloth": cloth_t,
        "gt": person_t.clone(),          # paired benchmark: gt = person
        "mask": torch.stack(masks),
        "meta": metas,
    }


def canonical_collate_vitonhd(batch: List[Dict]) -> Dict:
    """
    Collate VITON-HD samples into the canonical format.
    Converts key names and re-normalises images from [-1,1] → [0,1].
    """
    person_images = []
    cloth_images = []
    masks = []
    metas = []

    for item in batch:
        person_images.append(item["person_image"])
        cloth_images.append(item["cloth_image"])
        masks.append(item["mask"].unsqueeze(0) if item["mask"].dim() == 2 else item["mask"])
        metas.append({
            "id": item["filename"],
            "dataset": "vitonhd",
        })

    person_t = _renorm_to_01(torch.stack(person_images))
    cloth_t = _renorm_to_01(torch.stack(cloth_images))

    return {
        "person": person_t,
        "cloth": cloth_t,
        "gt": person_t.clone(),
        "mask": torch.stack(masks),
        "meta": metas,
    }


def canonical_collate_street_tryon(batch: List[Dict]) -> Dict:
    """
    Collate StreetTryOn / GeneralTryOnDataset samples into canonical format.
    Keys: pimg→person, gimg→cloth, pseg→mask.
    Images are already in [-1,1]; convert to [0,1].
    """
    persons = []
    cloths = []
    masks = []
    metas = []

    for item in batch:
        persons.append(item["pimg"])
        cloths.append(item["gimg"])
        # pseg is (1, H, W)
        masks.append(item["pseg"].float())
        metas.append({
            "id": item["person_fn"],
            "dataset": "street_tryon",
            "garment_fn": item["garment_fn"],
        })

    person_t = _renorm_to_01(torch.stack(persons))
    cloth_t = _renorm_to_01(torch.stack(cloths))

    return {
        "person": person_t,
        "cloth": cloth_t,
        "gt": person_t.clone(),
        "mask": torch.stack(masks),
        "meta": metas,
    }


def canonical_collate_laion(batch: List[Dict]) -> Dict:
    """
    Collate LAION-RVS-Fashion samples into canonical format.
    """
    persons = []
    cloths = []
    metas = []

    for item in batch:
        persons.append(item["person"])
        cloths.append(item["cloth"])
        metas.append(item.get("meta", {"id": str(item.get("idx", 0)), "dataset": "laion"}))

    person_t = torch.stack(persons)
    cloth_t = torch.stack(cloths)

    return {
        "person": person_t,
        "cloth": cloth_t,
        "gt": person_t.clone(),
        "mask": torch.ones(len(batch), 1, person_t.shape[-2], person_t.shape[-1]),
        "meta": metas,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

DATALOADER_REGISTRY = {
    "dresscode_standalone":   (Dresscode, canonical_collate_dresscode),
    "vitonhd_standalone":     (VITONHDDataset, canonical_collate_vitonhd),
    "street_tryon_standalone": (GeneralTryOnDataset, canonical_collate_street_tryon),
    "laion_standalone":       (LAIONRVSFashionDataset, canonical_collate_laion),
}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader_adapted(
    name: str,
    dataset_kwargs: dict,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    use_canonical: bool = True,
) -> DataLoader:
    """
    Build a DataLoader from the standalone dataloaders, optionally wrapping
    output in the canonical format expected by pretrained_metrics/ and EDA/.

    Parameters
    ----------
    name : str
        One of the keys in DATALOADER_REGISTRY.
    dataset_kwargs : dict
        Keyword arguments forwarded to the dataset constructor.
    batch_size : int
    num_workers : int
    shuffle : bool
    use_canonical : bool
        If True, use the canonical collate function that outputs
        {person, cloth, gt, mask, meta} in [0,1] range.
        If False, use the original collate function from the dataloader module.

    Returns
    -------
    DataLoader
    """
    if name not in DATALOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataloader '{name}'. Available: {list(DATALOADER_REGISTRY)}"
        )

    DatasetCls, canonical_collate = DATALOADER_REGISTRY[name]
    ds = DatasetCls(**dataset_kwargs)

    # Choose collate fn
    if use_canonical:
        collate_fn = canonical_collate
    else:
        # Use original collate functions
        if "dresscode" in name:
            collate_fn = dresscode_collate_fn
        elif "vitonhd" in name:
            collate_fn = vitonhd_collate_fn
        else:
            collate_fn = None  # default PyTorch collate

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
    )
