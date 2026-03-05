"""
on_the_wild_evaluation/triplet_dataloader.py
=============================================
Triplet Dataloader for In-the-Wild Virtual Try-On Evaluation

Provides (person_image, cloth_image, tryon_image) triplets for evaluation.

Triplet Structure:
------------------
  1. person_image  : Original person image (pose reference)
  2. cloth_image   : Target garment/clothing image (appearance reference)
  3. tryon_image   : Generated try-on result (what we evaluate)

Metric → Input Mapping:
-----------------------
  - VLM Score          : person_image + cloth_image + tryon_image (all three)
  - Pose Consistency   : person_image + tryon_image (pose preservation)
  - NIQE               : tryon_image only (no-reference quality)
  - MUSIQ              : tryon_image only (no-reference quality)
  - JEPA               : tryon_image only (self-consistency)
  - CLIP Garment       : cloth_image + tryon_image (garment fidelity)

Usage:
------
    from on_the_wild_evaluation.triplet_dataloader import TripletDataLoader
    
    loader = TripletDataLoader(
        person_dir="path/to/persons",
        cloth_dir="path/to/clothes",
        tryon_dir="path/to/tryons",
        batch_size=8,
    )
    
    for batch in loader:
        person_imgs = batch["person"]      # (B, 3, H, W)
        cloth_imgs = batch["cloth"]        # (B, 3, H, W)
        tryon_imgs = batch["tryon"]        # (B, 3, H, W)
        paths = batch["paths"]             # List of triplet paths
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from PIL import Image
import torchvision.transforms.functional as TF


class TripletDataLoader:
    """
    DataLoader for virtual try-on triplets.
    
    Supports multiple matching strategies:
      1. Filename matching: person_001.jpg ↔ cloth_001.jpg ↔ tryon_001.jpg
      2. CSV/JSON manifest file
      3. Separate directories with aligned order
      4. Custom matching function
    """
    
    def __init__(
        self,
        # Directory-based loading
        person_dir: Optional[str] = None,
        cloth_dir: Optional[str] = None,
        tryon_dir: Optional[str] = None,
        # Path list-based loading
        person_paths: Optional[List[str]] = None,
        cloth_paths: Optional[List[str]] = None,
        tryon_paths: Optional[List[str]] = None,
        # Manifest-based loading
        manifest_path: Optional[str] = None,
        # Settings
        batch_size: int = 4,
        img_size: Tuple[int, int] = (512, 384),  # (H, W)
        match_by: str = "filename",  # "filename", "order", "manifest"
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        """
        Args:
            person_dir: Directory containing person images
            cloth_dir: Directory containing cloth/garment images
            tryon_dir: Directory containing generated try-on results
            person_paths: Direct list of person image paths
            cloth_paths: Direct list of cloth image paths
            tryon_paths: Direct list of try-on image paths
            manifest_path: Path to JSON/CSV manifest file
            batch_size: Batch size for iteration
            img_size: Target image size (H, W)
            match_by: How to match triplets ("filename", "order", "manifest")
            extensions: Valid image extensions
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.match_by = match_by
        self.extensions = extensions
        
        # Build triplet list
        self._triplets: List[Tuple[str, str, str]] = []
        
        if manifest_path:
            self._load_from_manifest(manifest_path)
        elif person_paths and cloth_paths and tryon_paths:
            self._build_from_paths(person_paths, cloth_paths, tryon_paths)
        elif person_dir and cloth_dir and tryon_dir:
            self._build_from_dirs(person_dir, cloth_dir, tryon_dir)
        else:
            raise ValueError(
                "Must provide either (person_dir, cloth_dir, tryon_dir), "
                "(person_paths, cloth_paths, tryon_paths), or manifest_path"
            )
        
        print(f"[TripletDataLoader] Loaded {len(self._triplets)} triplets")
    
    def _list_images(self, directory: str) -> Dict[str, str]:
        """List images in directory, returns {basename: full_path}."""
        directory = Path(directory)
        images = {}
        for ext in self.extensions:
            for p in directory.glob(f"*{ext}"):
                # Extract base name without extension
                base = self._extract_base_name(p.name)
                images[base] = str(p)
            for p in directory.glob(f"*{ext.upper()}"):
                base = self._extract_base_name(p.name)
                images[base] = str(p)
        return images
    
    def _extract_base_name(self, filename: str) -> str:
        """
        Extract base identifier from filename.
        
        Handles patterns like:
          - person_001.jpg → 001
          - cloth_001.jpg → 001
          - 001_person.jpg → 001
          - tryon_result_001.png → 001
        """
        # Remove extension
        name = Path(filename).stem
        
        # Try to extract numeric ID
        # Pattern 1: prefix_ID (e.g., person_001)
        match = re.search(r'[_-](\d+)$', name)
        if match:
            return match.group(1)
        
        # Pattern 2: ID_suffix (e.g., 001_person)
        match = re.search(r'^(\d+)[_-]', name)
        if match:
            return match.group(1)
        
        # Pattern 3: just numeric (e.g., 00001.jpg)
        if name.isdigit():
            return name
        
        # Pattern 4: contains numeric anywhere
        match = re.search(r'(\d+)', name)
        if match:
            return match.group(1)
        
        # Fallback: use full name
        return name
    
    def _build_from_dirs(self, person_dir: str, cloth_dir: str, tryon_dir: str):
        """Build triplet list from directories."""
        person_imgs = self._list_images(person_dir)
        cloth_imgs = self._list_images(cloth_dir)
        tryon_imgs = self._list_images(tryon_dir)
        
        if self.match_by == "filename":
            # Match by extracted base name
            common_ids = set(person_imgs.keys()) & set(cloth_imgs.keys()) & set(tryon_imgs.keys())
            for base_id in sorted(common_ids):
                self._triplets.append((
                    person_imgs[base_id],
                    cloth_imgs[base_id],
                    tryon_imgs[base_id],
                ))
            
            if len(common_ids) < len(person_imgs):
                print(f"[TripletDataLoader] Warning: Only {len(common_ids)} matched "
                      f"out of {len(person_imgs)} person images")
        
        elif self.match_by == "order":
            # Match by sorted order
            person_list = sorted(person_imgs.values())
            cloth_list = sorted(cloth_imgs.values())
            tryon_list = sorted(tryon_imgs.values())
            
            n = min(len(person_list), len(cloth_list), len(tryon_list))
            for i in range(n):
                self._triplets.append((person_list[i], cloth_list[i], tryon_list[i]))
    
    def _build_from_paths(
        self, 
        person_paths: List[str], 
        cloth_paths: List[str], 
        tryon_paths: List[str]
    ):
        """Build triplet list from path lists."""
        n = min(len(person_paths), len(cloth_paths), len(tryon_paths))
        for i in range(n):
            self._triplets.append((person_paths[i], cloth_paths[i], tryon_paths[i]))
    
    def _load_from_manifest(self, manifest_path: str):
        """Load triplets from manifest file (JSON or CSV)."""
        manifest_path = Path(manifest_path)
        
        if manifest_path.suffix.lower() == ".json":
            import json
            with open(manifest_path) as f:
                data = json.load(f)
            
            # Expected format: [{"person": "...", "cloth": "...", "tryon": "..."}, ...]
            for entry in data:
                self._triplets.append((
                    entry["person"],
                    entry["cloth"],
                    entry["tryon"],
                ))
        
        elif manifest_path.suffix.lower() == ".csv":
            import csv
            with open(manifest_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._triplets.append((
                        row["person"],
                        row["cloth"],
                        row["tryon"],
                    ))
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]), Image.LANCZOS)
        return TF.to_tensor(img)  # (3, H, W) float32 [0, 1]
    
    def __len__(self) -> int:
        return len(self._triplets)
    
    def __iter__(self) -> Iterator[Dict[str, Union[torch.Tensor, List[Tuple[str, str, str]]]]]:
        """
        Iterate over batches.
        
        Yields:
            Dict with keys:
                "person": (B, 3, H, W) person images
                "cloth": (B, 3, H, W) cloth images
                "tryon": (B, 3, H, W) try-on images
                "paths": List of (person_path, cloth_path, tryon_path) tuples
        """
        for i in range(0, len(self._triplets), self.batch_size):
            batch_triplets = self._triplets[i:i + self.batch_size]
            
            person_imgs = []
            cloth_imgs = []
            tryon_imgs = []
            
            for person_path, cloth_path, tryon_path in batch_triplets:
                person_imgs.append(self._load_image(person_path))
                cloth_imgs.append(self._load_image(cloth_path))
                tryon_imgs.append(self._load_image(tryon_path))
            
            yield {
                "person": torch.stack(person_imgs),
                "cloth": torch.stack(cloth_imgs),
                "tryon": torch.stack(tryon_imgs),
                "paths": batch_triplets,
            }
    
    def get_triplet(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple[str, str, str]]]:
        """Get a single triplet by index."""
        person_path, cloth_path, tryon_path = self._triplets[idx]
        return {
            "person": self._load_image(person_path),
            "cloth": self._load_image(cloth_path),
            "tryon": self._load_image(tryon_path),
            "paths": (person_path, cloth_path, tryon_path),
        }
    
    @property
    def triplets(self) -> List[Tuple[str, str, str]]:
        """Get all triplet paths."""
        return self._triplets


class TripletDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for triplets.
    
    Can be used with torch.utils.data.DataLoader for multi-worker loading.
    """
    
    def __init__(
        self,
        person_dir: Optional[str] = None,
        cloth_dir: Optional[str] = None,
        tryon_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        img_size: Tuple[int, int] = (512, 384),
        **kwargs,
    ):
        # Create internal loader for triplet management
        self._loader = TripletDataLoader(
            person_dir=person_dir,
            cloth_dir=cloth_dir,
            tryon_dir=tryon_dir,
            manifest_path=manifest_path,
            batch_size=1,  # We handle batching via DataLoader
            img_size=img_size,
            **kwargs,
        )
    
    def __len__(self) -> int:
        return len(self._loader)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple[str, str, str]]]:
        return self._loader.get_triplet(idx)


def collate_triplets(
    batch: List[Dict]
) -> Dict[str, Union[torch.Tensor, List[Tuple[str, str, str]]]]:
    """
    Collate function for TripletDataset with DataLoader.
    
    Usage:
        dataset = TripletDataset(...)
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_triplets)
    """
    return {
        "person": torch.stack([b["person"] for b in batch]),
        "cloth": torch.stack([b["cloth"] for b in batch]),
        "tryon": torch.stack([b["tryon"] for b in batch]),
        "paths": [b["paths"] for b in batch],
    }
