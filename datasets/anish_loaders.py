import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets.base_dataset import BaseTryOnDataset, default_transform, mask_transform

# ─────────────────────────────────────────────────────────────────────────────
# Specialized DressCode Loader
# ─────────────────────────────────────────────────────────────────────────────
class AnishDressCodeDataset(BaseTryOnDataset):
    """Dedicated DressCode loader with overlay generation."""
    def __init__(self, root, split="test", category="upper_body", **kwargs):
        self.category = category
        super().__init__(root, split, **kwargs)

    def _load_samples(self):
        root_dir = self.root
        categories = ['upper_body', 'lower_body', 'dresses'] if self.category == "all" else [self.category]
        data = []
        for cat in categories:
            cat_path = root_dir / cat
            person_dir = cat_path / "image"
            cloth_dir = cat_path / "cloth"
            mask_dir = cat_path / "mask"
            if not person_dir.exists(): continue

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                base_name, ext = os.path.splitext(img_file)
                # DressCode naming convention: person = *_0.jpg, cloth = *_1.jpg
                if base_name.endswith('_0'):
                    cloth_file = base_name[:-1] + '1' + ext
                else:
                    cloth_file = img_file   # fallback: same name
                data.append({
                    'id': base_name,
                    'person_path': person_dir / img_file,
                    'cloth_path': cloth_dir / cloth_file,
                    'mask_path': mask_dir / f"{base_name}.png",
                    'category': cat
                })
        return data

    def __getitem__(self, idx):
        sample = self.samples[idx]
        person_pil = Image.open(sample['person_path']).convert("RGB")
        cloth_pil = Image.open(sample['cloth_path']).convert("RGB")
        mask_pil = Image.open(sample['mask_path']).convert("L")

        # Overlay generation
        person_np = np.array(person_pil)
        mask_np = np.array(mask_pil)
        blurred_mask = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
        alpha = blurred_mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        grey = np.full_like(person_np, 128)
        overlay_np = (person_np * (1 - alpha) + grey * alpha).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        return {
            'person': self.transform(person_pil),
            'cloth': self.transform(cloth_pil),
            'mask': self.mask_tf(mask_pil),
            'overlay_image': self.transform(overlay_pil),
            'meta': {'id': sample['id'], 'category': sample['category'], 'dataset': 'dresscode'}
        }

# ─────────────────────────────────────────────────────────────────────────────
# Specialized VITON-HD Loader
# ─────────────────────────────────────────────────────────────────────────────
class AnishVITONHDDataset(BaseTryOnDataset):
    """Dedicated VITON-HD loader with overlay generation."""
    def _load_samples(self):
        pair_txt = self.root / 'train_pairs_unpaired.txt'
        if not pair_txt.exists():
            pair_txt = self.root / f"{self.split}_pairs.txt"
        
        with open(pair_txt) as f:
            lines = f.readlines()
        
        split_dir = self.root / (self.split if self.split != "train_unpaired" else "train")
        samples = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            p_img, c_img = parts[0], parts[1]
            samples.append({
                'id': os.path.splitext(p_img)[0],
                'person_path': split_dir / 'image' / p_img,
                'cloth_path': split_dir / 'cloth' / c_img,
                'mask_path': split_dir / 'agnostic-mask' / p_img.replace('.jpg', '_mask.png')
            })
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        person_pil = Image.open(sample['person_path']).convert("RGB")
        cloth_pil = Image.open(sample['cloth_path']).convert("RGB")
        mask_pil = Image.open(sample['mask_path']).convert("L")

        # Overlay
        person_np = np.array(person_pil)
        mask_np = np.array(mask_pil)
        blurred_mask = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
        alpha = blurred_mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        grey = np.full_like(person_np, 128)
        overlay_np = (person_np * (1 - alpha) + grey * alpha).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        return {
            'person': self.transform(person_pil),
            'cloth': self.transform(cloth_pil),
            'mask': self.mask_tf(mask_pil),
            'overlay_image': self.transform(overlay_pil),
            'meta': {'id': sample['id'], 'dataset': 'vitonhd'}
        }

# ─────────────────────────────────────────────────────────────────────────────
# Specialized Street-TryOn Loader
# ─────────────────────────────────────────────────────────────────────────────
class AnishStreetTryOnDataset(AnishVITONHDDataset):
    """Dedicated Street-TryOn loader with overlay generation."""
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['meta']['dataset'] = 'street_tryon'
        return out

# ─────────────────────────────────────────────────────────────────────────────
# Specialized CurvTON Loader
# ─────────────────────────────────────────────────────────────────────────────
class AnishCurvTONDataset(AnishVITONHDDataset):
    """Dedicated CurvTON loader with overlay generation."""
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['meta']['dataset'] = 'curvton'
        return out

# ─────────────────────────────────────────────────────────────────────────────
# LAION Streaming Loader
# ─────────────────────────────────────────────────────────────────────────────
class AnishLAIONDataset(Dataset):
    """LAION loader: prefers local HuggingFace-on-disk cache, falls back to streaming."""
    def __init__(self, split="test", limit=1000, img_size=(512, 384), local_dir=None, **kwargs):
        from pathlib import Path as _Path
        self.transform = default_transform(img_size)
        self.data = []

        # Resolve local dataset directory
        if local_dir is None:
            _workspace = _Path(__file__).parent.parent
            local_dir = _workspace / "benchmark_datasets" / "LAION-RVS-Fashion"
        else:
            local_dir = _Path(local_dir)

        if local_dir.exists():
            try:
                import sys as _sys, pathlib as _pathlib
                _ws   = str(_pathlib.Path(__file__).parent.parent)
                _pbak = _sys.path[:]
                _mbak = {k: v for k, v in _sys.modules.items()
                         if k == "datasets" or k.startswith("datasets.")}
                _sys.path = [p for p in _sys.path if p not in (_ws, "")]
                for _k in list(_mbak): _sys.modules.pop(_k, None)
                try:
                    from datasets import load_from_disk as _load_from_disk
                finally:
                    _sys.path[:] = _pbak
                    _sys.modules.update(_mbak)

                ds = _load_from_disk(str(local_dir))
                # DatasetDict or plain Dataset saved with save_to_disk
                try:
                    hf_ds = ds[split]
                except (KeyError, TypeError):
                    hf_ds = ds
                for i in range(min(len(hf_ds), limit)):
                    self.data.append(hf_ds[i])
                return   # loaded from disk – done
            except Exception:
                pass  # fall through to streaming

        # Fallback: stream from HuggingFace Hub
        import sys as _sys, pathlib as _pathlib
        _ws   = str(_pathlib.Path(__file__).parent.parent)
        _pbak = _sys.path[:]
        _mbak = {k: v for k, v in _sys.modules.items()
                 if k == "datasets" or k.startswith("datasets.")}
        _sys.path = [p for p in _sys.path if p not in (_ws, "")]
        for _k in list(_mbak): _sys.modules.pop(_k, None)
        try:
            from datasets import load_dataset as _load_dataset
        finally:
            _sys.path[:] = _pbak
            _sys.modules.update(_mbak)
        self.hf_dataset = _load_dataset("Slep/LAION-RVS-Fashion", streaming=True)[split]
        self.limit = limit
        self._prepare()

        # If the primary stream yielded nothing (e.g. CastError on a parquet shard
        # whose schema differs from the dataset card – typically
        # distractors_metadata.parquet has an extra CATEGORY column), retry by
        # loading only the well-formed data-*.parquet shards via the raw parquet
        # builder, which auto-detects features per-file and never raises CastError.
        if not self.data:
            import warnings
            warnings.warn(
                "[AnishLAIONDataset] Streaming load failed (likely schema mismatch "
                "in distractors_metadata.parquet). Retrying with data-*.parquet shards only.",
                stacklevel=2,
            )
            try:
                _glob = f"hf://datasets/Slep/LAION-RVS-Fashion/data/{split}/data-*.parquet"
                self.hf_dataset = _load_dataset(
                    "parquet",
                    data_files={split: _glob},
                    streaming=True,
                )[split]
                self._prepare()
            except Exception:
                pass  # leave self.data empty; caller handles missing data

    def _prepare(self):
        it = iter(self.hf_dataset)
        for _ in range(self.limit):
            try:
                self.data.append(next(it))
            except StopIteration:
                break
            except Exception:
                # CastError (schema mismatch) or other per-shard failure terminates
                # the HuggingFace generator entirely – break rather than spinning.
                break

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_pil = item.get('person_image') or item.get('image') or Image.new('RGB', (512, 512), (128,128,128))
        c_pil = item.get('cloth_image') or item.get('cloth') or Image.new('RGB', (512, 512), (128,128,128))
        
        if not isinstance(p_pil, Image.Image): p_pil = Image.fromarray(np.array(p_pil))
        if not isinstance(c_pil, Image.Image): c_pil = Image.fromarray(np.array(c_pil))

        return {
            'person': self.transform(p_pil.convert("RGB")),
            'cloth': self.transform(c_pil.convert("RGB")),
            'mask': torch.ones(1, 512, 512), # Dummy mask
            'meta': {'id': f"laion_{idx}", 'dataset': 'laion'}
        }

# ─────────────────────────────────────────────────────────────────────────────
# Multi-purpose Collate
# ─────────────────────────────────────────────────────────────────────────────
def anish_collate_fn(batch):
    """Standardize collation for specialized loaders."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == 'meta':
            out[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
