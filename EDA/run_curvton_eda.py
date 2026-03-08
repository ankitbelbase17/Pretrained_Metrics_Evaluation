"""
EDA/run_curvton_eda.py
========================
CURVTON Dataset EDA Pipeline

Generates publication-quality EDA plots for CURVTON dataset:
- Individual plots for Easy, Medium, Hard splits
- Combined overlay plots comparing difficulty levels
- Support for different sample ratios (10%, 20%, ..., 100%)

Usage:
    # Single GPU:
    python EDA/run_curvton_eda.py \
        --base_path /path/to/dataset_ultimate \
        --out_dir figures/curvton \
        --sample_ratio 1.0

    # Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 EDA/run_curvton_eda.py \
        --base_path /path/to/dataset_ultimate \
        --out_dir figures/curvton \
        --sample_ratio 1.0

    # For test set:
    python EDA/run_curvton_eda.py \
        --base_path /path/to/dataset_ultimate_test \
        --out_dir figures/curvton_test \
        --sample_ratio 1.0
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess as _sp
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders.curvton_dataloader import CURVTONDataloader
from EDA.feature_extractor import FeatureExtractor
from EDA.plot_style import (
    apply_paper_style, DATASET_COLORS, DATASET_MARKERS, DATASET_LINESTYLES,
    FILL_ALPHA, LINE_ALPHA, CURVTON_COLORS,
)

# Import all EDA plot modules
from EDA.plots.p1_pose_eda import plot_pose_umap, plot_joint_angle_distributions
from EDA.plots.p2_occlusion_eda import plot_occlusion_histogram, plot_occlusion_heatmap
from EDA.plots.p3_background_eda import plot_bg_entropy_histogram, plot_entropy_vs_objects
from EDA.plots.p4_illumination_eda import plot_luminance_spectrum, plot_illumination_pca
from EDA.plots.p5_body_shape_eda import plot_shape_pca, plot_shape_coefficient_histograms
from EDA.plots.p6_appearance_eda import plot_face_umap, plot_pairwise_distance_distribution
from EDA.plots.p7_garment_eda import plot_garment_umap, plot_eigenvalue_spectrum
from EDA.plots.p11_clip_embedding_eda import run_clip_embedding_eda

apply_paper_style()

import torch.distributed as dist

def _setup_distributed():
    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return rank, world, device

def _is_distributed(): return dist.is_available() and dist.is_initialized()
def _print_rank0(msg, rank=0):
    if rank == 0:
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# CURVTON-specific colors — ColorBrewer Dark2 (Maximally Distinct)
# Teal vs Orange vs Purple — avoids perceptual confusion
# ═══════════════════════════════════════════════════════════════════════════════

# Use the centralized CURVTON_COLORS from plot_style.py
# Fallback if not imported:
_CURVTON_COLORS = {
    "Easy":   "#1B9E77",  # Dark teal-green (cool tone)
    "Medium": "#D95F02",  # Dark orange (warm tone)  
    "Hard":   "#7570B3",  # Muted purple-blue (distinct from both)
}

# Merge with imported CURVTON_COLORS if available
try:
    _CURVTON_COLORS.update(CURVTON_COLORS)
except:
    pass

CURVTON_MARKERS = {
    "Easy":   "o",  # circle (round = easy)
    "Medium": "s",  # square (angular = moderate)
    "Hard":   "^",  # triangle (sharp = difficult)
}

CURVTON_LINESTYLES = {
    "Easy":   "-",   # solid
    "Medium": "--",  # dashed
    "Hard":   "-.",  # dash-dot
}

# Line widths for emphasis
CURVTON_LINEWIDTHS = {
    "Easy":   2.0,
    "Medium": 2.2,
    "Hard":   2.4,
}


# ── Batched feature-extraction constants ──────────────────────────────────────
_EXTRACT_BATCH_SIZE = 32        # images per GPU batch
_CHECKPOINT_EVERY   = 2000      # save intermediate .npz every N images
_FEATURE_KEYS       = [
    "pose_vecs", "angles", "occlusion", "bg_entropy",
    "bg_obj_count", "lum_mean", "lum_grad_var",
    "betas", "face_embs", "garment_embs",
]


def _batched(loader, tf, batch_size, verbose=False,
             shard_rank=None, shard_world_size=None, num_workers=16):
    """Yield (person_batch, cloth_batch) tensors from the loader using parallel PyTorch DataLoader."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    
    _TRANSFORM = T.Compose([T.Resize((512, 384)), T.ToTensor()])
    
    class CURVTONTensorDataset(Dataset):
        def __init__(self, c_loader):
            self.samples = c_loader.samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            from PIL import Image
            s = self.samples[idx]
            try:
                person = _TRANSFORM(Image.open(s["person_path"]).convert("RGB"))
                cloth = _TRANSFORM(Image.open(s["cloth_path"]).convert("RGB"))
            except Exception:
                person = torch.zeros(3, 512, 384)
                cloth = torch.zeros(3, 512, 384)
            return person, cloth

    dataset = CURVTONTensorDataset(loader)
    
    if shard_world_size is not None and shard_world_size > 1:
        indices = list(range(shard_rank, len(dataset), shard_world_size))
        dataset = torch.utils.data.Subset(dataset, indices)
        
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    total = len(dataset)
    for i, (p_batch, c_batch) in enumerate(dl):
        if verbose and (i * batch_size) % 500 == 0:
            print(f"    {i * batch_size}/{total} ...")
        yield p_batch, c_batch


# ── Per-metric extraction functions (one model loaded at a time) ──────────────

def _extract_pose(loader, tf, device, batch_size, verbose, **kw):
    import math as _math
    from pretrained_metrics.metrics.m1_pose import (
        _KeypointExtractor, _normalise_pose, _joint_angle, TRIPLET_IDX,
    )
    if verbose:
        print("\n    [pose] Loading backend...")
    backend = _KeypointExtractor(device)
    pose_vecs, angles = [], []
    for person_batch, _ in _batched(loader, tf, batch_size, verbose, **kw):
        kps_raw = backend(person_batch)
        kps_norm, valid = _normalise_pose(kps_raw)
        for i in range(person_batch.shape[0]):
            if valid[i]:
                pn = kps_norm[i]
                pose_vecs.append(pn.flatten().astype(np.float32))
                ang = [_joint_angle(pn[ia], pn[ib], pn[ic]) for ia, ib, ic in TRIPLET_IDX]
                angles.append(np.array(
                    [a if not _math.isnan(a) else 0.0 for a in ang], dtype=np.float32))
            else:
                pose_vecs.append(np.zeros(34, dtype=np.float32))
                angles.append(np.zeros(len(TRIPLET_IDX), dtype=np.float32))
    del backend; _free_gpu()
    if verbose:
        print(f"    [pose] Done ({len(pose_vecs)} samples)")
    return {"pose_vecs": pose_vecs, "angles": angles}


def _extract_occlusion(loader, tf, device, batch_size, verbose, **kw):
    from pretrained_metrics.metrics.m2_occlusion import _SegBackend
    if verbose:
        print("\n    [occlusion] Loading backend...")
    backend = _SegBackend(device)
    occ_vals = []
    for person_batch, _ in _batched(loader, tf, batch_size, verbose, **kw):
        seg = backend.segment(person_batch)
        G = seg["garment"].float()
        occ = ((seg["arms"].float() + seg["hair"].float() + seg["other"].float()) > 0).float()
        overlap = G * occ
        for i in range(person_batch.shape[0]):
            g_area = G[i].sum().item()
            occ_vals.append(float(min(overlap[i].sum().item() / max(g_area, 1.0), 1.0)))
    del backend; _free_gpu()
    if verbose:
        print(f"    [occlusion] Done ({len(occ_vals)} samples)")
    return {"occlusion": occ_vals}


def _extract_background(loader, tf, device, batch_size, verbose, **kw):
    import math as _math
    from pretrained_metrics.metrics.m3_background import (
        _PersonSegmenter, _texture_entropy, _ObjectDetector,
    )
    if verbose:
        print("\n    [background] Loading backends...")
    per_seg = _PersonSegmenter(device)
    obj_det = _ObjectDetector(device)
    bg_entropy, bg_obj_count = [], []
    for person_batch, _ in _batched(loader, tf, batch_size, verbose, **kw):
        pmask = per_seg(person_batch)
        obj_c = obj_det.count_objects(person_batch, pmask)
        for i in range(person_batch.shape[0]):
            ent = _texture_entropy(person_batch[i], pmask[i])
            bg_entropy.append(float(ent) if not _math.isnan(ent) else 0.0)
            bg_obj_count.append(int(obj_c[i]))
    del per_seg, obj_det; _free_gpu()
    if verbose:
        print(f"    [background] Done ({len(bg_entropy)} samples)")
    return {"bg_entropy": bg_entropy, "bg_obj_count": bg_obj_count}


def _extract_illumination(loader, tf, device, batch_size, verbose, **kw):
    from pretrained_metrics.metrics.m4_illumination import (
        _rgb_to_lab_l, _sobel_gradient_variance,
    )
    if verbose:
        print("\n    [illumination] Extracting (no model)...")
    lum_mean, lum_grad_var = [], []
    for person_batch, _ in _batched(loader, tf, batch_size, verbose, **kw):
        mean_L, L_maps = _rgb_to_lab_l(person_batch.cpu())
        for i in range(person_batch.shape[0]):
            lum_mean.append(float(mean_L[i]))
            lum_grad_var.append(_sobel_gradient_variance(L_maps[i]))
    if verbose:
        print(f"    [illumination] Done ({len(lum_mean)} samples)")
    return {"lum_mean": lum_mean, "lum_grad_var": lum_grad_var}


def _extract_body_shape(loader, tf, device, batch_size, verbose, **kw):
    from pretrained_metrics.metrics.m5_body_shape import _ShapeExtractor
    if verbose:
        print("\n    [body_shape] Loading backend...")
    backend = _ShapeExtractor(device)
    betas = []
    for person_batch, _ in _batched(loader, tf, batch_size, verbose, **kw):
        b = backend(person_batch)
        for bi in b:
            betas.append(bi.astype(np.float32))
    del backend; _free_gpu()
    if verbose:
        print(f"    [body_shape] Done ({len(betas)} samples)")
    return {"betas": betas}


def _extract_appearance(loader, tf, device, batch_size, verbose, **kw):
    from pretrained_metrics.metrics.m6_appearance import _FaceEmbedder
    if verbose:
        print("\n    [appearance] Loading backend...")
    backend = _FaceEmbedder(device)
    face_embs = []
    for person_batch, _ in _batched(loader, tf, batch_size, verbose, **kw):
        f = backend(person_batch)
        for fi in f:
            face_embs.append(fi.astype(np.float32))
    del backend; _free_gpu()
    if verbose:
        print(f"    [appearance] Done ({len(face_embs)} samples)")
    return {"face_embs": face_embs}


def _extract_garment(loader, tf, device, batch_size, verbose, **kw):
    from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder
    if verbose:
        print("\n    [garment] Loading backend...")
    backend = _GarmentEncoder(device)
    garment_embs = []
    for _, cloth_batch in _batched(loader, tf, batch_size, verbose, **kw):
        g = backend(cloth_batch)
        for gi in g:
            garment_embs.append(gi.astype(np.float32))
    del backend; _free_gpu()
    if verbose:
        print(f"    [garment] Done ({len(garment_embs)} samples)")
    return {"garment_embs": garment_embs}


# Ordered list: (name, feature_keys, extraction_function)
_METRIC_EXTRACTORS = [
    ("pose",         ["pose_vecs", "angles"],        _extract_pose),
    ("occlusion",    ["occlusion"],                  _extract_occlusion),
    ("background",   ["bg_entropy", "bg_obj_count"], _extract_background),
    ("illumination", ["lum_mean", "lum_grad_var"],   _extract_illumination),
    ("body_shape",   ["betas"],                      _extract_body_shape),
    ("appearance",   ["face_embs"],                  _extract_appearance),
    ("garment",      ["garment_embs"],               _extract_garment),
]


def extract_features_for_difficulty(
    loader: CURVTONDataloader,
    extractor: Optional["FeatureExtractor"],
    cache_path: Path,
    force_recompute: bool = False,
    batch_size: int = _EXTRACT_BATCH_SIZE,
    device: str = None,
    num_workers: int = 16,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, np.ndarray]:
    """Extract EDA features, automatically sharding if world_size > 1."""
    if cache_path.exists() and not force_recompute:
        _print_rank0(f"  Loading cached features from {cache_path}", rank)
        if rank == 0:
            return dict(np.load(cache_path, allow_pickle=True))
        return {}
    
    N = len(loader)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = None

    _print_rank0(f"  Extracting features for {N} samples (batch_size={batch_size})...", rank)
    
    shard_dir = cache_path.parent / f".shards_{cache_path.stem}"
    if rank == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
    if _is_distributed():
        dist.barrier()
        
    metric_cache_dir = shard_dir / f"rank_{rank}"
    metric_cache_dir.mkdir(parents=True, exist_ok=True)
    
    features: Dict[str, list] = {k: [] for k in _FEATURE_KEYS}

    for name, keys, extract_fn in _METRIC_EXTRACTORS:
        metric_file = metric_cache_dir / f"{name}.npz"
        if metric_file.exists() and not force_recompute:
            if rank == 0:
                print(f"    [{name}] Loading from rank {rank} metric cache...")
            cached = dict(np.load(metric_file, allow_pickle=True))
            for k in keys:
                if k in cached and cached[k].size > 0:
                    features[k] = list(cached[k])
            del cached
            continue

        try:
            sub = extract_fn(loader, tf, device, batch_size, verbose=(rank==0),
                             shard_rank=rank, shard_world_size=world_size, num_workers=num_workers)
            for k in keys:
                features[k] = sub[k]
            arrays = {}
            for k in keys:
                if sub[k]:
                    arrays[k] = (np.array(sub[k]) if np.isscalar(sub[k][0]) else np.stack(sub[k]))
            np.savez_compressed(metric_file, **arrays)
        except Exception as e:
            print(f"    [Rank {rank}] Warning: {name} extraction failed: {e}")

    result: Dict[str, np.ndarray] = {}
    for k in _FEATURE_KEYS:
        if features[k]:
            result[k] = (np.array(features[k]) if np.isscalar(features[k][0]) else np.stack(features[k]))
        else:
            result[k] = np.array([])
            
    shard_path = shard_dir / f"shard_{rank}.npz"
    np.savez_compressed(shard_path, **result)
    
    del features
    gc.collect()
    
    if _is_distributed():
        dist.barrier()
        
    if rank == 0:
        return _merge_feature_shards(shard_dir, world_size, cache_path)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-GPU extraction via subprocesses
# ═══════════════════════════════════════════════════════════════════════════════




def _merge_feature_shards(
    shard_dir: Path,
    num_shards: int,
    final_path: Path,
) -> Dict[str, np.ndarray]:
    """Concatenate per-GPU shard ``.npz`` files into a single cache."""
    merged: Dict[str, list] = {k: [] for k in _FEATURE_KEYS}

    for rank in range(num_shards):
        sp = shard_dir / f"shard_{rank}.npz"
        if not sp.exists():
            print(f"  Warning: shard {sp} missing, skipping")
            continue
        d = dict(np.load(sp, allow_pickle=True))
        for k in _FEATURE_KEYS:
            if k in d and d[k].size > 0:
                merged[k].append(d[k])
        del d

    result: Dict[str, np.ndarray] = {}
    for k in _FEATURE_KEYS:
        result[k] = np.concatenate(merged[k]) if merged[k] else np.array([])

    final_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(final_path, **result)

    total = max((len(v) for v in result.values() if v.size > 0), default=0)
    print(f"  Merged {num_shards} shards → {final_path}  ({total} samples)")

    # Clean up shard files and directory
    for rank in range(num_shards):
        sp = shard_dir / f"shard_{rank}.npz"
        if sp.exists():
            sp.unlink()
    try:
        shard_dir.rmdir()
    except OSError:
        pass

    return result





def run_curvton_eda(
    base_path: str,
    out_dir: str = "figures/curvton",
    cache_dir: str = "eda_cache/curvton",
    sample_ratio: float = 1.0,
    difficulties: List[str] = ["easy", "medium", "hard"],
    force_recompute: bool = False,
    batch_size: int = _EXTRACT_BATCH_SIZE,
    num_workers: int = 16,
):
    """Run full CURVTON EDA pipeline with native PyTorch distributed scaling."""
    rank, world, device = _setup_distributed()

    out_path = Path(out_dir)
    cache_path = Path(cache_dir)
    
    if rank == 0:
        out_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print("CURVTON EDA Pipeline (Distributed)")
        print("=" * 70)
        print(f"  Base path:    {base_path}")
        print(f"  Output:       {out_dir}")
        print(f"  Device:       {device} (World Size: {world})")
        print(f"  Batch size:   {batch_size}")
        print(f"  Sample ratio: {sample_ratio:.0%}")
        print(f"  Difficulties: {difficulties}")
        print("=" * 70)
    
    if _is_distributed(): dist.barrier()

    all_features = {}
    
    for diff in difficulties:
        _print_rank0(f"\n[{diff.upper()}] Loading dataset...", rank)
        
        loader = CURVTONDataloader(
            base_path=base_path,
            difficulty=diff,
            sample_ratio=sample_ratio,
            return_paths=True,
        )
        
        cache_file = cache_path / f"curvton_{diff}_{int(sample_ratio*100)}pct.npz"
        features = extract_features_for_difficulty(
            loader, None, cache_file, force_recompute,
            batch_size=batch_size, device=device, num_workers=num_workers,
            rank=rank, world_size=world,
        )
        
        display_name = diff.capitalize()
        all_features[display_name] = features
    
    # ── Generators wait at barrier, only rank 0 does EDA plotting ──
    if _is_distributed(): dist.barrier()
    if rank != 0:
        return

    for diff in ["Easy", "Medium", "Hard"]:
        DATASET_COLORS[diff] = CURVTON_COLORS[diff]
        DATASET_MARKERS[diff] = CURVTON_MARKERS[diff]
        DATASET_LINESTYLES[diff] = CURVTON_LINESTYLES[diff]
    
    print("\n" + "=" * 70)
    print("Generating EDA Plots")
    print("=" * 70)
    
    # --- 1. Pose EDA ---
    print("\n[1/7] Pose Distribution...")
    pose_data = {k: v["pose_vecs"] for k, v in all_features.items() if len(v.get("pose_vecs", [])) > 0}
    angle_data = {k: v["angles"] for k, v in all_features.items() if len(v.get("angles", [])) > 0}
    
    if pose_data:
        plot_pose_umap(pose_data, str(out_path / "pose"))
    if angle_data:
        plot_joint_angle_distributions(angle_data, str(out_path / "pose"))
    
    # --- 2. Occlusion EDA ---
    print("\n[2/7] Occlusion Distribution...")
    occ_data = {k: v["occlusion"] for k, v in all_features.items() if len(v.get("occlusion", [])) > 0}
    
    if occ_data:
        plot_occlusion_histogram(occ_data, str(out_path / "occlusion"))
    
    # --- 3. Background EDA ---
    print("\n[3/7] Background Complexity...")
    bg_ent_data = {k: v["bg_entropy"] for k, v in all_features.items() if len(v.get("bg_entropy", [])) > 0}
    bg_obj_data = {k: v["bg_obj_count"] for k, v in all_features.items() if len(v.get("bg_obj_count", [])) > 0}
    
    if bg_ent_data:
        plot_bg_entropy_histogram(bg_ent_data, str(out_path / "background"))
    if bg_ent_data and bg_obj_data:
        plot_entropy_vs_objects(bg_ent_data, bg_obj_data, str(out_path / "background"))
    
    # --- 4. Illumination EDA ---
    print("\n[4/7] Illumination Analysis...")
    lum_data = {k: v["lum_mean"] for k, v in all_features.items() if len(v.get("lum_mean", [])) > 0}
    grad_data = {k: v["lum_grad_var"] for k, v in all_features.items() if len(v.get("lum_grad_var", [])) > 0}
    
    if lum_data:
        plot_luminance_spectrum(lum_data, grad_data, str(out_path / "illumination"))
    
    # --- 5. Body Shape EDA ---
    print("\n[5/7] Body Shape Distribution...")
    beta_data = {k: v["betas"] for k, v in all_features.items() if len(v.get("betas", [])) > 0}
    
    if beta_data:
        plot_shape_pca(beta_data, str(out_path / "body_shape"))
        plot_shape_coefficient_histograms(beta_data, str(out_path / "body_shape"))
    
    # --- 6. Appearance EDA ---
    print("\n[6/7] Appearance Diversity...")
    face_data = {k: v["face_embs"] for k, v in all_features.items() if len(v.get("face_embs", [])) > 0}
    
    if face_data:
        plot_face_umap(face_data, str(out_path / "appearance"))
        plot_pairwise_distance_distribution(face_data, str(out_path / "appearance"))
    
    # --- 7. Garment EDA ---
    print("\n[7/8] Garment Diversity...")
    garment_data = {k: v["garment_embs"] for k, v in all_features.items() if len(v.get("garment_embs", [])) > 0}
    
    if garment_data:
        plot_garment_umap(garment_data, str(out_path / "garment"))
        plot_eigenvalue_spectrum(garment_data, str(out_path / "garment"))
    
    # --- 8. CLIP Embedding EDA (Image + Text) ---
    print("\n[8/8] CLIP Embedding Analysis (20% sample)...")
    # Run CLIP embedding EDA at 20% sample ratio for efficiency
    clip_sample_ratio = min(sample_ratio, 0.2)  # Cap at 20% for CLIP analysis
    run_clip_embedding_eda(
        base_path=base_path,
        out_dir=str(out_path / "clip_embeddings"),
        cache_dir=str(cache_path.parent / "clip"),
        sample_ratio=clip_sample_ratio,
        device="cuda" if torch.cuda.is_available() else "cpu",
        force_recompute=force_recompute,
    )
    
    print("\n" + "=" * 70)
    print(f"EDA Complete! Figures saved to: {out_path}")
    print("=" * 70)


def run_multi_ratio_eda(
    base_path: str,
    out_dir: str = "figures/curvton",
    cache_dir: str = "eda_cache/curvton",
    ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    batch_size: int = _EXTRACT_BATCH_SIZE,
    num_workers: int = 16,
):
    """
    Run EDA for multiple sample ratios to analyze scaling behavior.
    """
    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        print(f"\n{'#' * 70}")
        print(f"# Running EDA for {ratio_pct}% of CURVTON dataset")
        print(f"{'#' * 70}")
        
        run_curvton_eda(
            base_path=base_path,
            out_dir=f"{out_dir}/{ratio_pct}pct",
            cache_dir=cache_dir,
            sample_ratio=ratio,
            batch_size=batch_size,
            num_workers=num_workers,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CURVTON Dataset EDA Pipeline")
    parser.add_argument(
        "--base_path", type=str, required=True,
        help="Path to CURVTON dataset (dataset_ultimate or dataset_ultimate_test)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="figures/curvton",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="eda_cache/curvton",
        help="Cache directory for extracted features"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=1.0,
        help="Fraction of dataset to use (0.1 to 1.0)"
    )
    parser.add_argument(
        "--multi_ratio", action="store_true",
        help="Run EDA for multiple sample ratios (10%%, 20%%, ..., 100%%)"
    )
    parser.add_argument(
        "--difficulties", type=str, nargs="+", default=["easy", "medium", "hard"],
        help="Difficulty levels to process"
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Force recomputation of cached features"
    )


    args = parser.parse_args()

    if args.multi_ratio:
        # Multi-ratio does percentages
        ratios = [0.1, 0.2, 0.3, 0.4, 1.0]
        run_multi_ratio_eda(
            base_path=args.base_path,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            ratios=ratios,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        run_curvton_eda(
            base_path=args.base_path,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            sample_ratio=args.sample_ratio,
            difficulties=args.difficulties,
            force_recompute=args.force_recompute,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
