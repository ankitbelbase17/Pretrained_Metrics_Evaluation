"""
EDA/run_curvton_eda.py
========================
CURVTON Dataset EDA Pipeline

Generates publication-quality EDA plots for CURVTON dataset:
- Individual plots for Easy, Medium, Hard splits
- Combined overlay plots comparing difficulty levels
- Support for different sample ratios (10%, 20%, ..., 100%)

Usage:
    python EDA/run_curvton_eda.py \
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


def _load_image_tensor(path: str, tf) -> torch.Tensor:
    """Load one image → (3, H, W) float32 [0, 1]."""
    from PIL import Image
    return tf(Image.open(path).convert("RGB"))


def extract_features_for_difficulty(
    loader: CURVTONDataloader,
    extractor: FeatureExtractor,
    cache_path: Path,
    force_recompute: bool = False,
    batch_size: int = _EXTRACT_BATCH_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Extract EDA features for a CURVTON difficulty split.

    Uses **batched** model inference (not per-image) so that GPU
    utilisation is high and wall-clock time is reasonable.  Writes
    intermediate checkpoints every ``_CHECKPOINT_EVERY`` images so
    progress survives OOM / time-limit kills.

    Returns dict with keys: pose_vecs, angles, occlusion, bg_entropy,
                           lum_mean, betas, face_embs, garment_embs, etc.
    """
    # ── fast-return if already cached ─────────────────────────────────
    if cache_path.exists() and not force_recompute:
        print(f"  Loading cached features from {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))

    N = len(loader)
    print(f"  Extracting features for {N} samples (batch_size={batch_size})...")

    # Check for partial checkpoint
    ckpt_path = cache_path.with_suffix(".ckpt.npz")
    start_idx = 0
    features: Dict[str, list] = {k: [] for k in _FEATURE_KEYS}

    if ckpt_path.exists() and not force_recompute:
        ckpt = dict(np.load(ckpt_path, allow_pickle=True))
        start_idx = int(ckpt.get("_next_idx", 0))
        for k in _FEATURE_KEYS:
            if k in ckpt and len(ckpt[k]) > 0:
                features[k] = list(ckpt[k])
        print(f"  Resuming from checkpoint at sample {start_idx}")
        del ckpt

    tf = extractor._get_transform()

    # ── iterate in batches ────────────────────────────────────────────
    person_buf: List[torch.Tensor] = []
    cloth_buf:  List[torch.Tensor] = []
    processed = start_idx

    def _flush_batch():
        """Run all 7 metric backends on the accumulated mini-batch."""
        nonlocal person_buf, cloth_buf
        if not person_buf:
            return
        import math as _math
        from pretrained_metrics.metrics.m1_pose import _normalise_pose, _joint_angle, TRIPLET_IDX

        person_t = torch.stack(person_buf)      # (B, 3, H, W)
        cloth_t  = torch.stack(cloth_buf)
        B = person_t.shape[0]

        # M1 – Pose
        kps_raw = extractor._kp_ext(person_t)
        kps_norm, valid = _normalise_pose(kps_raw)
        for i in range(B):
            if valid[i]:
                pn = kps_norm[i]
                features["pose_vecs"].append(pn.flatten().astype(np.float32))
                ang = [_joint_angle(pn[ia], pn[ib], pn[ic]) for ia, ib, ic in TRIPLET_IDX]
                features["angles"].append(np.array(
                    [a if not _math.isnan(a) else 0.0 for a in ang], dtype=np.float32))
            else:
                features["pose_vecs"].append(np.zeros(34, dtype=np.float32))
                features["angles"].append(np.zeros(len(TRIPLET_IDX), dtype=np.float32))

        # M2 – Occlusion
        seg = extractor._seg.segment(person_t)
        G   = seg["garment"].float()
        occ = ((seg["arms"].float() + seg["hair"].float() + seg["other"].float()) > 0).float()
        overlap = G * occ
        for i in range(B):
            g_area = G[i].sum().item()
            features["occlusion"].append(float(min(overlap[i].sum().item() / max(g_area, 1.0), 1.0)))

        # M3 – Background
        from pretrained_metrics.metrics.m3_background import _texture_entropy
        pmask  = extractor._per_seg(person_t)
        obj_c  = extractor._obj_det.count_objects(person_t, pmask)
        for i in range(B):
            ent = _texture_entropy(person_t[i], pmask[i])
            features["bg_entropy"].append(float(ent) if not _math.isnan(ent) else 0.0)
            features["bg_obj_count"].append(int(obj_c[i]))

        # M4 – Illumination
        from pretrained_metrics.metrics.m4_illumination import _rgb_to_lab_l, _sobel_gradient_variance
        mean_L, L_maps = _rgb_to_lab_l(person_t.cpu())
        for i in range(B):
            features["lum_mean"].append(float(mean_L[i]))
            features["lum_grad_var"].append(_sobel_gradient_variance(L_maps[i]))

        # M5 – Body shape
        b = extractor._shape_ex(person_t)
        for bi in b:
            features["betas"].append(bi.astype(np.float32))

        # M6 – Appearance
        f = extractor._face_ex(person_t)
        for fi in f:
            features["face_embs"].append(fi.astype(np.float32))

        # M7 – Garment
        g = extractor._garment_ex(cloth_t)
        for gi in g:
            features["garment_embs"].append(gi.astype(np.float32))

        # free GPU memory
        person_buf.clear()
        cloth_buf.clear()
        del person_t, cloth_t, seg, G, occ, overlap, pmask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── main loop ─────────────────────────────────────────────────────
    for i, (person_path, cloth_path, _tryon_path, _meta) in enumerate(loader):
        if i < start_idx:
            continue

        try:
            person_buf.append(_load_image_tensor(person_path, tf))
            cloth_buf.append(_load_image_tensor(cloth_path, tf))
        except Exception as e:
            print(f"    Error loading sample {i}: {e}")
            continue

        if len(person_buf) >= batch_size:
            _flush_batch()

        processed = i + 1

        # progress + checkpoint
        if processed % 500 == 0:
            print(f"    {processed}/{N} ...")
        if processed % _CHECKPOINT_EVERY == 0:
            _save_checkpoint(features, processed, ckpt_path)
            gc.collect()

    _flush_batch()  # final partial batch

    # ── convert to arrays & save final cache ──────────────────────────
    result: Dict[str, np.ndarray] = {}
    for k in _FEATURE_KEYS:
        if features[k]:
            result[k] = np.array(features[k]) if np.isscalar(features[k][0]) else np.stack(features[k])
        else:
            result[k] = np.array([])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    print(f"  Cached features to {cache_path}  ({processed} samples)")

    # remove checkpoint now that final cache is written
    if ckpt_path.exists():
        ckpt_path.unlink()

    # free the large lists
    del features
    gc.collect()

    return result


def _save_checkpoint(features: dict, next_idx: int, path: Path):
    """Write an intermediate .ckpt.npz so extraction can resume after a kill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"_next_idx": np.array(next_idx)}
    for k, v in features.items():
        if v:
            arrays[k] = np.array(v) if np.isscalar(v[0]) else np.stack(v)
    np.savez_compressed(path, **arrays)
    print(f"    [checkpoint] saved at sample {next_idx} → {path}")


def run_curvton_eda(
    base_path: str,
    out_dir: str = "figures/curvton",
    cache_dir: str = "eda_cache/curvton",
    sample_ratio: float = 1.0,
    difficulties: List[str] = ["easy", "medium", "hard"],
    force_recompute: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = _EXTRACT_BATCH_SIZE,
):
    """
    Run full CURVTON EDA pipeline.
    
    Generates:
    1. Individual plots per difficulty level
    2. Combined overlay plots comparing all difficulties
    """
    out_path = Path(out_dir)
    cache_path = Path(cache_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CURVTON EDA Pipeline")
    print("=" * 70)
    print(f"  Base path:    {base_path}")
    print(f"  Output:       {out_dir}")
    print(f"  Device:       {device}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Sample ratio: {sample_ratio:.0%}")
    print(f"  Difficulties: {difficulties}")
    print("=" * 70)
    
    # Initialize feature extractor with correct device
    extractor = FeatureExtractor(device=device)
    
    # Collect features for each difficulty
    all_features = {}
    
    for diff in difficulties:
        print(f"\n[{diff.upper()}] Loading dataset...")
        
        loader = CURVTONDataloader(
            base_path=base_path,
            difficulty=diff,
            sample_ratio=sample_ratio,
            return_paths=True,
        )
        
        cache_file = cache_path / f"curvton_{diff}_{int(sample_ratio*100)}pct.npz"
        features = extract_features_for_difficulty(
            loader, extractor, cache_file, force_recompute,
            batch_size=batch_size,
        )
        
        # Use display names for plots
        display_name = diff.capitalize()
        all_features[display_name] = features
    
    # Update global color/marker mappings for CURVTON
    for diff in ["Easy", "Medium", "Hard"]:
        DATASET_COLORS[diff] = CURVTON_COLORS[diff]
        DATASET_MARKERS[diff] = CURVTON_MARKERS[diff]
        DATASET_LINESTYLES[diff] = CURVTON_LINESTYLES[diff]
    
    # ══════════════════════════════════════════════════════════════════════════
    # Generate EDA Plots
    # ══════════════════════════════════════════════════════════════════════════
    
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = _EXTRACT_BATCH_SIZE,
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
            device=device,
            batch_size=batch_size,
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
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference (cuda / cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=_EXTRACT_BATCH_SIZE,
        help="Batch size for feature extraction (lower if OOM)"
    )
    
    args = parser.parse_args()
    
    if args.multi_ratio:
        run_multi_ratio_eda(
            base_path=args.base_path,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        run_curvton_eda(
            base_path=args.base_path,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            sample_ratio=args.sample_ratio,
            difficulties=args.difficulties,
            force_recompute=args.force_recompute,
            device=args.device,
            batch_size=args.batch_size,
        )
