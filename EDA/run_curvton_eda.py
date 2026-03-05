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


def extract_features_for_difficulty(
    loader: CURVTONDataloader,
    extractor: FeatureExtractor,
    cache_path: Path,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract EDA features for a CURVTON difficulty split.
    
    Returns dict with keys: pose_vecs, angles, occlusion, bg_entropy, 
                           lum_mean, betas, face_embs, garment_embs, etc.
    """
    if cache_path.exists() and not force_recompute:
        print(f"  Loading cached features from {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))
    
    print(f"  Extracting features for {len(loader)} samples...")
    
    # Initialize feature arrays
    features = {
        "pose_vecs": [],
        "angles": [],
        "occlusion": [],
        "bg_entropy": [],
        "bg_obj_count": [],
        "lum_mean": [],
        "lum_grad_var": [],
        "betas": [],
        "face_embs": [],
        "garment_embs": [],
    }
    
    for i, (person_path, cloth_path, tryon_path, meta) in enumerate(loader):
        if i % 500 == 0:
            print(f"    Processing {i}/{len(loader)}...")
        
        try:
            # Extract features using the feature extractor
            feat = extractor.extract_all(person_path, cloth_path)
            
            for key in features:
                if key in feat and feat[key] is not None:
                    features[key].append(feat[key])
        except Exception as e:
            print(f"    Error at {i}: {e}")
            continue
    
    # Convert to numpy arrays
    for key in features:
        if features[key]:
            features[key] = np.array(features[key])
        else:
            features[key] = np.array([])
    
    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **features)
    print(f"  Cached features to {cache_path}")
    
    return features


def run_curvton_eda(
    base_path: str,
    out_dir: str = "figures/curvton",
    cache_dir: str = "eda_cache/curvton",
    sample_ratio: float = 1.0,
    difficulties: List[str] = ["easy", "medium", "hard"],
    force_recompute: bool = False,
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
    print(f"  Sample ratio: {sample_ratio:.0%}")
    print(f"  Difficulties: {difficulties}")
    print("=" * 70)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
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
            loader, extractor, cache_file, force_recompute
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
        run_multi_ratio_eda(
            base_path=args.base_path,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
        )
    else:
        run_curvton_eda(
            base_path=args.base_path,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            sample_ratio=args.sample_ratio,
            difficulties=args.difficulties,
            force_recompute=args.force_recompute,
        )
