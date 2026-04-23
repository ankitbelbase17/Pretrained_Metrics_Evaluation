"""
EDA/run_eric_plots.py
======================
Eric's 6 Required EDA Plots — ECCV Publication Quality

Generates plots in two output directories:
  plots_eric/curvton/     — CurvTON pooled (Easy+Medium+Hard combined, single dataset)
  plots_eric/comparison/  — All datasets overlaid for cross-dataset comparison

The 6 required plots:
  1. Garment Diversity    — t-SNE / UMAP of CLIP embeddings
  2. Face Embeddings      — t-SNE / UMAP of appearance vectors
  3. Body Shape Diversity — PCA(β) with 1σ confidence ellipses
  4. Spatial Occlusion    — Heatmaps (clean, no crosses or arrows)
  5. Background           — Entropy vs Object Density scatter
  6. Illumination PCA     — Z_light scatter (no variance barcharts)

Usage:
    # From cached features:
    python EDA/run_eric_plots.py --cache_dir ./eda_cache --out_dir plots_eric

    # Dry run with synthetic data:
    python EDA/run_eric_plots.py --dry_run --out_dir plots_eric
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

# ── plot modules (only the 6 we need) ───────────────────────────────────────
from plot_style import (
    apply_paper_style, save_fig, set_global_prefix,
    DATASET_COLORS, DATASET_MARKERS, DATASET_LINESTYLES,
    CURVTON_COLORS,
)
from plots.p2_occlusion_eda   import plot_occlusion_heatmap
from plots.p3_background_eda  import plot_entropy_vs_objects
from plots.p4_illumination_eda import plot_illumination_pca
from plots.p5_body_shape_eda  import plot_shape_pca
from plots.p6_appearance_eda  import plot_face_umap
from plots.p7_garment_eda     import plot_garment_umap

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# Plot runner for Eric's 6 required figures
# ═══════════════════════════════════════════════════════════════════════════════

def _run_eric_6(
    all_data: Dict[str, dict],
    out_dir: str,
    tag: str = "",
):
    """
    Generate exactly the 6 required plots from the provided datasets.

    Parameters
    ----------
    all_data : dict
        {dataset_name: {feature_key: np.ndarray, ...}}
    out_dir : str
        Root output directory for this batch of plots.
    tag : str
        Prefix tag for saved figure filenames.
    """
    P = Path(out_dir)
    if tag:
        set_global_prefix(tag)
    else:
        set_global_prefix("")

    n_ds = len(all_data)
    print(f"\n{'═' * 60}")
    print(f"  Eric's 6 Required Plots — {n_ds} dataset(s)")
    print(f"  Output → {P.resolve()}")
    print(f"{'═' * 60}")

    # ── 1. Garment Diversity (t-SNE / UMAP of CLIP embeddings) ────────────
    garment = {n: d["garment_embs"] for n, d in all_data.items()
               if "garment_embs" in d and len(d.get("garment_embs", [])) > 0}
    if garment:
        print("\n  [1/6] Garment Diversity (t-SNE of CLIP embeddings) …")
        plot_garment_umap(garment, out_dir=str(P))
    else:
        print("\n  [1/6] ⚠ Skipped — no garment_embs in cache")

    # ── 2. Face Embeddings (Appearance) ───────────────────────────────────
    print("  [2/6] Appearance face plots disabled (removed globally)")

    # ── 3. Body Shape Diversity (PCA β with 1σ ellipses) ──────────────────
    betas = {n: d["betas"] for n, d in all_data.items()
             if "betas" in d and len(d.get("betas", [])) > 0}
    if betas:
        print("  [3/6] Body Shape Diversity (PCA with 1σ ellipses) …")
        plot_shape_pca(betas, out_dir=str(P))
    else:
        print("  [3/6] ⚠ Skipped — no betas in cache")

    # ── 4. Spatial Occlusion Heatmaps (clean, no crosses) ─────────────────
    occ_maps = {n: d["occ_maps"] for n, d in all_data.items()
                if "occ_maps" in d and len(d.get("occ_maps", [])) > 0}
    if occ_maps:
        print("  [4/6] Spatial Occlusion Heatmaps …")
        plot_occlusion_heatmap(occ_maps, out_dir=str(P))
    else:
        print("  [4/6] ⚠ Skipped — no occ_maps in cache")

    # ── 5. Background: Entropy vs Object Density ─────────────────────────
    bg_ent = {n: d["bg_entropy"] for n, d in all_data.items()
              if "bg_entropy" in d and len(d.get("bg_entropy", [])) > 0}
    bg_obj = {n: d["bg_obj_count"].astype(float) for n, d in all_data.items()
              if "bg_obj_count" in d and len(d.get("bg_obj_count", [])) > 0}
    if bg_ent and bg_obj:
        print("  [5/6] Background Complexity (Entropy vs Object Density) …")
        plot_entropy_vs_objects(bg_ent, bg_obj, out_dir=str(P))
    else:
        print("  [5/6] ⚠ Skipped — no bg_entropy / bg_obj_count in cache")

    # ── 6. PCA Illumination Maps (Z_light, no variance barchart) ──────────
    lum_maps = {n: d["lum_maps"] for n, d in all_data.items()
                if "lum_maps" in d and len(d.get("lum_maps", [])) > 0}
    if lum_maps:
        print("  [6/6] PCA Illumination Maps (Z_light) …")
        plot_illumination_pca(lum_maps, out_dir=str(P))
    else:
        print("  [6/6] ⚠ Skipped — no lum_maps in cache")

    print(f"\n  ✓  All 6 Eric plots → {P.resolve()}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_npz(path: Path) -> dict:
    """Load a cached .npz feature file into a plain dict."""
    d = dict(np.load(path, allow_pickle=True))
    return d


def _pool_curvton(cache_dir: Path, ratio: int = 100) -> dict:
    """
    Pool Easy + Medium + Hard into a single 'CurvTON' dataset.
    Concatenates all feature arrays along axis 0.
    """
    parts = {}
    for diff in ["easy", "medium", "hard"]:
        p = cache_dir / f"curvton_{diff}_{ratio}pct.npz"
        if p.exists():
            parts[diff] = _load_npz(p)
            print(f"  Loaded {p.name}")

    if not parts:
        return {}

    # Collect all feature keys
    all_keys = set()
    for d in parts.values():
        all_keys.update(d.keys())

    pooled = {}
    for k in all_keys:
        arrays = [d[k] for d in parts.values() if k in d and d[k].size > 0]
        if arrays:
            pooled[k] = np.concatenate(arrays, axis=0)

    return pooled


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data for dry-run
# ═══════════════════════════════════════════════════════════════════════════════

def _make_synthetic(n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    H, W = 64, 48
    return {
        "garment_embs": rng.normal(0, 1, (n, 512)).astype(np.float32),
        "face_embs":    rng.normal(0, 1, (n, 68)).astype(np.float32),
        "betas":        rng.normal(0, 1, (n, 10)).astype(np.float32),
        "occ_maps":     rng.random((n, H, W)).astype(np.float32),
        "bg_entropy":   rng.uniform(3, 5, n).astype(np.float32),
        "bg_obj_count": rng.integers(0, 15, n).astype(np.int32),
        "lum_maps":     rng.random((n, H, W)).astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate Eric's 6 required EDA plots (curvton + comparison)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./eda_cache",
        help="Root cache directory containing <label>_features.npz files"
    )
    parser.add_argument(
        "--curvton_cache_dir", type=str, default="./eda_cache/curvton",
        help="Cache directory for CurvTON difficulty splits"
    )
    parser.add_argument(
        "--out_dir", type=str, default="plots_eric",
        help="Output root (will contain curvton/ and comparison/ subdirs)"
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Dataset labels for comparison (e.g., viton_hd dresscode street_tryon)"
    )
    parser.add_argument(
        "--curvton_ratio", type=int, default=100,
        help="Sample ratio percentage for CurvTON cache files (default: 100)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Use synthetic data for smoke testing"
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Dry run
    # ══════════════════════════════════════════════════════════════════════════
    if args.dry_run:
        print("\n[DRY RUN] Generating with synthetic data …")

        # curvton/ — single pooled dataset
        curvton_data = {"CurvTON": _make_synthetic(600, seed=42)}
        DATASET_COLORS["CurvTON"] = "#E7298A"
        DATASET_MARKERS["CurvTON"] = "D"
        _run_eric_6(curvton_data, str(out_root / "curvton"), tag="curvton")

        # comparison/ — all datasets overlaid
        comp_data = {
            "VITON-HD":     _make_synthetic(200, seed=0),
            "DressCode":    _make_synthetic(200, seed=1),
            "StreetTryOn":  _make_synthetic(200, seed=2),
            "CurvTON":      _make_synthetic(200, seed=3),
        }
        for i, name in enumerate(comp_data):
            DATASET_COLORS.setdefault(name, ["#0077BB", "#EE7733", "#009988", "#E7298A"][i])
            DATASET_MARKERS.setdefault(name, ["o", "s", "^", "D"][i])
        _run_eric_6(comp_data, str(out_root / "comparison"), tag="comparison")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CurvTON (pooled Easy+Medium+Hard → single "CurvTON")
    # ══════════════════════════════════════════════════════════════════════════
    curvton_cache = Path(args.curvton_cache_dir)
    pooled = _pool_curvton(curvton_cache, ratio=args.curvton_ratio)

    if pooled:
        print(f"\n[CurvTON] Pooled {sum(v.shape[0] for v in pooled.values() if v.ndim > 0)} total feature rows")
        DATASET_COLORS["CurvTON"] = "#E7298A"
        DATASET_MARKERS["CurvTON"] = "D"
        DATASET_LINESTYLES["CurvTON"] = "-"
        _run_eric_6({"CurvTON": pooled}, str(out_root / "curvton"), tag="curvton")
    else:
        print("[WARN] No CurvTON cache found. Skipping curvton/ plots.")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Comparison (all datasets overlaid)
    # ══════════════════════════════════════════════════════════════════════════
    cache_dir = Path(args.cache_dir)
    all_data: Dict[str, dict] = {}

    # Add CurvTON pooled
    if pooled:
        all_data["CurvTON"] = pooled

    # Add other datasets from cache
    if args.labels:
        for lbl in args.labels:
            p = cache_dir / f"{lbl}_features.npz"
            if p.exists():
                all_data[lbl] = _load_npz(p)
                print(f"  Loaded {p.name}")
            else:
                print(f"  [WARN] {p} not found, skipping")
    else:
        # Auto-discover all *_features.npz in cache_dir
        for p in sorted(cache_dir.glob("*_features.npz")):
            lbl = p.stem.replace("_features", "")
            if lbl not in all_data:
                all_data[lbl] = _load_npz(p)
                print(f"  Auto-discovered {p.name}")

    if len(all_data) >= 2:
        print(f"\n[Comparison] {len(all_data)} datasets: {list(all_data.keys())}")
        _run_eric_6(all_data, str(out_root / "comparison"), tag="comparison")
    elif len(all_data) == 1:
        print("\n[Comparison] Only 1 dataset found — comparison requires ≥2. Skipping.")
    else:
        print("\n[Comparison] No datasets found for comparison.")

    print(f"\n{'═' * 60}")
    print(f"  Eric's plots complete → {out_root.resolve()}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
