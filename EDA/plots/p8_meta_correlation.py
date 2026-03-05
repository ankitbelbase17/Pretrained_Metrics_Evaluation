"""
EDA/plots/p8_meta_correlation.py
==================================
Meta-EDA — ECCV Publication Figures

Constructs a per-image feature vector:
    x_i = [pose_norm, O_i, H_bg, L_i, ||β_i||, ||f_i||, ||g_i||]

Computes Pearson correlation matrix → seaborn heatmap.

Also produces a scatter-matrix (pairplot) for visual inspection of
pairwise relationships.

Usage:
    python EDA/plots/p8_meta_correlation.py \
        --features eda_cache/viton_features.npz \
        --label VITON --out_dir figures/meta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
)

apply_paper_style()

# Feature names for display
FEATURE_NAMES = [
    "Pose\n$||v_i||$",
    "Occlusion\n$O_i$",
    "BG Entropy\n$H_{bg}$",
    "Luminance\n$L_i$",
    "Shape\n$||\\beta_i||$",
    "Face\n$||f_i||$",
    "Garment\n$||g_i||$",
]

FEATURE_NAMES_SHORT = [
    "Pose", "Occ", "BG",
    "Lum", "Shape", "Face", "Garment",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Build feature matrix from cache
# ═══════════════════════════════════════════════════════════════════════════════

def _build_feature_matrix(d: dict) -> np.ndarray:
    """
    d: loaded .npz dict.
    Returns (N, 7) float32 matrix — one row per image.
    """
    pose_norm    = np.linalg.norm(d["pose_vecs"],    axis=1)
    occ          = d["occ_ratios"].astype(np.float32)
    bg_ent       = d["bg_entropy"].astype(np.float32)
    lum          = d["lum_mean"].astype(np.float32)
    shape_norm   = np.linalg.norm(d["betas"],         axis=1).astype(np.float32)
    face_norm    = np.linalg.norm(d["face_embs"],     axis=1).astype(np.float32)
    garment_norm = np.linalg.norm(d["garment_embs"], axis=1).astype(np.float32)

    X = np.stack(
        [pose_norm, occ, bg_ent, lum, shape_norm, face_norm, garment_norm],
        axis=1
    )
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Main correlation heatmap (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(
    datasets: Dict[str, np.ndarray],   # {name: feature_matrix (N,7)}
    out_dir: str = "figures/meta",
):
    """
    ECCV-style correlation heatmap with clean annotations.
    Per-dataset panels + pooled summary.
    """
    n = len(datasets)
    pooled_X = np.concatenate(list(datasets.values()), axis=0)

    # ── Per-dataset panels ─────────────────────────────────────────────────
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.0 * cols, 2.8 * rows),
        squeeze=False
    )
    axes = axes.flatten()

    def _corr_heatmap(X, ax, title):
        df = pd.DataFrame(X, columns=FEATURE_NAMES_SHORT)
        corr = df.corr(method="pearson")
        
        # Create mask for upper triangle (optional - keep full for clarity)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        sns.heatmap(
            corr, ax=ax, 
            vmin=-1, vmax=1, center=0,
            cmap="RdBu_r",
            annot=True, fmt=".2f", 
            annot_kws={"size": 6, "weight": "normal"},
            linewidths=0.3, linecolor="white",
            square=True, 
            cbar_kws={"shrink": 0.7, "aspect": 20},
            xticklabels=FEATURE_NAMES_SHORT,
            yticklabels=FEATURE_NAMES_SHORT,
        )
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
        
        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)

    for i, (name, X) in enumerate(datasets.items()):
        X_clean = np.nan_to_num(X)
        _corr_heatmap(X_clean, axes[i], name)
        if i == 0:
            add_subplot_label(axes[i], "(a)", x=-0.15, y=1.10, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Correlation Matrix — Per Dataset",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "correlation_matrix_per_dataset")

    # ── Pooled (all datasets) ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.0))
    _corr_heatmap(np.nan_to_num(pooled_X), ax2, "All Datasets (Pooled)")
    add_subplot_label(ax2, "(b)", x=-0.12, y=1.08, fontsize=10)
    
    fig2.suptitle("Pooled Complexity Correlation Matrix",
                  fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig2, Path(out_dir), "correlation_matrix_pooled")


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter-matrix (pairplot) — ECCV Style
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scatter_matrix(
    datasets: Dict[str, np.ndarray],   # {name: feature_matrix (N,7)}
    out_dir: str = "figures/meta",
    max_per_ds: int = 300,
):
    """
    ECCV-style pairplot with dataset hue.
    Subsampled for readability.
    """
    rows_list = []
    for name, X in datasets.items():
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), min(max_per_ds, len(X)), replace=False)
        Xs  = np.nan_to_num(X[idx])
        df  = pd.DataFrame(Xs, columns=FEATURE_NAMES_SHORT)
        df["Dataset"] = name
        rows_list.append(df)

    df_all = pd.concat(rows_list, ignore_index=True)
    palette = {n: DATASET_COLORS.get(n, PALETTE[i % len(PALETTE)])
               for i, n in enumerate(datasets.keys())}

    g = sns.pairplot(
        df_all, 
        hue="Dataset", 
        plot_kws=dict(alpha=0.35, s=5, linewidth=0, rasterized=True),
        diag_kind="kde", 
        diag_kws=dict(fill=True, alpha=0.2, linewidth=0.8),
        palette=palette, 
        corner=True,
    )
    
    # Style improvements
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(labelsize=6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
    
    g.fig.suptitle("Pairwise Scatter Matrix — Complexity Features",
                   fontsize=11, fontweight="bold", y=1.01)
    
    # Improve legend
    g._legend.set_title("Dataset")
    g._legend.get_title().set_fontsize(8)
    g._legend.get_title().set_fontweight("bold")
    for text in g._legend.get_texts():
        text.set_fontsize(7)
    
    save_fig(g.fig, Path(out_dir), "scatter_matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/meta")
    p.add_argument("--no_pairplot", action="store_true",
                   help="Skip slow pairplot (useful for large datasets)")
    args = p.parse_args()

    Xs = {}
    for f, lbl in zip(args.features, args.labels):
        d     = dict(np.load(f, allow_pickle=True))
        Xs[lbl] = _build_feature_matrix(d)

    plot_correlation_matrix(Xs, args.out_dir)
    if not args.no_pairplot:
        plot_scatter_matrix(Xs, args.out_dir)


if __name__ == "__main__":
    _cli()
