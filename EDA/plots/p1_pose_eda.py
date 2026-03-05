"""
EDA/plots/p1_pose_eda.py
=========================
Pose EDA — ECCV Publication Figures

  Figure 1A:  Pose UMAP / t-SNE scatter
              Input : pose_vecs (N×34)
              Output: 2-D scatter coloured by dataset

  Figure 1B:  Joint angle distribution (KDE)
              Input : angles (N×8)  — 8 limb triplet angles per image
              Output: Overlaid KDE distributions per limb

Usage (standalone):
    python EDA/plots/p1_pose_eda.py \
        --features eda_cache/viton_features.npz \
        --label VITON --out_dir figures/pose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))  # EDA/
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
)

apply_paper_style()

LIMB_NAMES = [
    "L-Elbow", "R-Elbow",
    "L-Knee",  "R-Knee",
    "L-Shoulder", "R-Shoulder",
    "L-Torso", "R-Torso",
]

# Short names for compact display
LIMB_NAMES_SHORT = ["L-Elb", "R-Elb", "L-Kn", "R-Kn", "L-Sh", "R-Sh", "L-Tor", "R-Tor"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1A — UMAP scatter of pose vectors (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pose_umap(
    datasets: Dict[str, np.ndarray],   # {dataset_name: pose_vecs (N,34)}
    out_dir: str = "figures/pose",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    use_tsne: bool = False,
):
    """
    Jointly embeds normalised pose vectors from multiple datasets.
    Colours each point by its source dataset.
    ECCV-optimized: clean scatter with distinct markers, proper legend.
    """
    labels, vecs = [], []
    for name, V in datasets.items():
        labels.extend([name] * len(V))
        vecs.append(V)

    V_all = np.concatenate(vecs, axis=0)  # (N_total, 34)
    # Normalise
    mu = V_all.mean(0); sig = V_all.std(0) + 1e-8
    V_norm = (V_all - mu) / sig

    print(f"  [PoseEDA] Embedding {len(V_norm)} poses …")

    if use_tsne:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(V_norm)//4)).fit_transform(V_norm)
        method_name = "t-SNE"
    else:
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                random_state=42, n_jobs=1)
            Z = reducer.fit_transform(V_norm)
            method_name = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            Z = PCA(n_components=2, random_state=42).fit_transform(V_norm)
            method_name = "PCA (umap not installed)"

    # ── ECCV-style figure ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    
    unique_ds = list(datasets.keys())
    legend_handles = []
    
    for i, name in enumerate(unique_ds):
        mask = np.array(labels) == name
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=8,
            alpha=0.6,
            linewidths=0,
            color=color,
            marker=marker,
            rasterized=True,  # Smaller PDF size
        )
        
        # Legend handle
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w", 
                   markerfacecolor=color, markeredgecolor=color,
                   markersize=8, linewidth=0, label=name)
        )

    ax.set_title(f"Pose Distribution ({method_name})", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(f"{method_name}-1", fontsize=9)
    ax.set_ylabel(f"{method_name}-2", fontsize=9)
    
    # Clean axes for embedding plot
    ax.set_xticks([])
    ax.set_yticks([])
    despine_axes(ax, keep_left=False, keep_bottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")
    
    # Legend
    ncol = 2 if len(unique_ds) > 4 else 1
    legend = ax.legend(
        handles=legend_handles,
        title="Dataset",
        loc="upper right" if len(unique_ds) <= 4 else "center left",
        bbox_to_anchor=(1.0, 1.0) if len(unique_ds) <= 4 else (1.02, 0.5),
        framealpha=0.95,
        edgecolor="#cccccc",
        ncol=ncol,
        fontsize=8,
        markerscale=0.9,
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    save_fig(fig, Path(out_dir), "pose_umap_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset scatter plots
    # ══════════════════════════════════════════════════════════════════════════
    for i, name in enumerate(unique_ds):
        mask = np.array(labels) == name
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        fig_ind, ax_ind = plt.subplots(figsize=(4.0, 3.5))
        
        ax_ind.scatter(
            Z[mask, 0], Z[mask, 1],
            s=10, alpha=0.5, linewidths=0,
            color=color, marker=marker, rasterized=True,
        )
        
        # Add centroid marker
        centroid = Z[mask].mean(axis=0)
        ax_ind.scatter(
            centroid[0], centroid[1],
            s=150, color=color, marker="X",
            edgecolors="white", linewidths=1.5, zorder=10,
            label=f"Centroid"
        )
        
        ax_ind.set_title(f"{name} — Pose Distribution ({method_name})",
                         fontsize=10, fontweight="bold", pad=8)
        ax_ind.set_xlabel(f"{method_name}-1", fontsize=9)
        ax_ind.set_ylabel(f"{method_name}-2", fontsize=9)
        ax_ind.set_xticks([])
        ax_ind.set_yticks([])
        
        # Stats text box
        n_samples = mask.sum()
        ax_ind.text(
            0.02, 0.98, f"n = {n_samples:,}",
            transform=ax_ind.transAxes, fontsize=8,
            verticalalignment="top", fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                      edgecolor="#cccccc", alpha=0.9)
        )
        
        despine_axes(ax_ind, keep_left=False, keep_bottom=False)
        ax_ind.set_frame_on(True)
        for spine in ax_ind.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#cccccc")
        
        plt.tight_layout()
        save_fig(fig_ind, Path(out_dir), f"pose_umap_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1B — Joint angle distributions (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_joint_angle_distributions(
    datasets: Dict[str, np.ndarray],   # {dataset_name: angles (N,8)}
    out_dir: str = "figures/pose",
):
    """
    ECCV-style joint angle visualization:
    - Compact 2x4 grid of KDE distribution plots
    - Maximum overlap visibility with distinct colors/line styles
    - Clean typography and professional styling
    """
    import pandas as pd
    
    # Import overlap-friendly constants
    try:
        from plot_style import FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES
    except ImportError:
        FILL_ALPHA, LINE_ALPHA = 0.25, 0.95
        DATASET_LINESTYLES = {}
    
    n_joints = len(LIMB_NAMES)
    unique_ds = list(datasets.keys())

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A: Multi-dataset KDE per joint (2×4 grid)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.5))
    axes = axes.flatten()

    for j, limb in enumerate(LIMB_NAMES):
        ax = axes[j]
        
        for i, name in enumerate(unique_ds):
            angles_deg = datasets[name][:, j] * (180 / np.pi)
            angles_deg = angles_deg[np.isfinite(angles_deg)]
            
            if len(angles_deg) < 5:
                continue
            
            color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
            linestyle = DATASET_LINESTYLES.get(name, "-")
            
            # KDE with low fill alpha, strong border line
            sns.kdeplot(
                angles_deg, ax=ax,
                fill=True, 
                alpha=FILL_ALPHA,      # Low alpha for overlap visibility
                linewidth=1.5,         # Strong border
                color=color,
                linestyle=linestyle,
                label=name if j == 0 else None,
            )
            
            # Mean indicator line
            ax.axvline(
                angles_deg.mean(), color=color,
                linestyle="--", linewidth=1.2, alpha=0.9, zorder=10
            )
        
        ax.set_title(LIMB_NAMES_SHORT[j], fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Angle (°)" if j >= 4 else "", fontsize=8)
        ax.set_ylabel("Density" if j % 4 == 0 else "", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        
        # Subtle grid
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        despine_axes(ax)
        
        # Add panel label
        if j == 0:
            add_subplot_label(ax, "(a)", x=-0.2, y=1.15, fontsize=10)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(unique_ds), 5),
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=8,
        )

    fig.suptitle("Joint Angle Distribution per Limb", fontsize=11,
                 fontweight="bold", y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "joint_angle_distribution")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel B: Combined KDE overlay (cleaner summary view)
    # ══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
    
    legend_handles = []
    for i, (name, angles) in enumerate(datasets.items()):
        all_angles = (angles * (180 / np.pi)).flatten()
        all_angles = all_angles[np.isfinite(all_angles)]
        if len(all_angles) < 5:
            continue
            
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        linestyle = DATASET_LINESTYLES.get(name, "-")
        marker = DATASET_MARKERS.get(name, "o")
        
        # KDE with strong line, subtle fill
        sns.kdeplot(
            all_angles, ax=ax2, 
            fill=True, 
            alpha=FILL_ALPHA,     # Low alpha for overlap visibility
            linewidth=1.8,        # Strong border line
            color=color,
            linestyle=linestyle,
        )
        
        # Mean line
        ax2.axvline(
            all_angles.mean(), color=color, 
            linestyle="--", linewidth=1.3, alpha=0.9, zorder=10
        )
        
        # Legend with marker + line style + mean value
        legend_handles.append(
            Line2D([0], [0], 
                   marker=marker, color=color,
                   markerfacecolor=color, markersize=7,
                   linewidth=2.0, linestyle=linestyle,
                   label=f"{name} (μ={all_angles.mean():.1f}°)")
        )
    
    ax2.set_xlabel("Joint Angle (degrees)", fontsize=9, fontweight="medium")
    ax2.set_ylabel("Density", fontsize=9, fontweight="medium")
    ax2.set_title("Joint Angle Distribution (All Limbs)", fontsize=10, fontweight="bold", pad=8)
    
    # Legend
    legend = ax2.legend(
        handles=legend_handles,
        title="Dataset", 
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
        loc="upper right",
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")
    
    despine_axes(ax2)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax2.set_axisbelow(True)
    
    add_subplot_label(ax2, "(b)", x=-0.08, y=1.08, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig2, Path(out_dir), "joint_angle_kde_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset plots (for detailed analysis)
    # ══════════════════════════════════════════════════════════════════════════
    for ds_name, angles in datasets.items():
        color = DATASET_COLORS.get(ds_name, PALETTE[0])
        
        # Individual joint angle grid for this dataset
        fig_ind, axes_ind = plt.subplots(2, 4, figsize=(6.5, 4.0))
        axes_ind = axes_ind.flatten()
        
        for j, limb in enumerate(LIMB_NAMES):
            ax = axes_ind[j]
            angles_deg = angles[:, j] * (180 / np.pi)
            angles_deg = angles_deg[np.isfinite(angles_deg)]
            
            if len(angles_deg) < 5:
                continue
            
            sns.kdeplot(
                angles_deg, ax=ax, fill=True,
                alpha=0.35, linewidth=1.8, color=color,
            )
            ax.axvline(
                angles_deg.mean(), color=color,
                linestyle="--", linewidth=1.3, alpha=0.9,
                label=f"μ={angles_deg.mean():.1f}°"
            )
            ax.axvline(
                np.median(angles_deg), color=color,
                linestyle=":", linewidth=1.3, alpha=0.9,
                label=f"med={np.median(angles_deg):.1f}°"
            )
            
            ax.set_title(LIMB_NAMES_SHORT[j], fontsize=9, fontweight="bold", pad=4)
            ax.set_xlabel("Angle (°)" if j >= 4 else "", fontsize=8)
            ax.set_ylabel("Density" if j % 4 == 0 else "", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
            despine_axes(ax)
            if j == 0:
                ax.legend(fontsize=7, loc="upper right")
        
        fig_ind.suptitle(f"{ds_name} — Joint Angle Distributions", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_fig(fig_ind, Path(out_dir), f"joint_angles_{ds_name.lower().replace(' ', '_')}")
        
        # Individual combined KDE for this dataset
        fig_single, ax_single = plt.subplots(figsize=(4.0, 3.0))
        all_angles = (angles * (180 / np.pi)).flatten()
        all_angles = all_angles[np.isfinite(all_angles)]
        
        if len(all_angles) >= 5:
            sns.kdeplot(all_angles, ax=ax_single, fill=True,
                        alpha=0.35, linewidth=2.0, color=color)
            ax_single.axvline(all_angles.mean(), color=color, linestyle="--",
                              linewidth=1.5, alpha=0.9, label=f"μ={all_angles.mean():.1f}°")
            ax_single.axvline(np.median(all_angles), color=color, linestyle=":",
                              linewidth=1.5, alpha=0.9, label=f"med={np.median(all_angles):.1f}°")
            
            ax_single.set_xlabel("Joint Angle (degrees)", fontsize=9)
            ax_single.set_ylabel("Density", fontsize=9)
            ax_single.set_title(f"{ds_name} — All Joint Angles", fontsize=10, fontweight="bold")
            ax_single.legend(loc="upper right", fontsize=8)
            ax_single.yaxis.grid(True, linestyle="--", alpha=0.3)
            despine_axes(ax_single)
            
            plt.tight_layout()
            save_fig(fig_single, Path(out_dir), f"joint_angle_kde_{ds_name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  nargs="+", required=True, help=".npz feature files")
    p.add_argument("--labels",    nargs="+", required=True, help="Dataset label per file")
    p.add_argument("--out_dir",   default="figures/pose")
    p.add_argument("--tsne",      action="store_true", help="Use t-SNE instead of UMAP")
    args = p.parse_args()

    assert len(args.features) == len(args.labels), "features and labels must match"
    datasets_pose = {}
    datasets_angles = {}
    for fpath, label in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        datasets_pose[label]   = d["pose_vecs"]
        datasets_angles[label] = d["angles"]

    plot_pose_umap(datasets_pose, args.out_dir, use_tsne=args.tsne)
    plot_joint_angle_distributions(datasets_angles, args.out_dir)


if __name__ == "__main__":
    _cli()
