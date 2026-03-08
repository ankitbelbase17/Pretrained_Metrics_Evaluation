"""
EDA/plots/p7_garment_eda.py
=============================
Garment Texture Diversity EDA — ECCV Publication Figures

  Figure 7A:  UMAP of garment (CLIP) embeddings — style/pattern/colour clusters
  Figure 7B:  Embedding covariance eigenvalue spectrum
              Fast eigenvalue decay → low diversity
              Slow decay           → high diversity (broad manifold)

Usage:
    python EDA/plots/p7_garment_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/garment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# 7A — UMAP of garment embeddings (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_garment_umap(
    datasets: Dict[str, np.ndarray],   # {name: garment_embs (N,D)}
    out_dir: str = "figures/garment",
    n_neighbors: int = 15,
    min_dist: float = 0.05,
    max_per_ds: int = 3000,
):
    """
    ECCV-style UMAP of garment CLIP features.
    Reveals style clusters (patterns, solids, formals, casuals).
    """
    labels_all, embs = [], []
    for name, E in datasets.items():
        idx = np.random.default_rng(0).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])

    if not embs:
        return

    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    E_all = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)

    print(f"  [GarmentEDA] Embedding {len(E_all)} garment vectors …")
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=0, n_jobs=1, metric="cosine")
        Z = reducer.fit_transform(E_all)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=0, metric="cosine",
                 perplexity=min(30, len(E_all) // 4)).fit_transform(E_all)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    
    legend_handles = []
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        ax.scatter(
            Z[mask, 0], Z[mask, 1], 
            s=6, alpha=0.5,
            linewidths=0,
            color=c, marker=marker,
            rasterized=True,
        )
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=c, markeredgecolor=c,
                   markersize=7, linewidth=0, label=name)
        )

    ax.set_title(f"Garment Diversity — {method} of CLIP Embeddings",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(f"{method}-1", fontsize=9)
    ax.set_ylabel(f"{method}-2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    legend = ax.legend(
        handles=legend_handles,
        title="Dataset",
        markerscale=1.2,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
        ncol=2 if len(datasets) > 4 else 1,
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")
    
    despine_axes(ax, keep_left=False, keep_bottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")
    
    add_subplot_label(ax, "(a)", x=-0.05, y=1.05, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "garment_umap")


# ═══════════════════════════════════════════════════════════════════════════════
# 7B — Eigenvalue spectrum of Cov(g) (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenvalue_spectrum(
    datasets: Dict[str, np.ndarray],   # {name: garment_embs (N,D)}
    out_dir: str = "figures/garment",
    top_k: int = 50,
):
    """
    ECCV-style eigenvalue spectrum analysis.
    Slow decay = high diversity.
    Generates both combined and individual plots.
    """
    # ══════════════════════════════════════════════════════════════════════════
    # Combined overlay plot
    # ══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.8))

    legend_handles = []
    
    for i, (name, E) in enumerate(datasets.items()):
        E = E.astype(np.float32)
        E = np.nan_to_num(E)
        
        # Scale: L2 normalize embeddings so standard scale variance doesn't explode the covariance matrix
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        
        mu  = E.mean(0, keepdims=True)
        Ec  = E - mu
        
        # Economy SVD
        U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
        evs  = s ** 2 / max(len(E) - 1, 1)
        evs  = evs[:top_k]
        evs_norm = evs / evs.sum()
        cumvar   = np.cumsum(evs_norm) * 100

        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        linestyle = DATASET_LINESTYLES.get(name, "-")
        ks = np.arange(1, len(evs) + 1)

        # Strong lines for overlap visibility
        ax1.plot(
            ks, evs_norm, linestyle, 
            color=c, linewidth=1.8, alpha=LINE_ALPHA,
            marker=marker, markersize=3.0, markevery=5,
        )
        ax2.plot(
            ks, cumvar, linestyle, 
            color=c, linewidth=1.8, alpha=LINE_ALPHA,
            marker=marker, markersize=3.0, markevery=5,
        )
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color=c,
                   markerfacecolor=c, markersize=6,
                   linewidth=1.8, linestyle=linestyle, label=name)
        )

    # Style eigenvalue panel
    ax1.set_xlabel("Eigenvalue Index $k$", fontsize=9)
    ax1.set_ylabel("Normalized Eigenvalue", fontsize=9)
    ax1.set_title("Eigenvalue Spectrum\n(slow decay → high diversity)",
                  fontsize=9, fontweight="bold", pad=4)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.25, linewidth=0.4)
    despine_axes(ax1)
    add_subplot_label(ax1, "(b)", x=-0.14, y=1.08, fontsize=10)

    # Style cumulative variance panel
    ax2.set_xlabel("Number of Components $k$", fontsize=9)
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=9)
    ax2.set_title("Cumulative Explained Variance", fontsize=9, fontweight="bold", pad=4)
    ax2.axhline(90, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.axhline(99, color="#aaaaaa", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.text(top_k * 0.98, 91, "90%", fontsize=7, color="#666666", ha="right")
    ax2.text(top_k * 0.98, 100, "99%", fontsize=7, color="#888888", ha="right")
    ax2.set_ylim(0, 101)
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    despine_axes(ax2)
    add_subplot_label(ax2, "(c)", x=-0.14, y=1.08, fontsize=10)

    # Shared legend
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(datasets), 5),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
    )

    fig.suptitle("Garment Embedding Covariance — Eigenvalue Analysis",
                 fontsize=11, fontweight="bold", y=1.12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "garment_eigenvalue_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset plots
    # ══════════════════════════════════════════════════════════════════════════
    for i, (name, E) in enumerate(datasets.items()):
        E = E.astype(np.float32)
        E = np.nan_to_num(E)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        mu = E.mean(0, keepdims=True)
        Ec = E - mu
        
        U, s, Vt = np.linalg.svd(Ec, full_matrices=False)
        evs = s ** 2 / max(len(E) - 1, 1)
        evs = evs[:top_k]
        evs_norm = evs / evs.sum()
        cumvar = np.cumsum(evs_norm) * 100
        
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        ks = np.arange(1, len(evs) + 1)
        
        fig_ind, (ax_ev, ax_cum) = plt.subplots(1, 2, figsize=(6.0, 2.5))
        
        # Eigenvalue spectrum
        ax_ev.plot(ks, evs_norm, "-", color=c, linewidth=2.0, alpha=0.9,
                   marker="o", markersize=3.5, markevery=5)
        ax_ev.set_xlabel("Eigenvalue Index $k$", fontsize=9)
        ax_ev.set_ylabel("Normalized Eigenvalue", fontsize=9)
        ax_ev.set_title("Eigenvalue Spectrum", fontsize=10, fontweight="bold")
        ax_ev.set_yscale("log")
        ax_ev.grid(True, which="both", alpha=0.25, linewidth=0.4)
        despine_axes(ax_ev)
        
        # Cumulative variance
        ax_cum.plot(ks, cumvar, "-", color=c, linewidth=2.0, alpha=0.9,
                    marker="o", markersize=3.5, markevery=5)
        ax_cum.axhline(90, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_cum.axhline(99, color="#aaaaaa", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_cum.set_xlabel("Number of Components $k$", fontsize=9)
        ax_cum.set_ylabel("Cumulative Variance (%)", fontsize=9)
        ax_cum.set_title("Cumulative Variance", fontsize=10, fontweight="bold")
        ax_cum.set_ylim(0, 101)
        ax_cum.grid(True, alpha=0.25, linewidth=0.4)
        despine_axes(ax_cum)
        
        fig_ind.suptitle(f"{name} — Garment Eigenvalue Analysis", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(fig_ind, Path(out_dir), f"garment_eigenvalue_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/garment")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    garment_data = {}
    for fpath, lbl in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        garment_data[lbl] = d.get("garment_embs", np.array([]))

    plot_garment_umap(garment_data, args.out_dir)
    plot_eigenvalue_spectrum(garment_data, args.out_dir)


if __name__ == "__main__":
    _cli()
