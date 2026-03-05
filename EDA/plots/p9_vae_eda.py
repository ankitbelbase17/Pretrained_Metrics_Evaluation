"""
EDA/plots/p9_vae_eda.py
========================
Stable Diffusion VAE Latent Space EDA — ECCV Publication Figures

  Figure 9A:  PCA of VAE encoder embeddings
              Reveals image-level structure and dataset characteristics
  Figure 9B:  Explained variance ratio (scree plot)
              Shows dimensionality of the learned latent manifold

VAE embeddings capture holistic image statistics:
  - Global composition / layout
  - Colour distribution
  - Texture characteristics
  - Overall "look" suitable for reconstruction

Pretrained Model
-----------------
Stable Diffusion VAE (stabilityai/sd-vae-ft-mse or CompVis/stable-diffusion-v1-4)
Encoder produces (B, 4, H/8, W/8) latent codes.

Usage:
    python EDA/plots/p9_vae_eda.py \\
        --features eda_cache/*.npz --labels ... --out_dir figures/vae
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
    add_subplot_label, despine_axes, FIG_DOUBLE,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# 9A — PCA of VAE Latent Embeddings (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_vae_pca(
    datasets: Dict[str, np.ndarray],   # {name: vae_embs (N, D)}
    out_dir: str = "figures/vae",
    max_per_ds: int = 3000,
    n_components: int = 2,
):
    """
    ECCV-style PCA projection of VAE latent vectors.
    
    VAE embeddings capture global image structure — colour, composition,
    texture patterns — making PCA ideal for visualizing dataset distributions.
    """
    from sklearn.decomposition import PCA
    
    labels_all, embs = [], []
    for name, E in datasets.items():
        idx = np.random.default_rng(0).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])
    
    if not embs:
        return None
    
    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    
    # Standardize before PCA
    E_mean = E_all.mean(axis=0)
    E_std = E_all.std(axis=0) + 1e-8
    E_norm = (E_all - E_mean) / E_std
    
    print(f"  [VAE-EDA] PCA on {len(E_all)} VAE embeddings (dim={E_all.shape[1]}) …")
    
    pca = PCA(n_components=min(n_components, E_all.shape[1]))
    Z = pca.fit_transform(E_norm)
    explained_var = pca.explained_variance_ratio_
    
    # ── Figure ──────────────────────────────────────────────────────────────
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
    
    # Axis labels with explained variance
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var.)")
    ax.set_title("VAE Latent Space (Stable Diffusion)")
    
    # Legend
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        handletextpad=0.3,
        borderpad=0.4,
    )
    
    despine_axes(ax)
    add_subplot_label(ax, "(a)")
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, f"{out_dir}/vae_pca")
    plt.close(fig)
    
    return pca  # Return for eigenvalue spectrum plot


def plot_vae_pca_combined(
    datasets: Dict[str, np.ndarray],   # {name: vae_embs (N, D)}
    out_dir: str = "figures/vae",
    max_per_ds: int = 3000,
    n_components_full: int = 50,
):
    """
    ECCV-style combined figure: PCA scatter + explained variance scree plot.
    
    Panel (a): 2D PCA projection showing dataset distributions
    Panel (b): Cumulative explained variance showing latent dimensionality
    """
    from sklearn.decomposition import PCA
    
    labels_all, embs = [], []
    for name, E in datasets.items():
        idx = np.random.default_rng(0).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])
    
    if not embs:
        return
    
    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    
    # Standardize
    E_mean = E_all.mean(axis=0)
    E_std = E_all.std(axis=0) + 1e-8
    E_norm = (E_all - E_mean) / E_std
    
    print(f"  [VAE-EDA] PCA on {len(E_all)} VAE embeddings (dim={E_all.shape[1]}) …")
    
    n_comp = min(n_components_full, E_all.shape[1], len(E_all) - 1)
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(E_norm)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # ── Combined Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    
    # Panel (a): PCA scatter
    ax = axes[0]
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
                   markersize=6, linewidth=0, label=name)
        )
    
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    ax.set_title("VAE Latent Space")
    
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        handletextpad=0.3,
        borderpad=0.3,
    )
    
    despine_axes(ax)
    add_subplot_label(ax, "(a)")
    
    # Panel (b): Scree plot (explained variance)
    ax2 = axes[1]
    
    n_show = min(20, len(explained_var))
    x = np.arange(1, n_show + 1)
    
    # Individual variance bars
    ax2.bar(
        x, explained_var[:n_show] * 100,
        color=PALETTE[0], alpha=0.6,
        edgecolor=PALETTE[0], linewidth=0.5,
        label="Individual",
    )
    
    # Cumulative line
    ax2.plot(
        x, cumulative_var[:n_show] * 100,
        color=PALETTE[1], linewidth=1.5,
        marker="o", markersize=4,
        label="Cumulative",
    )
    
    # 90% and 95% variance thresholds
    ax2.axhline(90, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(95, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7)
    
    # Find components for thresholds
    n_90 = np.searchsorted(cumulative_var, 0.90) + 1
    n_95 = np.searchsorted(cumulative_var, 0.95) + 1
    
    ax2.annotate(
        f"90% @ PC{n_90}",
        xy=(n_90, 90), xytext=(n_90 + 2, 85),
        fontsize=7, color="#666666",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.5),
    )
    
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("VAE Latent Dimensionality")
    ax2.set_xlim(0.5, n_show + 0.5)
    ax2.set_ylim(0, 105)
    
    ax2.legend(loc="center right", fontsize=7, frameon=True,
               framealpha=0.9, edgecolor="none")
    
    despine_axes(ax2)
    add_subplot_label(ax2, "(b)")
    
    fig.tight_layout()
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, f"{out_dir}/vae_pca_combined")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 9B — Explained Variance Spectrum (Standalone)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_vae_explained_variance(
    datasets: Dict[str, np.ndarray],   # {name: vae_embs (N, D)}
    out_dir: str = "figures/vae",
    n_components: int = 50,
):
    """
    Per-dataset explained variance curves overlaid.
    
    Steeper curves indicate lower intrinsic dimensionality.
    Shallower curves suggest richer, more diverse latent representations.
    """
    from sklearn.decomposition import PCA
    
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    legend_handles = []
    for i, (name, E) in enumerate(datasets.items()):
        E = np.nan_to_num(E.astype(np.float32))
        
        # Standardize
        E_mean = E.mean(axis=0)
        E_std = E.std(axis=0) + 1e-8
        E_norm = (E - E_mean) / E_std
        
        n_comp = min(n_components, E.shape[1], len(E) - 1)
        pca = PCA(n_components=n_comp)
        pca.fit(E_norm)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        x = np.arange(1, len(cumvar) + 1)
        
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        
        ax.plot(x, cumvar, color=c, linewidth=1.2, alpha=0.85)
        
        # Mark 90% threshold
        n_90 = np.searchsorted(cumvar, 90) + 1
        ax.scatter([n_90], [90], color=c, s=25, marker="o", zorder=5)
        
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=1.5, label=f"{name} (d₉₀={n_90})")
        )
    
    ax.axhline(90, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(95, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)
    
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("VAE Latent Intrinsic Dimensionality")
    ax.set_xlim(1, n_components)
    ax.set_ylim(0, 102)
    
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )
    
    despine_axes(ax)
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, f"{out_dir}/vae_explained_variance")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 9C — t-SNE / UMAP Alternative (for comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_vae_tsne(
    datasets: Dict[str, np.ndarray],
    out_dir: str = "figures/vae",
    max_per_ds: int = 2000,
    perplexity: int = 30,
):
    """
    t-SNE visualization of VAE embeddings for non-linear structure discovery.
    Use alongside PCA for complementary views.
    """
    from sklearn.manifold import TSNE
    
    labels_all, embs = [], []
    for name, E in datasets.items():
        idx = np.random.default_rng(0).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])
    
    if not embs:
        return
    
    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    
    # Standardize
    E_mean = E_all.mean(axis=0)
    E_std = E_all.std(axis=0) + 1e-8
    E_norm = (E_all - E_mean) / E_std
    
    print(f"  [VAE-EDA] t-SNE on {len(E_all)} VAE embeddings …")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(E_all) // 4),
        random_state=0,
        n_iter=1000,
    )
    Z = tsne.fit_transform(E_norm)
    
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
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("VAE Latent Space (t-SNE)")
    
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )
    
    despine_axes(ax)
    add_subplot_label(ax, "(a)")
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, f"{out_dir}/vae_tsne")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="VAE Latent Space EDA")
    ap.add_argument("--features", nargs="+", required=True,
                    help="Paths to *_features.npz files")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Dataset labels (defaults to filename stems)")
    ap.add_argument("--out_dir", default="figures/vae")
    ap.add_argument("--max_samples", type=int, default=3000)
    args = ap.parse_args()

    paths = [Path(p) for p in args.features]
    labels = args.labels or [p.stem.replace("_features", "") for p in paths]

    datasets = {}
    for path, label in zip(paths, labels):
        data = np.load(path, allow_pickle=True)
        if "vae_embs" in data:
            datasets[label] = data["vae_embs"]
            print(f"  Loaded {label}: {data['vae_embs'].shape}")
        else:
            print(f"  [WARN] {label}: no 'vae_embs' in cache — run feature extraction first")

    if not datasets:
        print("  No VAE embeddings found. Ensure feature extraction includes VAE encoder.")
        return

    plot_vae_pca(datasets, args.out_dir, max_per_ds=args.max_samples)
    plot_vae_pca_combined(datasets, args.out_dir, max_per_ds=args.max_samples)
    plot_vae_explained_variance(datasets, args.out_dir)
    plot_vae_tsne(datasets, args.out_dir, max_per_ds=args.max_samples)


if __name__ == "__main__":
    main()
