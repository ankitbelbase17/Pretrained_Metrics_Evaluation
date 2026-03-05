"""
EDA/plots/p11_clip_embedding_eda.py
=====================================
CLIP Embedding EDA — PCA/t-SNE Visualization for CURVTON Dataset

Plots:
  Figure 11A: CLIP Image Embeddings PCA/t-SNE (cloth images)
  Figure 11B: CLIP Text Embeddings PCA/t-SNE (cloth names extracted from filenames)

Cloth naming convention:
  fc_010632_tracksuit → cloth name = "tracksuit" (after last underscore)
  mc_000123_blazer    → cloth name = "blazer"

Usage:
    python EDA/plots/p11_clip_embedding_eda.py \
        --base_path /path/to/curvton \
        --out_dir figures/clip_embeddings \
        --sample_ratio 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from EDA.plot_style import (
    apply_paper_style, save_fig, add_stat_box,
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, CURVTON_COLORS,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP Encoder for Images and Text
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPEmbedder:
    """
    CLIP encoder for both image and text embeddings.
    Uses openai/clip or open_clip or HuggingFace CLIP as fallback.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._backend = "stub"
        self.embed_dim = 512
        self._load()
    
    def _load(self):
        # Try openai/clip
        try:
            import clip as _oa_clip
            if hasattr(_oa_clip, "load"):
                self._clip, self._preprocess = _oa_clip.load(
                    "ViT-B/32", device=self.device
                )
                self._clip.eval()
                self._backend = "openai_clip"
                self.embed_dim = 512
                print("[CLIPEmbedder] Using CLIP ViT-B/32 (openai)")
                return
        except Exception as e:
            print(f"[CLIPEmbedder] openai/clip unavailable ({e})")
        
        # Try open_clip
        try:
            import open_clip
            self._oc_model, _, self._oc_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._oc_model = self._oc_model.to(self.device).eval()
            self._oc_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._backend = "open_clip"
            self.embed_dim = 512
            print("[CLIPEmbedder] Using open_clip ViT-B/32")
            return
        except Exception as e:
            print(f"[CLIPEmbedder] open_clip unavailable ({e})")
        
        # Try HuggingFace CLIP
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._hf_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device).eval()
            self._hf_proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._backend = "hf_clip"
            self.embed_dim = 512
            print("[CLIPEmbedder] Using HuggingFace CLIP")
            return
        except Exception as e:
            print(f"[CLIPEmbedder] HuggingFace CLIP unavailable ({e})")
        
        self._backend = "stub"
        print("[CLIPEmbedder] WARNING: No CLIP backend available, using stub")
    
    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode PIL images to CLIP embeddings. Returns (N, D) array."""
        if self._backend == "stub":
            return np.random.default_rng(42).normal(0, 1, (len(images), self.embed_dim)).astype(np.float32)
        
        if self._backend == "openai_clip":
            import clip as _oa_clip
            inp = torch.stack([self._preprocess(img) for img in images]).to(self.device)
            emb = self._clip.encode_image(inp)
            emb = F.normalize(emb.float(), dim=-1)
            return emb.cpu().numpy()
        
        if self._backend == "open_clip":
            inp = torch.stack([self._oc_preprocess(img) for img in images]).to(self.device)
            emb = self._oc_model.encode_image(inp)
            emb = F.normalize(emb.float(), dim=-1)
            return emb.cpu().numpy()
        
        if self._backend == "hf_clip":
            inputs = self._hf_proc(images=images, return_tensors="pt", padding=True).to(self.device)
            emb = self._hf_model.get_image_features(**inputs)
            emb = F.normalize(emb.float(), dim=-1)
            return emb.cpu().numpy()
        
        return np.zeros((len(images), self.embed_dim), dtype=np.float32)
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text strings to CLIP embeddings. Returns (N, D) array."""
        if self._backend == "stub":
            return np.random.default_rng(42).normal(0, 1, (len(texts), self.embed_dim)).astype(np.float32)
        
        if self._backend == "openai_clip":
            import clip as _oa_clip
            tokens = _oa_clip.tokenize(texts, truncate=True).to(self.device)
            emb = self._clip.encode_text(tokens)
            emb = F.normalize(emb.float(), dim=-1)
            return emb.cpu().numpy()
        
        if self._backend == "open_clip":
            tokens = self._oc_tokenizer(texts).to(self.device)
            emb = self._oc_model.encode_text(tokens)
            emb = F.normalize(emb.float(), dim=-1)
            return emb.cpu().numpy()
        
        if self._backend == "hf_clip":
            inputs = self._hf_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            emb = self._hf_model.get_text_features(**inputs)
            emb = F.normalize(emb.float(), dim=-1)
            return emb.cpu().numpy()
        
        return np.zeros((len(texts), self.embed_dim), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Extract cloth name from filename
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cloth_name(cloth_id: str) -> str:
    """
    Extract cloth category name from cloth ID.
    
    Convention: fc_010632_tracksuit → "tracksuit"
                mc_000123_blazer    → "blazer"
    """
    parts = cloth_id.split("_")
    if len(parts) >= 3:
        # Last part after the last underscore is the cloth name
        return parts[-1].lower()
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Load CURVTON data and extract features
# ═══════════════════════════════════════════════════════════════════════════════

def load_curvton_clip_features(
    base_path: str,
    sample_ratio: float = 0.2,
    device: str = "cuda",
    batch_size: int = 32,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load CURVTON dataset and extract CLIP features.
    
    Returns:
        image_embeddings: (N, D) CLIP image embeddings
        text_embeddings: (N, D) CLIP text embeddings  
        cloth_names: List of cloth category names
        difficulties: List of difficulty labels for coloring
    """
    from dataloaders.curvton_dataloader import CURVTONDataloader
    
    # Check cache
    if cache_path and cache_path.exists() and not force_recompute:
        print(f"  Loading cached CLIP features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["image_embeddings"],
            data["text_embeddings"],
            data["cloth_names"].tolist(),
            data["difficulties"].tolist(),
        )
    
    # Load dataloader
    print(f"  Loading CURVTON at {sample_ratio:.0%} sample ratio...")
    loader = CURVTONDataloader(
        base_path=base_path,
        difficulty="all",
        sample_ratio=sample_ratio,
        return_paths=True,
    )
    
    print(f"  Total samples: {len(loader)}")
    
    # Initialize CLIP embedder
    embedder = CLIPEmbedder(device=device)
    
    # Collect data
    cloth_images = []
    cloth_names = []
    difficulties = []
    
    print("  Loading cloth images...")
    for i, (person_path, cloth_path, tryon_path, meta) in enumerate(tqdm(loader)):
        try:
            cloth_img = Image.open(cloth_path).convert("RGB")
            cloth_images.append(cloth_img)
            
            # Extract cloth name from cloth_id
            cloth_id = meta.get("cloth_id", Path(cloth_path).stem)
            cloth_name = extract_cloth_name(cloth_id)
            cloth_names.append(cloth_name)
            
            difficulties.append(meta.get("difficulty", "unknown").capitalize())
        except Exception as e:
            print(f"    Error loading {cloth_path}: {e}")
            continue
    
    print(f"  Loaded {len(cloth_images)} cloth images")
    
    # Extract image embeddings in batches
    print("  Extracting CLIP image embeddings...")
    image_embeddings = []
    for i in tqdm(range(0, len(cloth_images), batch_size)):
        batch = cloth_images[i:i+batch_size]
        embs = embedder.encode_images(batch)
        image_embeddings.append(embs)
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    
    # Get unique cloth names for text embeddings
    unique_names = list(set(cloth_names))
    print(f"  Found {len(unique_names)} unique cloth categories")
    
    # Create text descriptions for CLIP
    text_descriptions = [f"a photo of a {name}" for name in cloth_names]
    
    # Extract text embeddings in batches
    print("  Extracting CLIP text embeddings...")
    text_embeddings = []
    for i in tqdm(range(0, len(text_descriptions), batch_size)):
        batch = text_descriptions[i:i+batch_size]
        embs = embedder.encode_texts(batch)
        text_embeddings.append(embs)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    
    # Cache results
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            cloth_names=np.array(cloth_names),
            difficulties=np.array(difficulties),
        )
        print(f"  Cached features to {cache_path}")
    
    return image_embeddings, text_embeddings, cloth_names, difficulties


# ═══════════════════════════════════════════════════════════════════════════════
# 11A — CLIP Image Embeddings PCA/t-SNE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_clip_image_embeddings(
    embeddings: np.ndarray,
    cloth_names: List[str],
    difficulties: List[str],
    out_dir: str = "figures/clip_embeddings",
    method: str = "both",  # "pca", "tsne", or "both"
    max_samples: int = 5000,
    color_by: str = "difficulty",  # "difficulty" or "cloth_name"
):
    """
    Plot CLIP image embeddings using PCA and/or t-SNE.
    
    Args:
        embeddings: (N, D) CLIP embeddings
        cloth_names: List of cloth category names
        difficulties: List of difficulty labels
        out_dir: Output directory for figures
        method: "pca", "tsne", or "both"
        max_samples: Max samples to plot (for t-SNE speed)
        color_by: Color points by "difficulty" or "cloth_name"
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Subsample if needed
    N = len(embeddings)
    if N > max_samples:
        idx = np.random.default_rng(42).choice(N, max_samples, replace=False)
        embeddings = embeddings[idx]
        cloth_names = [cloth_names[i] for i in idx]
        difficulties = [difficulties[i] for i in idx]
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    
    # ── PCA ───────────────────────────────────────────────────────────────
    if method in ["pca", "both"]:
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        Z_pca = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        
        if color_by == "difficulty":
            # Color by difficulty
            for diff in ["Easy", "Medium", "Hard"]:
                mask = np.array(difficulties) == diff
                if mask.sum() > 0:
                    c = CURVTON_COLORS.get(diff, "#333333")
                    ax.scatter(
                        Z_pca[mask, 0], Z_pca[mask, 1],
                        s=8, alpha=0.5, color=c, label=diff,
                        linewidths=0, rasterized=True,
                    )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        else:
            # Color by cloth name (top categories)
            name_counts = Counter(cloth_names)
            top_names = [n for n, _ in name_counts.most_common(10)]
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            for i, name in enumerate(top_names):
                mask = np.array(cloth_names) == name
                if mask.sum() > 0:
                    ax.scatter(
                        Z_pca[mask, 0], Z_pca[mask, 1],
                        s=8, alpha=0.5, color=colors[i], label=name,
                        linewidths=0, rasterized=True,
                    )
            
            # Plot "other" category
            other_mask = ~np.isin(cloth_names, top_names)
            if other_mask.sum() > 0:
                ax.scatter(
                    Z_pca[other_mask, 0], Z_pca[other_mask, 1],
                    s=5, alpha=0.3, color="#999999", label="other",
                    linewidths=0, rasterized=True,
                )
            
            ax.legend(loc="upper right", fontsize=6, framealpha=0.9, ncol=2)
        
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_title("CLIP Image Embeddings (PCA)", fontsize=11, fontweight="bold")
        
        var_explained = pca.explained_variance_ratio_
        add_stat_box(ax, [
            f"PC1: {var_explained[0]:.1%} var",
            f"PC2: {var_explained[1]:.1%} var",
            f"n = {len(embeddings):,}",
        ])
        
        despine_axes(ax)
        fig.tight_layout()
        
        suffix = f"_by_{color_by}"
        save_fig(fig, out_path / f"clip_image_pca{suffix}.pdf")
        save_fig(fig, out_path / f"clip_image_pca{suffix}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: clip_image_pca{suffix}.pdf")
    
    # ── t-SNE ─────────────────────────────────────────────────────────────
    if method in ["tsne", "both"]:
        print("  Computing t-SNE (this may take a while)...")
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, len(embeddings) // 4),
            n_iter=1000,
            metric="cosine",
        )
        Z_tsne = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        
        if color_by == "difficulty":
            for diff in ["Easy", "Medium", "Hard"]:
                mask = np.array(difficulties) == diff
                if mask.sum() > 0:
                    c = CURVTON_COLORS.get(diff, "#333333")
                    ax.scatter(
                        Z_tsne[mask, 0], Z_tsne[mask, 1],
                        s=8, alpha=0.5, color=c, label=diff,
                        linewidths=0, rasterized=True,
                    )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        else:
            name_counts = Counter(cloth_names)
            top_names = [n for n, _ in name_counts.most_common(10)]
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            for i, name in enumerate(top_names):
                mask = np.array(cloth_names) == name
                if mask.sum() > 0:
                    ax.scatter(
                        Z_tsne[mask, 0], Z_tsne[mask, 1],
                        s=8, alpha=0.5, color=colors[i], label=name,
                        linewidths=0, rasterized=True,
                    )
            
            other_mask = ~np.isin(cloth_names, top_names)
            if other_mask.sum() > 0:
                ax.scatter(
                    Z_tsne[other_mask, 0], Z_tsne[other_mask, 1],
                    s=5, alpha=0.3, color="#999999", label="other",
                    linewidths=0, rasterized=True,
                )
            
            ax.legend(loc="upper right", fontsize=6, framealpha=0.9, ncol=2)
        
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.set_title("CLIP Image Embeddings (t-SNE)", fontsize=11, fontweight="bold")
        
        add_stat_box(ax, [f"n = {len(embeddings):,}"])
        
        despine_axes(ax)
        fig.tight_layout()
        
        suffix = f"_by_{color_by}"
        save_fig(fig, out_path / f"clip_image_tsne{suffix}.pdf")
        save_fig(fig, out_path / f"clip_image_tsne{suffix}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: clip_image_tsne{suffix}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# 11B — CLIP Text Embeddings PCA/t-SNE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_clip_text_embeddings(
    text_embeddings: np.ndarray,
    cloth_names: List[str],
    out_dir: str = "figures/clip_embeddings",
    method: str = "both",
):
    """
    Plot CLIP text embeddings using PCA and/or t-SNE.
    
    Groups embeddings by unique cloth categories for visualization.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Get unique cloth names and their average embeddings
    unique_names = list(set(cloth_names))
    name_to_idx = {name: i for i, name in enumerate(unique_names)}
    
    # Compute mean embedding per category
    print(f"  Computing mean embeddings for {len(unique_names)} cloth categories...")
    category_embeddings = []
    category_counts = []
    
    for name in unique_names:
        mask = np.array(cloth_names) == name
        mean_emb = text_embeddings[mask].mean(axis=0)
        category_embeddings.append(mean_emb)
        category_counts.append(mask.sum())
    
    category_embeddings = np.array(category_embeddings)
    category_embeddings = category_embeddings / (np.linalg.norm(category_embeddings, axis=1, keepdims=True) + 1e-12)
    
    # Sort by count for coloring
    sorted_indices = np.argsort(category_counts)[::-1]
    top_k = min(15, len(unique_names))
    
    # ── PCA ───────────────────────────────────────────────────────────────
    if method in ["pca", "both"] and len(unique_names) > 2:
        print("  Computing PCA for text embeddings...")
        pca = PCA(n_components=2, random_state=42)
        Z_pca = pca.fit_transform(category_embeddings)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Color by frequency (top categories)
        colors = plt.cm.viridis(np.linspace(0, 0.9, top_k))
        
        for rank, idx in enumerate(sorted_indices[:top_k]):
            name = unique_names[idx]
            count = category_counts[idx]
            ax.scatter(
                Z_pca[idx, 0], Z_pca[idx, 1],
                s=50 + count * 0.5, alpha=0.7, color=colors[rank],
                linewidths=1, edgecolors="white",
            )
            ax.annotate(
                name, (Z_pca[idx, 0], Z_pca[idx, 1]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
            )
        
        # Plot remaining as small gray dots
        for idx in sorted_indices[top_k:]:
            ax.scatter(
                Z_pca[idx, 0], Z_pca[idx, 1],
                s=20, alpha=0.3, color="#999999",
                linewidths=0,
            )
        
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_title("CLIP Text Embeddings (PCA)", fontsize=11, fontweight="bold")
        
        var_explained = pca.explained_variance_ratio_
        add_stat_box(ax, [
            f"PC1: {var_explained[0]:.1%} var",
            f"PC2: {var_explained[1]:.1%} var",
            f"Categories: {len(unique_names)}",
        ])
        
        despine_axes(ax)
        fig.tight_layout()
        
        save_fig(fig, out_path / "clip_text_pca.pdf")
        save_fig(fig, out_path / "clip_text_pca.png", dpi=150)
        plt.close(fig)
        print("  Saved: clip_text_pca.pdf")
    
    # ── t-SNE ─────────────────────────────────────────────────────────────
    if method in ["tsne", "both"] and len(unique_names) > 5:
        print("  Computing t-SNE for text embeddings...")
        perplexity = min(30, max(5, len(unique_names) // 3))
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=1000,
            metric="cosine",
        )
        Z_tsne = tsne.fit_transform(category_embeddings)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, top_k))
        
        for rank, idx in enumerate(sorted_indices[:top_k]):
            name = unique_names[idx]
            count = category_counts[idx]
            ax.scatter(
                Z_tsne[idx, 0], Z_tsne[idx, 1],
                s=50 + count * 0.5, alpha=0.7, color=colors[rank],
                linewidths=1, edgecolors="white",
            )
            ax.annotate(
                name, (Z_tsne[idx, 0], Z_tsne[idx, 1]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
            )
        
        for idx in sorted_indices[top_k:]:
            ax.scatter(
                Z_tsne[idx, 0], Z_tsne[idx, 1],
                s=20, alpha=0.3, color="#999999",
                linewidths=0,
            )
        
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.set_title("CLIP Text Embeddings (t-SNE)", fontsize=11, fontweight="bold")
        
        add_stat_box(ax, [f"Categories: {len(unique_names)}"])
        
        despine_axes(ax)
        fig.tight_layout()
        
        save_fig(fig, out_path / "clip_text_tsne.pdf")
        save_fig(fig, out_path / "clip_text_tsne.png", dpi=150)
        plt.close(fig)
        print("  Saved: clip_text_tsne.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Image-Text Embedding Space
# ═══════════════════════════════════════════════════════════════════════════════

def plot_combined_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    cloth_names: List[str],
    out_dir: str = "figures/clip_embeddings",
    max_samples: int = 2000,
):
    """
    Plot image and text embeddings in the same space to show alignment.
    """
    from sklearn.decomposition import PCA
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Subsample images
    N_img = len(image_embeddings)
    if N_img > max_samples:
        idx = np.random.default_rng(42).choice(N_img, max_samples, replace=False)
        image_embeddings = image_embeddings[idx]
        cloth_names_sub = [cloth_names[i] for i in idx]
    else:
        cloth_names_sub = cloth_names
    
    # Get unique text embeddings
    unique_names = list(set(cloth_names_sub))
    text_emb_unique = []
    
    for name in unique_names:
        mask = np.array(cloth_names_sub) == name
        if mask.sum() > 0:
            # Use first matching text embedding
            first_idx = np.where(mask)[0][0]
            text_emb_unique.append(text_embeddings[idx[first_idx]] if N_img > max_samples else text_embeddings[first_idx])
    
    text_emb_unique = np.array(text_emb_unique)
    
    # Combine embeddings
    combined = np.vstack([image_embeddings, text_emb_unique])
    combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-12)
    
    # PCA on combined space
    print("  Computing PCA on combined image-text space...")
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(combined)
    
    Z_img = Z[:len(image_embeddings)]
    Z_txt = Z[len(image_embeddings):]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot image embeddings
    ax.scatter(
        Z_img[:, 0], Z_img[:, 1],
        s=6, alpha=0.3, color="#4A90D9", label="Images",
        linewidths=0, rasterized=True,
    )
    
    # Plot text embeddings with labels
    ax.scatter(
        Z_txt[:, 0], Z_txt[:, 1],
        s=80, alpha=0.8, color="#E74C3C", label="Text",
        marker="*", linewidths=0.5, edgecolors="white",
    )
    
    # Annotate text points
    for i, name in enumerate(unique_names[:15]):  # Top 15 labels
        ax.annotate(
            name, (Z_txt[i, 0], Z_txt[i, 1]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
            color="#E74C3C",
        )
    
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("CLIP Joint Embedding Space (Image + Text)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    
    despine_axes(ax)
    fig.tight_layout()
    
    save_fig(fig, out_path / "clip_combined_space.pdf")
    save_fig(fig, out_path / "clip_combined_space.png", dpi=150)
    plt.close(fig)
    print("  Saved: clip_combined_space.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Cloth Category Distribution Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cloth_category_distribution(
    cloth_names: List[str],
    difficulties: List[str],
    out_dir: str = "figures/clip_embeddings",
    top_k: int = 20,
):
    """
    Plot distribution of cloth categories.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Count categories
    name_counts = Counter(cloth_names)
    top_names = [n for n, _ in name_counts.most_common(top_k)]
    top_counts = [name_counts[n] for n in top_names]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.barh(range(len(top_names)), top_counts, color="#4A90D9", alpha=0.8)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title(f"Top {top_k} Cloth Categories (CURVTON 20%)", fontsize=11, fontweight="bold")
    
    # Add count labels
    for bar, count in zip(bars, top_counts):
        ax.text(
            bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f"{count:,}", va="center", fontsize=8,
        )
    
    despine_axes(ax)
    fig.tight_layout()
    
    save_fig(fig, out_path / "cloth_category_distribution.pdf")
    save_fig(fig, out_path / "cloth_category_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved: cloth_category_distribution.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def run_clip_embedding_eda(
    base_path: str,
    out_dir: str = "figures/clip_embeddings",
    cache_dir: str = "eda_cache/clip",
    sample_ratio: float = 0.2,
    device: str = "cuda",
    force_recompute: bool = False,
):
    """
    Run full CLIP embedding EDA pipeline for CURVTON dataset.
    """
    print("=" * 70)
    print("CLIP Embedding EDA Pipeline — CURVTON")
    print("=" * 70)
    print(f"  Base path:    {base_path}")
    print(f"  Output:       {out_dir}")
    print(f"  Sample ratio: {sample_ratio:.0%}")
    print(f"  Device:       {device}")
    print("=" * 70)
    
    cache_path = Path(cache_dir) / f"curvton_clip_{int(sample_ratio*100)}pct.npz"
    
    # Load/extract features
    image_embs, text_embs, cloth_names, difficulties = load_curvton_clip_features(
        base_path=base_path,
        sample_ratio=sample_ratio,
        device=device,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )
    
    print(f"\n  Image embeddings shape: {image_embs.shape}")
    print(f"  Text embeddings shape:  {text_embs.shape}")
    print(f"  Unique cloth categories: {len(set(cloth_names))}")
    
    # Generate plots
    print("\n[1/5] Cloth category distribution...")
    plot_cloth_category_distribution(cloth_names, difficulties, out_dir)
    
    print("\n[2/5] CLIP image embeddings by difficulty...")
    plot_clip_image_embeddings(image_embs, cloth_names, difficulties, out_dir, 
                               method="both", color_by="difficulty")
    
    print("\n[3/5] CLIP image embeddings by cloth category...")
    plot_clip_image_embeddings(image_embs, cloth_names, difficulties, out_dir,
                               method="both", color_by="cloth_name")
    
    print("\n[4/5] CLIP text embeddings...")
    plot_clip_text_embeddings(text_embs, cloth_names, out_dir, method="both")
    
    print("\n[5/5] Combined image-text embedding space...")
    plot_combined_embeddings(image_embs, text_embs, cloth_names, out_dir)
    
    print("\n" + "=" * 70)
    print("CLIP Embedding EDA Complete!")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP Embedding EDA for CURVTON Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_path", type=str, required=True,
        help="Path to CURVTON dataset root",
    )
    parser.add_argument(
        "--out_dir", type=str, default="figures/clip_embeddings",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="eda_cache/clip",
        help="Cache directory for extracted features",
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.2,
        help="Dataset sample ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device for CLIP inference",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recompute even if cache exists",
    )
    
    args = parser.parse_args()
    
    run_clip_embedding_eda(
        base_path=args.base_path,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        sample_ratio=args.sample_ratio,
        device=args.device,
        force_recompute=args.force,
    )
