from __future__ import annotations

import os
import re

# Limit BLAS thread counts early to avoid OpenBLAS over-threading crashes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from dress_info import DRESS_INFO


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CATEGORY_ORDER = ["upper_body", "lower_body", "dresses", "uncertain"]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    category: str
    cloth_name: str


class SD15CLIPMultimodalEmbedder:
    """SD1.x-compatible CLIP embedder for both image and text branches."""

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def _extract_image_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.get_image_features(**inputs)
        if torch.is_tensor(out):
            return out
        if hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
            return out.image_embeds
        if hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state.mean(dim=1)

        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output
        if hasattr(self.model, "visual_projection"):
            pooled = self.model.visual_projection(pooled)
        return pooled

    def _extract_text_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.get_text_features(**inputs)
        if torch.is_tensor(out):
            return out
        if hasattr(out, "text_embeds") and torch.is_tensor(out.text_embeds):
            return out.text_embeds
        if hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state.mean(dim=1)

        text_out = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        pooled = text_out.pooler_output
        if hasattr(self.model, "text_projection"):
            pooled = self.model.text_projection(pooled)
        return pooled

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding image CLIP embeddings"):
            batch_paths = image_paths[start : start + batch_size]
            images: List[Image.Image] = []
            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))

            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feats = self._extract_image_features(inputs)
            feats = F.normalize(feats.float(), dim=-1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text CLIP embeddings"):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feats = self._extract_text_features(inputs)
            feats = F.normalize(feats.float(), dim=-1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


def normalize_garment_name(name: str) -> str:
    cleaned = name.lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    return " ".join(cleaned.split())


def build_name_to_category() -> Dict[str, str]:
    name_to_category: Dict[str, str] = {}
    for entry in DRESS_INFO:
        raw_name = str(entry.get("name", "")).strip()
        category = str(entry.get("category", "uncertain")).strip()
        if not raw_name:
            continue
        name_to_category[normalize_garment_name(raw_name)] = category
    return name_to_category


def resolve_category(cloth_name: str, name_to_category: Dict[str, str]) -> str:
    key = normalize_garment_name(cloth_name)
    if key in name_to_category:
        return name_to_category[key]
    return "uncertain"


def extract_cloth_name_from_stem(stem: str) -> str:
    """
    Parse cloth name from naming convention:
      fh_001652_e03_fc_006199_Shirt_Dress -> Shirt Dress
      mh_002849_e03_mc_011708_barong_tagalog -> barong tagalog
    """
    parts = stem.split("_")

    if len(parts) >= 6:
        cloth_tokens = parts[5:]
    elif len(parts) >= 1:
        cloth_tokens = [parts[-1]]
    else:
        cloth_tokens = ["unknown"]

    cloth_name = " ".join([tok for tok in cloth_tokens if tok]).strip()
    if not cloth_name:
        cloth_name = "unknown"
    return cloth_name.replace("-", " ")


def discover_samples(root_dir: Path, name_to_category: Dict[str, str]) -> List[Sample]:
    """
    Expected tree:
      root/female/cloth_image/*.png
      root/male/cloth_image/*.png
    """
    samples: List[Sample] = []

    for gender in ("female", "male"):
        cloth_dir = root_dir / gender / "cloth_image"
        if not cloth_dir.exists():
            print(f"[warn] Missing folder: {cloth_dir}")
            continue

        for path in sorted(cloth_dir.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
                continue

            cloth_name = extract_cloth_name_from_stem(path.stem)
            category = resolve_category(cloth_name, name_to_category)
            samples.append(Sample(image_path=path, category=category, cloth_name=cloth_name))

    return samples


def sample_by_ratio_per_category(samples: Sequence[Sample], sample_ratio: float, seed: int) -> List[Sample]:
    """Random stratified sampling per category (never sequential slicing)."""
    if sample_ratio >= 1.0:
        return list(samples)
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be > 0")

    rng = np.random.default_rng(seed)
    out: List[Sample] = []

    categories = [c for c in CATEGORY_ORDER if any(s.category == c for s in samples)]
    for category in categories:
        cls = [s for s in samples if s.category == category]
        if not cls:
            continue

        n_keep = max(1, int(round(len(cls) * sample_ratio)))
        n_keep = min(n_keep, len(cls))
        idx = rng.choice(len(cls), size=n_keep, replace=False)
        idx_sorted = np.sort(idx)
        out.extend([cls[i] for i in idx_sorted])

    return out


def fuse_embeddings(image_emb: np.ndarray, text_emb: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fuse image and text embeddings:
      fused = normalize([alpha * image_emb, (1-alpha) * text_emb])
    """
    if image_emb.shape[0] != text_emb.shape[0]:
        raise ValueError("image_emb and text_emb must have same length")

    a = float(np.clip(alpha, 0.0, 1.0))
    fused = np.concatenate([a * image_emb, (1.0 - a) * text_emb], axis=1)
    norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8
    return (fused / norms).astype(np.float32)


def run_tsne(embeddings: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    n = embeddings.shape[0]
    p = min(perplexity, max(5.0, (n - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=p,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n = embeddings.shape[0]
    k = max(2, min(n_clusters, max(2, n // 15)))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return kmeans.fit_predict(embeddings)


def cloth_name_to_semantic_class(name: str) -> str:
    """Map garment names to mid-level semantic classes (broad, but informative)."""
    n = name.lower().replace("_", " ").replace("-", " ")

    keyword_groups: List[Tuple[str, Tuple[str, ...]]] = [
        ("South Asian Traditions", (
            "sari", "saree", "jamdani", "kurta", "kurung", "qipao", "kalash", "kashta", "lehenga", "ajrak"
        )),
        ("MENA Traditions", (
            "thobe", "kandura", "dishdasha", "djellaba", "ghutra", "keffiyeh", "shemagh", "abaya"
        )),
        ("East Asian Traditions", (
            "hanfu", "hanbok", "montsuki", "hakama", "ao ba ba", "shenyi"
        )),
        ("African/Indigenous Traditions", (
            "kente", "tuareg", "naga", "cholita", "ewe", "atlas"
        )),
        ("Ceremonial/Heritage Wear", (
            "barong", "ta'ovala", "taov", "poncho", "angrakha", "churidar"
        )),
        ("Tailored & Formal", (
            "suit", "blazer", "guayabera", "formal", "office"
        )),
        ("Dresses & One-Piece", (
            "dress", "gown", "playsuit", "jumpsuit", "slip", "a line", "wrap"
        )),
        ("Bottomwear & Separates", (
            "jeans", "chinos", "shorts", "skirt", "maxi", "pants", "trousers"
        )),
        ("Contemporary Casual", (
            "streetwear", "sweater", "set", "shirt", "top"
        )),
    ]

    for cls, keywords in keyword_groups:
        if any(k in n for k in keywords):
            return cls
    return "Mixed Global Styles"


def cluster_label_text(cloth_names: Sequence[str], cluster_ids: np.ndarray) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c in sorted(set(cluster_ids.tolist())):
        names = [cloth_names[i] for i in range(len(cloth_names)) if int(cluster_ids[i]) == c]
        classes = [cloth_name_to_semantic_class(n) for n in names]
        cnt = Counter(classes)
        top = cnt.most_common(2)

        if top:
            top_name, top_count = top[0]
            if len(top) > 1:
                second_name, second_count = top[1]
                top_share = top_count / max(1, len(classes))
                second_share = second_count / max(1, len(classes))
                # Keep moderately broad labels: mixed label only when genuinely blended.
                if top_share < 0.62 and second_share >= 0.22:
                    labels[c] = f"{top_name} + {second_name}"
                else:
                    labels[c] = top_name
            else:
                labels[c] = top_name
        else:
            labels[c] = "Mixed Global Styles"
    return labels


def compute_centroids(coords: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for c in sorted(set(cluster_ids.tolist())):
        pts = coords[cluster_ids == c]
        centroids[c] = np.mean(pts, axis=0)
    return centroids


def _eccv_axes_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def plot_clustered_tsne(
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    cloth_names: Sequence[str],
    categories: Sequence[str],
    out_dir: Path,
    stem: str,
    max_labels: int,
    min_cluster_fraction_for_label: float,
) -> Tuple[Path, Path]:
    _eccv_axes_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_clusters = sorted(set(cluster_ids.tolist()))
    cmap = plt.get_cmap("tab20")
    cluster_colors = {c: cmap(i % 20) for i, c in enumerate(unique_clusters)}
    category_markers = {
        "upper_body": "^",
        "lower_body": "s",
        "dresses": "o",
        "uncertain": "x",
    }

    fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)

    coords_arr = np.asarray(coords)
    cluster_arr = np.asarray(cluster_ids)
    category_arr = np.asarray(categories)

    for c in unique_clusters:
        for category in CATEGORY_ORDER:
            mask = (cluster_arr == c) & (category_arr == category)
            if not np.any(mask):
                continue
            ax.scatter(
                coords_arr[mask, 0],
                coords_arr[mask, 1],
                s=22,
                color=cluster_colors[c],
                marker=category_markers[category],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.2,
            )

    cluster_sizes = {c: int(np.sum(cluster_arr == c)) for c in unique_clusters}
    centroids = compute_centroids(coords_arr, cluster_arr)
    label_texts = cluster_label_text(cloth_names, cluster_arr)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)


    # Compact legends: cluster color mapping and category marker mapping.
    cluster_handles = []
    for c in unique_clusters:
        label_preview = label_texts.get(c, f"cluster_{c}")
        cluster_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=cluster_colors[c],
                markeredgecolor="white", markeredgewidth=0.3, markersize=7,
                label=f"C{c}: {label_preview}"
            )
        )

    category_handles = []
    for category in CATEGORY_ORDER:
        if category not in set(categories):
            continue
        category_handles.append(
            plt.Line2D(
                [0], [0], marker=category_markers[category], color="#333333",
                linestyle="None", markersize=6, label=category.replace("_", " ").title()
            )
        )

    leg1 = ax.legend(handles=cluster_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Cluster legend")
    ax.add_artist(leg1)
    ax.legend(handles=category_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), title="Garment category")

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster multimodal garment embeddings (image + cloth-name text) and plot labeled t-SNE by body category."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_ultimate/easy"),
        help="Dataset root containing female/cloth_image and male/cloth_image.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs/garment_multimodal_tsne_easy"),
        help="Directory to save plots and embedding cache.",
    )
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model id (SD1.5-compatible family).",
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.25,
        help="Fraction of each category to use (default: 0.25 = 25%), sampled randomly.",
    )
    parser.add_argument(
        "--fuse-alpha",
        type=float,
        default=0.5,
        help="Weight for image branch in fusion. Text branch weight is (1-alpha).",
    )
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--max-labels", type=int, default=8)
    parser.add_argument(
        "--min-cluster-fraction-for-label",
        type=float,
        default=0.05,
        help="Only annotate clusters with at least this fraction of samples (reduces label crowding).",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save multimodal embeddings and metadata to npz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1].")
    if not (0.0 <= args.fuse_alpha <= 1.0):
        raise ValueError("--fuse-alpha must be in [0, 1].")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    name_to_category = build_name_to_category()
    samples = discover_samples(args.data_root, name_to_category)
    if not samples:
        raise RuntimeError(
            "No garment images found. Expected: root/female/cloth_image/* and root/male/cloth_image/*"
        )

    samples = sample_by_ratio_per_category(samples, args.sample_ratio, args.seed)

    image_paths = [s.image_path for s in samples]
    categories = [s.category for s in samples]
    cloth_names = [s.cloth_name for s in samples]

    print(f"Found {len(samples)} sampled garments")
    print(f"Sampling ratio per category: {args.sample_ratio:.0%} (random, seed={args.seed})")
    for category in CATEGORY_ORDER:
        count = sum(1 for c in categories if c == category)
        if count:
            print(f"  {category}: {count}")
    print(f"Using device: {args.device}")
    print(f"CLIP model: {args.clip_model_id}")

    embedder = SD15CLIPMultimodalEmbedder(model_id=args.clip_model_id, device=args.device)
    image_emb = embedder.encode_images(image_paths, batch_size=args.image_batch_size)
    text_emb = embedder.encode_texts(cloth_names, batch_size=args.text_batch_size)

    fused = fuse_embeddings(image_emb, text_emb, alpha=args.fuse_alpha)
    cluster_ids = cluster_embeddings(fused, n_clusters=args.n_clusters, seed=args.seed)
    tsne_coords = run_tsne(fused, seed=args.seed, perplexity=args.tsne_perplexity)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        npz_path = args.out_dir / "garment_multimodal_embeddings_easy.npz"
        np.savez_compressed(
            npz_path,
            image_embeddings=image_emb,
            text_embeddings=text_emb,
            fused_embeddings=fused,
            cluster_ids=cluster_ids,
            cloth_names=np.asarray(cloth_names),
            categories=np.asarray(categories),
            image_paths=np.asarray([str(p) for p in image_paths]),
        )
        print(f"Saved embeddings to: {npz_path}")

    tsne_png, tsne_pdf = plot_clustered_tsne(
        coords=tsne_coords,
        cluster_ids=cluster_ids,
        cloth_names=cloth_names,
        categories=categories,
        out_dir=args.out_dir,
        stem="garment_multimodal_clustered_tsne",
        max_labels=args.max_labels,
        min_cluster_fraction_for_label=args.min_cluster_fraction_for_label,
    )

    print("Saved plots:")
    print(f"  {tsne_png}")
    print(f"  {tsne_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
