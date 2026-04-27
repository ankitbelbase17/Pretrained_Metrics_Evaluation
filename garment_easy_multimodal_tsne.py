from __future__ import annotations

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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    gender: str
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


def discover_samples(root_dir: Path) -> List[Sample]:
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
            samples.append(Sample(image_path=path, gender=gender, cloth_name=cloth_name))

    return samples


def sample_by_ratio_per_gender(samples: Sequence[Sample], sample_ratio: float, seed: int) -> List[Sample]:
    """Random stratified sampling per gender (never sequential slicing)."""
    if sample_ratio >= 1.0:
        return list(samples)
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be > 0")

    rng = np.random.default_rng(seed)
    out: List[Sample] = []

    for gender in ("female", "male"):
        cls = [s for s in samples if s.gender == gender]
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


def cluster_label_text(cloth_names: Sequence[str], cluster_ids: np.ndarray, top_n_terms: int = 2) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c in sorted(set(cluster_ids.tolist())):
        names = [cloth_names[i] for i in range(len(cloth_names)) if int(cluster_ids[i]) == c]
        cnt = Counter(names)
        top_terms = [name for name, _ in cnt.most_common(top_n_terms)]
        if top_terms:
            labels[c] = " / ".join(top_terms)
        else:
            labels[c] = f"cluster_{c}"
    return labels


def compute_centroids(coords: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for c in sorted(set(cluster_ids.tolist())):
        pts = coords[cluster_ids == c]
        centroids[c] = np.mean(pts, axis=0)
    return centroids


def place_non_overlapping_annotations(
    ax,
    centroids: Dict[int, np.ndarray],
    label_texts: Dict[int, str],
    cluster_sizes: Dict[int, int],
    max_labels: int,
) -> None:
    """
    Add sparse cluster labels with greedy collision avoidance in data space.
    """
    if not centroids:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = max(1e-8, xlim[1] - xlim[0])
    span_y = max(1e-8, ylim[1] - ylim[0])
    min_dx = 0.07 * span_x
    min_dy = 0.07 * span_y

    ranked = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:max_labels]

    placed: List[Tuple[float, float]] = []

    for c, _sz in ranked:
        base = centroids[c]
        bx, by = float(base[0]), float(base[1])

        found_x, found_y = bx, by
        found = False

        for step in range(20):
            if step == 0:
                cand_x, cand_y = bx, by
            else:
                ring = (step + 1) // 2
                sign = -1.0 if step % 2 == 0 else 1.0
                cand_x = bx + sign * ring * 0.02 * span_x
                cand_y = by + sign * ring * 0.02 * span_y

            ok = True
            for px, py in placed:
                if abs(cand_x - px) < min_dx and abs(cand_y - py) < min_dy:
                    ok = False
                    break
            if ok:
                found_x, found_y = cand_x, cand_y
                found = True
                break

        if not found:
            found_x, found_y = bx, by

        placed.append((found_x, found_y))

        ax.annotate(
            label_texts.get(c, f"cluster_{c}"),
            xy=(bx, by),
            xytext=(found_x, found_y),
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.86, "edgecolor": "#666"},
            arrowprops={"arrowstyle": "-", "color": "#777", "lw": 0.7, "alpha": 0.8},
        )


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
    genders: Sequence[str],
    out_dir: Path,
    stem: str,
    max_labels: int,
) -> Tuple[Path, Path]:
    _eccv_axes_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_clusters = sorted(set(cluster_ids.tolist()))
    cmap = plt.get_cmap("tab20")
    cluster_colors = {c: cmap(i % 20) for i, c in enumerate(unique_clusters)}
    gender_markers = {"female": "^", "male": "o"}

    fig, ax = plt.subplots(figsize=(9.5, 7.4), dpi=150)

    coords_arr = np.asarray(coords)
    cluster_arr = np.asarray(cluster_ids)
    gender_arr = np.asarray(genders)

    for c in unique_clusters:
        for gender in ("female", "male"):
            mask = (cluster_arr == c) & (gender_arr == gender)
            if not np.any(mask):
                continue
            ax.scatter(
                coords_arr[mask, 0],
                coords_arr[mask, 1],
                s=22,
                color=cluster_colors[c],
                marker=gender_markers[gender],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.2,
            )

    cluster_sizes = {c: int(np.sum(cluster_arr == c)) for c in unique_clusters}
    centroids = compute_centroids(coords_arr, cluster_arr)
    label_texts = cluster_label_text(cloth_names, cluster_arr, top_n_terms=2)
    place_non_overlapping_annotations(ax, centroids, label_texts, cluster_sizes, max_labels=max_labels)

    ax.set_title("Multimodal Garment Clusters (t-SNE)", pad=12)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    total = len(cloth_names)
    n_unique_names = len(set(cloth_names))
    ax.text(
        0.01,
        0.99,
        f"Samples: {total} | Clusters: {len(unique_clusters)} | Unique names: {n_unique_names}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.88, "edgecolor": "#888"},
    )

    # Compact legends: cluster color mapping and gender marker mapping.
    cluster_handles = []
    for c in unique_clusters:
        label_preview = label_texts.get(c, f"cluster_{c}")
        if len(label_preview) > 34:
            label_preview = label_preview[:31] + "..."
        cluster_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=cluster_colors[c],
                markeredgecolor="white", markeredgewidth=0.3, markersize=7,
                label=f"C{c} (n={cluster_sizes[c]}): {label_preview}"
            )
        )

    gender_handles = [
        plt.Line2D([0], [0], marker="^", color="#333333", linestyle="None", markersize=6, label="Female"),
        plt.Line2D([0], [0], marker="o", color="#333333", linestyle="None", markersize=6, label="Male"),
    ]

    leg1 = ax.legend(handles=cluster_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Cluster legend")
    ax.add_artist(leg1)
    ax.legend(handles=gender_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), title="Gender marker")

    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster multimodal garment embeddings (image + cloth-name text) and plot labeled t-SNE."
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
        default=0.1,
        help="Fraction of each gender to use (default: 0.1 = 10%), sampled randomly.",
    )
    parser.add_argument(
        "--fuse-alpha",
        type=float,
        default=0.5,
        help="Weight for image branch in fusion. Text branch weight is (1-alpha).",
    )
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--max-labels", type=int, default=10)
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

    samples = discover_samples(args.data_root)
    if not samples:
        raise RuntimeError(
            "No garment images found. Expected: root/female/cloth_image/* and root/male/cloth_image/*"
        )

    samples = sample_by_ratio_per_gender(samples, args.sample_ratio, args.seed)

    image_paths = [s.image_path for s in samples]
    genders = [s.gender for s in samples]
    cloth_names = [s.cloth_name for s in samples]

    print(f"Found {len(samples)} sampled garments")
    print(f"Sampling ratio per gender: {args.sample_ratio:.0%} (random, seed={args.seed})")
    print(f"  Female: {sum(1 for g in genders if g == 'female')}")
    print(f"  Male:   {sum(1 for g in genders if g == 'male')}")
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
            genders=np.asarray(genders),
            image_paths=np.asarray([str(p) for p in image_paths]),
        )
        print(f"Saved embeddings to: {npz_path}")

    tsne_png, tsne_pdf = plot_clustered_tsne(
        coords=tsne_coords,
        cluster_ids=cluster_ids,
        cloth_names=cloth_names,
        genders=genders,
        out_dir=args.out_dir,
        stem="garment_multimodal_clustered_tsne",
        max_labels=args.max_labels,
    )

    print("Saved plots:")
    print(f"  {tsne_png}")
    print(f"  {tsne_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
