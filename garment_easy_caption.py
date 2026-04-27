from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class TextSample:
    image_path: Path
    gender: str
    cloth_name: str


class SD15CLIPTextEmbedder:
    """SD1.x-compatible CLIP text embedder with robust HF output handling."""

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

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
    def encode_texts(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding garment text CLIP embeddings"):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feats = self._extract_text_features(inputs)
            feats = F.normalize(feats.float(), dim=-1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


def extract_cloth_name_from_stem(stem: str) -> str:
    """
    Parse cloth name from filename stem convention:
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


def discover_text_samples(root_dir: Path) -> List[TextSample]:
    """
    Expected tree:
      root/female/cloth_image/*.png
      root/male/cloth_image/*.png
    """
    samples: List[TextSample] = []

    for gender in ("female", "male"):
        cloth_dir = root_dir / gender / "cloth_image"
        if not cloth_dir.exists():
            print(f"[warn] Missing folder: {cloth_dir}")
            continue

        for path in sorted(cloth_dir.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
                continue

            cloth_name = extract_cloth_name_from_stem(path.stem)
            samples.append(TextSample(image_path=path, gender=gender, cloth_name=cloth_name))

    return samples


def sample_by_ratio_per_gender(samples: Sequence[TextSample], sample_ratio: float, seed: int) -> List[TextSample]:
    """Random, stratified sampling per gender (not sequential)."""
    if sample_ratio >= 1.0:
        return list(samples)
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be > 0")

    rng = np.random.default_rng(seed)
    out: List[TextSample] = []

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


def run_umap(embeddings: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        import umap
    except Exception as e:
        raise RuntimeError("UMAP is not installed. Install with: pip install umap-learn") from e

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _eccv_axes_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def _top_categories(labels: Sequence[str], top_k: int) -> List[str]:
    counts: Dict[str, int] = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:top_k]]


def plot_text_projection(
    coords: np.ndarray,
    cloth_names: Sequence[str],
    genders: Sequence[str],
    method_name: str,
    out_dir: Path,
    stem: str,
    top_k_categories: int = 12,
) -> Tuple[Path, Path]:
    _eccv_axes_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    top_cats = set(_top_categories(cloth_names, top_k_categories))
    category_labels = [c if c in top_cats else "Other" for c in cloth_names]

    palette = [
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
        "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#4C78A8", "#F58518", "#9A9A9A",
    ]

    unique_cats = sorted(set(category_labels), key=lambda x: (x == "Other", x.lower()))
    color_map: Dict[str, str] = {}
    for i, cat in enumerate(unique_cats):
        color_map[cat] = palette[i % len(palette)]

    marker_map = {"female": "^", "male": "o"}

    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=150)

    coords_arr = np.asarray(coords)
    cat_arr = np.asarray(category_labels)
    gender_arr = np.asarray(genders)

    for cat in unique_cats:
        for gender in ("female", "male"):
            mask = (cat_arr == cat) & (gender_arr == gender)
            if not np.any(mask):
                continue
            ax.scatter(
                coords_arr[mask, 0],
                coords_arr[mask, 1],
                s=24,
                c=color_map[cat],
                marker=marker_map[gender],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.2,
            )

    ax.set_title(f"Garment Name Text Embeddings ({method_name})", pad=12)
    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")

    total = len(cloth_names)
    n_unique = len(set(cloth_names))
    ax.text(
        0.01,
        0.99,
        f"Samples: {total} | Unique names: {n_unique}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.88, "edgecolor": "#888"},
    )

    # Category legend (top categories + Other)
    cat_handles = []
    for cat in unique_cats:
        cnt = int(np.sum(cat_arr == cat))
        cat_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color_map[cat],
                markeredgecolor="white", markeredgewidth=0.3, markersize=7,
                label=f"{cat} (n={cnt})"
            )
        )

    gender_handles = [
        plt.Line2D([0], [0], marker="^", color="#333333", linestyle="None", markersize=6, label="Female"),
        plt.Line2D([0], [0], marker="o", color="#333333", linestyle="None", markersize=6, label="Male"),
    ]

    leg1 = ax.legend(handles=cat_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Cloth name groups")
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
        description="Create t-SNE and UMAP plots from SD1.5 CLIP text embeddings of garment names parsed from easy split cloth_image filenames."
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
        default=Path("./outputs/garment_text_embeddings_easy"),
        help="Directory to save plots and embedding cache.",
    )
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model id (SD1.5-compatible family).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.25,
        help="Fraction of each gender to use (default: 0.25 = 25%%), sampled randomly.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--top-k-categories", type=int, default=12)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save text embeddings and metadata to npz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1].")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = discover_text_samples(args.data_root)
    if not samples:
        raise RuntimeError(
            "No garment images found. Expected: root/female/cloth_image/* and root/male/cloth_image/*"
        )

    samples = sample_by_ratio_per_gender(samples, args.sample_ratio, args.seed)

    cloth_names = [s.cloth_name for s in samples]
    genders = [s.gender for s in samples]
    image_paths = [s.image_path for s in samples]

    print(f"Found {len(samples)} sampled garment captions total")
    print(f"Sampling ratio per gender: {args.sample_ratio:.0%} (random, seed={args.seed})")
    print(f"  Female samples: {sum(1 for x in genders if x == 'female')}")
    print(f"  Male samples:   {sum(1 for x in genders if x == 'male')}")
    print(f"Using device: {args.device}")
    print(f"CLIP model: {args.clip_model_id}")

    embedder = SD15CLIPTextEmbedder(model_id=args.clip_model_id, device=args.device)
    text_embeddings = embedder.encode_texts(cloth_names, batch_size=args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        npz_path = args.out_dir / "clip_text_embeddings_garment_easy.npz"
        np.savez_compressed(
            npz_path,
            text_embeddings=text_embeddings,
            cloth_names=np.asarray(cloth_names),
            genders=np.asarray(genders),
            image_paths=np.asarray([str(p) for p in image_paths]),
        )
        print(f"Saved text embeddings to: {npz_path}")

    tsne_coords = run_tsne(text_embeddings, seed=args.seed, perplexity=args.tsne_perplexity)
    umap_coords = run_umap(
        text_embeddings,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    tsne_png, tsne_pdf = plot_text_projection(
        coords=tsne_coords,
        cloth_names=cloth_names,
        genders=genders,
        method_name="t-SNE",
        out_dir=args.out_dir,
        stem="garment_caption_clip_tsne",
        top_k_categories=args.top_k_categories,
    )
    umap_png, umap_pdf = plot_text_projection(
        coords=umap_coords,
        cloth_names=cloth_names,
        genders=genders,
        method_name="UMAP",
        out_dir=args.out_dir,
        stem="garment_caption_clip_umap",
        top_k_categories=args.top_k_categories,
    )

    print("Saved plots:")
    print(f"  {tsne_png}")
    print(f"  {tsne_pdf}")
    print(f"  {umap_png}")
    print(f"  {umap_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
