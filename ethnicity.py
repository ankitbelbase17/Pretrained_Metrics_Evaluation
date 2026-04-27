from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: str


class SD15CLIPImageEmbedder:
    """
    CLIP image embedder configured with the CLIP-L/14 family used with SD1.x.

    For SD v1.5 workflows, CLIP encoders are from the OpenAI CLIP family.
    This script uses the CLIP image branch to produce normalized image embeddings.
    """

    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def _extract_image_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return image embeddings as a tensor across transformers versions."""
        out = self.model.get_image_features(**inputs)
        if torch.is_tensor(out):
            return out

        # Some versions may return a model output object.
        if hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
            return out.image_embeds
        if hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            # Mean pool as a robust fallback if only token features are exposed.
            return out.last_hidden_state.mean(dim=1)

        # Final fallback: run vision branch directly and apply visual projection when available.
        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output
        if hasattr(self.model, "visual_projection"):
            pooled = self.model.visual_projection(pooled)
        return pooled

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding CLIP embeddings"):
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


def discover_samples(root_dir: Path) -> List[Sample]:
    samples: List[Sample] = []

    for label in ("female", "male"):
        subdir = root_dir / label
        if not subdir.exists():
            print(f"[warn] Missing folder: {subdir}")
            continue

        for path in sorted(subdir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append(Sample(image_path=path, label=label))

    return samples


def maybe_limit_per_class(samples: Sequence[Sample], limit_per_class: int, seed: int) -> List[Sample]:
    if limit_per_class <= 0:
        return list(samples)

    rng = np.random.default_rng(seed)
    out: List[Sample] = []

    for label in ("female", "male"):
        cls = [s for s in samples if s.label == label]
        if len(cls) <= limit_per_class:
            out.extend(cls)
            continue

        idx = rng.choice(len(cls), size=limit_per_class, replace=False)
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
        raise RuntimeError(
            "UMAP is not installed. Install with: pip install umap-learn"
        ) from e

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
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def plot_projection(
    coords: np.ndarray,
    labels: Sequence[str],
    method_name: str,
    out_dir: Path,
    stem: str,
) -> Tuple[Path, Path]:
    _eccv_axes_style()

    out_dir.mkdir(parents=True, exist_ok=True)

    color_map: Dict[str, str] = {
        "male": "#1B6CA8",
        "female": "#D14D41",
    }
    marker_map: Dict[str, str] = {
        "male": "o",
        "female": "^",
    }

    fig, ax = plt.subplots(figsize=(8.0, 6.6), dpi=150)

    labels_arr = np.asarray(labels)
    for label in ("female", "male"):
        mask = labels_arr == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            c=color_map[label],
            marker=marker_map[label],
            alpha=0.74,
            edgecolors="white",
            linewidths=0.25,
            label=f"{label.capitalize()} (n={int(mask.sum())})",
        )

    ax.set_title(f"CLIP Embedding Space ({method_name})", pad=12)
    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")

    total = len(labels)
    ax.text(
        0.01,
        0.99,
        f"Total samples: {total}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.85, "edgecolor": "#888"},
    )

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate t-SNE and UMAP plots from SD1.5 CLIP image embeddings for male/female people images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1_1/humans"),
        help="Dataset root containing female/ and male/ directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs/ethnicity"),
        help="Directory to save plots and embedding cache.",
    )
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model id (SD1.5-compatible family).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 = use all images).",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save embeddings and metadata to npz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = discover_samples(args.data_root)
    if not samples:
        raise RuntimeError(
            "No images found under female/ and male/ folders. "
            "Expected dataset tree: root/female/*.png and root/male/*.png"
        )

    samples = maybe_limit_per_class(samples, args.limit_per_class, args.seed)
    image_paths = [s.image_path for s in samples]
    labels = [s.label for s in samples]

    print(f"Found {len(samples)} images total")
    print(f"  Female: {sum(1 for x in labels if x == 'female')}")
    print(f"  Male:   {sum(1 for x in labels if x == 'male')}")
    print(f"Using device: {args.device}")
    print(f"CLIP model: {args.clip_model_id}")

    embedder = SD15CLIPImageEmbedder(model_id=args.clip_model_id, device=args.device)
    embeddings = embedder.encode_images(image_paths, batch_size=args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        npz_path = args.out_dir / "clip_embeddings_ethnicity.npz"
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,
            labels=np.asarray(labels),
            image_paths=np.asarray([str(p) for p in image_paths]),
        )
        print(f"Saved embeddings to: {npz_path}")

    tsne_coords = run_tsne(embeddings, seed=args.seed, perplexity=args.tsne_perplexity)
    umap_coords = run_umap(
        embeddings,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    tsne_png, tsne_pdf = plot_projection(
        coords=tsne_coords,
        labels=labels,
        method_name="t-SNE",
        out_dir=args.out_dir,
        stem="clip_embeddings_tsne",
    )
    umap_png, umap_pdf = plot_projection(
        coords=umap_coords,
        labels=labels,
        method_name="UMAP",
        out_dir=args.out_dir,
        stem="clip_embeddings_umap",
    )

    print("Saved plots:")
    print(f"  {tsne_png}")
    print(f"  {tsne_pdf}")
    print(f"  {umap_png}")
    print(f"  {umap_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
