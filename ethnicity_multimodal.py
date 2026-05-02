from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm


@dataclass(frozen=True)
class EthnicitySample:
    caption: str
    gender: str
    country: str
    ethnicity_group: str


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

        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding ethnicity text CLIP embeddings"):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feats = self._extract_text_features(inputs)
            feats = F.normalize(feats.float(), dim=-1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


# Broad, readable semantic groups for cluster labeling.
ETHNICITY_GROUPS: Dict[str, List[str]] = {
    "East Asian Traditions": ["Japan", "Korea", "China", "Taiwan", "Mongolia", "Vietnam"],
    "South Asian Traditions": ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Bhutan"],
    "Southeast Asian Traditions": ["Thailand", "Indonesia", "Malaysia", "Philippines", "Cambodia", "Laos"],
    "Middle Eastern Traditions": ["Egypt", "Saudi Arabia", "Jordan", "Lebanon", "Turkey", "Iran"],
    "North African Traditions": ["Morocco", "Algeria", "Tunisia", "Libya", "Sudan", "Ethiopia"],
    "Sub-Saharan African Traditions": ["Nigeria", "Ghana", "Kenya", "Senegal", "Ethiopia", "South Africa"],
    "European Traditions": ["France", "Germany", "Italy", "Spain", "Poland", "Greece"],
    "Latin American Traditions": ["Mexico", "Brazil", "Peru", "Colombia", "Chile", "Argentina"],
    "North American Traditions": ["United States", "Canada"],
    "Oceanic Traditions": ["Australia", "New Zealand", "Samoa", "Tonga", "Fiji"],
    "Indigenous Traditions": ["Inuit", "Maori", "Aboriginal", "Ainu", "Mapuche"],
}

GENDER_WORDS = ["female", "male"]


def build_caption(gender: str, country: str, ethnicity_group: str) -> str:
    """Create a short CLIP-friendly caption with gender + country + ethnicity group."""
    return f"{gender} {country} {ethnicity_group}".strip()


def generate_ethnicity_samples(num_samples: int, seed: int) -> List[EthnicitySample]:
    """Randomly sample text captions; the sampling is not sequential."""
    rng = np.random.default_rng(seed)
    groups = list(ETHNICITY_GROUPS.keys())
    samples: List[EthnicitySample] = []

    for _ in range(num_samples):
        gender = rng.choice(GENDER_WORDS)
        group = rng.choice(groups)
        country = rng.choice(ETHNICITY_GROUPS[group])
        caption = build_caption(gender, country, group)
        samples.append(
            EthnicitySample(
                caption=caption,
                gender=str(gender),
                country=str(country),
                ethnicity_group=str(group),
            )
        )

    return samples


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
    k = max(2, min(n_clusters, max(2, n // 250)))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return kmeans.fit_predict(embeddings)


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


def semantic_label_from_captions(captions: Sequence[str], cluster_ids: np.ndarray) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c in sorted(set(cluster_ids.tolist())):
        cluster_caps = [captions[i] for i in range(len(captions)) if int(cluster_ids[i]) == c]
        if not cluster_caps:
            labels[c] = "Mixed Identity"
            continue

        group_counts = Counter()
        country_counts = Counter()
        gender_counts = Counter()
        for cap in cluster_caps:
            parts = cap.split()
            if len(parts) >= 2:
                gender_counts[parts[0]] += 1
                country_counts[parts[1]] += 1
            if len(parts) >= 3:
                group_counts[" ".join(parts[2:])] += 1

        top_group = group_counts.most_common(1)
        top_country = country_counts.most_common(1)
        top_gender = gender_counts.most_common(1)

        if top_group:
            group_name = top_group[0][0]
        else:
            group_name = "Mixed Identity"

        if top_country and len(cluster_caps) > 30:
            # Keep the label broad: prefer region-level grouping, not country-level labels.
            labels[c] = group_name
        elif top_gender and len(cluster_caps) < 30:
            labels[c] = f"{group_name} ({top_gender[0][0]})"
        else:
            labels[c] = group_name

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
    min_points_to_label: int,
) -> None:
    if not centroids:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = max(1e-8, xlim[1] - xlim[0])
    span_y = max(1e-8, ylim[1] - ylim[0])
    min_dx = 0.08 * span_x
    min_dy = 0.08 * span_y

    ranked = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(c, s) for c, s in ranked if s >= min_points_to_label]
    if not ranked:
        ranked = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:max_labels]

    placed: List[Tuple[float, float]] = []
    for c, _sz in ranked:
        base = centroids[c]
        bx, by = float(base[0]), float(base[1])
        found_x, found_y = bx, by

        for step in range(18):
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
                break

        placed.append((found_x, found_y))
        ax.annotate(
            label_texts.get(c, f"cluster_{c}"),
            xy=(bx, by),
            xytext=(found_x, found_y),
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "alpha": 0.86, "edgecolor": "#666"},
            arrowprops={"arrowstyle": "-", "color": "#777", "lw": 0.7, "alpha": 0.8},
        )


def plot_ethnicity_multimodal_tsne(
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    samples: Sequence[EthnicitySample],
    out_dir: Path,
    stem: str,
    max_labels: int,
    min_cluster_fraction_for_label: float,
) -> Tuple[Path, Path]:
    _eccv_axes_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_groups = sorted({s.ethnicity_group for s in samples})
    cmap = plt.get_cmap("tab20")
    group_colors = {g: cmap(i % 20) for i, g in enumerate(unique_groups)}
    gender_markers = {"female": "^", "male": "o"}

    fig, ax = plt.subplots(figsize=(10.0, 7.6), dpi=150)

    coords_arr = np.asarray(coords)
    gender_arr = np.asarray([s.gender for s in samples])
    group_arr = np.asarray([s.ethnicity_group for s in samples])

    for group in unique_groups:
        for gender in ("female", "male"):
            mask = (group_arr == group) & (gender_arr == gender)
            if not np.any(mask):
                continue
            ax.scatter(
                coords_arr[mask, 0],
                coords_arr[mask, 1],
                s=18,
                color=group_colors[group],
                marker=gender_markers[gender],
                alpha=0.68,
                edgecolors="white",
                linewidths=0.18,
            )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    gender_handles = [
        plt.Line2D([0], [0], marker="^", color="#333333", linestyle="None", markersize=6, label="Female"),
        plt.Line2D([0], [0], marker="o", color="#333333", linestyle="None", markersize=6, label="Male"),
    ]

    ethnicity_handles = []
    for group in unique_groups:
        ethnicity_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=group_colors[group],
                markeredgecolor="white",
                markeredgewidth=0.3,
                markersize=6,
                label=group,
            )
        )

    leg1 = ax.legend(handles=ethnicity_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Ethnicity group")
    ax.add_artist(leg1)
    ax.legend(handles=gender_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), title="Gender marker")

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random gender+country+ethnicity captions, embed them with CLIP, and plot clustered t-SNE."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs/ethnicity_multimodal"),
        help="Directory to save plots and embeddings.",
    )
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model id (SD1.5-compatible family).",
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=14)
    parser.add_argument("--max-labels", type=int, default=8)
    parser.add_argument(
        "--min-cluster-fraction-for-label",
        type=float,
        default=0.04,
        help="Only annotate clusters with at least this fraction of samples.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=40.0)
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

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not (0.0 <= args.min_cluster_fraction_for_label <= 1.0):
        raise ValueError("--min-cluster-fraction-for-label must be in [0, 1]")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = generate_ethnicity_samples(args.num_samples, seed=args.seed)
    captions = [s.caption for s in samples]

    print(f"Generated {len(samples)} text samples")
    print(f"Using device: {args.device}")
    print(f"CLIP model: {args.clip_model_id}")

    embedder = SD15CLIPTextEmbedder(model_id=args.clip_model_id, device=args.device)
    embeddings = embedder.encode_texts(captions, batch_size=args.batch_size)

    cluster_ids = cluster_embeddings(embeddings, n_clusters=args.n_clusters, seed=args.seed)
    tsne_coords = run_tsne(embeddings, seed=args.seed, perplexity=args.tsne_perplexity)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        npz_path = args.out_dir / "ethnicity_multimodal_embeddings.npz"
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,
            captions=np.asarray(captions),
            genders=np.asarray([s.gender for s in samples]),
            countries=np.asarray([s.country for s in samples]),
            ethnicity_groups=np.asarray([s.ethnicity_group for s in samples]),
            cluster_ids=cluster_ids,
        )
        print(f"Saved embeddings to: {npz_path}")

    png_path, pdf_path = plot_ethnicity_multimodal_tsne(
        coords=tsne_coords,
        cluster_ids=cluster_ids,
        samples=samples,
        out_dir=args.out_dir,
        stem="ethnicity_multimodal_clustered_tsne",
        max_labels=args.max_labels,
        min_cluster_fraction_for_label=args.min_cluster_fraction_for_label,
    )

    print("Saved plots:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
