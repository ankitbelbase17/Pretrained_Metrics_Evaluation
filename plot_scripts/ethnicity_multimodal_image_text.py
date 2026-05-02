from __future__ import annotations

import argparse
import ast
import json
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
    caption: str
    country: str
    ethnicity: str
    broad_group: str


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


def _load_people_spec(spec_path: Path) -> Dict:
    if not spec_path.exists():
        raise FileNotFoundError(f"People spec not found: {spec_path}")
    text = spec_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    if not isinstance(data, dict):
        raise ValueError("People spec must be a dict-like JSON object")
    return data


def _weighted_choice(rng: np.random.Generator, items: Sequence, probs: Sequence[float]):
    p = np.array(probs, dtype=float)
    if p.sum() <= 0:
        p = np.ones_like(p)
    p = p / p.sum()
    idx = int(rng.choice(len(items), p=p))
    return items[idx]


def _sample_from_people_spec(
    spec: Dict,
    rng: np.random.Generator,
    gender: str,
    include_body_type: bool,
    include_clothing: bool,
    include_background: bool,
    include_photo_style: bool,
) -> Tuple[str, str, str]:
    countries = spec.get("countries_ethnicities", {})
    country_names = list(countries.keys())
    country_probs = [countries[c].get("prob", 1.0) for c in country_names]
    country = str(_weighted_choice(rng, country_names, country_probs)) if country_names else "Unknown"
    ethnicity_list = countries.get(country, {}).get("ethnicities", [])
    ethnicity = str(rng.choice(ethnicity_list)) if ethnicity_list else "Unknown"

    tokens: List[str] = [gender, country, ethnicity]

    if include_body_type:
        body_types = spec.get("body_type_descriptions", {}).get("body_types", {})
        bt_names = list(body_types.keys())
        bt_probs = [body_types[b].get("prob", 1.0) for b in bt_names]
        if bt_names:
            bt_key = _weighted_choice(rng, bt_names, bt_probs)
            bt_desc = body_types.get(bt_key, {}).get("description", "")
            if bt_desc:
                tokens.append(str(bt_desc))

    if include_clothing:
        clothing = spec.get("gender", {}).get(gender, {}).get("clothing", [])
        if clothing:
            tokens.append(str(rng.choice(clothing)))

    if include_background:
        backgrounds = spec.get("background", {})
        bg_names = list(backgrounds.keys())
        bg_probs = [backgrounds[b].get("prob", 1.0) for b in bg_names]
        if bg_names:
            bg = _weighted_choice(rng, bg_names, bg_probs)
            tokens.append(str(bg).replace("_", " "))

    if include_photo_style:
        styles = spec.get("photo_style", {})
        st_names = list(styles.keys())
        st_probs = [styles[s].get("prob", 1.0) for s in st_names]
        if st_names:
            st = _weighted_choice(rng, st_names, st_probs)
            tokens.append(f"{st} photo")

    caption = " ".join([t for t in tokens if t]).strip()
    return caption, country, ethnicity


def _build_country_to_broad_group(spec: Dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    groups = spec.get("broad_ethnicity_groups", {})
    if not isinstance(groups, dict):
        return mapping
    for group_name, countries in groups.items():
        if not isinstance(countries, list):
            continue
        for c in countries:
            mapping[str(c)] = str(group_name)
    return mapping


def discover_samples(root_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []

    for label in ("female", "male"):
        subdir = root_dir / label
        if not subdir.exists():
            continue

        for path in sorted(subdir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((path, label))

    if not samples:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((path, "unknown"))

    return samples


def sample_by_ratio(samples: Sequence[Tuple[Path, str]], sample_ratio: float, seed: int) -> List[Tuple[Path, str]]:
    if sample_ratio >= 1.0:
        return list(samples)
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be > 0")
    n = len(samples)
    if n <= 1:
        return list(samples)

    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return [samples[i] for i in idx]


def fuse_embeddings(image_emb: np.ndarray, text_emb: np.ndarray, alpha: float) -> np.ndarray:
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


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n = embeddings.shape[0]
    k = max(2, min(n_clusters, max(2, n // 200)))
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


def _cluster_label(samples: Sequence[Sample], cluster_ids: np.ndarray) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c in sorted(set(cluster_ids.tolist())):
        idxs = [i for i in range(len(samples)) if int(cluster_ids[i]) == c]
        if not idxs:
            labels[c] = "Mixed"
            continue
        broad_groups = [samples[i].broad_group for i in idxs]
        g1 = Counter(broad_groups).most_common(1)
        if g1:
            labels[c] = str(g1[0][0])
        else:
            labels[c] = "Mixed"
    return labels


def _compute_centroids(coords: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for c in sorted(set(cluster_ids.tolist())):
        pts = coords[cluster_ids == c]
        centroids[c] = np.mean(pts, axis=0)
    return centroids


def _place_labels(ax, centroids: Dict[int, np.ndarray], label_texts: Dict[int, str], max_labels: int) -> None:
    if not centroids:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span_x = max(1e-8, xlim[1] - xlim[0])
    span_y = max(1e-8, ylim[1] - ylim[0])
    min_dx = 0.08 * span_x
    min_dy = 0.08 * span_y

    placed: List[Tuple[float, float]] = []
    for i, c in enumerate(sorted(centroids.keys())):
        if i >= max_labels:
            break
        bx, by = float(centroids[c][0]), float(centroids[c][1])
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
        )


def plot_multimodal_tsne(
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    samples: Sequence[Sample],
    out_dir: Path,
    stem: str,
    max_labels: int,
    show_labels: bool,
) -> Tuple[Path, Path]:
    _eccv_axes_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_clusters = sorted(set(cluster_ids.tolist()))
    gender_markers = {"female": "^", "male": "o", "unknown": "s"}

    fig, ax = plt.subplots(figsize=(12.0, 7.6), dpi=150)

    coords_arr = np.asarray(coords)
    cluster_arr = np.asarray(cluster_ids)
    gender_arr = np.asarray([s.gender for s in samples])
    broad_arr = np.asarray([s.broad_group for s in samples])
    unique_groups = sorted(set(broad_arr.tolist()))
    eth_cmap = plt.get_cmap("tab20")
    ethnicity_colors = {e: eth_cmap(i % 20) for i, e in enumerate(unique_groups)}

    for eth in unique_groups:
        for gender in ("female", "male", "unknown"):
            mask = (broad_arr == eth) & (gender_arr == gender)
            if not np.any(mask):
                continue
            ax.scatter(
                coords_arr[mask, 0],
                coords_arr[mask, 1],
                s=20,
                color=ethnicity_colors[eth],
                marker=gender_markers[gender],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.15,
            )

    if show_labels:
        centroids = _compute_centroids(coords_arr, cluster_arr)
        label_texts = _cluster_label(samples, cluster_arr)
        _place_labels(ax, centroids, label_texts, max_labels=max_labels)

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
    if "unknown" in gender_arr:
        gender_handles.append(
            plt.Line2D([0], [0], marker="s", color="#333333", linestyle="None", markersize=6, label="Unknown")
        )

    ethnicity_handles = []
    for eth in unique_groups:
        ethnicity_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=ethnicity_colors[eth],
                linestyle="None",
                markersize=5,
                label=str(eth),
            )
        )

    leg1 = ax.legend(
        handles=ethnicity_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.95,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=gender_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), framealpha=0.95)

    fig.subplots_adjust(right=0.76)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    fig.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal ethnicity plot with paired image+text CLIP embeddings."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1_1/humans"),
        help="Dataset root containing female/ and male/ directories.",
    )
    parser.add_argument(
        "--people-spec",
        type=Path,
        default=Path("./people_combined.py"),
        help="Path to people spec JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./outputs/ethnicity_multimodal_image_text"),
        help="Directory to save plots and embeddings.",
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
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--fuse-alpha", type=float, default=0.5)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--max-labels", type=int, default=8)
    parser.add_argument("--no-cluster-labels", action="store_true")
    parser.add_argument("--tsne-perplexity", type=float, default=40.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument(
        "--include-body-type",
        action="store_true",
        help="Include body type description from people spec.",
    )
    parser.add_argument(
        "--include-clothing",
        action="store_true",
        help="Include clothing token from people spec.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include background token from people spec.",
    )
    parser.add_argument(
        "--include-photo-style",
        action="store_true",
        help="Include photo style token from people spec.",
    )
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
    if not args.people_spec.exists():
        raise FileNotFoundError(f"People spec not found: {args.people_spec}")
    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1]")
    if not (0.0 <= args.fuse_alpha <= 1.0):
        raise ValueError("--fuse-alpha must be in [0, 1]")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = _load_people_spec(args.people_spec)
    country_to_group = _build_country_to_broad_group(spec)
    rng = np.random.default_rng(args.seed)

    raw_samples = discover_samples(args.data_root)
    if not raw_samples:
        raise RuntimeError("No images found under the provided data root.")

    raw_samples = sample_by_ratio(raw_samples, args.sample_ratio, args.seed)

    samples: List[Sample] = []
    for image_path, gender in raw_samples:
        if gender not in ("female", "male"):
            gender = str(_weighted_choice(rng, ["female", "male"], [0.5, 0.5]))
        caption, country, ethnicity = _sample_from_people_spec(
            spec,
            rng,
            gender,
            include_body_type=args.include_body_type,
            include_clothing=args.include_clothing,
            include_background=args.include_background,
            include_photo_style=args.include_photo_style,
        )
        samples.append(
            Sample(
                image_path=image_path,
                gender=gender,
                caption=caption,
                country=country,
                ethnicity=ethnicity,
                broad_group=country_to_group.get(country, "Other"),
            )
        )

    print(f"Images sampled: {len(samples)}")
    print(f"Using device: {args.device}")
    print(f"CLIP model: {args.clip_model_id}")

    embedder = SD15CLIPMultimodalEmbedder(model_id=args.clip_model_id, device=args.device)
    image_paths = [s.image_path for s in samples]
    captions = [s.caption for s in samples]

    image_emb = embedder.encode_images(image_paths, batch_size=args.image_batch_size)
    text_emb = embedder.encode_texts(captions, batch_size=args.text_batch_size)

    fused = fuse_embeddings(image_emb, text_emb, alpha=args.fuse_alpha)
    cluster_ids = cluster_embeddings(fused, n_clusters=args.n_clusters, seed=args.seed)
    tsne_coords = run_tsne(fused, seed=args.seed, perplexity=args.tsne_perplexity)
    umap_coords = run_umap(
        fused,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_embeddings:
        npz_path = args.out_dir / "ethnicity_multimodal_image_text.npz"
        np.savez_compressed(
            npz_path,
            image_embeddings=image_emb,
            text_embeddings=text_emb,
            fused_embeddings=fused,
            cluster_ids=cluster_ids,
            captions=np.asarray(captions),
            genders=np.asarray([s.gender for s in samples]),
            countries=np.asarray([s.country for s in samples]),
            ethnicities=np.asarray([s.ethnicity for s in samples]),
            broad_groups=np.asarray([s.broad_group for s in samples]),
            image_paths=np.asarray([str(p) for p in image_paths]),
        )
        print(f"Saved embeddings to: {npz_path}")

    png_path, pdf_path = plot_multimodal_tsne(
        coords=tsne_coords,
        cluster_ids=cluster_ids,
        samples=samples,
        out_dir=args.out_dir,
        stem="ethnicity_multimodal_image_text_tsne",
        max_labels=args.max_labels,
        show_labels=not args.no_cluster_labels,
    )
    umap_png, umap_pdf = plot_multimodal_tsne(
        coords=umap_coords,
        cluster_ids=cluster_ids,
        samples=samples,
        out_dir=args.out_dir,
        stem="ethnicity_multimodal_image_text_umap",
        max_labels=args.max_labels,
        show_labels=not args.no_cluster_labels,
    )

    print("Saved plots:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {umap_png}")
    print(f"  {umap_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
