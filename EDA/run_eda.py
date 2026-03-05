"""
EDA/run_eda.py
===============
Master EDA runner for the VTON Dataset Analytics Pipeline.

Step 1:  Feature extraction  (once per dataset, cached to .npz)
Step 2:  Generate all 8 plot families (P1–P7 + Meta)

The pipeline works in two modes:
  (A) Real datasets — reads images via dataloaders
  (B) Synthetic mode (--dry_run) — generates random tensors, skips dataloaders

Usage
------
  # Dry-run smoke-test (no dataset needed):
  python EDA/run_eda.py --dry_run

  # Single real dataset:
  python EDA/run_eda.py \\
      --dataset viton --root /data/VITON --batch_size 16 --device cuda

  # Multiple datasets (YAML config):
  python EDA/run_eda.py --config configs/pretrained_metrics_datasets.yaml

  # Skip re-extraction if cache exists:
  python EDA/run_eda.py --dataset viton --root /data/VITON --skip_extraction

  # Generate figures only (cache must exist):
  python EDA/run_eda.py --figs_only --cache_dir ./eda_cache \\
      --labels viton viton_hd --out_dir figures/

Output
-------
  eda_cache/<dataset>_features.npz      ← per-image feature arrays
  figures/pose/           ← P1 figures (PDF+PNG)
  figures/occlusion/      ← P2
  figures/background/     ← P3
  figures/illumination/   ← P4
  figures/body_shape/     ← P5
  figures/appearance/     ← P6
  figures/garment/        ← P7
  figures/meta/           ← P8 correlation matrix
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))               # EDA/
sys.path.insert(0, str(_HERE.parent))        # workspace root

# ── plot modules ────────────────────────────────────────────────────────────
from plots.p1_pose_eda        import plot_pose_umap, plot_joint_angle_distributions
from plots.p2_occlusion_eda   import plot_occlusion_histogram, plot_occlusion_heatmap
from plots.p3_background_eda  import plot_bg_entropy_histogram, plot_entropy_vs_objects
from plots.p4_illumination_eda import plot_luminance_spectrum, plot_illumination_pca
from plots.p5_body_shape_eda  import plot_shape_pca, plot_shape_coefficient_histograms
from plots.p6_appearance_eda  import plot_face_umap, plot_pairwise_distance_distribution
from plots.p7_garment_eda     import plot_garment_umap, plot_eigenvalue_spectrum
from plots.p8_meta_correlation import (
    plot_correlation_matrix, plot_scatter_matrix, _build_feature_matrix
)
from plots.p9_vae_eda          import (
    plot_vae_pca, plot_vae_pca_combined, plot_vae_explained_variance, plot_vae_tsne
)


# ─────────────────────────────────────────────────────────────────────────────
# All-plot runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_plots(
    all_data: Dict[str, dict],   # {dataset_name: loaded .npz dict}
    out_root: str = "figures",
    skip_figures: List[str] = None,
    no_pairplot: bool = False,
):
    skip = set(skip_figures or [])
    P    = Path(out_root)

    print("\n" + "═" * 60)
    print("  Generating EDA figures …")
    print("═" * 60)

    # ── P1: Pose ─────────────────────────────────────────────────────────
    if "p1" not in skip:
        print("\n  [P1] Pose …")
        plot_pose_umap(
            {n: d["pose_vecs"] for n, d in all_data.items()},
            out_dir=str(P / "pose"),
        )
        plot_joint_angle_distributions(
            {n: d["angles"] for n, d in all_data.items()},
            out_dir=str(P / "pose"),
        )

    # ── P2: Occlusion ────────────────────────────────────────────────────
    if "p2" not in skip:
        print("\n  [P2] Occlusion …")
        plot_occlusion_histogram(
            {n: d["occ_ratios"] for n, d in all_data.items()},
            out_dir=str(P / "occlusion"),
        )
        plot_occlusion_heatmap(
            {n: d["occ_maps"] for n, d in all_data.items()},
            out_dir=str(P / "occlusion"),
        )

    # ── P3: Background ───────────────────────────────────────────────────
    if "p3" not in skip:
        print("\n  [P3] Background …")
        plot_bg_entropy_histogram(
            {n: d["bg_entropy"] for n, d in all_data.items()},
            out_dir=str(P / "background"),
        )
        plot_entropy_vs_objects(
            {n: d["bg_entropy"]    for n, d in all_data.items()},
            {n: d["bg_obj_count"].astype(float) for n, d in all_data.items()},
            out_dir=str(P / "background"),
        )

    # ── P4: Illumination ─────────────────────────────────────────────────
    if "p4" not in skip:
        print("\n  [P4] Illumination …")
        plot_luminance_spectrum(
            {n: d["lum_mean"]     for n, d in all_data.items()},
            {n: d["lum_grad_var"] for n, d in all_data.items()},
            out_dir=str(P / "illumination"),
        )
        plot_illumination_pca(
            {n: d["lum_maps"] for n, d in all_data.items()},
            out_dir=str(P / "illumination"),
        )

    # ── P5: Body Shape ───────────────────────────────────────────────────
    if "p5" not in skip:
        print("\n  [P5] Body Shape …")
        plot_shape_pca(
            {n: d["betas"] for n, d in all_data.items()},
            out_dir=str(P / "body_shape"),
        )
        plot_shape_coefficient_histograms(
            {n: d["betas"] for n, d in all_data.items()},
            out_dir=str(P / "body_shape"),
        )

    # ── P6: Appearance ───────────────────────────────────────────────────
    if "p6" not in skip:
        print("\n  [P6] Appearance …")
        plot_face_umap(
            {n: d["face_embs"] for n, d in all_data.items()},
            out_dir=str(P / "appearance"),
        )
        plot_pairwise_distance_distribution(
            {n: d["face_embs"] for n, d in all_data.items()},
            out_dir=str(P / "appearance"),
        )

    # ── P7: Garment ──────────────────────────────────────────────────────
    if "p7" not in skip:
        print("\n  [P7] Garment …")
        plot_garment_umap(
            {n: d["garment_embs"] for n, d in all_data.items()},
            out_dir=str(P / "garment"),
        )
        plot_eigenvalue_spectrum(
            {n: d["garment_embs"] for n, d in all_data.items()},
            out_dir=str(P / "garment"),
        )

    # ── P8: Meta correlation ─────────────────────────────────────────────
    if "p8" not in skip:
        print("\n  [P8] Meta correlation …")
        Xs = {n: _build_feature_matrix(d) for n, d in all_data.items()}
        plot_correlation_matrix(Xs, out_dir=str(P / "meta"))
        if not no_pairplot:
            plot_scatter_matrix(Xs, out_dir=str(P / "meta"))
    # ── P9: VAE Latent Space ────────────────────────────────────────
    if "p9" not in skip:
        # Check if VAE embeddings exist
        has_vae = any("vae_embs" in d for d in all_data.values())
        if has_vae:
            print("\n  [P9] VAE Latent Space …")
            vae_data = {n: d["vae_embs"] for n, d in all_data.items() if "vae_embs" in d}
            plot_vae_pca(vae_data, out_dir=str(P / "vae"))
            plot_vae_pca_combined(vae_data, out_dir=str(P / "vae"))
            plot_vae_explained_variance(vae_data, out_dir=str(P / "vae"))
            plot_vae_tsne(vae_data, out_dir=str(P / "vae"))
        else:
            print("\n  [P9] Skipping VAE plots (no vae_embs in cache)")
    print("\n  ✓  All EDA figures complete.")
    print(f"     Output → {P.resolve()}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run: generate synthetic .npz data
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_data(n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    H, W = 64, 48
    return dict(
        pose_vecs   = rng.normal(0, 1, (n, 34)).astype(np.float32),
        angles      = rng.uniform(0, np.pi, (n, 8)).astype(np.float32),
        occ_ratios  = rng.beta(2, 5, n).astype(np.float32),
        occ_maps    = rng.random((n, H, W)).astype(np.float32),
        bg_entropy  = rng.uniform(3, 5, n).astype(np.float32),
        bg_obj_count= rng.integers(0, 15, n).astype(np.int32),
        lum_mean    = rng.uniform(0.2, 0.8, n).astype(np.float32),
        lum_grad_var= rng.exponential(0.01, n).astype(np.float32),
        lum_maps    = rng.random((n, H, W)).astype(np.float32),
        betas       = rng.normal(0, 1, (n, 10)).astype(np.float32),
        face_embs   = rng.normal(0, 1, (n, 512)).astype(np.float32),
        garment_embs= rng.normal(0, 1, (n, 512)).astype(np.float32),
        vae_embs    = rng.normal(0, 1, (n, 4*64*48)).astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="EDA: Dataset Analytics Pipeline for VTON datasets"
    )
    p.add_argument("--config",          type=str,   default=None)
    p.add_argument("--dataset",         type=str,   default=None)
    p.add_argument("--root",            type=str,   default=None)
    p.add_argument("--dry_run",         action="store_true")
    p.add_argument("--figs_only",       action="store_true",
                   help="Skip extraction, load from cache only")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Use cached .npz if it exists, else extract")
    p.add_argument("--cache_dir",       type=str,   default="./eda_cache")
    p.add_argument("--out_dir",         type=str,   default="./figures")
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--img_size",        type=int,   nargs=2,
                   default=[512, 384], metavar=("H", "W"))
    p.add_argument("--split",           type=str,   default="test")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--labels",          nargs="+",  default=None,
                   help="Dataset labels (used with --figs_only)")
    p.add_argument("--skip",            nargs="*",  default=[],
                   help="Plot families to skip, e.g. --skip p1 p6")
    p.add_argument("--no_pairplot",     action="store_true")
    p.add_argument("--dresscode_category", type=str, default="upper_body")
    p.add_argument("--use_anish", action="store_true", help="Use dedicated dataloaders from dataloaders_anish/")
    p.add_argument("--no_resume",  action="store_true",
                   help="Ignore checkpoint and re-extract all datasets from scratch.")
    return p.parse_args()


# Configuration management
try:
    import config
except ImportError:
    config = None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eda_cache_label(name: str, cfg: dict) -> str:
    """Unique label used for both the cache filename and the checkpoint key."""
    cat = cfg.get("dresscode_category", "")
    return f"{name}_{cat}" if cat else name


def _load_eda_checkpoint(cache_dir: str) -> set:
    """Returns a set of cache_labels that have already been extracted."""
    import json
    path = Path(cache_dir) / "eda_checkpoint.json"
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        done = set(data.get("completed", []))
        if done:
            print(f"  [Resume] {len(done)} dataset(s) already done: {', '.join(sorted(done))}")
        return done
    except Exception as e:
        print(f"  [Resume] Could not read EDA checkpoint ({e}). Starting fresh.")
        return set()


def _write_eda_checkpoint(done: set, cache_dir: str):
    """Persist the set of completed cache_labels."""
    import json
    path = Path(cache_dir) / "eda_checkpoint.json"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"completed": sorted(done)}, f, indent=2)


def main():
    args = _parse()

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[EDA] DRY RUN — Synthetic tensors (1 dataset × 200 images)")
        ds_names = ["synth_dataset_A", "synth_dataset_B"]
        all_data = {
            n: _make_synthetic_data(n=200, seed=i)
            for i, n in enumerate(ds_names)
        }
        run_all_plots(all_data, out_root=args.out_dir,
                      skip_figures=args.skip, no_pairplot=args.no_pairplot)
        return

    # ── Figures only (load from cache) ───────────────────────────────────
    if args.figs_only:
        labels = args.labels or []
        cache  = Path(args.cache_dir)
        all_data = {}
        for lbl in labels:
            cp = cache / f"{lbl}_features.npz"
            if not cp.exists():
                print(f"  [WARN] Cache not found: {cp}")
                continue
            all_data[lbl] = dict(np.load(cp, allow_pickle=True))
        if not all_data:
            print("[ERR] No cached data found. Run without --figs_only first.")
            return
        run_all_plots(all_data, out_root=args.out_dir,
                      skip_figures=args.skip, no_pairplot=args.no_pairplot)
        return

    # ── Real extraction ───────────────────────────────────────────────────
    from feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(device=args.device, cache_dir=args.cache_dir)

    resume  = not args.no_resume
    force   = args.no_resume          # only force when explicitly asked
    done    = _load_eda_checkpoint(args.cache_dir) if resume else set()

    base_kwargs = dict(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        force=force,
        use_anish=args.use_anish,
    )

    all_data: Dict[str, dict] = {}

    # Pre-load already-completed datasets so figures can still be generated
    for label in done:
        cp = Path(args.cache_dir) / f"{label}_features.npz"
        if cp.exists():
            all_data[label] = dict(np.load(cp, allow_pickle=True))

    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        defaults = raw.get("defaults", {})
        for entry in raw.get("datasets", []):
            cfg         = {**defaults, **entry}
            name        = cfg.get("name")
            root        = cfg.get("root")
            cache_label = _eda_cache_label(name, cfg)

            if cache_label in done:
                print(f"\n  [Skip] {cache_label} (already extracted — use --no_resume to redo)")
                continue

            if not root and config:
                root = config.get_root(name)
            kw = {**base_kwargs, "cache_label": cache_label}
            if "dresscode" in name.lower():
                kw["dresscode_category"] = cfg.get("dresscode_category",
                                                    args.dresscode_category)
            try:
                all_data[cache_label] = extractor.extract(name, root, **kw)
                done.add(cache_label)
                _write_eda_checkpoint(done, args.cache_dir)
                print(f"  [Checkpoint] Saved ({len(done)} dataset(s) done).")
            except Exception as e:
                print(f"  [ERROR] {cache_label}: {e}")
                import traceback; traceback.print_exc()

    elif args.dataset:
        root        = args.root
        cache_label = _eda_cache_label(args.dataset, {"dresscode_category": args.dresscode_category})
        if cache_label in done:
            print(f"\n  [Skip] {cache_label} already extracted. Use --no_resume to redo.")
        else:
            if not root and config:
                root = config.get_root(args.dataset)
            kw = {**base_kwargs, "cache_label": cache_label}
            if "dresscode" in args.dataset.lower():
                kw["dresscode_category"] = args.dresscode_category
            all_data[cache_label] = extractor.extract(args.dataset, root, **kw)
            done.add(cache_label)
            _write_eda_checkpoint(done, args.cache_dir)

    else:
        print("[EDA] No action. Use --dry_run, --dataset+--root, --config, or --figs_only.")
        return

    if not all_data:
        print("\n  [EDA] No data to plot.")
        return

    run_all_plots(all_data, out_root=args.out_dir,
                  skip_figures=args.skip, no_pairplot=args.no_pairplot)


if __name__ == "__main__":
    main()
