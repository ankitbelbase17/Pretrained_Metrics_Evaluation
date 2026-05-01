from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


def add_autogen_args(parser: argparse.ArgumentParser) -> None:
    existing_opts = set()
    for action in parser._actions:
        for opt in getattr(action, "option_strings", []):
            existing_opts.add(opt)

    def _add_argument_if_missing(*names, **kwargs):
        if any(name in existing_opts for name in names):
            return
        parser.add_argument(*names, **kwargs)
        for name in names:
            existing_opts.add(name)

    _add_argument_if_missing(
        "--curvton-root",
        type=Path,
        default=None,
        help="CurvTON base path (overrides CURVTON_ROOT).",
    )
    _add_argument_if_missing(
        "--no-auto-generate",
        action="store_true",
        help="Disable auto-generation of missing CurvTON caches.",
    )
    _add_argument_if_missing("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    _add_argument_if_missing("--batch-size", type=int, default=16)
    _add_argument_if_missing("--num-workers", type=int, default=8)
    _add_argument_if_missing("--gender", type=str, default="all", choices=["all", "male", "female"])
    _add_argument_if_missing("--max-samples", type=int, default=None)
    _add_argument_if_missing("--img-size", type=int, nargs=2, default=(512, 384))
    _add_argument_if_missing(
        "--garment-backend",
        type=str,
        default="ensemble",
        choices=["ensemble", "fashion_clip", "dinov2", "sd_clip"],
        help="Garment embedding backend for cache generation.",
    )
    _add_argument_if_missing(
        "--sd-clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Model id to use when --garment-backend=sd_clip.",
    )
    _add_argument_if_missing(
        "--force-cache",
        action="store_true",
        help="Overwrite existing cache files during auto-generation.",
    )
    _add_argument_if_missing(
        "--occ-maps-dir",
        type=Path,
        default=None,
        help="Directory for separate occ_maps-only caches (required for occlusion map workflows).",
    )


def default_cache_paths(cache_dir: Path, sample_ratio: float) -> Dict[str, Path]:
    pct = int(round(sample_ratio * 100))
    return {
        "easy": cache_dir / f"curvton_easy_{pct}pct.npz",
        "medium": cache_dir / f"curvton_medium_{pct}pct.npz",
        "hard": cache_dir / f"curvton_hard_{pct}pct.npz",
    }


def resolve_curvton_root(curvton_root: Path | None) -> Path:
    if curvton_root is not None:
        return curvton_root
    env_root = os.getenv("CURVTON_ROOT") or os.getenv("CURVTON_BASE_PATH")
    if env_root:
        return Path(env_root)
    raise RuntimeError(
        "CurvTON base path not set. Provide --curvton-root or set CURVTON_ROOT."
    )


def _has_required_keys(npz_path: Path, required_keys: Iterable[str] | None) -> bool:
    if required_keys is None:
        return True
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            keys = set(data.files)
    except Exception:
        return False
    return set(required_keys).issubset(keys)


def caches_requiring_generation(
    cache_dir: Path,
    sample_ratio: float,
    difficulties: Iterable[str],
    required_keys: Iterable[str] | None = None,
) -> List[str]:
    defaults = default_cache_paths(cache_dir, sample_ratio)
    to_generate: List[str] = []
    for diff in difficulties:
        path = defaults.get(diff)
        if path is None:
            continue
        if not _has_required_keys(path, required_keys):
            to_generate.append(diff)
    return to_generate


def ensure_curvton_caches(
    difficulties: Iterable[str],
    args: argparse.Namespace,
    sample_ratio: float,
    required_keys: Iterable[str] | None = None,
) -> None:
    if args.no_auto_generate:
        return
    diffs = [d for d in difficulties if d]
    if not diffs:
        return
    diffs = caches_requiring_generation(
        cache_dir=args.cache_dir,
        sample_ratio=sample_ratio,
        difficulties=diffs,
        required_keys=required_keys,
    )
    if not diffs:
        return
    from generate_curvton_plot_features import generate_curvton_caches

    base_path = resolve_curvton_root(args.curvton_root)

    generate_curvton_caches(
        base_path=str(base_path),
        cache_dir=args.cache_dir,
        occ_maps_dir=getattr(args, "occ_maps_dir", None),
        sample_ratio=sample_ratio,
        difficulties=list(diffs),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gender=args.gender,
        max_samples=args.max_samples,
        img_size=tuple(args.img_size),
        device=args.device,
        garment_backend=args.garment_backend,
        sd_clip_model_id=args.sd_clip_model_id,
        force=True,
        required_keys=required_keys,
    )
