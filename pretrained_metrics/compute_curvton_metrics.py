"""
pretrained_metrics/compute_curvton_metrics.py
===============================================
Compute pretrained metrics for CURVTON dataset

Supports:
- Individual metrics for Easy, Medium, Hard splits
- Combined metrics (weighted average)
- Multiple sample ratios (10%, 20%, ..., 100%)
- Comparison with benchmark datasets (VITON-HD, DressCode)

Usage:
    # Single ratio
    python pretrained_metrics/compute_curvton_metrics.py \
        --base_path ../dataset_v3/dataset_ultimate \
        --out_dir metrics_output/curvton \
        --sample_ratio 1.0

    # Multiple ratios
    python pretrained_metrics/compute_curvton_metrics.py \
        --base_path /path/to/dataset_ultimate \
        --multi_ratio

    # Test set
    python pretrained_metrics/compute_curvton_metrics.py \
        --base_path /path/to/dataset_ultimate_test \
        --out_dir metrics_output/curvton_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.curvton_dataloader import CURVTONDataloader

# Import class-based metric modules
from pretrained_metrics.metrics.m1_pose import PoseMetrics
from pretrained_metrics.metrics.m2_occlusion import OcclusionMetrics
from pretrained_metrics.metrics.m3_background import BackgroundMetrics
from pretrained_metrics.metrics.m4_illumination import IlluminationMetrics
from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics
from pretrained_metrics.metrics.m6_appearance import AppearanceMetrics
from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics
from pretrained_metrics.metrics.unified_index import UnifiedComplexityIndex

# ── Image → tensor helper ────────────────────────────────────────────────────
_TO_TENSOR = T.Compose([T.Resize((512, 384)), T.ToTensor()])


def _load_image_tensor(path: str) -> torch.Tensor:
    """Load an image and return a (3, H, W) float32 tensor in [0, 1]."""
    return _TO_TENSOR(Image.open(path).convert("RGB"))


def _iterate_loader(loader, verbose=False):
    """Yield (person_tensor, cloth_tensor) from loader, skipping failures."""
    for i, (person_img, cloth_img, _tryon_img, _meta) in enumerate(loader):
        if verbose and i % 500 == 0:
            print(f"    Processing {i}/{len(loader)}...")
        try:
            if loader.return_paths:
                yield _load_image_tensor(person_img), _load_image_tensor(cloth_img)
            else:
                yield _TO_TENSOR(person_img), _TO_TENSOR(cloth_img)
        except Exception as e:
            if verbose:
                print(f"    Error at sample {i}: {e}")
            continue


def _run_single_metric(label, metric_cls, metric_kwargs, loader,
                       use_cloth, batch_size, device, verbose):
    """
    Instantiate one metric, iterate the full loader, compute, then free
    the metric (and its models) before returning results.

    Only one heavy model is resident in GPU at a time.
    """
    if verbose:
        print(f"\n    [{label}] Loading model ...")

    metric_obj = metric_cls(**metric_kwargs)

    buf: List[torch.Tensor] = []
    n = 0

    def _flush():
        nonlocal buf
        if not buf:
            return
        batch = torch.stack(buf)
        metric_obj.update(batch)
        buf = []

    for person_t, cloth_t in _iterate_loader(loader, verbose=verbose):
        buf.append(cloth_t if use_cloth else person_t)
        n += 1
        if len(buf) >= batch_size:
            _flush()
    _flush()

    sub = metric_obj.compute()
    if verbose:
        print(f"    [{label}] Done ({n} samples). Keys: {list(sub.keys())}")

    # ── Free GPU memory ──────────────────────────────────────────────────
    del metric_obj
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sub, n


def compute_metrics_for_split(
    loader: CURVTONDataloader,
    split_name: str,
    device: str = "cpu",
    batch_size: int = 16,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute all pretrained metrics for a CURVTON split.

    Metrics are computed **sequentially** so that only one heavy model is
    loaded into GPU memory at a time, preventing OOM on large datasets.

    Returns
    -------
    Dict with metric names as keys and aggregated values.
    """
    if verbose:
        print(f"\n  Computing metrics for {split_name} ({len(loader)} samples)...")

    # (label, class, constructor kwargs, uses_cloth_images)
    metric_specs = [
        ("pose",            PoseMetrics,          {"device": device}, False),
        ("occlusion",       OcclusionMetrics,     {"device": device}, False),
        ("background",      BackgroundMetrics,    {"device": device}, False),
        ("illumination",    IlluminationMetrics,  {},                 False),
        ("body_shape",      BodyShapeMetrics,     {"device": device}, False),
        ("appearance",      AppearanceMetrics,    {"device": device}, False),
        ("garment_texture", GarmentTextureMetrics,{"device": device}, True),
    ]

    results: Dict[str, float] = {}
    n_processed = 0

    for label, cls, kwargs, use_cloth in metric_specs:
        try:
            sub, n = _run_single_metric(
                label, cls, kwargs, loader,
                use_cloth=use_cloth,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
            for k, v in sub.items():
                results[k] = v
            n_processed = max(n_processed, n)
        except Exception as e:
            if verbose:
                print(f"    Warning: {label} metric failed: {e}")

    if verbose:
        print(f"    Processed {n_processed} samples successfully.")

    results["n_samples"] = n_processed
    return results


def compute_curvton_metrics(
    base_path: str,
    out_dir: str = "metrics_output/curvton",
    sample_ratio: float = 1.0,
    difficulties: List[str] = ["easy", "medium", "hard"],
    device: str = "cpu",
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Compute pretrained metrics for CURVTON dataset.
    
    Returns metrics for each difficulty level + combined average.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CURVTON Pretrained Metrics Computation")
    print("=" * 70)
    print(f"  Base path:    {base_path}")
    print(f"  Output:       {out_dir}")
    print(f"  Sample ratio: {sample_ratio:.0%}")
    print(f"  Difficulties: {difficulties}")
    print(f"  Device:       {device}")
    print("=" * 70)
    
    all_results = {}
    sample_counts = {}
    
    for diff in difficulties:
        print(f"\n[{diff.upper()}]")
        
        loader = CURVTONDataloader(
            base_path=base_path,
            difficulty=diff,
            sample_ratio=sample_ratio,
            seed=seed,
            return_paths=True,
        )
        
        metrics = compute_metrics_for_split(
            loader, diff, device=device, batch_size=batch_size,
        )
        all_results[diff] = metrics
        sample_counts[diff] = len(loader)
    
    # ── Unified Complexity Index ──────────────────────────────────────────
    uci = UnifiedComplexityIndex()
    for diff in difficulties:
        uci.add_dataset(f"CURVTON-{diff}", all_results[diff])

    # Combined: merge all difficulty metrics into one pass
    # (re-run on full dataset or just report per-difficulty)
    uci_scores = uci.compute_scores()

    # Attach unified scores to results
    for entry in uci_scores:
        diff_key = entry["dataset"].split("-")[-1].lower()
        if diff_key in all_results:
            all_results[diff_key]["unified_score"] = entry["unified_score"]
    
    # Add metadata
    output = {
        "dataset": "CURVTON",
        "base_path": base_path,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat(),
        "sample_counts": sample_counts,
        "total_samples": sum(sample_counts.values()),
        "metrics": all_results,
        "unified_index_report": uci_scores,
    }
    
    # Save results
    ratio_pct = int(sample_ratio * 100)
    output_file = out_path / f"curvton_metrics_{ratio_pct}pct.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Saved metrics to {output_file}")
    
    # Print summary
    uci.print_report(uci_scores)
    
    return all_results


def compute_multi_ratio_metrics(
    base_path: str,
    out_dir: str = "metrics_output/curvton",
    ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, Dict]:
    """
    Compute metrics for multiple sample ratios.
    """
    all_ratio_results = {}
    
    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        print(f"\n{'#' * 70}")
        print(f"# Computing metrics for {ratio_pct}% of CURVTON dataset")
        print(f"{'#' * 70}")
        
        results = compute_curvton_metrics(
            base_path=base_path,
            out_dir=out_dir,
            sample_ratio=ratio,
            device=device,
            batch_size=batch_size,
        )
        all_ratio_results[f"{ratio_pct}%"] = results
    
    # Save combined multi-ratio results
    summary_file = Path(out_dir) / "curvton_metrics_all_ratios.json"
    with open(summary_file, "w") as f:
        json.dump({
            "dataset": "CURVTON",
            "ratios": [f"{int(r*100)}%" for r in ratios],
            "results": all_ratio_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n  Saved multi-ratio summary to {summary_file}")
    
    return all_ratio_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CURVTON pretrained metrics")
    parser.add_argument(
        "--base_path", type=str, required=True,
        help="Path to CURVTON dataset"
    )
    parser.add_argument(
        "--out_dir", type=str, default="metrics_output/curvton",
        help="Output directory for metrics"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=1.0,
        help="Fraction of dataset to use (0.1 to 1.0)"
    )
    parser.add_argument(
        "--multi_ratio", action="store_true",
        help="Compute metrics for multiple sample ratios"
    )
    parser.add_argument(
        "--difficulties", type=str, nargs="+", default=["easy", "medium", "hard"],
        help="Difficulty levels to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for metric computation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for metric computation"
    )
    
    args = parser.parse_args()
    
    if args.multi_ratio:
        compute_multi_ratio_metrics(
            base_path=args.base_path,
            out_dir=args.out_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        compute_curvton_metrics(
            base_path=args.base_path,
            out_dir=args.out_dir,
            sample_ratio=args.sample_ratio,
            difficulties=args.difficulties,
            device=args.device,
            batch_size=args.batch_size,
        )
