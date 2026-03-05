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
        --base_path /path/to/dataset_ultimate \
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.curvton_dataloader import CURVTONDataloader

# Import metric modules
from pretrained_metrics.metrics.m1_pose import compute_pose_error
from pretrained_metrics.metrics.m2_occlusion import compute_occlusion_ratio
from pretrained_metrics.metrics.m3_background import compute_background_complexity
from pretrained_metrics.metrics.m4_illumination import compute_illumination_consistency
from pretrained_metrics.metrics.m5_body_shape import compute_body_shape_preservation
from pretrained_metrics.metrics.m6_appearance import compute_appearance_diversity
from pretrained_metrics.metrics.m7_garment_texture import compute_garment_texture_fidelity
from pretrained_metrics.metrics.unified_index import compute_unified_index


def compute_metrics_for_split(
    loader: CURVTONDataloader,
    split_name: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute all pretrained metrics for a CURVTON split.
    
    Returns
    -------
    Dict with metric names as keys and mean values
    """
    if verbose:
        print(f"\n  Computing metrics for {split_name} ({len(loader)} samples)...")
    
    # Collect per-sample metrics
    metrics_per_sample = {
        "pose_error": [],
        "occlusion_ratio": [],
        "bg_complexity": [],
        "illumination_consistency": [],
        "body_shape_preservation": [],
        "appearance_diversity": [],
        "garment_texture_fidelity": [],
    }
    
    for i, (person_path, cloth_path, tryon_path, meta) in enumerate(loader):
        if verbose and i % 200 == 0:
            print(f"    Processing {i}/{len(loader)}...")
        
        try:
            # M1: Pose Error
            pose_err = compute_pose_error(person_path, tryon_path)
            if pose_err is not None:
                metrics_per_sample["pose_error"].append(pose_err)
            
            # M2: Occlusion Ratio
            occ = compute_occlusion_ratio(person_path, cloth_path)
            if occ is not None:
                metrics_per_sample["occlusion_ratio"].append(occ)
            
            # M3: Background Complexity
            bg = compute_background_complexity(person_path)
            if bg is not None:
                metrics_per_sample["bg_complexity"].append(bg)
            
            # M4: Illumination Consistency
            illum = compute_illumination_consistency(person_path, tryon_path)
            if illum is not None:
                metrics_per_sample["illumination_consistency"].append(illum)
            
            # M5: Body Shape Preservation
            shape = compute_body_shape_preservation(person_path, tryon_path)
            if shape is not None:
                metrics_per_sample["body_shape_preservation"].append(shape)
            
            # M6: Appearance Diversity (computed at dataset level, placeholder here)
            # This will be computed separately
            
            # M7: Garment Texture Fidelity
            texture = compute_garment_texture_fidelity(cloth_path, tryon_path)
            if texture is not None:
                metrics_per_sample["garment_texture_fidelity"].append(texture)
                
        except Exception as e:
            if verbose:
                print(f"    Error at {i}: {e}")
            continue
    
    # Compute mean metrics
    results = {}
    for metric_name, values in metrics_per_sample.items():
        if values:
            results[metric_name] = float(np.mean(values))
            results[f"{metric_name}_std"] = float(np.std(values))
            results[f"{metric_name}_n"] = len(values)
        else:
            results[metric_name] = None
    
    # Compute unified index
    results["unified_index"] = compute_unified_index(results)
    
    return results


def compute_curvton_metrics(
    base_path: str,
    out_dir: str = "metrics_output/curvton",
    sample_ratio: float = 1.0,
    difficulties: List[str] = ["easy", "medium", "hard"],
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
        
        metrics = compute_metrics_for_split(loader, diff)
        all_results[diff] = metrics
        sample_counts[diff] = len(loader)
    
    # Compute combined metrics (weighted average by sample count)
    print("\n[COMBINED] Computing weighted average...")
    total_samples = sum(sample_counts.values())
    combined_metrics = {}
    
    metric_names = [
        "pose_error", "occlusion_ratio", "bg_complexity",
        "illumination_consistency", "body_shape_preservation",
        "garment_texture_fidelity"
    ]
    
    for metric in metric_names:
        weighted_sum = 0
        valid_samples = 0
        
        for diff in difficulties:
            if all_results[diff].get(metric) is not None:
                weight = sample_counts[diff]
                weighted_sum += all_results[diff][metric] * weight
                valid_samples += weight
        
        if valid_samples > 0:
            combined_metrics[metric] = weighted_sum / valid_samples
    
    combined_metrics["unified_index"] = compute_unified_index(combined_metrics)
    all_results["combined"] = combined_metrics
    
    # Add metadata
    output = {
        "dataset": "CURVTON",
        "base_path": base_path,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat(),
        "sample_counts": sample_counts,
        "total_samples": total_samples,
        "metrics": all_results,
    }
    
    # Save results
    ratio_pct = int(sample_ratio * 100)
    output_file = out_path / f"curvton_metrics_{ratio_pct}pct.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Saved metrics to {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    
    header = f"{'Metric':<30} {'Easy':>10} {'Medium':>10} {'Hard':>10} {'Combined':>10}"
    print(header)
    print("-" * len(header))
    
    for metric in metric_names + ["unified_index"]:
        row = f"{metric:<30}"
        for diff in difficulties + ["combined"]:
            val = all_results[diff].get(metric)
            if val is not None:
                row += f" {val:>9.4f}"
            else:
                row += f" {'N/A':>9}"
        print(row)
    
    print("=" * 70)
    
    return all_results


def compute_multi_ratio_metrics(
    base_path: str,
    out_dir: str = "metrics_output/curvton",
    ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
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
    
    args = parser.parse_args()
    
    if args.multi_ratio:
        compute_multi_ratio_metrics(
            base_path=args.base_path,
            out_dir=args.out_dir,
        )
    else:
        compute_curvton_metrics(
            base_path=args.base_path,
            out_dir=args.out_dir,
            sample_ratio=args.sample_ratio,
            difficulties=args.difficulties,
        )
