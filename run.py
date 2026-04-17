#!/usr/bin/env python
"""
run.py
======
Unified orchestrator for the Pretrained Metrics + EDA pipeline.

Sequentially executes:
  Phase 1: Pretrained metrics for ALL datasets (YAML config)
  Phase 2: CurvTON-only EDA (easy/medium/hard plots)
  Phase 3: Baseline dataset EDA (VITON-HD, DressCode, StreetTryOn)
  Phase 4: Comparative overlay plots (CurvTON vs all baselines)
  Phase 5: Quantitative Radar Chart (Complexity comparison)

All outputs (metrics JSON, EDA plots, CSVs, Radar charts) are saved to the project
root directory by default.

Usage (single GPU or CPU):
    python run.py

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 run.py

Usage (SLURM + multi-GPU):
    srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 run.py

Customise paths:
    python run.py \\
        --curvton_path /path/to/dataset_ultimate \\
        --vitonhd_root /path/to/vitonhd \\
        --dresscode_root /path/to/dresscode \\
        --street_tryon_root /path/to/street_tryon \\
        --output_dir .
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
# Default dataset paths (edit these or override via CLI)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "curvton_base":    "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate",
    "curvton_test":    "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test",
    "vitonhd_root":    "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/viton_hd",
    "dresscode_root":  "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/dresscode",
    "street_tryon_root": "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon",
}

BATCH_SIZE   = 40
NUM_WORKERS  = 16
SAMPLE_RATIO = 0.25  # CurvTON EDA sample ratio (25% for speed)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _banner(phase: int, total: int, title: str):
    print("\n" + "=" * 70)
    print(f"  [{phase}/{total}] {title}")
    print("=" * 70)


def _run_python(script: str, args: list, desc: str, gpus: int = 1, gpu_id: int = None) -> bool:
    """Run a Python script. If gpu_id is provided, runs on that specific GPU."""
    # Special case: CurvTON EDA natively supports torchrun
    if gpus > 1 and "run_curvton_eda" in script:
        cmd = ["torchrun", f"--nproc_per_node={gpus}", script] + args
        env = os.environ.copy()
    else:
        cmd = [sys.executable, script] + args
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
    print(f"\n  → [GPU {gpu_id if gpu_id is not None else 'ALL'}] {' '.join(cmd)}\n")
    print(f"  [DEBUG] Executing with Python: {sys.executable}")
    
    t0 = time.time()
    
    # We use Popen so we can stream output live, but if it crashes we capture the error explicitly
    process = subprocess.Popen(cmd, env=env, cwd=str(Path(__file__).parent))
    process.wait()
    
    elapsed = time.time() - t0
    
    if process.returncode != 0:
        print(f"\n" + "!" * 80)
        print(f"  [FATAL ERROR] {desc} FAILED (exit code {process.returncode}) after {elapsed:.1f}s.")
        print(f"  [DEBUG TIP] Please check your SLURM error log (pipeline_unified_*_error.log) for the exact Python traceback!")
        print(f"  [DEBUG TIP] Was the Conda environment successfully activated on the compute node?")
        print(f"  [DEBUG TIP] Path used: {sys.executable}")
        print("!" * 80 + "\n")
        return False
        
    print(f"  ✓ {desc} completed ({elapsed:.1f}s)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase runners
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_pretrained_metrics(args):
    """Phase 1: Compute pretrained metrics for all datasets via YAML config."""
    _banner(1, 4, "Pretrained Metrics — All Datasets (YAML config)")

    script = "pretrained_metrics/compute_pretrained_metrics.py"
    # ── Multi-GPU Dataset Parallelism ──
    if args.gpus > 1:
        print(f"\n  [Phase 1] Launching {args.gpus} datasets concurrently across GPUs...")
        with open("configs/pretrained_metrics_datasets.yaml") as f:
            cfg = yaml.safe_load(f)
        datasets = cfg.get("datasets", [])
        
        all_ok = True
        with ThreadPoolExecutor(max_workers=args.gpus) as executor:
            futures = []
            for i, ds in enumerate(datasets):
                ds_name = ds["name"]
                ds_args = [
                    "--dataset", ds_name,
                    "--output_dir", str(Path(args.output_dir) / "results" / "unified"),
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                ]
                # Modulo GPU assignment
                gpu_id = i % args.gpus
                futures.append(executor.submit(_run_python, script, ds_args, f"Metrics ({ds_name})", 1, gpu_id))
            
            for f in as_completed(futures):
                if not f.result():
                    all_ok = False
        
        # After parallel extraction, we must run the aggregation/normalization script once
        if all_ok:
            print("\n  [Phase 1] All datasets completed. Aggregating Unified Complexity Index...")
            return _run_python(script, [
                "--config", "configs/pretrained_metrics_datasets.yaml",
                "--output_dir", str(Path(args.output_dir) / "results" / "unified"),
            ], "Metrics Aggregation", gpus=1)
        return all_ok
    else:
        cli_args = [
            "--config", "configs/pretrained_metrics_datasets.yaml",
            "--output_dir", str(Path(args.output_dir) / "results" / "unified"),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ]
        return _run_python(script, cli_args, "Pretrained Metrics (all datasets)", gpus=1)


def phase2_curvton_eda(args):
    """Phase 2: CurvTON-only EDA (easy/medium/hard difficulty plots)."""
    _banner(2, 4, "CurvTON EDA — Difficulty-Level Plots (easy/medium/hard)")

    script = "EDA/run_curvton_eda.py"
    cli_args = [
        "--base_path", args.curvton_path,
        "--out_dir", str(Path(args.output_dir) / "plots"),
        "--cache_dir", str(Path(args.output_dir) / "eda_cache" / "curvton"),
        "--sample_ratio", str(args.sample_ratio),
    ]
    return _run_python(script, cli_args, "CurvTON EDA (difficulty splits)", gpus=args.gpus)


def phase3_baseline_eda(args):
    """Phase 3: EDA for each baseline dataset individually."""
    _banner(3, 4, "Baseline Dataset EDA — VITON-HD, DressCode, StreetTryOn")

    baselines = [
        ("vitonhd",      args.vitonhd_root,      "VITON-HD"),
        ("dresscode",    args.dresscode_root,     "DressCode"),
        ("street_tryon", args.street_tryon_root,  "StreetTryOn"),
    ]

    script = "EDA/run_eda.py"
    all_ok = True
    
    if args.gpus > 1:
        print(f"\n  [Phase 3] Launching baseline EDAs concurrently across {args.gpus} GPUs...")
        with ThreadPoolExecutor(max_workers=args.gpus) as executor:
            futures = []
            for i, (ds_name, ds_root, display_name) in enumerate(baselines):
                if not ds_root: continue
                cli_args = [
                    "--dataset", ds_name,
                    "--root", ds_root,
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                    "--cache_dir", str(Path(args.output_dir) / "eda_cache" / ds_name),
                    "--out_dir", str(Path(args.output_dir) / "plots"),
                ]
                gpu_id = i % args.gpus
                futures.append(executor.submit(_run_python, script, cli_args, f"EDA ({display_name})", 1, gpu_id))
            
            for f in as_completed(futures):
                if not f.result():
                    all_ok = False
    else:
        for ds_name, ds_root, display_name in baselines:
            if not ds_root:
                print(f"\n  [SKIP] {display_name} — no root path provided")
                continue

            print(f"\n  ── {display_name} ──")
            cli_args = [
                "--dataset", ds_name,
                "--root", ds_root,
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--cache_dir", str(Path(args.output_dir) / "eda_cache" / ds_name),
                "--out_dir", str(Path(args.output_dir) / "plots"),
            ]
            ok = _run_python(script, cli_args, f"EDA ({display_name})", gpus=1)
            if not ok:
                all_ok = False

    return all_ok


def phase4_comparison_plots(args):
    """Phase 4: Overlapped comparison plots — CurvTON vs all baselines."""
    _banner(4, 4, "Comparative Overlay Plots — CurvTON vs Baselines")

    # Collect all available cache directories and labels
    cache_base = Path(args.output_dir) / "eda_cache"
    cache_dirs = []
    labels = []

    # CurvTON difficulty caches (from run_curvton_eda.py — stored as npz per difficulty)
    curvton_cache = cache_base / "curvton"
    for diff in ["easy", "medium", "hard"]:
        # run_curvton_eda saves as curvton_{diff}_{pct}pct.npz
        ratio_pct = int(args.sample_ratio * 100)
        npz_file = curvton_cache / f"curvton_{diff}_{ratio_pct}pct.npz"
        if npz_file.exists():
            cache_dirs.append(str(npz_file))
            labels.append(f"CurvTON-{diff.capitalize()}")

    # Baseline caches (from run_eda.py — stored as {name}_features.npz)
    for ds_name, display_name in [
        ("vitonhd", "VITON-HD"),
        ("dresscode", "DressCode"),
        ("street_tryon", "StreetTryOn"),
    ]:
        ds_cache = cache_base / ds_name / f"{ds_name}_features.npz"
        if ds_cache.exists():
            cache_dirs.append(str(ds_cache))
            labels.append(display_name)

    if len(labels) < 2:
        print("  [SKIP] Need at least 2 cached datasets for comparison plots")
        return True

    # Use run_eda.py --figs_only to generate comparison plots from cached features
    # We need to load the npz files and pass them with matching labels
    print(f"  Datasets found for comparison: {', '.join(labels)}")

    # Generate comparison plots using a small inline script
    # (run_eda.py --figs_only expects cache_dir with {label}_features.npz naming)
    comparison_script = str(Path(args.output_dir) / "_run_comparison.py")
    _write_comparison_script(comparison_script, cache_dirs, labels,
                             str(Path(args.output_dir) / "plots"))

    ok = _run_python(comparison_script, [], "Comparison overlay plots")

    # Clean up temporary script
    try:
        Path(comparison_script).unlink()
    except OSError:
        pass

    return ok


def _write_comparison_script(path: str, cache_files: list, labels: list, out_dir: str):
    """Write a small script that loads cached npz and generates comparison plots."""
    code = f'''#!/usr/bin/env python
"""Auto-generated comparison plot script."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "EDA"))
from EDA.run_eda import run_all_plots

cache_files = {cache_files!r}
labels = {labels!r}
out_dir = {out_dir!r}

all_data = {{}}
for cf, lbl in zip(cache_files, labels):
    p = Path(cf)
    if p.exists():
        all_data[lbl] = dict(np.load(str(p), allow_pickle=True))
        print(f"  Loaded {{lbl}}: {{len(list(all_data[lbl].keys()))}} feature keys")
    else:
        print(f"  [WARN] Not found: {{cf}}")

if len(all_data) >= 2:
    run_all_plots(all_data, out_root=out_dir, no_pairplot=True)
else:
    print("  [SKIP] Not enough cached datasets for comparison.")
'''
    Path(path).write_text(code, encoding="utf-8")


def phase5_radar_chart(args):
    """Phase 5: Generate quantitative radar chart from comprehensive metrics JSON."""
    _banner(5, 5, "Quantitative Radar Chart — Dataset Complexity Comparison")
    
    # Find the most recent comprehensive JSON
    import glob
    json_files = glob.glob(str(Path(args.output_dir) / "pretrained_metrics_comprehensive_*.json"))
    if not json_files:
        print("  [SKIP] No comprehensive JSON metrics found. Run Phase 1 first.")
        return True
        
    latest_json = max(json_files, key=os.path.getctime)
    print(f"  Using metrics from: {latest_json}")
    
    comparison_script = str(Path(args.output_dir) / "_run_radar.py")
    code = f'''#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, "{Path(__file__).parent}")
from EDA.plots.p12_radar_chart import generate_radar_chart

generate_radar_chart("{latest_json}", "{Path(args.output_dir) / 'plots' / 'radar'}")
'''
    Path(comparison_script).write_text(code, encoding="utf-8")
    
    ok = _run_python(comparison_script, [], "Radar Chart Generation")
    
    try:
        Path(comparison_script).unlink()
    except OSError:
        pass
        
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified pipeline: Pretrained Metrics + EDA for all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Dataset paths ─────────────────────────────────────────────────────
    p.add_argument("--curvton_path",      type=str, default=DEFAULTS["curvton_base"],
                   help="CurvTON dataset_ultimate root path")
    p.add_argument("--vitonhd_root",      type=str, default=DEFAULTS["vitonhd_root"],
                   help="VITON-HD test root path")
    p.add_argument("--dresscode_root",    type=str, default=DEFAULTS["dresscode_root"],
                   help="DressCode test root path")
    p.add_argument("--street_tryon_root", type=str, default=DEFAULTS["street_tryon_root"],
                   help="StreetTryOn test root path")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default=".",
                   help="Project root output directory (default: current directory)")
    p.add_argument("--metrics_config", type=str,
                   default="configs/pretrained_metrics_datasets.yaml",
                   help="YAML config for pretrained metrics datasets")

    # ── Processing ────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers",  type=int, default=NUM_WORKERS)
    p.add_argument("--gpus", type=int, default=1,
                   help="Number of GPUs to use for DataParallel processing (torchrun)")
    p.add_argument("--sample_ratio", type=float, default=SAMPLE_RATIO,
                   help="CurvTON EDA sample ratio (default: 0.25)")

    # ── Phase selection ───────────────────────────────────────────────────
    p.add_argument("--skip_metrics",    action="store_true",
                   help="Skip Phase 1 (pretrained metrics)")
    p.add_argument("--skip_curvton_eda", action="store_true",
                   help="Skip Phase 2 (CurvTON-only EDA)")
    p.add_argument("--skip_baseline_eda", action="store_true",
                   help="Skip Phase 3 (baseline dataset EDA)")
    p.add_argument("--skip_comparison", action="store_true",
                   help="Skip Phase 4 (comparison overlay plots)")
    p.add_argument("--skip_radar", action="store_true",
                   help="Skip Phase 5 (radar chart)")
    p.add_argument("--only", type=int, nargs="+", default=None,
                   help="Run only specific phases, e.g. --only 1 5")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("  UNIFIED PIPELINE: Pretrained Metrics + EDA")
    print("=" * 70)
    print(f"  Output directory : {Path(args.output_dir).resolve()}")
    print(f"  Metrics config   : {args.metrics_config}")
    print(f"  CurvTON path     : {args.curvton_path}")
    print(f"  VITON-HD root    : {args.vitonhd_root}")
    print(f"  DressCode root   : {args.dresscode_root}")
    print(f"  StreetTryOn root : {args.street_tryon_root}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  GPUs (Parallel)  : {args.gpus}")
    print(f"  Sample ratio     : {args.sample_ratio}")
    print("=" * 70)

    # Determine which phases to run
    if args.only:
        phases = set(args.only)
    else:
        phases = {1, 2, 3, 4, 5}
        if args.skip_metrics:
            phases.discard(1)
        if args.skip_curvton_eda:
            phases.discard(2)
        if args.skip_baseline_eda:
            phases.discard(3)
        if args.skip_comparison:
            phases.discard(4)
        if args.skip_radar:
            phases.discard(5)

    results = {}

    # ── Phase 1: Pretrained Metrics ───────────────────────────────────────
    if 1 in phases:
        results[1] = phase1_pretrained_metrics(args)
    else:
        print("\n  [SKIP] Phase 1: Pretrained Metrics")

    # ── Phase 2: CurvTON-only EDA ────────────────────────────────────────
    if 2 in phases:
        results[2] = phase2_curvton_eda(args)
    else:
        print("\n  [SKIP] Phase 2: CurvTON EDA")

    # ── Phase 3: Baseline EDA ─────────────────────────────────────────────
    if 3 in phases:
        results[3] = phase3_baseline_eda(args)
    else:
        print("\n  [SKIP] Phase 3: Baseline EDA")

    # ── Phase 4: Comparison plots ─────────────────────────────────────────
    if 4 in phases:
        results[4] = phase4_comparison_plots(args)
    else:
        print("\n  [SKIP] Phase 4: Comparison Plots")

    # ── Phase 5: Radar Chart ──────────────────────────────────────────────
    if 5 in phases:
        results[5] = phase5_radar_chart(args)
    else:
        print("\n  [SKIP] Phase 5: Radar Chart")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    for phase_id, ok in sorted(results.items()):
        status = "✓" if ok else "✗"
        names = {1: "Pretrained Metrics", 2: "CurvTON EDA",
                 3: "Baseline EDA", 4: "Comparison Plots",
                 5: "Radar Chart"}
        print(f"  {status} Phase {phase_id}: {names[phase_id]}")
    print(f"\n  Outputs saved to: {Path(args.output_dir).resolve()}")
    print(f"    Metrics JSON  : pretrained_metrics_comprehensive_*.json")
    print(f"    Plots         : plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
