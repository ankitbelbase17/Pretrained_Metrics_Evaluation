"""
pretrained_metrics/compute_pretrained_metrics.py
==================================================
Main entry-point for the Pretrained-Model Dataset Complexity Pipeline.

Computes 7 dataset-complexity metrics for each of the 10 try-on datasets:

  ┌────┬──────────────────────────────┬────────────────────────────────────┐
  │ M  │ Metric Family                │ Key outputs                        │
  ├────┼──────────────────────────────┼────────────────────────────────────┤
  │ 1  │ Pose Diversity               │ D_pose (log-det cov)               │
  │    │ Pose Articulation Complexity │ C_artic (Σ joint-angle var)        │
  │ 2  │ Occlusion Complexity         │ C_occ = E[O] + Var(O)              │
  │ 3  │ Background Complexity        │ H_bg (entropy), C_obj (density)    │
  │ 4  │ Illumination Complexity      │ C_light = E[Var(G_i)]              │
  │ 5  │ Body Shape Diversity         │ D_shape (log-det β cov)            │
  │ 6  │ Appearance Diversity         │ D_face (mean cosine distance)      │
  │ 7  │ Garment Texture Diversity    │ D_garment (log-det CLIP cov)       │
  │UCI │ Unified Complexity Index     │ Weighted z-score sum               │
  └────┴──────────────────────────────┴────────────────────────────────────┘

Each dataloader provides batches of:
    { "person": Tensor(B,3,H,W), "cloth": Tensor(B,3,H,W), ... }

Metric modules only see the tensors they need:
  M1, M2, M3, M4, M5, M6  → person_imgs
  M7                        → cloth_imgs

Usage
------
  # Evaluate all 10 datasets via YAML config:
  python compute_pretrained_metrics.py --config configs/datasets.yaml

  # Evaluate a single dataset:
  python compute_pretrained_metrics.py \
      --dataset viton --root /data/VITON \
      --batch_size 16 --device cuda

  # Dry-run smoke-test (synthetic tensors, no real dataset):
  python compute_pretrained_metrics.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Add workspace root to path ───────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))   # workspace root

# Thin shim → delegates entirely to datasets/loaders.py (the only source of truth)
from pretrained_metrics.dataloader import get_dataloader, ALL_DATASETS
from pretrained_metrics.metrics.m1_pose           import PoseMetrics
from pretrained_metrics.metrics.m2_occlusion      import OcclusionMetrics
from pretrained_metrics.metrics.m3_background     import BackgroundMetrics
from pretrained_metrics.metrics.m4_illumination   import IlluminationMetrics
from pretrained_metrics.metrics.m5_body_shape     import BodyShapeMetrics
from pretrained_metrics.metrics.m6_appearance     import AppearanceMetrics
from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics
from pretrained_metrics.metrics.unified_index     import UnifiedComplexityIndex

# Configuration management
try:
    import config
except ImportError:
    config = None


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation for a single dataset
# ─────────────────────────────────────────────────────────────────────────────

def _all_nan():
    return float("nan")


def evaluate_one_dataset(
    dataset_name: str,
    root: str,
    cfg: dict,
) -> Dict:
    """
    Run all pretrained-metric modules on a single dataset.
    Returns a flat dict of metric values.
    """
    device     = cfg.get("device", "cpu")
    batch_size = cfg.get("batch_size", 16)
    num_workers= cfg.get("num_workers", 4)
    img_size   = tuple(cfg.get("img_size", [512, 384]))
    split      = cfg.get("split", "test")
    use_anish  = cfg.get("use_anish", False)

    # Selective metric flags
    run_pose    = cfg.get("run_pose", True)
    run_occ     = cfg.get("run_occ", True)
    run_bg      = cfg.get("run_bg", True)
    run_illum   = cfg.get("run_illum", True)
    run_shape   = cfg.get("run_shape", True)
    run_appear  = cfg.get("run_appear", True)
    run_garment = cfg.get("run_garment", True)

    # Resolve root if not provided
    if not root and config:
        root = config.get_root(dataset_name)

    # Use dedicated anish version if flag is set
    if use_anish and not dataset_name.endswith("_anish"):
        if dataset_name.lower() in ["dresscode", "vitonhd", "laion"]:
            dataset_name = f"{dataset_name.lower()}_anish"

    print(f"\n{'='*65}")
    print(f"  Dataset : {dataset_name.upper()}")
    print(f"  Root    : {root}")
    print(f"  Batch   : {batch_size}  |  Device : {device}  |  Split : {split}")
    print(f"{'='*65}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    try:
        loader = get_dataloader(
            dataset_name, root,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            **({} if "dresscode" not in dataset_name.lower()
               else {"category": cfg.get("dresscode_category", "upper_body")}),
        )
        n_samples = len(loader.dataset)
    except (FileNotFoundError, RuntimeError, Exception) as e:
        print(f"  [SKIP] {e}")
        import traceback
        traceback.print_exc()
        return {}

    print(f"  Samples : {n_samples}")

    # ── Metric objects ────────────────────────────────────────────────────────
    m1 = PoseMetrics(device=device)           if run_pose    else None
    m2 = OcclusionMetrics(device=device)      if run_occ     else None
    m3 = BackgroundMetrics(device=device)     if run_bg      else None
    m4 = IlluminationMetrics()                if run_illum   else None
    m5 = BodyShapeMetrics(device=device)      if run_shape   else None
    m6 = AppearanceMetrics(device=device)     if run_appear  else None
    m7 = GarmentTextureMetrics(device=device) if run_garment else None

    # ── Batch loop ────────────────────────────────────────────────────────────
    t0 = time.time()
    for batch in tqdm(loader, desc=f"  {dataset_name}", unit="batch"):
        person = batch["person"].float()   # (B,3,H,W)
        cloth  = batch["cloth"].float()    # (B,3,H,W)

        if m1: m1.update(person)
        if m2: m2.update(person)
        if m3: m3.update(person)
        if m4: m4.update(person)
        if m5: m5.update(person)
        if m6: m6.update(person)
        if m7: m7.update(cloth)

    elapsed = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    r1 = m1.compute() if m1 else {}
    r2 = m2.compute() if m2 else {}
    r3 = m3.compute() if m3 else {}
    r4 = m4.compute() if m4 else {}
    r5 = m5.compute() if m5 else {}
    r6 = m6.compute() if m6 else {}
    r7 = m7.compute() if m7 else {}

    dresscode_cat = cfg.get("dresscode_category") if "dresscode" in dataset_name.lower() else None
    result = {
        "dataset":   dataset_name,
        **({"dresscode_category": dresscode_cat} if dresscode_cat else {}),
        "n_samples": n_samples,
        "elapsed_s": round(elapsed, 2),
        **r1, **r2, **r3, **r4, **r5, **r6, **r7,
    }

    _print_result_box(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_KEYS = [
    # M1
    ("pose_diversity",              "Pose Diversity (log-det)"),
    ("pose_artic_complexity",       "Pose Artic. Complexity"),
    ("pose_artic_mean_per_image",   "Pose Artic. Mean/img"),
    # M2
    ("occlusion_mean",              "Occlusion Mean"),
    ("occlusion_var",               "Occlusion Var"),
    ("occlusion_complexity",        "Occlusion Complexity"),
    # M3
    ("bg_entropy_mean",             "BG Entropy Mean"),
    ("bg_entropy_var",              "BG Entropy Var"),
    ("bg_object_density_mean",      "BG Object Density"),
    # M4
    ("luminance_mean_global",       "Luminance Mean"),
    ("luminance_var_global",        "Luminance Var"),
    ("illumination_gradient_mean",  "Illum. Gradient Mean"),
    ("illumination_complexity",     "Illum. Complexity"),
    # M5
    ("shape_diversity_logdet",      "Shape Diversity (log-det)"),
    ("shape_variance_total",        "Shape Variance Total"),
    # M6
    ("appearance_diversity_mean",   "Appearance Diversity Mean"),
    ("appearance_diversity_std",    "Appearance Diversity Std"),
    # M7
    ("garment_diversity_logdet",    "Garment Diversity (log-det)"),
    ("garment_variance_total",      "Garment Variance Total"),
]


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.4f}"


def _print_result_box(r: dict):
    W = 62
    name = r["dataset"].upper()
    if "dresscode_category" in r:
        name = f"{name} [{r['dresscode_category']}]"
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │{'Results — ' + name:^{W}}│")
    print(f"  ├{'─'*W}┤")
    for key, label in DISPLAY_KEYS:
        if key in r:
            print(f"  │  {label:<38} {_fmt(r[key]):>20}  │")
    print(f"  └{'─'*W}┘")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run smoke-test (no real dataset needed)
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(device: str = "cpu"):
    """Validates the pipeline using random image tensors."""
    print("\n" + "=" * 60)
    print("  DRY RUN — Synthetic Tensors (No Dataset Required)")
    print("=" * 60)

    B, H, W = 4, 256, 192
    person = torch.rand(B, 3, H, W)
    cloth  = torch.rand(B, 3, H, W)

    metrics = [
        ("Pose",       PoseMetrics(device=device)),
        ("Occlusion",  OcclusionMetrics(device=device)),
        ("Background", BackgroundMetrics(device=device)),
        ("Illumination", IlluminationMetrics()),
        ("BodyShape",  BodyShapeMetrics(device=device)),
        ("Appearance", AppearanceMetrics(device=device)),
        ("GarmentTex", GarmentTextureMetrics(device=device)),
    ]

    N_ITER = 3
    for name, m in metrics:
        print(f"\n  ── {name} ──")
        for _ in range(N_ITER):
            if hasattr(m, "update"):
                if name == "GarmentTex":
                    m.update(cloth)
                else:
                    m.update(person)
        res = m.compute()
        for k, v in res.items():
            print(f"    {k:<40} {_fmt(v) if isinstance(v, float) else v}")

    print("\n  ✓ Dry run complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_key(name: str, cfg: dict) -> str:
    """Unique string key for a dataset entry (includes dresscode category)."""
    cat = cfg.get("dresscode_category", "")
    return f"{name}__{cat}" if cat else name


def _load_checkpoint(output_dir: str) -> dict:
    """
    Load existing checkpoint from output_dir/checkpoint.json.
    Returns a dict mapping checkpoint_key → result dict.
    """
    path = Path(output_dir) / "checkpoint.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        # data is {key: result_dict}
        print(f"  [Resume] Loaded {len(data)} completed dataset(s) from {path}")
        return data
    except Exception as e:
        print(f"  [Resume] Could not read checkpoint ({e}). Starting fresh.")
        return {}


def _write_checkpoint(checkpoint: dict, output_dir: str):
    """Persist the checkpoint dict to disk (overwrite)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def _save(all_results: List[dict], output_dir: str, uci_scores: List[dict]):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Main metrics
    json_path = out / f"pretrained_metrics_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({"metrics": all_results, "unified_scores": uci_scores}, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # CSV
    try:
        import pandas as pd
        rows = []
        for i, r in enumerate(all_results):
            row = {k: v for k, v in r.items()}
            if i < len(uci_scores):
                row["unified_score"] = uci_scores[i].get("unified_score", float("nan"))
            rows.append(row)
        df = pd.DataFrame(rows)
        csv = out / f"pretrained_metrics_{ts}.csv"
        df.to_csv(csv, index=False)
        print(f"  Results saved → {csv}")
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Pretrained-Model Dataset Complexity Pipeline (7 metrics, 10 datasets)"
    )
    p.add_argument("--config",    type=str, default=None,
                   help="YAML config (multi-dataset mode).")
    p.add_argument("--dataset",   type=str, default=None,
                   help=f"Single dataset name. One of: {ALL_DATASETS}")
    p.add_argument("--root",      type=str, default=None,
                   help="Dataset root directory.")
    p.add_argument("--dry_run",   action="store_true",
                   help="Smoke-test with random tensors (no dataset needed).")
    p.add_argument("--output_dir",type=str, default="./results_pretrained")
    p.add_argument("--batch_size",type=int, default=16)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--img_size",  type=int, nargs=2, default=[512, 384], metavar=("H","W"))
    p.add_argument("--split",     type=str, default="test")
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dresscode_category", type=str, default="upper_body")
    p.add_argument("--use_anish", action="store_true", help="Use dedicated dataloaders from dataloaders_anish/")
    p.add_argument("--no_resume",  action="store_true",
                   help="Ignore existing checkpoint and re-evaluate all datasets.")
    # Selective metrics
    p.add_argument("--no_pose",    action="store_true")
    p.add_argument("--no_occ",     action="store_true")
    p.add_argument("--no_bg",      action="store_true")
    p.add_argument("--no_illum",   action="store_true")
    p.add_argument("--no_shape",   action="store_true")
    p.add_argument("--no_appear",  action="store_true")
    p.add_argument("--no_garment", action="store_true")
    return p.parse_args()


def main():
    args = _parse()

    if args.dry_run:
        dry_run(args.device)
        return

    base_cfg = dict(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        split=args.split,
        dresscode_category=args.dresscode_category,
        use_anish=args.use_anish,
        run_pose=    not args.no_pose,
        run_occ=     not args.no_occ,
        run_bg=      not args.no_bg,
        run_illum=   not args.no_illum,
        run_shape=   not args.no_shape,
        run_appear=  not args.no_appear,
        run_garment= not args.no_garment,
    )

    output_dir = args.output_dir

    # ── Checkpoint (resume) ───────────────────────────────────────────────────
    resume = not args.no_resume
    checkpoint: dict = _load_checkpoint(output_dir) if resume else {}
    all_results: List[dict] = list(checkpoint.values())   # seed with already-done

    # ── Multi-dataset YAML ────────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        defaults = {**base_cfg, **raw.get("defaults", {})}
        for entry in raw.get("datasets", []):
            cfg  = {**defaults, **entry}
            name = cfg.pop("name")
            root = cfg.pop("root")
            cfg.pop("pred_dir", None)

            key = _checkpoint_key(name, cfg)
            if key in checkpoint:
                print(f"\n  [Skip] {name} (already in checkpoint — use --no_resume to recompute)")
                continue

            res = evaluate_one_dataset(name, root, cfg)
            if res:
                all_results.append(res)
                checkpoint[key] = res
                _write_checkpoint(checkpoint, output_dir)   # save after every dataset
                print(f"  [Checkpoint] Saved progress ({len(checkpoint)} dataset(s) done).")

    # ── Single dataset ────────────────────────────────────────────────────────
    elif args.dataset and args.root:
        key = _checkpoint_key(args.dataset, base_cfg)
        if key in checkpoint and resume:
            print(f"\n  [Skip] {args.dataset} already in checkpoint. Use --no_resume to recompute.")
        else:
            res = evaluate_one_dataset(args.dataset, args.root, base_cfg)
            if res:
                all_results.append(res)
                checkpoint[key] = res
                _write_checkpoint(checkpoint, output_dir)

    else:
        print("[INFO] No action specified. Use --dry_run, --dataset+--root, or --config.")
        print(f"[INFO] Available datasets: {ALL_DATASETS}")
        return

    if not all_results:
        print("\n  No results to save.")
        return

    # ── Unified Complexity Index ───────────────────────────────────────────────
    uci = UnifiedComplexityIndex()
    for r in all_results:
        uci.add_dataset(r["dataset"], r)
    uci_scores = uci.compute_scores()
    uci.print_report(uci_scores)

    # ── Final consolidated save ───────────────────────────────────────────────
    _save(all_results, output_dir, uci_scores)


if __name__ == "__main__":
    main()
