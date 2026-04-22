"""
evaluate.py
============
Main entry-point for the Virtual Try-On Evaluation Pipeline.

Metrics computed
----------------
  ┌──────────────────────────────┬───────────────────────────────────────────────┐
  │ Metric                       │ Description                                   │
  ├──────────────────────────────┼───────────────────────────────────────────────┤
  │ PSNR                         │ Peak Signal-to-Noise Ratio                    │
  │ SSIM                         │ Structural Similarity Index                   │
  │ Masked SSIM                  │ SSIM computed only over the garment mask      │
  │ LPIPS                        │ Learned Perceptual Image Patch Similarity     │
  │ FID                          │ Fréchet Inception Distance                    │
  │ IS (mean ± std)              │ Inception Score                               │
  │ KID (mean ± std)             │ Kernel Inception Distance                     │
    │ Pose Error (PE)              │ MPJPE in pixels (MMPose HRNet keypoints)      │
  │ VLM Score                    │ BLIP-2 plausibility score 1-10                │
  │ JEPA EPE                     │ Embedding Prediction Error (MSE)              │
  │ JEPA Trace Σ                 │ Tr(Cov) of target embeddings                  │
  └──────────────────────────────┴───────────────────────────────────────────────┘

Supported datasets (--dataset flag)
------------------------------------
  viton | viton_hd | dresscode | mpv | deepfashion_tryon |
  acgpn | cp_vton  | hr_vton   | ladi_vton | ovnet

Usage
-----
  python evaluate.py \
      --dataset viton \
      --root /path/to/dataset \
      --pred_dir /path/to/tryon_outputs \
      --output_dir ./results \
      --batch_size 8 \
      --device cuda \
      [--no_vlm] [--no_jepa] [--no_pose]

  # Evaluate ALL 10 datasets in one go (requires --config):
  python evaluate.py --config configs/all_datasets.yaml

Note on --pred_dir
------------------
If your model is loaded inline (not saved to disk), pass --pred_dir "" and
subclass / monkey-patch the `_run_model` function below with your own forward pass.
By default the script loads PNG/JPG images from pred_dir whose filenames match
the sample IDs from the dataloader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from datasets.loaders import get_dataset, DATASET_REGISTRY, _STANDALONE_NAMES

# Standalone dataloader canonical collates (for dataloaders/ package)
try:
    from dataloaders import (
        canonical_collate_dresscode,
        canonical_collate_vitonhd,
        canonical_collate_street_tryon,
        canonical_collate_laion,
    )
    _STANDALONE_COLLATES = {
        "dresscode_standalone":    canonical_collate_dresscode,
        "vitonhd_standalone":      canonical_collate_vitonhd,
        "street_tryon_standalone": canonical_collate_street_tryon,
        "laion_standalone":        canonical_collate_laion,
    }
except ImportError:
    _STANDALONE_COLLATES = {}

from metrics.image_metrics import (
    compute_psnr_batch,
    compute_ssim_batch,
    compute_masked_ssim_batch,
    LPIPSMetric,
)
from metrics.distribution_metrics import DistributionMetrics
from metrics.pose_error import PoseErrorMetric
from metrics.vlm_score import VLMScoreMetric
from metrics.jepa_metrics import JEPAMetrics

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a predicted image from disk (if pred_dir supplied)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pred_from_dir(
    ids: List[str],
    pred_dir: Path,
    img_size: Tuple[int, int],
    transform,
) -> Optional[torch.Tensor]:
    """
    Try to load images named <id>.jpg or <id>.png from pred_dir.
    Returns (B, C, H, W) float tensor or None if no pred_dir / files missing.
    """
    if pred_dir is None:
        return None
    imgs = []
    for img_id in ids:
        stem = Path(img_id).stem
        found = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = pred_dir / (stem + ext)
            if p.exists():
                found = p
                break
        if found is None:
            return None       # At least one missing → signal failure
        img = Image.open(found).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs)


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder: swap this for your model's forward pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_model(cloth: torch.Tensor, person: torch.Tensor) -> torch.Tensor:
    """
    Default stub — returns the person image as a no-op prediction.
    REPLACE with your actual model forward pass, e.g.:
        return model(person, cloth)
    """
    return person.clone()


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop for one dataset
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    dataset_root: str,
    pred_dir: Optional[str],
    cfg: dict,
) -> Dict:
    """
    Run the full metric suite on a single dataset.

    Returns a dict of aggregated metrics.
    """
    device    = cfg.get("device", "cpu")
    batch_sz  = cfg.get("batch_size", 8)
    num_work  = cfg.get("num_workers", 4)
    img_size  = tuple(cfg.get("img_size", [512, 384]))
    split     = cfg.get("split", "test")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    dataset_kwargs = {}
    if dataset_name == "dresscode":
        dataset_kwargs["category"] = cfg.get("dresscode_category", "upper_body")

    print(f"\n{'='*60}")
    print(f" Dataset : {dataset_name.upper()}")
    print(f" Root    : {dataset_root}")
    print(f" Split   : {split}  |  Batch: {batch_sz}  |  Device: {device}")
    print(f"{'='*60}")

    try:
        dataset = get_dataset(
            dataset_name, dataset_root,
            split=split,
            img_size=img_size,
            **dataset_kwargs,
        )
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return {}

    if len(dataset) == 0:
        print(f"[SKIP] Dataset is empty.")
        return {}

    transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    pred_path = Path(pred_dir) if pred_dir else None

    # Use canonical collate for standalone dataloaders to ensure consistent
    # output format {person, cloth, gt, mask, meta}
    ds_name_norm = dataset_name.lower().replace("-", "_")
    collate_fn = _STANDALONE_COLLATES.get(ds_name_norm, None)

    loader = DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=num_work,
        pin_memory=("cuda" in device),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ── Metric objects ────────────────────────────────────────────────────────
    lpips_metric = LPIPSMetric(device=device)
    dist_metric  = DistributionMetrics(device=device)
    pose_metric  = PoseErrorMetric(device=device)      if cfg.get("compute_pose", True) else None
    vlm_metric   = VLMScoreMetric(device=device)       if cfg.get("compute_vlm",  True) else None
    jepa_metric  = JEPAMetrics(device=device)          if cfg.get("compute_jepa", True) else None

    # Accumulators
    acc: Dict[str, List[float]] = {
        k: [] for k in [
            "psnr", "ssim", "masked_ssim", "lpips",
            "pose_error",
            # VLM sub-scores (S1-S4) + weighted composite
            "vlm_s1", "vlm_s2", "vlm_s3", "vlm_s4", "vlm_score",
            "jepa_epe",
        ]
    }

    t0 = time.time()
    for batch in tqdm(loader, desc=dataset_name, unit="batch"):
        cloth  = batch["cloth"]   # (B, 3, H, W) or None
        person = batch["person"]  # (B, 3, H, W)
        gt     = batch["gt"]      # (B, 3, H, W)
        mask   = batch["mask"]    # (B, 1, H, W)
        ids    = [m["id"] for m in batch["meta"]]

        # Skip if any element is None (missing file)
        if cloth is None or person is None or gt is None:
            continue

        cloth  = cloth.float()
        person = person.float()
        gt     = gt.float()
        mask   = mask.float()

        # ── Obtain model prediction ──────────────────────────────────────────
        pred = _load_pred_from_dir(ids, pred_path, img_size, transform)
        if pred is None:
            # Fall back to calling the stub / your model
            with torch.no_grad():
                pred = _run_model(cloth.to(device), person.to(device)).cpu()

        pred = pred.float().clamp(0, 1)

        # ── Per-image metrics ────────────────────────────────────────────────
        acc["psnr"].extend(compute_psnr_batch(pred, gt))
        acc["ssim"].extend(compute_ssim_batch(pred, gt))
        acc["masked_ssim"].extend(compute_masked_ssim_batch(pred, gt, mask))
        acc["lpips"].extend(lpips_metric.compute_batch(pred, gt))

        if pose_metric:
            acc["pose_error"].extend(pose_metric.compute_batch(pred, gt))

        if vlm_metric:
            vlm_results = vlm_metric.compute_batch(pred)   # list[dict]
            for entry in vlm_results:
                acc["vlm_s1"].append(entry["s1"])
                acc["vlm_s2"].append(entry["s2"])
                acc["vlm_s3"].append(entry["s3"])
                acc["vlm_s4"].append(entry["s4"])
                acc["vlm_score"].append(entry["vlm_score"])

        if jepa_metric:
            acc["jepa_epe"].extend(jepa_metric.compute_epe_batch(person, pred))
            jepa_metric.update_embeddings(pred)

        # ── Distributional (accumulate) ──────────────────────────────────────
        dist_metric.update(pred, gt)

    # ── Dataset-level ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  Computing FID / IS / KID …")
    try:
        dist_results = dist_metric.compute()
    except Exception as e:
        print(f"  [WARN] FID/IS/KID failed: {e}")
        dist_results = {"fid": float("nan"), "is_mean": float("nan"),
                        "is_std": float("nan"), "kid_mean": float("nan"),
                        "kid_std": float("nan")}
    dist_metric.cleanup()

    jepa_trace = float("nan")
    if jepa_metric and acc["jepa_epe"]:
        jepa_trace = jepa_metric.compute_embedding_trace()
        jepa_metric.reset()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _mean(lst):
        lst = [v for v in lst if not math.isnan(v)]
        return float(np.mean(lst)) if lst else float("nan")

    results = {
        "dataset":        dataset_name,
        "n_samples":      len(dataset),
        "elapsed_s":      round(elapsed, 2),
        # ── pixel-level
        "psnr":           _mean(acc["psnr"]),
        "ssim":           _mean(acc["ssim"]),
        "masked_ssim":    _mean(acc["masked_ssim"]),
        "lpips":          _mean(acc["lpips"]),
        # ── distributional
        "fid":            dist_results["fid"],
        "is_mean":        dist_results["is_mean"],
        "is_std":         dist_results["is_std"],
        "kid_mean":       dist_results["kid_mean"],
        "kid_std":        dist_results["kid_std"],
        # ── structural / semantic
        "pose_error_px":  _mean(acc["pose_error"]),
        # VLM sub-scores (↑ higher is better, scale 1–10)
        "vlm_s1_garment_fidelity":      _mean(acc["vlm_s1"]),
        "vlm_s2_geometric_naturalness": _mean(acc["vlm_s2"]),
        "vlm_s3_identity_preservation": _mean(acc["vlm_s3"]),
        "vlm_s4_scene_coherence":       _mean(acc["vlm_s4"]),
        "vlm_score":                    _mean(acc["vlm_score"]),
        # ── JEPA
        "jepa_epe":       _mean(acc["jepa_epe"]),
        "jepa_trace_cov": jepa_trace,
    }

    _print_results_table(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, float):
        return "N/A" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def _print_results_table(r: dict):
    rows = [
        ("PSNR (↑)",                        r["psnr"]),
        ("SSIM (↑)",                        r["ssim"]),
        ("Masked SSIM (↑)",                 r["masked_ssim"]),
        ("LPIPS (↓)",                       r["lpips"]),
        ("FID (↓)",                         r["fid"]),
        ("IS mean (↑)",                     r["is_mean"]),
        ("IS std",                          r["is_std"]),
        ("KID mean (↓)",                    r["kid_mean"]),
        ("KID std",                         r["kid_std"]),
        ("Pose Error px (↓)",               r["pose_error_px"]),
        ("VLM S1 Garment Fidelity (↑)",     r["vlm_s1_garment_fidelity"]),
        ("VLM S2 Geometric Natural. (↑)",   r["vlm_s2_geometric_naturalness"]),
        ("VLM S3 Identity Preserv. (↑)",    r["vlm_s3_identity_preservation"]),
        ("VLM S4 Scene Coherence (↑)",      r["vlm_s4_scene_coherence"]),
        ("VLM Score (weighted, ↑)",         r["vlm_score"]),
        ("JEPA EPE (↓)",                    r["jepa_epe"]),
        ("JEPA Tr(Sigma) (↑)",              r["jepa_trace_cov"]),
    ]
    width = 38
    print(f"\n  ┌{'─'*width}┐")
    print(f"  │{'Results — ' + r['dataset'].upper():^{width}}│")
    print(f"  ├{'─'*width}┤")
    for name, val in rows:
        print(f"  │  {name:<22} {_fmt(val):>12}  │")
    print(f"  └{'─'*width}┘")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Virtual Try-On Evaluation Pipeline (11 metrics, 10 datasets)"
    )
    # ── Single dataset mode ───────────────────────────────────────────────────
    p.add_argument("--dataset", type=str, default=None,
                   help=f"Dataset name. One of: {list(DATASET_REGISTRY)}")
    p.add_argument("--root",    type=str, default=None,
                   help="Path to dataset root directory.")
    p.add_argument("--pred_dir", type=str, default=None,
                   help="Directory with predicted try-on PNG/JPGs (filename = sample ID).")
    # ── Multi-dataset (config) mode ───────────────────────────────────────────
    p.add_argument("--config",  type=str, default=None,
                   help="YAML config file for evaluating multiple datasets.")
    # ── Shared options ────────────────────────────────────────────────────────
    p.add_argument("--output_dir",  type=str,   default="./results",
                   help="Directory to save JSON / CSV results.")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--img_size",    type=int,   nargs=2, default=[512, 384],
                   metavar=("H", "W"))
    p.add_argument("--split",       type=str,   default="test")
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dresscode_category", type=str, default="upper_body",
                   help="DressCode category: upper_body | lower_body | dresses")
    # ── Selective metric flags ────────────────────────────────────────────────
    p.add_argument("--no_vlm",   action="store_true", help="Skip VLM scoring.")
    p.add_argument("--no_jepa",  action="store_true", help="Skip JEPA metrics.")
    p.add_argument("--no_pose",  action="store_true", help="Skip pose error.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-dataset YAML config format
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_CONFIG = """
# configs/all_datasets.yaml
# --------------------------
# List of dataset entries. Each entry must have 'name' and 'root'.
# All shared options (device, batch_size, …) can be overridden per dataset.

defaults:
  device: cuda
  batch_size: 8
  num_workers: 4
  img_size: [512, 384]
  split: test
  compute_pose: true
  compute_vlm: true
  compute_jepa: true

datasets:
  - name: viton
    root: /data/VITON/
    pred_dir: /results/viton/

  - name: viton_hd
    root: /data/VITON-HD/
    pred_dir: /results/viton_hd/

  - name: dresscode
    root: /data/DressCode/
    pred_dir: /results/dresscode/
    dresscode_category: upper_body

  - name: mpv
    root: /data/MPV/
    pred_dir: /results/mpv/

  - name: deepfashion_tryon
    root: /data/DeepFashion-TryOn/
    pred_dir: /results/deepfashion_tryon/

  - name: acgpn
    root: /data/ACGPN/
    pred_dir: /results/acgpn/

  - name: cp_vton
    root: /data/CP-VTON/
    pred_dir: /results/cp_vton/

  - name: hr_vton
    root: /data/HR-VTON/
    pred_dir: /results/hr_vton/

  - name: ladi_vton
    root: /data/LaDI-VTON/
    pred_dir: /results/ladi_vton/

  - name: ovnet
    root: /data/OVNet/
    pred_dir: /results/ovnet/
"""


def _load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(all_results: List[dict], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out / f"metrics_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV / Excel (per-dataset row)
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = out / f"metrics_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Results saved → {json_path}")
        print(f"  Results saved → {csv_path}")
    else:
        print(f"\n  Results saved → {json_path}")

    # Pretty summary table
    if all_results:
        _print_summary_table(all_results)


def _print_summary_table(all_results: List[dict]):
    cols = [
        "dataset", "psnr", "ssim", "masked_ssim", "lpips",
        "fid", "is_mean", "kid_mean", "pose_error_px",
        "vlm_s1_garment_fidelity",
        "vlm_s2_geometric_naturalness",
        "vlm_s3_identity_preservation",
        "vlm_s4_scene_coherence",
        "vlm_score",
        "jepa_epe", "jepa_trace_cov",
    ]
    # Only keep columns that exist in at least one result dict
    avail = [c for c in cols if any(c in r for r in all_results)]
    df = pd.DataFrame(all_results)[avail]
    sep = "═" * max(130, len(avail) * 14)
    print("\n" + sep)
    print("  SUMMARY — All Datasets  (VLM: S1=GarmentFidelity S2=GeomNaturalness S3=Identity S4=SceneCoherence)")
    print(sep)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    base_cfg = dict(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        split=args.split,
        dresscode_category=args.dresscode_category,
        compute_pose=not args.no_pose,
        compute_vlm=not args.no_vlm,
        compute_jepa=not args.no_jepa,
    )

    all_results: List[dict] = []

    # ── Config (multi-dataset) mode ───────────────────────────────────────────
    if args.config:
        raw = _load_yaml_config(args.config)
        defaults = raw.get("defaults", {})
        defaults = {**base_cfg, **defaults}   # CLI overrides YAML defaults

        for entry in raw.get("datasets", []):
            cfg = {**defaults, **entry}         # entry overrides defaults
            name     = cfg.pop("name")
            root     = cfg.pop("root")
            pred_dir = cfg.pop("pred_dir", None)
            res = evaluate_dataset(name, root, pred_dir, cfg)
            if res:
                all_results.append(res)

    # ── Single dataset mode ───────────────────────────────────────────────────
    elif args.dataset and args.root:
        res = evaluate_dataset(args.dataset, args.root, args.pred_dir, base_cfg)
        if res:
            all_results.append(res)

    # ── No-arg: generate sample config and exit ───────────────────────────────
    else:
        cfg_path = Path("configs") / "all_datasets.yaml"
        cfg_path.parent.mkdir(exist_ok=True)
        if not cfg_path.exists():
            cfg_path.write_text(_SAMPLE_CONFIG)
            print(f"[INFO] Sample config written to: {cfg_path}")
            print("[INFO] Edit dataset roots/pred_dirs then run:")
            print(f"       python evaluate.py --config {cfg_path}")
        else:
            print("[INFO] Run with --help for usage.")
        return

    _save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
