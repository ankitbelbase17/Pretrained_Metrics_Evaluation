"""
test_m9_camera_angle.py
=======================
Standalone smoke-test for M9 Camera Angle Diversity metric.

Pretrained models: HMR2.0 → ViTPose → KeypointRCNN → DINOv2

Usage
-----
python test_m9_camera_angle.py --max_batches 2
python test_m9_camera_angle.py --device cpu --max_batches 4
"""

from __future__ import annotations

import argparse, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pretrained_metrics.cache_setup import configure_model_caches, DEFAULT_MODEL_BASE

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))

def _g(s): return f"\033[92m{s}\033[0m"
def _r(s): return f"\033[91m{s}\033[0m"

def _fv(v):
    if isinstance(v, float):
        return "NA" if np.isnan(v) else f"{v:.6g}"
    if isinstance(v, (np.floating,)):
        fv = float(v); return "NA" if np.isnan(fv) else f"{fv:.6g}"
    return str(v) if v is not None else "NA"

def _to_bchw(x, name):
    if x.ndim != 4:
        raise RuntimeError(f"[{name}] Expected 4-D tensor, got {tuple(x.shape)}")
    if x.shape[1] != 3 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.shape[1] != 3:
        raise RuntimeError(f"[{name}] Expected C=3, got {tuple(x.shape)}")
    return x

def _collect_batches(args):
    from pretrained_metrics.dataloader import get_dataloader
    loader = get_dataloader(
        dataset_name=args.dataset_name, root=args.curvton_root,
        split=args.split, batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=tuple(args.img_size),
    )
    batches, n = [], 0
    for i, batch in enumerate(loader):
        person = _to_bchw(batch["person"].float(), "person")
        cloth  = _to_bchw(batch["cloth"].float(),  "cloth")
        batches.append({"person": person, "cloth": cloth})
        n += int(person.shape[0])
        if args.max_batches > 0 and (i + 1) >= args.max_batches:
            break
    if not batches:
        raise RuntimeError("No batches loaded.")
    return batches, len(batches), n

def _probe_m9(device, batches):
    print(f"\n  Loading M9 Camera Angle metric (device={device})...")
    from pretrained_metrics.metrics.m9_camera_angle import CameraAngleMetrics
    obj = CameraAngleMetrics(device=device)
    backend = getattr(obj._backend, "_backend", "unknown")
    print(f"  Backend: {backend}")

    print(f"  Running update on {sum(b['person'].shape[0] for b in batches)} images...")
    t0 = time.time()
    for b in batches:
        obj.update(b["person"])
    result = obj.compute()
    dt = time.time() - t0
    return result, backend, dt

def main():
    p = argparse.ArgumentParser(description="Smoke-test M9 Camera Angle metric")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--download_base", default=DEFAULT_MODEL_BASE)
    p.add_argument("--curvton_root", default="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test/hard")
    p.add_argument("--dataset_name", default="curvton")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_size", type=int, nargs=2, default=[512, 384])
    p.add_argument("--max_batches", type=int, default=0)
    args = p.parse_args()

    cache_info = configure_model_caches(args.download_base, set_home_for_hmr2=True)

    print("\n" + "=" * 90)
    print("  M9 Camera Angle Metric — Standalone Smoke-Test")
    print(f"  device={args.device} | batch_size={args.batch_size} | max_batches={args.max_batches}")
    print("=" * 90)

    try:
        batches, n_batches, n_images = _collect_batches(args)
        print(f"  Loaded {n_batches} batches ({n_images} images)")
    except Exception as e:
        print(f"  {_r('FAILED')} dataloader: {e}")
        if args.verbose: traceback.print_exc()
        return 1

    try:
        result, backend, dt = _probe_m9(args.device, batches)
        print(f"\n  {_g('LOADED')}  M9 Camera Angle  (backend={backend}, {dt:.1f}s)")
        print("  Computed values:")
        for k in sorted(result):
            print(f"    {k:40s} = {_fv(result[k])}")
        status = 0
    except Exception as e:
        print(f"\n  {_r('NOT LOADED')}  M9 Camera Angle")
        print(f"    error: {type(e).__name__}: {e}")
        if args.verbose: traceback.print_exc()
        status = 1

    print("\n" + "=" * 90)
    print(f"  Result: {_g('PASSED') if status == 0 else _r('FAILED')}")
    print("=" * 90 + "\n")
    return status

if __name__ == "__main__":
    sys.exit(main())
