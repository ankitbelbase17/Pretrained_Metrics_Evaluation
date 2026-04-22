"""
test.py
=======
Pretrained model load-status checker.

This script ONLY checks whether pretrained-model backends can be initialized.
It does not run forward passes or metric computations.

Run:
    python test.py
    python test.py --device cuda
    python test.py --skip m2 m6
    python test.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


# (name, loader_fn)
CHECKS: List[Tuple[str, Callable[[str], str]]] = []


def register(name: str):
    def _wrap(fn: Callable[[str], str]):
        CHECKS.append((name, fn))
        return fn
    return _wrap


@register("M1 Pose backend")
def check_m1(device: str) -> str:
    from pretrained_metrics.metrics.m1_pose import _KeypointExtractor
    obj = _KeypointExtractor(device)
    return f"backend={obj._backend}"


@register("M2 Occlusion segmentation backend")
def check_m2(device: str) -> str:
    from pretrained_metrics.metrics.m2_occlusion import _SegBackend
    obj = _SegBackend(device)
    return f"backend={obj._backend}"


@register("M3 Background person-segmenter")
def check_m3_person(device: str) -> str:
    from pretrained_metrics.metrics.m3_background import _PersonSegmenter
    obj = _PersonSegmenter(device)
    return f"model_loaded={obj._model is not None}"


@register("M3 Background object-detector")
def check_m3_obj(device: str) -> str:
    from pretrained_metrics.metrics.m3_background import _ObjectDetector
    obj = _ObjectDetector(device)
    return f"backend={obj._backend}"


@register("M5 Body-shape backend")
def check_m5(device: str) -> str:
    from pretrained_metrics.metrics.m5_body_shape import _ShapeExtractor
    obj = _ShapeExtractor(device)
    return f"backend={obj._backend}"


@register("M6 Appearance face-embedder backend")
def check_m6(device: str) -> str:
    from pretrained_metrics.metrics.m6_appearance import _FaceEmbedder
    obj = _FaceEmbedder(device)
    return f"backend={obj._backend}"


@register("M7 Garment encoder backend")
def check_m7(device: str) -> str:
    from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder
    obj = _GarmentEncoder(device)
    return f"backend={obj._backend}"


@register("M8 VAE encoder backend")
def check_m8(device: str) -> str:
    from pretrained_metrics.metrics.m8_vae_latent import _VAEEncoder
    obj = _VAEEncoder(device)
    return f"backend={obj._backend}"


@register("M9 Camera-angle backend")
def check_m9(device: str) -> str:
    from pretrained_metrics.metrics.m9_camera_angle import _CameraAngleBackend
    obj = _CameraAngleBackend(device)
    return f"backend={obj._backend}"


@register("VLM score backend")
def check_vlm(device: str) -> str:
    from metrics.vlm_score import VLMScoreMetric
    obj = VLMScoreMetric(device=device)
    return f"backend={obj._backend}"


def run_checks(device: str, skip: List[str], verbose: bool) -> int:
    print("\n" + "=" * 72)
    print("  Pretrained Model Load Status")
    print(f"  device={device}  |  skip={skip}")
    print("=" * 72)

    loaded: List[str] = []
    failed: List[str] = []
    loaded_info: List[Tuple[str, str]] = []
    failed_info: List[Tuple[str, str]] = []
    skipped = 0

    for name, fn in CHECKS:
        if any(s.lower() in name.lower() for s in skip):
            print(f"  {'SKIP':<12} {name}")
            skipped += 1
            continue

        t0 = time.time()
        try:
            info = fn(device)
            dt = time.time() - t0
            print(f"  {_green('LOADED'):<21} [{dt:5.1f}s]  {name}")
            print(f"           -> {info}")
            loaded.append(name)
            loaded_info.append((name, info))
        except Exception as e:
            dt = time.time() - t0
            print(f"  {_red('NOT LOADED'):<21} [{dt:5.1f}s]  {name}")
            print(f"           -> {_red(type(e).__name__)}: {e}")
            if verbose:
                traceback.print_exc()
            failed.append(name)
            failed_info.append((name, f"{type(e).__name__}: {e}"))

    print("\n" + "-" * 72)
    print(f"  {_green('Loaded')}     : {len(loaded)}")
    for n in loaded:
        print(f"    - {n}")

    print(f"  {_red('Not loaded')} : {len(failed)}")
    for n in failed:
        print(f"    - {n}")

    print(f"  {_yellow('Skipped')}    : {skipped}")
    print("-" * 72 + "\n")

    print("=" * 72)
    print("  Final Per-Metric Status")
    print("=" * 72)
    if loaded_info:
        print(f"  {_green('Loaded Metrics/Models')}")
        for name, info in loaded_info:
            print(f"    - {name}")
            print(f"      info: {info}")
    else:
        print(f"  {_green('Loaded Metrics/Models')}: none")

    if failed_info:
        print(f"  {_red('Not Loaded Metrics/Models')}")
        for name, info in failed_info:
            print(f"    - {name}")
            print(f"      reason: {info}")
    else:
        print(f"  {_red('Not Loaded Metrics/Models')}: none")
    print("=" * 72 + "\n")

    # Return non-zero if any model failed to load.
    return len(failed)


def _parse():
    p = argparse.ArgumentParser(description="Check pretrained model load status")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for backend initialization",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Substring keywords to skip checks (case-insensitive)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full traceback for failed checks",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    n_failed = run_checks(args.device, args.skip, args.verbose)
    sys.exit(0 if n_failed == 0 else 1)
