from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))

from fallback_debug.common import (
    first_success_index,
    parse_common_args,
    print_attempt,
    print_env,
    print_summary,
    run_attempt,
)


def try_hmr2(device: str) -> str:
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
    import torch.serialization as _ts
    from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig

    _ts.add_safe_globals([_DictConfig, _ListConfig])
    download_models()
    model, _cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    return f"camera_backend=hmr2, model={type(model).__name__}"


def try_vitpose(device: str) -> str:
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple",
        use_safetensors=True,
    ).to(device).eval()
    return f"camera_backend=vitpose, model={model.__class__.__name__}, processor={processor.__class__.__name__}"


def try_keypointrcnn(device: str) -> str:
    import torchvision

    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights).to(device).eval()
    return f"camera_backend=keypointrcnn, model={model.__class__.__name__}"


def try_dino(device: str) -> str:
    from transformers import AutoImageProcessor, AutoModel

    _ = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    return f"camera_backend=dino, model={model.__class__.__name__}"


def main():
    args = parse_common_args("M9 fallback debug (HMR2 -> ViTPose -> KeypointRCNN -> DINOv2)")
    print_env(args.device)

    attempts = [
        ("HMR2.0 (primary)", lambda: try_hmr2(args.device)),
        ("ViTPose (fallback #1)", lambda: try_vitpose(args.device)),
        ("KeypointRCNN (fallback #2)", lambda: try_keypointrcnn(args.device)),
        ("DINOv2 (fallback #3)", lambda: try_dino(args.device)),
    ]

    rows: List[Tuple[str, bool, float]] = []
    selected = None
    for i, (name, fn) in enumerate(attempts):
        ok, detail, elapsed = run_attempt(name, fn, args.verbose)
        rows.append((name, ok, elapsed))
        print_attempt(name, ok, detail, elapsed)
        if ok and selected is None:
            selected = i
            if not args.all:
                break

    print_summary(rows)
    idx = first_success_index(rows)
    if idx is not None:
        print(f"Selected backend chain step: {attempts[idx][0]}")
    else:
        print("No backend loaded. This explains full metric failure.")


if __name__ == "__main__":
    main()
