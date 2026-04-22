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
    setup_caches,
)


def _try_object_detector(device: str) -> str:
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor

        _ = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        _ = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            use_safetensors=True,
        ).to(device).eval()
        return "object_detector=detr"
    except Exception:
        pass

    from ultralytics import YOLO

    _ = YOLO("yolov8n.pt")
    return "object_detector=yolo"


def try_mask2former(device: str) -> str:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    _ = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    _ = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-coco-panoptic",
        use_safetensors=True,
    ).to(device).eval()
    return "segmentation_backend=mask2former"


def try_segformer(device: str) -> str:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    _ = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    _ = SegformerForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        use_safetensors=True,
    ).to(device).eval()
    det = _try_object_detector(device)
    return f"segmentation_backend=segformer, {det}"


def try_deeplab(device: str) -> str:
    import torchvision.models.segmentation as seg_models

    _ = seg_models.deeplabv3_resnet101(weights=seg_models.DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()
    det = _try_object_detector(device)
    return f"segmentation_backend=deeplabv3_skin, {det}"


def main():
    args = parse_common_args("M2 fallback debug (Mask2Former -> SegFormer -> DeepLabV3)")
    print_env(args.device)
    setup_caches(args.download_base)

    attempts = [
        ("Mask2Former (primary)", lambda: try_mask2former(args.device)),
        ("SegFormer + object detector (fallback #1)", lambda: try_segformer(args.device)),
        ("DeepLabV3 + object detector (fallback #2)", lambda: try_deeplab(args.device)),
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
