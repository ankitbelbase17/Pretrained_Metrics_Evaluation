from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _probe_m6(device: str, batches: List[Dict[str, torch.Tensor]], args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m6_appearance import AppearanceMetrics

    obj = AppearanceMetrics(
        device=device,
        face_detector_backend=args.m6_face_detector_backend,
        retinaface_repo_dir=args.m6_retinaface_repo_dir,
        retinaface_weights=args.m6_retinaface_weights,
        retinaface_backbone=args.m6_retinaface_backbone,
        retinaface_device=args.m6_retinaface_device,
    )
    backend = getattr(getattr(obj, "_embedder", None), "_backend", "unknown")
    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, backend, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M6 Appearance metric test across all configured datasets")
    add_common_args(parser)
    parser.add_argument("--m6_face_detector_backend", type=str, default="retinaface_pytorch",
                        choices=["auto", "retinaface_pytorch", "insightface", "haar"])
    parser.add_argument("--m6_retinaface_repo_dir", type=str, default=None)
    parser.add_argument("--m6_retinaface_weights", type=str, default=None)
    parser.add_argument("--m6_retinaface_backbone", type=str, default="mobilenetv1_0.25")
    parser.add_argument("--m6_retinaface_device", type=str, default="cpu")
    args = parser.parse_args()
    return run_metric_on_all_datasets(
        args=args,
        metric_title="M6 Appearance",
        probe_fn=_probe_m6,
        paper_score_key="appearance_diversity_mean",
        paper_score_label="Appearance Diversity Mean",
        set_home_for_hmr2=False,
    )


if __name__ == "__main__":
    sys.exit(main())
