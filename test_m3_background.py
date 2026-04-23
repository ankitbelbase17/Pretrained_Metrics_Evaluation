from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _probe_m3(device: str, batches: List[Dict[str, torch.Tensor]], _args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m3_background import BackgroundMetrics

    obj = BackgroundMetrics(device=device)
    seg_backend = getattr(obj._segmenter, "_backend", "unknown") if hasattr(obj, "_segmenter") else "unknown"
    det_backend = getattr(obj._detector, "_backend", "unknown") if hasattr(obj, "_detector") else "unknown"
    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, f"seg={seg_backend},det={det_backend}", time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 Background metric test across all configured datasets")
    add_common_args(parser)
    args = parser.parse_args()
    return run_metric_on_all_datasets(
        args=args,
        metric_title="M3 Background",
        probe_fn=_probe_m3,
        paper_score_key="bg_overall_complexity",
        paper_score_label="Background Overall Complexity",
        set_home_for_hmr2=False,
    )


if __name__ == "__main__":
    sys.exit(main())
