from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _probe_m5(device: str, batches: List[Dict[str, torch.Tensor]], _args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics

    obj = BodyShapeMetrics(device=device)
    backend = getattr(obj._extractor, "_backend", "unknown")
    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, backend, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 Body Shape metric test across all configured datasets")
    add_common_args(parser)
    args = parser.parse_args()
    return run_metric_on_all_datasets(
        args=args,
        metric_title="M5 Body Shape",
        probe_fn=_probe_m5,
        paper_score_key="shape_diversity_logdet",
        paper_score_label="Body Shape Diversity (logdet)",
        set_home_for_hmr2=True,
    )


if __name__ == "__main__":
    sys.exit(main())
