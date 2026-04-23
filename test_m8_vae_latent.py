from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _probe_m8(device: str, batches: List[Dict[str, torch.Tensor]], _args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m8_vae_latent import VAELatentMetric

    obj = VAELatentMetric(device=device)
    backend = getattr(obj._encoder, "_backend", "unknown")
    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, backend, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M8 VAE Latent metric test across all configured datasets")
    add_common_args(parser)
    args = parser.parse_args()
    return run_metric_on_all_datasets(
        args=args,
        metric_title="M8 VAE Latent",
        probe_fn=_probe_m8,
        paper_score_key="vae_diversity_logdet",
        paper_score_label="VAE Latent Diversity (logdet)",
        set_home_for_hmr2=False,
    )


if __name__ == "__main__":
    sys.exit(main())
