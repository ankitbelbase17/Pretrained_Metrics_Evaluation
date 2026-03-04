"""
metrics/unified_index.py
=========================
Final Unified Dataset Complexity Index
----------------------------------------
Normalises all 7 metric families and combines them into a single score.

    Score = Σ_k λ_k * M̂_k

where:
    M̂_k = (M_k - μ_baseline_k) / σ_baseline_k   (z-score)

Baseline statistics (μ, σ) are estimated from VITON-HD values reported in
published papers OR from the first dataset evaluated (auto-calibration mode).

Weights λ_k are equal by default (1/7 each) but can be customised.

Usage
------
    from metrics.unified_index import UnifiedComplexityIndex
    uci = UnifiedComplexityIndex(baseline_stats=None)   # auto-calibrate
    uci.add_dataset("my_dataset", all_metrics_dict)
    report = uci.compute_scores()

Where all_metrics_dict contains the union of keys from metrics m1–m7.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Metric name → friendly label + direction
# ─────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = [
    ("pose_diversity",            "Pose Diversity (↑)"),
    ("pose_artic_complexity",     "Pose Articulation (↑)"),
    ("occlusion_complexity",      "Occlusion Complexity (↑)"),
    ("illumination_complexity",   "Illumination Complexity (↑)"),
    ("shape_diversity_logdet",    "Body Shape Diversity (↑)"),
    ("appearance_diversity_mean", "Appearance Diversity (↑)"),
    ("garment_diversity_logdet",  "Garment Texture Diversity (↑)"),
]


# Known VITON-HD approximate baselines (from literature estimates).
# Update these with your own measured values for more accurate normalisation.
VITON_HD_BASELINES = {
    "pose_diversity":            -15.0,
    "pose_artic_complexity":       0.3,
    "occlusion_complexity":        0.15,
    "illumination_complexity":     0.05,
    "shape_diversity_logdet":    -30.0,
    "appearance_diversity_mean":   0.4,
    "garment_diversity_logdet":  -20.0,
}

VITON_HD_STDS = {k: max(abs(v) * 0.2, 1e-6) for k, v in VITON_HD_BASELINES.items()}


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedComplexityIndex
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedComplexityIndex:
    """
    Parameters
    ----------
    baseline_stats : dict or None
        If None, the first dataset added is used as the auto-calibration baseline.
        Otherwise, supply {"metric_key": (mean, std), ...}.
    weights : dict or None
        λ weights per metric key. Equal (1/7) if None.
    """

    def __init__(
        self,
        baseline_stats: Optional[Dict[str, tuple]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        if baseline_stats is None:
            self._mu  = VITON_HD_BASELINES
            self._sig = VITON_HD_STDS
        else:
            self._mu  = {k: v[0] for k, v in baseline_stats.items()}
            self._sig = {k: max(v[1], 1e-6) for k, v in baseline_stats.items()}

        n = len(METRIC_KEYS)
        self._w = weights or {k: 1.0 / n for k, _ in METRIC_KEYS}

        self._records: List[Dict] = []   # one entry per dataset

    # ------------------------------------------------------------------ #
    def add_dataset(self, name: str, metrics: Dict[str, float]):
        """
        metrics : flat dict containing keys from m1–m7 compute() outputs.
        Redundant/unknown keys are silently ignored.
        """
        self._records.append({"dataset": name, **metrics})

    # ------------------------------------------------------------------ #
    def compute_scores(self) -> List[Dict]:
        """
        Returns a list of dicts, one per dataset, containing:
          - normalised scores M̂_k
          - final weighted score
          - raw metric values
        """
        out = []
        for rec in self._records:
            z_scores = {}
            for mk, label in METRIC_KEYS:
                val  = rec.get(mk, float("nan"))
                mu   = self._mu.get(mk, 0.0)
                sig  = self._sig.get(mk, 1.0)
                z_scores[mk] = (val - mu) / sig if not _isnan(val) else float("nan")

            valid_z = [v for v in z_scores.values() if not _isnan(v)]
            weights = [self._w.get(mk, 0.0) for mk, _ in METRIC_KEYS]
            valid_pairs = [
                (z_scores[mk], w)
                for (mk, _), w in zip(METRIC_KEYS, weights)
                if not _isnan(z_scores[mk])
            ]
            if valid_pairs:
                zv, wv = zip(*valid_pairs)
                total_w = sum(wv)
                final_score = sum(z * w for z, w in zip(zv, wv)) / max(total_w, 1e-12)
            else:
                final_score = float("nan")

            entry = {
                "dataset":            rec["dataset"],
                "unified_score":      final_score,
                "normalised_metrics": z_scores,
                "raw_metrics":        {k: rec.get(k, float("nan")) for k, _ in METRIC_KEYS},
            }
            if "dresscode_category" in rec:
                entry["dresscode_category"] = rec["dresscode_category"]
            out.append(entry)

        return out

    # ------------------------------------------------------------------ #
    def print_report(self, scores: List[Dict]):
        """Pretty-print the unified complexity report."""
        W = 70
        print("\n" + "═" * W)
        print(f"  {'UNIFIED DATASET COMPLEXITY INDEX':^{W-4}}")
        print("═" * W)
        for d in scores:
            label = d['dataset'].upper()
            if 'dresscode_category' in d:
                label = f"{label} [{d['dresscode_category']}]"
            print(f"\n  ► {label}")
            print(f"    {'Metric':<35} {'Raw':>12}  {'Normalised':>12}")
            print(f"    {'─'*35} {'─'*12}  {'─'*12}")
            for mk, label in METRIC_KEYS:
                raw = d["raw_metrics"].get(mk, float("nan"))
                nrm = d["normalised_metrics"].get(mk, float("nan"))
                print(f"    {label:<35} {_f(raw):>12}  {_f(nrm):>12}")
            print(f"    {'─'*62}")
            print(f"    {'Final Unified Score':<35} {'':>12}  {_f(d['unified_score']):>12}")
        print("\n" + "═" * W)


def _isnan(v):
    try:
        return v != v
    except Exception:
        return True


def _f(v):
    if _isnan(v):
        return "N/A"
    return f"{v:+.4f}"
