"""
metrics/unified_index.py
=========================
Final Unified Dataset Complexity Index
----------------------------------------
Normalises all 7 metric families and combines them into a single score
that is **always positive** and **higher = more complex / diverse**.

Pipeline (per metric k):
    1.  z_k  = (M_k − μ_k) / σ_k              (z-score, removes raw scale)
    2.  s_k  = sigmoid(z_k / τ)                (maps to (0, 1), always > 0)
    3.  Final = 100 × Σ_k  w_k × s_k           (weighted average × 100)

The sigmoid squashes every metric to the same (0, 1) range, so a metric with
a raw magnitude of -800 doesn't overshadow one with magnitude 0.1.

Temperature τ controls the sigmoid's sensitivity:
    • τ = 1  ⇒ standard sigmoid, z-scores map ~linearly near 0
    • τ > 1  ⇒ smoother / more forgiving (larger range maps to ~0.5)
    • τ < 1  ⇒ sharper / more discriminating

Baseline statistics (μ, σ) default to approximate VITON-HD values.

Usage
------
    from metrics.unified_index import UnifiedComplexityIndex
    uci = UnifiedComplexityIndex()
    uci.add_dataset("my_dataset", all_metrics_dict)
    report = uci.compute_scores()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Metric name → friendly label  (all ↑ = higher raw value is better)
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

_MK_SET = {k for k, _ in METRIC_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
# VITON-HD approximate baselines  (μ)  and spread  (σ)
# ─────────────────────────────────────────────────────────────────────────────
# These let us z-score each metric *before* the sigmoid, so that a score of
# 0.5 corresponds roughly to "VITON-HD level".
#
# σ is set to a meaningful fraction of |μ| so that ±1σ represents a real diff.
# For metrics whose baseline μ is near 0, σ is set to a reasonable absolute
# value so that the z-score doesn't explode.
# ─────────────────────────────────────────────────────────────────────────────

VITON_HD_BASELINES = {
    "pose_diversity":            -15.0,
    "pose_artic_complexity":       0.3,
    "occlusion_complexity":        0.15,
    "illumination_complexity":     0.05,
    "shape_diversity_logdet":    -30.0,
    "appearance_diversity_mean":   0.4,
    "garment_diversity_logdet":  -20.0,
}

# σ: choose max(20% of |μ|, sensible_floor) so near-zero baselines work too
_FLOOR = {
    "pose_artic_complexity":     0.15,
    "occlusion_complexity":      0.10,
    "illumination_complexity":   0.05,
    "appearance_diversity_mean": 0.20,
}

VITON_HD_STDS = {
    k: max(abs(v) * 0.2, _FLOOR.get(k, 1e-6))
    for k, v in VITON_HD_BASELINES.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


def _isnan(v) -> bool:
    try:
        return v != v
    except Exception:
        return True


def _f(v, signed: bool = True) -> str:
    """Format a float for display."""
    if _isnan(v):
        return "N/A"
    if signed:
        return f"{v:+.4f}"
    return f"{v:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedComplexityIndex
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedComplexityIndex:
    """
    Produces a unified complexity score in (0, 100) — always positive,
    higher = more complex / diverse.

    Uses **adaptive per-metric sigmoid temperature** so that every metric
    retains discriminative information regardless of its z-score scale.

    For each metric k the temperature τ_k is set so that the most extreme
    z-score across all evaluated datasets maps to sigmoid(±TARGET), where
    TARGET ≈ 2.5  →  sigmoid range ≈ [0.08, 0.92].

    This means:
        • Metrics with huge z-scores (e.g.  garment -197)  get a large τ,
          keeping them in sigmoid's informative middle region.
        • Metrics with small z-scores (e.g.  occlusion -0.5) get a small τ,
          so the sigmoid can still distinguish between datasets.

    Parameters
    ----------
    baseline_stats : dict or None
        If None, use built-in VITON-HD baselines.
        Otherwise supply {"metric_key": (mean, std), ...}.
    weights : dict or None
        Per-metric weights λ.  Equal (1/K) by default.
    sigmoid_target : float
        The sigmoid input magnitude that the most extreme z-score for each
        metric is mapped to.  Default 2.5 → sigmoid range ≈ [0.08, 0.92].
        Larger → tighter range around 0.5; smaller → more polarised.
    """

    # The sigmoid input at which we want the most extreme z-score to land.
    # sigmoid(2.5)  ≈ 0.924
    # sigmoid(-2.5) ≈ 0.076
    # This keeps ALL metrics in the informative [0.08, 0.92] band.
    _DEFAULT_TARGET = 2.5

    def __init__(
        self,
        baseline_stats: Optional[Dict[str, tuple]] = None,
        weights: Optional[Dict[str, float]] = None,
        sigmoid_target: float = 2.5,
    ):
        if baseline_stats is None:
            self._mu  = dict(VITON_HD_BASELINES)
            self._sig = dict(VITON_HD_STDS)
        else:
            self._mu  = {k: v[0] for k, v in baseline_stats.items()}
            self._sig = {k: max(v[1], 1e-6) for k, v in baseline_stats.items()}

        n = len(METRIC_KEYS)
        self._w = weights or {k: 1.0 / n for k, _ in METRIC_KEYS}
        self._target = max(sigmoid_target, 0.5)    # clamp to sensible floor

        self._records: List[Dict] = []

    # ------------------------------------------------------------------ #
    def add_dataset(self, name: str, metrics: Dict[str, float]):
        """
        metrics : flat dict containing keys produced by m1–m7 compute().
        Unknown keys are kept (for detailed sub-metric output).
        """
        self._records.append({"dataset": name, **metrics})

    # ------------------------------------------------------------------ #
    def compute_scores(self) -> List[Dict]:
        """
        Two-pass scoring:

        Pass 1 — Compute z-scores for every (dataset, metric) pair and
                 determine the per-metric adaptive temperature τ_k.

        Pass 2 — Apply sigmoid(z / τ_k) and combine into the final score.

        Returns a list of dicts, one per dataset, each containing:
            dataset              : str
            unified_score        : float   in (0, 100)
            normalised_metrics   : dict    sigmoid scores per metric (0–1)
            z_scores             : dict    raw z-scores (for reference)
            temperatures         : dict    per-metric τ_k used
            raw_metrics          : dict    every key that was passed in
        """

        # ── Pass 1: z-scores + per-metric max |z| ────────────────────────
        all_z: List[Dict[str, float]] = []         # one dict per record
        max_abs_z: Dict[str, float] = {mk: 0.0 for mk, _ in METRIC_KEYS}

        for rec in self._records:
            z_dict = {}
            for mk, _ in METRIC_KEYS:
                val = rec.get(mk, float("nan"))
                mu  = self._mu.get(mk, 0.0)
                sig = self._sig.get(mk, 1.0)

                if _isnan(val):
                    z_dict[mk] = float("nan")
                else:
                    z = (val - mu) / sig
                    z_dict[mk] = z
                    max_abs_z[mk] = max(max_abs_z[mk], abs(z))

            all_z.append(z_dict)

        # ── Per-metric adaptive temperature ──────────────────────────────
        # τ_k = max(|z_k|) / TARGET
        # Floor at 1.0 so well-behaved metrics (small z) aren't over-amplified.
        tau: Dict[str, float] = {}
        for mk, _ in METRIC_KEYS:
            maz = max_abs_z[mk]
            if maz < 1e-9:
                # All values were identical or missing → default τ
                tau[mk] = 1.0
            else:
                tau[mk] = max(maz / self._target, 1.0)

        # ── Pass 2: sigmoid scores + final combination ───────────────────
        out = []
        for rec, z_dict in zip(self._records, all_z):
            pre_norm = rec.get("category_metrics_normalized_0_1", {})
            sig_scores = {}
            for mk, _ in METRIC_KEYS:
                if isinstance(pre_norm, dict) and mk in pre_norm and not _isnan(pre_norm[mk]):
                    # Reuse precomputed per-category normalized score when provided.
                    sig_scores[mk] = float(pre_norm[mk])
                else:
                    z = z_dict[mk]
                    if _isnan(z):
                        sig_scores[mk] = float("nan")
                    else:
                        sig_scores[mk] = _sigmoid(z / tau[mk])

            # Weighted average of sigmoid scores (only valid ones)
            valid_pairs = [
                (sig_scores[mk], self._w.get(mk, 0.0))
                for mk, _ in METRIC_KEYS
                if not _isnan(sig_scores[mk])
            ]
            if valid_pairs:
                sv, wv = zip(*valid_pairs)
                total_w = sum(wv)
                # Re-normalise weights so missing metrics don't deflate score
                final_score = (
                    sum(s * w for s, w in zip(sv, wv)) / max(total_w, 1e-12)
                ) * 100.0            # scale to (0, 100)
            else:
                final_score = float("nan")

            entry = {
                "dataset":            rec["dataset"],
                "unified_score":      final_score,
                "normalised_metrics": sig_scores,
                "z_scores":           z_dict,
                "temperatures":       dict(tau),
                "raw_metrics":        {
                    k: v for k, v in rec.items()
                    if k not in {"dataset", "dresscode_category"}
                },
            }
            if "dresscode_category" in rec:
                entry["dresscode_category"] = rec["dresscode_category"]
            out.append(entry)

        return out

    # ------------------------------------------------------------------ #
    def get_comprehensive_results(self, scores: List[Dict]) -> Dict:
        """
        Return a comprehensive dict with raw, normalized metrics, and
        all normalization parameters — suitable for JSON serialization.

        Structure:
            normalization_parameters : baselines (μ, σ), temperatures (τ), weights
            metric_definitions      : ordered list of metric keys and labels
            datasets                : per-dataset raw + normalized + sub-metrics
        """
        temps = scores[0].get("temperatures", {}) if scores else {}

        results = {
            "normalization_parameters": {
                "description": (
                    "VITON-HD baselines used for z-score normalization. "
                    "z = (raw - mu) / sigma, then sigmoid(z / tau) maps to (0,1). "
                    "Final score = 100 * weighted_avg(sigmoid_scores)."
                ),
                "baselines_mu": {k: self._mu.get(k, 0.0) for k, _ in METRIC_KEYS},
                "baselines_sigma": {k: self._sig.get(k, 1.0) for k, _ in METRIC_KEYS},
                "temperatures_tau": dict(temps),
                "sigmoid_target": self._target,
                "weights": {k: self._w.get(k, 0.0) for k, _ in METRIC_KEYS},
            },
            "metric_definitions": [
                {"key": mk, "label": label} for mk, label in METRIC_KEYS
            ],
            "datasets": [],
        }

        for entry in scores:
            ds_entry = {
                "dataset": entry["dataset"],
                "unified_complexity_score": entry["unified_score"],
                "per_metric": {},
                "all_raw_sub_metrics": {
                    k: (v if not _isnan(v) else None)
                    for k, v in entry["raw_metrics"].items()
                    if k not in _MK_SET and k != "n_samples"
                },
                "n_samples": entry["raw_metrics"].get("n_samples", 0),
            }
            if "dresscode_category" in entry:
                ds_entry["dresscode_category"] = entry["dresscode_category"]

            for mk, label in METRIC_KEYS:
                raw  = entry["raw_metrics"].get(mk, float("nan"))
                z    = entry["z_scores"].get(mk, float("nan"))
                norm = entry["normalised_metrics"].get(mk, float("nan"))

                ds_entry["per_metric"][mk] = {
                    "label": label,
                    "raw_value": raw if not _isnan(raw) else None,
                    "z_score": z if not _isnan(z) else None,
                    "normalized_0_1": norm if not _isnan(norm) else None,
                    "baseline_mu": self._mu.get(mk, 0.0),
                    "baseline_sigma": self._sig.get(mk, 1.0),
                    "temperature_tau": temps.get(mk, 1.0),
                }

            results["datasets"].append(ds_entry)

        return results

    # ------------------------------------------------------------------ #
    def print_report(self, scores: List[Dict]):
        """Pretty-print the unified complexity report with normalization details."""
        W = 105
        print("\n" + "═" * W)
        print(f"  {'UNIFIED DATASET COMPLEXITY INDEX':^{W-4}}")
        print("═" * W)

        # ── Normalization baselines ────────────────────────────────────────
        temps = scores[0].get("temperatures", {}) if scores else {}
        print(f"\n  Normalization Baselines (VITON-HD reference):")
        print(f"    {'Metric':<35} {'μ (baseline)':>14} {'σ (scale)':>12} {'τ (temp)':>12}")
        print(f"    {'─'*35} {'─'*14} {'─'*12} {'─'*12}")
        for mk, label in METRIC_KEYS:
            mu  = self._mu.get(mk, 0.0)
            sig = self._sig.get(mk, 1.0)
            tau = temps.get(mk, 1.0)
            print(f"    {label:<35} {mu:>14.4f} {sig:>12.4f} {tau:>12.4f}")

        # ── Per-dataset results ────────────────────────────────────────────
        for d in scores:
            ds_label = d['dataset'].upper()
            if 'dresscode_category' in d:
                ds_label = f"{ds_label} [{d['dresscode_category']}]"
            print(f"\n  ► {ds_label}")
            print(f"    {'Metric':<35} {'Raw (unnorm.)':>14}  {'z-score':>10}  {'τ':>8}  {'Score₀₋₁':>10}")
            print(f"    {'─'*35} {'─'*14}  {'─'*10}  {'─'*8}  {'─'*10}")

            for mk, label in METRIC_KEYS:
                raw = d["raw_metrics"].get(mk, float("nan"))
                z   = d["z_scores"].get(mk, float("nan"))
                s   = d["normalised_metrics"].get(mk, float("nan"))
                tau = temps.get(mk, 1.0)
                print(
                    f"    {label:<35} {_f(raw):>14}  {_f(z):>10}  {tau:>8.2f}  {_f(s, signed=False):>10}"
                )

            print(f"    {'─'*85}")
            score_str = _f(d['unified_score'], signed=False)
            print(f"    {'OVERALL COMPLEXITY SCORE (0–100)':<35} {'':>14}  {'':>10}  {'':>8}  {score_str:>10}")

            # ── Detailed sub-metrics ──────────────────────────────────────
            other_keys = sorted(
                k for k in d["raw_metrics"]
                if k not in _MK_SET and k != "n_samples"
            )
            if other_keys:
                print(f"\n    {'[Detailed Sub-Metrics]'}")
                for ok in other_keys:
                    raw_val = d["raw_metrics"][ok]
                    try:
                        print(f"      {ok:<40} {_f(float(raw_val)):>12}")
                    except (ValueError, TypeError):
                        print(f"      {ok:<40} {str(raw_val):>12}")

        print("\n" + "═" * W)
