from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
from datetime import datetime
import traceback

import numpy as np


def load_npz_dict(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def find_dataset_cache(cache_root: Path, dataset: str) -> Optional[Path]:
    """
    Find a cache file for a dataset label with common naming variants.
    """
    key = normalize_label(dataset)

    # Common direct forms.
    candidates = [
        cache_root / f"{key}_features.npz",
        cache_root / key / f"{key}_features.npz",
        cache_root / key / f"{key}_all_features.npz",
    ]

    # Known aliases.
    if key in {"viton_hd", "vitonhd"}:
        candidates.extend([
            cache_root / "viton_hd_features.npz",
            cache_root / "vitonhd_features.npz",
            cache_root / "vitonhd" / "vitonhd_features.npz",
            cache_root / "viton_hd" / "viton_hd_features.npz",
        ])
    elif key == "dresscode":
        candidates.extend([
            cache_root / "dresscode" / "dresscode_all_features.npz",
            cache_root / "dresscode" / "dresscode_upper_body_features.npz",
        ])
    elif key in {"street_tryon", "streettryon"}:
        candidates.extend([
            cache_root / "street_tryon_features.npz",
            cache_root / "street_tryon" / "street_tryon_features.npz",
            cache_root / "street_tryon" / "street_tryon_all_features.npz",
        ])

    p = first_existing(candidates)
    if p is not None:
        return p

    # Broad fallback search.
    for pat in [f"**/{key}*_features.npz", f"**/*{key}*features.npz"]:
        found = sorted(cache_root.glob(pat))
        if found:
            return found[0]
    return None


def load_curvton_split_caches(curvton_cache_dir: Path, ratio_pct: int = 100) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    mapping = {
        "CurvTON-Easy": curvton_cache_dir / f"curvton_easy_{ratio_pct}pct.npz",
        "CurvTON-Medium": curvton_cache_dir / f"curvton_medium_{ratio_pct}pct.npz",
        "CurvTON-Hard": curvton_cache_dir / f"curvton_hard_{ratio_pct}pct.npz",
        "CurvTON-All": curvton_cache_dir / f"curvton_all_{ratio_pct}pct.npz",
    }
    for label, path in mapping.items():
        if path.exists():
            out[label] = load_npz_dict(path)
    return out


def log_plot_error(script_name: str, error_log: str, exc: Exception) -> None:
    """Append a detailed plotting failure record to a log file."""
    p = Path(error_log)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {script_name} FAILED\n")
        f.write(f"Exception: {type(exc).__name__}: {exc}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-" * 80 + "\n")
