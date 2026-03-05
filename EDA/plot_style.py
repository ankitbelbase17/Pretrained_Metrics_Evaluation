"""
EDA/plot_style.py
==================
ECCV Publication-Quality matplotlib/seaborn configuration.
Import this module first in any plotting script.

Designed for:
  - ECCV / CVPR / ICCV / NeurIPS conference papers
  - Single-column (3.25in) and double-column (6.875in) figures
  - Professional typography using Computer Modern (LaTeX-compatible)
  - Accessible, colorblind-friendly palette

Provides:
  - apply_paper_style()  → sets RC params for ECCV figures
  - PALETTE              → colour palette (colorblind-friendly, up to 10)
  - DATASET_COLORS       → dict mapping dataset name → hex colour
  - DATASET_MARKERS      → dict mapping dataset name → marker style
  - save_fig()           → saves PDF + high-DPI PNG side-by-side
  - add_stat_box()       → adds μ ± σ text box to an Axes
  - add_legend_outside() → creates professional legend outside plot
  - create_figure()      → helper to create ECCV-sized figures
  - FIG_SINGLE, FIG_DOUBLE, FIG_FULL → standard ECCV figure dimensions
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════════════════
# ECCV Standard Figure Dimensions (inches)
# ═══════════════════════════════════════════════════════════════════════════════
FIG_SINGLE = (3.25, 2.5)      # Single-column width
FIG_DOUBLE = (6.875, 2.75)    # Double-column width
FIG_FULL   = (6.875, 5.0)     # Full-page width, taller
FIG_SQUARE = (3.25, 3.25)     # Square for UMAP/scatter


# ═══════════════════════════════════════════════════════════════════════════════
# ECCV/CVPR Publication Palette
# - Maximally distinct colors for overlapping plots
# - High contrast for both color and grayscale printing
# - Colorblind-friendly (tested with Coblis simulator)
# - Based on ColorBrewer and Wong's colorblind-safe palette
# ═══════════════════════════════════════════════════════════════════════════════

# Primary palette: maximally distinct, high-saturation colors
# Ordered by perceptual distinctiveness (most distinct first)
# Based on ColorBrewer + Wong's colorblind-safe palette
PALETTE = [
    "#0077BB",  # strong blue   (viton_hd)    - primary, highest contrast
    "#EE7733",  # orange        (dresscode)   - warm, distinct from blue
    "#009988",  # teal          (street_tryon)- green-blue, unique hue
    "#CC3311",  # vermillion    (alternative) - red, high visibility
    "#33BBEE",  # cyan          (alternative) - light blue, distinct
    "#EE3377",  # magenta       (alternative) - pink-red
    "#BBBBBB",  # gray          (baseline)    - neutral reference
    "#AA4499",  # purple        (alternative) - violet range
    "#44BB99",  # mint          (alternative) - light green
    "#DDCC77",  # sand/gold     (alternative) - warm yellow
]

# ═══════════════════════════════════════════════════════════════════════════════
# CURVTON Difficulty Colors — Traffic Light Scheme (maximally distinct)
# Green → Yellow/Orange → Red progression for intuitive difficulty understanding
# ═══════════════════════════════════════════════════════════════════════════════
CURVTON_COLORS = {
    "Easy":   "#1B9E77",  # Dark teal-green (distinct from pure green)
    "Medium": "#D95F02",  # Dark orange (distinct from yellow/red)  
    "Hard":   "#7570B3",  # Muted purple-blue (NOT red - avoids confusion with orange)
}

# Alternative CURVTON scheme using maximally distinct hues
CURVTON_COLORS_ALT = {
    "Easy":   "#66C2A5",  # Soft teal
    "Medium": "#FC8D62",  # Coral orange
    "Hard":   "#8DA0CB",  # Periwinkle blue
}

# Dataset-specific colors (maximally distinct for the 3 main datasets)
DATASET_COLORS: Dict[str, str] = {
    # Primary datasets (most distinct colors)
    "viton_hd":       "#0077BB",  # Strong blue
    "dresscode":      "#EE7733",  # Orange
    "street_tryon":   "#009988",  # Teal
    # DressCode categories (variations of orange)
    "dresscode_upper_body": "#EE7733",  # Orange
    "dresscode_lower_body": "#CC3311",  # Vermillion
    "dresscode_dresses":    "#EE3377",  # Magenta
    # Legacy/additional datasets
    "viton":          "#33BBEE",  # Cyan
    "mpv":            "#AA4499",  # Purple  
    "deepfashion_tryon": "#44BB99",  # Mint
    "acgpn":          "#DDCC77",  # Sand
    "cp_vton":        "#882255",  # Wine
    "hr_vton":        "#117733",  # Forest green
    "ladi_vton":      "#88CCEE",  # Light blue
    "ovnet":          "#CC6677",  # Rose
    # ═══ CURVTON Dataset (difficulty-based) — MAXIMALLY DISTINCT ═══
    # Using ColorBrewer Dark2 palette for maximum separation
    "curvton_easy":   "#1B9E77",  # Dark teal-green
    "curvton_medium": "#D95F02",  # Dark orange
    "curvton_hard":   "#7570B3",  # Muted purple (NOT red - avoids orange confusion)
    "curvton_all":    "#E7298A",  # Magenta pink - combined
    # Display names (for legend)
    "Easy":           "#1B9E77",
    "Medium":         "#D95F02",
    "Hard":           "#7570B3",
    # CURVTON gender splits (lighter variants)
    "curvton_easy_female":   "#1B9E77",
    "curvton_easy_male":     "#66C2A5",
    "curvton_medium_female": "#D95F02",
    "curvton_medium_male":   "#FC8D62",
    "curvton_hard_female":   "#7570B3",
    "curvton_hard_male":     "#8DA0CB",
}

# Distinct markers for each dataset (essential for B&W printing)
DATASET_MARKERS: Dict[str, str] = {
    "viton_hd":       "o",   # circle
    "dresscode":      "s",   # square
    "street_tryon":   "^",   # triangle up
    "dresscode_upper_body": "s",
    "dresscode_lower_body": "D",   # diamond
    "dresscode_dresses":    "v",   # triangle down
    "viton":          "o",
    "mpv":            "p",   # pentagon
    "deepfashion_tryon": "h",  # hexagon
    "acgpn":          "*",   # star
    "cp_vton":        "X",   # X marker
    "hr_vton":        "P",   # plus filled
    "ladi_vton":      "d",   # thin diamond
    "ovnet":          ">",   # triangle right
    # ═══ CURVTON markers (distinct shapes) ═══
    "curvton_easy":   "o",   # circle
    "curvton_medium": "s",   # square
    "curvton_hard":   "^",   # triangle up
    "curvton_all":    "D",   # diamond
    "Easy":           "o",
    "Medium":         "s",
    "Hard":           "^",
}

# Line styles for overlapping line plots (use with colors)
# CRITICAL: Different line styles ensure visibility even with similar colors
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
DATASET_LINESTYLES: Dict[str, str] = {
    "viton_hd":       "-",    # solid
    "dresscode":      "--",   # dashed
    "street_tryon":   "-.",   # dash-dot
    "dresscode_upper_body": "--",
    "dresscode_lower_body": ":",
    "dresscode_dresses":    "-.",
    # ═══ CURVTON line styles (maximally distinct) ═══
    "curvton_easy":   "-",    # solid (thick)
    "curvton_medium": "--",   # dashed
    "curvton_hard":   "-.",   # dash-dot
    "curvton_all":    ":",    # dotted
    "Easy":           "-",
    "Medium":         "--",
    "Hard":           "-.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Alpha values for overlapping elements — OPTIMIZED FOR VISIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
FILL_ALPHA = 0.20        # Very subtle fill — overlaps clearly visible
LINE_ALPHA = 0.95        # Strong lines — always visible on top
SCATTER_ALPHA = 0.55     # Points semi-transparent for density
HIST_ALPHA = 0.25        # Histogram bars more transparent
KDE_LINE_WIDTH = 2.0     # Thick KDE lines for visibility
BORDER_LINE_WIDTH = 1.8  # Strong borders on filled regions

# Grayscale-friendly markers (ensures distinction when printed B&W)
PALETTE_GRAYSCALE = [
    "#000000",  # black
    "#404040",  # dark gray
    "#808080",  # medium gray
    "#B0B0B0",  # light gray
    "#D0D0D0",  # very light gray
]

# Sequential palette for heatmaps (perceptually uniform)
PALETTE_SEQUENTIAL = [
    "#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
    "#FC8D59", "#EF6548", "#D7301F", "#B30000", "#7F0000"
]

# Diverging palette for correlation matrices
PALETTE_DIVERGING = "RdBu_r"  # Red-Blue diverging (standard for correlations)

DATASET_ORDER = [
    "viton_hd", "dresscode", "street_tryon",  # Primary (most used)
    "viton", "mpv", "deepfashion_tryon",
    "acgpn", "cp_vton", "hr_vton", "ladi_vton", "ovnet",
]

MARKER_STYLES = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Overlapping KDE Plot with Maximum Visibility
# ═══════════════════════════════════════════════════════════════════════════════

def plot_overlapping_kde(
    ax,
    datasets: Dict[str, np.ndarray],
    xlabel: str = "",
    ylabel: str = "Density",
    title: str = "",
    show_mean: bool = True,
    show_legend: bool = True,
):
    """
    Plot overlapping KDE distributions with maximum visibility.
    
    Uses:
    - Distinct colors from DATASET_COLORS
    - Different line styles for each dataset
    - Very low fill alpha (0.15-0.25) for overlaps
    - Strong border lines (2.0+ width)
    - Drawing order: most samples first (background)
    
    Parameters
    ----------
    ax : matplotlib Axes
    datasets : dict
        {dataset_name: np.ndarray of values}
    """
    import seaborn as sns
    
    # Sort by sample count (largest first = background)
    sorted_datasets = sorted(datasets.items(), key=lambda x: len(x[1]), reverse=True)
    
    legend_handles = []
    
    for i, (name, values) in enumerate(sorted_datasets):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if len(values) < 5:
            continue
        
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        linestyle = DATASET_LINESTYLES.get(name, LINE_STYLES[i % len(LINE_STYLES)])
        marker = DATASET_MARKERS.get(name, MARKER_STYLES[i % len(MARKER_STYLES)])
        
        # Variable fill alpha based on order (first = most transparent)
        fill_alpha = FILL_ALPHA + (i * 0.02)  # Slight increase for later datasets
        fill_alpha = min(fill_alpha, 0.30)    # Cap at 0.30
        
        # KDE with subtle fill, strong line
        sns.kdeplot(
            values, ax=ax,
            fill=True, alpha=fill_alpha,
            linewidth=KDE_LINE_WIDTH + (i * 0.1),  # Slightly thicker for later
            color=color,
            linestyle=linestyle,
        )
        
        # Mean indicator line
        if show_mean:
            ax.axvline(
                values.mean(), color=color,
                linestyle=":", linewidth=1.5, alpha=0.85,
            )
        
        # Legend handle with marker + line
        legend_handles.append(
            Line2D([0], [0],
                   color=color, marker=marker,
                   markerfacecolor=color, markersize=6,
                   linewidth=KDE_LINE_WIDTH,
                   linestyle=linestyle,
                   label=f"{name} (n={len(values):,})")
        )
    
    # Style axes
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    despine_axes(ax)
    
    if show_legend and legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=8,
        )
    
    return legend_handles


def plot_overlapping_histogram(
    ax,
    datasets: Dict[str, np.ndarray],
    bins: int = 40,
    xlabel: str = "",
    ylabel: str = "Density",
    title: str = "",
):
    """
    Plot overlapping histograms with step style for maximum visibility.
    
    Uses step histograms (outlines only) + very subtle fills.
    """
    sorted_datasets = sorted(datasets.items(), key=lambda x: len(x[1]), reverse=True)
    
    legend_handles = []
    
    for i, (name, values) in enumerate(sorted_datasets):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if len(values) < 5:
            continue
        
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        linestyle = DATASET_LINESTYLES.get(name, "-")
        
        # Step histogram (outline) - strong line
        ax.hist(
            values, bins=bins, density=True, histtype="step",
            color=color, linewidth=1.8, alpha=0.95,
        )
        
        # Filled histogram - very subtle
        ax.hist(
            values, bins=bins, density=True, histtype="stepfilled",
            color=color, alpha=0.15, edgecolor="none",
        )
        
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2.0,
                   linestyle=linestyle, label=name)
        )
    
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    despine_axes(ax)
    
    return legend_handles


# ═══════════════════════════════════════════════════════════════════════════════
# RC Params for ECCV Publication
# ═══════════════════════════════════════════════════════════════════════════════

def apply_paper_style(
    font_scale: float = 1.0,
    use_latex: bool = False,
    context: str = "paper",
):
    """
    Apply ECCV/CVPR publication-ready style.
    
    Design principles:
    - Clean, minimal aesthetic (no chartjunk)
    - High information density
    - Readable at single-column width
    - Print-friendly (works in grayscale)
    
    Parameters
    ----------
    font_scale : float
        Scale factor for all fonts (default 1.0 for ECCV)
    use_latex : bool
        If True, uses LaTeX for text rendering (requires LaTeX installation)
    context : str
        One of 'paper', 'poster', 'talk' for different sizing
    """
    # Base font sizes optimized for ECCV single-column (3.25 inches)
    base_sizes = {
        "paper":  {"title": 9, "label": 8, "tick": 7, "legend": 7, "annotation": 6},
        "poster": {"title": 14, "label": 12, "tick": 11, "legend": 11, "annotation": 10},
        "talk":   {"title": 16, "label": 14, "tick": 12, "legend": 12, "annotation": 11},
    }
    sizes = base_sizes.get(context, base_sizes["paper"])
    
    # Set seaborn theme - use 'white' for cleanest academic look
    sns.set_theme(
        style="white",
        font_scale=font_scale,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    
    # Font configuration - prefer serif fonts for academic papers
    if use_latex:
        font_config = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{times}",
        }
    else:
        font_config = {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
            "mathtext.fontset": "stix",  # STIX fonts match Times well
        }
    
    matplotlib.rcParams.update(font_config)
    matplotlib.rcParams.update({
        # ── Typography (academic standard) ──────────────────────────────────
        "axes.titlesize":       sizes["title"] * font_scale,
        "axes.titleweight":     "bold",
        "axes.titlepad":        6,
        "axes.labelsize":       sizes["label"] * font_scale,
        "axes.labelweight":     "normal",
        "axes.labelpad":        3,
        "xtick.labelsize":      sizes["tick"] * font_scale,
        "ytick.labelsize":      sizes["tick"] * font_scale,
        "legend.fontsize":      sizes["legend"] * font_scale,
        "legend.title_fontsize": sizes["legend"] * font_scale,
        
        # ── Axes & Spines (minimal, clean) ──────────────────────────────────
        "axes.linewidth":       0.6,
        "axes.edgecolor":       "#000000",
        "axes.labelcolor":      "#000000",
        "xtick.color":          "#000000",
        "ytick.color":          "#000000",
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.minor.size":     1.5,
        "ytick.minor.size":     1.5,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        
        # ── Grid (subtle, non-distracting) ──────────────────────────────────
        "axes.grid":            False,  # Off by default for clean look
        "grid.linewidth":       0.3,
        "grid.alpha":           0.4,
        "grid.color":           "#CCCCCC",
        "grid.linestyle":       "-",
        
        # ── Legend (compact, unobtrusive) ───────────────────────────────────
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        "legend.fancybox":      False,  # Square corners for academic look
        "legend.borderpad":     0.3,
        "legend.labelspacing":  0.25,
        "legend.handlelength":  1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.borderaxespad": 0.3,
        
        # ── Figure ──────────────────────────────────────────────────────────
        "figure.dpi":           150,
        "figure.facecolor":     "white",
        "figure.edgecolor":     "white",
        "figure.autolayout":    False,
        "figure.constrained_layout.use": True,
        
        # ── Saving (high quality for print) ─────────────────────────────────
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.01,
        "savefig.facecolor":    "white",
        "savefig.edgecolor":    "none",
        "savefig.format":       "pdf",
        "savefig.transparent":  False,
        "pdf.fonttype":         42,   # TrueType (editable in Illustrator)
        "ps.fonttype":          42,
        
        # ── Colors ──────────────────────────────────────────────────────────
        "axes.prop_cycle":      matplotlib.cycler(color=PALETTE),
        "image.cmap":           "viridis",
        
        # ── Lines & Markers (visible but not heavy) ─────────────────────────
        "lines.linewidth":      1.0,
        "lines.markersize":     4,
        "lines.markeredgewidth": 0.5,
        "scatter.edgecolors":   "none",
        
        # ── Histograms & Patches ────────────────────────────────────────────
        "patch.linewidth":      0.5,
        "patch.edgecolor":      "#000000",
        "hist.bins":            "auto",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Figure Creation Helper
# ═══════════════════════════════════════════════════════════════════════════════

def create_figure(
    size: Union[str, Tuple[float, float]] = "double",
    nrows: int = 1,
    ncols: int = 1,
    height_ratios: Optional[List[float]] = None,
    width_ratios: Optional[List[float]] = None,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create an ECCV-sized figure with proper dimensions.
    
    Parameters
    ----------
    size : str or tuple
        'single', 'double', 'full', 'square', or custom (width, height) tuple
    nrows, ncols : int
        Number of subplot rows and columns
    height_ratios, width_ratios : list
        GridSpec ratios for rows/columns
        
    Returns
    -------
    fig, axes : Figure and array of Axes
    """
    size_map = {
        "single": FIG_SINGLE,
        "double": FIG_DOUBLE,
        "full":   FIG_FULL,
        "square": FIG_SQUARE,
    }
    
    figsize = size_map.get(size, size) if isinstance(size, str) else size
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        gridspec_kw={
            "height_ratios": height_ratios,
            "width_ratios": width_ratios,
        } if height_ratios or width_ratios else None,
        **kwargs
    )
    
    return fig, np.atleast_2d(axes) if nrows > 1 or ncols > 1 else axes


# ═══════════════════════════════════════════════════════════════════════════════
# Save Helper (PDF + PNG for ECCV submission)
# ═══════════════════════════════════════════════════════════════════════════════

def save_fig(
    fig: plt.Figure, 
    out_dir: Path, 
    stem: str,
    formats: Tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
):
    """
    Save figure as both PDF (vector) and PNG (raster) for ECCV submission.
    
    PDF: Primary format for paper submission (vector, fonts embedded)
    PNG: Backup for compatibility issues (600 DPI for print quality)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(
            p,
            format=ext,
            dpi=dpi if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            edgecolor="none",
        )
        print(f"  ✓ Saved → {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Overlapping Plot Utilities (KDE, Histograms)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_overlapping_kde(
    ax: plt.Axes,
    datasets: Dict[str, np.ndarray],
    xlabel: str = "",
    show_mean: bool = True,
    show_legend: bool = True,
):
    """
    Plot overlapping KDE distributions with maximum visibility.
    
    Key design choices for visibility:
    - Low fill alpha (0.2) so overlaps remain visible
    - Strong line borders (linewidth 1.5) for each distribution
    - Distinct colors from DATASET_COLORS
    - Optional dashed mean lines per dataset
    
    Parameters
    ----------
    ax : matplotlib Axes
    datasets : dict
        {dataset_name: values_array}
    xlabel : str
        X-axis label
    show_mean : bool
        If True, adds vertical dashed line at each dataset's mean
    show_legend : bool
        If True, adds legend
    """
    legend_handles = []
    
    for i, (name, values) in enumerate(datasets.items()):
        values = np.asarray(values).flatten()
        values = values[np.isfinite(values)]
        if len(values) < 5:
            continue
        
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        linestyle = DATASET_LINESTYLES.get(name, "-")
        marker = DATASET_MARKERS.get(name, "o")
        
        # KDE with low fill alpha, strong border
        sns.kdeplot(
            values, ax=ax,
            fill=True, 
            alpha=FILL_ALPHA,        # Low alpha for overlap visibility
            linewidth=1.5,           # Strong border line
            color=color,
            linestyle=linestyle,
        )
        
        # Mean indicator line
        if show_mean:
            mean_val = values.mean()
            ax.axvline(
                mean_val, 
                color=color, 
                linestyle="--", 
                linewidth=1.2, 
                alpha=0.85,
                zorder=10,
            )
        
        # Legend handle with both marker and line
        legend_handles.append(
            Line2D([0], [0], 
                   marker=marker, 
                   color=color,
                   markerfacecolor=color, 
                   markersize=7,
                   linewidth=1.8, 
                   linestyle=linestyle,
                   label=f"{name} (μ={values.mean():.2f})")
        )
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    
    if show_legend and legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=8,
        )
    
    return legend_handles


def plot_overlapping_histogram(
    ax: plt.Axes,
    datasets: Dict[str, np.ndarray],
    bins: int = 40,
    xlabel: str = "",
    show_mean: bool = True,
):
    """
    Plot overlapping histograms with step style for clarity.
    
    Uses 'step' histtype so all distributions remain visible.
    """
    legend_handles = []
    
    for i, (name, values) in enumerate(datasets.items()):
        values = np.asarray(values).flatten()
        values = values[np.isfinite(values)]
        if len(values) < 5:
            continue
        
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        linestyle = DATASET_LINESTYLES.get(name, "-")
        marker = DATASET_MARKERS.get(name, "o")
        
        # Step histogram (unfilled, just borders)
        ax.hist(
            values, bins=bins, 
            histtype="step",
            density=True,
            color=color,
            linewidth=1.5,
            linestyle=linestyle,
            alpha=LINE_ALPHA,
        )
        
        # Light fill for context
        ax.hist(
            values, bins=bins,
            histtype="stepfilled",
            density=True,
            color=color,
            alpha=FILL_ALPHA * 0.6,  # Even lighter fill
            linewidth=0,
        )
        
        # Mean line
        if show_mean:
            ax.axvline(
                values.mean(),
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
        
        legend_handles.append(
            Line2D([0], [0], 
                   marker=marker,
                   color=color,
                   markerfacecolor=color,
                   markersize=7,
                   linewidth=1.8,
                   linestyle=linestyle,
                   label=f"{name} (μ={values.mean():.2f})")
        )
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    
    return legend_handles


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Annotation Box (Publication Style)
# ═══════════════════════════════════════════════════════════════════════════════

def add_stat_box(
    ax: plt.Axes, 
    values: np.ndarray, 
    x: float = 0.97, 
    y: float = 0.95,
    fontsize: int = 8,
    show_n: bool = False,
):
    """
    Add elegant μ ± σ statistics box in the corner of axes.
    
    Parameters
    ----------
    ax : matplotlib Axes
    values : array-like
        Data to compute statistics from
    x, y : float
        Position in axes coordinates (0-1)
    fontsize : int
        Font size for annotation
    show_n : bool
        If True, also displays sample size n
    """
    values = np.asarray(values).flatten()
    values = values[np.isfinite(values)]
    
    if len(values) == 0:
        return
    
    mu  = np.nanmean(values)
    sig = np.nanstd(values)
    med = np.nanmedian(values)
    
    if show_n:
        text = f"n = {len(values):,}\nμ = {mu:.3f}\nσ = {sig:.3f}"
    else:
        text = f"μ = {mu:.3f}\nσ = {sig:.3f}"
    
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4,rounding_size=0.2",
            facecolor="white",
            edgecolor="#888888",
            alpha=0.92,
            linewidth=0.6,
        ),
        zorder=100,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Legend Utilities for Multi-Dataset Plots
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataset_legend(
    ax: plt.Axes,
    dataset_names: List[str],
    loc: str = "upper right",
    ncol: int = 1,
    title: str = "Dataset",
    outside: bool = False,
    marker_scale: float = 1.5,
):
    """
    Create a consistent, publication-quality legend for dataset comparisons.
    
    Parameters
    ----------
    ax : matplotlib Axes
    dataset_names : list of str
        Names of datasets to include in legend
    loc : str
        Legend location (see matplotlib docs)
    ncol : int
        Number of legend columns
    title : str
        Legend title
    outside : bool
        If True, places legend outside the axes
    marker_scale : float
        Scale factor for legend markers
    """
    handles = []
    for name in dataset_names:
        color = DATASET_COLORS.get(name, PALETTE[len(handles) % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        handle = Line2D(
            [0], [0],
            marker=marker,
            color="white",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8 * marker_scale,
            linewidth=0,
            label=name,
        )
        handles.append(handle)
    
    if outside:
        legend = ax.legend(
            handles=handles,
            title=title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.95,
            edgecolor="#cccccc",
            ncol=ncol,
        )
    else:
        legend = ax.legend(
            handles=handles,
            title=title,
            loc=loc,
            framealpha=0.95,
            edgecolor="#cccccc",
            ncol=ncol,
        )
    
    legend.get_title().set_fontweight("bold")
    return legend


def add_subplot_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.12,
    y: float = 1.08,
    fontsize: int = 12,
):
    """
    Add panel label (a), (b), (c), etc. for multi-panel figures.
    
    Standard practice for ECCV/CVPR papers.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="right",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Color Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def get_cmap_for_heatmap(style: str = "diverging") -> str:
    """
    Get appropriate colormap for different heatmap types.
    
    Parameters
    ----------
    style : str
        'diverging' for correlation matrices (RdBu_r)
        'sequential' for frequency/density (viridis)
        'hot' for occlusion/attention maps
    """
    cmap_map = {
        "diverging":  "RdBu_r",
        "sequential": "viridis", 
        "hot":        "inferno",
        "cool":       "cividis",
    }
    return cmap_map.get(style, "viridis")


def lighten_color(color: str, amount: float = 0.3) -> str:
    """
    Lighten a color by a given amount for fills/backgrounds.
    """
    import colorsys
    import matplotlib.colors as mcolors
    
    try:
        c = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        l = min(1, l + amount)
        rgb = colorsys.hls_to_rgb(h, l, s)
        return mcolors.to_hex(rgb)
    except:
        return color


# ═══════════════════════════════════════════════════════════════════════════════
# Grid & Layout Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def despine_axes(ax: plt.Axes, keep_left: bool = True, keep_bottom: bool = True):
    """
    Remove top and right spines for cleaner ECCV-style plots.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)


def format_axis_labels(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlabel_style: dict = None,
    ylabel_style: dict = None,
    title_style: dict = None,
):
    """
    Apply consistent formatting to axis labels and title.
    """
    default_label_style = {"fontsize": 9, "fontweight": "medium"}
    default_title_style = {"fontsize": 10, "fontweight": "bold", "pad": 8}
    
    if xlabel:
        style = {**default_label_style, **(xlabel_style or {})}
        ax.set_xlabel(xlabel, **style)
    if ylabel:
        style = {**default_label_style, **(ylabel_style or {})}
        ax.set_ylabel(ylabel, **style)
    if title:
        style = {**default_title_style, **(title_style or {})}
        ax.set_title(title, **style)
