import matplotlib.pyplot as plt
import seaborn as sns

def apply_conference_style():
    """
    Apply professional, academic conference style to all matplotlib/seaborn plots.
    Designed to match ECCV / NeurIPS visual standards.
    """
    # Use seaborn's whitegrid as a clean base
    sns.set_theme(style="whitegrid", context="paper")

    # Matplotlib rcParams for academic style
    plt.rcParams.update({
        # Fonts (Times New Roman is standard for CV conferences)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "text.usetex": False,  # Keep False to avoid hard LaTeX dependency
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,

        # Figure aesthetics
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "figure.autolayout": True,
        
        # Axes aesthetics
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "grid.color": "#CCCCCC",
        
        # Ticks
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        
        # Legend
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        
        # Lines and markers
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "scatter.edgecolors": "white",
        
        # Savefig
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
    })
    
    # Professional, colorblind-friendly color palette
    colors = sns.color_palette("colorblind")
    sns.set_palette(colors)
