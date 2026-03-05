"""
EDA/plots/__init__.py
=====================
Export all EDA plot functions.
"""

from .p1_pose_eda import plot_pose_umap, plot_joint_angle_distributions
from .p2_occlusion_eda import plot_occlusion_histogram, plot_occlusion_heatmap
from .p3_background_eda import plot_bg_entropy_histogram, plot_entropy_vs_objects
from .p4_illumination_eda import plot_luminance_spectrum, plot_illumination_pca
from .p5_body_shape_eda import plot_shape_pca, plot_shape_coefficient_histograms
from .p6_appearance_eda import plot_face_umap, plot_pairwise_distance_distribution
from .p7_garment_eda import plot_garment_umap, plot_eigenvalue_spectrum
from .p11_clip_embedding_eda import run_clip_embedding_eda

__all__ = [
    # Pose EDA
    "plot_pose_umap",
    "plot_joint_angle_distributions",
    # Occlusion EDA
    "plot_occlusion_histogram",
    "plot_occlusion_heatmap",
    # Background EDA
    "plot_bg_entropy_histogram",
    "plot_entropy_vs_objects",
    # Illumination EDA
    "plot_luminance_spectrum",
    "plot_illumination_pca",
    # Body Shape EDA
    "plot_shape_pca",
    "plot_shape_coefficient_histograms",
    # Appearance EDA
    "plot_face_umap",
    "plot_pairwise_distance_distribution",
    # Garment EDA
    "plot_garment_umap",
    "plot_eigenvalue_spectrum",
    # CLIP Embedding EDA
    "run_clip_embedding_eda",
]
