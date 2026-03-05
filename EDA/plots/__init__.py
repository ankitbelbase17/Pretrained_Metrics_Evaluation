"""
EDA/plots/__init__.py
=====================
Export all EDA plot functions.
"""

from .p1_pose_eda import run_pose_eda
from .p2_occlusion_eda import run_occlusion_eda
from .p3_background_eda import run_background_eda
from .p4_illumination_eda import run_illumination_eda
from .p5_body_shape_eda import run_body_shape_eda
from .p6_appearance_eda import run_appearance_eda
from .p7_garment_eda import run_garment_eda
from .p8_meta_correlation import run_meta_correlation_eda
from .p10_camera_angle_eda import run_camera_angle_eda

__all__ = [
    "run_pose_eda",
    "run_occlusion_eda",
    "run_background_eda",
    "run_illumination_eda",
    "run_body_shape_eda",
    "run_appearance_eda",
    "run_garment_eda",
    "run_meta_correlation_eda",
    "run_camera_angle_eda",
]
