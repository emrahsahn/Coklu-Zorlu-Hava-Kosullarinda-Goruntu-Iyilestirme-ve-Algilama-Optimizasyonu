"""
Unified restoration + detection pipeline package.
"""

from .metrics import (
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    compare_detections,
    generate_report,
    summarize_pipeline_result,
)
from .model_loader import (
    DDNWrapper,
    FFANetWrapper,
    HVI_CIDNetWrapper,
    default_device,
    repo_root,
)
from .pipeline import PipelineResult, RestorationMode, UnifiedPipeline
from .weather_detector import WeatherDetector, WeatherResult, condition_to_mode
from .yolo_detector import YoloV8Detector, detections_to_json

__all__ = [
    "DDNWrapper",
    "FFANetWrapper",
    "HVI_CIDNetWrapper",
    "YoloV8Detector",
    "UnifiedPipeline",
    "PipelineResult",
    "RestorationMode",
    "WeatherDetector",
    "WeatherResult",
    "condition_to_mode",
    "detections_to_json",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_mae",
    "compare_detections",
    "summarize_pipeline_result",
    "generate_report",
    "default_device",
    "repo_root",
]
