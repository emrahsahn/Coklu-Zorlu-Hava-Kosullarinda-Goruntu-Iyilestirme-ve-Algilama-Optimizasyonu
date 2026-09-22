"""
Lightweight statistical weather / degradation classifier (no extra ML model).

Maps to pipeline restoration modes via ``condition_to_mode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

WeatherCondition = Literal["low_light", "haze", "rain", "clear"]


@dataclass
class WeatherResult:
    condition: WeatherCondition
    confidence: float
    stats: dict[str, float]
    suggested_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "confidence": float(self.confidence),
            "stats": self.stats,
            "suggested_mode": self.suggested_mode,
        }


def condition_to_mode(condition: WeatherCondition, *, haze_variant: Literal["its", "ots"] = "its") -> str:
    """Map weather label to ``UnifiedPipeline`` restoration mode string."""
    if condition == "low_light":
        return "low_light_hvi"
    if condition == "haze":
        return "haze_ffa_its" if haze_variant == "its" else "haze_ffa_ots"
    if condition == "rain":
        return "rain_ddn"
    return "none"


def _to_bgr_gray(image: np.ndarray | Image.Image) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb.astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _dark_channel_mean(bgr: np.ndarray, patch: int = 15) -> float:
    """Lower-level DCP statistic: mean of per-pixel min channel in local patches."""
    b, g, r = cv2.split(bgr)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(dc, kernel)
    return float(dark.mean() / 255.0)


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean() / 255.0)


def _local_contrast_var(gray: np.ndarray, ksize: int = 15) -> float:
    blur = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    local_var = cv2.GaussianBlur((gray.astype(np.float32) - blur) ** 2, (ksize, ksize), 0)
    return float(local_var.mean() / (255.0 ** 2))


def _bright_spot_ratio(gray: np.ndarray, thresh: int = 245) -> float:
    return float((gray >= thresh).mean())


class WeatherDetector:
    """
    Scores low_light, haze, rain (streaks); picks argmax if above threshold else ``clear``.
    """

    def __init__(
        self,
        brightness_thresh: float = 0.30,
        dark_pixel_ratio_thresh: float = 0.55,
        haze_dcp_thresh: float = 0.55,
        haze_contrast_thresh: float = 0.08,
        raindrop_var_thresh: float = 0.012,
        raindrop_spot_thresh: float = 0.002,
        min_confidence: float = 0.35,
        haze_variant: Literal["its", "ots"] = "its",
    ) -> None:
        self.brightness_thresh = brightness_thresh
        self.dark_pixel_ratio_thresh = dark_pixel_ratio_thresh
        self.haze_dcp_thresh = haze_dcp_thresh
        self.haze_contrast_thresh = haze_contrast_thresh
        self.raindrop_var_thresh = raindrop_var_thresh
        self.raindrop_spot_thresh = raindrop_spot_thresh
        self.min_confidence = min_confidence
        self.haze_variant = haze_variant

    def detect(self, image: np.ndarray | Image.Image) -> WeatherResult:
        bgr, gray = _to_bgr_gray(image)
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return WeatherResult(
                condition="clear",
                confidence=0.0,
                stats={"mean_brightness": 0.0},
                suggested_mode="none",
            )

        mean_brightness = float(gray.mean() / 255.0)
        dark_ratio = float((gray < 50).mean())
        dcp_mean = _dark_channel_mean(bgr)
        edge_den = _edge_density(gray)
        contrast_std = float(gray.std() / 255.0)
        local_var = _local_contrast_var(gray)
        bright_spots = _bright_spot_ratio(gray)

        # Normalized scores in [0, 1]
        low_light_score = 0.0
        if mean_brightness < self.brightness_thresh:
            low_light_score += 0.5 * (1.0 - mean_brightness / max(self.brightness_thresh, 1e-6))
        if dark_ratio > self.dark_pixel_ratio_thresh:
            low_light_score += 0.5 * min(1.0, (dark_ratio - self.dark_pixel_ratio_thresh) / 0.35)
        low_light_score = float(np.clip(low_light_score, 0, 1))

        haze_score = 0.0
        if dcp_mean > self.haze_dcp_thresh:
            haze_score += 0.45 * min(1.0, (dcp_mean - self.haze_dcp_thresh) / 0.25)
        if contrast_std < self.haze_contrast_thresh:
            haze_score += 0.35 * (1.0 - contrast_std / max(self.haze_contrast_thresh, 1e-6))
        if edge_den < 0.06:
            haze_score += 0.2 * (1.0 - edge_den / 0.06)
        haze_score = float(np.clip(haze_score, 0, 1))

        rain_score = 0.0
        if local_var > self.raindrop_var_thresh:
            rain_score += 0.55 * min(1.0, (local_var - self.raindrop_var_thresh) / 0.03)
        if bright_spots > self.raindrop_spot_thresh:
            rain_score += 0.45 * min(1.0, bright_spots / 0.02)
        rain_score = float(np.clip(rain_score, 0, 1))

        scores = {
            "low_light": low_light_score,
            "haze": haze_score,
            "rain": rain_score,
        }
        best_cond = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_cond]

        if best_score < self.min_confidence:
            condition: WeatherCondition = "clear"
            confidence = 1.0 - best_score
        else:
            condition = best_cond  # type: ignore[assignment]
            confidence = best_score

        stats = {
            "mean_brightness": mean_brightness,
            "dark_pixel_ratio": dark_ratio,
            "dcp_mean": dcp_mean,
            "edge_density": edge_den,
            "contrast_std": contrast_std,
            "local_contrast_var": local_var,
            "bright_spot_ratio": bright_spots,
            "score_low_light": low_light_score,
            "score_haze": haze_score,
            "score_rain": rain_score,
        }

        return WeatherResult(
            condition=condition,
            confidence=float(confidence),
            stats=stats,
            suggested_mode=condition_to_mode(condition, haze_variant=self.haze_variant),
        )


__all__ = [
    "WeatherDetector",
    "WeatherResult",
    "WeatherCondition",
    "condition_to_mode",
]
