"""
PSNR, SSIM, MAE, detection comparison, and batch report helpers.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:  # pragma: no cover
    peak_signal_noise_ratio = None
    structural_similarity = None


def pair_same_hwc(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop both images to the overlapping top-left window ``min(H1,H2) x min(W1,W2)``.

    When restored size differs from input, metrics must compare
    the same spatial region.
    """
    a = np.asarray(img1)
    b = np.asarray(img2)
    if a.ndim < 3 or b.ndim < 3:
        raise ValueError("Expected HWC images")
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h, w = min(ha, hb), min(wa, wb)
    a2 = np.ascontiguousarray(a[:h, :w, : min(a.shape[2], b.shape[2])])
    b2 = np.ascontiguousarray(b[:h, :w, : min(a.shape[2], b.shape[2])])
    return a2, b2


def _as_float_hwc(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.dtype == np.uint8:
        x = x.astype(np.float64) / 255.0
    else:
        x = x.astype(np.float64)
        if x.max() > 1.5:
            x = x / 255.0
    return np.clip(x, 0, 1)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two RGB images. If shapes differ, compares the shared top-left crop."""
    a0, b0 = pair_same_hwc(img1, img2) if np.asarray(img1).shape != np.asarray(img2).shape else (img1, img2)
    a = _as_float_hwc(a0)
    b = _as_float_hwc(b0)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch after align: {a.shape} vs {b.shape}")
    if np.allclose(a, b, rtol=0.0, atol=1e-6):
        return float("inf")
    if peak_signal_noise_ratio is not None:
        return float(peak_signal_noise_ratio(a, b, data_range=1.0))
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM between two RGB images. If shapes differ, uses the shared top-left crop."""
    a0, b0 = pair_same_hwc(img1, img2) if np.asarray(img1).shape != np.asarray(img2).shape else (img1, img2)
    a = _as_float_hwc(a0)
    b = _as_float_hwc(b0)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch after align: {a.shape} vs {b.shape}")
    if structural_similarity is not None:
        return float(
            structural_similarity(a, b, data_range=1.0, channel_axis=2)
        )
    gray_a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    gray_b = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = gray_a.mean(), gray_b.mean()
    var_a, var_b = gray_a.var(), gray_b.var()
    cov = ((gray_a - mu_a) * (gray_b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den > 0 else 1.0


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean absolute error on [0,1] RGB. Lower is better. Uses shared top-left crop if shapes differ."""
    a0, b0 = pair_same_hwc(img1, img2) if np.asarray(img1).shape != np.asarray(img2).shape else (img1, img2)
    a = _as_float_hwc(a0)
    b = _as_float_hwc(b0)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch after align: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def _mean_confidence(detections: List[dict[str, Any]]) -> float:
    if not detections:
        return 0.0
    return float(sum(d["confidence"] for d in detections) / len(detections))


def compare_detections(
    det_original: List[dict[str, Any]],
    det_restored: List[dict[str, Any]],
) -> dict[str, Any]:
    n_o = len(det_original)
    n_r = len(det_restored)
    c_o = _mean_confidence(det_original)
    c_r = _mean_confidence(det_restored)
    return {
        "count_original": n_o,
        "count_restored": n_r,
        "count_delta": n_r - n_o,
        "mean_confidence_original": c_o,
        "mean_confidence_restored": c_r,
        "mean_confidence_delta": c_r - c_o,
    }


def summarize_pipeline_result(
    *,
    filename: str,
    mode: str,
    original_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    detections_original: List[dict[str, Any]],
    detections_restored: List[dict[str, Any]],
    reference_rgb: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """One row of metrics for batch tables / reports."""
    row: dict[str, Any] = {
        "filename": filename,
        "mode": mode,
        **compare_detections(detections_original, detections_restored),
    }
    if reference_rgb is not None:
        row["psnr"] = calculate_psnr(reference_rgb, restored_rgb)
        row["ssim"] = calculate_ssim(reference_rgb, restored_rgb)
        row["mae"] = calculate_mae(reference_rgb, restored_rgb)
    else:
        row["psnr"] = calculate_psnr(original_rgb, restored_rgb)
        row["ssim"] = calculate_ssim(original_rgb, restored_rgb)
        row["mae"] = calculate_mae(original_rgb, restored_rgb)
    return row


def generate_report(
    results: Sequence[dict[str, Any]],
    output_dir: Path | str,
    *,
    basename: str = "batch_report",
) -> tuple[Path, Path]:
    """
    Write CSV and JSON summaries under ``output_dir``. Returns paths to both files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{basename}.csv"
    json_path = out / f"{basename}.json"

    if not results:
        csv_path.write_text("filename,mode\n", encoding="utf-8")
        json_path.write_text("[]", encoding="utf-8")
        return csv_path, json_path

    keys = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in keys})

    psnrs = [r.get("psnr", 0) for r in results]
    finite_psnr = [float(x) for x in psnrs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    summary = {
        "num_images": len(results),
        "avg_psnr": float(np.mean(finite_psnr)) if finite_psnr else 0.0,
        "avg_ssim": float(np.mean([r.get("ssim", 0) for r in results])),
        "avg_mae": float(np.mean([r.get("mae", 0) for r in results])),
        "avg_count_original": float(np.mean([r.get("count_original", 0) for r in results])),
        "avg_count_restored": float(np.mean([r.get("count_restored", 0) for r in results])),
        "rows": list(results),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def load_image_rgb(path: Path | str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


__all__ = [
    "pair_same_hwc",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_mae",
    "compare_detections",
    "summarize_pipeline_result",
    "generate_report",
    "load_image_rgb",
]
