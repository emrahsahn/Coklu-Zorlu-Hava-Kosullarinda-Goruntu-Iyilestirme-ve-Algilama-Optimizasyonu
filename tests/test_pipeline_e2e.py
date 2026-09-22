"""
End-to-end smoke tests (stdlib only — run: python tests/test_pipeline_e2e.py).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_weather_detector():
    from src.weather_detector import WeatherDetector

    x = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    w = WeatherDetector().detect(x)
    assert w.condition in ("low_light", "haze", "rain", "clear")
    assert w.suggested_mode


def test_metrics():
    from src.metrics import calculate_mae, calculate_psnr, calculate_ssim, compare_detections

    a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    b = a.copy()
    assert calculate_ssim(a, b) > 0.99
    assert calculate_mae(a, b) < 1e-6
    cmp = compare_detections([], [{"confidence": 0.9, "class_name": "car"}])
    assert cmp["count_delta"] == 1


def test_pipeline_and_yolo():
    from src.pipeline import UnifiedPipeline

    x = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    pl = UnifiedPipeline()
    out = pl.process(x, condition="none", conf=0.5)
    assert out.original_rgb.shape == out.restored_rgb.shape
    assert "psnr" in out.metrics
    assert "mae" in out.metrics


def test_restoration_modes():
    from src.pipeline import UnifiedPipeline

    x = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    pl = UnifiedPipeline()
    for mode in ("low_light_hvi", "haze_ffa_its", "rain_ddn"):
        res = pl.run(x, mode=mode, conf=0.5)
        assert res.restored_rgb.shape == x.shape


def test_batch():
    from PIL import Image

    from src.pipeline import UnifiedPipeline

    x = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.png"
        Image.fromarray(x).save(p)
        pl = UnifiedPipeline()
        rows, csv_p, json_p = pl.process_batch(
            td, condition="none", auto_weather=False, limit=1
        )
        assert len(rows) == 1
        assert "yol_panel" in rows[0]
        assert rows[0]["yol_panel"] == "bypass"
        assert csv_p.is_file()
        assert json_p.is_file()


def test_gradio_build():
    from src.app import build_ui

    assert build_ui() is not None


def main() -> None:
    tests = [
        test_weather_detector,
        test_metrics,
        test_pipeline_and_yolo,
        test_restoration_modes,
        test_batch,
        test_gradio_build,
    ]
    for fn in tests:
        print(f"  {fn.__name__}...", end=" ")
        fn()
        print("ok")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
