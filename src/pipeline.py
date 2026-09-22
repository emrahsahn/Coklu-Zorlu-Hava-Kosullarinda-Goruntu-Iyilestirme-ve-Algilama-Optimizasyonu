"""
Orchestrates restoration (HVI-CIDNet / FFA-Net / DDN) and YOLOv8 detection on original vs restored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .metrics import (
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    compare_detections,
    generate_report,
    load_image_rgb,
    summarize_pipeline_result,
)
from .model_loader import (
    DDNVariant,
    DDNWrapper,
    DEFAULT_DDN_VARIANT,
    FFANetWrapper,
    HVI_CIDNetWrapper,
    ddn_variant_meta,
    default_device,
    repo_root,
)
from .weather_detector import WeatherDetector, WeatherResult
from .yolo_detector import YoloV8Detector

RestorationMode = Literal[
    "none",
    "low_light_hvi",
    "haze_ffa_its",
    "haze_ffa_ots",
    "rain_ddn",
]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _build_mode_specific_metrics(
    mode: str,
    *,
    use_auto: bool,
    weather: Optional[WeatherResult],
    gamma: float,
    alpha_s: float,
    alpha_i: float,
    orig_hw: tuple[int, int],
    rest_hw: tuple[int, int],
    ddn_variant: DDNVariant = DEFAULT_DDN_VARIANT,
) -> dict[str, Any]:
    """UI / rapor için moda özet bilgi (her restorasyon dalı için farklı alanlar)."""
    ho, wo = orig_hw
    hr, wr = rest_hw
    out: dict[str, Any] = {
        "geometry": {
            "original_hw": [int(ho), int(wo)],
            "restored_hw": [int(hr), int(wr)],
            "sizes_match": bool(ho == hr and wo == wr),
        },
    }
    if use_auto and weather is not None:
        out["auto_detection"] = weather.to_dict()

    if mode == "low_light_hvi":
        out["panel"] = "cidnet"
        out["cidnet"] = {
            "gamma": float(gamma),
            "alpha_s": float(alpha_s),
            "alpha_i": float(alpha_i),
            "notes": (
                "Girdi üzerinde gamma (x**gamma), RGB_HVI uzayında doygunluk (alpha_s) ve "
                "yoğunluk (alpha_i). Yalnızca bu modda anlamlıdır."
            ),
        }
    elif mode == "haze_ffa_its":
        out["panel"] = "ffa"
        out["ffa"] = {
            "variant": "ITS",
            "weights": "its_train_ffa_3_20.pk",
            "description": "Indoor / genel sis (RESIDE-ITS tarzı eğitim).",
            "input_normalize_mean": [0.64, 0.6, 0.58],
            "input_normalize_std": [0.14, 0.15, 0.152],
        }
    elif mode == "haze_ffa_ots":
        out["panel"] = "ffa"
        out["ffa"] = {
            "variant": "OTS",
            "weights": "ots_train_ffa_3_19.pk",
            "description": "Outdoor sis (RESIDE-outdoor tarzı eğitim).",
            "input_normalize_mean": [0.64, 0.6, 0.58],
            "input_normalize_std": [0.14, 0.15, 0.152],
        }
    elif mode == "rain_ddn":
        ddn_meta = ddn_variant_meta(ddn_variant)
        out["panel"] = "ddn"
        out["ddn"] = {
            "model": "Deep Detailed Network (DDN)",
            "variant": ddn_variant,
            "label": ddn_meta["label"],
            "weights": f"Deep_Detailed_Network-PyTorch-master/{ddn_meta['weights']}",
            "task": ddn_meta["task"],
            "notes": ddn_meta["notes"],
        }
    elif mode == "none":
        out["panel"] = "bypass"
        out["bypass"] = {"note": "Restorasyon atlandı; görüntü doğrudan geçirildi."}
    else:
        out["panel"] = "unknown"
    return out


def _batch_display_columns_from_mode_specific(ms: Any) -> dict[str, Any]:
    """Dataframe / CSV için kısa, okunur sütunlar (JSON yerine veya yanında)."""
    if not isinstance(ms, dict):
        return {
            "yol_panel": "",
            "girdi_hw": "",
            "cikti_hw": "",
            "boyut_ayni": True,
            "otomatik_ozet": "",
            "yol_ozet": "",
        }
    geo = ms.get("geometry") or {}
    oh, ow = geo.get("original_hw", [0, 0])
    rh, rw = geo.get("restored_hw", [0, 0])
    row_ex: dict[str, Any] = {
        "yol_panel": str(ms.get("panel", "")),
        "girdi_hw": f"{oh}×{ow}" if oh or ow else "",
        "cikti_hw": f"{rh}×{rw}" if rh or rw else "",
        "boyut_ayni": bool(geo.get("sizes_match", True)),
    }
    ad = ms.get("auto_detection")
    if isinstance(ad, dict):
        row_ex["otomatik_ozet"] = f"{ad.get('condition')} → {ad.get('suggested_mode')}"
    else:
        row_ex["otomatik_ozet"] = ""
    panel = ms.get("panel")
    if panel == "cidnet":
        c = ms.get("cidnet") or {}
        row_ex["yol_ozet"] = f"CIDNet γ={c.get('gamma')} αs={c.get('alpha_s')} αi={c.get('alpha_i')}"
    elif panel == "ffa":
        f = ms.get("ffa") or {}
        row_ex["yol_ozet"] = f"FFA {f.get('variant', '')} ({f.get('weights', '')})"
    elif panel == "ddn":
        d = ms.get("ddn") or {}
        label = d.get("label") or d.get("variant") or "DDN"
        row_ex["yol_ozet"] = f"DDN {label} ({d.get('task', 'yağmur çizgisi')})"
    elif panel == "bypass":
        row_ex["yol_ozet"] = "İyileştirme yok (bypass)"
    else:
        row_ex["yol_ozet"] = ""
    return row_ex


@dataclass
class PipelineResult:
    """RGB uint8 numpy images, detections, and optional metric summaries."""

    original_rgb: np.ndarray
    restored_rgb: np.ndarray
    original_annotated_rgb: np.ndarray
    restored_annotated_rgb: np.ndarray
    detections_original: list[dict[str, Any]]
    detections_restored: list[dict[str, Any]]
    json_original: str
    json_restored: str
    mode: str
    weather: Optional[WeatherResult] = None
    metrics: dict[str, Any] = field(default_factory=dict)


def _pil_to_tensor_rgb01(pil: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def _tensor_rgb01_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().clamp(0, 1).squeeze(0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(np.transpose(x, (1, 2, 0)))


class UnifiedPipeline:
    """
    Lazy-loads restoration models and YOLO.

    - ``detect_weather``: statistical degradation hint
    - ``restore`` / ``detect_objects``: single-step APIs
    - ``process``: full restore + dual YOLO + metrics
    - ``process_batch``: folder of images + CSV/JSON report
  """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        haze_variant: Literal["its", "ots"] = "its",
        ddn_variant: DDNVariant = DEFAULT_DDN_VARIANT,
    ) -> None:
        self.device = device or default_device()
        self.haze_variant = haze_variant
        self.ddn_variant = ddn_variant
        self._weather = WeatherDetector(haze_variant=haze_variant)
        self._yolo: Optional[YoloV8Detector] = None
        self._ddn: Optional[DDNWrapper] = None
        self._ddn_loaded_variant: Optional[DDNVariant] = None
        self._ffa_its: Optional[FFANetWrapper] = None
        self._ffa_ots: Optional[FFANetWrapper] = None
        self._hvi: Optional[HVI_CIDNetWrapper] = None

    def set_ddn_variant(self, variant: DDNVariant) -> None:
        """Yağmur modeli varyantını değiştir; bir sonraki inference'ta yeniden yüklenir."""
        if variant != self.ddn_variant:
            self.ddn_variant = variant
            self._ddn = None
            self._ddn_loaded_variant = None

    def detect_weather(self, image: np.ndarray | Image.Image) -> WeatherResult:
        return self._weather.detect(image)

    def yolo(self) -> YoloV8Detector:
        if self._yolo is None:
            self._yolo = YoloV8Detector(device=self.device).load()
        return self._yolo

    def _ddn_rain(self) -> DDNWrapper:
        v = self.ddn_variant
        if self._ddn is None or self._ddn_loaded_variant != v:
            self._ddn = DDNWrapper(variant=v, device=self.device).load()
            self._ddn_loaded_variant = v
        return self._ddn

    def _ffa(self, task: Literal["its", "ots"]) -> FFANetWrapper:
        if task == "its":
            if self._ffa_its is None:
                self._ffa_its = FFANetWrapper("its", device=self.device).load()
            return self._ffa_its
        if self._ffa_ots is None:
            self._ffa_ots = FFANetWrapper("ots", device=self.device).load()
        return self._ffa_ots

    def _hvi_cidnet(self) -> HVI_CIDNetWrapper:
        if self._hvi is None:
            self._hvi = HVI_CIDNetWrapper(device=self.device).load()
        return self._hvi

    def restore_tensor(
        self,
        mode: RestorationMode,
        rgb_01: torch.Tensor,
        *,
        gamma: float = 1.0,
        alpha_s: float = 1.0,
        alpha_i: float = 1.0,
    ) -> torch.Tensor:
        if mode == "none":
            return rgb_01
        if mode == "low_light_hvi":
            return self._hvi_cidnet().restore(rgb_01, gamma=gamma, alpha_s=alpha_s, alpha_i=alpha_i)
        if mode == "haze_ffa_its":
            return self._ffa("its").restore(rgb_01)
        if mode == "haze_ffa_ots":
            return self._ffa("ots").restore(rgb_01)
        if mode == "rain_ddn":
            return self._ddn_rain().restore(rgb_01)
        if mode == "raindrop":
            # Eski mod adı → DDN
            return self._ddn_rain().restore(rgb_01)
        raise ValueError(f"Unknown restoration mode: {mode}")

    def restore(
        self,
        image: np.ndarray | Image.Image,
        mode: RestorationMode,
        *,
        gamma: float = 1.0,
        alpha_s: float = 1.0,
        alpha_i: float = 1.0,
    ) -> np.ndarray:
        pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8))
        rgb_01 = _pil_to_tensor_rgb01(pil, self.device)
        out = self.restore_tensor(mode, rgb_01, gamma=gamma, alpha_s=alpha_s, alpha_i=alpha_i)
        return _tensor_rgb01_to_uint8_hwc(out)

    def detect_objects(
        self,
        image: np.ndarray | Image.Image,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> dict[str, Any]:
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(image).astype(np.uint8)
        return self.yolo().predict(rgb, conf=conf, iou=iou)

    def process(
        self,
        image: np.ndarray | Image.Image,
        condition: Optional[str] = None,
        *,
        auto_weather: bool = False,
        use_yolo: bool = True,
        conf: float = 0.25,
        iou: float = 0.45,
        gamma: float = 1.0,
        alpha_s: float = 1.0,
        alpha_i: float = 1.0,
        reference_rgb: Optional[np.ndarray] = None,
        ddn_variant: Optional[DDNVariant] = None,
    ) -> PipelineResult:
        if ddn_variant is not None:
            self.set_ddn_variant(ddn_variant)
        pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8))
        orig_rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)

        use_auto = bool(auto_weather or condition in (None, "auto", ""))
        weather = self.detect_weather(orig_rgb) if use_auto else None
        if condition and condition not in ("auto", ""):
            mode = condition  # type: ignore[assignment]
        elif use_auto and weather is not None:
            mode = weather.suggested_mode  # type: ignore[assignment]
        else:
            mode = "none"

        rgb_01 = _pil_to_tensor_rgb01(pil, self.device)
        restored = self.restore_tensor(
            mode,
            rgb_01,
            gamma=gamma,
            alpha_s=alpha_s,
            alpha_i=alpha_i,
        )
        restored_rgb = _tensor_rgb01_to_uint8_hwc(restored)

        if use_yolo:
            yo = self.yolo()
            o = yo.predict(orig_rgb, conf=conf, iou=iou)
            r = yo.predict(restored_rgb, conf=conf, iou=iou)
            det_cmp = compare_detections(o["detections"], r["detections"])
        else:
            empty_json = "[]"
            o = {
                "annotated_rgb": np.ascontiguousarray(orig_rgb.copy()),
                "detections": [],
                "json": empty_json,
            }
            r = {
                "annotated_rgb": np.ascontiguousarray(restored_rgb.copy()),
                "detections": [],
                "json": empty_json,
            }
            det_cmp = compare_detections([], [])

        if reference_rgb is not None:
            psnr = calculate_psnr(reference_rgb, restored_rgb)
            ssim = calculate_ssim(reference_rgb, restored_rgb)
            mae = calculate_mae(reference_rgb, restored_rgb)
        else:
            psnr = calculate_psnr(orig_rgb, restored_rgb)
            ssim = calculate_ssim(orig_rgb, restored_rgb)
            mae = calculate_mae(orig_rgb, restored_rgb)

        orig_hw = (int(orig_rgb.shape[0]), int(orig_rgb.shape[1]))
        rest_hw = (int(restored_rgb.shape[0]), int(restored_rgb.shape[1]))
        mode_specific = _build_mode_specific_metrics(
            mode,
            use_auto=use_auto,
            weather=weather,
            gamma=gamma,
            alpha_s=alpha_s,
            alpha_i=alpha_i,
            orig_hw=orig_hw,
            rest_hw=rest_hw,
            ddn_variant=self.ddn_variant,
        )
        metrics = {
            **det_cmp,
            "psnr": psnr,
            "ssim": ssim,
            "mae": mae,
            "use_yolo": use_yolo,
            "mode_specific": mode_specific,
        }

        return PipelineResult(
            original_rgb=orig_rgb,
            restored_rgb=restored_rgb,
            original_annotated_rgb=o["annotated_rgb"],
            restored_annotated_rgb=r["annotated_rgb"],
            detections_original=o["detections"],
            detections_restored=r["detections"],
            json_original=o["json"],
            json_restored=r["json"],
            mode=mode,
            weather=weather,
            metrics=metrics,
        )

    def run(
        self,
        image: np.ndarray | Image.Image,
        mode: RestorationMode = "none",
        *,
        use_yolo: bool = True,
        conf: float = 0.25,
        iou: float = 0.45,
        gamma: float = 1.0,
        alpha_s: float = 1.0,
        alpha_i: float = 1.0,
    ) -> PipelineResult:
        """Backward-compatible alias: fixed ``mode``, no auto weather."""
        return self.process(
            image,
            condition=mode,
            auto_weather=False,
            use_yolo=use_yolo,
            conf=conf,
            iou=iou,
            gamma=gamma,
            alpha_s=alpha_s,
            alpha_i=alpha_i,
        )

    def process_batch(
        self,
        folder: Path | str,
        condition: Optional[str] = None,
        *,
        auto_weather: bool = True,
        use_yolo: bool = True,
        conf: float = 0.25,
        output_dir: Optional[Path | str] = None,
        reference_dir: Optional[Path | str] = None,
        gamma: float = 1.0,
        alpha_s: float = 1.0,
        alpha_i: float = 1.0,
        limit: Optional[int] = None,
        ddn_variant: Optional[DDNVariant] = None,
    ) -> tuple[list[dict[str, Any]], Path, Path]:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Batch folder not found: {folder}")

        paths = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if limit is not None:
            paths = paths[:limit]

        out_dir = Path(output_dir) if output_dir else repo_root() / "output" / "batch_pipeline"
        out_dir.mkdir(parents=True, exist_ok=True)
        ref_dir = Path(reference_dir) if reference_dir else None

        rows: list[dict[str, Any]] = []
        for path in paths:
            rgb = load_image_rgb(path)
            reference_rgb = None
            if ref_dir is not None and ref_dir.is_dir():
                ref_path = ref_dir / path.name
                if ref_path.is_file():
                    reference_rgb = load_image_rgb(ref_path)
            res = self.process(
                rgb,
                condition=condition,
                auto_weather=auto_weather,
                use_yolo=use_yolo,
                conf=conf,
                gamma=gamma,
                alpha_s=alpha_s,
                alpha_i=alpha_i,
                reference_rgb=reference_rgb,
                ddn_variant=ddn_variant,
            )
            row = summarize_pipeline_result(
                filename=path.name,
                mode=res.mode,
                original_rgb=res.original_rgb,
                restored_rgb=res.restored_rgb,
                detections_original=res.detections_original,
                detections_restored=res.detections_restored,
                reference_rgb=reference_rgb,
            )
            if res.weather is not None:
                row["weather_condition"] = res.weather.condition
                row["weather_confidence"] = res.weather.confidence
            ms = res.metrics.get("mode_specific", {})
            row["mode_specific_json"] = json.dumps(ms, ensure_ascii=False)
            row.update(_batch_display_columns_from_mode_specific(ms))
            rows.append(row)

            stem = path.stem
            Image.fromarray(res.restored_rgb).save(out_dir / f"{stem}_restored.png")

        csv_path, json_path = generate_report(rows, out_dir, basename="batch_report")
        return rows, csv_path, json_path


__all__ = [
    "UnifiedPipeline",
    "PipelineResult",
    "RestorationMode",
]
