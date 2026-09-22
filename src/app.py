"""
Gradio web UI — tekil analiz, toplu işleme ve performans grafikleri.
Run from repo root::
    .venv\\Scripts\\python -m src.app
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from .model_loader import DEFAULT_DDN_VARIANT, repo_root
from .pipeline import UnifiedPipeline
_pipeline: Optional[UnifiedPipeline] = None
_last_batch_rows: list[dict[str, Any]] = []
_USAGE_HELP_MD = """
- **Otomatik (istatistik):** Mod olarak bunu seçtiğinizde, çalıştırınca görüntüye göre en uygun iyileştirme
  dalı otomatik seçilir.
- **Otomatik hava koşulu algıla** kutusu: Açıkken hava / bozulma istatistikleri hesaplanır ve
  **özet metriklerde** gösterilir; **seçtiğiniz modu değiştirmez** (manuel sis / düşük ışık vb.
  yine o modda çalışır).
- **Gamma / alpha_s / alpha_i:** Yalnızca **Düşük ışık — HVI-CIDNet** modunda modele gider.
- **PSNR / SSIM / MAE:** Çıktı boyutu girdiden farklıysa metrikler üst-sol ortak pencerede hesaplanır. MAE düşük daha iyidir.
- **Karşılaştırma kaydırıcısı:** Orijinal ve restore görüntüleri üst üste sürükleyerek karşılaştırın.
"""
def get_pipeline() -> UnifiedPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedPipeline()
    return _pipeline
MODE_CHOICES: List[tuple[str, str]] = [
    ("Otomatik (istatistik)", "auto"),
    ("Yok (iyileştirme yok, sadece geçiş)", "none"),
    ("Düşük ışık — HVI-CIDNet", "low_light_hvi"),
    ("Sis — FFA indoor (ITS)", "haze_ffa_its"),
    ("Sis — FFA outdoor (OTS)", "haze_ffa_ots"),
    ("Yağmur çizgisi — DDN", "rain_ddn"),
]
_MODE_LABEL_TR: dict[str, str] = {
    "none": "Yok (iyileştirme yok)",
    "low_light_hvi": "Düşük ışık (HVI-CIDNet)",
    "haze_ffa_its": "Sis — FFA indoor (ITS)",
    "haze_ffa_ots": "Sis — FFA outdoor (OTS)",
    "rain_ddn": "Yağmur çizgisi (DDN)",
}
DDN_VARIANT_CHOICES: List[tuple[str, str]] = [
    ("Rain1400 fine-tune (önerilen)", "rain1400"),
    ("Rain100L — hafif yağmur", "rain100L"),
    ("Rain100H — şiddetli yağmur", "rain100H"),
]
def _mode_label_tr(mode: str) -> str:
    return _MODE_LABEL_TR.get(mode, mode)
# Toplu tablo: önemli sütunlar önce (mode_specific_json en sonda kalır)
_BATCH_COL_PRIORITY: List[str] = [
    "filename",
    "mode",
    "yol_panel",
    "yol_ozet",
    "otomatik_ozet",
    "girdi_hw",
    "cikti_hw",
    "boyut_ayni",
    "weather_condition",
    "weather_confidence",
    "psnr",
    "ssim",
    "mae",
    "mode_specific_json",
]
_DETECTION_TABLE_COLS = frozenset({
    "count_original",
    "count_restored",
    "count_delta",
    "mean_confidence_original",
    "mean_confidence_restored",
    "mean_confidence_delta",
})


def _reorder_batch_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in _DETECTION_TABLE_COLS if c in df.columns])
    front = [c for c in _BATCH_COL_PRIORITY if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest] if front or rest else df
def _notify_warning(message: str) -> None:
    gr.Warning(message)
def _notify_info(message: str) -> None:
    gr.Info(message)


def _coerce_inp_preview(image: Any) -> Any:
    """Yükleme / yapıştırma sonrası önizlemenin RGB uint8 olarak kalması."""
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.dtype in (np.float32, np.float64):
        if float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if isinstance(image, np.ndarray) and image.shape == arr.shape and image.dtype == arr.dtype:
        if np.array_equal(image, arr):
            return image
    return arr


def auto_detect_weather(image: Any) -> Tuple[str, str]:
    if image is None:
        _notify_warning("Görüntü yükleyin, ardından hava koşulunu tahmin edin.")
        return "none", "*Görüntü yükleyin.*"
    pl = get_pipeline()
    w = pl.detect_weather(image)
    text = (
        f"**Tahmin:** `{w.condition}` (güven: {w.confidence:.2f})\n\n"
        f"**Önerilen mod:** `{w.suggested_mode}`\n\n"
        f"```json\n{json.dumps(w.stats, indent=2)}\n```"
    )
    _notify_info(
        f"Hava koşulu: {w.condition} (güven {w.confidence:.2f}). "
        f"Mod önerisi uygulandı."
    )
    return w.suggested_mode, text
def _format_mode_specific_markdown(mode: str, ms: dict[str, Any]) -> str:
    """Gradio özeti: her restorasyon dalına uygun notlar ve otomatik tespit özeti."""
    if not ms:
        return ""
    lines: list[str] = ["#### Detaylı yol / model bilgisi", ""]
    ad = ms.get("auto_detection")
    if isinstance(ad, dict):
        lines.append(
            f"- **Otomatik sinyal:** `{ad.get('condition')}` "
            f"(güven {float(ad.get('confidence', 0)):.2f})"
        )
        lines.append(f"- **Önerilen mod:** `{ad.get('suggested_mode')}`")
        stats = ad.get("stats") or {}
        if stats:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(stats, indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")
    panel = ms.get("panel")
    if panel == "cidnet":
        c = ms.get("cidnet") or {}
        lines.append("**HVI-CIDNet (düşük ışık)**")
        lines.append("")
        lines.append("| gamma | alpha_s (doygunluk) | alpha_i (yoğunluk) |")
        lines.append("|:---:|:---:|:---:|")
        lines.append(
            f"| {c.get('gamma')} | {c.get('alpha_s')} | {c.get('alpha_i')} |"
        )
        lines.append("")
        lines.append(str(c.get("notes", "")))
    elif panel == "ffa":
        f = ms.get("ffa") or {}
        lines.append("**FFA-Net (sis giderme)**")
        lines.append("")
        lines.append(f"- **Varyant:** `{f.get('variant')}` — {f.get('description', '')}")
        lines.append(f"- **Ağırlık dosyası:** `{f.get('weights', '')}`")
        lines.append(
            f"- **Girdi normalizasyonu (mean/std):** `{f.get('input_normalize_mean')}` / "
            f"`{f.get('input_normalize_std')}`"
        )
    elif panel == "ddn":
        r = ms.get("ddn") or {}
        lines.append("**DDN (Deep Detailed Network — yağmur çizgisi)**")
        lines.append("")
        lines.append(f"- **Model:** {r.get('model', 'DDN')}")
        lines.append(f"- **Varyant:** `{r.get('variant', '')}` ({r.get('label', '')})")
        lines.append(f"- **Ağırlık:** `{r.get('weights', '')}`")
        lines.append(f"- **Görev:** {r.get('task', '')}")
        lines.append(f"- {r.get('notes', '')}")
    elif panel == "bypass":
        b = ms.get("bypass") or {}
        lines.append("**İyileştirme yok**")
        lines.append("")
        lines.append(str(b.get("note", "")))
    elif panel == "unknown":
        lines.append(f"*Bilinmeyen panel (`{mode}`).*")
    geo = ms.get("geometry") or {}
    if geo and not geo.get("sizes_match"):
        lines.append("")
        lines.append(
            "**Geometri:** Çıktı boyutu girdiden farklı; PSNR/SSIM/MAE üst-sol ortak pencerede hesaplanır."
        )
    return "\n".join(lines).strip() + "\n"


def _compose_horizontal_metric_strip(path_md: str, metrics_md: str) -> str:
    """Yol/mod ve kalite metriklerini tek satırda iki sütun halinde gösterir."""
    return (
        '<div class="metric-strip-outer">\n'
        f'<div class="metric-strip-card">\n\n{path_md.strip()}\n\n</div>\n'
        f'<div class="metric-strip-card">\n\n{metrics_md.strip()}\n\n</div>\n'
        "</div>\n"
    )


def process_single(
    image: Any,
    mode: str,
    gamma: float,
    alpha_s: float,
    alpha_i: float,
    use_auto_detect: bool,
    ddn_variant: str,
) -> Tuple[Any, str]:
    if image is None:
        _notify_warning("Önce bir görüntü yükleyin.")
        return None, "*Görüntü gerekli.*"
    pl = get_pipeline()
    res = pl.process(
        image,
        condition=mode,
        auto_weather=bool(use_auto_detect),
        use_yolo=False,
        gamma=float(gamma),
        alpha_s=float(alpha_s),
        alpha_i=float(alpha_i),
        ddn_variant=ddn_variant or DEFAULT_DDN_VARIANT,  # type: ignore[arg-type]
    )
    m = res.metrics
    ms = m.get("mode_specific") or {}
    mode_block = _format_mode_specific_markdown(res.mode, ms if isinstance(ms, dict) else {})
    col_path = (
        f"#### Yol / mod\n\n"
        f"| Mod (kod) | Mod |\n"
        f"|:---|:---|\n"
        f"| `{res.mode}` | {_mode_label_tr(res.mode)} |\n\n"
        f"{mode_block if mode_block.strip() else '*Ek yol notu yok.*'}\n"
    )
    psnr_note = ""
    if res.original_rgb.shape[:2] != res.restored_rgb.shape[:2]:
        psnr_note = (
            "\n\n*PSNR/SSIM/MAE: Çıktı boyutu orijinalden farklı olduğu için "
            "üst-sol ortak bölge üzerinden hesaplanır.*"
        )
    psnr_v = m.get("psnr", 0)
    psnr_str = "∞" if isinstance(psnr_v, float) and psnr_v == float("inf") else f"{float(psnr_v):.3f}"
    col_metrics = (
        f"#### Görüntü kalitesi (orijinal → restore)\n\n"
        f"| PSNR ↑ | SSIM ↑ | MAE ↓ |\n"
        f"|:---:|:---:|:---:|\n"
        f"| {psnr_str} | {m.get('ssim', 0):.4f} | {m.get('mae', 0):.6f} |\n"
        f"{psnr_note}\n"
    )
    metric_strip = _compose_horizontal_metric_strip(col_path, col_metrics)
    _notify_info(f"İşlem tamamlandı — mod: {_mode_label_tr(res.mode)}")
    return (res.original_rgb, res.restored_rgb), metric_strip
def _empty_batch_outputs(
    message: str,
) -> Tuple[Any, str, list, Optional[str], Optional[str]]:
    return pd.DataFrame(), message, [], None, None


def _batch_result_message(df: pd.DataFrame, rows: list, csv_path: Path, json_path: Path) -> str:
    gt_note = ""
    if rows and rows[0].get("reference_metrics"):
        gt_note = (
            "\n\n*Metrikler eşleştirilmiş **ground truth** klasörüne göre "
            "(restore → GT) hesaplanmıştır.*\n"
        )
    return (
        f"**{len(rows)}** görüntü işlendi.\n\n"
        f"Raporlar `output/batch_pipeline/` altına yazıldı:\n"
        f"- **CSV:** `{csv_path}`\n"
        f"- **JSON:** `{json_path}`\n\n"
        "Aşağıdan dosyaları indirebilir veya tabloyu kopyalayabilirsiniz.\n\n"
        f"Ortalama PSNR: **{df['psnr'].mean():.3f}** | Ortalama SSIM: **{df['ssim'].mean():.4f}** "
        f"| Ortalama MAE: **{df['mae'].mean():.6f}**"
        f"{gt_note}"
    )


def _run_batch_on_paths(
    file_paths: list[Path],
    *,
    mode: str,
    gamma: float,
    alpha_s: float,
    alpha_i: float,
    use_auto: bool,
    reference_dir: Optional[Path],
    ddn_variant: str,
    progress: gr.Progress,
) -> Tuple[Any, str, list, Optional[str], Optional[str]]:
    """Ortak toplu işleme: restore + CSV/JSON + galeri."""
    global _last_batch_rows
    from .metrics import generate_report, load_image_rgb, summarize_pipeline_result
    from .pipeline import _batch_display_columns_from_mode_specific

    if not file_paths:
        _notify_warning("İşlenecek görüntü yok.")
        return _empty_batch_outputs("İşlenecek görüntü yok.")

    pl = get_pipeline()
    out_dir = repo_root() / "output" / "batch_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    gallery_items: list[tuple[str, str]] = []
    sorted_paths = sorted(file_paths, key=lambda p: p.name)
    total = len(sorted_paths)
    ref_dir = reference_dir if reference_dir and reference_dir.is_dir() else None

    for i, path in enumerate(sorted_paths):
        progress((i + 1) / total, desc=f"İşleniyor ({i + 1}/{total}): {path.name}")
        reference_rgb = None
        if ref_dir is not None:
            ref_path = ref_dir / path.name
            if ref_path.is_file():
                reference_rgb = load_image_rgb(ref_path)
        rgb = load_image_rgb(path)
        res = pl.process(
            rgb,
            condition=mode,
            auto_weather=bool(use_auto),
            use_yolo=False,
            gamma=float(gamma),
            alpha_s=float(alpha_s),
            alpha_i=float(alpha_i),
            reference_rgb=reference_rgb,
            ddn_variant=ddn_variant or DEFAULT_DDN_VARIANT,  # type: ignore[arg-type]
        )
        row = summarize_pipeline_result(
            filename=path.name,
            mode=res.mode,
            original_rgb=res.original_rgb,
            restored_rgb=res.restored_rgb,
            detections_original=[],
            detections_restored=[],
            reference_rgb=reference_rgb,
        )
        row["reference_metrics"] = reference_rgb is not None
        if res.weather is not None:
            row["weather_condition"] = res.weather.condition
            row["weather_confidence"] = res.weather.confidence
        ms = res.metrics.get("mode_specific", {})
        row["mode_specific_json"] = json.dumps(ms, ensure_ascii=False)
        row.update(_batch_display_columns_from_mode_specific(ms))
        rows.append(row)
        restored_path = out_dir / f"{path.stem}_restored.png"
        Image.fromarray(res.restored_rgb).save(restored_path)
        gallery_items.append((str(restored_path), path.name))

    csv_path, json_path = generate_report(rows, out_dir, basename="batch_report")
    _last_batch_rows = rows
    df = _reorder_batch_dataframe(pd.DataFrame(rows))
    msg = _batch_result_message(df, rows, csv_path, json_path)
    _notify_info(f"Toplu işleme tamamlandı: {len(rows)} görüntü — CSV/JSON hazır.")
    return df, msg, gallery_items, str(csv_path), str(json_path)


def run_batch(
    source: str,
    files: Any,
    folder_path: str,
    gt_folder_path: str,
    mode: str,
    gamma: float,
    alpha_s: float,
    alpha_i: float,
    use_auto: bool,
    limit: int,
    ddn_variant: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Any, str, list, Optional[str], Optional[str]]:
    """
    Toplu işleme: dosya listesi veya klasör yolu.
    Her çalıştırmada ``output/batch_pipeline/batch_report.csv`` ve ``.json`` üretilir.
    """
    lim = int(limit) if limit and limit > 0 else None
    ref_dir: Optional[Path] = None
    if gt_folder_path and str(gt_folder_path).strip():
        ref_dir = Path(str(gt_folder_path).strip())

    src = (source or "folder").strip().lower()
    if src == "files":
        if not files:
            _notify_warning("Önce **Girdi görüntüleri** alanından dosya seçin.")
            return _empty_batch_outputs("*Dosya seçilmedi.*")
        if isinstance(files, str):
            file_paths = [Path(files)]
        else:
            file_paths = [Path(str(f)) for f in files]
        file_paths = [
            p for p in file_paths
            if p.suffix.lower() in _IMAGE_EXTS_BATCH and p.is_file()
        ]
        if not file_paths:
            _notify_warning("Seçilen dosyalarda desteklenen görüntü yok.")
            return _empty_batch_outputs(
                "Seçilen dosyalarda desteklenen görüntü bulunamadı (.jpg, .png, …)."
            )
        if lim is not None:
            file_paths = file_paths[:lim]
        return _run_batch_on_paths(
            file_paths,
            mode=mode,
            gamma=gamma,
            alpha_s=alpha_s,
            alpha_i=alpha_i,
            use_auto=use_auto,
            reference_dir=ref_dir,
            ddn_variant=ddn_variant,
            progress=progress,
        )

    if not folder_path or not str(folder_path).strip():
        _notify_warning("**Girdi klasörü** yolunu girin (ör. data/lol_dataset/eval15/low).")
        return _empty_batch_outputs(
            "Klasör yolu boş. Örnek: `data/lol_dataset/eval15/low`"
        )
    folder = Path(str(folder_path).strip())
    if not folder.is_dir():
        _notify_warning(f"Klasör bulunamadı: {folder}")
        return _empty_batch_outputs(f"Klasör bulunamadı: `{folder}`")

    file_paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS_BATCH
    )
    if lim is not None:
        file_paths = file_paths[:lim]
    return _run_batch_on_paths(
        file_paths,
        mode=mode,
        gamma=gamma,
        alpha_s=alpha_s,
        alpha_i=alpha_i,
        use_auto=use_auto,
        reference_dir=ref_dir,
        ddn_variant=ddn_variant,
        progress=progress,
    )
def build_performance_charts() -> Tuple[Any, Any]:
    if not _last_batch_rows:
        empty = go.Figure()
        empty.update_layout(title="Önce Toplu İşleme sekmesinde bir klasör çalıştırın")
        return empty, empty
    df = pd.DataFrame(_last_batch_rows)
    fig_metrics = px.bar(
        df,
        x="filename",
        y=["psnr", "ssim"],
        barmode="group",
        title="PSNR / SSIM (orijinal vs restore)",
        labels={"value": "Skor", "filename": "Dosya"},
    )
    fig_metrics.update_layout(xaxis_tickangle=-45, template="plotly_dark")
    fig_mae = px.bar(
        df,
        x="filename",
        y="mae",
        title="MAE (orijinal vs restore, düşük daha iyi)",
        labels={"mae": "MAE", "filename": "Dosya"},
    )
    fig_mae.update_layout(xaxis_tickangle=-45, template="plotly_dark")
    return fig_metrics, fig_mae
def _theme_and_css() -> tuple[gr.themes.Base, str]:
    css = """
    .gradio-container { background: #12131C !important; color: #e8e8ef !important; }
    footer { visibility: hidden; }
    .gr-button-primary {
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%) !important;
        border: none !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(0, 242, 254, 0.35) !important;
    }
    .gr-panel, .gr-box {
        background: rgba(26, 27, 36, 0.85) !important;
        border: 1px solid rgba(79, 172, 254, 0.25) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    h1, h2, h3 { color: #00F2FE !important; }
    .metric-strip-outer {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.75rem !important;
        width: 100% !important;
        align-items: start !important;
    }
    .metric-strip-card {
        min-width: 0 !important;
        padding: 0.5rem 0.65rem !important;
        border: 1px solid rgba(79, 172, 254, 0.2) !important;
        border-radius: 10px !important;
        background: rgba(20, 22, 30, 0.6) !important;
    }
    .metric-strip-card table { width: 100% !important; font-size: 0.9rem !important; }
    @media (max-width: 1100px) {
        .metric-strip-outer { grid-template-columns: 1fr !important; }
    }
    .app-header { text-align: center; padding: 0.75rem 0 0.25rem; }
    .app-header h1 { margin: 0; font-size: 1.75rem; color: #00F2FE !important; }
    .app-header p { margin: 0.5rem 0 0; opacity: 0.75; font-size: 0.95rem; }
    /* Tekil girdi: .image-preview tam ekran katmanıdır — dokunmayın */
    #tekil-girdi-image .image-container {
        min-height: 360px !important;
    }
    #tekil-girdi-image .gr-box {
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }
    #tekil-girdi-image .upload-container {
        min-height: 300px !important;
        border: 2px dashed rgba(79, 172, 254, 0.35) !important;
        border-radius: 12px !important;
        background-color: #2a2d3a !important;
        background-image: repeating-conic-gradient(
            #3a3f52 0% 25%, #2a2d3a 0% 50%
        ) !important;
        background-size: 16px 16px !important;
    }
    #tekil-girdi-image .image-frame {
        min-height: 300px !important;
        width: 100% !important;
        background-color: #2a2d3a !important;
        border-radius: 12px !important;
    }
    #tekil-girdi-image .image-frame img {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        max-width: 100% !important;
        max-height: 340px !important;
        width: auto !important;
        height: auto !important;
        margin: 0 auto !important;
        object-fit: contain !important;
    }
    """
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    return theme, css
def _mode_help_visibility(
    mode_val: str,
) -> tuple[Any, Any, Any, Any, Any]:
    m = (mode_val or "auto").strip()
    return (
        gr.update(visible=(m == "auto")),
        gr.update(visible=(m == "low_light_hvi")),
        gr.update(visible=(m in ("haze_ffa_its", "haze_ffa_ots"))),
        gr.update(visible=(m == "rain_ddn")),
        gr.update(visible=(m == "none")),
    )
def _batch_hvi_acc_visibility(mode_val: str):
    return gr.update(visible=((mode_val or "").strip() == "low_light_hvi"))
_IMAGE_EXTS_BATCH = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
def _batch_source_visibility(source: str) -> tuple[gr.update, gr.update]:
    use_files = (source or "").strip().lower() == "files"
    return (
        gr.update(visible=use_files),
        gr.update(visible=not use_files),
    )


def _batch_files_selection_summary(files: Any) -> str:
    """Dosyalar seçilince sayı ve isim önizlemesi."""
    if not files:
        return "*Henüz dosya seçilmedi.*"
    if isinstance(files, str):
        paths = [Path(files)]
    else:
        paths = [Path(str(f)) for f in files]
    images = [p for p in paths if p.suffix.lower() in _IMAGE_EXTS_BATCH]
    n = len(images)
    sample = ", ".join(p.name for p in images[:5])
    if n > 5:
        sample += f", … (+{n - 5})"
    lines = [f"**Seçilen görüntü sayısı:** {n}"]
    if sample:
        lines.append(f"**Dosyalar:** {sample}")
    if n == 0:
        lines.append("Desteklenen görüntü formatı yok (.jpg, .png, …).")
        _notify_warning("Seçilen dosyalarda desteklenen görüntü formatı yok.")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hibrit Görüntü Restorasyonu") as demo:
        gr.Markdown(
            """
            <div class="app-header">
                <h1>Çoklu Zorlu Hava Koşullarında Görüntü İyileştirme — Hibrit Pipeline</h1>
                <p>Görüntü İyileştirme ve Otomatik Hava Koşulu Algılama</p>
            </div>
            """
        )
        with gr.Accordion("Nasıl kullanılır?", open=False):
            gr.Markdown(
                "**Hızlı başlangıç:** Görüntü yükleyin → **İyileştirme modu** seçin → **Çalıştır**.\n\n"
                + _USAGE_HELP_MD
            )
        with gr.Tabs():
            with gr.Tab("Tekil analiz"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inp = gr.Image(
                            label="Girdi görüntüsü",
                            type="numpy",
                            image_mode="RGB",
                            height=360,
                            sources=["upload", "clipboard"],
                            elem_id="tekil-girdi-image",
                            placeholder="Görsel yükleyin veya panodan yapıştırın",
                            interactive=True,
                        )
                        btn_detect = gr.Button(
                            "Hava koşulunu tahmin et (önerilen moda ayarla)"
                        )
                        weather_md = gr.Markdown()
                        mode = gr.Dropdown(
                            label="İyileştirme modu",
                            choices=MODE_CHOICES,
                            value="auto",
                        )
                        use_auto = gr.Checkbox(
                            label="Otomatik hava koşulu algıla (özet)",
                            value=True,
                            info="Açık: istatistiksel tahmin özet panelde gösterilir. Seçilen modu ezmez.",
                        )
                        with gr.Accordion(
                            "Otomatik (istatistik) modu",
                            open=False,
                            visible=True,
                        ) as acc_auto:
                            gr.Markdown(
                                "Açılır listede **Otomatik (istatistik)** seçiliyken **Çalıştır**, "
                                "görüntüye göre düşük ışık / sis / damlacık / yok arasından bir dal seçer. "
                                "Üstteki *Otomatik hava koşulu algıla* kutusu bunu değiştirmez; sadece "
                                "istatistik özetini ekler."
                            )
                        with gr.Accordion(
                            "Düşük ışık (HVI-CIDNet) — gamma / doygunluk / yoğunluk",
                            open=False,
                            visible=False,
                        ) as acc_hvi:
                            gr.Markdown(
                                "Bu kaydırıcılar **yalnızca** mod *Düşük ışık — HVI-CIDNet* seçiliyken modele gider."
                            )
                            gamma = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Gamma (girdi^gamma)")
                            alpha_s = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Doygunluk (alpha_s)")
                            alpha_i = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Yoğunluk (alpha_i)")
                        with gr.Accordion(
                            "Sis (FFA-Net) — girdi ön-işleme",
                            open=False,
                            visible=False,
                        ) as acc_ffa:
                            gr.Markdown(
                                "Girdi tensörü **[0,1]** aralığında **mean** `[0.64, 0.6, 0.58]` ve **std** "
                                "`[0.14, 0.15, 0.152]` ile normalize edilir. **ITS** indoor / genel sis; "
                                "**OTS** outdoor sis ağırlığıdır. Gamma / alpha bu dalda **kullanılmaz**."
                            )
                        with gr.Accordion(
                            "Yağmur çizgisi (DDN)",
                            open=False,
                            visible=False,
                        ) as acc_rain:
                            gr.Markdown(
                                "**Deep Detailed Network (DDN)** — `Deep_Detailed_Network-PyTorch-master` "
                                "içindeki eğitilmiş ağırlıklar. Varsayılan: **Rain1400 fine-tune** "
                                "(`rain1400_finetune_new/model_best.pth`). Guided filter + detay ağı; "
                                "çıktı girdi ile aynı çözünürlükte."
                            )
                            ddn_variant = gr.Dropdown(
                                label="DDN ağırlık varyantı",
                                choices=DDN_VARIANT_CHOICES,
                                value=DEFAULT_DDN_VARIANT,
                            )
                        with gr.Accordion(
                            "İyileştirme yok",
                            open=False,
                            visible=False,
                        ) as acc_none:
                            gr.Markdown(
                                "Restorasyon atlanır; görüntü doğrudan geçirilir. PSNR/SSIM/MAE (kendi kendine) "
                                "referansıyla karşılaştırılır."
                            )
                        btn_run = gr.Button("Çalıştır", variant="primary")
                    with gr.Column(scale=2):
                        out_compare = gr.ImageSlider(
                            label="Orijinal ↔ Restore (sürükleyerek karşılaştırın)",
                            type="numpy",
                            slider_position=50,
                            max_height=480,
                        )
                        out_metric_strip = gr.Markdown(elem_classes=["metric-strip-host"])
                mode.change(
                    _mode_help_visibility,
                    inputs=[mode],
                    outputs=[acc_auto, acc_hvi, acc_ffa, acc_rain, acc_none],
                )
                demo.load(
                    _mode_help_visibility,
                    inputs=[mode],
                    outputs=[acc_auto, acc_hvi, acc_ffa, acc_rain, acc_none],
                )
                inp.upload(_coerce_inp_preview, inputs=inp, outputs=inp)
                inp.input(_coerce_inp_preview, inputs=inp, outputs=inp)
                btn_detect.click(auto_detect_weather, [inp], [mode, weather_md])
                btn_run.click(
                    process_single,
                    [inp, mode, gamma, alpha_s, alpha_i, use_auto, ddn_variant],
                    [out_compare, out_metric_strip],
                )
            with gr.Tab("Toplu işleme"):
                gr.Markdown(
                    "Klasördeki veya seçtiğiniz görüntüleri toplu işler; sonuçlar "
                    "**`output/batch_pipeline/batch_report.csv`** ve **`.json`** olarak kaydedilir. "
                    "LOL gibi eşleştirilmiş setlerde **ground truth klasörü** verirseniz PSNR/SSIM/MAE "
                    "restore → GT olarak hesaplanır."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_source = gr.Radio(
                            label="Girdi kaynağı",
                            choices=[
                                ("Klasör yolu (önerilen)", "folder"),
                                ("Dosya yükle", "files"),
                            ],
                            value="folder",
                        )
                        with gr.Group(visible=True) as batch_folder_group:
                            _default_low = str(
                                repo_root() / "data" / "lol_dataset" / "eval15" / "low"
                            )
                            _default_high = str(
                                repo_root() / "data" / "lol_dataset" / "eval15" / "high"
                            )
                            batch_folder_path = gr.Textbox(
                                label="Girdi klasörü",
                                value=_default_low,
                                placeholder="ör. data/lol_dataset/eval15/low",
                            )
                            batch_gt_folder = gr.Textbox(
                                label="Ground truth klasörü (isteğe bağlı)",
                                value=_default_high,
                                placeholder="Aynı dosya adları; boş bırakılırsa metrikler orijinal→restore",
                            )
                        with gr.Group(visible=False) as batch_files_group:
                            batch_folder_inp = gr.File(
                                label="Girdi görüntüleri (birden fazla seçilebilir)",
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                                height=280,
                            )
                            batch_folder_info = gr.Markdown(
                                value="*Dosyalar seçilince burada özet görünür.*"
                            )
                        batch_mode = gr.Dropdown(
                            label="İyileştirme modu",
                            choices=MODE_CHOICES,
                            value="auto",
                        )
                        batch_auto = gr.Checkbox(
                            label="Otomatik hava koşulu (mod seçimi)",
                            value=True,
                            info="Açık: her görüntü için istatistikle mod seçilir (manuel modu ezmez).",
                        )
                        with gr.Accordion(
                            "Düşük ışık (HVI-CIDNet) parametreleri",
                            open=False,
                            visible=False,
                        ) as batch_acc_hvi:
                            batch_gamma = gr.Slider(
                                0.5, 2.0, value=1.0, step=0.05, label="Gamma"
                            )
                            batch_alpha_s = gr.Slider(
                                0.5, 1.5, value=1.0, step=0.05, label="alpha_s"
                            )
                            batch_alpha_i = gr.Slider(
                                0.5, 1.5, value=1.0, step=0.05, label="alpha_i"
                            )
                        batch_ddn_variant = gr.Dropdown(
                            label="DDN ağırlık varyantı (yağmur modu)",
                            choices=DDN_VARIANT_CHOICES,
                            value=DEFAULT_DDN_VARIANT,
                        )
                        batch_limit = gr.Number(
                            label="Maks. görüntü (0 = hepsi)",
                            value=0,
                            precision=0,
                        )
                        batch_btn = gr.Button("Toplu çalıştır ve rapor oluştur", variant="primary")
                    with gr.Column(scale=2):
                        batch_table = gr.Dataframe(
                            label="Sonuçlar",
                            interactive=False,
                        )
                        batch_msg = gr.Markdown()
                        with gr.Row():
                            batch_csv_dl = gr.File(
                                label="İndir — batch_report.csv",
                                interactive=False,
                            )
                            batch_json_dl = gr.File(
                                label="İndir — batch_report.json",
                                interactive=False,
                            )
                        batch_gallery = gr.Gallery(
                            label="Restore sonuçları",
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                batch_source.change(
                    _batch_source_visibility,
                    inputs=[batch_source],
                    outputs=[batch_files_group, batch_folder_group],
                )
                demo.load(
                    _batch_source_visibility,
                    inputs=[batch_source],
                    outputs=[batch_files_group, batch_folder_group],
                )
                batch_folder_inp.change(
                    _batch_files_selection_summary,
                    inputs=[batch_folder_inp],
                    outputs=[batch_folder_info],
                )
                batch_mode.change(
                    _batch_hvi_acc_visibility,
                    inputs=[batch_mode],
                    outputs=[batch_acc_hvi],
                )
                demo.load(
                    _batch_hvi_acc_visibility,
                    inputs=[batch_mode],
                    outputs=[batch_acc_hvi],
                )
            with gr.Tab("Performans raporu"):
                gr.Markdown(
                    "Toplu işleme çalıştırıldıktan sonra PSNR/SSIM ve MAE grafikleri "
                    "(toplu işlem bitince otomatik güncellenir)."
                )
                refresh_charts = gr.Button("Grafikleri güncelle")
                chart_metrics = gr.Plot(label="PSNR / SSIM")
                chart_mae = gr.Plot(label="MAE")
                refresh_charts.click(
                    build_performance_charts,
                    outputs=[chart_metrics, chart_mae],
                )
        batch_btn.click(
            run_batch,
            [
                batch_source,
                batch_folder_inp,
                batch_folder_path,
                batch_gt_folder,
                batch_mode,
                batch_gamma,
                batch_alpha_s,
                batch_alpha_i,
                batch_auto,
                batch_limit,
                batch_ddn_variant,
            ],
            [batch_table, batch_msg, batch_gallery, batch_csv_dl, batch_json_dl],
        ).then(
            build_performance_charts,
            outputs=[chart_metrics, chart_mae],
        )
    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False) -> None:
    theme, css = _theme_and_css()
    app = build_ui()
    app.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    launch()
