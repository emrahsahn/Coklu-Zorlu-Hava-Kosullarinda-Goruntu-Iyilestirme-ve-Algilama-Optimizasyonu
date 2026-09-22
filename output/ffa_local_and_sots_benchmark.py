"""
FFA-Net sis testi: yerel FFA-Net görselleri + (varsa) SOTS eşli GT.

1) FFA-Net/FFA-Net içindeki sisli girdiler (fig/, samples/) → pipeline FFA çıktısı
2) Mevcut pred_FFA_* ile uyum (yeniden üretim PSNR/SSIM)
3) SOTS indoor/outdoor varsa GT'ye karşı PSNR/SSIM/MAE

Çıktı: output/ffa_benchmark_results.csv, output/FFA_benchmark_report.md
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FFA_ROOT = ROOT / "FFA-Net" / "FFA-Net"
sys.path.insert(0, str(ROOT / "src"))

from metrics import calculate_psnr, calculate_ssim  # noqa: E402
from model_loader import FFANetWrapper  # noqa: E402

SOTS_TARGET = FFA_ROOT / "data" / "RESIDE" / "SOTS"
HF_DATASET = "desonglll/reside-SOTS"


def _tensor_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().clamp(0, 1).squeeze(0).numpy()
    return np.ascontiguousarray(np.transpose((x * 255.0).round().astype(np.uint8), (1, 2, 0)))


def _mae_np(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64) / 255.0
    bf = b.astype(np.float64) / 255.0
    h, w = min(af.shape[0], bf.shape[0]), min(af.shape[1], bf.shape[1])
    return float(np.mean(np.abs(af[:h, :w] - bf[:h, :w])))


def ensure_sots_layout() -> bool:
    """İndir ve SOTS klasör yapısını FFA-Net/data/RESIDE/SOTS altına kur."""
    indoor_h = SOTS_TARGET / "indoor" / "hazy"
    outdoor_h = SOTS_TARGET / "outdoor" / "hazy"
    if indoor_h.is_dir() and (SOTS_TARGET / "indoor" / "clear").is_dir():
        if outdoor_h.is_dir() and (SOTS_TARGET / "outdoor" / "clear").is_dir():
            return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARN] huggingface_hub yok; SOTS atlanıyor.")
        return False

    cache = ROOT / "data" / "hf_cache" / "SOTS.zip"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        print("[DL] SOTS.zip (~435 MB) indiriliyor...")
        zip_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename="SOTS.zip",
            repo_type="dataset",
            local_dir=str(cache.parent),
        )
        cache = Path(zip_path)
    else:
        print(f"[OK] Önbellek: {cache}")

    extract_root = cache.parent / "SOTS_extract"
    if not extract_root.is_dir():
        print("[ZIP] Açılıyor...")
        with zipfile.ZipFile(cache, "r") as zf:
            zf.extractall(extract_root)

    # Zip içi: SOTS/indoor/{hazy,clear}, SOTS/outdoor/{hazy,clear} veya benzeri
    candidates = list(extract_root.rglob("indoor"))
    src_base = None
    for c in candidates:
        if (c / "hazy").is_dir() and (c / "clear").is_dir():
            src_base = c.parent
            break
    if src_base is None:
        for name in ("SOTS", "RESIDE", "SOTS/outdoor"):
            p = extract_root / name
            if p.is_dir():
                src_base = p
                break
    if src_base is None:
        # doğrudan indoor/outdoor kökte olabilir
        if (extract_root / "indoor").is_dir():
            src_base = extract_root

    if src_base is None:
        print("[WARN] SOTS zip yapısı tanınamadı.")
        return False

    SOTS_TARGET.mkdir(parents=True, exist_ok=True)
    for split in ("indoor", "outdoor"):
        for sub in ("hazy", "clear", "gt"):
            src = src_base / split / sub
            dst = SOTS_TARGET / split / sub
            if src.is_dir() and not dst.is_dir():
                print(f"[CP] {src} -> {dst}")
                shutil.copytree(src, dst)
        # HF zip: ground truth `gt/` → benchmark `clear/`
        gt_dir = SOTS_TARGET / split / "gt"
        clear_dir = SOTS_TARGET / split / "clear"
        if gt_dir.is_dir() and not clear_dir.is_dir():
            print(f"[LINK] {gt_dir} -> clear/")
            shutil.copytree(gt_dir, clear_dir)
    ok = (
        (SOTS_TARGET / "indoor" / "hazy").is_dir()
        and (SOTS_TARGET / "indoor" / "clear").is_dir()
        and (SOTS_TARGET / "outdoor" / "hazy").is_dir()
        and (SOTS_TARGET / "outdoor" / "clear").is_dir()
    )
    return ok


def discover_local_hazy() -> list[tuple[Path, str]]:
    """(path, task) — task: its | ots."""
    out: list[tuple[Path, str]] = []
    fig = FFA_ROOT / "fig"
    if fig.is_dir():
        for p in sorted(fig.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            if "_FFA" in p.stem:
                continue
            # 0099_* ve 1400_* tipik SOTS indoor; aachen_* outdoor
            task = "ots" if "aachen" in p.name.lower() or p.stem.isdigit() else "its"
            if "0099" in p.name or "1400" in p.name:
                task = "its"
            out.append((p, task))
    sp = FFA_ROOT / "samples" / "043.png"
    if sp.is_file():
        out.append((sp, "ots"))
    return out


def _sots_clear_id(hazy_path: Path, split: str) -> str:
    """SOTS: indoor 1400_2.png → 1400; outdoor 0001_0.8_0.2.jpg → 0001."""
    stem = hazy_path.stem
    if split == "outdoor":
        return stem.split("_")[0]
    return stem.split("_")[0]


def find_sots_pairs(split: str) -> list[tuple[Path, Path]]:
    hazy_dir = SOTS_TARGET / split / "hazy"
    clear_dir = SOTS_TARGET / split / "clear"
    if not clear_dir.is_dir():
        gt_dir = SOTS_TARGET / split / "gt"
        if gt_dir.is_dir():
            clear_dir = gt_dir
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    pairs: list[tuple[Path, Path]] = []
    for hp in sorted(p for p in hazy_dir.iterdir() if p.suffix.lower() in exts):
        base = _sots_clear_id(hp, split)
        cp = None
        for ext in (".png", ".jpg", ".jpeg"):
            t = clear_dir / f"{base}{ext}"
            if t.is_file():
                cp = t
                break
        if cp is not None:
            pairs.append((hp, cp))
    return pairs


def eval_pair(
    wrapper: FFANetWrapper,
    hazy_path: Path,
    clear_path: Path | None,
    ref_pred_path: Path | None,
    device: torch.device,
    tag: str,
) -> dict:
    to_tensor = transforms.ToTensor()
    hazy_pil = Image.open(hazy_path).convert("RGB")
    hazy_t = to_tensor(hazy_pil).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_t = wrapper.restore(hazy_t)
    ms = (time.perf_counter() - t0) * 1000.0

    hazy_rgb = np.asarray(hazy_pil, dtype=np.uint8)
    rest_rgb = _tensor_to_hwc_uint8(out_t)
    row: dict = {
        "tag": tag,
        "image": hazy_path.name,
        "task": wrapper.task,
        "time_ms": ms,
    }
    if clear_path and clear_path.is_file():
        clear_rgb = np.asarray(Image.open(clear_path).convert("RGB"), dtype=np.uint8)
        row["baseline_psnr"] = calculate_psnr(hazy_rgb, clear_rgb)
        row["ffa_psnr"] = calculate_psnr(rest_rgb, clear_rgb)
        row["baseline_ssim"] = calculate_ssim(hazy_rgb, clear_rgb)
        row["ffa_ssim"] = calculate_ssim(rest_rgb, clear_rgb)
        row["baseline_mae"] = _mae_np(hazy_rgb, clear_rgb)
        row["ffa_mae"] = _mae_np(rest_rgb, clear_rgb)
        b_mae = row["baseline_mae"]
        row["mae_improvement_percent"] = (b_mae - row["ffa_mae"]) / (b_mae + 1e-8) * 100.0
        row["has_gt"] = True
    else:
        row["has_gt"] = False

    if ref_pred_path and ref_pred_path.is_file():
        ref_rgb = np.asarray(Image.open(ref_pred_path).convert("RGB"), dtype=np.uint8)
        row["repro_psnr_vs_pred_ffa"] = calculate_psnr(rest_rgb, ref_rgb)
        row["repro_ssim_vs_pred_ffa"] = calculate_ssim(rest_rgb, ref_rgb)
    return row


def ref_pred_path(hazy_path: Path, task: str) -> Path | None:
    stem = hazy_path.stem
    p = FFA_ROOT / f"pred_FFA_{task}" / f"{stem}_FFA.png"
    return p if p.is_file() else None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    # Yerel görseller
    for hazy_path, task in discover_local_hazy():
        w = FFANetWrapper(task=task, device=device).load()
        ref = ref_pred_path(hazy_path, task)
        rows.append(
            eval_pair(
                w,
                hazy_path,
                None,
                ref,
                device,
                tag=f"local_{task}",
            )
        )
        print(f"[local] {hazy_path.name} ({task}) repro={rows[-1].get('repro_psnr_vs_pred_ffa', 'n/a')}")

    # SOTS tam benchmark
    has_sots = ensure_sots_layout()
    if has_sots:
        for split, task in (("indoor", "its"), ("outdoor", "ots")):
            pairs = find_sots_pairs(split)
            if not pairs:
                print(f"[SKIP] SOTS {split}: çift yok")
                continue
            w = FFANetWrapper(task=task, device=device).load()
            print(f"[SOTS] {split}: {len(pairs)} çift")
            for hp, cp in pairs:
                ref = ref_pred_path(hp, task)
                rows.append(
                    eval_pair(w, hp, cp, ref, device, tag=f"sots_{split}")
                )

    csv_out = ROOT / "output" / "ffa_benchmark_results.csv"
    md_out = ROOT / "output" / "FFA_benchmark_report.md"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        md_out.write_text("# FFA benchmark\n\nVeri bulunamadı.\n", encoding="utf-8")
        return

    keys = sorted({k for r in rows for k in r.keys()})
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def avg(subset: list[dict], key: str) -> float:
        vals = [r[key] for r in subset if key in r and isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    lines = [
        "# FFA-Net Sis Benchmark",
        "",
        f"- Cihaz: `{device}`",
        f"- Yerel girdi: `FFA-Net/FFA-Net/fig/`, `samples/`",
        f"- SOTS: `{'evet' if has_sots else 'hayır (indirilemedi veya eksik)'}`",
        "",
    ]

    local = [r for r in rows if str(r.get("tag", "")).startswith("local")]
    if local:
        lines.append("## Yerel görseller (GT yok)")
        lines.append("")
        lines.append("| Görüntü | Görev | Süre (ms) | pred_FFA repro PSNR | repro SSIM |")
        lines.append("|---------|-------|-----------|---------------------|------------|")
        for r in local:
            if "repro_psnr_vs_pred_ffa" in r:
                lines.append(
                    f"| {r['image']} | {r['task']} | {r['time_ms']:.1f} | "
                    f"{r['repro_psnr_vs_pred_ffa']:.4f} | {r['repro_ssim_vs_pred_ffa']:.4f} |"
                )
            else:
                lines.append(
                    f"| {r['image']} | {r['task']} | {r['time_ms']:.1f} | — | — |"
                )
        lines.append("")

    for split, task in (("indoor", "its"), ("outdoor", "ots")):
        sub = [r for r in rows if r.get("tag") == f"sots_{split}"]
        if not sub:
            continue
        lines.append(f"## SOTS {split} ({task.upper()}, N={len(sub)}) — GT'ye karşı")
        lines.append("")
        lines.append(
            f"- Ort. PSNR: Sisli `{avg(sub, 'baseline_psnr'):.4f}` → FFA `{avg(sub, 'ffa_psnr'):.4f}` dB "
            f"(Δ {avg(sub, 'ffa_psnr') - avg(sub, 'baseline_psnr'):+.4f})"
        )
        lines.append(
            f"- Ort. SSIM: Sisli `{avg(sub, 'baseline_ssim'):.4f}` → FFA `{avg(sub, 'ffa_ssim'):.4f}`"
        )
        lines.append(
            f"- Ort. MAE: Sisli `{avg(sub, 'baseline_mae'):.6f}` → FFA `{avg(sub, 'ffa_mae'):.6f}` "
            f"(iyileşme %{avg(sub, 'mae_improvement_percent'):.2f})"
        )
        lines.append(f"- Ort. süre: `{avg(sub, 'time_ms'):.2f}` ms/görüntü")
        lines.append("")

    lines.append(f"## Dosyalar\n- `{csv_out.relative_to(ROOT)}`\n")
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {csv_out}")
    print(f"[OK] {md_out}")


if __name__ == "__main__":
    main()
