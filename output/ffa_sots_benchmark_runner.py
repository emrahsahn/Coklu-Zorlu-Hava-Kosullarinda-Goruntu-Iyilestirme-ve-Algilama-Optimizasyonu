"""
FFA-Net SOTS benchmark: paired hazy/clear → PSNR, SSIM, MAE (baseline vs restored).

Searches common RESIDE/SOTS paths under repo. Writes:
  output/ffa_sots_benchmark_results.csv
  output/FFA_SOTS_benchmark.md
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import calculate_psnr, calculate_ssim  # noqa: E402
from model_loader import FFANetWrapper  # noqa: E402


def _find_sots_pairs(split: str) -> tuple[Path, Path, list[tuple[Path, Path]]]:
    """split: 'indoor' (ITS weights) or 'outdoor' (OTS weights)."""
    candidates = [
        ROOT / "FFA-Net" / "FFA-Net" / "data" / "RESIDE" / "SOTS" / split,
        ROOT / "data" / "RESIDE" / "SOTS" / split,
        ROOT / "data" / "SOTS" / split,
    ]
    hazy_dir = clear_dir = None
    for base in candidates:
        h = base / "hazy"
        c = base / "clear"
        if h.is_dir() and c.is_dir():
            hazy_dir, clear_dir = h, c
            break
    if hazy_dir is None or clear_dir is None:
        return Path(), Path(), []

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    hazy_files = sorted(p for p in hazy_dir.iterdir() if p.suffix.lower() in exts)
    pairs: list[tuple[Path, Path]] = []
    gt_dir = base / "gt"
    if not clear_dir.is_dir() and gt_dir.is_dir():
        clear_dir = gt_dir

    def clear_id(hp: Path) -> str:
        return hp.stem.split("_")[0]

    for hp in hazy_files:
        base_id = clear_id(hp)
        cp = None
        for ext in (".png", ".jpg", ".jpeg"):
            t = clear_dir / f"{base_id}{ext}"
            if t.is_file():
                cp = t
                break
        if cp is not None:
            pairs.append((hp, cp))
    return hazy_dir, clear_dir, pairs


def _tensor_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().clamp(0, 1).squeeze(0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(np.transpose(x, (1, 2, 0)))


def _mae_np(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64) / 255.0
    bf = b.astype(np.float64) / 255.0
    h = min(af.shape[0], bf.shape[0])
    w = min(af.shape[1], bf.shape[1])
    return float(np.mean(np.abs(af[:h, :w] - bf[:h, :w])))


def run_split(task: str, split: str, device: torch.device) -> list[dict]:
    hazy_dir, clear_dir, pairs = _find_sots_pairs(split)
    if not pairs:
        print(f"[SKIP] SOTS {split}: no pairs under {hazy_dir.parent if hazy_dir else 'N/A'}")
        return []

    wrapper = FFANetWrapper(task=task, device=device).load()
    to_tensor = transforms.ToTensor()
    rows: list[dict] = []

    for hazy_path, clear_path in pairs:
        hazy_pil = Image.open(hazy_path).convert("RGB")
        clear_pil = Image.open(clear_path).convert("RGB")
        hazy_t = to_tensor(hazy_pil).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_t = wrapper.restore(hazy_t)
        ms = (time.perf_counter() - t0) * 1000.0

        hazy_rgb = np.asarray(hazy_pil, dtype=np.uint8)
        clear_rgb = np.asarray(clear_pil, dtype=np.uint8)
        rest_rgb = _tensor_to_hwc_uint8(out_t)

        base_psnr = calculate_psnr(hazy_rgb, clear_rgb)
        rest_psnr = calculate_psnr(rest_rgb, clear_rgb)
        base_ssim = calculate_ssim(hazy_rgb, clear_rgb)
        rest_ssim = calculate_ssim(rest_rgb, clear_rgb)
        base_mae = _mae_np(hazy_rgb, clear_rgb)
        rest_mae = _mae_np(rest_rgb, clear_rgb)
        imp = (base_mae - rest_mae) / (base_mae + 1e-8) * 100.0

        rows.append(
            {
                "split": split,
                "task": task,
                "image": hazy_path.name,
                "baseline_psnr": base_psnr,
                "ffa_psnr": rest_psnr,
                "baseline_ssim": base_ssim,
                "ffa_ssim": rest_ssim,
                "baseline_mae": base_mae,
                "ffa_mae": rest_mae,
                "mae_improvement_percent": imp,
                "time_ms": ms,
            }
        )
    print(f"[OK] SOTS {split} ({task}): N={len(rows)} from {hazy_dir}")
    return rows


def _avg(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indoor = run_split("its", "indoor", device)
    outdoor = run_split("ots", "outdoor", device)
    all_rows = indoor + outdoor

    csv_out = ROOT / "output" / "ffa_sots_benchmark_results.csv"
    md_out = ROOT / "output" / "FFA_SOTS_benchmark.md"

    if not all_rows:
        md_out.write_text(
            "# FFA-Net SOTS Benchmark\n\n"
            "SOTS veri seti bulunamadı. Aşağıdaki yollardan birine `hazy/` ve `clear/` klasörlerini yerleştirin:\n"
            "- `FFA-Net/FFA-Net/data/RESIDE/SOTS/indoor`\n"
            "- `FFA-Net/FFA-Net/data/RESIDE/SOTS/outdoor`\n"
            "- `data/RESIDE/SOTS/indoor` ve `outdoor`\n\n"
            "Ardından: `python output/ffa_sots_benchmark_runner.py`\n",
            encoding="utf-8",
        )
        print(f"[WARN] No SOTS data. Wrote stub: {md_out}")
        return

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    lines = [
        "# FFA-Net SOTS Benchmark (Pipeline entegrasyonu)",
        "",
        f"- Device: `{device}`",
        f"- ITS ağırlık: `its_train_ffa_3_20.pk` → SOTS **indoor**",
        f"- OTS ağırlık: `ots_train_ffa_3_19.pk` → SOTS **outdoor**",
        "",
    ]

    for split, task, subset in (
        ("indoor", "its", indoor),
        ("outdoor", "ots", outdoor),
    ):
        if not subset:
            continue
        lines.append(f"## SOTS {split} ({task.upper()}, N={len(subset)})")
        lines.append("")
        lines.append(
            f"- Ort. PSNR: Sisli girdi `{_avg(subset, 'baseline_psnr'):.4f}` | FFA `{_avg(subset, 'ffa_psnr'):.4f}` dB"
        )
        lines.append(
            f"- Ort. SSIM: Sisli girdi `{_avg(subset, 'baseline_ssim'):.4f}` | FFA `{_avg(subset, 'ffa_ssim'):.4f}`"
        )
        lines.append(
            f"- Ort. MAE: Sisli girdi `{_avg(subset, 'baseline_mae'):.6f}` | FFA `{_avg(subset, 'ffa_mae'):.6f}` "
            f"(iyileşme %{_avg(subset, 'mae_improvement_percent'):.2f})"
        )
        lines.append(f"- Ort. süre: `{_avg(subset, 'time_ms'):.2f}` ms/görüntü")
        lines.append("")

    lines.append("## Dosyalar")
    lines.append(f"- `{csv_out.relative_to(ROOT)}`")
    lines.append(f"- `{md_out.relative_to(ROOT)}`")
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] CSV: {csv_out}")
    print(f"[OK] MD: {md_out}")


if __name__ == "__main__":
    main()
