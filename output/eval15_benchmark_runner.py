import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    return (10 * torch.log10(1.0 / (mse + 1e-12))).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def run_benchmark() -> None:
    root = Path(__file__).resolve().parents[1]
    low_dir = root / "data" / "lol_dataset" / "eval15" / "low"
    high_dir = root / "data" / "lol_dataset" / "eval15" / "high"

    ccm_ckpt = root / "src_low_light" / "checkpoints" / "ccm" / "best_combined_model.pth"
    hvi_ckpt = root / "HVI-CIDNet" / "weights" / "train" / "epoch_100.pth"

    csv_out = root / "output" / "eval15_benchmark_results.csv"
    md_out = root / "output" / "CCM_vs_HVI-CIDNet_eval15_benchmark.md"

    sys.path.append(str(root / "src_low_light" / "CCM Combined"))
    from ccm_model import load_trained_model  # type: ignore

    sys.path.append(str(root / "HVI-CIDNet"))
    from net.CIDNet import CIDNet  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    ccm_model = load_trained_model(str(ccm_ckpt), device)
    ccm_model.eval()

    hvi_model = CIDNet().to(device)
    hvi_model.load_state_dict(torch.load(hvi_ckpt, map_location=device))
    hvi_model.eval()
    hvi_model.trans.gated = True
    hvi_model.trans.gated2 = True

    image_names = sorted([p.name for p in low_dir.glob("*.png") if (high_dir / p.name).exists()])
    if not image_names:
        raise RuntimeError("eval15 icinde eslesmis low/high PNG bulunamadi.")

    rows = []
    ccm_win = 0
    hvi_win = 0
    tie = 0

    for name in image_names:
        low_img = Image.open(low_dir / name).convert("RGB")
        gt_img = Image.open(high_dir / name).convert("RGB")

        low = to_tensor(low_img).unsqueeze(0).to(device)
        gt = to_tensor(gt_img).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            ccm_out, _, _, _, _ = ccm_model(low)
            ccm_out = torch.clamp(ccm_out, 0, 1)
        ccm_time_ms = (time.perf_counter() - t0) * 1000.0

        _, _, h, w = low.shape
        factor = 8
        h_aligned = ((h + factor) // factor) * factor
        w_aligned = ((w + factor) // factor) * factor
        pad_h = h_aligned - h if h % factor != 0 else 0
        pad_w = w_aligned - w if w % factor != 0 else 0
        low_padded = F.pad(low, (0, pad_w, 0, pad_h), mode="reflect")

        t1 = time.perf_counter()
        with torch.no_grad():
            hvi_out = hvi_model(low_padded**1.0)
            hvi_out = torch.clamp(hvi_out, 0, 1)[:, :, :h, :w]
        hvi_time_ms = (time.perf_counter() - t1) * 1000.0

        baseline_mae = mae(low, gt)
        ccm_mae = mae(ccm_out, gt)
        hvi_mae = mae(hvi_out, gt)

        baseline_psnr = psnr(low, gt)
        ccm_psnr = psnr(ccm_out, gt)
        hvi_psnr = psnr(hvi_out, gt)

        ccm_imp = (baseline_mae - ccm_mae) / (baseline_mae + 1e-8) * 100.0
        hvi_imp = (baseline_mae - hvi_mae) / (baseline_mae + 1e-8) * 100.0

        if hvi_mae < ccm_mae:
            hvi_win += 1
            winner = "HVI-CIDNet"
        elif ccm_mae < hvi_mae:
            ccm_win += 1
            winner = "CCM"
        else:
            tie += 1
            winner = "Tie"

        rows.append(
            {
                "image": name,
                "baseline_mae": baseline_mae,
                "ccm_mae": ccm_mae,
                "hvi_mae": hvi_mae,
                "baseline_psnr": baseline_psnr,
                "ccm_psnr": ccm_psnr,
                "hvi_psnr": hvi_psnr,
                "ccm_improvement_percent": ccm_imp,
                "hvi_improvement_percent": hvi_imp,
                "ccm_time_ms": ccm_time_ms,
                "hvi_time_ms": hvi_time_ms,
                "winner_by_mae": winner,
            }
        )

    def avg(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    summary = {
        "n_images": len(rows),
        "avg_baseline_mae": avg("baseline_mae"),
        "avg_ccm_mae": avg("ccm_mae"),
        "avg_hvi_mae": avg("hvi_mae"),
        "avg_baseline_psnr": avg("baseline_psnr"),
        "avg_ccm_psnr": avg("ccm_psnr"),
        "avg_hvi_psnr": avg("hvi_psnr"),
        "avg_ccm_improvement_percent": avg("ccm_improvement_percent"),
        "avg_hvi_improvement_percent": avg("hvi_improvement_percent"),
        "avg_ccm_time_ms": avg("ccm_time_ms"),
        "avg_hvi_time_ms": avg("hvi_time_ms"),
        "ccm_win": ccm_win,
        "hvi_win": hvi_win,
        "tie": tie,
    }

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "baseline_mae",
                "ccm_mae",
                "hvi_mae",
                "baseline_psnr",
                "ccm_psnr",
                "hvi_psnr",
                "ccm_improvement_percent",
                "hvi_improvement_percent",
                "ccm_time_ms",
                "hvi_time_ms",
                "winner_by_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("# CCM vs HVI-CIDNet Eval15 Toplu Benchmark Raporu")
    lines.append("")
    lines.append("Bu rapor `eval15` paired setinin tamaminda olusturulmustur.")
    lines.append("")
    lines.append("## Kullanilan modeller")
    lines.append(f"- CCM: `{ccm_ckpt.relative_to(root)}`")
    lines.append(f"- HVI-CIDNet: `{hvi_ckpt.relative_to(root)}`")
    lines.append(f"- Device: `{device}`")
    lines.append("")
    lines.append("## Ozet Sonuclar")
    lines.append(f"- Goruntu sayisi: `{summary['n_images']}`")
    lines.append(
        f"- Ortalama MAE (dusuk daha iyi): Baseline `{summary['avg_baseline_mae']:.6f}` | CCM `{summary['avg_ccm_mae']:.6f}` | HVI-CIDNet `{summary['avg_hvi_mae']:.6f}`"
    )
    lines.append(
        f"- Ortalama PSNR (yuksek daha iyi): Baseline `{summary['avg_baseline_psnr']:.4f}` | CCM `{summary['avg_ccm_psnr']:.4f}` | HVI-CIDNet `{summary['avg_hvi_psnr']:.4f}`"
    )
    lines.append(
        f"- Ortalama MAE iyilesme (%): CCM `{summary['avg_ccm_improvement_percent']:.2f}%` | HVI-CIDNet `{summary['avg_hvi_improvement_percent']:.2f}%`"
    )
    lines.append(
        f"- Ortalama sure (ms/goruntu): CCM `{summary['avg_ccm_time_ms']:.2f}` | HVI-CIDNet `{summary['avg_hvi_time_ms']:.2f}`"
    )
    lines.append(
        f"- MAE bazli kazanan sayisi: CCM `{summary['ccm_win']}` | HVI-CIDNet `{summary['hvi_win']}` | Beraberlik `{summary['tie']}`"
    )
    lines.append("")
    lines.append("## Dosyalar")
    lines.append(f"- Ayrintili satir bazli sonuclar: `{csv_out.relative_to(root)}`")
    lines.append(f"- Bu rapor: `{md_out.relative_to(root)}`")

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Benchmark tamamlandi. N={len(rows)}")
    print(f"[OK] CSV: {csv_out}")
    print(f"[OK] Rapor: {md_out}")


if __name__ == "__main__":
    run_benchmark()
