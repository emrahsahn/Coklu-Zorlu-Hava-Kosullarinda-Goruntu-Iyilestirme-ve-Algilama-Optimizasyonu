"""
Generate 4-panel thesis figure for LOL eval15 / 79.png:
(a) Low-light input, (b) Ground truth, (c) CCM, (d) HVI-CIDNet
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    return (10 * torch.log10(1.0 / (mse + 1e-12))).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image_id = "79"

    low_path = root / "data" / "lol_dataset" / "eval15" / "low" / f"{image_id}.png"
    gt_path = root / "data" / "lol_dataset" / "eval15" / "high" / f"{image_id}.png"
    ccm_ckpt = root / "src_low_light" / "checkpoints" / "ccm" / "best_combined_model.pth"
    hvi_ckpt = root / "HVI-CIDNet" / "weights" / "train" / "epoch_100.pth"
    out_path = root / "output" / "comparison_figures" / f"{image_id}_four_panel_comparison.png"

    sys.path.insert(0, str(root / "src_low_light" / "CCM Combined"))
    from ccm_model import load_trained_model  # type: ignore

    sys.path.insert(0, str(root / "HVI-CIDNet"))
    from net.CIDNet import CIDNet  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    low = to_tensor(Image.open(low_path).convert("RGB")).unsqueeze(0).to(device)
    gt = to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)

    ccm_model = load_trained_model(str(ccm_ckpt), device)
    ccm_model.eval()
    with torch.no_grad():
        ccm_out, _, _, _, _ = ccm_model(low)
        ccm_out = torch.clamp(ccm_out, 0, 1)

    hvi_model = CIDNet().to(device)
    hvi_model.load_state_dict(torch.load(hvi_ckpt, map_location=device))
    hvi_model.eval()
    hvi_model.trans.gated = True
    hvi_model.trans.gated2 = True

    _, _, h, w = low.shape
    factor = 8
    h_aligned = ((h + factor) // factor) * factor
    w_aligned = ((w + factor) // factor) * factor
    pad_h = h_aligned - h if h % factor != 0 else 0
    pad_w = w_aligned - w if w % factor != 0 else 0
    low_padded = F.pad(low, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        hvi_out = hvi_model(low_padded**1.0)
        hvi_out = torch.clamp(hvi_out, 0, 1)[:, :, :h, :w]

    baseline_mae = mae(low, gt)
    ccm_mae_v = mae(ccm_out, gt)
    hvi_mae_v = mae(hvi_out, gt)
    baseline_psnr = psnr(low, gt)
    ccm_psnr_v = psnr(ccm_out, gt)
    hvi_psnr_v = psnr(hvi_out, gt)
    ccm_imp = (baseline_mae - ccm_mae_v) / (baseline_mae + 1e-8) * 100.0
    hvi_imp = (baseline_mae - hvi_mae_v) / (baseline_mae + 1e-8) * 100.0

    panels = [
        (low, "(a) Düşük Işıklı Girdi", f"MAE={baseline_mae:.3f}  PSNR={baseline_psnr:.2f} dB"),
        (gt, "(b) Hedef (Ground Truth)", ""),
        (ccm_out, "(c) CCM Combined", f"MAE={ccm_mae_v:.3f}  PSNR={ccm_psnr_v:.2f} dB  İyileşme %{ccm_imp:.1f}"),
        (hvi_out, "(d) HVI-CIDNet", f"MAE={hvi_mae_v:.3f}  PSNR={hvi_psnr_v:.2f} dB  İyileşme %{hvi_imp:.1f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"LOL eval15 — {image_id}.png: Düşük Işık Restorasyon Karşılaştırması", fontsize=14, fontweight="bold")

    for ax, (tensor, title, metrics) in zip(axes.flat, panels):
        img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"{title}\n{metrics}" if metrics else title, fontsize=11)
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_path}")
    print(f"baseline_mae={baseline_mae:.6f} ccm_mae={ccm_mae_v:.6f} hvi_mae={hvi_mae_v:.6f}")
    print(f"baseline_psnr={baseline_psnr:.4f} ccm_psnr={ccm_psnr_v:.4f} hvi_psnr={hvi_psnr_v:.4f}")


if __name__ == "__main__":
    main()
