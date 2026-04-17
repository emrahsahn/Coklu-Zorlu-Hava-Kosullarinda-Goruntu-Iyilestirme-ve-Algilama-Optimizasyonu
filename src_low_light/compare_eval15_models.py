"""
Fair comparison on the same LOL eval split (eval15) used by CCM training.

Metrics (RGB, tensors in [0,1]): mean MAE vs GT, mean PSNR vs GT, baseline MAE (low vs GT).

Example:
  python src_low_light/compare_eval15_models.py ^
    --dataset data/lol_dataset ^
    --cidnet_ckpt src_low_light/checkpoints/cidnet/best_model.pth ^
    --ccm_ckpt src_low_light/checkpoints/ccm/best_combined_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "CIDNet"))
sys.path.insert(0, str(_ROOT / "CCM Combined"))

from app import CIDNetPipeline, LOLDataset  # noqa: E402
from ccm_model import load_trained_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compare CIDNet vs CCM on eval15")
    parser.add_argument("--dataset", type=str, default="data/lol_dataset", help="LOL root (our485 + eval15)")
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Resize for eval (use same value you trained CIDNet with, often 64; CCM can use 256).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--cidnet_ckpt",
        type=str,
        default=str(_ROOT / "checkpoints" / "cidnet" / "best_model.pth"),
        help="CIDNet checkpoint (best_model.pth). If missing, CIDNet is skipped.",
    )
    parser.add_argument(
        "--ccm_ckpt",
        type=str,
        default=str(_ROOT / "checkpoints" / "ccm" / "best_combined_model.pth"),
        help="CCM Combined checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    val_ds = LOLDataset(args.dataset, subset="test", image_size=args.image_size)
    if len(val_ds) == 0:
        print("[ERROR] Validation set is empty. Check dataset path and folder layout.")
        return 1

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cidnet = None
    use_cidnet = bool(args.cidnet_ckpt) and os.path.isfile(args.cidnet_ckpt)
    if use_cidnet:
        cidnet = CIDNetPipeline(device=str(device))
        cidnet.load_checkpoint(args.cidnet_ckpt)
        cidnet.eval_mode()
        print(f"[CIDNet] Loaded {args.cidnet_ckpt}")
    else:
        print("[CIDNet] Skipped (checkpoint missing or path empty).")

    ccm = None
    use_ccm = bool(args.ccm_ckpt) and os.path.isfile(args.ccm_ckpt)
    if use_ccm:
        ccm = load_trained_model(args.ccm_ckpt, device)
        ccm.eval()
        print(f"[CCM] Loaded {args.ccm_ckpt}")
    else:
        print("[CCM] Skipped (checkpoint missing).")

    if not use_cidnet and not use_ccm:
        print("[ERROR] No model checkpoint available.")
        return 1

    sum_low_mae = 0.0
    sum_cid_mae = sum_cid_psnr = 0.0
    sum_ccm_mae = sum_ccm_psnr = 0.0
    n_img = 0

    with torch.no_grad():
        for batch in loader:
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)
            b = low.size(0)
            n_img += b

            mae_low = torch.mean(torch.abs(low - high), dim=(1, 2, 3))
            sum_low_mae += mae_low.sum().item()

            if use_cidnet and cidnet is not None:
                out_c = cidnet.forward(low, gamma=1.0).clamp(0, 1)
                mae_c = torch.mean(torch.abs(out_c - high), dim=(1, 2, 3))
                sum_cid_mae += mae_c.sum().item()
                mse_c = torch.mean((out_c - high) ** 2, dim=(1, 2, 3))
                sum_cid_psnr += torch.sum(10.0 * torch.log10(1.0 / (mse_c + 1e-10))).item()

            if use_ccm and ccm is not None:
                corrected, *_ = ccm(low)
                out_m = corrected.clamp(0, 1)
                mae_m = torch.mean(torch.abs(out_m - high), dim=(1, 2, 3))
                sum_ccm_mae += mae_m.sum().item()
                mse_m = torch.mean((out_m - high) ** 2, dim=(1, 2, 3))
                sum_ccm_psnr += torch.sum(10.0 * torch.log10(1.0 / (mse_m + 1e-10))).item()

    print("\n" + "=" * 72)
    print("EVAL15 SUMMARY (per-image mean over pixels, then mean over images; RGB [0,1])")
    print("=" * 72)
    print(f"  Images: {n_img}")
    print(f"  Baseline MAE (low -> GT):     {sum_low_mae / n_img:.6f}")
    if use_cidnet:
        print(f"  CIDNet MAE (out -> GT):       {sum_cid_mae / n_img:.6f}")
        print(f"  CIDNet PSNR (dB):             {sum_cid_psnr / n_img:.4f}")
    if use_ccm:
        print(f"  CCM MAE (out -> GT):          {sum_ccm_mae / n_img:.6f}")
        print(f"  CCM PSNR (dB):                {sum_ccm_psnr / n_img:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
