import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    return (10 * torch.log10(1.0 / mse)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCM vs HVI-CIDNet comparison on one paired sample")
    parser.add_argument("--image_id", type=str, default="79", help="Image id/name without extension, e.g. 79")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    low_path = root / "data" / "lol_dataset" / "eval15" / "low" / f"{args.image_id}.png"
    gt_path = root / "data" / "lol_dataset" / "eval15" / "high" / f"{args.image_id}.png"
    ccm_ckpt = root / "src_low_light" / "checkpoints" / "ccm" / "best_combined_model.pth"
    hvi_ckpt = root / "HVI-CIDNet" / "weights" / "train" / "epoch_100.pth"

    sys.path.append(str(root / "src_low_light" / "CCM Combined"))
    from ccm_model import load_trained_model  # type: ignore

    sys.path.append(str(root / "HVI-CIDNet"))
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

    baseline_mae = torch.mean(torch.abs(low - gt)).item()
    ccm_mae = torch.mean(torch.abs(ccm_out - gt)).item()
    hvi_mae = torch.mean(torch.abs(hvi_out - gt)).item()

    print(f"device={device}")
    print(f"baseline_mae={baseline_mae:.6f}")
    print(f"ccm_mae={ccm_mae:.6f}")
    print(f"hvi_mae={hvi_mae:.6f}")
    print(f"ccm_improvement={(baseline_mae - ccm_mae) / (baseline_mae + 1e-8) * 100:.2f}%")
    print(f"hvi_improvement={(baseline_mae - hvi_mae) / (baseline_mae + 1e-8) * 100:.2f}%")
    print(f"baseline_psnr={psnr(low, gt):.4f}")
    print(f"ccm_psnr={psnr(ccm_out, gt):.4f}")
    print(f"hvi_psnr={psnr(hvi_out, gt):.4f}")


if __name__ == "__main__":
    main()
