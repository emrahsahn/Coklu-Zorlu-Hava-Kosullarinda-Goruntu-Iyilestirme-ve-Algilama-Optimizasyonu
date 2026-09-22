"""FFA hızlı test: FFA-Net klasöründeki görseller + SOTS GT (varsa)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FFA = ROOT / "FFA-Net" / "FFA-Net"
SOTS = FFA / "data" / "RESIDE" / "SOTS"
sys.path.insert(0, str(ROOT / "src"))
from metrics import calculate_psnr, calculate_ssim  # noqa: E402
from model_loader import FFANetWrapper  # noqa: E402


def clear_for(hazy: Path, split: str) -> Path | None:
    base = hazy.stem.split("_")[0]
    d = SOTS / split / "clear"
    if not d.is_dir():
        d = SOTS / split / "gt"
    for ext in (".png", ".jpg"):
        p = d / f"{base}{ext}"
        if p.is_file():
            return p
    return None


def run(hazy: Path, task: str, split: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = FFANetWrapper(task=task, device=device).load()
    to_t = transforms.ToTensor()
    hazy_pil = Image.open(hazy).convert("RGB")
    with torch.no_grad():
        out = w.restore(to_t(hazy_pil).unsqueeze(0).to(device))
    h = np.asarray(hazy_pil, np.uint8)
    r = (out.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    cp = clear_for(hazy, split)
    if not cp:
        print(f"{hazy.name}: GT yok")
        return
    c = np.asarray(Image.open(cp).convert("RGB"), np.uint8)
    print(
        f"{hazy.name} ({task}): "
        f"PSNR sisli={calculate_psnr(h,c):.2f} -> FFA={calculate_psnr(r,c):.2f} dB | "
        f"SSIM sisli={calculate_ssim(h,c):.4f} -> FFA={calculate_ssim(r,c):.4f}"
    )


def main() -> None:
    run(FFA / "fig" / "1400_2.png", "its", "indoor")
    run(FFA / "fig" / "0099_0.9_0.16.jpg", "ots", "outdoor")
    run(FFA / "samples" / "043.png", "ots", "outdoor")


if __name__ == "__main__":
    main()
