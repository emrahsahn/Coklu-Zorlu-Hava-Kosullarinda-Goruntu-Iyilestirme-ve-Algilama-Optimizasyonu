"""
CIDNet test script.

Usage:
    python "src_low_light/CIDNet/cidnet_test.py"
    python "src_low_light/CIDNet/cidnet_test.py" --image "path/to/image.png"
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from app import CIDNetPipeline


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_CIDNET_CHECKPOINT = str(PROJECT_ROOT / "src_low_light" / "checkpoints" / "cidnet" / "best_model.pth")


def check_device_info():
    """Print device information and return selected torch.device."""
    print("\n" + "=" * 80)
    print("[DEVICE INFO] - Donanim Bilgisi")
    print("=" * 80)
    print(f"[INFO] PyTorch versiyonu: {torch.__version__}")
    print(f"[INFO] CUDA kullanilabilir mi: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU sayisi: {torch.cuda.device_count()}")
        print(f"[INFO] Aktif GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = torch.device("cuda")
        device_name = "GPU (CUDA)"
    else:
        print("[WARNING] GPU bulunamadi! CPU kullanilacak.")
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"\n[SELECTED] Kullanilacak Device: {device_name}")
    print("=" * 80 + "\n")
    return device


def print_channel_analysis(name, tensor):
    """Print RGB statistics for a [1,3,H,W] tensor."""
    r_mean = tensor[0, 0].mean().item()
    g_mean = tensor[0, 1].mean().item()
    b_mean = tensor[0, 2].mean().item()
    r_std = tensor[0, 0].std().item()
    g_std = tensor[0, 1].std().item()
    b_std = tensor[0, 2].std().item()
    print(f"\n[STATS] {name}:")
    print(f"  R -> Mean: {r_mean:.4f}, Std: {r_std:.4f}")
    print(f"  G -> Mean: {g_mean:.4f}, Std: {g_std:.4f}")
    print(f"  B -> Mean: {b_mean:.4f}, Std: {b_std:.4f}")
    print(f"  RGB Farki: R-G={r_mean - g_mean:.4f}, G-B={g_mean - b_mean:.4f}")


def load_cidnet_pipeline(checkpoint_path, device, base_channels, num_heads):
    """Load CIDNet pipeline and optional checkpoint."""
    pipeline = CIDNetPipeline(device=str(device), base_channels=base_channels, num_heads=num_heads)
    if checkpoint_path and os.path.exists(checkpoint_path):
        pipeline.load_checkpoint(checkpoint_path)
        print(f"[INFO] Egitilmis model kullaniliyor: {checkpoint_path}")
    else:
        if checkpoint_path:
            print(f"[WARNING] Checkpoint bulunamadi: {checkpoint_path}")
        print("[INFO] Egitimsiz model kullaniliyor")
    pipeline.eval_mode()
    return pipeline


def save_single_visualization(input_tensor, output_tensor, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    inp = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    out = output_tensor[0].permute(1, 2, 0).cpu().numpy()

    axes[0].imshow(np.clip(inp, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(out, 0, 1))
    axes[1].set_title("CIDNet Output")
    axes[1].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Sonuc kaydedildi: {output_path}")


def save_comparison_visualization(low, output, high, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    low_np = low[0].permute(1, 2, 0).cpu().numpy()
    out_np = output[0].permute(1, 2, 0).cpu().numpy()
    high_np = high[0].permute(1, 2, 0).cpu().numpy()

    axes[0].imshow(np.clip(low_np, 0, 1))
    axes[0].set_title("Low-Light")
    axes[0].axis("off")

    axes[1].imshow(np.clip(out_np, 0, 1))
    axes[1].set_title("CIDNet Output")
    axes[1].axis("off")

    axes[2].imshow(np.clip(high_np, 0, 1))
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Karsilastirma kaydedildi: {output_path}")


def test_single_image(image_path, checkpoint_path, out_dir, image_size, gamma, device, base_channels, num_heads):
    if not os.path.exists(image_path):
        print(f"[ERROR] Goruntu bulunamadi: {image_path}")
        return

    print("\n" + "=" * 80)
    print("TEKIL GORUNTU TESTI (CIDNet)")
    print("=" * 80)
    print(f"[LOADING] Goruntu: {image_path}")

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    print(f"[INFO] Input shape: {image_tensor.shape}")

    pipeline = load_cidnet_pipeline(checkpoint_path, device, base_channels, num_heads)
    with torch.no_grad():
        output = pipeline.forward(image_tensor, gamma=gamma).clamp(0, 1)

    print_channel_analysis("Original", image_tensor)
    print_channel_analysis("CIDNet Output", output)

    basename = Path(image_path).stem
    output_path = os.path.join(out_dir, f"{basename}_cidnet_single_test.png")
    save_single_visualization(image_tensor, output, output_path)


def compare_with_ground_truth(low_path, high_path, checkpoint_path, out_dir, image_size, gamma, device, base_channels, num_heads):
    if not os.path.exists(low_path):
        print(f"[ERROR] Dusuk isik dosyasi bulunamadi: {low_path}")
        return
    if not os.path.exists(high_path):
        print(f"[ERROR] Ground truth dosyasi bulunamadi: {high_path}")
        return

    print("\n" + "=" * 80)
    print("KARSILASTIRMALI TEST (CIDNet): Low -> Output -> GT")
    print("=" * 80)

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    low_tensor = transform(Image.open(low_path).convert("RGB")).unsqueeze(0).to(device)
    high_tensor = transform(Image.open(high_path).convert("RGB")).unsqueeze(0).to(device)

    pipeline = load_cidnet_pipeline(checkpoint_path, device, base_channels, num_heads)
    with torch.no_grad():
        output = pipeline.forward(low_tensor, gamma=gamma).clamp(0, 1)

    diff_low_gt = torch.abs(low_tensor - high_tensor).mean()
    diff_out_gt = torch.abs(output - high_tensor).mean()
    improvement = (diff_low_gt - diff_out_gt) / (diff_low_gt + 1e-8) * 100.0

    print_channel_analysis("Low-Light", low_tensor)
    print_channel_analysis("CIDNet Output", output)
    print_channel_analysis("Ground Truth", high_tensor)

    print("\n[ERROR] MAE ANALIZI:")
    print("-" * 80)
    print(f"Low-Light -> GT:    {diff_low_gt:.6f} (baseline)")
    print(f"CIDNet -> GT:       {diff_out_gt:.6f} ({improvement:+.2f}%)")
    print("-" * 80)

    basename = Path(low_path).stem
    output_path = os.path.join(out_dir, f"{basename}_cidnet_detailed_comparison.png")
    save_comparison_visualization(low_tensor, output, high_tensor, output_path)


def main():
    parser = argparse.ArgumentParser(description="CIDNet Test Script")
    parser.add_argument("--low", type=str, default="data/lol_dataset/eval15/low/79.png", help="Dusuk isik goruntu yolu")
    parser.add_argument("--high", type=str, default="data/lol_dataset/eval15/high/79.png", help="Ground truth goruntu yolu")
    parser.add_argument("--image", type=str, default=None, help="Tekil goruntu testi icin goruntu yolu")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CIDNET_CHECKPOINT, help="CIDNet checkpoint yolu")
    parser.add_argument("--out_dir", type=str, default="output/cidnet_tests", help="Cikti klasoru")
    parser.add_argument("--image_size", type=int, default=64, help="Model giris boyutu")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction degeri")
    parser.add_argument("--base_channels", type=int, default=32, help="Model base_channels (checkpoint ile ayni olmali)")
    parser.add_argument("--num_heads", type=int, default=4, help="Model num_heads (checkpoint ile ayni olmali)")
    args = parser.parse_args()

    print("=" * 80)
    print("CIDNet Test")
    print("=" * 80)

    device = check_device_info()
    if args.image:
        test_single_image(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            out_dir=args.out_dir,
            image_size=args.image_size,
            gamma=args.gamma,
            device=device,
            base_channels=args.base_channels,
            num_heads=args.num_heads,
        )
    else:
        compare_with_ground_truth(
            low_path=args.low,
            high_path=args.high,
            checkpoint_path=args.checkpoint,
            out_dir=args.out_dir,
            image_size=args.image_size,
            gamma=args.gamma,
            device=device,
            base_channels=args.base_channels,
            num_heads=args.num_heads,
        )

    print("\n[COMPLETE] CIDNet test tamamlandi!")


if __name__ == "__main__":
    main()
