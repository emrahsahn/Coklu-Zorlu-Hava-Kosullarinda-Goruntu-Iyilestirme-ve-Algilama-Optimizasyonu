"""
CCM Test Script
Düşük ışık görüntü iyileştirme test ve karşılaştırma fonksiyonları

Kullanım:
    python src/ccm_complete2.py                          # Varsayılan test
    python src/ccm_complete2.py --image data/test.png    # Belirli bir görüntü
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import argparse

# Model import
from ccm_model import (
    CombinedEnhancementModule,
    ColorCorrectionModule,
    ColorConstancyLoss,
    check_device_info,
    load_trained_model
)


def print_channel_analysis(name, tensor):
    """Kanal detaylı analiz ekrana yazdır"""
    r_mean = tensor[0, 0].mean().item()
    g_mean = tensor[0, 1].mean().item()
    b_mean = tensor[0, 2].mean().item()
    
    r_max = tensor[0, 0].max().item()
    g_max = tensor[0, 1].max().item()
    b_max = tensor[0, 2].max().item()
    
    r_std = tensor[0, 0].std().item()
    g_std = tensor[0, 1].std().item()
    b_std = tensor[0, 2].std().item()
    
    print(f"\n[STATS] {name}:")
    print(f"  R -> Mean: {r_mean:.4f}, Max: {r_max:.4f}, Std: {r_std:.4f}")
    print(f"  G -> Mean: {g_mean:.4f}, Max: {g_max:.4f}, Std: {g_std:.4f} {'[HIGH]' if g_mean > 0.45 else ''}")
    print(f"  B -> Mean: {b_mean:.4f}, Max: {b_max:.4f}, Std: {b_std:.4f}")
    print(f"  RGB Farki: R-G={r_mean-g_mean:.4f}, G-B={g_mean-b_mean:.4f}")


def save_detailed_comparison(low, ccm_out, combined_out, high, enhanced_illum, illum_map, output_path='output/ccm_detailed_comparison.png'):
    """6 panellik detaylı karşılaştırma"""
    
    # Numpy'ye çevir
    low_np = low[0].permute(1, 2, 0).cpu().numpy()
    ccm_np = ccm_out[0].permute(1, 2, 0).cpu().numpy()
    combined_np = combined_out[0].permute(1, 2, 0).cpu().numpy()
    high_np = high[0].permute(1, 2, 0).cpu().numpy()
    enhanced_np = enhanced_illum[0].permute(1, 2, 0).cpu().numpy()
    illum_np = illum_map[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize
    low_np = np.clip(low_np, 0, 1)
    ccm_np = np.clip(ccm_np, 0, 1)
    combined_np = np.clip(combined_np, 0, 1)
    high_np = np.clip(high_np, 0, 1)
    enhanced_np = np.clip(enhanced_np, 0, 1)
    illum_np = np.clip(illum_np, 0, 1)
    
    # 3x2 grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1
    axes[0, 0].imshow(low_np)
    axes[0, 0].set_title('1. Dusuk Isik (Original)', fontsize=12, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced_np)
    axes[0, 1].set_title('2. Iyilestirme (Illum. Enhanced)', fontsize=12, fontweight='bold', color='orange')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ccm_np)
    axes[0, 2].set_title('3. CCM Ciktisi (Color Corr.)', fontsize=12, fontweight='bold', color='blue')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(combined_np)
    axes[1, 0].set_title('4. Combined (Illum+CCM)', fontsize=12, fontweight='bold', color='green')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(high_np)
    axes[1, 1].set_title('5. Ground Truth (Normal)', fontsize=12, fontweight='bold', color='darkgreen')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(illum_np)
    axes[1, 2].set_title('6. Illumination Haritasi', fontsize=12, fontweight='bold', color='purple')
    axes[1, 2].axis('off')
    
    # Kaydet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\n[SAVED] Detayli karsilastirma kaydedildi: {output_path}")
    plt.close()


def compare_with_ground_truth(low_light_path, high_light_path, checkpoint_path=None, out_dir='output/ccm_tests'):
    """
    Düşük ışık, model çıktısı ve ground truth'i karşılaştır
    
    Args:
        low_light_path: Düşük ışık görüntü yolu
        high_light_path: Normal ışık (ground truth) görüntü yolu
        checkpoint_path: Eğitilmiş model checkpoint yolu (opsiyonel)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"KARSILASTIRMALI TEST: Dusuk Isik -> Model Ciktisi -> Ground Truth")
    print(f"{'='*80}\n")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Check files
    if not os.path.exists(low_light_path):
        print(f"[ERROR] Dusuk isik dosyasi bulunamadi: {low_light_path}")
        return
    
    if not os.path.exists(high_light_path):
        print(f"[ERROR] Normal isik dosyasi bulunamadi: {high_light_path}")
        return
    
    # Load images
    print(f"[LOADING] Dusuk isik yukleniyor: {low_light_path}")
    low_img_pil = Image.open(low_light_path).convert('RGB')
    low_tensor = transform(low_img_pil).unsqueeze(0).to(device)
    
    print(f"[LOADING] Normal isik yukleniyor: {high_light_path}")
    high_img_pil = Image.open(high_light_path).convert('RGB')
    high_tensor = transform(high_img_pil).unsqueeze(0).to(device)
    
    # Test 1: Sadece ColorCorrectionModule
    print(f"\n[TEST 1] Sadece ColorCorrectionModule")
    print(f"{'-'*80}")
    ccm = ColorCorrectionModule().to(device)
    ccm.eval()
    
    with torch.no_grad():
        corrected_ccm, illum_ccm = ccm(low_tensor)
    
    diff_ccm = torch.abs(corrected_ccm - high_tensor).mean()
    
    # Test 2: CombinedEnhancementModule
    print(f"\n[TEST 2] CombinedEnhancementModule (Illumination + Color Correction)")
    print(f"{'-'*80}")
    
    # Load trained model if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        combined = load_trained_model(checkpoint_path, device)
        print(f"[INFO] Egitilmis model kullaniliyor")
    else:
        combined = CombinedEnhancementModule().to(device)
        if checkpoint_path:
            print(f"[WARNING] Checkpoint bulunamadi: {checkpoint_path}")
        print(f"[INFO] Egtimsiz model kullaniliyor")
    
    combined.eval()
    
    with torch.no_grad():
        corrected_combined, enhanced_illum, gamma_curves, intensity_scale, illum_map = combined(low_tensor)
    
    diff_combined = torch.abs(corrected_combined - high_tensor).mean()
    
    # Channel analysis
    print(f"\n[ANALYSIS] RENK KANAL ANALIZI:")
    print(f"{'─'*80}")
    print_channel_analysis("Dusuk Isik (Original)", low_tensor)
    print_channel_analysis("CCM Ciktisi", corrected_ccm)
    print_channel_analysis("Enhanced Illumination", enhanced_illum)
    print_channel_analysis("Combined Ciktisi", corrected_combined)
    print_channel_analysis("Ground Truth (Normal)", high_tensor)
    print(f"{'─'*80}")
    
    # Error analysis
    diff_low_gt = torch.abs(low_tensor - high_tensor).mean()
    improvement_ccm = (diff_low_gt - diff_ccm) / diff_low_gt * 100
    improvement_combined = (diff_low_gt - diff_combined) / diff_low_gt * 100
    
    print(f"\n[ERROR] HATA ANALIZI (MAE - Mean Absolute Error):")
    print(f"{'─'*80}")
    print(f"Dusuk Isik -> Ground Truth:         {diff_low_gt:.6f} (baseline)")
    print(f"CCM Ciktisi -> Ground Truth:        {diff_ccm:.6f} ({improvement_ccm:+.2f}%)")
    print(f"Combined Ciktisi -> Ground Truth:   {diff_combined:.6f} ({improvement_combined:+.2f}%)")
    print(f"{'─'*80}\n")
    
    if improvement_combined > improvement_ccm:
        print(f"[SUCCESS] Combined model DAHA IYI! ({improvement_combined - improvement_ccm:.2f}% fark)")
    else:
        print(f"[INFO] CCM daha iyi sonuc veriyor")
    
    # Target check
    if improvement_combined >= 83:
        print(f"[TARGET] Hedef basariya ulasildi! {improvement_combined:.2f}% >= 83%")
    else:
        print(f"[TARGET] Hedef: 83%, Mevcut: {improvement_combined:.2f}%")
    
    # Save visualization
    filename = os.path.basename(low_light_path)
    basename, _ = os.path.splitext(filename)
    output_path = os.path.join(out_dir, f"{basename}_detailed_comparison.png")
    save_detailed_comparison(low_tensor, corrected_ccm, corrected_combined, high_tensor, enhanced_illum, illum_map, output_path=output_path)
    
    return {
        'low': low_tensor,
        'ccm_out': corrected_ccm,
        'combined_out': corrected_combined,
        'high': high_tensor,
        'improvement_ccm': improvement_ccm.item(),
        'improvement_combined': improvement_combined.item()
    }


def test_single_image(image_path, checkpoint_path=None, out_dir='output/ccm_tests'):
    """Tek bir görüntü üzerinde test yap"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"TEKIL GORUNTU TESTI")
    print(f"{'='*80}\n")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"[ERROR] Goruntu bulunamadi: {image_path}")
        return
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    print(f"[LOADING] Goruntu yuklendi: {image_path}")
    print(f"[INFO] Shape: {img_tensor.shape}")
    
    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_trained_model(checkpoint_path, device)
    else:
        model = CombinedEnhancementModule().to(device)
        print(f"[INFO] Egtimsiz model kullaniliyor")
    
    model.eval()
    
    # Process
    with torch.no_grad():
        corrected, enhanced, gamma, intensity, illum = model(img_tensor)
    
    # Stats
    print_channel_analysis("Original", img_tensor)
    print_channel_analysis("Enhanced", enhanced)
    print_channel_analysis("Corrected", corrected)
    
    # Save
    filename = os.path.basename(image_path)
    basename, _ = os.path.splitext(filename)
    output_path = os.path.join(out_dir, f"{basename}_single_test.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    orig_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    enh_np = enhanced[0].permute(1, 2, 0).cpu().numpy()
    corr_np = corrected[0].permute(1, 2, 0).cpu().numpy()
    
    axes[0].imshow(np.clip(orig_np, 0, 1))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(enh_np, 0, 1))
    axes[1].set_title('Enhanced')
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(corr_np, 0, 1))
    axes[2].set_title('Corrected')
    axes[2].axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\n[SAVED] Sonuc kaydedildi: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='CCM Test Script')
    parser.add_argument('--low', type=str, default='/data/lol_dataset/eval15/low/79.png',
                       help='Dusuk isik goruntu yolu')
    parser.add_argument('--high', type=str, default='/data/lol_dataset/eval15/high/79.png',
                       help='Normal isik (ground truth) goruntu yolu')
    parser.add_argument('--image', type=str, default=None,
                       help='Tekil goruntu testi icin goruntu yolu')
    parser.add_argument('--checkpoint', type=str, default='/src/checkpoints/ccm/best_combined_model.pth',
                       help='Egitilmis model checkpoint yolu')
    parser.add_argument('--out_dir', type=str, default='/output/ccm_tests',
                       help='Cikti klasoru')
    args = parser.parse_args()
    
    print("="*80)
    print("CCM - Combined Enhancement Module Test")
    print("="*80)
    
    # Device info
    device = check_device_info()
    
    if args.image:
        # Single image test
        test_single_image(args.image, args.checkpoint, args.out_dir)
    else:
        # Ground truth comparison
        compare_with_ground_truth(args.low, args.high, args.checkpoint, args.out_dir)
    
    print("\n[COMPLETE] Test tamamlandi!")


if __name__ == "__main__":
    main()
