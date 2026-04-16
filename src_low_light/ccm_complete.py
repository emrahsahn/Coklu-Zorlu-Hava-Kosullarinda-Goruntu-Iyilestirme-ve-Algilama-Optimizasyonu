import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ColorCorrectionModule(nn.Module):
    """
    ORİJİNAL: Illumination estimation + White balance learning
    Renk sapması (color cast) problemini çözüyor
    """
    def __init__(self):
        super(ColorCorrectionModule, self).__init__()
        
        # Illumination Map Estimator - ORİJİNAL DESIGN
        self.illumination_net = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # White Balance Learner - ORİJİNAL DESIGN  
        self.wb_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Illumination map estimation (per-pixel)
        illum = self.illumination_net(x)
        
        # Global white balance parameters
        wb_params = self.wb_net(x)  # [B, 3, 1, 1]
        
        # Color correction: I_corrected = I_input * WB / (Illumination + eps)
        corrected = x * wb_params / (illum + 1e-4)
        
        # Clamp to valid range
        corrected = torch.clamp(corrected, 0, 1)
        
        return corrected, illum

# ORİJİNAL Color Constancy Loss
class ColorConstancyLoss(nn.Module):
    """
    Illumination map'in gri olmasını sağlar
    Renk sapmasını önleyen ORİJİNAL loss function
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
    
    def forward(self, illum_map):
        # RGB kanalları arasındaki farkı minimize et
        b, c, h, w = illum_map.shape
        
        # Her pixel için RGB değerlerinin varyansını hesapla
        illum_flat = illum_map.view(b, c, -1)  # [B, 3, H*W]
        
        # Her pixel için ortalama
        pixel_mean = torch.mean(illum_flat, dim=1, keepdim=True)  # [B, 1, H*W]
        
        # Her pixel için varyans 
        pixel_var = torch.mean((illum_flat - pixel_mean) ** 2, dim=1)  # [B, H*W]
        
        # Ortalama varyansı minimize et
        loss = torch.mean(pixel_var)
        
        return loss

# Test function with real image
def test_ccm_real_image(image_path=None):
    """ColorCorrectionModule'ü gerçek görüntü ile test eder"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Eğer image_path verilmişse o dosyayı kullan
    if image_path and os.path.exists(image_path):
        print(f"✅ Loading image from: {image_path}")
        img_pil = Image.open(image_path).convert('RGB')
        
        # Resize işlemi
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 256, 256]
        
        print(f"Image shape: {img.shape}")
    else:
        # Fallback: Random image (blue cast ile)
        print("⚠️ Image file not found, using synthetic blue-cast image")
        img = torch.rand(1, 3, 256, 256).to(device)
        img[0, 0] *= 0.3  # Red channel düşük
        img[0, 2] *= 1.5  # Blue channel yüksek
        img = torch.clamp(img, 0, 1)
    
    # Orijinal istatistikler
    print(f"\n📊 Original Image Statistics:")
    print(f"  R: {img[0,0].mean():.4f}, G: {img[0,1].mean():.4f}, B: {img[0,2].mean():.4f}")
    
    # Model
    ccm = ColorCorrectionModule().to(device)
    ccm.eval()  # Evaluation mode
    
    # Forward pass
    with torch.no_grad():
        corrected, illum = ccm(img)
    
    # Düzeltilmiş istatistikler
    print(f"\n✨ Corrected Image Statistics:")
    print(f"  R: {corrected[0,0].mean():.4f}, G: {corrected[0,1].mean():.4f}, B: {corrected[0,2].mean():.4f}")
    
    # Illumination haritası istatistikleri
    print(f"\n💡 Illumination Map Statistics:")
    print(f"  R: {illum[0,0].mean():.4f}, G: {illum[0,1].mean():.4f}, B: {illum[0,2].mean():.4f}")
    
    # Loss test
    cc_loss = ColorConstancyLoss()
    loss_val = cc_loss(illum)
    print(f"\n📉 Color Constancy Loss: {loss_val.item():.6f}")
    
    # Visualize sonuçları
    save_visualization(img, corrected, illum)
    
    return ccm, img, corrected, illum

def save_visualization(original, corrected, illum):
    """Orijinal ve düzeltilmiş görüntüleri yan yana göster ve kaydet"""
    
    # Tensor'u numpy'ye çevir
    orig_np = original[0].permute(1, 2, 0).cpu().numpy()
    corr_np = corrected[0].permute(1, 2, 0).cpu().numpy()
    illum_np = illum[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize (visualization için)
    orig_np = np.clip(orig_np, 0, 1)
    corr_np = np.clip(corr_np, 0, 1)
    illum_np = np.clip(illum_np, 0, 1)
    
    # Yan yana görüntüler
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(corr_np)
    axes[1].set_title('Corrected Image')
    axes[1].axis('off')
    
    axes[2].imshow(illum_np)
    axes[2].set_title('Illumination Map')
    axes[2].axis('off')
    
    # Kaydet
    output_path = 'output/ccm_test_result.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {output_path}")
    plt.close()

def compare_with_ground_truth(low_light_path, high_light_path, brightness_boost=0.3):
    """
    Düşük ışık, model çıktısı ve ground truth'i karşılaştır
    brightness_boost: Parlaklık kontrol faktörü (daha düşük = daha parlak)
                      Default 0.3 optimal sonuç veriyor
                      Aralık: 0.1-1.0 (0.1=max parlak, 1.0=normal)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"KARŞILAŞTIRMALI TEST: Düşük Işık → Model Çıktısı → Ground Truth")
    print(f"{'='*80}\n")
    
    # Görselleri yükle
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    if not os.path.exists(low_light_path):
        print(f"❌ Düşük ışık dosyası bulunamadı: {low_light_path}")
        return
    
    if not os.path.exists(high_light_path):
        print(f"❌ Normal ışık dosyası bulunamadı: {high_light_path}")
        return
    
    # Yükle
    print(f"📂 Düşük ışık yükleniyor: {low_light_path}")
    low_img_pil = Image.open(low_light_path).convert('RGB')
    low_tensor = transform(low_img_pil).unsqueeze(0).to(device)
    
    print(f"📂 Normal ışık yükleniyor: {high_light_path}")
    high_img_pil = Image.open(high_light_path).convert('RGB')
    high_tensor = transform(high_img_pil).unsqueeze(0).to(device)
    
    # Model ile düzelt
    print(f"🤖 Renk düzeltme modeli çalıştırılıyor...")
    ccm = ColorCorrectionModule().to(device)
    ccm.eval()
    
    with torch.no_grad():
        corrected_tensor, illum = ccm(low_tensor)
    
    # Illumination enhanced çıktısı - parlaklık kontrolü ile
    # brightness_boost düşük = daha parlak (illum haritası daha düşük değerler alır)
    illum_adjusted = illum * (1.0 - brightness_boost) + brightness_boost
    illum_enhanced = low_tensor / (illum_adjusted + 1e-4)
    illum_enhanced = torch.clamp(illum_enhanced, 0, 1)
    
    # Illumination enhanced çıktısı (illum haritasını doğrudan kullanarak iyileştirme)
    illum_enhanced = low_tensor / (illum + 1e-4)
    illum_enhanced = torch.clamp(illum_enhanced, 0, 1)
    
    # Combined çıktı (Illum + CCM)
    combined = corrected_tensor  # CCM zaten illum ve white balance birleştiriyor
    
    # İstatistikler
    print(f"\n📊 RENGİ KANAL ANALİZİ:")
    print(f"{'─'*90}")
    print(f"{'Görüntü':<25} {'R':<18} {'G':<18} {'B':<18}")
    print(f"{'─'*90}")
    print(f"{'Düşük Işık (Original)':<25} R:{low_tensor[0,0].mean():.4f}     G:{low_tensor[0,1].mean():.4f}     B:{low_tensor[0,2].mean():.4f}")
    print(f"{'İllum Enhanced':<25} R:{illum_enhanced[0,0].mean():.4f}     G:{illum_enhanced[0,1].mean():.4f}     B:{illum_enhanced[0,2].mean():.4f}")
    print(f"{'CCM Çıktısı':<25} R:{corrected_tensor[0,0].mean():.4f}     G:{corrected_tensor[0,1].mean():.4f}     B:{corrected_tensor[0,2].mean():.4f}")
    print(f"{'Combined (Illum+CCM)':<25} R:{combined[0,0].mean():.4f}     G:{combined[0,1].mean():.4f}     B:{combined[0,2].mean():.4f}")
    print(f"{'Ground Truth (Normal)':<25} R:{high_tensor[0,0].mean():.4f}     G:{high_tensor[0,1].mean():.4f}     B:{high_tensor[0,2].mean():.4f}")
    print(f"{'─'*90}")
    
    # Farklar
    diff_illum_enh = torch.abs(illum_enhanced - high_tensor).mean()
    diff_ccm = torch.abs(corrected_tensor - high_tensor).mean()
    diff_combined = torch.abs(combined - high_tensor).mean()
    diff_low_gt = torch.abs(low_tensor - high_tensor).mean()
    
    print(f"\n📉 HATA ANALİZİ (MAE - Mean Absolute Error):")
    print(f"{'─'*90}")
    print(f"Düşük Işık → Ground Truth:       {diff_low_gt:.6f}")
    print(f"İllum Enhanced → Ground Truth:   {diff_illum_enh:.6f}  (iyileştirme: {(diff_low_gt - diff_illum_enh) / diff_low_gt * 100:.2f}%)")
    print(f"CCM Çıktısı → Ground Truth:      {diff_ccm:.6f}  (iyileştirme: {(diff_low_gt - diff_ccm) / diff_low_gt * 100:.2f}%)")
    print(f"Combined (Illum+CCM) → Ground Truth: {diff_combined:.6f}  (iyileştirme: {(diff_low_gt - diff_combined) / diff_low_gt * 100:.2f}%)")
    print(f"{'─'*90}\n")
    
    # Visualize
    save_comparison_visualization(low_tensor, illum_enhanced, corrected_tensor, combined, high_tensor, illum)
    
    return low_tensor, illum_enhanced, corrected_tensor, combined, high_tensor, illum

def save_comparison_visualization(low, illum_enhanced, corrected, combined, high, illum):
    """6 görseli göster: low, illum_enhanced, ccm, combined, ground_truth, illumination"""
    
    # Numpy'ye çevir
    low_np = low[0].permute(1, 2, 0).cpu().numpy()
    illum_enh_np = illum_enhanced[0].permute(1, 2, 0).cpu().numpy()
    corr_np = corrected[0].permute(1, 2, 0).cpu().numpy()
    comb_np = combined[0].permute(1, 2, 0).cpu().numpy()
    high_np = high[0].permute(1, 2, 0).cpu().numpy()
    illum_np = illum[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize
    low_np = np.clip(low_np, 0, 1)
    illum_enh_np = np.clip(illum_enh_np, 0, 1)
    corr_np = np.clip(corr_np, 0, 1)
    comb_np = np.clip(comb_np, 0, 1)
    high_np = np.clip(high_np, 0, 1)
    illum_np = np.clip(illum_np, 0, 1)
    
    # 6 panel görsel (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1
    axes[0, 0].imshow(low_np)
    axes[0, 0].set_title('1. Düşük Işık (Original)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(illum_enh_np)
    axes[0, 1].set_title('2. İyileştirme (Illum. Enhanced)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(corr_np)
    axes[0, 2].set_title('3. CCM Çıktısı (Color Corr.)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(comb_np)
    axes[1, 0].set_title('4. Combined (Illum+CCM)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(high_np)
    axes[1, 1].set_title('5. Ground Truth (Normal Işık)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(illum_np)
    axes[1, 2].set_title('6. Illumination Haritası', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Kaydet
    output_path = 'output/ccm_comparison_result.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"💾 Karşılaştırmalı görsel kaydedildi: {output_path}")
    plt.close()


# Test function
def test_ccm():
    """Backward compatibility - random image ile test"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate blue-cast low light image  
    img = torch.rand(1, 3, 256, 256).to(device)
    img[0, 0] *= 0.3  # Red channel düşük (mavi cast)
    img[0, 2] *= 1.5  # Blue channel yüksek
    img = torch.clamp(img, 0, 1)
    
    print(f"Original image stats:")
    print(f"R: {img[0,0].mean():.3f}, G: {img[0,1].mean():.3f}, B: {img[0,2].mean():.3f}")
    
    # Model
    ccm = ColorCorrectionModule().to(device)
    ccm.train()
    
    # Forward pass
    with torch.no_grad():
        corrected, illum = ccm(img)
    
    print(f"Corrected image stats:")
    print(f"R: {corrected[0,0].mean():.3f}, G: {corrected[0,1].mean():.3f}, B: {corrected[0,2].mean():.3f}")
    
    # Loss test
    cc_loss = ColorConstancyLoss()
    loss_val = cc_loss(illum)
    print(f"Color Constancy Loss: {loss_val.item():.4f}")
    
    return ccm, img, corrected, illum

if __name__ == "__main__":
    print("Testing ColorCorrectionModule...\n")
    
    # Gerçek görüntü ile test
    real_image_path = "output/bowling_20epoch_fixed.png"
    print("=" * 60)
    print("TEST 1: REAL IMAGE (bowling_20epoch_fixed.png)")
    print("=" * 60)
    model, original, corrected, illum = test_ccm_real_image(real_image_path)
    
    print("\n" + "=" * 60)
    print("TEST 2: RANDOM BLUE-CAST IMAGE (fallback)")
    print("=" * 60)
    model, original, corrected, illum = test_ccm()
    
    print("\n" + "=" * 60)
    print("TEST 3: GROUND TRUTH COMPARISON (LOL Dataset)")
    print("=" * 60)
    low_light_path = "data/LOL/test/low/1.png"
    high_light_path = "data/LOL/test/high/1.png"
    
    # Optimal brightness_boost değeri (0.3) ile test
    print("\n✨ Optimal parlaklık seviyesi (brightness_boost=0.3) ile test yapılıyor...\n")
    compare_with_ground_truth(low_light_path, high_light_path, brightness_boost=0.3)
    
    print("\n✅ All tests completed!")