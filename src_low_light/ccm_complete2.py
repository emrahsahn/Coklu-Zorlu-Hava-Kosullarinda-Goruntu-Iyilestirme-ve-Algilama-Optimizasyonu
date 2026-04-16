import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

def check_device_info():
    """GPU/CPU bilgisini kontrol et ve yazdır"""
    print("\n" + "="*80)
    print("[DEVICE INFO] - Donanım Bilgisi")
    print("="*80)
    
    # CUDA kontrol
    print(f"[INFO] PyTorch versiyonu: {torch.__version__}")
    print(f"[INFO] CUDA kullanılabilir mi: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU sayısı: {torch.cuda.device_count()}")
        print(f"[INFO] Aktif GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = torch.device('cuda')
        device_name = "GPU (CUDA)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
        print("[WARNING] GPU bulunamadı! CPU kullanılacak (YAVAŞ)")
    
    print(f"\n[SELECTED] Kullanılacak Device: {device_name}")
    print("="*80 + "\n")
    
    return device

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class IlluminationEnhancementModule(nn.Module):
    """
    Düşük ışık görüntülerini parlaklaştırır
    Adaptive brightness ve contrast enhancement
    Optimized for better RGB balance
    """
    def __init__(self):
        super(IlluminationEnhancementModule, self).__init__()
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
        )
        
        # Global intensity boost (channel-wise)
        # Her kanal için ayrı boost faktörü
        self.channel_boost = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
            nn.Sigmoid()
        )
        
        # Gamma correction per-pixel (daha ılımlı)
        self.gamma_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()  # [0, 1]
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_net(x)
        
        # Channel-wise intensity boost [1.0, 4.0]
        boost = self.channel_boost(x)  # [0, 1]
        boost = 1.0 + boost * 3.0  # [1.0, 4.0]
        
        # Gamma curves [0.3, 0.9] - gamma < 1 means brighten
        gamma_raw = self.gamma_net(features)  # [0, 1]
        gamma_curves = 0.3 + gamma_raw * 0.6  # [0.3, 0.9]
        
        # Apply gamma: I_out = I_in ^ gamma
        # Smaller gamma = more brightening
        enhanced = torch.pow(x + 1e-4, gamma_curves)
        
        # Apply channel-wise boost
        enhanced = enhanced * boost
        
        # Clamp to valid range [0, 1]
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced, gamma_curves, boost

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

class CombinedEnhancementModule(nn.Module):
    """
    Illumination Enhancement + Color Correction = Tam çözüm!
    Adım 1: Düşük ışığı parlaklaştır
    Adım 2: Renk sapmasını düzelt
    """
    def __init__(self):
        super(CombinedEnhancementModule, self).__init__()
        self.illumination_enhancement = IlluminationEnhancementModule()
        self.color_correction = ColorCorrectionModule()
    
    def forward(self, x):
        # Step 1: Enhance illumination (darken lighter)
        enhanced_illum, gamma_curves, intensity_scale = self.illumination_enhancement(x)
        
        # Step 2: Correct color cast
        corrected_color, illum_map = self.color_correction(enhanced_illum)
        
        return corrected_color, enhanced_illum, gamma_curves, intensity_scale, illum_map

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
    print(f"\n[SAVED] Visualization saved to: {output_path}")
    plt.close()

def compare_with_ground_truth(low_light_path, high_light_path):
    """
    Dusuk isik, model ciktisi ve ground truth'i karsilastir
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"KARSILASTIRMALI TEST: Dusuk Isik -> Model Ciktisi -> Ground Truth")
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
    
    # Test 2: CombinedEnhancementModule (Illumination + Color Correction)
    print(f"\n[TEST 2] CombinedEnhancementModule (Illumination + Color Correction)")
    print(f"{'-'*80}")
    combined = CombinedEnhancementModule().to(device)
    
    # EGITILMIS MODELI YUKLE
    ckpt_path = 'src/checkpoints/ccm/best_combined_model.pth'
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            combined.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Egitilmis model basariyla yuklendi: {ckpt_path}")
        except Exception as e:
            print(f"⚠️ Model dosyasi yuklenirken hata olustu: {e}")
    else:
        print(f"⚠️ Egitilmis model bulunamadi, sifir (untrained) model kullaniliyor!")
        
    combined.eval()
    
    with torch.no_grad():
        corrected_combined, enhanced_illum, gamma_curves, intensity_scale, illum_map = combined(low_tensor)
    
    diff_combined = torch.abs(corrected_combined - high_tensor).mean()
    
    # Karşılaştırma
    print(f"\n[ANALYSIS] RENK KANAL ANALIZI:")
    print(f"{'─'*80}")
    print_channel_analysis("Dusuk Isik (Original)", low_tensor)
    print_channel_analysis("CCM Ciktisi", corrected_ccm)
    print_channel_analysis("Enhanced Illumination", enhanced_illum)
    print_channel_analysis("Combined Ciktisi", corrected_combined)
    print_channel_analysis("Ground Truth (Normal)", high_tensor)
    print(f"{'─'*80}")
    
    # Hata analizi
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
    
    # Visualize
    save_detailed_comparison(low_tensor, corrected_ccm, corrected_combined, high_tensor, enhanced_illum, illum_map)
    
    return low_tensor, corrected_ccm, corrected_combined, high_tensor

def print_channel_analysis(name, tensor):
    """Kanal detayli analiz ekrana yazdir"""
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

def save_detailed_comparison(low, ccm_out, combined_out, high, enhanced_illum, illum_map):
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
    output_path = 'output/ccm_detailed_comparison_22.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\n[SAVED] Detayli karsilastirma kaydedildi: {output_path}")
    plt.close()

def save_comparison_visualization(low, corrected, high, illum):
    """4 görseli yan yana göster: low, corrected, ground_truth, illumination"""
    
    # Numpy'ye çevir
    low_np = low[0].permute(1, 2, 0).cpu().numpy()
    corr_np = corrected[0].permute(1, 2, 0).cpu().numpy()
    high_np = high[0].permute(1, 2, 0).cpu().numpy()
    illum_np = illum[0].permute(1, 2, 0).cpu().numpy()
    
    # Normalize
    low_np = np.clip(low_np, 0, 1)
    corr_np = np.clip(corr_np, 0, 1)
    high_np = np.clip(high_np, 0, 1)
    illum_np = np.clip(illum_np, 0, 1)
    
    # 4 panel görsel
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Row 1
    axes[0, 0].imshow(low_np)
    axes[0, 0].set_title('Düşük Işık (Original)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(corr_np)
    axes[0, 1].set_title('Model Çıktısı (CCM)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2
    axes[1, 0].imshow(high_np)
    axes[1, 0].set_title('Ground Truth (Normal Işık)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(illum_np)
    axes[1, 1].set_title('Illumination Haritası', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Kaydet
    output_path = 'output/ccm_comparison_result.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"[SAVED] Karsilastirmali gorsel kaydedildi: {output_path}")
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
    
    # NOT: Seed kaldırıldı - Eğitimli model kullanılacak
    # Eğitimsiz modelde seed gereksiz, eğitimli model tutarlı sonuç verecek
    
    # Device bilgisini göster
    device = check_device_info()
    
    # Test 1: Düşük ışık, model çıktısı ve ground truth karşılaştırması
    low_light_path = "data/lol_dataset/eval15/low/22.png"
    high_light_path = "data/lol_dataset/eval15/high/22.png"
    
    print("=" * 80)
    print("GERCEK VERI SETI ILE KARSILASTIRMALI TEST")
    print("=" * 80)
    compare_with_ground_truth(low_light_path, high_light_path)
    
    # print("\n" + "=" * 80)
    # print("BONUS TEST: bowling_20epoch_fixed.png")
    # print("=" * 80)
    # real_image_path = "output/bowling_20epoch_fixed.png"
    # test_ccm_real_image(real_image_path)
    
    print("\n[COMPLETE] Tum testler tamamlandi!")