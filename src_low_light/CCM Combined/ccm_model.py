"""
CCM Model - Color Correction Module
Düşük ışık görüntü iyileştirme için model sınıfları

Modüller:
- ConvBlock: Temel convolution bloğu
- IlluminationEnhancementModule: Parlaklık iyileştirme
- ColorCorrectionModule: Renk düzeltme
- CombinedEnhancementModule: İkisinin birleşimi
- Loss fonksiyonları
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class ConvBlock(nn.Module):
    """Temel Convolution bloğu: Conv2d + BatchNorm + ReLU"""
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
    Illumination estimation + White balance learning
    Renk sapması (color cast) problemini çözüyor
    """
    def __init__(self):
        super(ColorCorrectionModule, self).__init__()
        
        # Illumination Map Estimator
        self.illumination_net = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # White Balance Learner
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
        # Step 1: Enhance illumination
        enhanced_illum, gamma_curves, intensity_scale = self.illumination_enhancement(x)
        
        # Step 2: Correct color cast
        corrected_color, illum_map = self.color_correction(enhanced_illum)
        
        return corrected_color, enhanced_illum, gamma_curves, intensity_scale, illum_map


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ColorConstancyLoss(nn.Module):
    """
    Illumination map'in gri olmasını sağlar
    Renk sapmasını önleyen loss function
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


class CombinedLoss(nn.Module):
    """Combined loss for training: L1 + Color Constancy + SSIM"""
    def __init__(self, lambda_l1=1.0, lambda_cc=0.1, lambda_ssim=0.5):
        super(CombinedLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_cc = lambda_cc
        self.lambda_ssim = lambda_ssim
        
        self.l1_loss = nn.L1Loss()
        self.cc_loss = ColorConstancyLoss()
    
    def ssim_loss(self, pred, target, window_size=11):
        """Simple SSIM loss approximation"""
        return 1.0 - F.cosine_similarity(pred.view(pred.size(0), -1), 
                                          target.view(target.size(0), -1)).mean()
    
    def forward(self, pred, target, illum_map=None):
        # L1 Loss
        loss_l1 = self.l1_loss(pred, target)
        
        # SSIM Loss
        loss_ssim = self.ssim_loss(pred, target)
        
        # Color Constancy Loss (if illumination map provided)
        loss_cc = 0.0
        if illum_map is not None:
            loss_cc = self.cc_loss(illum_map)
        
        # Total loss
        total_loss = self.lambda_l1 * loss_l1 + self.lambda_ssim * loss_ssim + self.lambda_cc * loss_cc
        
        return total_loss, {
            'l1': loss_l1.item(),
            'ssim': loss_ssim.item(),
            'cc': loss_cc.item() if isinstance(loss_cc, torch.Tensor) else loss_cc,
            'total': total_loss.item()
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_CCM_CHECKPOINT_DIR = PROJECT_ROOT / 'src_low_light' / 'checkpoints' / 'ccm'


def _safe_torch_load(checkpoint_path, device):
    """
    PyTorch 2.6 defaults torch.load(..., weights_only=True), which breaks
    older training checkpoints that contain optimizer/state metadata.
    """
    return torch.load(checkpoint_path, map_location=device, weights_only=False)

def check_device_info():
    """GPU/CPU bilgisini kontrol et ve yazdır"""
    print("\n" + "="*80)
    print("[DEVICE INFO] - Donanim Bilgisi")
    print("="*80)
    
    print(f"[INFO] PyTorch versiyonu: {torch.__version__}")
    print(f"[INFO] CUDA kullanilabilir mi: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU sayisi: {torch.cuda.device_count()}")
        print(f"[INFO] Aktif GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = torch.device('cuda')
        device_name = "GPU (CUDA)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
        print("[WARNING] GPU bulunamadi! CPU kullanilacak (YAVAS)")
    
    print(f"\n[SELECTED] Kullanilacak Device: {device_name}")
    print("="*80 + "\n")
    
    return device


def load_trained_model(checkpoint_path, device):
    """Eğitilmiş CombinedEnhancementModule yükle"""
    import os
    model = CombinedEnhancementModule()
    
    if os.path.exists(checkpoint_path):
        checkpoint = _safe_torch_load(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', None)
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        print(f"[LOADED] CombinedModel loaded from {checkpoint_path}")
        print(f"[INFO]   Epoch: {epoch}, Val Loss: {val_loss_str}")
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print(f"[INFO] Using untrained CombinedModel")
    
    return model.to(device)


def load_illum_model(checkpoint_path, device):
    """Eğitilmiş IlluminationEnhancementModule yükle"""
    import os
    model = IlluminationEnhancementModule()
    
    if os.path.exists(checkpoint_path):
        checkpoint = _safe_torch_load(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', None)
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        print(f"[LOADED] IllumModel loaded from {checkpoint_path}")
        print(f"[INFO]   Epoch: {epoch}, Val Loss: {val_loss_str}")
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print(f"[INFO] Using untrained IllumModel")
    
    return model.to(device)


def load_ccm_model(checkpoint_path, device):
    """Eğitilmiş ColorCorrectionModule yükle"""
    import os
    model = ColorCorrectionModule()
    
    if os.path.exists(checkpoint_path):
        checkpoint = _safe_torch_load(checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', None)
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        print(f"[LOADED] CCMModel loaded from {checkpoint_path}")
        print(f"[INFO]   Epoch: {epoch}, Val Loss: {val_loss_str}")
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print(f"[INFO] Using untrained CCMModel")
    
    return model.to(device)


# Test
if __name__ == "__main__":
    device = check_device_info()
    
    # Model test
    print("[TEST] CombinedEnhancementModule")
    model = CombinedEnhancementModule().to(device)
    
    # Dummy input
    x = torch.rand(1, 3, 256, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        output, enhanced, gamma, intensity, illum = model(x)
    
    print(f"[OK] Input shape: {x.shape}")
    print(f"[OK] Output shape: {output.shape}")
    print(f"[OK] Enhanced shape: {enhanced.shape}")
    print(f"[OK] Illum map shape: {illum.shape}")
    
    # Loss test
    print("\n[TEST] CombinedLoss")
    criterion = CombinedLoss().to(device)
    target = torch.rand(1, 3, 256, 256).to(device)
    loss, loss_dict = criterion(output, target, illum)
    print(f"[OK] Total loss: {loss_dict['total']:.4f}")
    print(f"[OK] L1: {loss_dict['l1']:.4f}, SSIM: {loss_dict['ssim']:.4f}, CC: {loss_dict['cc']:.4f}")
    
    print("\n[COMPLETE] All model tests passed!")
