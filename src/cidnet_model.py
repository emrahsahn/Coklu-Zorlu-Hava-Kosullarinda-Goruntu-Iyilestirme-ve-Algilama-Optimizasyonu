"""
CIDNet: Color and Intensity Decoupling Network
Based on paper: https://arxiv.org/abs/2402.05809v3

Main architecture with dual-branch UNet and Lighten Cross-Attention (LCA) modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedding(nn.Module):
    """Feature embedding with depth-wise and group convolution"""
    def __init__(self, in_channels, out_channels):
        super(FeatureEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1 depth-wise
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     padding=1, groups=out_channels//4 if out_channels >= 4 else 1, bias=False),  # 3x3 group conv
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.conv(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block (CAB)
    Facilitates interaction between HV-branch and I-branch
    """
    def __init__(self, channels, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Feature embedding for Q, K, V
        self.query_embed = FeatureEmbedding(channels, channels)
        self.key_embed = FeatureEmbedding(channels, channels)
        self.value_embed = FeatureEmbedding(channels, channels)
        
        # Output projection
        self.output_proj = FeatureEmbedding(channels, channels)
        
    def forward(self, query_features, key_value_features):
        """
        Args:
            query_features: Features from one branch [B, C, H, W]
            key_value_features: Features from opposite branch [B, C, H, W]
        """
        B, C, H, W = query_features.shape
        
        # Generate Q, K, V
        Q = self.query_embed(query_features)  # [B, C, H, W]
        K = self.key_embed(key_value_features)  # [B, C, H, W]
        V = self.value_embed(key_value_features)  # [B, C, H, W]
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        K = K.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        V = V.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        
        # Attention: Q @ K^T / sqrt(head_dim)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # [B, heads, HW, head_dim]
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Output projection and residual
        out = self.output_proj(out) + query_features
        
        return out


class IntensityEnhanceLayer(nn.Module):
    """
    Intensity Enhance Layer (IEL)
    Decomposes features into illumination and reflectance
    Based on Retinex theory (Equation 11)
    """
    def __init__(self, channels):
        super(IntensityEnhanceLayer, self).__init__()
        self.illum_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Tanh()
        )
        self.reflect_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Tanh()
        )
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
    
    def forward(self, x):
        """
        Y_I = tanh(W_s * Y_I) + Y_I
        Y_R = tanh(W_s * Y_R) + Y_R
        Output = W_s((Y_I ⊙ Y_R))
        """
        Y_I = self.illum_conv(x) + x
        Y_R = self.reflect_conv(x) + x
        
        # Element-wise multiplication and fusion
        out = self.fusion_conv(Y_I * Y_R)
        
        return out + x  # Residual connection


class ColorDenoiseLayer(nn.Module):
    """
    Color Denoise Layer (CDL)
    Suppresses noise and color artifacts in HV branch
    """
    def __init__(self, channels):
        super(ColorDenoiseLayer, self).__init__()
        self.denoise_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.denoise_conv(x)


class LightenCrossAttention(nn.Module):
    """
    Lighten Cross-Attention (LCA) Module
    Dual-branch processing with cross-attention between HV and I branches
    """
    def __init__(self, channels, num_heads=4):
        super(LightenCrossAttention, self).__init__()
        
        # Cross Attention Blocks for both branches
        self.cab_hv = CrossAttentionBlock(channels, num_heads)
        self.cab_i = CrossAttentionBlock(channels, num_heads)
        
        # Branch-specific layers
        self.color_denoise = ColorDenoiseLayer(channels)  # For HV branch
        self.intensity_enhance = IntensityEnhanceLayer(channels)  # For I branch
        
    def forward(self, hv_features, i_features):
        """
        Args:
            hv_features: Features from HV branch [B, C, H, W]
            i_features: Features from I branch [B, C, H, W]
        
        Returns:
            Enhanced HV and I features
        """
        # Cross attention: HV uses I as key/value, I uses HV as key/value
        hv_attended = self.cab_hv(hv_features, i_features)
        i_attended = self.cab_i(i_features, hv_features)
        
        # Branch-specific processing
        hv_out = self.color_denoise(hv_attended)
        i_out = self.intensity_enhance(i_attended)
        
        return hv_out, i_out


class DownBlock(nn.Module):
    """Downsampling block for encoder"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class CIDNet(nn.Module):
    """
    Color and Intensity Decoupling Network
    Dual-branch UNet with Lighten Cross-Attention modules
    """
    def __init__(self, base_channels=32, num_heads=4):
        super(CIDNet, self).__init__()
        
        # Initial convolutions for each branch
        self.hv_init_conv = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)  # HV has 2 channels
        self.i_init_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)   # I has 1 channel
        
        # Encoder (3 LCA blocks with downsampling)
        self.encoder_lca1 = LightenCrossAttention(base_channels, num_heads)
        self.hv_down1 = DownBlock(base_channels, base_channels * 2)
        self.i_down1 = DownBlock(base_channels, base_channels * 2)
        
        self.encoder_lca2 = LightenCrossAttention(base_channels * 2, num_heads)
        self.hv_down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.i_down2 = DownBlock(base_channels * 2, base_channels * 4)
        
        self.encoder_lca3 = LightenCrossAttention(base_channels * 4, num_heads)
        
        # Bottleneck
        self.bottleneck_lca = LightenCrossAttention(base_channels * 4, num_heads)
        
        # Decoder (3 LCA blocks with upsampling)
        self.hv_up1 = UpBlock(base_channels * 4, base_channels * 2)
        self.i_up1 = UpBlock(base_channels * 4, base_channels * 2)
        self.decoder_lca1 = LightenCrossAttention(base_channels * 2, num_heads)
        
        self.hv_up2 = UpBlock(base_channels * 2, base_channels)
        self.i_up2 = UpBlock(base_channels * 2, base_channels)
        self.decoder_lca2 = LightenCrossAttention(base_channels, num_heads)
        
        # Output convolutions
        self.hv_out_conv = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)
        self.i_out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
    
    def forward(self, hv_map, intensity_map):
        """
        Args:
            hv_map: Input HV color map [B, 2, H, W]
            intensity_map: Input intensity map [B, 1, H, W]
        
        Returns:
            enhanced_hv: Enhanced HV map [B, 2, H, W]
            enhanced_i: Enhanced intensity map [B, 1, H, W]
        """
        # Initial feature extraction
        hv_feat = self.hv_init_conv(hv_map)
        i_feat = self.i_init_conv(intensity_map)
        
        # Encoder
        hv1, i1 = self.encoder_lca1(hv_feat, i_feat)
        hv_down1 = self.hv_down1(hv1)
        i_down1 = self.i_down1(i1)
        
        hv2, i2 = self.encoder_lca2(hv_down1, i_down1)
        hv_down2 = self.hv_down2(hv2)
        i_down2 = self.i_down2(i2)
        
        hv3, i3 = self.encoder_lca3(hv_down2, i_down2)
        
        # Bottleneck
        hv_bottle, i_bottle = self.bottleneck_lca(hv3, i3)
        
        # Decoder with skip connections
        hv_up1 = self.hv_up1(hv_bottle) + hv2  # Skip connection
        i_up1 = self.i_up1(i_bottle) + i2
        hv_dec1, i_dec1 = self.decoder_lca1(hv_up1, i_up1)
        
        hv_up2 = self.hv_up2(hv_dec1) + hv1  # Skip connection
        i_up2 = self.i_up2(i_dec1) + i1
        hv_dec2, i_dec2 = self.decoder_lca2(hv_up2, i_up2)
        
        # Output with residual connection
        enhanced_hv = self.hv_out_conv(hv_dec2) + hv_map
        enhanced_i = self.i_out_conv(i_dec2) + intensity_map
        
        return enhanced_hv, enhanced_i


# Test the model
if __name__ == "__main__":
    # Create dummy input
    batch_size = 2
    height, width = 256, 256
    
    hv_map = torch.randn(batch_size, 2, height, width)
    intensity_map = torch.randn(batch_size, 1, height, width)
    
    # Initialize model
    model = CIDNet(base_channels=32, num_heads=4)
    
    # Forward pass
    enhanced_hv, enhanced_i = model(hv_map, intensity_map)
    
    print(f"Input HV shape: {hv_map.shape}")
    print(f"Input I shape: {intensity_map.shape}")
    print(f"Output HV shape: {enhanced_hv.shape}")
    print(f"Output I shape: {enhanced_i.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
