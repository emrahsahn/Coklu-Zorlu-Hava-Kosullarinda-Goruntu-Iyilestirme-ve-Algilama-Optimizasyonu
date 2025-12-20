"""
HVI (Horizontal/Vertical-Intensity) Color Space Transformation
Based on CIDNet paper: https://arxiv.org/abs/2402.05809v3
"""
import torch
import torch.nn as nn
import numpy as np


class HVITransform(nn.Module):
    """
    Trainable HVI color space transformation.
    Converts sRGB images to HVI space with trainable parameters.
    """
    def __init__(self, k=1.0, gamma_G=0.5, gamma_B=0.5):
        super(HVITransform, self).__init__()
        # Trainable parameters
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        self.gamma_G = nn.Parameter(torch.tensor(gamma_G, dtype=torch.float32))
        self.gamma_B = nn.Parameter(torch.tensor(gamma_B, dtype=torch.float32))
        self.eps = 1e-8
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        
        max_val = torch.max(torch.max(r, g), b)
        min_val = torch.min(torch.min(r, g), b)
        diff = max_val - min_val + self.eps
        
        # Hue calculation
        h = torch.zeros_like(max_val)
        mask_r = (max_val == r)
        mask_g = (max_val == g)
        mask_b = (max_val == b)
        
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        h = h / 360.0  # Normalize to [0, 1]
        
        # Saturation
        s = torch.where(max_val == 0, torch.zeros_like(diff), diff / (max_val + self.eps))
        
        # Value
        v = max_val
        
        return h, s, v
    
    def compute_color_perceptual_map(self, H):
        """
        Compute adaptive linear Color-Perceptual map P_γ
        Equation 3 from the paper
        """
        gamma_G = torch.clamp(self.gamma_G, 0.01, 0.99)
        gamma_B = torch.clamp(self.gamma_B, 0.01, 0.99)
        
        P_gamma = torch.zeros_like(H)
        
        # Region 1: 0 <= H < 1/3
        mask1 = (H >= 0) & (H < 1/3)
        P_gamma[mask1] = 3 * gamma_G * H[mask1]
        
        # Region 2: 1/3 <= H < 2/3
        mask2 = (H >= 1/3) & (H < 2/3)
        P_gamma[mask2] = 3 * (gamma_B - gamma_G) * (H[mask2] - 1/3) + gamma_G
        
        # Region 3: 2/3 <= H <= 1
        mask3 = (H >= 2/3) & (H <= 1)
        P_gamma[mask3] = 3 * (1 - gamma_B) * (H[mask3] - 1) + 1
        
        return P_gamma
    
    def compute_function_density(self, P_gamma):
        """
        Compute Function-Density-T
        Equation 4: D_T = T(P_γ)
        Using a simple sinusoidal function that satisfies T(0) = T(1)
        """
        # Simple implementation: D_T = 1 + sin(2π * P_γ) / 4
        # This ensures D_T >= 0 and T(0) = T(1)
        D_T = 1.0 + torch.sin(2 * np.pi * P_gamma) / 4.0
        return D_T
    
    def forward(self, rgb_image):
        """
        Transform sRGB image to HVI color space
        
        Args:
            rgb_image: Input RGB image [B, 3, H, W], values in [0, 1]
        
        Returns:
            hv_map: HV color map [B, 2, H, W]
            intensity_map: Intensity map [B, 1, H, W]
        """
        # Step 1: Compute Intensity Map (Equation 1)
        # I_max = max(R, G, B)
        intensity_map = torch.max(rgb_image, dim=1, keepdim=True)[0]
        
        # Step 2: Convert to HSV
        H, S, V = self.rgb_to_hsv(rgb_image)
        
        # Step 3: Compute Color-Density-k (Equation 2)
        # C_k = k * sin(π * I_max / 2) + ε
        C_k = torch.abs(self.k) * torch.sin(np.pi * intensity_map / 2.0) + self.eps
        
        # Step 4: Compute Color-Perceptual map P_γ (Equation 3)
        P_gamma = self.compute_color_perceptual_map(H)
        
        # Step 5: Compute Function-Density-T (Equation 4)
        D_T = self.compute_function_density(P_gamma)
        
        # Step 6: Compute HV map (Equation 5)
        # Intermediate variables for bijective mapping
        h = torch.cos(2 * np.pi * P_gamma)
        v = torch.sin(2 * np.pi * P_gamma)
        
        # H_hat = C_k ⊙ S ⊙ D_T ⊙ h
        # V_hat = C_k ⊙ S ⊙ D_T ⊙ v
        H_hat = C_k * S * D_T * h
        V_hat = C_k * S * D_T * v
        
        hv_map = torch.cat([H_hat, V_hat], dim=1)  # [B, 2, H, W]
        
        return hv_map, intensity_map


class InverseHVITransform(nn.Module):
    """
    Perceptual-inverse HVI Transformation (PHVIT)
    Converts HVI back to sRGB
    """
    def __init__(self, hvi_transform, alpha_s=1.0, alpha_i=1.0):
        super(InverseHVITransform, self).__init__()
        self.hvi_transform = hvi_transform  # Access trainable parameters
        self.alpha_s = alpha_s  # Saturation adjustment
        self.alpha_i = alpha_i  # Intensity/brightness adjustment
        self.eps = 1e-8
    
    def compute_inverse_perceptual_map(self, X):
        """
        Inverse linear function F_γ (Equation 8)
        """
        gamma_G = torch.clamp(self.hvi_transform.gamma_G, 0.01, 0.99)
        gamma_B = torch.clamp(self.hvi_transform.gamma_B, 0.01, 0.99)
        
        F_gamma = torch.zeros_like(X)
        
        # Region 1: 0 <= X < γ_G
        mask1 = (X >= 0) & (X < gamma_G)
        F_gamma[mask1] = X[mask1] / (3 * gamma_G + self.eps)
        
        # Region 2: γ_G <= X < γ_B
        mask2 = (X >= gamma_G) & (X < gamma_B)
        F_gamma[mask2] = (X[mask2] - gamma_G) / (3 * (gamma_B - gamma_G) + self.eps) + 1/3
        
        # Region 3: γ_B <= X <= 1
        mask3 = (X >= gamma_B) & (X <= 1)
        F_gamma[mask3] = (X[mask3] - 1) / (3 * (1 - gamma_B) + self.eps) + 1
        
        return F_gamma
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h = h * 360.0  # Convert back to degrees
        
        c = v * s
        x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
        m = v - c
        
        rgb = torch.zeros_like(torch.cat([h, s, v], dim=1))
        
        mask1 = (h >= 0) & (h < 60)
        mask2 = (h >= 60) & (h < 120)
        mask3 = (h >= 120) & (h < 180)
        mask4 = (h >= 180) & (h < 240)
        mask5 = (h >= 240) & (h < 300)
        mask6 = (h >= 300) & (h < 360)
        
        rgb[:, 0:1][mask1] = c[mask1]
        rgb[:, 1:2][mask1] = x[mask1]
        
        rgb[:, 0:1][mask2] = x[mask2]
        rgb[:, 1:2][mask2] = c[mask2]
        
        rgb[:, 1:2][mask3] = c[mask3]
        rgb[:, 2:3][mask3] = x[mask3]
        
        rgb[:, 1:2][mask4] = x[mask4]
        rgb[:, 2:3][mask4] = c[mask4]
        
        rgb[:, 0:1][mask5] = x[mask5]
        rgb[:, 2:3][mask5] = c[mask5]
        
        rgb[:, 0:1][mask6] = c[mask6]
        rgb[:, 2:3][mask6] = x[mask6]
        
        rgb = rgb + m
        return torch.clamp(rgb, 0, 1)
    
    def forward(self, hv_map, intensity_map, original_rgb=None):
        """
        Transform HVI back to sRGB
        
        Args:
            hv_map: Enhanced HV color map [B, 2, H, W]
            intensity_map: Enhanced intensity map [B, 1, H, W]
            original_rgb: Original RGB for reference (optional)
        
        Returns:
            rgb_image: Output RGB image [B, 3, H, W]
        """
        H_hat = hv_map[:, 0:1, :, :]
        V_hat = hv_map[:, 1:2, :, :]
        I_hat = intensity_map
        
        # Recompute intermediate values (Equation 6)
        # Need D_T and C_k from original transformation
        # For simplicity, we compute them again
        C_k = torch.abs(self.hvi_transform.k) * torch.sin(np.pi * I_hat / 2.0) + self.eps
        
        # Approximate D_T (this is a simplification; ideally store from forward pass)
        # Using average D_T ≈ 1.0 for inverse
        D_T = torch.ones_like(I_hat)
        
        # Compute h_hat and v_hat
        h_hat = H_hat / (D_T * C_k + self.eps)
        v_hat = V_hat / (D_T * C_k + self.eps)
        
        # Compute Hue (Equation 7)
        angle = torch.atan2(v_hat, h_hat)
        angle = (angle + 2 * np.pi) % (2 * np.pi)  # Ensure positive
        X = angle / (2 * np.pi)
        H = self.compute_inverse_perceptual_map(X)
        
        # Compute Saturation and Value (Equation 9)
        S = self.alpha_s * torch.sqrt(h_hat**2 + v_hat**2 + self.eps)
        V = self.alpha_i * I_hat
        
        # Clamp values
        H = torch.clamp(H, 0, 1)
        S = torch.clamp(S, 0, 1)
        V = torch.clamp(V, 0, 1)
        
        # Convert HSV to RGB
        rgb_image = self.hsv_to_rgb(H, S, V)
        
        return rgb_image


# Example usage and testing
if __name__ == "__main__":
    # Test HVI transformation
    batch_size = 2
    height, width = 256, 256
    
    # Create dummy RGB image
    rgb_image = torch.rand(batch_size, 3, height, width)
    
    # Initialize HVI transform
    hvi_transform = HVITransform(k=1.0, gamma_G=0.5, gamma_B=0.5)
    
    # Forward transform
    hv_map, intensity_map = hvi_transform(rgb_image)
    
    print(f"Input RGB shape: {rgb_image.shape}")
    print(f"HV map shape: {hv_map.shape}")
    print(f"Intensity map shape: {intensity_map.shape}")
    print(f"Trainable parameters: k={hvi_transform.k.item():.3f}, "
          f"γ_G={hvi_transform.gamma_G.item():.3f}, "
          f"γ_B={hvi_transform.gamma_B.item():.3f}")
    
    # Inverse transform
    inverse_transform = InverseHVITransform(hvi_transform)
    reconstructed_rgb = inverse_transform(hv_map, intensity_map, rgb_image)
    
    print(f"Reconstructed RGB shape: {reconstructed_rgb.shape}")
    
    # Check reconstruction error
    mse = torch.mean((rgb_image - reconstructed_rgb)**2)
    print(f"Reconstruction MSE: {mse.item():.6f}")
