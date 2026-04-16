"""
Loss functions for CIDNet training
Based on Equations 12-13 from the paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EdgeLoss(nn.Module):
    """
    Edge loss to preserve image structure
    Uses Sobel filters to compute edges
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_edges(self, x):
        """Compute image edges using Sobel filters"""
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Apply Sobel to each channel
        edges_x = F.conv2d(x.reshape(B*C, 1, H, W), self.sobel_x, padding=1)
        edges_y = F.conv2d(x.reshape(B*C, 1, H, W), self.sobel_y, padding=1)
        
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)
        edges = edges.reshape(B, C, H, W)
        
        return edges
    
    def forward(self, pred, target):
        """
        Compute edge loss between prediction and target
        """
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        
        return F.l1_loss(pred_edges, target_edges)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    Compares high-level features between pred and target
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Extract specific layers
        self.feature_extractors = nn.ModuleDict()
        
        layer_mapping = {
            'relu1_2': 3,   # After first ReLU
            'relu2_2': 8,   # After second block
            'relu3_3': 15,  # After third block
        }
        
        for name, idx in layer_mapping.items():
            if name in layers:
                self.feature_extractors[name] = nn.Sequential(*list(vgg.children())[:idx+1])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization values for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize image for VGG"""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        pred, target: [B, 3, H, W] in range [0, 1]
        """
        # Normalize
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        loss = 0.0
        for name, extractor in self.feature_extractors.items():
            pred_feat = extractor(pred_norm)
            target_feat = extractor(target_norm)
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(self.feature_extractors)


class CIDNetLoss(nn.Module):
    """
    Combined loss for CIDNet training
    L = λ_c * l(HVI) + l(sRGB)
    where l(·) = λ_1 * L1 + λ_e * L_edge + λ_p * L_perceptual
    """
    def __init__(self, 
                 lambda_l1=1.0, 
                 lambda_edge=0.5, 
                 lambda_perceptual=0.1,
                 lambda_color_space=1.0,
                 use_perceptual=True):
        super(CIDNetLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_edge = lambda_edge
        self.lambda_perceptual = lambda_perceptual
        self.lambda_color_space = lambda_color_space
        self.use_perceptual = use_perceptual
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def compute_loss_for_space(self, pred, target, name=""):
        """
        Compute l(·) = λ_1 * L1 + λ_e * L_edge + λ_p * L_perceptual
        """
        # L1 loss
        loss_l1 = self.l1_loss(pred, target)
        
        # Edge loss
        loss_edge = self.edge_loss(pred, target)
        
        # Perceptual loss (only for RGB images)
        loss_perceptual = 0.0
        if self.use_perceptual and pred.shape[1] == 3:  # RGB has 3 channels
            loss_perceptual = self.perceptual_loss(pred, target)
        
        # Combined loss
        total_loss = (self.lambda_l1 * loss_l1 + 
                     self.lambda_edge * loss_edge + 
                     self.lambda_perceptual * loss_perceptual)
        
        loss_dict = {
            f'{name}_l1': loss_l1.item(),
            f'{name}_edge': loss_edge.item(),
        }
        
        if self.use_perceptual and pred.shape[1] == 3:
            loss_dict[f'{name}_perceptual'] = loss_perceptual.item()
        
        return total_loss, loss_dict
    
    def forward(self, pred_rgb, target_rgb, pred_hvi=None, target_hvi=None):
        """
        Compute total CIDNet loss
        
        Args:
            pred_rgb: Predicted RGB image [B, 3, H, W]
            target_rgb: Target RGB image [B, 3, H, W]
            pred_hvi: Predicted HVI (HV + I concatenated) [B, 3, H, W] (optional)
            target_hvi: Target HVI [B, 3, H, W] (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Loss in sRGB space
        loss_rgb, rgb_dict = self.compute_loss_for_space(pred_rgb, target_rgb, name="rgb")
        
        total_loss = loss_rgb
        loss_dict = rgb_dict
        
        # Loss in HVI space (if provided)
        if pred_hvi is not None and target_hvi is not None:
            loss_hvi, hvi_dict = self.compute_loss_for_space(pred_hvi, target_hvi, name="hvi")
            total_loss = self.lambda_color_space * loss_hvi + loss_rgb
            loss_dict.update(hvi_dict)
            loss_dict['hvi_total'] = loss_hvi.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Test the loss functions
if __name__ == "__main__":
    # Create dummy predictions and targets
    batch_size = 2
    pred_rgb = torch.rand(batch_size, 3, 256, 256)
    target_rgb = torch.rand(batch_size, 3, 256, 256)
    
    pred_hvi = torch.rand(batch_size, 3, 256, 256)  # HV (2 channels) + I (1 channel)
    target_hvi = torch.rand(batch_size, 3, 256, 256)
    
    # Initialize loss
    criterion = CIDNetLoss(
        lambda_l1=1.0,
        lambda_edge=0.5,
        lambda_perceptual=0.1,
        lambda_color_space=1.0,
        use_perceptual=True
    )
    
    # Compute loss
    total_loss, loss_dict = criterion(pred_rgb, target_rgb, pred_hvi, target_hvi)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nTotal loss: {total_loss.item():.6f}")
    print(f"Loss can be backpropagated: {total_loss.requires_grad}")
