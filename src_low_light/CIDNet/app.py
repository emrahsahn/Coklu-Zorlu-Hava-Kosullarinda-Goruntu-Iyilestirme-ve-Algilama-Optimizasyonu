import torch
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CIDNet components
from hvi_transform import RGB_HVI
from cidnet_model import CIDNet
from losses import CIDNetLoss


def _lol_low_high_dirs(dataset_root, subset):
    """
    Resolve low/high folders for LOL-style datasets.

    1) LOL-v2 / CCM layout (same as train_ccm.py):
         dataset_root/our485/{low,high}  -> train
         dataset_root/eval15/{low,high}  -> val/test

    2) Classic layout:
         dataset_root/train/{low,high}, dataset_root/test/{low,high}
    """
    root = dataset_root
    v2_train = os.path.join(root, 'our485', 'low')
    v2_eval = os.path.join(root, 'eval15', 'low')
    if os.path.isdir(v2_train) and os.path.isdir(v2_eval):
        if subset == 'train':
            return os.path.join(root, 'our485', 'low'), os.path.join(root, 'our485', 'high')
        return os.path.join(root, 'eval15', 'low'), os.path.join(root, 'eval15', 'high')

    split = 'train' if subset == 'train' else 'test'
    return os.path.join(root, split, 'low'), os.path.join(root, split, 'high')


class LOLDataset(Dataset):
    """
    LOL (Low-Light) Dataset Loader
    Expected structure (classic):
      dataset_root/train/low, train/high
      dataset_root/test/low, test/high

    Or LOL-v2 / CCM Combined layout:
      dataset_root/our485/low, our485/high
      dataset_root/eval15/low, eval15/high
    """
    def __init__(self, dataset_root, subset='train', image_size=256):
        super(LOLDataset, self).__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.image_size = image_size
        
        # Paths for low and high light images
        self.low_dir, self.high_dir = _lol_low_high_dirs(dataset_root, subset)
        
        # Get list of image files
        if os.path.exists(self.low_dir):
            self.image_files = sorted([f for f in os.listdir(self.low_dir) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            print(f"Warning: {self.low_dir} not found. Using empty dataset.")
            self.image_files = []
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load low-light and high-light images
        low_path = os.path.join(self.low_dir, img_name)
        high_path = os.path.join(self.high_dir, img_name)
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply transforms
        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)
        
        return {
            'low': low_tensor,
            'high': high_tensor,
            'filename': img_name
        }


class CIDNetPipeline:
    """
    Complete CIDNet pipeline: RGB -> HVI -> Enhancement -> RGB
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 base_channels=32, num_heads=4):
        self.device = device
        
        # Initialize CIDNet with integrated HVI transforms
        self.cidnet = CIDNet(base_channels=base_channels, num_heads=num_heads).to(device)
        
        # Set transform parameters like in original app.py
        self.cidnet.trans.gated = True
        self.cidnet.trans.gated2 = True
        
        # Fix saturation issue - use lower default values
        self.cidnet.trans.alpha_s = 1.0  # was 1.3, causing blue tint
        self.cidnet.trans.alpha = 1.0    # intensity
        
        self.criterion = CIDNetLoss(
            lambda_l1=1.0,
            lambda_edge=0.5,
            lambda_perceptual=0.1,
            lambda_color_space=1.0,
            use_perceptual=True
        ).to(device)
    
    def forward(self, low_rgb, gamma=1.0):
        """
        Forward pass: RGB -> HVI -> Enhanced HVI -> Enhanced RGB
        
        Args:
            low_rgb: Low-light RGB image [B, 3, H, W]
            gamma: Gamma correction parameter (default 1.0)
        
        Returns:
            enhanced_rgb: Enhanced RGB image [B, 3, H, W]
        """
        # Apply gamma correction to input like in original app.py
        input_corrected = low_rgb ** gamma
        
        # Forward through CIDNet (includes HVI transforms)
        enhanced_rgb = self.cidnet(input_corrected)
        
        return enhanced_rgb
    
    
    def compute_loss(self, low_rgb, high_rgb, gamma=1.0):
        """
        Compute loss for training
        
        Args:
            low_rgb: Low-light RGB [B, 3, H, W]
            high_rgb: Ground truth RGB [B, 3, H, W]
            gamma: Gamma correction parameter
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
            enhanced_rgb: Enhanced output
        """
        # Forward pass
        enhanced_rgb = self.forward(low_rgb, gamma)
        
        # Compute loss (simplified for RGB-to-RGB comparison)
        loss, loss_dict = self.criterion(enhanced_rgb, high_rgb, enhanced_rgb, high_rgb)
        
        return loss, loss_dict, enhanced_rgb
    
    def train_mode(self):
        """Set to training mode"""
        self.cidnet.train()
    
    def eval_mode(self):
        """Set to evaluation mode"""
        self.cidnet.eval()
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'cidnet': self.cidnet.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.cidnet.load_state_dict(checkpoint['cidnet'])
        print(f"Checkpoint loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Test dataset loader
    dataset_root = "./data/LOL"  # Update this path
    
    if os.path.exists(dataset_root):
        dataset = LOLDataset(dataset_root, subset='train', image_size=256)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
            
            # Test one batch
            for batch in dataloader:
                low = batch['low']
                high = batch['high']
                print(f"Low-light batch shape: {low.shape}")
                print(f"High-light batch shape: {high.shape}")
                break
    
    # Test CIDNet pipeline
    print("\nTesting CIDNet Pipeline...")
    pipeline = CIDNetPipeline(device='cpu', base_channels=32, num_heads=4)
    
    # Dummy input
    low_rgb = torch.rand(2, 3, 256, 256)
    high_rgb = torch.rand(2, 3, 256, 256)
    
    # Forward pass and loss
    pipeline.train_mode()
    loss, loss_dict, enhanced = pipeline.compute_loss(low_rgb, high_rgb)
    
    print(f"\nOutput shape: {enhanced.shape}")
    print(f"Total loss: {loss.item():.6f}")
    print("Loss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")


