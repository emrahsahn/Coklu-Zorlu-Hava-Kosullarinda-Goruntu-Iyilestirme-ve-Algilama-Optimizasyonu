import torch
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Import CIDNet components
from hvi_transform import HVITransform, InverseHVITransform
from cidnet_model import CIDNet
from losses import CIDNetLoss


class LOLDataset(Dataset):
    """
    LOL (Low-Light) Dataset Loader
    Expected structure:
      dataset_root/
        low/  (low-light images)
        high/ (normal-light/ground truth images)
    """
    def __init__(self, dataset_root, subset='train', image_size=256):
        super(LOLDataset, self).__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.image_size = image_size
        
        # Paths for low and high light images
        self.low_dir = os.path.join(dataset_root, subset, 'low')
        self.high_dir = os.path.join(dataset_root, subset, 'high')
        
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
        
        # Initialize components
        self.hvi_transform = HVITransform(k=1.0, gamma_G=0.5, gamma_B=0.5).to(device)
        self.cidnet = CIDNet(base_channels=base_channels, num_heads=num_heads).to(device)
        self.inverse_hvi = InverseHVITransform(self.hvi_transform).to(device)
        self.criterion = CIDNetLoss(
            lambda_l1=1.0,
            lambda_edge=0.5,
            lambda_perceptual=0.1,
            lambda_color_space=1.0,
            use_perceptual=True
        ).to(device)
    
    def forward(self, low_rgb):
        """
        Forward pass: RGB -> HVI -> Enhanced HVI -> Enhanced RGB
        
        Args:
            low_rgb: Low-light RGB image [B, 3, H, W]
        
        Returns:
            enhanced_rgb: Enhanced RGB image [B, 3, H, W]
        """
        # Step 1: Transform to HVI
        hv_map, intensity_map = self.hvi_transform(low_rgb)
        
        # Step 2: Enhance in HVI space
        enhanced_hv, enhanced_i = self.cidnet(hv_map, intensity_map)
        
        # Step 3: Transform back to RGB
        enhanced_rgb = self.inverse_hvi(enhanced_hv, enhanced_i)
        
        return enhanced_rgb, enhanced_hv, enhanced_i, hv_map, intensity_map
    
    def compute_loss(self, low_rgb, high_rgb):
        """
        Compute loss for training
        
        Args:
            low_rgb: Low-light RGB [B, 3, H, W]
            high_rgb: Ground truth RGB [B, 3, H, W]
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
            enhanced_rgb: Enhanced output
        """
        # Forward pass
        enhanced_rgb, enhanced_hv, enhanced_i, hv_map, intensity_map = self.forward(low_rgb)
        
        # Get target HVI
        target_hv, target_i = self.hvi_transform(high_rgb)
        
        # Concatenate HVI for loss computation
        pred_hvi = torch.cat([enhanced_hv, enhanced_i], dim=1)
        target_hvi = torch.cat([target_hv, target_i], dim=1)
        
        # Compute loss
        loss, loss_dict = self.criterion(enhanced_rgb, high_rgb, pred_hvi, target_hvi)
        
        return loss, loss_dict, enhanced_rgb
    
    def train_mode(self):
        """Set to training mode"""
        self.hvi_transform.train()
        self.cidnet.train()
    
    def eval_mode(self):
        """Set to evaluation mode"""
        self.hvi_transform.eval()
        self.cidnet.eval()
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'hvi_transform': self.hvi_transform.state_dict(),
            'cidnet': self.cidnet.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.hvi_transform.load_state_dict(checkpoint['hvi_transform'])
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


#######################################

class LowLightDataset(Dataset):
    def __init__(self, data_root_path, split='train', img_size=(512, 512)):
        """
        Başlatıcı metod. Veri yollarını ve parametreleri ayarlar.
        
        Args:
            data_root_path (str): LOL veya SID veri setinin kök dizini.
            split (str): 'train', 'val' veya 'test' olarak veri kümesi türü.
            img_size (tuple): Görüntülerin yeniden boyutlandırılacağı boyut.
        """
        self.img_size = img_size
        # Veri setinizin iç yapısına göre 'input' ve 'ground_truth' klasörlerini ayarlayın
        # LOL veri seti için genellikle 'our485' içinde 'low' ve 'high' klasörleri bulunur.
        self.low_light_dir = os.path.join(data_root_path, split, 'low')
        self.high_light_dir = os.path.join(data_root_path, split, 'high')

        self.file_names = sorted(os.listdir(self.low_light_dir))

        print(f"{split.capitalize()} kümesi için {len(self.file_names)} çift bulundu.")

    def __len__(self):
        """Toplam görüntü çifti sayısını döndürür."""
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Veri kümesinden bir görüntü çiftini yükler, işler ve döndürür.
        """

        filename = self.file_names[idx]

        # 1. görüntüleri yükleme
        low_path = os.path.join(self.low_light_dir, filename)
        high_path = os.path.join(self.high_light_dir, filename)

        # hata kontrolü eklenmesi önerilir
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # 2. Ön İşleme: (Resize, Normalize, Totensor)

        #Resize
        low_img = low_img.resize(self.img_size)
        high_img = high_img.resize(self.img_size)

        # PIL'den Numpy'a ardından tensöre dnüştürme ve normalizasyon


