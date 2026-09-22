import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# CIDNet klasörünü import yollarına ekliyoruz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src_low_light', 'CIDNet')))

from app import CIDNetPipeline
from train import train_epoch

class DummyDataset(Dataset):
    """Sadece test amaçlı rastgele veriler üreten sahte veri seti"""
    def __init__(self, num_samples=8, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            'low': torch.rand(3, self.image_size, self.image_size),
            'high': torch.rand(3, self.image_size, self.image_size),
            'filename': f'dummy_{idx}.png'
        }

def test_training_pipeline():
    print("1. Sanal ortam dataseti oluşturuluyor (Dummy Dataset)...")
    # CUDA OOM (Out of Memory) hatasını engellemek için image_size küçük tutuluyor
    dataset = DummyDataset(num_samples=4, image_size=128)  
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"2. Kullanılacak cihaz: {device}")
    
    print("3. CIDNet (Low Light) pipeline başlatılıyor...")
    pipeline = CIDNetPipeline(device=device, base_channels=16, num_heads=2)
    
    # Optimizer hazırlığı
    params = list(pipeline.cidnet.parameters())
    optimizer = optim.AdamW(params, lr=1e-4)
    
    # Tensorboard logları için sahte dizin
    log_dir = os.path.join('logs', 'dummy_test')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print("4. Eğitim işlemi 1 epoch olarak sahte verilerle çalıştırılıyor...")
    try:
        avg_loss, loss_components = train_epoch(
            pipeline=pipeline, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            epoch=0, 
            writer=writer, 
            device=device
        )
        print("\n" + "="*50)
        print("✅ TEST BAŞARILI! Eğitim kodları CUDA/CPU ile sorunsuz çalışıyor.")
        print(f"Ortalama Kayıp (Average Loss): {avg_loss:.6f}")
        for k, v in loss_components.items():
            print(f"  - {k}: {v:.6f}")
        print("="*50)
            
    except Exception as e:
        print(f"\n❌ TEST BAŞARISIZ! Şöyle bir hata oluştu: {str(e)}")
        
if __name__ == "__main__":
    test_training_pipeline()
