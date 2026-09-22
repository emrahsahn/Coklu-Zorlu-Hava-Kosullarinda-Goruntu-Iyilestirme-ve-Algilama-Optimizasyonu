"""
Dataset yapısını LOL formatına dönüştüren ve test için hazırlayan script
"""
import os
import shutil
from pathlib import Path

def prepare_lol_dataset():
    """
    Mevcut lol_dataset klasörünü train/test/low/high yapısına dönüştürür
    """
    base_dir = Path("data/lol_dataset")
    
    # Hedef yapıyı oluştur
    train_low = Path("data/LOL/train/low")
    train_high = Path("data/LOL/train/high")
    test_low = Path("data/LOL/test/low")
    test_high = Path("data/LOL/test/high")
    
    for path in [train_low, train_high, test_low, test_high]:
        path.mkdir(parents=True, exist_ok=True)
    
    print("Dataset yapısı oluşturuldu:")
    print("data/LOL/")
    print("  train/")
    print("    low/")
    print("    high/")
    print("  test/")
    print("    low/")
    print("    high/")
    
    # Mevcut dosyaları kontrol et
    our485_path = base_dir / "our485"
    eval15_path = base_dir / "eval15"
    
    if our485_path.exists():
        print(f"\n✓ our485 klasörü bulundu: {our485_path}")
        # our485'i train için kullan
        low_path = our485_path / "low"
        high_path = our485_path / "high"
        
        if low_path.exists() and high_path.exists():
            print(f"  - low: {len(list(low_path.glob('*')))} dosya")
            print(f"  - high: {len(list(high_path.glob('*')))} dosya")
            
            # Dosyaları kopyala (veya symlink oluştur)
            for img in low_path.glob("*"):
                if img.is_file():
                    shutil.copy(img, train_low / img.name)
            for img in high_path.glob("*"):
                if img.is_file():
                    shutil.copy(img, train_high / img.name)
            
            print(f"  → {len(list(train_low.glob('*')))} train/low dosya kopyalandı")
            print(f"  → {len(list(train_high.glob('*')))} train/high dosya kopyalandı")
    
    if eval15_path.exists():
        print(f"\n✓ eval15 klasörü bulundu: {eval15_path}")
        # eval15'i test için kullan
        low_path = eval15_path / "low"
        high_path = eval15_path / "high"
        
        if low_path.exists() and high_path.exists():
            print(f"  - low: {len(list(low_path.glob('*')))} dosya")
            print(f"  - high: {len(list(high_path.glob('*')))} dosya")
            
            # Dosyaları kopyala
            for img in low_path.glob("*"):
                if img.is_file():
                    shutil.copy(img, test_low / img.name)
            for img in high_path.glob("*"):
                if img.is_file():
                    shutil.copy(img, test_high / img.name)
            
            print(f"  → {len(list(test_low.glob('*')))} test/low dosya kopyalandı")
            print(f"  → {len(list(test_high.glob('*')))} test/high dosya kopyalandı")
    
    print("\n" + "="*60)
    print("Dataset hazır! Train komutu:")
    print("python src/train.py --dataset_root ./data/LOL --batch_size 2 --epochs 5 --device cpu")
    print("="*60)


if __name__ == "__main__":
    prepare_lol_dataset()
