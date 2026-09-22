# CIDNet Prototip - Görüntü İyileştirme Projesi

**CIDNet (Color and Intensity Decoupling Network)** tabanlı düşük ışıklı görüntü iyileştirme prototipi.

## 📄 Proje Hakkında

Bu proje, [CIDNet makalesine](https://arxiv.org/abs/2402.05809v3) dayanan bir görüntü iyileştirme sistemidir. Temel özellikler:

- **HVI Renk Uzayı**: Trainable parametrelerle (k, γ_G, γ_B) RGB'den HVI'ye dönüşüm
- **Dual-Branch Architecture**: HV (renk) ve I (parlaklık) için ayrı işleme
- **Lighten Cross-Attention (LCA)**: Branch'ler arası etkileşim modülü
- **Multi-Loss Training**: sRGB ve HVI uzaylarında L1, edge ve perceptual loss

## 🏗️ Proje Yapısı

```
├── src/
│   ├── hvi_transform.py      # HVI renk uzayı dönüşümleri
│   ├── cidnet_model.py        # CIDNet model mimarisi
│   ├── losses.py              # Loss fonksiyonları
│   ├── app.py                 # Dataset loader ve pipeline
│   └── train.py               # Training script
├── data/
│   ├── data_file.py           # Dataset indirme (Kaggle)
│   └── LOL/                   # LOL dataset klasörü
├── document/
│   └── 2402.05809v3 (1).pdf   # CIDNet paper
├── checkpoints/               # Model checkpoint'leri
├── logs/                      # Tensorboard logs
└── requirement.txt            # Python dependencies
```

## 🚀 Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirement.txt
```

### 2. Dataset İndirme

LOL dataset'ini Kaggle'dan indirmek için (şu an 404 hatası var, alternatif dataset bulunmalı):

```bash
python data/data_file.py
```

**Not**: Dataset yolu çalışmıyorsa, manuel olarak LOL dataset'ini indirip `data/LOL/` klasörüne yerleştirin:
```
data/LOL/
  train/
    low/   # Düşük ışıklı görüntüler
    high/  # Normal ışıklı görüntüler
  test/
    low/
    high/
```

## 🧪 Kullanım

### Test - Modelleri Çalıştırma

Her modülü bağımsız olarak test edebilirsiniz:

```bash
# HVI dönüşümünü test et
python src/hvi_transform.py

# CIDNet modelini test et
python src/cidnet_model.py

# Loss fonksiyonlarını test et
python src/losses.py

# Pipeline'ı test et
python src/app.py
```

### Training

```bash
# Temel training (CPU)
python src/train.py --dataset_root ./data/LOL --device cpu --epochs 10

# GPU ile training
python src/train.py --dataset_root ./data/LOL --device cuda --batch_size 8 --epochs 100

# Tüm parametreler
python src/train.py \
  --dataset_root ./data/LOL \
  --batch_size 4 \
  --epochs 100 \
  --lr 1e-4 \
  --image_size 256 \
  --base_channels 32 \
  --num_heads 4 \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs \
  --device cuda
```

### Tensorboard ile İzleme

```bash
tensorboard --logdir ./logs
```

### Inference (Checkpoint Kullanarak)

```python
from app import CIDNetPipeline
import torch
from PIL import Image
import torchvision.transforms as transforms

# Pipeline oluştur
pipeline = CIDNetPipeline(device='cpu', base_channels=32)

# Checkpoint yükle
pipeline.load_checkpoint('./checkpoints/best_model.pth')
pipeline.eval_mode()

# Görüntü yükle
img = Image.open('test_image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
low_rgb = transform(img).unsqueeze(0)  # [1, 3, 256, 256]

# Enhancement
with torch.no_grad():
    enhanced_rgb, _, _, _, _ = pipeline.forward(low_rgb)

# Kaydet
enhanced_img = transforms.ToPILImage()(enhanced_rgb[0])
enhanced_img.save('enhanced_output.jpg')
```

## 📊 Model Detayları

### CIDNet Mimarisi

- **Base Channels**: 32 (ayarlanabilir)
- **Attention Heads**: 4
- **Parameters**: ~1.88M (paper'daki gibi hafif)
- **FLOPs**: ~7.57G

### HVI Color Space

- **I (Intensity)**: `max(R, G, B)` - parlaklık bilgisi
- **H (Horizontal)**: `C_k ⊙ S ⊙ D_T ⊙ cos(2π P_γ)`
- **V (Vertical)**: `C_k ⊙ S ⊙ D_T ⊙ sin(2π P_γ)`
- **Trainable**: k (density), γ_G, γ_B (color perception)

### Loss Functions

```python
L = λ_c * l(HVI) + l(sRGB)
l(·) = λ_1 * L1 + λ_e * L_edge + λ_p * L_perceptual
```

Varsayılan ağırlıklar:
- λ_1 = 1.0 (L1 loss)
- λ_e = 0.5 (Edge loss)
- λ_p = 0.1 (Perceptual loss)
- λ_c = 1.0 (HVI space weight)

## 🔬 Özellikler

### Tamamlananlar ✅
- [x] HVI renk uzayı dönüşümü (forward + inverse)
- [x] Trainable HVI parametreleri (k, γ_G, γ_B)
- [x] Dual-branch UNet mimarisi
- [x] Lighten Cross-Attention (LCA) modülü
- [x] Intensity Enhance Layer (IEL)
- [x] Color Denoise Layer (CDL)
- [x] Multi-loss (L1, Edge, Perceptual)
- [x] LOL Dataset loader
- [x] Training pipeline
- [x] Checkpoint save/load
- [x] Tensorboard logging

### Geliştirilebilir 🔧
- [ ] Dataset 404 hatası çözümü (alternatif kaynak)
- [ ] Data augmentation ekleme
- [ ] Mixed precision training (FP16)
- [ ] Multi-GPU support
- [ ] Evaluation metrics (PSNR, SSIM)
- [ ] Inference script optimization
- [ ] Model quantization

## 📚 Referanslar

- **Paper**: [You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement](https://arxiv.org/abs/2402.05809v3)
- **Authors**: Qingsen Yan, Yixu Feng, Cheng Zhang, et al.
- **GitHub**: https://github.com/Fediory/HVI-CIDNet (orijinal implementasyon)

## 🤝 Katkıda Bulunma

Bu bir prototip projedir. İyileştirmeler için:

1. Dataset sorununu çözün (LOL v1/v2 dataset bağlantısı)
2. Training'i çalıştırıp sonuçları değerlendirin
3. Hyperparameter tuning yapın
4. Test sonuçlarını paylaşın

## 📝 Lisans

Eğitim amaçlı prototip. CIDNet paper'ı referans alınmıştır.

---

**Not**: Bu implementasyon CIDNet makalesine dayanarak oluşturulmuş bir prototiptir. Orijinal paper'daki tüm detaylar ve optimizasyonlar içermeyebilir.
