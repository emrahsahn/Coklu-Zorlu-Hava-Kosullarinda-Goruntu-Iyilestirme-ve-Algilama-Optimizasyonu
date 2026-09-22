# Çoklu Hava Koşullarında Görüntü İyileştirme — Poster Metinleri

**Proje:** Çoklu Hava Koşullarında Görüntü İyileştirme ve Algilama Optimizasyonu  
**Kurum:** KOSTÜ — Mühendislik ve Doğa Bilimleri Fakültesi  
**Yazarlar:** Anıl Erdoğan, Emin Kayra Ertekin, Emrah Şahin  
**Danışman:** Fulya Akdeniz  

---

## GİRİŞ

Otonom sürüş sistemleri, insansız hava araçları (İHA) ve akıllı gözetim kameraları, operasyonel güvenlik için sürekli ve güvenilir görsel algıya ihtiyaç duyar. Sis, **yağmur çizgileri** ve düşük ışık gibi atmosferik bozulmalar; kontrastı düşürür, renkleri soluklaştırır ve nesne sınırlarını belirsizleştirir. Bu durum, sonraki aşamadaki **nesne tespiti** modellerinin performansını doğrudan düşürür.

Geleneksel görüntü iyileştirme yöntemleri (histogram eşitleme, CLAHE, karanlık kanal öncesi sis giderme vb.) bazen görüntüyü “daha parlak” gösterse de **yapısal bilgiyi** bozabilir; yapay kontrast ve renk kaymaları, derin öğrenme tabanlı dedektörlerde yanlış pozitif ve kaçırılan tespitlere yol açabilir.

Bu çalışmada, **koşula özel derin öğrenme restorasyon modelleri** tek bir **hibrit pipeline** (`UnifiedPipeline`) içinde birleştirilmiş ve iyileştirilmiş görüntü üzerinde **YOLOv8** nesne tespiti ile karşılaştırmalı değerlendirme yapılmıştır.

**Önemli:** Birleşik pipeline’da düşük ışık için yalnızca **HVI-CIDNet** kullanılmaktadır. **CCM Combined** aynı veri setinde karşılaştırmalı analiz içindir; **canlı hibrit sistemde yer almaz**. Yağmur tarafında eski **DeRaindrop** (lens damlası) kaldırılmış; yerine **DDN** (yağmur **çizgisi**) entegre edilmiştir.

---

## PROBLEM TANIMI

- Zorlu hava koşullarında görüntü kalitesinin düşmesi → algılama güvenilirliğinin azalması  
- Tek tip restorasyon modelinin tüm bozulma türlerinde yetersiz kalması  
- Restorasyon ile nesne tespitinin birbirinden bağımsız ele alınması  
- Otonom karar sistemleri için **uçtan uca**, modüler ve ölçülebilir bir çözüm ihtiyacı  

---

## YÖNTEM

### 1. Hibrit restorasyon pipeline’ı (`UnifiedPipeline`)

Girdi RGB görüntüsü için akış:

1. **İsteğe bağlı otomatik koşul tespiti** (`WeatherDetector`, `src/weather_detector.py`): Ek ML modeli yok; istatistiksel skorlar:
   - **Düşük ışık:** ortalama parlaklık, koyu piksel oranı  
   - **Sis:** karanlık kanal ortalaması (DCP), kontrast, kenar yoğunluğu  
   - **Yağmur (çizgi):** yerel kontrast varyansı, parlak nokta oranı  
   - En yüksek skor `min_confidence` üstündeyse seçilir; aksi halde `clear` → restorasyon atlanır (`none`).

2. **Koşul → pipeline modu** (`condition_to_mode`):

   | Algılanan koşul | Pipeline modu | Restorasyon modeli |
   |-----------------|---------------|-------------------|
   | `low_light` | `low_light_hvi` | HVI-CIDNet |
   | `haze` | `haze_ffa_its` veya `haze_ffa_ots` | FFA-Net (varsayılan ITS) |
   | `rain` | `rain_ddn` | DDN (Deep Detailed Network) |
   | `clear` | `none` | — |

3. **YOLOv8 Nano** (`yolov8n.pt`): orijinal ve restore görüntüde tespit; kutu sayısı ve güven karşılaştırması.

4. **Metrikler:** PSNR, SSIM (referans/GT varsa ona; yoksa orijinale); tespit farkları. Boyut uyumsuzluğunda üst-sol ortak pencere (`src/metrics.py`).

**Şekil 1 — Akış diyagramı:**  
`Girdi → [WeatherDetector?] → HVI-CIDNet | FFA-Net (ITS/OTS) | DDN → Restore → YOLOv8 (×2) → Gradio / CSV rapor`

**Gradio arayüzü** (`src/app.py`): modlar — Otomatik, Düşük ışık (HVI), Sis ITS, Sis OTS, **Yağmur çizgisi (DDN)**, Yok.

---

### 2. Düşük ışık: HVI-CIDNet (pipeline modeli)

- RGB → **HVI** renk uzayı; dual-branch CIDNet; Lighten Cross-Attention.  
- Girdi **8’in katına** reflect pad; isteğe bağlı **gamma**, **alpha_s**, **alpha_i**.  
- Ağırlık: `HVI-CIDNet/weights/train/epoch_100.pth`  
- Yükleyici: `HVI_CIDNetWrapper` (`src/model_loader.py`).

---

### 3. Sis: FFA-Net

- **Mimari:** Feature Fusion Attention Network (GPS=3).  
- **ITS (iç mekan):** 20 blok → `FFA-Net/FFA-Net/trained_models/its_train_ffa_3_20.pk`, mod `haze_ffa_its`.  
- **OTS (dış mekan):** 19 blok → `ots_train_ffa_3_19.pk`, mod `haze_ffa_ots`.  
- Girdi normalizasyonu: mean `[0.64, 0.6, 0.58]`, std `[0.14, 0.15, 0.152]`.  
- Yükleyici: `FFANetWrapper` — `DataParallel` (`module.`) önek temizleme destekli.

---

### 4. Yağmur: DDN (güncel — DeRaindrop yerine)

**Değişiklik özeti:** Önceki sürümde **DeRaindrop** (cam/lens **damlası**, GAN + maske) kullanılıyordu. Güncel kodda görev **yağmur çizgisi** gidermeye alındı; pipeline modu **`rain_ddn`**.

| Özellik | Eski (DeRaindrop) | Güncel (DDN) |
|---------|-------------------|--------------|
| Bozulma tipi | Lens üzeri damla | Görüntüde yağmur çizgileri |
| Mod adı | `raindrop` | **`rain_ddn`** (`raindrop` yalnızca geriye uyum alias) |
| Mimari | Attentive GAN + LSTM | Guided filter + 12× çift konv detay ağı |
| Çözünürlük | 4’e hizalı kırpma | **Girdi ile aynı H×W** |
| Ağırlık | `gen.pkl` | `Deep_Detailed_Network-PyTorch-master/model/rain100L/model_best.pth` |
| Kod | Harici DeRaindrop repo | `src/ddn_model.py` + `DDNWrapper` |

**DDN işleyişi** (`DeRain` sınıfı):

1. `base = guided_filter(girdi, girdi)` — taban ayrımı (CUDA sabitli upstream yerine **cihaz güvenli** `box_filter`).  
2. `detail = girdi − base` — çizgi/detay katmanı.  
3. Detay üzerinde derin konvolüsyon yığını → `neg_residual`.  
4. **`çıktı = girdi + neg_residual`** (çizgileri bastırır).

**Varyantlar:** `rain100L` (varsayılan, hafif yağmur) / `rain100H` (şiddetli); `DDNWrapper(variant=...)`.

**WeatherDetector:** Koşul etiketi `rain` (eski `raindrop` değil); skor anahtarı `score_rain`.

---

### 5. Nesne tespiti: YOLOv8

- **YOLOv8n**; güven ve IoU ayarlanabilir (`YoloV8Detector`).

---

### 6. CCM Combined (yalnızca karşılaştırma — pipeline dışı)

- LOL **eval15** üzerinde HVI-CIDNet ile kıyas (`output/CCM_vs_HVI-CIDNet_eval15_benchmark.md`).  
- Poster: *“Hibrit sistemde düşük ışık = HVI-CIDNet; CCM = benchmark.”*

---

## DENEYSEL BULGULAR

### Tablo 1 — Düşük ışık, LOL eval15 (15 görüntü ortalaması)

| Yöntem | Ort. MAE ↓ | Ort. PSNR ↑ | MAE iyileşme % |
|--------|------------|-------------|----------------|
| Düşük ışıklı girdi | 0,391 | 7,77 | — |
| CCM Combined *(benchmark)* | 0,110 | 19,01 | 70,2 |
| **HVI-CIDNet** *(pipeline)* | **0,082** | **21,26** | **73,5** |

- HVI-CIDNet, 15 örnekten **11’inde** MAE açısından CCM’den üstün.  
- Ortalama süre: CCM ~21 ms/görüntü, HVI-CIDNet ~115 ms/görüntü (CUDA).

---

### Tablo 2 — Sis giderme, SOTS (FFA-Net, pipeline ağırlıkları)

Pipeline’da kullanılan **ITS / OTS** eğitimli ağırlıklar, literatürde **SOTS** (RESIDE) üzerinde aynı FFA-Net mimarisiyle raporlanan sonuçlardır (Qin et al., AAAI 2020; `FFA-Net/FFA-Net/README.md`).

| Test seti | Ağırlık (pipeline) | PSNR ↑ (dB) | SSIM ↑ |
|-----------|-------------------|-------------|--------|
| **SOTS indoor** | ITS (`its_train_ffa_3_20.pk`) | **36,39** | **0,9886** |
| **SOTS outdoor** | OTS (`ots_train_ffa_3_19.pk`) | **33,57** | **0,9840** |

**Karşılaştırma (aynı makale, SOTS indoor/outdoor PSNR/SSIM):**

| Yöntem | Indoor | Outdoor |
|--------|--------|---------|
| DCP | 16,62 / 0,818 | 19,13 / 0,815 |
| DehazeNet | 21,14 / 0,847 | 22,46 / 0,851 |
| **FFA-Net (bizim pipeline)** | **36,39 / 0,989** | **33,57 / 0,984** |

*Poster grafik önerisi:* Indoor/outdoor için 2 çubuklu PSNR grafiği (DCP / DehazeNet / FFA-Net).

**Yerel yeniden ölçüm:** SOTS `hazy/` + `clear/` klasörleri eklendiğinde:

```bash
python output/ffa_sots_benchmark_runner.py
```

çıktı: `output/FFA_SOTS_benchmark.md`, `output/ffa_sots_benchmark_results.csv` (sisli girdi vs FFA, görüntü başına PSNR/SSIM/MAE).

---

### Tablo 3 — Yağmur çizgisi, Rain100 (DDN, pipeline ağırlığı)

Rain100L varsayılan ağırlık ile (DDN README / CVPR 2017):

| Test seti | PSNR ↑ (dB) | Pipeline ağırlığı |
|-----------|-------------|-------------------|
| **Rain100L** | **33,58** | `model/rain100L/model_best.pth` |
| Rain100H | 23,91 | `model/rain100H/model_best.pth` (isteğe bağlı) |

*Not:* Rain100H daha zor sahneler; Gradio’da `DDNWrapper(variant="rain100H")` ile seçilebilir.

---

### Şekil 2 — Görsel paneller (öneri)

| Panel | İçerik / dosya |
|--------|----------------|
| Düşük ışık | `output/comparison_figures/79_four_panel_comparison.png` |
| Sis (FFA) | SOTS veya pipeline çıktısı (önce/sonra) |
| Yağmur (DDN) | Rain100 örneği (çizgi giderme) |
| YOLO | Orijinal vs restore tespit kutuları |

---

## SONUÇ

- Sis, **yağmur çizgisi (DDN)** ve düşük ışık için **üç SOTA tabanlı modül** tek `UnifiedPipeline`’da birleştirildi.  
- **DeRaindrop → DDN** geçişi ile bozulma tipi (damla → çizgi) ve mod adı (`rain_ddn`) netleştirildi; çıktı çözünürlüğü korunur.  
- Düşük ışıkta pipeline’da **yalnızca HVI-CIDNet**; CCM karşılaştırmalı benchmark’tır.  
- Sis tarafında FFA-Net, SOTS’ta **36,39 / 33,57 dB PSNR** (literatür, entegre ağırlıklar).  
- **YOLOv8** ile restorasyonun tespite etkisi nicel ve görsel izlenebilir; Gradio + toplu CSV/JSON rapor.

**Gelecek çalışmalar:** SOTS/Rain100 üzerinde yerel PSNR/SSIM toplu ölçüm (script hazır); YOLO güven artışının istatistiksel testi; gerçek İHA verisi.

---

## TEŞEKKÜR

Danışman **Fulya Akdeniz**’e, KOSTÜ Mühendislik ve Doğa Bilimleri Fakültesi’ne ve bitirme projesi sergisine katkı sunan tüm akademik personele teşekkür ederiz.

---

## KAYNAKÇA (özet)

1. Fu, X. et al., *Removing Rain from Single Images via a Deep Detailed Network*, CVPR 2017 (DDN).  
2. HVI-CIDNet — low-light enhancement (arXiv:2402.05809).  
3. Qin, X. et al., *FFA-Net: Feature Fusion Attention Network for Single Image Dehazing*, AAAI 2020.  
4. Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics  
5. LOL Dataset — low-light paired benchmark.  
6. RESIDE / SOTS — dehazing benchmark.  
7. Rain100 — joint rain removal (Peking University).

---

## Canva’da kısa başlık önerileri

- **Ana başlık:** Çoklu Hava Koşullarında Görüntü İyileştirme ve Nesne Algılama  
- **Alt başlık:** HVI-CIDNet · FFA-Net · DDN · YOLOv8 Hibrit Pipeline  
- **Şekil 1:** Otomatik koşul seçimi ve koşula özel restorasyon + YOLOv8  
- **Şekil 2:** Düşük ışık, sis (FFA), yağmur çizgisi (DDN) örnek çıktıları  
