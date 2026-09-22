# CCM vs HVI-CIDNet Karsilastirma Raporu

Bu rapor, proje klasorundeki iki modelin mevcut egitilmis agirliklari ve urettigi ciktilar kullanilarak hazirlanmistir.

## 1) Kullanilan Checkpoint ve Cikti Konumlari

- CCM checkpoint klasoru:
  - `src_low_light/checkpoints/ccm`
  - Kullanilan agirlik: `best_combined_model.pth`
- HVI-CIDNet checkpoint klasoru:
  - `HVI-CIDNet/weights/train`
  - Kullanilan agirlik: `epoch_100.pth`
- Ciktilar:
  - CCM: `output/ccm_tests`
  - HVI-CIDNet: `output/hvi_cidnet_tests`
  - CIDNet (ayri test script ciktilari): `output/cidnet_tests`

Not: Kullanici mesajinda gecen `src_low_light/CCM Combined/checkpoints` klasoru bu repoda bulunmuyor; CCM agirliklari `src_low_light/checkpoints/ccm` altinda.

## 2) Karsilastirma Kurulumu

- Eslestirilmis (paired) test icin `LOL eval15` setinden `79.png` kullanildi:
  - Dusuk isik: `data/lol_dataset/eval15/low/79.png`
  - Ground truth: `data/lol_dataset/eval15/high/79.png`
- Her iki model de ayni giris ve ayni ground truth ile degerlendirildi.
- Hesaplanan metrikler:
  - MAE (Mean Absolute Error): Dusuk olmasi daha iyi
  - PSNR: Yuksek olmasi daha iyi
  - Baseline iyilesme (%): `((MAE_low - MAE_model) / MAE_low) * 100`

## 3) Sayisal Sonuclar (79.png)

- Baseline (Low-Light -> GT):
  - `MAE = 0.540976`
  - `PSNR = 5.0899`
- CCM (best_combined_model.pth):
  - `MAE = 0.166353`
  - `PSNR = 15.3453`
  - `Iyilesme = 69.25%`
- HVI-CIDNet (epoch_100.pth):
  - `MAE = 0.078296`
  - `PSNR = 21.7231`
  - `Iyilesme = 85.53%`

## 4) Yorum

- Her iki model de baseline dusuk isik goruntuye gore belirgin iyilesme sagliyor.
- Bu test orneginde (`79.png`) HVI-CIDNet, hem MAE hem PSNR tarafinda CCM'den daha iyi.
- `85.53%` iyilesme ile HVI-CIDNet, CCM'nin `69.25%` degerinin uzerinde performans vermis.

## 5) Gorsel Cikti Referanslari

- CCM ornekleri:
  - `output/ccm_tests/79_detailed_comparison.png`
  - `output/ccm_tests/98_detailed_comparison.png`
- HVI-CIDNet ornegi:
  - `output/hvi_cidnet_tests/98_hvi_cidnet_output.png`
  - `output/hvi_cidnet_tests/98_hvi_cidnet_comparison.png`

## 6) Sinirlar ve Sonraki Adim

- Bu rapordaki sayisal kiyas su an tek paired goruntu (`79.png`) uzerinden yapilmistir.
- Daha guclu sonuclandirima icin tum `eval15` setinde toplu olcum (ortalama MAE/PSNR/SSIM/LPIPS) onerilir.
- Istersen bir sonraki adimda iki modeli tum `eval15` icin batch calistirip otomatik bir "leaderboard" raporu (csv + md) olusturabilirim.
