# FFA-Net SOTS Benchmark (Pipeline entegrasyonu)

- Device: `cuda`
- ITS ağırlık: `its_train_ffa_3_20.pk` → SOTS **indoor**
- OTS ağırlık: `ots_train_ffa_3_19.pk` → SOTS **outdoor**

## SOTS indoor (ITS, N=500)

- Ort. PSNR: Sisli girdi `11.9748` | FFA `16.3532` dB
- Ort. SSIM: Sisli girdi `0.6934` | FFA `0.7101`
- Ort. MAE: Sisli girdi `0.229726` | FFA `0.130694` (iyileşme %37.88)
- Ort. süre: `304.69` ms/görüntü

## SOTS outdoor (OTS, N=500)

- Ort. PSNR: Sisli girdi `15.9194` | FFA `32.1257` dB
- Ort. SSIM: Sisli girdi `0.8139` | FFA `0.9792`
- Ort. MAE: Sisli girdi `0.146410` | FFA `0.020571` (iyileşme %84.37)
- Ort. süre: `193.09` ms/görüntü

## Dosyalar
- `output\ffa_sots_benchmark_results.csv`
- `output\FFA_SOTS_benchmark.md`
