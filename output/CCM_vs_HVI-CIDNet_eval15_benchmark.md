# CCM vs HVI-CIDNet Eval15 Toplu Benchmark Raporu

Bu rapor `eval15` paired setinin tamaminda olusturulmustur.

## Kullanilan modeller
- CCM: `src_low_light\checkpoints\ccm\best_combined_model.pth`
- HVI-CIDNet: `HVI-CIDNet\weights\train\epoch_100.pth`
- Device: `cuda`

## Ozet Sonuclar
- Goruntu sayisi: `15`
- Ortalama MAE (dusuk daha iyi): Baseline `0.391367` | CCM `0.110301` | HVI-CIDNet `0.082353`
- Ortalama PSNR (yuksek daha iyi): Baseline `7.7733` | CCM `19.0104` | HVI-CIDNet `21.2618`
- Ortalama MAE iyilesme (%): CCM `70.16%` | HVI-CIDNet `73.48%`
- Ortalama sure (ms/goruntu): CCM `20.56` | HVI-CIDNet `114.74`
- MAE bazli kazanan sayisi: CCM `4` | HVI-CIDNet `11` | Beraberlik `0`

## Dosyalar
- Ayrintili satir bazli sonuclar: `output\eval15_benchmark_results.csv`
- Bu rapor: `output\CCM_vs_HVI-CIDNet_eval15_benchmark.md`
