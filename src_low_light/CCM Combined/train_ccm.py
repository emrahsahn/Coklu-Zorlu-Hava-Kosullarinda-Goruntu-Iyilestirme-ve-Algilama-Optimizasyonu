"""
CCM Training Script
Her modül için ayrı eğitim ve test desteği

Kullanım:
  -- IlluminationEnhancementModule --
    python src/train_ccm.py --mode train_illum --epochs 50
    python src/train_ccm.py --mode test_illum  --checkpoint src/checkpoints/ccm/best_illum_model.pth

  -- ColorCorrectionModule --
    python src/train_ccm.py --mode train_ccm   --epochs 50
    python src/train_ccm.py --mode test_ccm    --checkpoint src/checkpoints/ccm/best_ccm_model.pth

  -- CombinedEnhancementModule (Illum + CCM birlikte) --
    python src/train_ccm.py --mode train_combined --epochs 50
    python src/train_ccm.py --mode test_combined  --checkpoint src/checkpoints/ccm/best_combined_model.pth

  -- Hepsini birden eğit ve karşılaştır --
    python src/train_ccm.py --mode compare --epochs 30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse
from datetime import datetime
from tqdm import tqdm

# Model import
from ccm_model import (
    IlluminationEnhancementModule,
    ColorCorrectionModule,
    CombinedEnhancementModule,
    CombinedLoss,
    ColorConstancyLoss,
    check_device_info,
    load_trained_model,
    load_illum_model,
    load_ccm_model,
)


class LOLDataset(Dataset):
    """LOL Dataset for low-light image enhancement training"""
    def __init__(self, root_dir, split='train', transform=None, image_size=256):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        
        # Find image pairs
        if split == 'train':
            low_dir = os.path.join(root_dir, 'our485', 'low')
            high_dir = os.path.join(root_dir, 'our485', 'high')
        else:  # eval
            low_dir = os.path.join(root_dir, 'eval15', 'low')
            high_dir = os.path.join(root_dir, 'eval15', 'high')
        
        # Check if directories exist
        if not os.path.exists(low_dir):
            raise FileNotFoundError(f"Low light directory not found: {low_dir}")
        if not os.path.exists(high_dir):
            raise FileNotFoundError(f"High light directory not found: {high_dir}")
        
        self.low_images = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.high_images = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"[DATASET] {split} set: {len(self.low_images)} images loaded")
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        low_img = Image.open(self.low_images[idx]).convert('RGB')
        high_img = Image.open(self.high_images[idx]).convert('RGB')
        
        if self.transform:
            low_tensor = self.transform(low_img)
            high_tensor = self.transform(high_img)
        
        return {
            'low': low_tensor, 
            'high': high_tensor, 
            'name': os.path.basename(self.low_images[idx])
        }


def _run_training_loop(model, train_loader, val_loader, device, epochs, lr,
                       save_dir, log_dir, checkpoint_name, forward_fn, loss_fn, label):
    """
    Ortak eğitim döngüsü.
    - GPU varsa: AMP (FP16 mixed precision) + cudnn.benchmark + non_blocking transfer
    - GPU yoksa: standart CPU eğitimi
    forward_fn(model, low, high) -> (loss, loss_dict)
    """
    use_gpu = device.type == 'cuda'

    # --- GPU optimizasyonları ---
    if use_gpu:
        # Sabit boyutlu input için conv algoritmasını otomatik seç -> hız artar
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] cudnn.benchmark = True")

        # Automatic Mixed Precision (FP16 + FP32 karma) -> ~2x hız, daha az VRAM
        scaler = torch.cuda.amp.GradScaler()
        print(f"[GPU] Automatic Mixed Precision (AMP) aktif")
    else:
        scaler = None
        print(f"[CPU] AMP devre disi (sadece GPU destekler)")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{label}_{timestamp}')
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    print(f"\n{'='*80}")
    print(f"[TRAINING] {label} | Device: {device} | Epochs: {epochs} | LR: {lr}")
    if use_gpu:
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[TRAINING] GPU: {torch.cuda.get_device_name(0)} ({mem_total:.1f} GB)")
    print(f"[TRAINING] Checkpoint: {save_dir}/{checkpoint_name}")
    print(f"[TRAINING] TensorBoard: {log_path}")
    print(f"{'='*80}\n")

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0
        train_l1 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [{label} TRAIN]")
        for batch in pbar:
            # non_blocking=True: pin_memory ile async GPU transfer
            low  = batch['low'].to(device, non_blocking=True)
            high = batch['high'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # None atamak sıfırlamaktan hızlı

            if use_gpu:
                # AMP: forward pass FP16, loss FP32
                with torch.cuda.amp.autocast():
                    loss, loss_dict = forward_fn(model, low, high)
                # Scaled backward + optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = forward_fn(model, low, high)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_l1 += loss_dict['l1']

            postfix = {'loss': f"{loss.item():.4f}", 'l1': f"{loss_dict['l1']:.4f}"}
            if use_gpu:
                mem_used = torch.cuda.memory_reserved(0) / 1e9
                postfix['GPU_GB'] = f"{mem_used:.1f}"
            pbar.set_postfix(postfix)

        train_loss /= len(train_loader)
        train_l1 /= len(train_loader)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0

        with torch.no_grad():
            for batch in val_loader:
                low  = batch['low'].to(device, non_blocking=True)
                high = batch['high'].to(device, non_blocking=True)

                if use_gpu:
                    with torch.cuda.amp.autocast():
                        loss, loss_dict = forward_fn(model, low, high)
                else:
                    loss, loss_dict = forward_fn(model, low, high)

                val_loss += loss.item()
                val_l1 += loss_dict['l1']

        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        scheduler.step()

        # Validation sonrası GPU cache temizle
        if use_gpu:
            torch.cuda.empty_cache()

        # ---------- LOGGING ----------
        writer.add_scalar(f'{label}/Loss_train', train_loss, epoch)
        writer.add_scalar(f'{label}/Loss_val', val_loss, epoch)
        writer.add_scalar(f'{label}/L1_val', val_l1, epoch)
        writer.add_scalar(f'{label}/LR', scheduler.get_last_lr()[0], epoch)
        if use_gpu:
            writer.add_scalar(f'{label}/GPU_memory_GB', torch.cuda.memory_reserved(0) / 1e9, epoch)

        gpu_info = f" | GPU: {torch.cuda.memory_reserved(0)/1e9:.1f}GB" if use_gpu else ""
        print(f"[{label}] Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | L1: {val_l1:.4f}{gpu_info}")

        # ---------- CHECKPOINT ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(save_dir, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'val_l1': val_l1,
                'label': label,
            }, path)
            print(f"  [SAVED] Best model -> {path}  (val_loss={val_loss:.4f})")

        if epoch % 10 == 0:
            path = os.path.join(save_dir, f'{label}_epoch_{epoch}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, path)
            print(f"  [SAVED] Checkpoint -> {path}")

    writer.close()
    print(f"\n[{label}] Training done! Best Val Loss: {best_val_loss:.4f}\n")
    return model


# ============================================================================
# FORWARD FUNCTIONS (her modüle özel)
# ============================================================================

def _forward_illum(model, low, high):
    """IlluminationEnhancementModule için forward + loss"""
    enhanced, gamma_curves, boost = model(low)
    criterion = CombinedLoss(lambda_l1=1.0, lambda_cc=0.0, lambda_ssim=0.5)
    loss, loss_dict = criterion(enhanced, high, illum_map=None)
    return loss, loss_dict


def _forward_ccm(model, low, high):
    """ColorCorrectionModule için forward + loss"""
    corrected, illum_map = model(low)
    criterion = CombinedLoss(lambda_l1=1.0, lambda_cc=0.1, lambda_ssim=0.5)
    loss, loss_dict = criterion(corrected, high, illum_map=illum_map)
    return loss, loss_dict


def _forward_combined(model, low, high):
    """CombinedEnhancementModule için forward + loss"""
    corrected, enhanced_illum, gamma_curves, intensity_scale, illum_map = model(low)
    criterion = CombinedLoss(lambda_l1=1.0, lambda_cc=0.1, lambda_ssim=0.5)
    loss, loss_dict = criterion(corrected, high, illum_map=illum_map)
    return loss, loss_dict


# ============================================================================
# PUBLIC TRAIN FUNCTIONS
# ============================================================================

def train_illumination_model(train_loader, val_loader, device, epochs=50, lr=1e-4,
                              save_dir='src/checkpoints/ccm', log_dir='src/logs/ccm'):
    """Sadece IlluminationEnhancementModule'ü eğitir"""
    model = IlluminationEnhancementModule()
    return _run_training_loop(
        model, train_loader, val_loader, device, epochs, lr,
        save_dir, log_dir,
        checkpoint_name='best_illum_model.pth',
        forward_fn=_forward_illum,
        loss_fn=None,
        label='ILLUM'
    )


def train_ccm_model(train_loader, val_loader, device, epochs=50, lr=1e-4,
                    save_dir='src/checkpoints/ccm', log_dir='src/logs/ccm'):
    """Sadece ColorCorrectionModule'ü eğitir"""
    model = ColorCorrectionModule()
    return _run_training_loop(
        model, train_loader, val_loader, device, epochs, lr,
        save_dir, log_dir,
        checkpoint_name='best_ccm_model.pth',
        forward_fn=_forward_ccm,
        loss_fn=None,
        label='CCM'
    )


def train_combined_model(train_loader, val_loader, device, epochs=50, lr=1e-4,
                          save_dir='src/checkpoints/ccm', log_dir='src/logs/ccm'):
    """Illum + CCM birlikte (CombinedEnhancementModule) eğitir"""
    model = CombinedEnhancementModule()
    return _run_training_loop(
        model, train_loader, val_loader, device, epochs, lr,
        save_dir, log_dir,
        checkpoint_name='best_combined_model.pth',
        forward_fn=_forward_combined,
        loss_fn=None,
        label='COMBINED'
    )


# ============================================================================
# EVALUATION
# ============================================================================

def _get_output(model, low, module_type):
    """Modül tipine göre çıktıyı al"""
    if module_type == 'illum':
        enhanced, _, _ = model(low)
        return enhanced
    elif module_type == 'ccm':
        corrected, _ = model(low)
        return corrected
    else:  # combined
        corrected, _, _, _, _ = model(low)
        return corrected


def evaluate_model(model, val_loader, device, module_type='combined'):
    """
    Herhangi bir modülü değerlendirir.
    module_type: 'illum' | 'ccm' | 'combined'
    """
    model.eval()
    total_mae_original = 0.0
    total_mae_enhanced = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating [{module_type.upper()}]"):
            low = batch['low'].to(device)
            high = batch['high'].to(device)

            output = _get_output(model, low, module_type)

            total_mae_original += torch.abs(low - high).mean().item()
            total_mae_enhanced += torch.abs(output - high).mean().item()

    avg_orig = total_mae_original / len(val_loader)
    avg_enh = total_mae_enhanced / len(val_loader)
    improvement = (avg_orig - avg_enh) / avg_orig * 100

    label = module_type.upper()
    print(f"\n{'='*80}")
    print(f"[EVAL] {label} Sonuclari")
    print(f"{'='*80}")
    print(f"  Baseline MAE  (Low  -> GT):    {avg_orig:.6f}")
    print(f"  Enhanced MAE  ({label} -> GT): {avg_enh:.6f}")
    print(f"  Iyilestirme:                   {improvement:.2f}%")
    if improvement >= 83:
        print(f"  [SUCCESS] Hedef basariya ulasildi! {improvement:.2f}% >= 83%")
    else:
        print(f"  [INFO] Hedef: 83% | Mevcut: {improvement:.2f}%")
    print(f"{'='*80}\n")

    return improvement


def _make_loaders(dataset_path, image_size, batch_size):
    """Dataset ve DataLoader'ları oluşturur"""
    train_ds = LOLDataset(dataset_path, split='train', image_size=image_size)
    val_ds   = LOLDataset(dataset_path, split='eval',  image_size=image_size)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    print(f"[INFO] Train: {len(train_ds)} goruntu ({len(train_loader)} batch)")
    print(f"[INFO] Val:   {len(val_ds)} goruntu ({len(val_loader)} batch)")
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='CCM Training and Testing')
    parser.add_argument('--mode', type=str, default='train_illum',
                        choices=[
                            'train_illum',    # Sadece IlluminationEnhancementModule egit
                            'train_ccm',      # Sadece ColorCorrectionModule egit
                            'train_combined', # Illum + CCM birlikte egit
                            'compare',        # Uc modulu de egit ve karsilastir
                            'test_illum',     # IlluminationEnhancementModule test
                            'test_ccm',       # ColorCorrectionModule test
                            'test_combined',  # CombinedEnhancementModule test
                        ],
                        help='Calistirma modu')
    parser.add_argument('--epochs',      type=int,   default=50,                               help='Epoch sayisi')
    parser.add_argument('--batch_size',  type=int,   default=8,                                help='Batch size')
    parser.add_argument('--lr',          type=float, default=1e-4,                             help='Learning rate')
    parser.add_argument('--image_size',  type=int,   default=256,                              help='Goruntu boyutu')
    parser.add_argument('--dataset',     type=str,   default='data/lol_dataset',               help='Dataset yolu')
    parser.add_argument('--checkpoint',  type=str,   default=None,                             help='Test icin checkpoint yolu')
    parser.add_argument('--save_dir',    type=str,   default='src/checkpoints/ccm',            help='Checkpoint kayit klasoru')
    parser.add_argument('--log_dir',     type=str,   default='src/logs/ccm',                   help='TensorBoard log klasoru')
    args = parser.parse_args()

    print("="*80)
    print(f"CCM Training | Mode: {args.mode}")
    print("="*80)

    device = check_device_info()

    # ================================================================
    # TRAIN MODES
    # ================================================================
    if args.mode == 'train_illum':
        print("\n[MODE] IlluminationEnhancementModule Egitimi")
        print("  Input : dusuk isikli goruntu")
        print("  Target: normal isikli goruntu (GT)")
        print("  Loss  : L1 + SSIM  (renk sabiti yok)\n")
        try:
            train_loader, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}"); return

        model = train_illumination_model(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir, log_dir=args.log_dir
        )
        evaluate_model(model, val_loader, device, module_type='illum')

    elif args.mode == 'train_ccm':
        print("\n[MODE] ColorCorrectionModule Egitimi")
        print("  Input : dusuk isikli goruntu")
        print("  Target: normal isikli goruntu (GT)")
        print("  Loss  : L1 + SSIM + ColorConstancy\n")
        try:
            train_loader, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}"); return

        model = train_ccm_model(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir, log_dir=args.log_dir
        )
        evaluate_model(model, val_loader, device, module_type='ccm')

    elif args.mode == 'train_combined':
        print("\n[MODE] CombinedEnhancementModule Egitimi (Illum + CCM birlikte)")
        print("  Input : dusuk isikli goruntu")
        print("  Target: normal isikli goruntu (GT)")
        print("  Loss  : L1 + SSIM + ColorConstancy (combined ciktiya)\n")
        try:
            train_loader, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}"); return

        model = train_combined_model(
            train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir, log_dir=args.log_dir
        )
        evaluate_model(model, val_loader, device, module_type='combined')

    elif args.mode == 'compare':
        print("\n[MODE] Karsilastirmali Egitim: Illum vs CCM vs Combined")
        try:
            train_loader, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}"); return

        results = {}

        print("\n" + "─"*80)
        print("ADIM 1/3: IlluminationEnhancementModule")
        print("─"*80)
        m_illum = train_illumination_model(train_loader, val_loader, device,
                                           epochs=args.epochs, lr=args.lr,
                                           save_dir=args.save_dir, log_dir=args.log_dir)
        results['illum'] = evaluate_model(m_illum, val_loader, device, 'illum')

        print("\n" + "─"*80)
        print("ADIM 2/3: ColorCorrectionModule")
        print("─"*80)
        m_ccm = train_ccm_model(train_loader, val_loader, device,
                                epochs=args.epochs, lr=args.lr,
                                save_dir=args.save_dir, log_dir=args.log_dir)
        results['ccm'] = evaluate_model(m_ccm, val_loader, device, 'ccm')

        print("\n" + "─"*80)
        print("ADIM 3/3: CombinedEnhancementModule")
        print("─"*80)
        m_combined = train_combined_model(train_loader, val_loader, device,
                                          epochs=args.epochs, lr=args.lr,
                                          save_dir=args.save_dir, log_dir=args.log_dir)
        results['combined'] = evaluate_model(m_combined, val_loader, device, 'combined')

        print("\n" + "="*80)
        print("KARSILASTIRMA OZETI")
        print("="*80)
        for name, imp in results.items():
            star = " <-- EN IYI" if imp == max(results.values()) else ""
            print(f"  {name.upper():12s}: {imp:.2f}% iyilestirme{star}")
        print("="*80)

    # ================================================================
    # TEST MODES
    # ================================================================
    elif args.mode == 'test_illum':
        ckpt = args.checkpoint or os.path.join(args.save_dir, 'best_illum_model.pth')
        print(f"\n[MODE] IlluminationEnhancementModule Test | Checkpoint: {ckpt}")
        model = load_illum_model(ckpt, device)
        try:
            _, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
            evaluate_model(model, val_loader, device, 'illum')
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")

    elif args.mode == 'test_ccm':
        ckpt = args.checkpoint or os.path.join(args.save_dir, 'best_ccm_model.pth')
        print(f"\n[MODE] ColorCorrectionModule Test | Checkpoint: {ckpt}")
        model = load_ccm_model(ckpt, device)
        try:
            _, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
            evaluate_model(model, val_loader, device, 'ccm')
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")

    elif args.mode == 'test_combined':
        ckpt = args.checkpoint or os.path.join(args.save_dir, 'best_combined_model.pth')
        print(f"\n[MODE] CombinedEnhancementModule Test | Checkpoint: {ckpt}")
        model = load_trained_model(ckpt, device)
        try:
            _, val_loader = _make_loaders(args.dataset, args.image_size, args.batch_size)
            evaluate_model(model, val_loader, device, 'combined')
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")

    print("\n[COMPLETE] Tum islemler tamamlandi!")


if __name__ == "__main__":
    main()
