"""
Training script for CIDNet
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import argparse
from datetime import datetime

from app import LOLDataset, CIDNetPipeline


def train_epoch(pipeline, dataloader, optimizer, epoch, writer, device):
    """Train for one epoch"""
    pipeline.train_mode()
    
    total_loss = 0.0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        low_rgb = batch['low'].to(device)
        high_rgb = batch['high'].to(device)
        
        # Forward pass and compute loss
        optimizer.zero_grad()
        loss, loss_dict, enhanced = pipeline.compute_loss(low_rgb, high_rgb, gamma=1.0)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, val in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            loss_components[key] += val
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/batch', loss.item(), global_step)
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, loss_components


def validate(pipeline, dataloader, epoch, writer, device):
    """Validation"""
    pipeline.eval_mode()
    
    total_loss = 0.0
    loss_components = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            low_rgb = batch['low'].to(device)
            high_rgb = batch['high'].to(device)
            
            # Forward pass
            loss, loss_dict, enhanced = pipeline.compute_loss(low_rgb, high_rgb, gamma=1.0)
            
            # Accumulate losses
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val
            
            # Save some sample images
            if batch_idx == 0:
                writer.add_images('Validation/low', low_rgb[:4], epoch)
                writer.add_images('Validation/enhanced', enhanced[:4], epoch)
                writer.add_images('Validation/target', high_rgb[:4], epoch)
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='Train CIDNet')
    parser.add_argument('--dataset_root', type=str, default='./data/LOL',
                       help='Path to LOL dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for training')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base channels for CIDNet')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, f'run_{timestamp}'))
    
    # Create datasets
    train_dataset = LOLDataset(args.dataset_root, subset='train', image_size=args.image_size)
    val_dataset = LOLDataset(args.dataset_root, subset='test', image_size=args.image_size)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows compatibility
        pin_memory=False  # More stable on Windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=False  # More stable on Windows
    )
    
    # Initialize pipeline
    print(f"\nInitializing CIDNet on device: {args.device}")
    pipeline = CIDNetPipeline(
        device=args.device,
        base_channels=args.base_channels,
        num_heads=args.num_heads
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in pipeline.cidnet.parameters())
    trainable_params = sum(p.numel() for p in pipeline.cidnet.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Setup optimizer
    params = list(pipeline.cidnet.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        pipeline.load_checkpoint(args.resume)
        # Optionally load optimizer state
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_components = train_epoch(
            pipeline, train_loader, optimizer, epoch, writer, args.device
        )
        
        # Log training metrics
        print(f"\nTraining - Average loss: {train_loss:.6f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        for key, val in train_components.items():
            writer.add_scalar(f'Loss/train_{key}', val, epoch)
            print(f"  {key}: {val:.6f}")
        
        # Validate
        if len(val_dataset) > 0:
            val_loss, val_components = validate(
                pipeline, val_loader, epoch, writer, args.device
            )
            
            print(f"\nValidation - Average loss: {val_loss:.6f}")
            writer.add_scalar('Loss/val', val_loss, epoch)
            for key, val in val_components.items():
                writer.add_scalar(f'Loss/val_{key}', val, epoch)
                print(f"  {key}: {val:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                pipeline.save_checkpoint(best_checkpoint_path)
                print(f"Saved best model with val_loss: {val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            pipeline.save_checkpoint(checkpoint_path)
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    pipeline.save_checkpoint(final_checkpoint_path)
    
    print("\nTraining completed!")
    writer.close()


if __name__ == "__main__":
    main()
