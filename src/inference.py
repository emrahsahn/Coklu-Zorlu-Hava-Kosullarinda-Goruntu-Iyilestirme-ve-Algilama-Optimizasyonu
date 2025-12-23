"""
Inference script for CIDNet
Enhance low-light images using trained CIDNet model
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from app import CIDNetPipeline


def enhance_image(image_path, model_path, output_path, device='cpu', image_size=256, 
                 base_channels=32, num_heads=4, gamma=1.0, alpha_s=1.0, alpha_i=1.0):
    """
    Enhance a single low-light image
    
    Args:
        image_path: Path to input low-light image
        model_path: Path to trained model checkpoint
        output_path: Path to save enhanced image
        device: Device to use ('cpu' or 'cuda')
        image_size: Size to resize image for processing
        base_channels: Base channels for CIDNet (must match checkpoint)
        num_heads: Number of attention heads (must match checkpoint)
        gamma: Gamma correction parameter
        alpha_s: Saturation parameter
        alpha_i: Intensity parameter
    """
    # Initialize pipeline
    print(f"Loading model from {model_path}...")
    pipeline = CIDNetPipeline(device=device, base_channels=base_channels, num_heads=num_heads)
    pipeline.load_checkpoint(model_path)
    pipeline.eval_mode()
    
    # Set alpha parameters
    pipeline.cidnet.trans.alpha_s = alpha_s
    pipeline.cidnet.trans.alpha = alpha_i
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    low_rgb = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Enhance
    with torch.no_grad():
        enhanced_rgb = pipeline.forward(low_rgb, gamma)
    
    # Post-processing: Basic clamp only (no aggressive correction)
    enhanced_tensor = enhanced_rgb[0].cpu()
    enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
    
    # Convert back to PIL image
    to_pil = transforms.ToPILImage()
    enhanced_img = to_pil(enhanced_tensor)
    
    # Resize to original size
    enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)
    
    # Save
    enhanced_img.save(output_path)
    print(f"Enhanced image saved to: {output_path}")
    
    return enhanced_img


def enhance_folder(input_folder, model_path, output_folder, device='cpu', image_size=256, 
                  base_channels=32, num_heads=4, gamma=1.0, alpha_s=1.0, alpha_i=1.0):
    """
    Enhance all images in a folder
    
    Args:
        input_folder: Folder containing low-light images
        model_path: Path to trained model checkpoint
        output_folder: Folder to save enhanced images
        device: Device to use ('cpu' or 'cuda')
        image_size: Size to resize images for processing
        base_channels: Base channels (must match checkpoint)
        num_heads: Number of attention heads (must match checkpoint)
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize pipeline once
    print(f"Loading model from {model_path}...")
    pipeline = CIDNetPipeline(device=device, base_channels=base_channels, num_heads=num_heads)
    pipeline.load_checkpoint(model_path)
    pipeline.eval_mode()
    
    # Set alpha parameters
    pipeline.cidnet.trans.alpha_s = alpha_s
    pipeline.cidnet.trans.alpha = alpha_i
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"enhanced_{filename}")
        
        print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
        
        try:
            # Load and preprocess
            img = Image.open(input_path).convert('RGB')
            original_size = img.size
            
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
            
            low_rgb = transform(img).unsqueeze(0).to(device)
            
            # Enhance
            with torch.no_grad():
                enhanced_rgb = pipeline.forward(low_rgb, gamma)
            
            # Post-processing: Normalize and prevent over-brightness
            enhanced_tensor = enhanced_rgb[0].cpu()
            
            # Clamp to valid range [0, 1]
            enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
            
            # Fix green bias - balance RGB channels
            r_mean, g_mean, b_mean = enhanced_tensor[0].mean(), enhanced_tensor[1].mean(), enhanced_tensor[2].mean()
            target_mean = (r_mean + g_mean + b_mean) / 3
            
            enhanced_tensor[0] = enhanced_tensor[0] * (target_mean / (r_mean + 1e-6))
            enhanced_tensor[1] = enhanced_tensor[1] * (target_mean / (g_mean + 1e-6))
            enhanced_tensor[2] = enhanced_tensor[2] * (target_mean / (b_mean + 1e-6))
            
            # Aggressive tone mapping
            enhanced_tensor = torch.pow(enhanced_tensor + 1e-6, 0.7)
            
            # Reduce saturation
            gray = enhanced_tensor.mean(dim=0, keepdim=True)
            enhanced_tensor = 0.7 * enhanced_tensor + 0.3 * gray
            
            # Final clamp
            enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
            
            # Convert and save
            to_pil = transforms.ToPILImage()
            enhanced_img = to_pil(enhanced_tensor)
            enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)
            enhanced_img.save(output_path)
            
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
    
    print(f"\nCompleted! Enhanced images saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Enhance low-light images using CIDNet')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image file or folder')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for processing (default: 256)')
    parser.add_argument('--base_channels', type=int, default=8,
                       help='Base channels for model (must match checkpoint)')
    parser.add_argument('--num_heads', type=int, default=2,
                       help='Number of attention heads (must match checkpoint)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Gamma correction parameter (default: 1.0)')
    parser.add_argument('--alpha_s', type=float, default=1.0,
                       help='Saturation adjustment (default: 1.0)')
    parser.add_argument('--alpha_i', type=float, default=1.0,
                       help='Intensity adjustment (default: 1.0)')
    
    args = parser.parse_args()
    
    # Check if input is file or folder
    if os.path.isfile(args.input):
        # Single image
        enhance_image(
            args.input,
            args.model,
            args.output,
            device=args.device,
            image_size=args.image_size,
            base_channels=args.base_channels,
            num_heads=args.num_heads,
            gamma=args.gamma,
            alpha_s=args.alpha_s,
            alpha_i=args.alpha_i
        )
    elif os.path.isdir(args.input):
        # Folder of images
        enhance_folder(
            args.input,
            args.model,
            args.output,
            device=args.device,
            image_size=args.image_size,
            base_channels=args.base_channels,
            num_heads=args.num_heads,
            gamma=args.gamma,
            alpha_s=args.alpha_s,
            alpha_i=args.alpha_i
        )
    else:
        print(f"Error: Input path not found: {args.input}")


if __name__ == "__main__":
    # Example usage:
    # python src/inference.py --input test_low.jpg --output test_enhanced.jpg --model checkpoints/best_model.pth
    # python src/inference.py --input ./test_images/ --output ./enhanced_images/ --model checkpoints/best_model.pth --device cuda
    
    main()
