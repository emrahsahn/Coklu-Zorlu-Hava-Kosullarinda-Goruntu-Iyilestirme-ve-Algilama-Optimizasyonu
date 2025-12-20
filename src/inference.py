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


def enhance_image(image_path, model_path, output_path, device='cpu', image_size=256):
    """
    Enhance a single low-light image
    
    Args:
        image_path: Path to input low-light image
        model_path: Path to trained model checkpoint
        output_path: Path to save enhanced image
        device: Device to use ('cpu' or 'cuda')
        image_size: Size to resize image for processing
    """
    # Initialize pipeline
    print(f"Loading model from {model_path}...")
    pipeline = CIDNetPipeline(device=device, base_channels=32, num_heads=4)
    pipeline.load_checkpoint(model_path)
    pipeline.eval_mode()
    
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
        enhanced_rgb, _, _, _, _ = pipeline.forward(low_rgb)
    
    # Convert back to PIL image
    enhanced_tensor = enhanced_rgb[0].cpu()
    to_pil = transforms.ToPILImage()
    enhanced_img = to_pil(enhanced_tensor)
    
    # Resize to original size
    enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)
    
    # Save
    enhanced_img.save(output_path)
    print(f"Enhanced image saved to: {output_path}")
    
    return enhanced_img


def enhance_folder(input_folder, model_path, output_folder, device='cpu', image_size=256):
    """
    Enhance all images in a folder
    
    Args:
        input_folder: Folder containing low-light images
        model_path: Path to trained model checkpoint
        output_folder: Folder to save enhanced images
        device: Device to use ('cpu' or 'cuda')
        image_size: Size to resize images for processing
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize pipeline once
    print(f"Loading model from {model_path}...")
    pipeline = CIDNetPipeline(device=device, base_channels=32, num_heads=4)
    pipeline.load_checkpoint(model_path)
    pipeline.eval_mode()
    
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
                enhanced_rgb, _, _, _, _ = pipeline.forward(low_rgb)
            
            # Convert and save
            enhanced_tensor = enhanced_rgb[0].cpu()
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
    
    args = parser.parse_args()
    
    # Check if input is file or folder
    if os.path.isfile(args.input):
        # Single image
        enhance_image(
            args.input,
            args.model,
            args.output,
            device=args.device,
            image_size=args.image_size
        )
    elif os.path.isdir(args.input):
        # Folder of images
        enhance_folder(
            args.input,
            args.model,
            args.output,
            device=args.device,
            image_size=args.image_size
        )
    else:
        print(f"Error: Input path not found: {args.input}")


if __name__ == "__main__":
    # Example usage:
    # python src/inference.py --input test_low.jpg --output test_enhanced.jpg --model checkpoints/best_model.pth
    # python src/inference.py --input ./test_images/ --output ./enhanced_images/ --model checkpoints/best_model.pth --device cuda
    
    main()
