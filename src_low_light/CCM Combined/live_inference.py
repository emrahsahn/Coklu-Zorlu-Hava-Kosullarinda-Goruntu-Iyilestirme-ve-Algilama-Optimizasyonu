"""
Real-time CCM Model Inference
Webcam veya video dosyasından düşük ışık görüntüsünü iyileştir

Kullanım:
  -- Webcam (kamera cihazı) --
    python "src_low_light/CCM Combined/live_inference.py" --mode webcam --checkpoint "src_low_light/checkpoints/ccm/best_combined_model.pth"

  -- Video dosyası --
    python "src_low_light/CCM Combined/live_inference.py" --mode video --input video.mp4 --checkpoint "src_low_light/checkpoints/ccm/best_combined_model.pth" --output output_enhanced.mp4

  -- Ekran üzerinde göster (real-time) --
    python "src_low_light/CCM Combined/live_inference.py" --mode webcam --display True --checkpoint "src_low_light/checkpoints/ccm/best_combined_model.pth"
"""

import torch
import cv2
import numpy as np
import argparse
import os
from datetime import datetime
import time
from pathlib import Path

# Model import
from ccm_model import CombinedEnhancementModule, check_device_info, load_trained_model

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_COMBINED_CHECKPOINT = str(PROJECT_ROOT / 'src_low_light' / 'checkpoints' / 'ccm' / 'best_combined_model.pth')


class LiveInference:
    """Real-time görüntü iyileştirme"""
    def __init__(self, checkpoint_path, device, image_size=256):
        self.device = device
        self.image_size = image_size
        
        # Model yükle
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = load_trained_model(checkpoint_path, device)
            print(f"[MODEL] Eğitilmiş model yüklendi: {checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint bulunamadı. Eğitimsiz model kullanılıyor.")
            self.model = CombinedEnhancementModule().to(device)
        
        self.model.eval()
    
    def preprocess(self, frame):
        """OpenCV BGR frame -> torch tensor [0, 1]"""
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        rgb_resized = cv2.resize(rgb_frame, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        rgb_tensor = torch.from_numpy(rgb_resized).float() / 255.0
        
        # CHW format: [H, W, C] -> [C, H, W]
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return rgb_tensor.to(self.device)
    
    def postprocess(self, output_tensor, original_h, original_w):
        """torch tensor -> OpenCV BGR frame"""
        # Squeeze batch dimension
        output_tensor = output_tensor.squeeze(0)  # [3, H, W]
        
        # CHW -> HWC
        output_np = output_tensor.permute(1, 2, 0).cpu().detach().numpy()
        
        # Clamp to [0, 1]
        output_np = np.clip(output_np, 0, 1)
        
        # Scale to [0, 255]
        output_np = (output_np * 255).astype(np.uint8)
        
        # RGB -> BGR
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        # Restore original size
        output_bgr = cv2.resize(output_bgr, (original_w, original_h))
        
        return output_bgr
    
    def infer(self, frame):
        """
        Tek bir frame'i iyileştir
        
        Returns:
            enhanced_frame: İyileştirilmiş BGR frame
            fps: Saniyede kaç frame işlendiği
        """
        start_time = time.time()
        
        original_h, original_w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Inference
        with torch.no_grad():
            output, enhanced_illum, gamma_curves, intensity_scale, illum_map = self.model(input_tensor)
        
        # Postprocess
        enhanced_frame = self.postprocess(output, original_h, original_w)
        
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        return enhanced_frame, fps


def webcam_inference(checkpoint_path, device, display=True):
    """WebCam'den canlı iyileştirme yap"""
    print("\n" + "="*80)
    print("[WEBCAM] Canlı kamera akışı")
    print("="*80)
    print("Kapatmak için 'q' tuşuna bas\n")
    
    inferencer = LiveInference(checkpoint_path, device)
    
    cap = cv2.VideoCapture(0)  # 0 = built-in camera
    
    if not cap.isOpened():
        print("[ERROR] Kamera açılamadı!")
        return
    
    frame_count = 0
    fps_hist = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame okunamadı")
            break
        
        # Infer
        enhanced_frame, fps = inferencer.infer(frame)
        fps_hist.append(fps)
        
        frame_count += 1
        
        if display:
            # Side-by-side comparison
            h, w = frame.shape[:2]
            
            # Resize for display
            display_h = 480
            display_w = int(w * display_h / h)
            
            original_display = cv2.resize(frame, (display_w, display_h))
            enhanced_display = cv2.resize(enhanced_frame, (display_w, display_h))
            
            # Birleştir
            combined = np.hstack([original_display, enhanced_display])
            
            # Stats ekle
            avg_fps = np.mean(fps_hist[-30:]) if fps_hist else 0
            cv2.putText(combined, f"Original | Enhanced (FPS: {avg_fps:.1f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("CCM Live Enhancement", combined)
            
            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 30 == 0:
            avg_fps = np.mean(fps_hist[-30:]) if fps_hist else 0
            print(f"[INFO] Frame {frame_count} | Avg FPS: {avg_fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[COMPLETE] İşlenen frame: {frame_count}")
    if fps_hist:
        print(f"[STATS] Ortalama FPS: {np.mean(fps_hist):.2f}")


def video_inference(input_path, output_path, checkpoint_path, device, display=False):
    """Video dosyasını iyileştir"""
    print("\n" + "="*80)
    print(f"[VIDEO] Video işleniyor: {input_path}")
    print("="*80 + "\n")
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Video dosyası bulunamadı: {input_path}")
        return
    
    inferencer = LiveInference(checkpoint_path, device)
    
    # Video capture
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("[ERROR] Video açılamadı!")
        return
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total frames: {total_frames}\n")
    
    # Video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None
    
    frame_count = 0
    fps_hist = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Infer
        enhanced_frame, infer_fps = inferencer.infer(frame)
        fps_hist.append(infer_fps)
        
        # Write
        if out:
            out.write(enhanced_frame)
        
        if display:
            combined = np.hstack([frame, enhanced_frame])
            cv2.imshow("Original | Enhanced", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            avg_infer_fps = np.mean(fps_hist[-30:])
            progress = (frame_count / total_frames) * 100
            print(f"[PROGRESS] {frame_count}/{total_frames} ({progress:.1f}%) | Infer FPS: {avg_infer_fps:.2f}")
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n[COMPLETE] İşlenen frame: {frame_count}")
    if fps_hist:
        print(f"[STATS] Ortalama inference FPS: {np.mean(fps_hist):.2f}")
    
    if output_path:
        print(f"[SAVED] Çıktı video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CCM Real-time Inference')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'video'],
                       help='webcam: Canlı kamera | video: Video dosyası')
    parser.add_argument('--input', type=str, default=None,
                       help='Video dosyası yolu (--mode video olduğunda gerekli)')
    parser.add_argument('--output', type=str, default=None,
                       help='Çıktı video dosyası yolu (opsiyonel)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_COMBINED_CHECKPOINT,
                       help='Eğitilmiş model checkpoint yolu')
    parser.add_argument('--display', type=bool, default=True,
                       help='Ekranda göster')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Model input boyutu')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CCM Real-time Enhancement")
    print("="*80)
    
    device = check_device_info()
    
    if args.mode == 'webcam':
        webcam_inference(args.checkpoint, device, display=args.display)
    
    elif args.mode == 'video':
        if not args.input:
            print("[ERROR] Video modu için --input argümanı gerekli")
            return
        
        video_inference(args.input, args.output, args.checkpoint, device, display=args.display)
    
    print("\n[COMPLETE] Tum islemler tamamlandi!")


if __name__ == "__main__":
    main()
