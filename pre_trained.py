

import torch
import numpy as np
import cv2
from PIL import Image
import argparse
import time
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# First install: pip install compressai
try:
    from compressai.zoo import (
        bmshj2018_factorized,
        mbt2018_mean,
        cheng2020_anchor,
        bmshj2018_hyperprior,
        mbt2018
    )
except ImportError:
    print("ERROR: CompressAI not installed!")
    print("Install it with: pip install compressai")
    print("Then run this script again.")
    exit(1)


class PretrainedCompressor:
    """Use pre-trained compression models"""
    
    def __init__(self, model_name='cheng2020_anchor', quality=5, device='cuda'):
        """
        Args:
            model_name: Model to use
                - 'bmshj2018_factorized': Basic (fast)
                - 'mbt2018_mean': Good
                - 'cheng2020_anchor': Excellent (RECOMMENDED)
                - 'bmshj2018_hyperprior': Advanced
            quality: 1-8 (higher = better quality, larger file)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.quality = quality
        
        print(f"Loading pre-trained model: {model_name} (quality={quality})")
        
        # Load pre-trained model
        if model_name == 'bmshj2018_factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True)
        elif model_name == 'mbt2018_mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True)
        elif model_name == 'cheng2020_anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True)
        elif model_name == 'bmshj2018_hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def compress_image(self, image_path):
        """Compress an image and return results"""
        
        
        img = Image.open(image_path).convert('RGB')
        
        
        min_size = 256
        if img.size[0] < min_size or img.size[1] < min_size:
            print(f"Image too small ({img.size[0]}x{img.size[1]}), resizing to {min_size}x{min_size}")
            img = img.resize((min_size, min_size), Image.LANCZOS)
        
       
        w, h = img.size
        new_w = ((w + 63) // 64) * 64
        new_h = ((h + 63) // 64) * 64
        
        if w != new_w or h != new_h:
            print(f"Adjusting size from {w}x{h} to {new_w}x{new_h} (must be multiple of 64)")
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor
        x = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Compress and decompress
        with torch.no_grad():
            
            start = time.time()
            compressed = self.model.compress(x)
            encode_time = (time.time() - start) * 1000
            
           
            start = time.time()
            decompressed = self.model.decompress(compressed["strings"], compressed["shape"])
            decode_time = (time.time() - start) * 1000
            
            reconstructed = decompressed["x_hat"].clamp(0, 1)
        
        # Convert back to numpy
        reconstructed_np = reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Calculate metrics
        mse = np.mean((img_array - reconstructed_np) ** 2)
        psnr_val = psnr(img_array, reconstructed_np, data_range=1.0)
        ssim_val = ssim(img_array, reconstructed_np, data_range=1.0, channel_axis=2)
        
        # Calculate compressed size
        num_bits = sum(len(s[0]) * 8 for s in compressed["strings"])
        compressed_size = num_bits / 8  # bytes
        original_size = img_array.nbytes
        compression_ratio = original_size / compressed_size
        
        results = {
            'original': img_array,
            'reconstructed': reconstructed_np,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'mse': mse,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'latency': encode_time + decode_time,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'num_bits': num_bits
        }
        
        return results
    
    def save_comparison(self, results, output_path):
        """Save visual comparison"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(results['original'])
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Reconstructed
        axes[1].imshow(results['reconstructed'])
        axes[1].set_title(f"Reconstructed\nPSNR: {results['psnr']:.2f} dB | SSIM: {results['ssim']:.4f}", 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(results['original'] - results['reconstructed']) * 5
        axes[2].imshow(diff)
        axes[2].set_title(f"Difference (5x)\nMSE: {results['mse']:.6f}", 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f"Pre-trained Model Compression\n"
                    f"Compression: {results['compression_ratio']:.1f}:1 | "
                    f"Latency: {results['latency']:.1f}ms | "
                    f"Size: {results['compressed_size']/1024:.2f} KB",
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Use Pre-trained Compression Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use recommended model (Cheng2020, quality 5)
  python use_pretrained.py --image sample.jpg
  
  # Higher quality (better PSNR, larger file)
  python use_pretrained.py --image sample.jpg --quality 7
  
  # Faster model
  python use_pretrained.py --image sample.jpg --model bmshj2018_factorized --quality 5
  
  # Best quality model
  python use_pretrained.py --image sample.jpg --model cheng2020_anchor --quality 8

Models:
  bmshj2018_factorized - Fast, basic quality (PSNR ~30-35 dB)
  mbt2018_mean         - Good quality (PSNR ~32-37 dB)
  cheng2020_anchor     - Excellent quality (PSNR ~35-40 dB) ⭐ RECOMMENDED
  bmshj2018_hyperprior - Advanced (PSNR ~33-38 dB)
  
Quality levels: 1 (low) to 8 (high)
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='cheng2020_anchor',
                       choices=['bmshj2018_factorized', 'mbt2018_mean', 
                               'cheng2020_anchor', 'bmshj2018_hyperprior', 'mbt2018'],
                       help='Pre-trained model to use')
    parser.add_argument('--quality', type=int, default=5, choices=range(1, 9),
                       help='Quality level (1-8, higher is better)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./pretrained_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("PRE-TRAINED IMAGE COMPRESSION")
    print("=" * 70)
    
    # Create compressor
    compressor = PretrainedCompressor(
        model_name=args.model,
        quality=args.quality,
        device=args.device
    )
    
    print(f"\nCompressing: {args.image}")
    print("-" * 70)
    
    results = compressor.compress_image(args.image)
    
    
    print(f"\nResults:")
    print(f"  PSNR: {results['psnr']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f}")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"\nPerformance:")
    print(f"  Encoding: {results['encode_time']:.2f} ms")
    print(f"  Decoding: {results['decode_time']:.2f} ms")
    print(f"  Total latency: {results['latency']:.2f} ms")
    print(f"\nCompression:")
    print(f"  Original size: {results['original_size']/1024:.2f} KB")
    print(f"  Compressed size: {results['compressed_size']/1024:.2f} KB")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}:1")
    print(f"  Bits per pixel: {results['num_bits']/(results['original'].shape[0]*results['original'].shape[1]):.2f}")
    
   
    basename = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join(args.output_dir, f'{basename}_q{args.quality}_result.png')
    compressor.save_comparison(results, output_path)
    
    print(f"\nComparison saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
