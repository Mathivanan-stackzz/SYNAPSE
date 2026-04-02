import torch
import numpy as np
import socket
import pickle
import struct
import cv2
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

try:
    from compressai.zoo import cheng2020_anchor
except ImportError:
    print("ERROR: pip install compressai")
    exit(1)

def load_model(model_path, quality=3):
    if model_path and os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            q = ckpt.get('quality', quality)
            m = cheng2020_anchor(quality=q, pretrained=False)
            m.load_state_dict(ckpt['model_state_dict'])
            print(f"✓ Custom model loaded (quality={q})")
            return m
        except Exception as e:
            print(f"✗ Custom model failed: {e}")
    
    print(f"Loading pre-trained model (quality={quality})...")
    m = cheng2020_anchor(quality=quality, pretrained=True)
    return m

def receive_one(conn):
    """Receive one photo from existing connection - SIMPLIFIED PROTOCOL"""
    conn.settimeout(30)
    
    raw = b''
    while len(raw) < 4:
        chunk = conn.recv(4 - len(raw))
        if not chunk:
            raise ConnectionError("Connection closed while reading size")
        raw += chunk
    
    size = struct.unpack('!I', raw)[0]
    print(f"    Expecting {size/1024:.1f} KB")
    
    data = b''
    while len(data) < size:
        chunk = conn.recv(min(65536, size - len(data)))
        if not chunk:
            raise ConnectionError(f"Connection closed, got {len(data)}/{size} bytes")
        data += chunk
    
    print(f"    Received {len(data)/1024:.1f} KB")
    
    packet = pickle.loads(data)
    conn.sendall(b'OK')
    
    return packet

def decompress_image(model, compressed_data):
    """Decompress image from compressed data"""
    shape = torch.Size(compressed_data['shape'])
    strings = compressed_data['strings']
    
    byte_strings = []
    for string_group in strings:
        byte_group = []
        for s in string_group:
            if isinstance(s, bytes):
                byte_group.append(s)
            elif isinstance(s, torch.Tensor):
                byte_group.append(bytes(s.cpu().numpy().tobytes()))
            else:
                byte_group.append(bytes(s))
        byte_strings.append(byte_group)
    
    with torch.no_grad():
        out = model.decompress(byte_strings, shape)
    
    x = out['x_hat'].squeeze(0).permute(1, 2, 0)
    x = torch.clamp(x, 0.0, 1.0)
    return (x.numpy() * 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V2X Receiver - 3 Photos')
    parser.add_argument('--port',    type=int, default=5000)
    parser.add_argument('--quality', type=int, default=3)
    parser.add_argument('--model',   default='./models/enhanced_v2x_best_q5.pth')
    parser.add_argument('--num_photos', type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print(f"V2X RECEIVER - {args.num_photos} PHOTOS MODE")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model = load_model(args.model, args.quality)
    model.eval()
    print("✓ Model ready")

    print(f"\n[2/3] Waiting for connection on port {args.port}...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', args.port))
    server.listen(1)
    
    conn, addr = server.accept()
    print(f"✓ Connected from {addr[0]}")
    print(f"\n[3/3] Receiving {args.num_photos} photos...")
    received_images = []
    
    try:
        for i in range(1, args.num_photos + 1):
            print(f"\n  {'─'*50}")
            print(f"  Photo {i}/{args.num_photos}")
            print(f"  {'─'*50}")
            
            print("  📡 Receiving...")
            packet = receive_one(conn)
    
            if isinstance(packet, dict) and 'compressed' in packet:
                compressed_data = packet['compressed']
                print(f"      Movement: {packet.get('movement_name', 'unknown')}")
            else:
                compressed_data = packet
            
            print("  🗜️  Decompressing...")
            raw_img = decompress_image(model, compressed_data)
            
            # Save the raw decompressed photo
            cv2.imwrite(f"received_photo_{i}.png", raw_img)
            print(f"  ✓ Saved photo {i}")
            
            received_images.append(raw_img)
            
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()
        server.close()
        print("\n  Connection closed")

    if len(received_images) == args.num_photos:
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: Received all {args.num_photos} photos")
        print(f"{'='*60}")
        
        print("\nCreating composite image...")
        h = min(img.shape[0] for img in received_images)
        resized = []
        for img in received_images:
            scale = h / img.shape[0]
            w = int(img.shape[1] * scale)
            resized.append(cv2.resize(img, (w, h)))
        
        composite = np.hstack(resized)
        cv2.imwrite("received_all_3_photos.png", composite)
        print("✓ Saved: received_all_3_photos.png")
        
        print("\nDisplaying composite...")
        cv2.imshow("V2X - All 3 Photos", composite)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print("Files saved:")
        for i in range(1, args.num_photos + 1):
            print(f"  • received_photo_{i}.png")
        print(f"  • received_all_{args.num_photos}_photos.png (composite)")
        print(f"{'='*60}")
        
    else:
        print(f"\n⚠ Only received {len(received_images)}/{args.num_photos} photos")
    
    print("\n[Done]")