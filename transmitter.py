"""
V2X TRANSMITTER - FIXED VERSION

Properly serializes compressed tensors for network transmission
"""

import torch
import numpy as np
import socket
import pickle
import struct
import time
import cv2
import argparse
import warnings
import hashlib
warnings.filterwarnings("ignore")

try:
    from picamera2 import Picamera2
    PICAMERA = True
except ImportError:
    PICAMERA = False

try:
    import RPi.GPIO as GPIO
    MOTORS = True
except ImportError:
    MOTORS = False

from compressai.zoo import cheng2020_anchor


# ============ CONFIGURATION ============
LAPTOP_IP = "10.252.241.222"
PORT = 5000

F_L_IN1, F_L_IN2 = 5, 11
F_R_IN3, F_R_IN4 = 9, 10
R_L_IN1, R_L_IN2 = 27, 22
R_R_IN3, R_R_IN4 = 12, 13


# ============ MOTORS ============
class Motors:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        self.pins = [F_L_IN1, F_L_IN2, F_R_IN3, F_R_IN4,
                     R_L_IN1, R_L_IN2, R_R_IN3, R_R_IN4]
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        print("[OK] Motors initialized")
    
    def _set_motor(self, motor, direction):
        pin_map = {
            'fl': (F_L_IN1, F_L_IN2), 'fr': (F_R_IN3, F_R_IN4),
            'rl': (R_L_IN1, R_L_IN2), 'rr': (R_R_IN3, R_R_IN4)
        }
        if motor not in pin_map:
            return
        in1, in2 = pin_map[motor]
        if direction == 1:
            GPIO.output(in1, 1); GPIO.output(in2, 0)
        elif direction == -1:
            GPIO.output(in1, 0); GPIO.output(in2, 1)
        else:
            GPIO.output(in1, 0); GPIO.output(in2, 0)
    
    def _set_side(self, side, direction):
        sides = {'left': ['fl', 'rl'], 'right': ['fr', 'rr']}
        for m in sides.get(side, []):
            self._set_motor(m, direction)
    
    def _set_all(self, direction):
        for m in ['fl', 'fr', 'rl', 'rr']:
            self._set_motor(m, direction)
    
    def forward(self, duration=2.0):
        print(f"    Forward ({duration}s)...")
        self._set_all(1)
        time.sleep(duration)
        self.stop()
    
    def right(self, duration=0.7):
        print(f"    Right ({duration}s)...")
        self._set_side('left', 1)
        self._set_side('right', -1)
        time.sleep(duration)
        self.stop()
    
    def turn_around(self, duration=1.5):
        print(f"    Turn around ({duration}s)...")
        self._set_side('left', -1)
        self._set_side('right', 1)
        time.sleep(duration)
        self.stop()
    
    def stop(self):
        for pin in self.pins:
            GPIO.output(pin, 0)
    
    def cleanup(self):
        self.stop()
        GPIO.cleanup()


# ============ CAMERA ============
_camera = None

def init_camera():
    global _camera
    if _camera:
        return
    
    if PICAMERA:
        _camera = Picamera2()
        config = _camera.create_still_configuration(
            main={"size": (256, 256), "format": "BGR888"}
        )
        _camera.configure(config)
        _camera.start()
        time.sleep(2.0)
        print("[OK] Camera ready")
    else:
        _camera = cv2.VideoCapture(0)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

def capture_image():
    if PICAMERA:
        _ = _camera.capture_array()
        time.sleep(0.2)
        bgr = _camera.capture_array()
    else:
        for _ in range(3):
            ret, bgr = _camera.read()
        ret, bgr = _camera.read()
        if not ret:
            raise RuntimeError("Camera failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def close_camera():
    global _camera
    if _camera:
        if PICAMERA:
            _camera.stop()
            _camera.close()
        else:
            _camera.release()
        _camera = None


# ============ NETWORK ============
def connect_to_receiver(ip, port):
    """Connect with proper settings"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)  # Increased timeout
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    for attempt in range(5):
        try:
            print(f"    Attempt {attempt+1}/5...")
            sock.connect((ip, port))
            print(f"    ✓ Connected")
            return sock
        except socket.error as e:
            print(f"      Failed: {e}")
            time.sleep(1)
    return None


def recv_ack(conn, timeout=10):
    """Receive acknowledgment"""
    conn.settimeout(timeout)
    try:
        ack = conn.recv(2)
        return ack == b'OK'
    except socket.timeout:
        return False


def prepare_compressed_for_network(compressed):
    """
    Convert compressed data to network-serializable format
    """
    # Convert shape to tuple
    shape = tuple(compressed['shape']) if isinstance(compressed['shape'], torch.Size) else tuple(compressed['shape'])
    
    # Convert strings: list of list of byte tensors -> list of list of bytes
    strings = compressed['strings']
    serializable_strings = []
    
    for string_group in strings:
        serializable_group = []
        for s in string_group:
            if isinstance(s, torch.Tensor):
                # Convert tensor to bytes
                byte_data = bytes(s.cpu().numpy().tobytes())
                serializable_group.append(byte_data)
            elif isinstance(s, bytes):
                serializable_group.append(s)
            else:
                serializable_group.append(bytes(s))
        serializable_strings.append(serializable_group)
    
    return {
        'strings': serializable_strings,
        'shape': shape
    }


def send_photo(sock, movement_id, movement_name, compressed):
    """
    Send photo with proper serialization - SIMPLIFIED PROTOCOL
    """
    # Convert compressed data to serializable format
    compressed_serializable = prepare_compressed_for_network(compressed)
    
    packet = {
        'movement_id': movement_id,
        'movement_name': movement_name,
        'compressed': compressed_serializable
    }
    
    # Serialize
    raw_data = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
    size = len(raw_data)
    
    print(f"    Sending {size/1024:.1f} KB...")
    
    try:
        # Send size (4 bytes)
        sock.sendall(struct.pack('!I', size))
        
        # Send data
        sock.sendall(raw_data)
        
        # Wait for ACK
        if recv_ack(sock):
            print(f"    ✓ Acknowledged")
            return True
        else:
            print(f"    ✗ No ACK received")
            return False
            
    except socket.error as e:
        print(f"    ✗ Send error: {e}")
        return False


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default=LAPTOP_IP)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--test-motors', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("  V2X TRANSMITTER - FIXED")
    print("=" * 60)

    motors = None
    if MOTORS:
        try:
            motors = Motors()
        except Exception as e:
            print(f"[WARN] Motors: {e}")

    if args.test_motors and motors:
        motors.forward(1)
        motors.right(0.5)
        motors.turn_around(1)
        motors.cleanup()
        return

    try:
        init_camera()
    except Exception as e:
        print(f"[ERROR] Camera: {e}")
        return

    try:
        print("\n[1/4] Loading model...")
        model = cheng2020_anchor(quality=3, pretrained=True).eval()
        device = 'cpu'
        model = model.to(device)
        print("      Model ready")
    except Exception as e:
        print(f"[ERROR] Model: {e}")
        close_camera()
        return

    print(f"\n[2/4] Connecting to {args.ip}:{args.port}...")
    sock = connect_to_receiver(args.ip, args.port)
    if not sock:
        print("[ERROR] Connection failed!")
        close_camera()
        if motors:
            motors.cleanup()
        return

    movements = [
        (0, "FORWARD", lambda: motors.forward(2.0) if motors else time.sleep(2.0)),
        (1, "RIGHT_TURN", lambda: motors.right(0.7) if motors else time.sleep(0.7)),
        (2, "TURN_AROUND", lambda: motors.turn_around(1.5) if motors else time.sleep(1.5))
    ]

    print(f"\n[3/4] Executing...")
    try:
        for move_id, move_name, action in movements:
            print(f"\n  [{move_name}]")
            
            action()
            time.sleep(0.5)
            
            print("    Capturing...")
            frame = capture_image()
            cv2.imwrite(f"car_{move_name}.jpg", 
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            print("    Compressing...")
            tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                compressed = model.compress(tensor)
            
            print(f"      Compressed data ready")
            
            if send_photo(sock, move_id, move_name, compressed):
                print("    ✓ Success")
                time.sleep(0.5)
            else:
                print("    ✗ Failed, stopping...")
                break
        
        print(f"\n[4/4] Complete!")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Cleanup]")
        sock.close()
        close_camera()
        if motors:
            motors.cleanup()
        print("[Done]")


if __name__ == "__main__":
    main()