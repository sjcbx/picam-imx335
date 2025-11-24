#!/usr/bin/env python3
import numpy as np
import cv2
import os
import glob

# ---- CONFIGURATION ----
INPUT_DIR = "/home/user/Pictures"
OUTPUT_DIR = "/home/user/Pictures/Processed"
# IMX335 is usually RGGB. If colors are wrong, try cv2.COLOR_BayerBG2RGB
BAYER_PATTERN = cv2.COLOR_BayerRG2RGB 
ENABLE_GAMMA = True

def unpack_raw10(packed_array, width, height):
    """
    Unpacks 10-bit packed CSI-2 data (SRGGB10) from Raspberry Pi.
    
    Structure (5 bytes -> 4 pixels):
    [P0_MSB][P1_MSB][P2_MSB][P3_MSB][LSBs]
    LSBs byte: [P3:2][P2:2][P1:2][P0:2]
    """
    raw_bytes = packed_array.flatten().astype(np.uint8)
    
    # 1. Determine Stride (Bytes per row)
    file_size = raw_bytes.size
    stride = file_size // height
    
    print(f"   -> Debug: Size={file_size}, Stride={stride} bytes/row")
    
    # 2. Reshape into rows and remove padding
    # 10-bit packed width = width * 1.25 bytes
    valid_packed_width = int(width * 1.25) 
    
    try:
        # Reshape to (Height, Stride)
        raw_rows = raw_bytes.reshape((height, stride))
        # Crop off the padding bytes at the end of each row
        raw_rows = raw_rows[:, :valid_packed_width]
    except ValueError:
        print(f"   -> Error: shape mismatch. Expected {height}x{stride}, got {raw_bytes.shape}")
        return None

    # 3. Reshape for unpacking (N blocks of 5 bytes)
    # We flatten again, then reshape into chunks of 5 bytes
    flat_packed = raw_rows.flatten()
    n_pixels = width * height
    n_blocks = n_pixels // 4
    
    blocks = flat_packed.reshape((n_blocks, 5))
    
    # 4. Extract Components
    # MSBs are simply the first 4 bytes
    p0_msb = blocks[:, 0].astype(np.uint16)
    p1_msb = blocks[:, 1].astype(np.uint16)
    p2_msb = blocks[:, 2].astype(np.uint16)
    p3_msb = blocks[:, 3].astype(np.uint16)
    
    # LSBs are in the 5th byte (index 4)
    lsb_byte = blocks[:, 4].astype(np.uint16)
    
    # 5. Shift and Combine
    # P0 gets bits 0-1 of LSB byte
    p0 = (p0_msb << 2) | (lsb_byte & 0x03)
    # P1 gets bits 2-3
    p1 = (p1_msb << 2) | ((lsb_byte >> 2) & 0x03)
    # P2 gets bits 4-5
    p2 = (p2_msb << 2) | ((lsb_byte >> 4) & 0x03)
    # P3 gets bits 6-7
    p3 = (p3_msb << 2) | ((lsb_byte >> 6) & 0x03)
    
    # 6. Interleave to final image
    unpacked = np.zeros((n_pixels,), dtype=np.uint16)
    unpacked[0::4] = p0
    unpacked[1::4] = p1
    unpacked[2::4] = p2
    unpacked[3::4] = p3
    
    return unpacked.reshape((height, width))

def process_file(filepath):
    print(f"Processing: {os.path.basename(filepath)}...")
    
    try:
        data = np.load(filepath)
        raw_packed = data['raw']
        
        # IMX335 Resolution
        W, H = 2592, 1944
        
        # Attempt unpack
        raw_16bit = unpack_raw10(raw_packed, W, H)
        if raw_16bit is None:
            return

        # Simple ISP (Demosaic + Gamma)
        bgr_image = cv2.cvtColor(raw_16bit, BAYER_PATTERN)

        if ENABLE_GAMMA:
            # 10-bit max is 1023. Normalize, Gamma, Scale to 16-bit
            norm = bgr_image.astype(np.float32) / 1023.0
            gamma = np.power(norm, 1/2.2)
            final_img = (np.clip(gamma, 0, 1) * 65535).astype(np.uint16)
        else:
            # Scale 10-bit to 16-bit explicitly
            final_img = (bgr_image * 64).astype(np.uint16)

        out_name = os.path.basename(filepath).replace('.npz', '.png')
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, final_img)
        print(f"   -> Saved: {out_name}")

    except Exception as e:
        print(f"   -> Failed: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    if not files:
        print("No .npz files found!")
        return

    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()
