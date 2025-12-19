import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Load Image ---
# Ensure the path is correct for your environment
img_path = r"C:\\Users\\barat\\Downloads\\L01 images\\dog.png"
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image at {img_path}")
else:
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    
    # Define sampling factors
    factors = [2, 4, 8, 16]
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/original_gray.jpg", gray)

    # --- 2. Spatial Sampling Loop ---
    for k in factors:
        # Downsample: Reduce resolution by factor k
        new_h = H // k
        new_w = W // k
        down = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Upsample: Restore size using Nearest Neighbor (creates blocky effect)
        up = cv2.resize(down, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Save Result
        cv2.imwrite(f"output/spatial_1_{k}.jpg", up)

    # --- 3. Frequency Sampling Loop ---
    # Compute FFT once
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)

    for k in factors:
        # Define Low-Pass Filter Dimensions
        keep_h = H // k
        keep_w = W // k
        
        # Create Rectangular Mask (Ideal Low Pass Filter)
        mask = np.zeros((H, W), np.uint8)
        r1 = H // 2 - keep_h // 2
        r2 = r1 + keep_h
        c1 = W // 2 - keep_w // 2
        c2 = c1 + keep_w
        mask[r1:r2, c1:c2] = 1
        
        # Apply Mask
        F_masked = F_shift * mask
        
        # Save Spectrum Visualization (Log Scale)
        mag = 20 * np.log(np.abs(F_masked) + 1)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f"output/spectrum_1_{k}.jpg", mag_norm)
        
        # Inverse FFT to reconstruct image
        F_ishift = np.fft.ifftshift(F_masked)
        img_back = np.fft.ifft2(F_ishift)
        img_back = np.real(img_back)
        
        # Normalize and Save
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f"output/frequency_1_{k}.jpg", img_back_norm)

    print("All outputs saved in the 'output' folder.")