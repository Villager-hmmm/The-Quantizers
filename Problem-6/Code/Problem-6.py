import cv2
import numpy as np
import os

def analyze_bitplanes(filename):
    # --- Configuration ---
    # AUTOMATIC PATH DETECTION
    # This gets the folder where THIS script is currently saved
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Robustly join paths to get the full location of the image
    image_path = os.path.join(script_dir, filename)

    print(f"Looking for image at: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{filename}'")
        print(f"Make sure '{filename}' is inside the folder: {script_dir}")
        return
    
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. Extract Bit Planes ---
    # Using bitwise AND to mask specific bits
    # Bit 0 is the Least Significant Bit (LSB)
    p1 = cv2.bitwise_and(gray, 1)  # Bit plane 1 (Value 1)
    p2 = cv2.bitwise_and(gray, 2)  # Bit plane 2 (Value 2)
    p3 = cv2.bitwise_and(gray, 4)  # Bit plane 3 (Value 4)

    # --- 2. Reconstruct Noise Floor ---
    # Sum the lowest planes to see the noise contribution
    union_low = p1 + p2 + p3

    # --- 3. Visualization ---
    # The raw values (0-7) are too dark to see.
    # Scale them to 0-255 range for visualization.
    union_vis = (union_low / 7.0) * 255.0
    
    # --- 4. Difference Map ---
    # Calculate absolute difference between original and noise floor
    diff = cv2.absdiff(gray, union_low)

    # Save results to the same directory
    prefix = filename.split('.')[0]
    
    output_vis_path = os.path.join(script_dir, f'bitplane_noise_{prefix}.jpg')
    output_diff_path = os.path.join(script_dir, f'bitplane_diff_{prefix}.jpg')
    
    cv2.imwrite(output_vis_path, union_vis)
    cv2.imwrite(output_diff_path, diff)
    
    print(f"Success! Saved: bitplane_noise_{prefix}.jpg and bitplane_diff_{prefix}.jpg")

if __name__ == "__main__":
    # Run on the two dataset images
    # Ensure these files exist in your directory
    print("--- Processing Low Light Image ---")
    analyze_bitplanes('low_light.jpg')
    
    print("\n--- Processing Bright Light Image ---")
    analyze_bitplanes('bright_light.jpg')