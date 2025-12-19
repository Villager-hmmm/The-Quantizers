import cv2
import numpy as np
import math
import os

def spatial_filtering():
    # --- Configuration ---
    # AUTOMATIC PATH DETECTION
    # This gets the folder where THIS script is currently saved
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Image name
    image_name = 'Torgya - Arunachal Festival.jpg'
    
    # robustly join paths
    image_path = os.path.join(script_dir, image_name)

    # Load input image
    print(f"Looking for image at: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image!")
        print(f"1. Make sure '{image_name}' is inside the folder: {script_dir}")
        print(f"2. Check if the filename extension matches exactly (e.g., is it .jpeg instead of .jpg?)")
        return

    # --- Part 1: Box Filters ---
    print("Applying Box Filters...")
    for size in [5, 20]:
        # A. Normalized Kernel (Correct Approach)
        kernel_norm = np.ones((size, size), np.float32) / (size * size)
        res_norm = cv2.filter2D(img, -1, kernel_norm)
        
        output_norm = os.path.join(script_dir, f'output_box_{size}x{size}_normalized.jpg')
        cv2.imwrite(output_norm, res_norm)

        # B. Unnormalized Kernel (Demonstration of Limitation)
        kernel_unnorm = np.ones((size, size), np.float32)
        res_unnorm = cv2.filter2D(img, -1, kernel_unnorm)
        
        output_unnorm = os.path.join(script_dir, f'output_box_{size}x{size}_unnormalized.jpg')
        cv2.imwrite(output_unnorm, res_unnorm)

    # --- Part 2: Separable Gaussian Filter ---
    print("Applying Separable Gaussian Filter...")
    sigma = 5.0
    k_size = int(2 * math.pi * sigma) + 1
    
    # Get 1D Gaussian Kernel
    gaussian_1d = cv2.getGaussianKernel(k_size, sigma)

    # Separable Convolution Optimization:
    # Step 1: Convolve along rows (Transpose 1D kernel to be horizontal)
    step1 = cv2.filter2D(img, -1, gaussian_1d.T) 
    
    # Step 2: Convolve along columns
    final_gaussian = cv2.filter2D(step1, -1, gaussian_1d) 
    
    output_gauss = os.path.join(script_dir, f'output_gaussian_separable_sigma{int(sigma)}.jpg')
    cv2.imwrite(output_gauss, final_gaussian)
    print(f"Success! Outputs saved to {script_dir}")

if __name__ == "__main__":
    spatial_filtering()