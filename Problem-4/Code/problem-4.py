import cv2
import numpy as np
import os

def compute_hdr():
    # --- Configuration ---
    # AUTOMATIC PATH DETECTION
    # This gets the folder where THIS script is currently saved (e.g., d:\DIP\Problem-4\)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We assume HDR_Set is inside that same folder
    hdr_folder = os.path.join(script_dir, 'HDR_Set')
    
    # Check if folder exists to avoid errors
    if not os.path.exists(hdr_folder):
        print(f"Error: Folder not found at: {hdr_folder}")
        print(f"Please ensure you created a folder named 'HDR_Set' inside '{script_dir}'")
        return

    # Exposure times for the dataset (inverse of shutter speed)
    exposure_times = np.array([
        32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25,
        0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        0.00195312, 0.00097656
    ], dtype=np.float32)

    # --- 1. Load Images ---
    print(f"Loading images from: {hdr_folder}")
    # Supports .ppm, .jpg, and .png
    filenames = sorted([f for f in os.listdir(hdr_folder) if f.lower().endswith(('.ppm', '.jpg', '.png'))])
    
    if not filenames:
        print("Error: No images found in HDR_Set folder! Please put the 16 memorial images inside it.")
        return

    img_list = [cv2.imread(os.path.join(hdr_folder, fn)) for fn in filenames]
    
    # Safety check for image count
    if len(img_list) != len(exposure_times):
        print(f"Warning: Found {len(img_list)} images but expected {len(exposure_times)}. The HDR merge might fail or look wrong.")

    # --- 2. Merge to HDR (Debevec Algorithm) ---
    print("Merging images using Debevec algorithm...")
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

    # --- 3. Tone Mapping (Drago) ---
    print("Applying Drago Tone Mapping...")
    tonemap = cv2.createTonemapDrago(gamma=2.2) 
    res_tonemap = tonemap.process(hdr_debevec)

    # --- 4. Save Result ---
    res_8bit = np.clip(res_tonemap * 255, 0, 255).astype('uint8')
    
    # Save output in the same folder as the script
    output_filename = os.path.join(script_dir, "output_memorial_hdr_final.jpg")
    cv2.imwrite(output_filename, res_8bit)
    print(f"Success! Saved HDR result to: {output_filename}")

if __name__ == "__main__":
    compute_hdr()