# The Quantizers

**Course**: 23ELC361 Digital Image Processing  
**Team Members:**  
**Harshaa V [EEE23113]**  
**Navendharan G [EEE23124]**  
**S Barath Pragdeesh [EEE231332]**  
**Amati Rithwik [EEE23151]**


## 📖 Overview
This repository contains the source code and results for the Digital Image Processing assignments. The project explores fundamental concepts such as Image Quantization, Sampling, Filtering, High Dynamic Range (HDR) imaging, and Bit-plane Slicing using Python.

## 📂 Project Structure

The repository is organized into six problem sets, each addressing a specific domain of image processing:

### [Problem 1: Image Quantization Techniques](./Problem-1)
**Objective**: Explore different methods of reducing the number of distinct colors in an image.
* **Techniques Implemented**:
    * **Luminosity Method**: Grayscale conversion preserving perceived brightness.
    * **Median Cut**: Adaptive palette generation.
    * **Octree Quantization**: Tree-based color reduction.
* **Key Files**: `problem-1.py`, `quant_results/`

### [Problem 2: Rate-Distortion & K-Means Clustering](./Problem-2)
**Objective**: Analyze the trade-off between image quality (PSNR) and compression rate (bits/pixel) using clustering.
* **Techniques Implemented**:
    * **K-Means Clustering**: Used to quantize image colors into $K$ clusters ($K=4, 8, 16, 32, 64$).
    * **Metrics**: Calculation of Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Entropy.
    * **Visualization**: Rate-Distortion curves.
* **Key Files**: `problem-2.py`, `Results.png`

### [Problem 3: Sampling & Aliasing](./Problem-3)
**Objective**: Demonstrate the effects of downsampling in both spatial and frequency domains.
* **Techniques Implemented**:
    * **Spatial Downsampling**: Reducing resolution and observing pixelation/aliasing.
    * **Frequency Domain Filtering**: Using FFT (Fast Fourier Transform) to apply ideal low-pass filters before reconstruction.
* **Key Files**: `problem-3.py`, `output/`

### [Problem 4: High Dynamic Range (HDR) Imaging](./Problem-4)
**Objective**: Create a high dynamic range image from a sequence of exposure-bracketed photographs.
* **Techniques Implemented**:
    * **Debevec Algorithm**: Merging multiple exposure images into a single HDR radiance map.
    * **Tone Mapping**: Using the Drago operator to convert the HDR image to a displayable 8-bit format.
* **Key Files**: `problem-4.py`, `output_memorial_hdr_final.jpg`

### [Problem 5: Spatial Filtering](./Problem-5)
**Objective**: Implement and compare different spatial smoothing filters.
* **Techniques Implemented**:
    * **Box Filter**: Implementation of both Normalized (correct brightness) and Unnormalized (saturated) box filters.
    * **Gaussian Filter**: Optimization using **Separable Convolution** (splitting 2D convolution into two 1D passes) for efficiency.
* **Key Files**: `Problem-5.py`, `output_gaussian_separable_sigma5.jpg`

### [Problem 6: Bit-Plane Slicing & Noise Analysis](./Problem-6)
**Objective**: Analyze image information content and noise distribution across bit planes.
* **Techniques Implemented**:
    * **Bit-Plane Extraction**: Isolating specific bits (LSB to MSB) for analysis.
    * **Noise Visualization**: Reconstructing the "noise floor" using the lowest 3 significant bits.
    * **Difference Mapping**: Highlighting the difference between the original image and the noise floor.
* **Key Files**: `Problem-6.py`, `bitplane_noise_low_light.jpg`

---

## 🛠️ Prerequisites & Installation

The projects are written in **Python**. You will need the following libraries installed:

```bash
pip install numpy opencv-python matplotlib pillow scikit-learn
