from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import math

# --- 1. Load and Preprocess Image ---
# NOTE: Update the path below to match the location on your computer
image_path = r"D:\\DIP\\Problem-2\\Input Image\\L01 images\\cycle.png"

# Open image and convert to RGB
img = Image.open(image_path).convert("RGB")

# Convert to numpy array and normalize to [0, 1] range
arr = np.asarray(img, dtype=np.float32) / 255.0
h, w, c = arr.shape

# Flatten the image array (rows * cols, 3 channels)
X = arr.reshape(-1, c)
N = X.shape[0]

# --- 2. Random Sampling for Efficiency ---
# Use a random sample of 10,000 pixels to train the K-Means algorithm faster
sample_size = min(10000, N)
idx = np.random.choice(N, sample_size, replace=False)
X_sample = X[idx]

# --- 3. K-Means Loop ---
K_values = [4, 8, 16, 32, 64]
rates = []
mses = []
psnrs = []

for K in K_values:
    # Initialize and train MiniBatchKMeans on the sample
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=2048, max_iter=50, n_init=1)
    kmeans.fit(X_sample)
    
    # Predict labels for the entire image
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    
    # Reconstruct the quantized image
    Xq = centers[labels]
    quant_arr = Xq.reshape(h, w, c)
    
    # Calculate MSE
    diff = arr - quant_arr
    mse = np.mean(diff ** 2)
    mses.append(mse)
    
    # Calculate PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * math.log10(1.0 / mse)
    psnrs.append(psnr)
    
    # Calculate Rate (Entropy)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]  # Remove zero probabilities to avoid log(0) error
    entropy = -np.sum(p * np.log2(p))
    rates.append(entropy)
    
    # Plot Specific K results
    if K in [4, 16, 64]:
        plt.figure()
        plt.imshow(quant_arr)
        plt.title(f"K = {K}, PSNR = {psnr:.2f} dB, R = {entropy:.3f} bits/pixel")
        plt.axis("off")

# Convert lists to numpy arrays for plotting
rates = np.array(rates)
mses = np.array(mses)
psnrs = np.array(psnrs)

# --- 4. Plot Rate-Distortion Curve ---
plt.figure()
plt.plot(rates, psnrs, marker="o")

# Annotate points on the graph
for i, K in enumerate(K_values):
    plt.text(rates[i], psnrs[i], f"K={K}")

plt.xlabel("Rate (bits/pixel)")
plt.ylabel("PSNR (dB)")
plt.title("Fast K-means Rate–Distortion Curve")
plt.grid(True)
plt.show()

# --- 5. Print Summary Table ---
print("K\tRate(bits/pixel)\tMSE\t\tPSNR(dB)")
print("-" * 60)
for K, R, mse, psnr in zip(K_values, rates, mses, psnrs):
    print(f"{K}\t{R:.4f}\t\t\t{mse:.6f}\t{psnr:.2f}")