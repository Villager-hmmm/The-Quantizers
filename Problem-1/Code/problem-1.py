from PIL import Image
import os
import matplotlib.pyplot as plt

def clamp(v):
     return max(0, min(255, v))

def luminosity(r, g, b):
    v = 0.21*r + 0.72*g + 0.07*b
    return clamp(int(round(v)))

def quantize(gray, levels):
     idx = round((gray / 255) * (levels - 1))
     return clamp(round(idx * (255 / (levels - 1))))

def convert_luminosity_quantized(img, levels):
    if levels < 2:
        levels = 256
    w, h = img.size
    out = Image.new("RGB", (w, h))
    px = img.load()
    opx = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            g1 = luminosity(r, g, b)
            g2 = quantize(g1, levels)
            opx[x, y] = (g2, g2, g2)
    return out

def median_cut_quantize(img, colors=16):
    return img.convert("P", palette=Image.ADAPTIVE, colors=colors).convert("RGB")

def octree_quantize(img, colors=16):
    return img.quantize(colors=colors, method=2).convert("RGB")


input_path = r"D:\\DIP\\Problem-1\\L01 images\\cycle.png"
os.makedirs("quant_results", exist_ok=True)

img = Image.open(input_path).convert("RGB")

lum_out = convert_luminosity_quantized(img, 16)
lum_out.save("quant_results/luminosity_quantized_16.png")

median_out = median_cut_quantize(img, 16)
median_out.save("quant_results/median_cut_16.png")

octree_out = octree_quantize(img, 16)
octree_out.save("quant_results/octree_16.png")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Luminosity + Uniform")
plt.imshow(lum_out)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Median Cut (16 colors)")
plt.imshow(median_out)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Octree (16 colors)")
plt.imshow(octree_out)
plt.axis("off")

plt.show()