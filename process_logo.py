from PIL import Image, ImageDraw
import numpy as np
import os

img_path = r"C:\Patrick\coding\save later\spect\prototype\assets\logo.png"
out_path = r"C:\Patrick\coding\save later\spect\prototype\assets\logo_transparent.png"

img = Image.open(img_path).convert("RGBA")
arr = np.array(img)
h, w = arr.shape[:2]

# 1. Detect background color from top-left pixel
bg_color = arr[0, 0, :3].astype(np.float32)

# 2. Calculate color difference
dist = np.linalg.norm(arr[:, :, :3].astype(np.float32) - bg_color, axis=2)

# 3. Create alpha channel based on difference (soft threshold)
# Multiplier makes the transition sharper
alpha_color = np.clip((dist - 15) * 15, 0, 255).astype(np.uint8)

# 4. Create a perfect circle mask for the "bulet" shape
mask_circle = Image.new('L', (w, h), 0)
draw = ImageDraw.Draw(mask_circle)
# Draw circle with 5px padding
pad = 5
draw.ellipse((pad, pad, w-pad, h-pad), fill=255)
alpha_circle = np.array(mask_circle)

# 5. Combine both masks (must be inside circle AND differ from background)
final_alpha = np.minimum(alpha_color, alpha_circle)

# Overwrite alpha channel
arr[:, :, 3] = final_alpha

# Save new transparent logo
out = Image.fromarray(arr)
out.save(out_path)
print("Berhasil diproses!")
