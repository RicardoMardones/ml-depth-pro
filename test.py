import torch
import depth_pro
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img_path = "huelmo_cap2_small.png"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
precision = torch.half if device.type == "cuda" else torch.float32

print(f"Using device: {device}")

model, transform = depth_pro.create_model_and_transforms(
    device=device,
    precision=precision,
)
model.eval()

image, _, f_px = depth_pro.load_rgb(img_path)
image = transform(image)

with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

depth = prediction["depth"]
focallength_px = prediction["focallength_px"]

print(f'{focallength_px=}')

print(f"Depth shape: {tuple(depth.shape)}")
print(f"Depth device: {depth.device}")
print(f"Depth min/max: {depth.min().item():.3f} / {depth.max().item():.3f} m")

if focallength_px is not None:
    print(f"Focal length: {focallength_px.item():.2f} px")

# -------------------------------
# Visualización del mapa de profundidad
# -------------------------------

depth_np = depth.detach().cpu().numpy()
depth_np = np.squeeze(depth_np)

depth_min = np.nanmin(depth_np)
depth_max = np.nanmax(depth_np)

depth_vis = (depth_np - depth_min) / (depth_max - depth_min)
depth_vis = (depth_vis * 255).astype(np.uint8)

Image.fromarray(depth_vis).save("depth_result_gray.png")

plt.figure(figsize=(10, 6))
plt.imshow(depth_vis, cmap="gray")
plt.title("Mapa de profundidad normalizado")
plt.axis("off")
plt.colorbar(label="Profundidad normalizada")
plt.show()