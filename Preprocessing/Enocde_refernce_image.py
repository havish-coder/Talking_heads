import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
REF_DIR = "reference_images_final_768"
REF_LATENT_DIR = "ref_latents"
HEIGHT = 768
WIDTH = 768

os.makedirs(REF_LATENT_DIR, exist_ok=True)

# --- Hardware Setup ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[SYSTEM] GPU Verified. Using: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("[WARN] CUDA is not available. Falling back to CPU (processing will be slow).")

# --- Model Initialization ---
print("[SYSTEM] Loading AutoencoderKLCogVideoX VAE...")

# Load the VAE in half-precision (FP16) for GPU memory efficiency, or FP32 for CPU
vae = AutoencoderKLCogVideoX.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="vae",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Freeze gradients and set to evaluation mode since we are only doing inference
vae.requires_grad_(False)
vae.eval()

# --- Image Processing Pipeline ---
# Resize the image, convert to tensor, and scale pixel values to [-1.0, 1.0] for the VAE
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Execution ---
img_paths = sorted(Path(REF_DIR).glob("*.jpg"))
print(f"[SYSTEM] Found {len(img_paths)} image(s) to process.")

for img_path in tqdm(img_paths, desc="Encoding Latents", unit="img"):
    out = Path(REF_LATENT_DIR) / f"{img_path.stem}.npy"
    
    if out.exists():
        tqdm.write(f"  -> Skipping {img_path.name} (File already exists)")
        continue
        
    img = Image.open(img_path).convert("RGB")
    
    # Transform outputs (Channels, Height, Width).
    # CogVideoX expects (Batch, Channels, Time, Height, Width).
    # We add Batch=1 and Time=1 using unsqueeze to satisfy the 3D video VAE architecture.
    tensor = transform(img).unsqueeze(0).unsqueeze(2)
    tensor = tensor.to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    
    # Encode the image into the latent space and sample from the distribution
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample() 
        
    # Save the resulting tensor (1, 16, 1, H/8, W/8) as a numpy array for later use
    np.save(str(out), latent.cpu().float().numpy())

print("[SYSTEM] Latent encoding completed successfully.")