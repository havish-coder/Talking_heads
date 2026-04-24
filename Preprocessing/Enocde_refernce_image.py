import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKLCogVideoX
from tqdm import tqdm  # Added for progress tracking

print("[STEP] Script initialized. Setting up directories...")
REF_DIR     = "reference_images_final_768"   # input: .jpg files
REF_LATENT_DIR = "ref_latents"               # output: .npy files
HEIGHT = 768
WIDTH  = 768

os.makedirs(REF_LATENT_DIR, exist_ok=True)
print("       -> Directories ready.")

# --- GPU VERIFICATION ---
print("[STEP] Verifying GPU availability...")
if torch.cuda.is_available():
    DEVICE = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"       -> SUCCESS: GPU Verified. Using: {gpu_name}")
else:
    DEVICE = "cpu"
    print("       -> WARNING: CUDA is not available! Falling back to CPU. This will be very slow.")

# --- LOAD VAE ---
print("[STEP] Started loading VAE model (AutoencoderKLCogVideoX)...")
vae = AutoencoderKLCogVideoX.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="vae",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
vae.requires_grad_(False)
vae.eval()
print("       -> VAE model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- GATHER FILES ---
print(f"[STEP] Scanning '{REF_DIR}' for images...")
img_paths = sorted(Path(REF_DIR).glob("*.jpg"))
print(f"       -> Found {len(img_paths)} image(s) to process.")

# --- ENCODING LOOP ---
print("[STEP] Starting latent encoding...")
for img_path in tqdm(img_paths, desc="Encoding Latents", unit="img"):
    out = Path(REF_LATENT_DIR) / (img_path.stem + ".npy")
    
    if out.exists():
        # Use tqdm.write instead of print to prevent messing up the progress bar UI
        tqdm.write(f"Skipping {img_path.name} (Already exists)")
        continue
        
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).unsqueeze(2)   # (1, 3, 1, H, W)
    tensor = tensor.to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    
    with torch.no_grad():
        # (1, 16, 1, H/8, W/8)
        latent = vae.encode(tensor).latent_dist.sample() 
        
    np.save(str(out), latent.cpu().float().numpy())

print("[STEP] Latent encoding finished.")
print("\nAll tasks completed successfully. Done.")