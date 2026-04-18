# ============================================================
# colab_setup.py — Run this ONCE at the start of every Colab session
# ============================================================
# Copy-paste each cell into a Colab notebook, or run:
#   exec(open('colab_setup.py').read())
# ============================================================

import subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# ── Cell 1: Verify GPU ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Checking GPU")
print("=" * 60)
import subprocess
gpu_info = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
if gpu_info.returncode != 0:
    raise RuntimeError("❌ No GPU found! Go to Runtime → Change runtime type → T4 GPU")
print(gpu_info.stdout)
print("✅ GPU found")

# ── Cell 2: Install PyTorch with CUDA FIRST ────────────────────────────────
print("=" * 60)
print("STEP 1: Installing PyTorch with CUDA 12.1")
print("This is the most important step — must come before everything else!")
print("=" * 60)
run(
    "pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "
    "--index-url https://download.pytorch.org/whl/cu121"
)

# Verify CUDA torch installed correctly
import importlib
torch = importlib.import_module("torch")
assert torch.cuda.is_available(), (
    "❌ CUDA not available after install! Restart runtime and try again."
)
print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Cell 3: Install all other dependencies ─────────────────────────────────
print("=" * 60)
print("STEP 2: Installing project dependencies")
print("=" * 60)
run("pip install -q -r requirements.txt")
print("✅ All dependencies installed")

# ── Cell 4: Mount Drive and set up paths ──────────────────────────────────
print("=" * 60)
print("STEP 3: Mounting Google Drive")
print("=" * 60)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("✅ Drive mounted at /content/drive")
except ImportError:
    print("⚠️  Not running in Colab — skipping Drive mount")

# ── Cell 5: Verify all imports work ───────────────────────────────────────
print("=" * 60)
print("STEP 4: Verifying all imports")
print("=" * 60)
try:
    import torch
    import diffusers
    import transformers
    import peft
    import librosa
    import lpips
    import cv2
    import imageio
    import numpy
    import scipy
    print(f"✅ torch         {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
    print(f"✅ diffusers     {diffusers.__version__}")
    print(f"✅ transformers  {transformers.__version__}")
    print(f"✅ peft          {peft.__version__}")
    print(f"✅ librosa       {librosa.__version__}")
    print(f"✅ lpips         {lpips.__version__}")
    print(f"✅ opencv        {cv2.__version__}")
    print(f"✅ imageio       {imageio.__version__}")
    print(f"✅ numpy         {numpy.__version__}")
    print(f"✅ scipy         {scipy.__version__}")
    print("\n🎉 Environment is ready! All packages imported successfully.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Try restarting the Colab runtime and re-running this script.")
