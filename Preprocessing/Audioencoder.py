import os
import gc
import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
import subprocess
import tempfile

# ==========================================
# CONFIGURATION
# ==========================================
AUDIO_DIR  = "Talking_heads\Preprocessing\audio_clips"
OUTPUT_DIR = "audio_embeddings_wav2vec2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL LOADING
# ==========================================
print(f"Device : {DEVICE}")
print("Loading Wav2Vec2 model...")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
model.eval()

print("Model loaded.\n")

# ==========================================
# AUDIO LOADING
# Converts any format to 16kHz mono WAV
# using ffmpeg (same as your Whisper script)
# ==========================================
def load_audio(path: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    audio, _ = sf.read(tmp_path)
    os.remove(tmp_path)
    return audio.astype(np.float32)

# ==========================================
# MAIN
# ==========================================
audio_files = sorted([
    p for p in Path(AUDIO_DIR).iterdir()
    if p.suffix.lower() in SUPPORTED
])
print(f"Found {len(audio_files)} audio files.\n")

completed = 0
skipped   = 0
failed    = 0

with torch.no_grad():
    for fpath in audio_files:
        out_path = Path(OUTPUT_DIR) / (fpath.stem + ".npy")

        # skip if already done
        if out_path.exists():
            print(f"Skipping {fpath.name} (already done)")
            skipped += 1
            continue

        print(f"Processing {fpath.name} ...")

        try:
            # load + resample to 16kHz mono
            audio = load_audio(str(fpath))

            # tokenize
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(DEVICE)

            # extract features
            output = model(input_values)

            # shape: (T, 768)
            # T = ~50 tokens per second of audio (frame-level, 20ms per token)
            # 768 = Wav2Vec2-base hidden size
            # directly matches EchoMimicV2 paper's audio encoder output
            embedding = output.last_hidden_state.squeeze(0).cpu().numpy()

            np.save(out_path, embedding.astype(np.float32))
            print(f"  Saved {embedding.shape} → {out_path}")
            completed += 1

        except Exception as e:
            print(f"  Failed {fpath.name}: {e}")
            failed += 1

        # free memory between files
        del audio
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# ==========================================
# SUMMARY
# ==========================================
print(f"""
==========================================
DONE
==========================================
  Completed : {completed}
  Skipped   : {skipped}
  Failed    : {failed}

Output shape per file : (T, 768)
  T   = ~50 tokens per second of audio
  768 = Wav2Vec2 hidden dim

vs your old Whisper output:
  (1500, 1280) — fixed 30s window, semantic features

Wav2Vec2 advantages for animation:
  - frame-level acoustic features (better lip sync)
  - 768 dim fits DiT model directly (no projection needed)
  - T scales with actual audio length (no padding waste)
  - exactly what EchoMimicV2 paper uses
""")