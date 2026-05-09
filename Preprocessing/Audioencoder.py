import os
import gc
import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
import subprocess
import tempfile

# Configuration
AUDIO_DIR  = "audio_clips"
OUTPUT_DIR = "audio_embeddings_wav2vec2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize model and processor
print(f"Device : {DEVICE}")
print("Loading Wav2Vec2 model...")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
model.eval()

print("Model loaded.\n")

def load_audio(path: str) -> np.ndarray:
    """Converts audio to 16kHz mono WAV and loads as float32 array."""
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

# Process audio files
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

        if out_path.exists():
            print(f"Skipping {fpath.name} (already done)")
            skipped += 1
            continue

        print(f"Processing {fpath.name} ...")

        try:
            audio = load_audio(str(fpath))

            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(DEVICE)
            output = model(input_values)
            embedding = output.last_hidden_state.squeeze(0).cpu().numpy()

            np.save(out_path, embedding.astype(np.float32))
            print(f"  Saved {embedding.shape} → {out_path}")
            completed += 1

        except Exception as e:
            print(f"  Failed {fpath.name}: {e}")
            failed += 1

        # Free memory between iterations
        del audio
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# Execution summary
print("\nProcessing Complete")
print(f"  Completed : {completed}")
print(f"  Skipped   : {skipped}")
print(f"  Failed    : {failed}")