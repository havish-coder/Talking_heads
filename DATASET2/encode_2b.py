"""
encode_2b.py — re-encode videos + reference images to CogVideoX-2B latents.

Why this replaces encode_vid.py:
  * Uses the CogVideoX-2B VAE (the transformer we now train is 2B; the 2B VAE
    has different latent statistics than the 5B VAE the old latents used).
  * Encodes each clip in a SINGLE causal pass with the VAE's built-in tiling
    (vae.enable_tiling()), which preserves temporal coherence AND bounds memory.
    The old script encoded independent 8-frame chunks, which broke causality and
    produced seams every chunk.
  * Saves RAW latents (latent_dist.sample()). The training loop applies
    vae.config.scaling_factor itself, so do NOT scale here.

Resolution (v2 fix):
  Encode at SIZE=480 (square) -> 60x60 latent, NOT 768 -> 96x96. CogVideoX-2B's
  sinusoidal positional prior is tuned for its native ~480-scale grid; the old
  96x96 latents ran off-distribution and at ~2.5x the spatial-token cost. 480 is
  a *uniform* downscale of the same square framing, so the [0,1]-normalised
  DWPose keypoints (dataset.py CANVAS) stay valid — NO pose re-extraction needed.
  Pass --size to override.

Outputs (matching training/dataset.py expectations):
  OUT_ROOT/video_latents_final_videos/{stem}.npy  -> (1, 16, T_lat, 60, 60)
  OUT_ROOT/ref_latents/{stem}.npy                  -> (1, 16, 1, 60, 60)

Run on a GPU box (e.g. Lightning A100). 480x480 -> 60x60 latent.
"""

import gc
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
VIDEO_DIR   = SCRIPT_DIR / "processed_dataset_wholebody" / "final_videos"
REF_IMG_DIR = SCRIPT_DIR / "processed_dataset_wholebody" / "reference_images_final_768"
OUT_ROOT    = SCRIPT_DIR / "DATASET"
VID_OUT     = OUT_ROOT / "video_latents_final_videos"
REF_OUT     = OUT_ROOT / "ref_latents"

MODEL  = "THUDM/CogVideoX-2b"
SIZE   = 480            # square; -> 60x60 latent (native CogVideoX-2B scale)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32


def load_vae():
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
    vae.requires_grad_(False)
    vae.eval()
    vae.enable_tiling()   # spatial + temporal tiling: bounded memory, causal-correct
    vae.enable_slicing()
    return vae


def frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """list of HxWx3 RGB uint8 -> (1, 3, T, SIZE, SIZE) in [-1, 1]."""
    arr = np.stack(frames).astype(np.float32)               # (T, H, W, 3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)            # (T, 3, H, W)
    t = t / 127.5 - 1.0
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    t = t.permute(1, 0, 2, 3).unsqueeze(0)                   # (1, 3, T, SIZE, SIZE)
    return t


@torch.no_grad()
def encode(vae, tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.to(DEVICE, dtype=DTYPE)
    latent = vae.encode(tensor).latent_dist.sample()         # (1, 16, T_lat, 96, 96)
    out = latent.float().cpu().numpy()
    del latent, tensor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    global SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=SIZE,
                    help="square encode resolution; 480 -> 60x60 latent (default)")
    args = ap.parse_args()
    SIZE = args.size
    assert SIZE % 8 == 0, "SIZE must be divisible by 8 (VAE spatial factor)"
    print(f"[encode_2b] encoding at {SIZE}x{SIZE} -> {SIZE // 8}x{SIZE // 8} latent")

    VID_OUT.mkdir(parents=True, exist_ok=True)
    REF_OUT.mkdir(parents=True, exist_ok=True)
    vae = load_vae()

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"[encode_2b] {len(videos)} videos. device={DEVICE} dtype={DTYPE}")

    for vid in videos:
        stem = vid.stem
        vid_out = VID_OUT / f"{stem}.npy"
        ref_out = REF_OUT / f"{stem}.npy"

        # ── video latents ──
        if not vid_out.exists():
            frames = read_video_frames(vid)
            if not frames:
                print(f"  [skip] {stem}: no frames")
                continue
            lat = encode(vae, frames_to_tensor(frames))
            np.save(str(vid_out), lat)
            print(f"  [video] {stem}: {lat.shape}")
            del frames
            gc.collect()

        # ── reference latent (from reference image; fallback: first video frame) ──
        if not ref_out.exists():
            ref_img_path = REF_IMG_DIR / f"{stem}.jpg"
            if ref_img_path.exists():
                img = cv2.cvtColor(cv2.imread(str(ref_img_path)), cv2.COLOR_BGR2RGB)
            else:
                img = read_video_frames(vid)[0]
            ref_lat = encode(vae, frames_to_tensor([img]))    # (1,16,1,96,96)
            np.save(str(ref_out), ref_lat)
            print(f"  [ref]   {stem}: {ref_lat.shape}")

    print("[encode_2b] done.")


if __name__ == "__main__":
    main()
