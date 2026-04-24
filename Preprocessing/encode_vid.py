import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import gc
import threading
import time
import multiprocessing
from queue import Queue
from pathlib import Path
from diffusers import AutoencoderKLCogVideoX

# --- CONFIGURATION ---
REF_DIR        = "final_videos"
REF_LATENT_DIR = "video_latents_final_videos"

#RTX 3050 (4GB) optimized settings
HEIGHT     = 768   # Reduced from 768 — huge VRAM saving, encode then upsample if needed
WIDTH      = 768
CHUNK_SIZE = 8     # Reduced from 16 — prevents mid-chunk OOM

START_INDEX = 8

# Auto-Restart / Cooldown Settings
BATCH_SIZE       = 3    # Reduced: respawn process more often to flush fragmented VRAM
COOLDOWN_SECONDS = 20   # Slightly longer cooldown for a laptop GPU

# --- Thermal monitoring threshold (Celsius) ---
# Script will pause if GPU temp exceeds this before starting a new video
GPU_TEMP_LIMIT = 85  # degrees C

os.makedirs(REF_LATENT_DIR, exist_ok=True)


def get_gpu_temp():
    """Try to read GPU temperature via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return None  # Can't read temp, skip check


def wait_for_gpu_cooldown(limit=GPU_TEMP_LIMIT, check_interval=10):
    """Block until GPU temperature drops below the limit."""
    while True:
        temp = get_gpu_temp()
        if temp is None or temp < limit:
            break
        print(f"   [THERMAL] GPU at {temp}°C (limit {limit}°C). Waiting {check_interval}s...")
        time.sleep(check_interval)


def cpu_video_reader(cap, chunk_queue, chunk_size):
    """Background thread: reads video frames and puts numpy chunks on a queue."""
    while True:
        chunk_frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            chunk_frames.append(frame_rgb)

        if not chunk_frames:
            break

        chunk_queue.put(np.stack(chunk_frames))
        del chunk_frames

    chunk_queue.put(None)   # sentinel


def encode_chunk(vae, chunk_numpy, device, height, width):
    """
    Encode a single numpy chunk through the VAE.
    Returns the latent on CPU. Handles OOM by halving the chunk.
    """
    chunk_tensor = torch.from_numpy(chunk_numpy).to(device)
    chunk_tensor = chunk_tensor.permute(0, 3, 1, 2).float() / 127.5 - 1.0
    chunk_tensor = F.interpolate(chunk_tensor, size=(height, width),
                                 mode='bilinear', align_corners=False)
    if device == "cuda":
        chunk_tensor = chunk_tensor.half()
    chunk_tensor = chunk_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

    try:
        with torch.no_grad():
            latent = vae.encode(chunk_tensor).latent_dist.sample()
        result = latent.cpu()
        del latent
        return result

    except torch.cuda.OutOfMemoryError:
        # ---- OOM Recovery: split chunk in half and encode separately ----
        print("   [OOM] Splitting chunk in half to recover...")
        torch.cuda.empty_cache()
        gc.collect()

        T = chunk_tensor.shape[2]
        half = T // 2
        if half == 0:
            raise  # Can't split further

        parts = []
        for sub in [chunk_tensor[:, :, :half], chunk_tensor[:, :, half:]]:
            with torch.no_grad():
                lat = vae.encode(sub).latent_dist.sample()
            parts.append(lat.cpu())
            del lat
            torch.cuda.empty_cache()

        result = torch.cat(parts, dim=2)
        del chunk_tensor
        return result

    finally:
        # Always clean up the input tensor
        try:
            del chunk_tensor
        except Exception:
            pass
        torch.cuda.empty_cache()


def process_video_batch(video_paths, height, width, chunk_size):
    """Runs in an isolated subprocess — loads VAE fresh, processes batch, exits cleanly."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   [Process] Using device: {device}")

    # Force deterministic allocator behavior to reduce fragmentation
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    vae.enable_tiling()
    vae.enable_slicing()

    for vid_path in video_paths:
        out = Path(REF_LATENT_DIR) / (vid_path.stem + ".npy")
        if out.exists():
            print(f"   Skipping {vid_path.name} (already exists)")
            continue

        # --- Thermal gate: don't start a hot video ---
        wait_for_gpu_cooldown()

        print(f"   -> Processing: {vid_path.name}")
        t0 = time.time()

        cap = cv2.VideoCapture(str(vid_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"      Frames: {total_frames}  FPS: {fps:.1f}  "
              f"Duration: {total_frames/max(fps,1):.1f}s")

        encoded_chunks = []
        chunk_queue = Queue(maxsize=3)   # slightly larger buffer
        reader_thread = threading.Thread(
            target=cpu_video_reader, args=(cap, chunk_queue, chunk_size)
        )
        reader_thread.daemon = True
        reader_thread.start()

        chunk_idx = 0
        while True:
            chunk_numpy = chunk_queue.get()
            if chunk_numpy is None:
                break

            try:
                latent = encode_chunk(vae, chunk_numpy, device, height, width)
                encoded_chunks.append(latent)
                chunk_idx += 1
                if chunk_idx % 5 == 0:
                    temp = get_gpu_temp()
                    temp_str = f"  GPU: {temp}°C" if temp else ""
                    print(f"      Chunk {chunk_idx} encoded.{temp_str}")
            except Exception as e:
                print(f"      [ERROR] Chunk {chunk_idx} failed: {e}. Skipping chunk.")
                torch.cuda.empty_cache()
                gc.collect()
            finally:
                del chunk_numpy

        reader_thread.join()
        cap.release()

        if not encoded_chunks:
            print(f"      [WARN] No chunks encoded for {vid_path.name}. Skipping save.")
            continue

        final_latent = torch.cat(encoded_chunks, dim=2)  # concat along time axis
        np.save(str(out), final_latent.float().numpy())
        elapsed = time.time() - t0
        print(f"      Saved: {out.name}  shape={tuple(final_latent.shape)}  "
              f"time={elapsed:.1f}s")

        del encoded_chunks, final_latent
        gc.collect()
        torch.cuda.empty_cache()

    # Clean shutdown of this subprocess
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    print("   [Process] Batch done. Subprocess exiting cleanly.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    print("[INIT] Scanning for videos...")
    all_vids = sorted(Path(REF_DIR).glob("*.mp4"))[START_INDEX:]
    pending  = [p for p in all_vids
                if not (Path(REF_LATENT_DIR) / (p.stem + ".npy")).exists()]
    print(f"       {len(pending)} videos to process.\n")

    if not pending:
        print("Nothing to do. Exiting.")
        exit(0)

    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(pending), BATCH_SIZE), start=1):
        batch = pending[i : i + BATCH_SIZE]
        names = [p.name for p in batch]
        print(f"[BATCH {batch_num}/{total_batches}] {names}")

        p = multiprocessing.Process(
            target=process_video_batch,
            args=(batch, HEIGHT, WIDTH, CHUNK_SIZE)
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"   [WARN] Batch {batch_num} subprocess exited with code {p.exitcode}.")

        # Cooldown between batches (skip after the last one)
        if i + BATCH_SIZE < len(pending):
            print(f"[COOLDOWN] Sleeping {COOLDOWN_SECONDS}s between batches...\n")
            time.sleep(COOLDOWN_SECONDS)

    print("\n[DONE] All videos processed.")
