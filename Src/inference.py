"""
inference.py

Talking Heads inference — generate a talking-head video from:
  1. A reference image  (.jpg / .png)
  2. A driving audio    (.wav / .m4a)
  3. A trained checkpoint (checkpoint.pt produced by train.py)

Usage (run from Src/):
  python inference.py \\
      --checkpoint  ../checkpoints/checkpoint_070000/checkpoint.pt \\
      --ref_image   path/to/portrait.jpg \\
      --audio       path/to/speech.wav \\
      --output      output.mp4 \\
      --steps       50 \\
      --fps         24 \\
      --cfg_scale   3.5

What this script does:
  1. Rebuild TalkingHeadsDiT + AudioEncoder (same arch as training).
  2. Load the trainable weights from the checkpoint (strict=False, so the
     frozen backbone weights come from CogVideoX pretrained).
  3. Encode the reference image with the CogVideoX VAE.
  4. Encode the driving audio with AudioEncoder.
  5. Run DDIM denoising for `--steps` steps (much faster than DDPM).
  6. Decode every latent frame with the VAE.
  7. Write frames to an MP4 via imageio / cv2.

Notes:
  - Pose keypoints are OPTIONAL at inference. If you provide a pose .npy
    file (same format as preprocessing) via --pose, it will be used.
    Otherwise the model runs audio-only (APDH stage 4 behaviour).
  - Classifier-free guidance (CFG) works by running two forward passes per
    step: one conditioned, one with zeroed audio.  cfg_scale=1.0 disables it.
  - The script runs fine on CPU but is very slow (minutes per frame).
    Use a GPU for real videos.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import imageio
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKLCogVideoX, DDIMScheduler
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2Model

# ── Local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from Models.talking_heads_dit import TalkingHeadsDiT
from Models.audio_encoder import AudioEncoder


# ── Constants ──────────────────────────────────────────────────────────────

PRETRAINED_MODEL   = "THUDM/CogVideoX-2b"
WAV2VEC2_MODEL     = "facebook/wav2vec2-base"
AUDIO_SR           = 16_000        # Hz — Wav2Vec2 expects 16 kHz
VIDEO_FPS          = 24
CLIP_FRAMES        = 24            # frames per latent clip (1 s at 24 fps)
LATENT_H = LATENT_W = 96          # CogVideoX VAE spatial compression: 768 / 8
LATENT_CHANNELS    = 16
INNER_DIM          = 1920          # CogVideoX-2B hidden dim (30 heads × 64)
AUDIO_INPUT_DIM    = 1920          # AudioEncoder output_dim (matches 2B)
AUDIO_TOKENS_PER_FRAME = 1         # reduced from 4 for memory (matches training)
CANVAS             = 768.0         # keypoint normalisation canvas


# ── Helpers ────────────────────────────────────────────────────────────────

def build_audio_encoder_with_lora(model_name: str) -> AudioEncoder:
    """Rebuild AudioEncoder with LoRA on the last transformer block (same as training)."""
    encoder = AudioEncoder(
        output_dim=AUDIO_INPUT_DIM,
        freeze_encoder=True,
        model_name=model_name,
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    encoder.wav2vec2.encoder.layers[-1] = get_peft_model(
        encoder.wav2vec2.encoder.layers[-1], lora_config
    )
    return encoder


def load_checkpoint(
    ckpt_path: str,
    dit: TalkingHeadsDiT,
    audio_enc: AudioEncoder,
) -> None:
    """Load trainable weights from a checkpoint.pt saved by train.py."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing_dit, unexpected_dit = dit.load_state_dict(
        ckpt["dit_trainable"], strict=False
    )
    missing_ae, unexpected_ae = audio_enc.load_state_dict(
        ckpt["audio_enc_lora"], strict=False
    )
    iteration = ckpt.get("iteration", "?")
    print(f"[CKPT] Loaded checkpoint from iteration {iteration}")
    if missing_dit:
        print(f"  [DiT]   Missing keys  : {len(missing_dit)}")
    if missing_ae:
        print(f"  [Audio] Missing keys  : {len(missing_ae)}")


def encode_reference_image(
    image_path: str,
    vae: AutoencoderKLCogVideoX,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Encode a reference portrait image to a latent using the CogVideoX VAE.

    Returns : (1, 16, 1, 96, 96)
    """
    img = Image.open(image_path).convert("RGB").resize((768, 768), Image.LANCZOS)
    img_t = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [-1, 1]
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)        # (1, 3, 1, H, W)
    img_t = img_t.to(device, dtype=torch.float32)

    with torch.no_grad():
        # CogVideoX VAE encode expects (B, C, T, H, W)
        latent = vae.encode(img_t).latent_dist.sample()              # (1, 16, 1, 96, 96)
        latent = latent * vae.config.scaling_factor
    return latent.to(dtype)


def load_full_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Load full audio at 16 kHz and calculate the total video frames it corresponds to.
    Returns: (waveform_tensor, total_frames)
    """
    waveform, _ = librosa.load(audio_path, sr=AUDIO_SR, mono=True)
    total_frames = math.ceil(len(waveform) / AUDIO_SR * VIDEO_FPS)
    return torch.from_numpy(waveform).float(), total_frames


def load_pose_keypoints(
    pose_path: str | None,
    total_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """
    Load pose keypoints from a .npy file (same format as preprocessing).
    Returns (1, T, 133, 2) normalised tensor, or None if pose_path is None.
    """
    if pose_path is None:
        return None

    pose_data = np.load(pose_path, allow_pickle=True)
    kp_frames = []
    for frame_dict in pose_data[:total_frames]:
        kp = frame_dict.get("keypoints", None)
        if kp is None or len(kp) == 0:
            kp = np.zeros((133, 2), dtype=np.float32)
        kp_frames.append(kp.astype(np.float32))

    while len(kp_frames) < total_frames:
        kp_frames.append(np.zeros((133, 2), dtype=np.float32))

    keypoints = torch.from_numpy(np.stack(kp_frames))    # (T, 133, 2)
    keypoints = (keypoints / CANVAS).clamp(0.0, 1.0)     # normalise
    return keypoints.unsqueeze(0).to(device, dtype=dtype) # (1, T, 133, 2)


def decode_latents_to_frames(
    latents: torch.Tensor,
    vae: AutoencoderKLCogVideoX,
) -> list[np.ndarray]:
    """
    Decode a batch of video latents to uint8 RGB frames.

    Args:
        latents : (1, 16, T, H, W)
        vae     : frozen CogVideoX VAE

    Returns:
        list of T numpy arrays, each (768, 768, 3) uint8
    """
    T = latents.shape[2]
    frames = []
    with torch.no_grad():
        for t in range(T):
            z = latents[:, :, t:t+1, :, :].float() / vae.config.scaling_factor
            rgb = vae.decode(z).sample          # (1, 3, 1, H, W) or (1, 3, H, W)
            if rgb.dim() == 5:
                rgb = rgb.squeeze(2)            # (1, 3, H, W)
            rgb = rgb.squeeze(0)                # (3, H, W)
            rgb = rgb.clamp(-1.0, 1.0)
            rgb = ((rgb + 1.0) * 127.5).byte()  # [0, 255]
            frames.append(rgb.permute(1, 2, 0).cpu().numpy())
    return frames


# ── DDIM sampling loop ─────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    dit: TalkingHeadsDiT,
    scheduler: DDIMScheduler,
    ref_latents: torch.Tensor,          # (1, 16, 1, 96, 96)
    audio_embeds: torch.Tensor,         # (1, T, AUDIO_INPUT_DIM)
    pose_keypoints: torch.Tensor | None,# (1, T, 133, 2) or None
    clip_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 50,
    cfg_scale: float = 3.5,
    seed: int = 42,
) -> torch.Tensor:
    """
    Run DDIM reverse diffusion to generate video latents.

    Returns : (1, 16, T, H, W) denoised latent
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    # Start from pure noise
    latents = torch.randn(
        1, LATENT_CHANNELS, clip_frames, LATENT_H, LATENT_W,
        generator=generator, device=device, dtype=dtype,
    )

    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)

    # For CFG: null audio = zeros
    null_audio = torch.zeros_like(audio_embeds)

    do_cfg = cfg_scale > 1.0

    for t in tqdm(timesteps, desc="  DDIM", leave=False):
        t_batch = t.unsqueeze(0)

        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(device.type == "cuda")):
            # Conditioned prediction
            noise_pred_cond = dit(
                video_latents  = latents,
                ref_latents    = ref_latents,
                timestep       = t_batch,
                audio_embeds   = audio_embeds,
                pose_keypoints = pose_keypoints,
            )

            if do_cfg:
                # Unconditioned prediction (null audio, no pose)
                noise_pred_uncond = dit(
                    video_latents  = latents,
                    ref_latents    = ref_latents,
                    timestep       = t_batch,
                    audio_embeds   = null_audio,
                    pose_keypoints = None,
                )
                # CFG guidance
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

        # Process all frames at once by treating Time as the Batch dimension
        # This prevents internal _step_index corruption in diffusers schedulers!
        noise_pred_batched = noise_pred.squeeze(0).transpose(0, 1)  # (T, 16, H, W)
        latents_batched = latents.squeeze(0).transpose(0, 1)        # (T, 16, H, W)
        
        step_out = scheduler.step(
            noise_pred_batched,
            t,
            latents_batched,
        )
        
        # Reshape back to (1, 16, T, H, W)
        latents = step_out.prev_sample.transpose(0, 1).unsqueeze(0)

    return latents


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TalkingHeads Inference — generate talking-head video from audio + reference image."
    )
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to checkpoint.pt (e.g. ../checkpoints/checkpoint_070000/checkpoint.pt)")
    parser.add_argument("--ref_image",   type=str, required=True,
                        help="Path to reference portrait image (.jpg / .png). Will be resized to 768×768.")
    parser.add_argument("--audio",       type=str, required=True,
                        help="Path to driving audio file (.wav / .m4a). First N seconds used.")
    parser.add_argument("--output",      type=str, default="output.mp4",
                        help="Output video path (default: output.mp4).")
    parser.add_argument("--pose",        type=str, default=None,
                        help="Optional .npy pose file (same format as preprocessing). "
                             "If omitted the model runs audio-only.")
    parser.add_argument("--steps",       type=int, default=50,
                        help="Number of DDIM denoising steps (default: 50).")
    parser.add_argument("--fps",         type=int, default=VIDEO_FPS,
                        help="Output video FPS (default: 24).")
    parser.add_argument("--cfg_scale",   type=float, default=3.5,
                        help="Classifier-free guidance scale (default: 3.5). "
                             "Set to 1.0 to disable.")
    parser.add_argument("--chunk_size",  type=int, default=CLIP_FRAMES,
                        help="Max frames to generate per chunk to save VRAM (default: 24). Total length is auto-determined by audio.")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--pretrained_model", type=str, default=PRETRAINED_MODEL,
                        help=f"HuggingFace model ID for the CogVideoX backbone (default: {PRETRAINED_MODEL}).")
    parser.add_argument("--wav2vec2_model",   type=str, default=WAV2VEC2_MODEL,
                        help=f"HuggingFace model ID for Wav2Vec2 (default: {WAV2VEC2_MODEL}).")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # H100 supports bfloat16 natively and prevents the NaN/static noise overflows common in fp16
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\n[INFERENCE] Device: {device} | dtype: {dtype}\n")

    # ── 1. Load models ─────────────────────────────────────────────────────
    print("[INFERENCE] Loading TalkingHeadsDiT (backbone from pretrained)...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        args.pretrained_model,
        freeze_backbone        = True,
        gradient_checkpointing = False,   # not needed for inference
        audio_input_dim        = AUDIO_INPUT_DIM,
        audio_tokens_per_frame = AUDIO_TOKENS_PER_FRAME,
    ).to(device, dtype=dtype)

    print("\n[INFERENCE] Loading AudioEncoder with LoRA...")
    audio_enc = build_audio_encoder_with_lora(args.wav2vec2_model).to(device)

    print("\n[INFERENCE] Loading VAE (frozen)...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    print("\n[INFERENCE] Loading DDIM scheduler...")
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    # ── 2. Load checkpoint ─────────────────────────────────────────────────
    print(f"\n[INFERENCE] Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, dit, audio_enc)
    dit.eval()
    audio_enc.eval()

    # ── 3. Encode inputs ───────────────────────────────────────────────────
    print(f"\n[INFERENCE] Loading full audio: {args.audio}")
    waveform, total_frames = load_full_audio(args.audio)
    print(f"  Total frames to generate based on audio length: {total_frames}")

    pose_keypoints = None
    if args.pose:
        print(f"\n[INFERENCE] Loading pose keypoints: {args.pose}")
        pose_keypoints = load_pose_keypoints(
            args.pose, total_frames, device, dtype
        )
        print(f"  pose_keypoints : {tuple(pose_keypoints.shape)}")
    else:
        print("\n[INFERENCE] No pose file provided — running audio-only mode.")

    print(f"\n[INFERENCE] Encoding reference image: {args.ref_image}")
    current_ref_latents = encode_reference_image(args.ref_image, vae, device, dtype)
    print(f"  ref_latents  : {tuple(current_ref_latents.shape)}")

    # ── 4. Autoregressive Chunking & DDIM sampling ─────────────────────────
    all_video_latents = []
    
    print(f"\n[INFERENCE] Running Autoregressive Chunking ({args.steps} steps, cfg={args.cfg_scale})...")
    for start_frame in range(0, total_frames, args.chunk_size):
        end_frame = min(start_frame + args.chunk_size, total_frames)
        chunk_frames = end_frame - start_frame
        
        print(f"\n  => Generating chunk: frames {start_frame} to {end_frame} ({chunk_frames} frames)...")
        
        # 4a. Extract corresponding audio chunk
        start_sample = int(start_frame * AUDIO_SR / VIDEO_FPS)
        end_sample = int(end_frame * AUDIO_SR / VIDEO_FPS)
        audio_chunk = waveform[start_sample:end_sample]
        
        expected_samples = int(chunk_frames * AUDIO_SR / VIDEO_FPS)
        if len(audio_chunk) < expected_samples:
            pad_len = expected_samples - len(audio_chunk)
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad_len))
            
        audio_chunk = audio_chunk.to(device)
        
        print("     Encoding audio chunk...")
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                audio_embeds = audio_enc(
                    audio_chunk.unsqueeze(0).float(),
                    target_frames=chunk_frames,
                )
                
        # 4b. Extract corresponding pose chunk
        pose_chunk = None
        if pose_keypoints is None:
            pose_chunk = None
        else:
            pose_chunk = pose_keypoints[:, start_frame:end_frame, :, :]
            
        # 4c. Generate DDIM latents
        chunk_latents = ddim_sample(
            dit            = dit,
            scheduler      = scheduler,
            ref_latents    = current_ref_latents,
            audio_embeds   = audio_embeds,
            pose_keypoints = pose_chunk,
            clip_frames    = chunk_frames,
            device         = device,
            dtype          = dtype,
            num_steps      = args.steps,
            cfg_scale      = args.cfg_scale,
            seed           = args.seed + start_frame,
        )
        
        # Move latents to CPU to save GPU VRAM
        all_video_latents.append(chunk_latents.cpu())
        
        # 4d. The last frame of this chunk becomes the reference image for the next chunk!
        current_ref_latents = chunk_latents[:, :, -1:, :, :]

    # ── 5. Decode to RGB frames ────────────────────────────────────────────
    print("\n[INFERENCE] Decoding all latents to RGB frames...")
    frames = []
    for chunk_latents in all_video_latents:
        # Move back to GPU chunk-by-chunk for decoding
        chunk_latents = chunk_latents.to(device)
        chunk_frames_rgb = decode_latents_to_frames(chunk_latents, vae)
        frames.extend(chunk_frames_rgb)
        
    print(f"  Decoded {len(frames)} total frame(s)")

    # ── 6. Write video ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFERENCE] Writing video → {out_path} @ {args.fps} fps...")

    try:
        with imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                                 quality=8, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"[INFERENCE] Done. Video saved to: {out_path.resolve()}")
    except Exception as e:
        # Fallback: save frames as PNG if video writer fails
        print(f"[WARN] Video writer failed ({e}). Saving frames as PNGs...")
        frames_dir = out_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"[INFERENCE] Frames saved to: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
