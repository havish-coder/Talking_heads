"""
inference.py

Talking Heads Inference Pipeline.

Generates a talking-head video utilizing:
  1. A reference image (.jpg / .png)
  2. A driving audio file (.wav / .m4a)
  3. A trained model checkpoint (`checkpoint.pt` produced by `train.py`)

Usage (from the Src/ directory):
  python inference.py \
      --checkpoint ../checkpoints/checkpoint_070000/checkpoint.pt \
      --ref_image path/to/portrait.jpg \
      --audio path/to/speech.wav \
      --output output.mp4 \
      --steps 50 \
      --fps 24 \
      --cfg_scale 3.5

Pipeline Overview:
  1. Initialize TalkingHeadsDiT + AudioEncoder (matching training architecture).
  2. Load trainable weights from the checkpoint (strict=False; frozen backbone 
     weights are pulled from the CogVideoX pretrained model).
  3. Encode the reference image using the CogVideoX VAE.
  4. Autoregressive Generation: Process the video in chunks. The model generates 
     a latent clip using the audio chunk, pose chunk (optional), and the reference 
     image. The final frame of the generated clip becomes the reference image for 
     the subsequent chunk.
  5. Run DDIM denoising for each chunk (substantially faster than DDPM).
  6. Decode the generated latents frame-by-frame via the VAE.
  7. Export the decoded frames to an MP4 file.

Notes:
  - Pose keypoints are OPTIONAL. If a pose `.npy` file is provided via `--pose`, 
    it will guide generation. Otherwise, the model defaults to audio-only 
    (mimicking APDH Stage 4 behavior).
  - Classifier-Free Guidance (CFG) runs two forward passes per step: one conditioned, 
    and one unconditioned (zeroed audio/pose). Disable CFG by setting `--cfg_scale 1.0`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, DDIMScheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import Wav2Vec2Model

# --- Local Imports ---
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from Models.audio_encoder import AudioEncoder
from Models.talking_heads_dit import TalkingHeadsDiT

# --- Constants ---
PRETRAINED_MODEL = "THUDM/CogVideoX-2b"
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
AUDIO_SR = 16_000        # Hz (Wav2Vec2 requirement)
VIDEO_FPS = 24
CLIP_FRAMES = 24         # Frames per latent clip (1 sec at 24 FPS)
LATENT_H = 96            # CogVideoX VAE spatial compression (768 / 8)
LATENT_W = 96
LATENT_CHANNELS = 16
INNER_DIM = 1920         # CogVideoX-2B hidden dimension (30 heads * 64)
AUDIO_INPUT_DIM = 1920   # AudioEncoder output dimension
AUDIO_TOKENS_PER_FRAME = 1  # Memory-optimized setting matching training
CANVAS = 768.0           # Normalization constant for keypoints


# --- Helpers ---

def build_audio_encoder_with_lora(model_name: str) -> AudioEncoder:
    """Rebuilds the AudioEncoder with LoRA injected into the final transformer block."""
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
    
    last_layer = encoder.wav2vec2.encoder.layers[-1]
    encoder.wav2vec2.encoder.layers[-1] = get_peft_model(last_layer, lora_config)
    
    return encoder


def load_checkpoint(ckpt_path: str, dit: TalkingHeadsDiT, audio_enc: AudioEncoder) -> None:
    """Loads trainable weights from a trained checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    missing_dit, _ = dit.load_state_dict(ckpt["dit_trainable"], strict=False)
    missing_ae, _ = audio_enc.load_state_dict(ckpt["audio_enc_lora"], strict=False)
    
    iteration = ckpt.get("iteration", "Unknown")
    print(f"[CKPT] Successfully loaded weights from iteration {iteration}")
    
    if missing_dit:
        print(f"  -> [DiT] Expected missing frozen keys: {len(missing_dit)}")
    if missing_ae:
        print(f"  -> [Audio] Expected missing frozen keys: {len(missing_ae)}")


def encode_reference_image(
    image_path: str,
    vae: AutoencoderKLCogVideoX,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encodes a reference portrait image into latent space using the CogVideoX VAE."""
    img = Image.open(image_path).convert("RGB").resize((768, 768), Image.LANCZOS)
    
    # Normalize to [-1, 1] and prepare shape: (1, 3, 1, H, W)
    img_t = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    img_t = img_t.to(device, dtype=torch.float32)

    with torch.no_grad():
        latent = vae.encode(img_t).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        
    return latent.to(dtype)


def load_full_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Loads driving audio at 16kHz and calculates the corresponding target video frames."""
    waveform, _ = librosa.load(audio_path, sr=AUDIO_SR, mono=True)
    total_frames = math.ceil(len(waveform) / AUDIO_SR * VIDEO_FPS)
    return torch.from_numpy(waveform).float(), total_frames


def load_pose_keypoints(
    pose_path: Optional[str],
    total_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Loads and normalizes pose keypoints from a .npy file."""
    if not pose_path:
        return None

    pose_data = np.load(pose_path, allow_pickle=True)
    kp_frames = []
    
    for frame_dict in pose_data[:total_frames]:
        kp = frame_dict.get("keypoints", None)
        if kp is None or len(kp) == 0:
            kp = np.zeros((133, 2), dtype=np.float32)
        kp_frames.append(kp.astype(np.float32))

    # Pad with zeros if pose data is shorter than audio duration
    while len(kp_frames) < total_frames:
        kp_frames.append(np.zeros((133, 2), dtype=np.float32))

    keypoints = torch.from_numpy(np.stack(kp_frames))
    keypoints = (keypoints / CANVAS).clamp(0.0, 1.0)
    
    return keypoints.unsqueeze(0).to(device, dtype=dtype)


def decode_latents_to_frames(
    latents: torch.Tensor,
    vae: AutoencoderKLCogVideoX,
) -> List[np.ndarray]:
    """Decodes a batch of generated video latents back into uint8 RGB frames."""
    T = latents.shape[2]
    frames = []
    
    with torch.no_grad():
        for t in range(T):
            z = latents[:, :, t:t+1, :, :].float() / vae.config.scaling_factor
            rgb = vae.decode(z).sample
            
            if rgb.dim() == 5:
                rgb = rgb.squeeze(2)
                
            rgb = rgb.squeeze(0).clamp(-1.0, 1.0)
            rgb = ((rgb + 1.0) * 127.5).byte()
            
            frames.append(rgb.permute(1, 2, 0).cpu().numpy())
            
    return frames


# --- DDIM Sampling Loop ---

@torch.no_grad()
def ddim_sample(
    dit: TalkingHeadsDiT,
    scheduler: DDIMScheduler,
    ref_latents: torch.Tensor,
    audio_embeds: torch.Tensor,
    pose_keypoints: Optional[torch.Tensor],
    clip_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 50,
    cfg_scale: float = 3.5,
    seed: int = 42,
) -> torch.Tensor:
    """Executes the DDIM reverse diffusion process to generate video latents."""
    generator = torch.Generator(device=device).manual_seed(seed)

    # Initialize pure noise
    latents = torch.randn(
        1, LATENT_CHANNELS, clip_frames, LATENT_H, LATENT_W,
        generator=generator, device=device, dtype=dtype,
    )

    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)

    do_cfg = cfg_scale > 1.0
    null_audio = torch.zeros_like(audio_embeds) if do_cfg else None

    for t in tqdm(timesteps, desc="  -> DDIM Denoising", leave=False):
        t_batch = t.unsqueeze(0)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            # Conditioned prediction
            noise_pred_cond = dit(
                video_latents=latents,
                ref_latents=ref_latents,
                timestep=t_batch,
                audio_embeds=audio_embeds,
                pose_keypoints=pose_keypoints,
            )

            if do_cfg:
                # Unconditioned prediction
                noise_pred_uncond = dit(
                    video_latents=latents,
                    ref_latents=ref_latents,
                    timestep=t_batch,
                    audio_embeds=null_audio,
                    pose_keypoints=None,
                )
                # Apply Classifier-Free Guidance
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

        # Step scheduler. 
        # Trick: Transpose Time into the Batch dimension to process all frames cleanly
        # and prevent internal `_step_index` corruption in diffusers schedulers.
        noise_pred_batched = noise_pred.squeeze(0).transpose(0, 1)
        latents_batched = latents.squeeze(0).transpose(0, 1)
        
        step_out = scheduler.step(noise_pred_batched, t, latents_batched)
        
        # Reshape back to standard layout
        latents = step_out.prev_sample.transpose(0, 1).unsqueeze(0)

    return latents


# --- Main Pipeline ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TalkingHeads Inference Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference portrait (.jpg/.png)")
    parser.add_argument("--audio", type=str, required=True, help="Path to driving audio (.wav/.m4a)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--pose", type=str, default=None, help="Optional .npy pose file (guides generation if provided)")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS, help="Output video FPS")
    parser.add_argument("--cfg_scale", type=float, default=3.5, help="CFG scale (1.0 disables it)")
    parser.add_argument("--chunk_size", type=int, default=CLIP_FRAMES, help="Max frames per generation chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pretrained_model", type=str, default=PRETRAINED_MODEL, help="CogVideoX backbone ID")
    parser.add_argument("--wav2vec2_model", type=str, default=WAV2VEC2_MODEL, help="Wav2Vec2 model ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- 1. Environment Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bfloat16 highly recommended on Ampere+ to prevent static noise overflows common in fp16
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    print(f"\n[SYSTEM] Inference Hardware Setup")
    print(f"  -> Device: {device}")
    print(f"  -> Dtype : {dtype}\n")

    # --- 2. Model Initialization ---
    print("[INFO] Initializing TalkingHeadsDiT Backbone...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        args.pretrained_model,
        freeze_backbone=True,
        gradient_checkpointing=False,
        audio_input_dim=AUDIO_INPUT_DIM,
        audio_tokens_per_frame=AUDIO_TOKENS_PER_FRAME,
    ).to(device, dtype=dtype)

    print("[INFO] Initializing AudioEncoder (Wav2Vec2 + LoRA)...")
    audio_enc = build_audio_encoder_with_lora(args.wav2vec2_model).to(device)

    print("[INFO] Initializing VAE and Scheduler...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # --- 3. Checkpoint Loading ---
    print(f"\n[INFO] Restoring Checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, dit, audio_enc)
    dit.eval()
    audio_enc.eval()

    # --- 4. Input Encoding ---
    print(f"\n[INFERENCE] Processing Inputs")
    waveform, total_frames = load_full_audio(args.audio)
    print(f"  -> Target Duration: {total_frames} frames (derived from audio)")

    if args.pose:
        print(f"  -> Pose Guidance  : Enabled ({args.pose})")
        pose_keypoints = load_pose_keypoints(args.pose, total_frames, device, dtype)
    else:
        print("  -> Pose Guidance  : Disabled (Audio-only mode active)")
        pose_keypoints = None

    print(f"  -> Encoding Reference Image: {args.ref_image}")
    current_ref_latents = encode_reference_image(args.ref_image, vae, device, dtype)

    # --- 5. Autoregressive Chunking & Generation ---
    all_video_latents = []
    
    print(f"\n[INFERENCE] Commencing Autoregressive DDIM Sampling ({args.steps} steps | CFG: {args.cfg_scale})")
    
    for start_frame in range(0, total_frames, args.chunk_size):
        end_frame = min(start_frame + args.chunk_size, total_frames)
        chunk_frames = end_frame - start_frame
        
        print(f"\n[INFERENCE] Generating Chunk [{start_frame} : {end_frame}] ({chunk_frames} frames)")
        
        # 5a. Process Audio Chunk
        start_sample = int(start_frame * AUDIO_SR / VIDEO_FPS)
        end_sample = int(end_frame * AUDIO_SR / VIDEO_FPS)
        audio_chunk = waveform[start_sample:end_sample]
        
        expected_samples = int(chunk_frames * AUDIO_SR / VIDEO_FPS)
        if len(audio_chunk) < expected_samples:
            pad_len = expected_samples - len(audio_chunk)
            audio_chunk = F.pad(audio_chunk, (0, pad_len))
            
        audio_chunk = audio_chunk.to(device)
        
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            audio_embeds = audio_enc(audio_chunk.unsqueeze(0).float(), target_frames=chunk_frames)
                
        # 5b. Extract Pose Chunk
        pose_chunk = pose_keypoints[:, start_frame:end_frame, :, :] if pose_keypoints is not None else None
            
        # 5c. Run Diffusion
        chunk_latents = ddim_sample(
            dit=dit,
            scheduler=scheduler,
            ref_latents=current_ref_latents,
            audio_embeds=audio_embeds,
            pose_keypoints=pose_chunk,
            clip_frames=chunk_frames,
            device=device,
            dtype=dtype,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed + start_frame,
        )
        
        # Offload to CPU to maintain VRAM stability
        all_video_latents.append(chunk_latents.cpu())
        
        # 5d. Autoregressive Update: Last frame becomes new reference image
        current_ref_latents = chunk_latents[:, :, -1:, :, :]

    # --- 6. Decode Latents ---
    print("\n[INFERENCE] Decoding Latents to RGB...")
    frames = []
    for chunk_latents in all_video_latents:
        # Move chunk back to GPU strictly for VAE decoding
        chunk_frames_rgb = decode_latents_to_frames(chunk_latents.to(device), vae)
        frames.extend(chunk_frames_rgb)
        
    print(f"  -> Successfully decoded {len(frames)} frames.")

    # --- 7. Export Video ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[SYSTEM] Exporting Video to {out_path.resolve()} (@ {args.fps} FPS)...")

    try:
        with imageio.get_writer(
            str(out_path), fps=args.fps, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        print("[SYSTEM] Export Complete. Inference Terminated Successfully.")
        
    except Exception as e:
        print(f"\n[WARN] Primary video writer failed ({e}).")
        print("[WARN] Executing Fallback: Saving individual frames as PNGs...")
        frames_dir = out_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"[SYSTEM] Fallback Complete. Frames saved to: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()