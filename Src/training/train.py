"""
training/train.py

Main training loop for the TalkingHeadsDiT model — heavily optimized for T4 (12GB) / Colab environments.

Memory Optimizations Applied:
  - Mixed Precision       : fp16/bf16 native support.
  - Batching              : batch_size=1, clip_frames=6.
  - VAE Offloading        : VAE is kept on CPU/GPU conditionally; moved to GPU only for PhD auxiliary loss decoding.
  - Audio Context         : Audio tokens per frame reduced from 4 to 1 (saves ~4x cross-attention memory).
  - Sparse Aux Loss       : PhD auxiliary loss is computed on only ~25% of iterations (saves ~4 GB/iter).
  - Memory Management     : Aggressive intermediate tensor cleanup (`del` + `empty_cache`).
  - Checkpointing         : Gradient checkpointing enabled on the DiT backbone.

APDH Curriculum Schedule:
  - Stage 1 (0-10k)       : Full pose, audio frozen.
  - Stage 2 (10k-20k)     : No lips, audio mapped to lips.
  - Stage 3 (20k-30k)     : No head, audio mapped to face.
  - Stage 4 (30k-40k)     : Hands only, audio mapped globally.

Usage:
  python training/train.py --config training/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKLCogVideoX, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.amp import GradScaler
from tqdm import tqdm
from transformers import Wav2Vec2Model

# Disable Python bytecode caching to prevent Google Drive __pycache__ corruption
sys.dont_write_bytecode = True

# Ensure local Models and training directories are importable
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from Models.audio_encoder import AudioEncoder
from Models.talking_heads_dit import TalkingHeadsDiT
from training.dataset import get_dataloader
from training.loss import PhDLoss

# --- APDH Curriculum Configurations ---

# Format: (start_iter, end_iter, stage, iterative_prob, unfreeze_n_layers)
APDH_SCHEDULE = [
    (0, 50, 1, 0.0, 0),
    (50, 1500, 2, 0.05, 4),
    (1500, 3000, 3, 0.10, 8),
    (3000, 4200, 4, 0.20, 15),
    (4200, 5400, 4, 0.20, 30),
    (5400, 99999, 4, 0.20, 30),
]


def get_apdh_stage(iteration: int) -> Tuple[int, float, int]:
    """Retrieves the APDH curriculum parameters for the current training iteration."""
    for start, end, stage, prob, layers in APDH_SCHEDULE:
        if start <= iteration < end:
            return stage, prob, layers
    return 4, 0.20, 30  # Default fallback


# --- Audio Encoder & LoRA Setup ---

def build_audio_encoder_with_lora(model_name: str = "facebook/wav2vec2-base") -> AudioEncoder:
    """
    Builds the AudioEncoder with LoRA injected strictly into the last transformer block.
    Rank=4, Alpha=8 reduces trainable parameters from ~7M to ~50k.
    Output dimension is mapped to 1920 to align with CogVideoX-2B's inner dimension.
    """
    encoder = AudioEncoder(
        output_dim=1920,
        freeze_encoder=True,
        model_name=model_name,
    )

    # Apply LoRA exclusively to the final block's attention projections
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    last_layer = encoder.wav2vec2.encoder.layers[-1]
    encoder.wav2vec2.encoder.layers[-1] = get_peft_model(last_layer, lora_config)

    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[INFO] AudioEncoder initialized with LoRA. Trainable params: {trainable_params:,}")
    return encoder


# --- Checkpoint Management ---

def save_checkpoint(
    iteration: int,
    dit: TalkingHeadsDiT,
    audio_enc: AudioEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    output_dir: str,
) -> None:
    """Saves only the unfrozen/trainable weights to conserve storage."""
    ckpt_dir = Path(output_dir) / f"checkpoint_{iteration:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Extract only the weights that require gradients or are explicitly managed
    dit_trainable = {
        k: v for k, v in dit.state_dict().items()
        if dit.state_dict()[k].requires_grad
        or k.startswith("audio_proj")
        or k.startswith("pose_encoder")
        or k.startswith("pose_scale")
    }

    torch.save({
        "iteration": iteration,
        "dit_trainable": dit_trainable,
        "audio_enc_lora": audio_enc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, ckpt_dir / "checkpoint.pt")

    print(f"[CKPT] Saved checkpoint -> {ckpt_dir}")


def load_checkpoint(
    ckpt_path: str,
    dit: TalkingHeadsDiT,
    audio_enc: AudioEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> int:
    """Loads weights and safely restores optimizer states if layer counts match."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. Load Model Weights
    dit.load_state_dict(ckpt["dit_trainable"], strict=False)
    audio_enc.load_state_dict(ckpt["audio_enc_lora"], strict=False)

    # 2. Safely Load Optimizer (Handles curriculum unfreezing mismatches)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except ValueError:
        print("\n[WARN] Optimizer size mismatch detected (due to curriculum unfreezing).")
        print("[WARN] Model weights loaded successfully, but optimizer momentum is reset to prevent crashes.\n")

    # 3. Safely Load GradScaler
    try:
        scaler.load_state_dict(ckpt["scaler"])
    except Exception:
        print("[WARN] Could not load GradScaler state. Resetting scaler.")

    iteration = ckpt["iteration"]
    print(f"[CKPT] Successfully resumed from iteration {iteration}")
    return iteration


def vae_decode_frame(
    latent: torch.Tensor,
    vae: AutoencoderKLCogVideoX,
    device: torch.device,
) -> torch.Tensor:
    """Decodes a single frame latent to RGB space."""
    with torch.no_grad():
        rgb = vae.decode(latent.to(device).float()).sample
        if rgb.dim() == 5:
            rgb = rgb.squeeze(2)
        rgb = rgb.clamp(-1.0, 1.0)
    return rgb


# --- Main Training Routine ---

def train(cfg: dict) -> None:
    # --- 1. Environment & Hardware Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Upgrade to bfloat16 natively if supported (A100/H100), fallback to fp16 (T4)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"[SYSTEM] Hardware -> Device: {device} | Dtype: {dtype}")
    if device.type == "cuda":
        print(f"[SYSTEM] GPU -> {torch.cuda.get_device_name(0)}")
        print(f"[SYSTEM] VRAM -> {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # --- 2. Model Initialization ---
    print("[INFO] Loading TalkingHeadsDiT (CogVideoX-2B Backbone)...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        cfg["pretrained_model"],
        freeze_backbone=True,
        gradient_checkpointing=True,
        audio_input_dim=1920,
        audio_tokens_per_frame=1,  # Critical memory save: reduced from 4 to 1
    ).to(device, dtype=dtype)
    print(dit.param_summary())

    # Force VRAM cleanup after heavy model load
    gc.collect()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        print(f"[MEM] VRAM Reserved after DiT load: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print("\n[INFO] Loading AudioEncoder & VAE...")
    audio_enc = build_audio_encoder_with_lora(cfg["wav2vec2_model"]).to(device)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        cfg["pretrained_model"], subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = DDPMScheduler.from_pretrained(cfg["pretrained_model"], subfolder="scheduler")

    loss_fn = PhDLoss(
        lambda_pose=cfg.get("lambda_pose", 0.1),
        lambda_detail=cfg.get("lambda_detail", 0.1),
        lambda_low=cfg.get("lambda_low", 0.1),
    ).to(device)

    # --- 3. Resume State & Curriculum Pre-Unfreeze ---
    start_iter = 0
    current_stage = -1

    if cfg.get("resume_from"):
        # Temporarily peek at checkpoint to align the curriculum stage before building optimizer
        ckpt_tmp = torch.load(cfg["resume_from"], map_location="cpu", weights_only=False)
        start_iter = ckpt_tmp["iteration"]
        del ckpt_tmp
        
        current_stage, iterative_prob, n_layers = get_apdh_stage(start_iter)
        dit.unfreeze_backbone_top_n_layers(n_layers)

    # --- 4. Optimizer Configurations ---
    trainable_params = (
        [p for p in dit.parameters() if p.requires_grad] +
        [p for p in audio_enc.parameters() if p.requires_grad]
    )
    
    # CRITICAL FIX: Force trainable parameters to fp32 to prevent GradScaler crashes
    for p in trainable_params:
        p.data = p.data.to(torch.float32)

    print(f"\n[INFO] Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.get("lr", 1e-5),
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    scaler = GradScaler(enabled=(dtype == torch.float16))

    # --- 5. Data Loading & Post-Resume ---
    dataset, loader = get_dataloader(
        root_dir=cfg["data_root"],
        batch_size=cfg.get("batch_size", 1),
        clip_frames=cfg.get("clip_frames", 6),
        num_workers=cfg.get("num_workers", 2),
        apdh_stage=1 if current_stage == -1 else current_stage,
    )

    if cfg.get("resume_from"):
        load_checkpoint(cfg["resume_from"], dit, audio_enc, optimizer, scaler)

    # --- 6. Execution Parameters ---
    aux_loss_prob = cfg.get("aux_loss_prob", 0.25)
    warmup_no_aux = cfg.get("warmup_no_aux", 500)
    total_iters   = cfg.get("total_iters", 40000)
    save_every    = cfg.get("save_every", 500)
    log_every     = cfg.get("log_every", 25)
    output_dir    = cfg.get("output_dir", "checkpoints")

    dit.train()
    audio_enc.train()
    data_iter = iter(loader)
    t0 = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"\n[MEM] Pre-loop VRAM Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # --- 7. Main Training Loop ---
    pbar = tqdm(range(start_iter, total_iters), initial=start_iter, total=total_iters, desc="Training")
    for iteration in pbar:

        # -- A. APDH Curriculum Check --
        stage, iterative_prob, n_layers = get_apdh_stage(iteration)
        if stage != current_stage:
            current_stage = stage
            dataset.set_apdh_stage(stage, iterative_prob)
            dit.unfreeze_backbone_top_n_layers(n_layers)
            
            # Re-initialize optimizer for newly unfrozen parameters
            trainable_params = (
                [p for p in dit.parameters() if p.requires_grad] +
                [p for p in audio_enc.parameters() if p.requires_grad]
            )
            for p in trainable_params:
                p.data = p.data.to(torch.float32)
                
            optimizer = torch.optim.AdamW(
                trainable_params, lr=cfg.get("lr", 1e-5), betas=(0.9, 0.999), weight_decay=1e-2
            )
            tqdm.write(f"\n[APDH] Transitioned to Stage {stage} | Trainable Params: {sum(p.numel() for p in trainable_params):,}")

        # -- B. Batch Extraction --
        try:
            batch = next(data_iter)
        except StopIteration:
            torch.cuda.empty_cache()
            data_iter = iter(loader)
            batch = next(data_iter)

        video_latents = batch["video_latents"].to(device, dtype=dtype)
        ref_latents = batch["ref_latents"].to(device, dtype=dtype)
        pose_keypoints = batch["pose_keypoints"].to(device, dtype=dtype)
        audio_waveform = batch["audio_waveform"].to(device)
        del batch

        # -- C. Audio Encoding --
        T = video_latents.shape[2]
        with torch.no_grad():
            # Process Wav2Vec2 in FP32 to prevent FP16 NaNs, then cast back to required dtype
            audio_embeds = audio_enc(audio_waveform.float(), target_frames=T).detach()
            audio_embeds = audio_embeds.to(dtype)
        del audio_waveform

        # -- D. Diffusion Forward Pass --
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        t_val = int(t.item())
        noise = torch.randn_like(video_latents)
        noisy_latent = scheduler.add_noise(video_latents.float(), noise.float(), t).to(dtype)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype):
            noise_pred = dit(
                video_latents=noisy_latent,
                ref_latents=ref_latents,
                timestep=t.expand(video_latents.shape[0]),
                audio_embeds=audio_embeds,
                pose_keypoints=pose_keypoints,
            )
        del ref_latents, pose_keypoints, audio_embeds

        # -- E. Loss Computation --
        use_aux = (iteration >= warmup_no_aux and random.random() < aux_loss_prob)

        if use_aux:
            # Requires expensive VAE Decode
            mid = T // 2
            z_target = video_latents[:, :, mid:mid+1, :, :].detach()
            target_rgb = vae_decode_frame(z_target, vae, device)
            del z_target

            total_loss, log_dict = loss_fn(
                noise_pred=noise_pred.float(),
                actual_noise=noise.float(),
                noisy_latent=noisy_latent.float(),
                timestep=t,
                target_rgb=target_rgb,
                scheduler=scheduler,
                vae=vae,
                vae_device=None,
            )
            del target_rgb
        else:
            # Efficient MSE-Only pass
            total_loss = F.mse_loss(noise_pred.float(), noise.float())
            log_dict = {
                "Llatent": total_loss.item(),
                "phase": "MSE-only",
                "t": t_val,
                "total_loss": total_loss.item(),
            }

        # -- F. Backpropagation --
        if torch.isnan(total_loss):
            tqdm.write(f"[WARN] NaN loss detected at iteration {iteration}! Skipping backward pass.")
        else:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        