"""
training/train.py

Main training loop for Talking_Heads — optimized for T4 12GB Colab.

Memory optimizations applied:
  - fp16 mixed precision (T4 native, NOT bf16 which T4 emulates slowly)
  - batch_size=1, clip_frames=6
  - VAE offloaded to CPU, moved to GPU only for PhD auxiliary loss decode
  - Audio tokens per frame reduced 4→1 (4× less cross-attention memory)
  - PhD auxiliary loss computed only 25% of iterations (saves ~4 GB/iter)
  - Aggressive intermediate tensor cleanup
  - Gradient checkpointing on backbone

APDH Stage Schedule:
  Stage 1 :     0 – 10k iters  full pose,  audio frozen
  Stage 2 : 10k – 20k iters  no lips,    audio → lips
  Stage 3 : 20k – 30k iters  no head,    audio → face
  Stage 4 : 30k – 40k iters  hands only, audio → global

Run from Src/:
  python training/train.py --config training/config.yaml
"""

from __future__ import annotations

import gc
import os
import sys
import inspect

# Disable Python bytecode caching to prevent Google Drive __pycache__ corruption
sys.dont_write_bytecode = True

import time
import argparse
import yaml
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from diffusers import AutoencoderKLCogVideoX, DDPMScheduler
from transformers import Wav2Vec2Model
from peft import LoraConfig, get_peft_model
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from Models.talking_heads_dit import TalkingHeadsDiT
from Models.audio_encoder import AudioEncoder
from training.dataset import get_dataloader
from training.loss import PhDLoss


# ── APDH Stage definitions ─────────────────────────────────────────────────

APDH_SCHEDULE = [
    # (start_iter, end_iter, stage, iterative_prob, unfreeze_n_layers)
    # CogVideoX-2B has 30 transformer blocks
    (0, 250, 1, 0.00, 0),
    (250, 500, 2, 0.05, 4),
    (500, 1000, 3, 0.10, 8),
    (1000, 1500, 4, 0.20, 15),
    (1500, 999999, 4, 0.20, 30),  # full fine-tune (all 30 blocks)
]


def get_apdh_stage(iteration: int) -> tuple[int, float, int]:
    """Return (stage, iterative_prob, n_layers_to_unfreeze) for current iteration."""
    for start, end, stage, prob, layers in APDH_SCHEDULE:
        if start <= iteration < end:
            return stage, prob, layers
    return 4, 0.20, 30


# ── LoRA setup for Wav2Vec2 ────────────────────────────────────────────────

def build_audio_encoder_with_lora(model_name: str = "facebook/wav2vec2-base") -> AudioEncoder:
    """
    Build AudioEncoder with LoRA injected into last transformer block only.
    Rank=4, Alpha=8 → ~50k trainable params instead of ~7M.
    Output dim = 1920 to match CogVideoX-2B inner_dim.
    """
    encoder = AudioEncoder(
        output_dim=1920,           # CogVideoX-2B inner_dim
        load_wav2vec2=True,        # Must be True to load backbone!
        freeze_encoder=True,       # freeze everything first
        model_name=model_name,
    )

    # Apply LoRA only to last transformer block's attention projections
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # Wrap only the last transformer layer
    last_layer = encoder.wav2vec2.encoder.layers[-1]
    encoder.wav2vec2.encoder.layers[-1] = get_peft_model(last_layer, lora_config)

    print(f"[AudioEncoder] LoRA injected into last block.")
    print(f"[AudioEncoder] Trainable params: "
          f"{sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    return encoder


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(
    iteration   : int,
    dit         : TalkingHeadsDiT,
    audio_enc   : AudioEncoder,
    optimizer   : torch.optim.Optimizer,
    scaler      : GradScaler,
    output_dir  : str,
):
    ckpt_dir = Path(output_dir) / f"checkpoint_{iteration:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save only trainable params — not the full frozen backbone
    torch.save({
        "iteration"         : iteration,
        "dit_trainable"     : {k: v for k, v in dit.state_dict().items()
                               if dit.state_dict()[k].requires_grad
                               or k.startswith("audio_proj")
                               or k.startswith("pose_encoder")
                               or k.startswith("pose_scale")},
        "audio_enc_lora"    : audio_enc.state_dict(),
        "optimizer"         : optimizer.state_dict(),
        "scaler"            : scaler.state_dict(),
    }, ckpt_dir / "checkpoint.pt")

    print(f"[CKPT] Saved → {ckpt_dir}")


def load_checkpoint(
    ckpt_path   : str,
    dit         : TalkingHeadsDiT,
    audio_enc   : AudioEncoder,
    optimizer   : torch.optim.Optimizer,
    scaler      : GradScaler,
) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dit.load_state_dict(ckpt["dit_trainable"], strict=False)
    audio_enc.load_state_dict(ckpt["audio_enc_lora"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    iteration = ckpt["iteration"]
    print(f"[CKPT] Resumed from iteration {iteration}")
    return iteration


# ── VAE decode helper with GPU offload ─────────────────────────────────────

def vae_decode_frame(
    latent: torch.Tensor,
    vae: AutoencoderKLCogVideoX,
    device: torch.device,
) -> torch.Tensor:
    """
    Decode a single frame latent to RGB, moving VAE to GPU temporarily.

    Args:
        latent : (B, 16, 1, H, W) — single frame latent
        vae    : frozen VAE on CPU
        device : target GPU device

    Returns:
        rgb : (B, 3, H*8, W*8) — decoded RGB in [-1, 1]
    """
    vae.to(device)
    with torch.no_grad():
        rgb = vae.decode(latent.to(device).float()).sample
        if rgb.dim() == 5:
            rgb = rgb.squeeze(2)
        rgb = rgb.clamp(-1.0, 1.0)
    vae.to("cpu")
    torch.cuda.empty_cache()
    return rgb


# ── Main training function ─────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # T4 natively supports fp16 but NOT bf16 (bf16 is emulated = slow + broken)
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[TRAIN] Device: {device} | dtype: {dtype}")
    if device.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[TRAIN] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\n")

    # ── 1. Models ──────────────────────────────────────────────────────────
    print("[TRAIN] Loading TalkingHeadsDiT (CogVideoX-2B)...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        cfg["pretrained_model"],
        freeze_backbone        = True,
        gradient_checkpointing = True,
        audio_input_dim        = 1920,
        audio_tokens_per_frame = 1,       # reduced from 4 — saves 4× attention memory
    ).to(device, dtype=dtype)
    print(dit.param_summary())

    # Force cleanup after model loading
    gc.collect()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        print(f"[MEM] After DiT load: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

    print("\n[TRAIN] Loading AudioEncoder with LoRA...")
    audio_enc = build_audio_encoder_with_lora(cfg["wav2vec2_model"]).to(device)

    # ── VAE: stays on CPU, moved to GPU only for PhD aux loss decode ──────
    print("\n[TRAIN] Loading VAE (frozen, CPU-offloaded to save ~0.8 GB VRAM)...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        cfg["pretrained_model"], subfolder="vae", torch_dtype=torch.float32
    )  # intentionally NOT .to(device) — stays on CPU
    vae.requires_grad_(False)
    vae.eval()

    print("\n[TRAIN] Loading scheduler...")
    scheduler = DDPMScheduler.from_pretrained(
        cfg["pretrained_model"], subfolder="scheduler"
    )

    # ── 2. Loss ────────────────────────────────────────────────────────────
    loss_fn = PhDLoss(
        lambda_pose   = cfg.get("lambda_pose",   0.1),
        lambda_detail = cfg.get("lambda_detail", 0.1),
        lambda_low    = cfg.get("lambda_low",    0.1),
    ).to(device)

    # ── 3. Optimizer ───────────────────────────────────────────────────────
    trainable_params = (
        [p for p in dit.parameters()       if p.requires_grad] +
        [p for p in audio_enc.parameters() if p.requires_grad]
    )
    
    # CRITICAL FIX: Force trainable parameters to float32 so GradScaler doesn't crash
    for p in trainable_params:
        p.data = p.data.to(torch.float32)

    print(f"\n[TRAIN] Total trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = cfg.get("lr", 1e-5),
        betas        = (0.9, 0.999),
        weight_decay = 1e-2,
    )
    # GradScaler for fp16 — essential for stable training on T4
    scaler = GradScaler(enabled=(dtype == torch.float16))

    # ── 4. DataLoader ──────────────────────────────────────────────────────
    dataset, loader = get_dataloader(
        root_dir    = cfg["data_root"],
        batch_size  = cfg.get("batch_size", 1),
        clip_frames = cfg.get("clip_frames", 6),
        num_workers = cfg.get("num_workers", 2),
        apdh_stage  = 1,
    )

    # ── 5. Resume ──────────────────────────────────────────────────────────
    start_iter = 0
    if cfg.get("resume_from"):
        start_iter = load_checkpoint(
            cfg["resume_from"], dit, audio_enc, optimizer, scaler
        )

    # ── 6. Memory optimization config ─────────────────────────────────────
    aux_loss_prob  = cfg.get("aux_loss_prob", 0.25)      # compute PhD aux 25% of iters
    warmup_no_aux  = cfg.get("warmup_no_aux", 500)       # first N iters: MSE only

    # ── 7. Training loop ───────────────────────────────────────────────────
    total_iters    = cfg.get("total_iters", 40000)
    save_every     = cfg.get("save_every",  500)
    log_every      = cfg.get("log_every",   25)
    output_dir     = cfg.get("output_dir",  "checkpoints")
    current_stage  = -1

    dit.train()
    audio_enc.train()

    data_iter = iter(loader)
    t0 = time.time()

    if device.type == "cuda":
        print(f"\n[MEM] Before training loop: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        print(f"[MEM] Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")
        torch.cuda.reset_peak_memory_stats()

    for iteration in range(start_iter, total_iters):

        # ── APDH curriculum check ──────────────────────────────────────────
        stage, iterative_prob, n_layers = get_apdh_stage(iteration)
        if stage != current_stage:
            current_stage = stage
            dataset.set_apdh_stage(stage, iterative_prob)
            dit.unfreeze_backbone_top_n_layers(n_layers)
            # Rebuild optimizer with newly unfrozen params
            trainable_params = (
                [p for p in dit.parameters()       if p.requires_grad] +
                [p for p in audio_enc.parameters() if p.requires_grad]
            )
            
            # CRITICAL FIX: Force new trainable parameters to float32
            for p in trainable_params:
                p.data = p.data.to(torch.float32)
                
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=cfg.get("lr", 1e-5),
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )
            print(f"\n[APDH] → Stage {stage} | "
                  f"trainable: {sum(p.numel() for p in trainable_params):,}")

        # ── Get batch ─────────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            torch.cuda.empty_cache()
            data_iter = iter(loader)
            batch = next(data_iter)

        video_latents  = batch["video_latents"].to(device, dtype=dtype)   # (B,16,T,96,96)
        ref_latents    = batch["ref_latents"].to(device, dtype=dtype)     # (B,16,1,96,96)
        pose_keypoints = batch["pose_keypoints"].to(device, dtype=dtype)  # (B,T,133,2)
        audio_waveform = batch["audio_waveform"].to(device)               # (B,T_audio)
        del batch  # free CPU memory

        # ── Audio encoding (no_grad + detach to free Wav2Vec2 graph) ──────
        T = video_latents.shape[2]
        with torch.no_grad():
            # Run Wav2Vec2 in FP32 (it frequently produces NaNs in FP16 autocast)
            audio_embeds = audio_enc(audio_waveform.float(), target_frames=T).detach()
            # Cast the embeddings to match the DiT's dtype (FP16)
            audio_embeds = audio_embeds.to(dtype)
        del audio_waveform  # free immediately

        # ── Sample timestep + add noise ───────────────────────────────────
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        t_val = int(t.item())
        noise        = torch.randn_like(video_latents)
        noisy_latent = scheduler.add_noise(
            video_latents.float(), noise.float(), t
        ).to(dtype)

        # ── Forward pass ──────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype):
            noise_pred = dit(
                video_latents  = noisy_latent,
                ref_latents    = ref_latents,
                timestep       = t.expand(video_latents.shape[0]),
                audio_embeds   = audio_embeds,
                pose_keypoints = pose_keypoints,
            )
        del ref_latents, pose_keypoints, audio_embeds  # free before loss

        # ── Loss computation ──────────────────────────────────────────────
        # Decide whether to compute expensive PhD auxiliary loss this iter
        use_aux = (
            iteration >= warmup_no_aux and
            random.random() < aux_loss_prob
        )

        if use_aux:
            # Full PhD loss — requires VAE decode (temporarily move VAE to GPU)
            mid = T // 2
            z_target = video_latents[:, :, mid:mid+1, :, :].detach()
            target_rgb = vae_decode_frame(z_target, vae, device)  # moves VAE to GPU and back
            del z_target

            total_loss, log_dict = loss_fn(
                noise_pred   = noise_pred.float(),
                actual_noise = noise.float(),
                noisy_latent = noisy_latent.float(),
                timestep     = t,
                target_rgb   = target_rgb,
                scheduler    = scheduler,
                vae          = vae,
                vae_device   = device,
            )
            del target_rgb
        else:
            # MSE-only loss — no VAE decode, saves ~4 GB
            total_loss = F.mse_loss(noise_pred.float(), noise.float())
            log_dict = {
                "Llatent": total_loss.item(),
                "phase": "MSE-only",
                "t": t_val,
                "total_loss": total_loss.item(),
            }

        # ── Backprop ──────────────────────────────────────────────────────
        if torch.isnan(total_loss):
            print(f"[WARN] NaN loss at iter {iteration}! Skipping backward pass to protect model weights.")
        else:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # ── Free all large intermediates immediately after backward ────────
        del noise_pred, noisy_latent, noise, video_latents, total_loss
        torch.cuda.empty_cache()

        # ── Logging ───────────────────────────────────────────────────────
        if iteration % log_every == 0:
            elapsed = time.time() - t0
            iters_per_sec = log_every / max(elapsed, 1e-6)
            t0 = time.time()
            mem_gb = torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else 0.0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(
                f"[{iteration:06d}] "
                f"stage={stage} | "
                f"phase={log_dict['phase']} | "
                f"t={log_dict['t']:4d} | "
                f"loss={log_dict['total_loss']:.4f} | "
                f"Llatent={log_dict['Llatent']:.4f} | "
                f"mem={mem_gb:.1f}GB peak={peak_gb:.1f}GB | "
                f"{iters_per_sec:.2f} it/s"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if iteration % save_every == 0 and iteration > 0:
            save_checkpoint(iteration, dit, audio_enc, optimizer, scaler, output_dir)
            torch.cuda.empty_cache()

    print("\n[TRAIN] Done.")
    save_checkpoint(total_iters, dit, audio_enc, optimizer, scaler, output_dir)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)