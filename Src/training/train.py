"""
training/train.py

Main training loop for Talking_Heads.

What this file does:
  1. Loads TalkingHeadsDiT (CogVideoX-5B backbone, frozen)
  2. Loads Wav2Vec2 with LoRA on last block only
  3. Loads frozen CogVideoX VAE (decode only, for PhD Loss)
  4. Builds DataLoader from preprocessed .npy files
  5. Runs training with APDH curriculum progression
  6. Computes PhD Loss per iteration
  7. Saves checkpoints

APDH Stage Schedule:
  Stage 1 :     0 – 10k iters  full pose,  audio frozen
  Stage 2 : 10k – 20k iters  no lips,    audio → lips
  Stage 3 : 20k – 30k iters  no head,    audio → face
  Stage 4 : 30k – 40k iters  hands only, audio → global

Run from Src/:
  python training/train.py --config training/config.yaml
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
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
    (0,     10000, 1, 0.00, 0),
    (10000, 20000, 2, 0.05, 5),
    (20000, 30000, 3, 0.10, 10),
    (30000, 40000, 4, 0.20, 20),
    (40000, 999999, 4, 0.20, 42),  # full fine-tune
]


def get_apdh_stage(iteration: int) -> tuple[int, float, int]:
    """Return (stage, iterative_prob, n_layers_to_unfreeze) for current iteration."""
    for start, end, stage, prob, layers in APDH_SCHEDULE:
        if start <= iteration < end:
            return stage, prob, layers
    return 4, 0.20, 42


# ── LoRA setup for Wav2Vec2 ────────────────────────────────────────────────

def build_audio_encoder_with_lora(model_name: str = "facebook/wav2vec2-base") -> AudioEncoder:
    """
    Build AudioEncoder with LoRA injected into last transformer block only.
    Rank=4, Alpha=8 → ~50k trainable params instead of ~7M.
    """
    encoder = AudioEncoder(
        output_dim=3072,
        freeze_encoder=True,   # freeze everything first
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


# ── Target RGB helper ──────────────────────────────────────────────────────

def get_target_rgb(video_latents: torch.Tensor, vae: AutoencoderKLCogVideoX) -> torch.Tensor:
    """
    Decode middle frame of clean video latent to RGB.
    Used as target for PhD Loss auxiliary terms.
    No grad — VAE is frozen.
    """
    with torch.no_grad():
        T   = video_latents.shape[2]
        mid = T // 2
        z   = video_latents[:, :, mid:mid+1, :, :].float()
        rgb = vae.decode(z).sample.squeeze(2)   # (B, 3, H*8, W*8)
        rgb = rgb.clamp(-1.0, 1.0)
    return rgb


# ── Main training function ─────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[TRAIN] Device: {device} | dtype: {dtype}\n")

    # ── 1. Models ──────────────────────────────────────────────────────────
    print("[TRAIN] Loading TalkingHeadsDiT...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        cfg["pretrained_model"],
        freeze_backbone        = True,
        gradient_checkpointing = True,
        audio_input_dim        = 3072,
        audio_tokens_per_frame = 4,
    ).to(device, dtype=dtype)
    print(dit.param_summary())

    print("\n[TRAIN] Loading AudioEncoder with LoRA...")
    audio_enc = build_audio_encoder_with_lora(cfg["wav2vec2_model"]).to(device)

    print("\n[TRAIN] Loading VAE (frozen)...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        cfg["pretrained_model"], subfolder="vae", torch_dtype=torch.float32
    ).to(device)
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
    print(f"\n[TRAIN] Total trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = cfg.get("lr", 1e-5),
        betas        = (0.9, 0.999),
        weight_decay = 1e-2,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── 4. DataLoader ──────────────────────────────────────────────────────
    dataset, loader = get_dataloader(
        root_dir    = cfg["data_root"],
        batch_size  = cfg.get("batch_size", 2),
        clip_frames = cfg.get("clip_frames", 24),
        num_workers = cfg.get("num_workers", 0),
        apdh_stage  = 1,
    )

    # ── 5. Resume ──────────────────────────────────────────────────────────
    start_iter = 0
    if cfg.get("resume_from"):
        start_iter = load_checkpoint(
            cfg["resume_from"], dit, audio_enc, optimizer, scaler
        )

    # ── 6. Training loop ───────────────────────────────────────────────────
    total_iters    = cfg.get("total_iters", 70000)
    save_every     = cfg.get("save_every",  1000)
    log_every      = cfg.get("log_every",   50)
    output_dir     = cfg.get("output_dir",  "checkpoints")
    current_stage  = -1

    dit.train()
    audio_enc.train()

    data_iter = iter(loader)
    t0 = time.time()

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
            data_iter = iter(loader)
            batch = next(data_iter)

        video_latents  = batch["video_latents"].to(device, dtype=dtype)   # (B,16,T,96,96)
        ref_latents    = batch["ref_latents"].to(device, dtype=dtype)     # (B,16,1,96,96)
        pose_keypoints = batch["pose_keypoints"].to(device, dtype=dtype)  # (B,T,133,2)
        audio_waveform = batch["audio_waveform"].to(device)               # (B,T_audio)

        # ── Audio encoding ─────────────────────────────────────────────────
        T = video_latents.shape[2]
        with torch.autocast(device_type=device.type, dtype=dtype):
            audio_embeds = audio_enc(audio_waveform.float(), target_frames=T)
            # (B, T, 3072)

        # ── Sample timestep + add noise ───────────────────────────────────
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noise        = torch.randn_like(video_latents)
        noisy_latent = scheduler.add_noise(
            video_latents.float(), noise.float(), t
        ).to(dtype)

        # ── Target RGB for auxiliary losses ───────────────────────────────
        target_rgb = get_target_rgb(video_latents, vae)   # (B, 3, H*8, W*8)

        # ── Forward pass ──────────────────────────────────────────────────
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=dtype):
            noise_pred = dit(
                video_latents  = noisy_latent,
                ref_latents    = ref_latents,
                timestep       = t.expand(video_latents.shape[0]),
                audio_embeds   = audio_embeds,
                pose_keypoints = pose_keypoints if stage < 4 else pose_keypoints,
            )

        # ── PhD Loss ──────────────────────────────────────────────────────
        total_loss, log_dict = loss_fn(
            noise_pred   = noise_pred.float(),
            actual_noise = noise.float(),
            noisy_latent = noisy_latent.float(),
            timestep     = t,
            target_rgb   = target_rgb,
            scheduler    = scheduler,
            vae          = vae,
        )

        # ── Backprop ──────────────────────────────────────────────────────
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ───────────────────────────────────────────────────────
        if iteration % log_every == 0:
            elapsed = time.time() - t0
            iters_per_sec = log_every / max(elapsed, 1e-6)
            t0 = time.time()
            print(
                f"[{iteration:06d}] "
                f"stage={stage} | "
                f"phase={log_dict['phase']} | "
                f"t={log_dict['t']:4d} | "
                f"loss={log_dict['total_loss']:.4f} | "
                f"Llatent={log_dict['Llatent']:.4f} | "
                f"{iters_per_sec:.2f} it/s"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if iteration % save_every == 0 and iteration > 0:
            save_checkpoint(iteration, dit, audio_enc, optimizer, scaler, output_dir)

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