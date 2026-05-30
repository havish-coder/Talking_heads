"""
training/train.py

Training loop for Talking_Heads on a CogVideoX-2B backbone.

  1. TalkingHeadsDiT — CogVideoX-2B backbone, frozen + LoRA + trainable
     conditioning (reference patch-embed channels, audio projection, pose guider)
  2. Wav2Vec2 audio encoder with LoRA on its last block
  3. Frozen CogVideoX-2B VAE (decode only, for the PhD-loss aux terms)
  4. DataLoader over the re-encoded latents / pose / audio
  5. APDH pose-dropout curriculum (spatial pose dropout schedule)
  6. PhD Loss (v-prediction aware)

Run from Src/:
  python training/train.py --config training/config.yaml
"""

from __future__ import annotations

import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from Models.talking_heads_dit import TalkingHeadsDiT
from Models.audio_encoder import AudioEncoder
from training.dataset import get_dataloader
from training.loss import PhDLoss


# (start_iter, end_iter, stage, iterative_prob) — controls POSE dropout.
APDH_SCHEDULE = [
    (0,     10000, 1, 0.00),
    (10000, 20000, 2, 0.05),
    (20000, 30000, 3, 0.10),
    (30000, 999999, 4, 0.20),
]

# Audio Diffusion curriculum (EchoMimicV2 §3.2.2): the audio cross-attention is
# muted during the Initial Pose phase (stage 1) so the model first learns motion
# from full pose, then audio is phased in as pose keypoints are dropped. We
# implement the *temporal* schedule here (audio off in stage 1, on from stage 2);
# the paper's spatial lips->head->global audio masking is a later refinement.
AUDIO_ON_FROM_STAGE = 2


def get_apdh_stage(iteration: int, schedule=APDH_SCHEDULE):
    for start, end, stage, prob in schedule:
        if start <= iteration < end:
            return stage, prob
    return schedule[-1][2], schedule[-1][3]


def build_audio_encoder_with_lora(model_name: str, output_dim: int) -> AudioEncoder:
    encoder = AudioEncoder(output_dim=output_dim, freeze_encoder=True, model_name=model_name)
    lora_config = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05, bias="none",
    )
    encoder.wav2vec2.encoder.layers[-1] = get_peft_model(
        encoder.wav2vec2.encoder.layers[-1], lora_config
    )
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[AudioEncoder] LoRA on last block. Trainable params: {trainable:,}")
    return encoder


def save_checkpoint(iteration, dit, audio_enc, optimizer, output_dir):
    ckpt_dir = Path(output_dir) / f"checkpoint_{iteration:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dit_trainable = {k: v.detach().cpu()
                     for k, v in dit.named_parameters() if v.requires_grad}
    audio_trainable = {k: v.detach().cpu()
                       for k, v in audio_enc.named_parameters() if v.requires_grad}
    torch.save({
        "iteration": iteration,
        "dit_trainable": dit_trainable,
        "audio_trainable": audio_trainable,
        "optimizer": optimizer.state_dict(),
    }, ckpt_dir / "checkpoint.pt")
    print(f"[CKPT] Saved -> {ckpt_dir}")


def load_checkpoint(ckpt_path, dit, audio_enc, optimizer):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dit.load_state_dict(ckpt["dit_trainable"], strict=False)
    audio_enc.load_state_dict(ckpt["audio_trainable"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[CKPT] Resumed from iteration {ckpt['iteration']}")
    return ckpt["iteration"]


def get_target_rgb(latents_raw: torch.Tensor, vae) -> torch.Tensor:
    """Decode the middle frame of the clean UNSCALED latent to RGB (B,3,H*8,W*8)."""
    with torch.no_grad():
        T = latents_raw.shape[2]
        mid = T // 2
        z = latents_raw[:, :, mid:mid + 1, :, :].float()
        rgb = vae.decode(z).sample.squeeze(2).clamp(-1.0, 1.0)
    return rgb


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[TRAIN] Device: {device} | dtype: {dtype}\n")

    audio_dim = cfg.get("audio_dim", 768)

    print("[TRAIN] Loading TalkingHeadsDiT (CogVideoX-2B + LoRA)...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        cfg["pretrained_model"],
        freeze_backbone=True,
        gradient_checkpointing=True,
        use_lora=cfg.get("use_lora", True),
        lora_rank=cfg.get("lora_rank", 64),
        lora_alpha=cfg.get("lora_alpha", 64),
        audio_input_dim=audio_dim,
    ).to(device, dtype=dtype)
    print(dit.param_summary())

    print("\n[TRAIN] Loading AudioEncoder with LoRA...")
    audio_enc = build_audio_encoder_with_lora(cfg["wav2vec2_model"], audio_dim).to(device)

    print("\n[TRAIN] Loading VAE (frozen)...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        cfg["pretrained_model"], subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    scaling_factor = float(vae.config.scaling_factor)
    print(f"[TRAIN] VAE scaling_factor = {scaling_factor}")

    print("\n[TRAIN] Loading scheduler...")
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        cfg["pretrained_model"], subfolder="scheduler"
    )
    prediction_type = scheduler.config.prediction_type
    num_train_timesteps = scheduler.config.num_train_timesteps
    print(f"[TRAIN] prediction_type = {prediction_type} | timesteps = {num_train_timesteps}")

    loss_fn = PhDLoss(
        lambda_pose=cfg.get("lambda_pose", 0.1),
        lambda_detail=cfg.get("lambda_detail", 0.1),
        lambda_low=cfg.get("lambda_low", 0.1),
    ).to(device)

    trainable_params = (
        [p for p in dit.parameters() if p.requires_grad] +
        [p for p in audio_enc.parameters() if p.requires_grad]
    )
    print(f"\n[TRAIN] Total trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.get("lr", 1e-5), betas=(0.9, 0.999), weight_decay=1e-2
    )

    dataset, loader = get_dataloader(
        root_dir=cfg["data_root"],
        batch_size=cfg.get("batch_size", 1),
        clip_frames=cfg.get("clip_frames", 13),
        num_workers=cfg.get("num_workers", 0),
        apdh_stage=1,
    )

    start_iter = 0
    if cfg.get("resume_from"):
        start_iter = load_checkpoint(cfg["resume_from"], dit, audio_enc, optimizer)

    total_iters = cfg.get("total_iters", 40000)
    save_every = cfg.get("save_every", 1000)
    log_every = cfg.get("log_every", 50)
    output_dir = cfg.get("output_dir", "checkpoints")
    current_stage = -1

    # APDH curriculum (override-able from config for short overfit runs)
    apdh_schedule = [tuple(x) for x in cfg.get("apdh_schedule", APDH_SCHEDULE)]
    audio_on_from_stage = cfg.get("audio_on_from_stage", AUDIO_ON_FROM_STAGE)
    print(f"[TRAIN] APDH schedule: {apdh_schedule} | audio on from stage {audio_on_from_stage}")

    dit.train()
    audio_enc.train()

    data_iter = iter(loader)
    t0 = time.time()

    for iteration in range(start_iter, total_iters):
        stage, iterative_prob = get_apdh_stage(iteration, apdh_schedule)
        if stage != current_stage:
            current_stage = stage
            dataset.set_apdh_stage(stage, iterative_prob)
            print(f"\n[APDH] -> pose-dropout stage {stage} (iterative_prob={iterative_prob})")

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        latents_raw = batch["video_latents"].to(device, dtype=torch.float32)  # (B,16,T,H,W)
        ref_raw = batch["ref_latents"].to(device, dtype=torch.float32)        # (B,16,1,H,W)
        pose_keypoints = batch["pose_keypoints"].to(device, dtype=dtype)      # (B,T,133,2)
        audio_waveform = batch["audio_waveform"].to(device)                   # (B,T_audio)

        latents = latents_raw * scaling_factor
        ref = ref_raw * scaling_factor
        T = latents.shape[2]
        # Feed audio at video-frame resolution; the DiT's audio cross-attention
        # resamples to the T latent frames and attends per-frame in a local window.
        n_video_frames = (T - 1) * 4 + 1

        # Audio Diffusion curriculum: mute audio during the Initial Pose phase.
        audio_active = stage >= audio_on_from_stage
        if audio_active:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                audio_embeds = audio_enc(audio_waveform.float(), target_frames=n_video_frames)
        else:
            audio_embeds = None

        # one timestep per batch (PhD loss is phase-per-batch)
        t = torch.randint(0, num_train_timesteps, (1,), device=device)
        t_b = t.expand(latents.shape[0])
        noise = torch.randn_like(latents)
        noisy = scheduler.add_noise(latents, noise, t_b)

        if prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, t_b)
        else:
            target = noise

        target_rgb = get_target_rgb(latents_raw, vae)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            model_pred = dit(
                video_latents=noisy.to(dtype),
                ref_latents=ref.to(dtype),
                timestep=t_b,
                audio_embeds=audio_embeds,
                pose_keypoints=pose_keypoints,
            )

        total_loss, log_dict = loss_fn(
            model_pred=model_pred.float(),
            target=target.float(),
            noisy_latent=noisy.float(),
            timestep=t_b,
            target_rgb=target_rgb,
            scheduler=scheduler,
            vae=vae,
            prediction_type=prediction_type,
            scaling_factor=scaling_factor,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if iteration % log_every == 0:
            elapsed = time.time() - t0
            its = log_every / max(elapsed, 1e-6)
            t0 = time.time()
            print(f"[{iteration:06d}] stage={stage} phase={log_dict['phase']} "
                  f"t={log_dict['t']:4d} loss={log_dict['total_loss']:.4f} "
                  f"Llatent={log_dict['Llatent']:.4f} {its:.2f} it/s")

        if iteration % save_every == 0 and iteration > 0:
            save_checkpoint(iteration, dit, audio_enc, optimizer, output_dir)

    print("\n[TRAIN] Done.")
    save_checkpoint(total_iters, dit, audio_enc, optimizer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
