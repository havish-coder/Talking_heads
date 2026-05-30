"""
smoke_gpu.py — GPU smoke of the REAL stack, one training step.

Loads the actual CogVideoX-2B transformer + VAE + scheduler + Wav2Vec2 encoder
(downloaded/cached on first run) and runs a few real training steps on SYNTHETIC
60x60 latents. This isolates "does the real model build and step on GPU" from
"is the re-encoded data correct" — run it BEFORE the encode/overfit steps.

Requires a big GPU (A100). Will OOM on a 4 GB laptop card — Lightning only.

Run (from Src/):  python smoke_gpu.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler

from Models.talking_heads_dit import TalkingHeadsDiT
from training.train import build_audio_encoder_with_lora, get_target_rgb
from training.loss import PhDLoss

MODEL = "THUDM/CogVideoX-2b"
assert torch.cuda.is_available(), "No CUDA device — this smoke is GPU-only."
device = torch.device("cuda")
dtype = torch.bfloat16
print(f"[smoke-gpu] device={device} dtype={dtype}")

# ── real model ───────────────────────────────────────────────────────────────
print("[smoke-gpu] building TalkingHeadsDiT (downloads CogVideoX-2B once)...")
dit = TalkingHeadsDiT.from_pretrained_cogvideox(
    MODEL, freeze_backbone=True, gradient_checkpointing=True,
    use_lora=True, lora_rank=64, lora_alpha=64, audio_input_dim=768,
).to(device, dtype=dtype)
print(dit.param_summary())

print("[smoke-gpu] loading VAE + scheduler + audio encoder...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to(device)
vae.requires_grad_(False); vae.eval()
scaling_factor = float(vae.config.scaling_factor)
scheduler = CogVideoXDDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
prediction_type = scheduler.config.prediction_type
num_train_timesteps = scheduler.config.num_train_timesteps
print(f"[smoke-gpu] prediction_type={prediction_type} scaling_factor={scaling_factor}")

audio_enc = build_audio_encoder_with_lora("facebook/wav2vec2-base", 768).to(device)
loss_fn = PhDLoss(0.1, 0.1, 0.1).to(device)

trainable = [p for p in dit.parameters() if p.requires_grad] + \
            [p for p in audio_enc.parameters() if p.requires_grad]
print(f"[smoke-gpu] total trainable: {sum(p.numel() for p in trainable):,}")
opt = torch.optim.AdamW(trainable, lr=1e-4)

# ── synthetic native-resolution batch (mirrors re-encoded data shapes) ───────
B, T, H, W = 1, 13, 60, 60               # 13 latent frames -> 49 video frames
n_video = (T - 1) * 4 + 1
sr = 16000
latents_raw = torch.randn(B, 16, T, H, W, device=device)
ref_raw     = torch.randn(B, 16, 1, H, W, device=device)
pose_kp     = torch.rand(B, T, 133, 2, device=device, dtype=dtype)
audio_wave  = torch.randn(B, int(n_video / 24 * sr), device=device)   # ~2s @ 16kHz

dit.train(); audio_enc.train()


def one_step(audio_active: bool):
    latents = latents_raw * scaling_factor
    ref = ref_raw * scaling_factor
    if audio_active:
        with torch.autocast("cuda", dtype=dtype):
            audio_embeds = audio_enc(audio_wave.float(), target_frames=n_video)
    else:
        audio_embeds = None

    t_b = torch.randint(0, num_train_timesteps, (1,), device=device).expand(B)
    noise = torch.randn_like(latents)
    noisy = scheduler.add_noise(latents, noise, t_b)
    target = scheduler.get_velocity(latents, noise, t_b) if prediction_type == "v_prediction" else noise
    target_rgb = get_target_rgb(latents_raw, vae)

    opt.zero_grad()
    with torch.autocast("cuda", dtype=dtype):
        pred = dit(video_latents=noisy.to(dtype), ref_latents=ref.to(dtype), timestep=t_b,
                   audio_embeds=audio_embeds, pose_keypoints=pose_kp)
    total_loss, log = loss_fn(
        model_pred=pred.float(), target=target.float(), noisy_latent=noisy.float(),
        timestep=t_b, target_rgb=target_rgb, scheduler=scheduler, vae=vae,
        prediction_type=prediction_type, scaling_factor=scaling_factor)
    total_loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    assert torch.isfinite(total_loss), "NaN/Inf loss!"
    return total_loss.item(), gnorm.item(), log


print("\n[smoke-gpu] running real training steps...")
for i, audio_active in enumerate([False, True, True, True]):
    t0 = time.time()
    l, g, log = one_step(audio_active)
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  step {i} audio={audio_active} phase={log['phase']} t={log['t']:4d} "
          f"loss={l:.4f} grad={g:.3f} {time.time()-t0:.1f}s peakmem={mem:.1f}GB")

print("\nGPU smoke passed: real CogVideoX-2B stack builds, steps, and backprops.")
