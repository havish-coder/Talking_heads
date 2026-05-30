"""
smoke_cpu.py — CPU integration smoke of ONE training step.

Pins device=cpu and wires together the REAL pieces that train.py uses, with a
TINY random-init transformer + a dummy VAE so nothing large is downloaded:
  * ConditionedCogVideoXTransformer (tiny) + audio cross-attention
  * CogVideoXDDIMScheduler (v_prediction) -> add_noise / get_velocity
  * PhDLoss across all three phases (S1 sobel, S2 sobel, S3 LPIPS)
  * the real APDH curriculum helpers from training.train (audio muting)
  * forward -> loss -> backward -> optimizer.step

Goal: catch wiring/shape/NaN bugs in the training step before spending GPU time.
Run:  python Src/smoke_cpu.py
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # belt-and-suspenders
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import CogVideoXDDIMScheduler

from Models.talking_heads_dit import ConditionedCogVideoXTransformer, TalkingHeadsDiT
from training.loss import PhDLoss
from training.train import get_apdh_stage, APDH_SCHEDULE, AUDIO_ON_FROM_STAGE, get_target_rgb

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Dummy VAE (decode 16ch latent -> 3ch RGB @ 8x), no download ──────────────
class _Dec:
    def __init__(self, s): self.sample = s

class DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(16, 3, 1)
        self.config = type("c", (), {"scaling_factor": 0.7})()
    def decode(self, z):
        x = self.proj(z)                              # (B,3,T,h,w)
        B, C, T, h, w = x.shape
        x = F.interpolate(x, size=(T, h * 8, w * 8), mode="trilinear", align_corners=False)
        return _Dec(x.clamp(-1, 1))


# ── tiny model ───────────────────────────────────────────────────────────────
cfg = dict(
    num_attention_heads=2, attention_head_dim=8, in_channels=32, out_channels=16,
    text_embed_dim=32, num_layers=2, time_embed_dim=32, patch_size=2,
    sample_height=60, sample_width=90, sample_frames=49, max_text_seq_length=226,
    temporal_compression_ratio=4, use_rotary_positional_embeddings=False,
    use_learned_positional_embeddings=False, spatial_interpolation_scale=1.875,
    temporal_interpolation_scale=1.0,
)
AUDIO_DIM = 768
backbone = ConditionedCogVideoXTransformer.from_config(cfg)
backbone.add_audio_conditioning(audio_dim=AUDIO_DIM, audio_layers=[0, 1], audio_window=2)
dit = TalkingHeadsDiT(backbone, inner_dim=16, text_embed_dim=32, latent_channels=16,
                      use_rotary=False, audio_input_dim=AUDIO_DIM, audio_seq_len=226).to(DEVICE)
print(dit.param_summary())

vae = DummyVAE().to(DEVICE).eval()
vae.requires_grad_(False)
scaling_factor = float(vae.config.scaling_factor)

scheduler = CogVideoXDDIMScheduler(num_train_timesteps=1000, prediction_type="v_prediction")
prediction_type = scheduler.config.prediction_type
print(f"scheduler prediction_type = {prediction_type}")

print("[smoke] building PhDLoss (downloads small LPIPS-vgg weights once)...")
loss_fn = PhDLoss(0.1, 0.1, 0.1).to(DEVICE)

opt = torch.optim.AdamW([p for p in dit.parameters() if p.requires_grad], lr=1e-4)

# ── synthetic batch (shapes mirror the real dataset) ─────────────────────────
B, T, H, W = 1, 5, 16, 16
n_video = (T - 1) * 4 + 1
latents_raw = torch.randn(B, 16, T, H, W, device=DEVICE)
ref_raw     = torch.randn(B, 16, 1, H, W, device=DEVICE)
pose_kp     = torch.rand(B, T, 133, 2, device=DEVICE)
audio_embeds_full = torch.randn(B, n_video, AUDIO_DIM, device=DEVICE)


def one_step(t_val: int, audio_active: bool):
    """Run a full train step at a fixed timestep; return loss + log."""
    latents = latents_raw * scaling_factor
    ref = ref_raw * scaling_factor
    audio_embeds = audio_embeds_full if audio_active else None

    t_b = torch.full((B,), t_val, device=DEVICE, dtype=torch.long)
    noise = torch.randn_like(latents)
    noisy = scheduler.add_noise(latents, noise, t_b)
    target = scheduler.get_velocity(latents, noise, t_b)
    target_rgb = get_target_rgb(latents_raw, vae)

    opt.zero_grad()
    model_pred = dit(video_latents=noisy, ref_latents=ref, timestep=t_b,
                     audio_embeds=audio_embeds, pose_keypoints=pose_kp)
    total_loss, log = loss_fn(
        model_pred=model_pred.float(), target=target.float(), noisy_latent=noisy.float(),
        timestep=t_b, target_rgb=target_rgb, scheduler=scheduler, vae=vae,
        prediction_type=prediction_type, scaling_factor=scaling_factor)
    total_loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_([p for p in dit.parameters() if p.requires_grad], 1.0)
    opt.step()
    assert torch.isfinite(total_loss), f"NaN/Inf loss at t={t_val}"
    return total_loss.item(), gnorm.item(), log


print("\n[smoke] curriculum check against the ACTUAL config.yaml schedule:")
import yaml
with open(os.path.join(os.path.dirname(__file__), "training", "config.yaml")) as f:
    _cfg = yaml.safe_load(f)
sched = [tuple(x) for x in _cfg.get("apdh_schedule", APDH_SCHEDULE)]
audio_from = _cfg.get("audio_on_from_stage", AUDIO_ON_FROM_STAGE)
print(f"  schedule={sched} | audio_on_from_stage={audio_from}")
saw_audio_on = False
for it in [0, 2499, 2500, 5000, 7500, 9999]:
    stage, prob = get_apdh_stage(it, sched)
    active = stage >= audio_from
    saw_audio_on = saw_audio_on or active
    print(f"  iter {it:5d} -> stage {stage} prob {prob} | audio_active={active}")
assert get_apdh_stage(2499, sched)[0] == 1, "expected pose-only before 2500"
assert get_apdh_stage(2500, sched)[0] >= audio_from, "audio should engage at iter 2500"
assert saw_audio_on, "audio never engages in the configured schedule!"
print("  [OK] audio is muted before iter 2500 and engages from 2500.")

print("\n[smoke] training steps across all PhD phases:")
# S1 pose-dominant (t>=700), S2 detail (100<=t<700), S3 quality (t<100)
for t_val, audio_active, tag in [(900, False, "S1/stage1-audio-muted"),
                                 (900, True,  "S1/audio-on"),
                                 (400, True,  "S2/detail"),
                                 (50,  True,  "S3/quality-LPIPS")]:
    l, g, log = one_step(t_val, audio_active)
    print(f"  [{tag:24s}] t={log['t']:4d} phase={log['phase']} "
          f"loss={l:.4f} Llatent={log['Llatent']:.4f} gradnorm={g:.3f}")

# A few repeated steps should not blow up; loss should stay finite.
print("\n[smoke] 5 consecutive steps (stability):")
for i in range(5):
    l, g, log = one_step(torch.randint(0, 1000, (1,)).item(), audio_active=True)
    print(f"  step {i}: phase={log['phase']} loss={l:.4f} gradnorm={g:.3f}")

print("\nAll CPU integration smoke checks passed.")
