import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from Models.talking_heads_dit import TalkingHeadsDiT, PoseEncoder, AudioProjection

print("Imports done")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device} | Dtype: {dtype}\n")

# ── Config ──────────────────────────────────────────────────────
B, T, H, W    = 1, 24, 96, 96
AUDIO_DIM     = 3072

# ── Test 1: PoseEncoder ─────────────────────────────────────────
pe  = PoseEncoder(inner_dim=1920).to(device, dtype=dtype)
kp  = torch.rand(B, T, 133, 2, device=device, dtype=dtype)
out = pe(kp)
assert out.shape == (B, T, 1920)
print(f"[PASS] PoseEncoder     : {tuple(kp.shape)} -> {tuple(out.shape)}")

# ── Test 2: AudioProjection ─────────────────────────────────────
ap  = AudioProjection(audio_dim=AUDIO_DIM, inner_dim=1920, n_tokens_per_frame=4).to(device, dtype=dtype)
ae  = torch.randn(B, T, AUDIO_DIM, device=device, dtype=dtype)
out = ap(ae)
assert out.shape == (B, T*4, 1920)
print(f"[PASS] AudioProjection : {tuple(ae.shape)} -> {tuple(out.shape)}")

# ── Test 3: Full model (NO pretrained weights, random init) ─────
model = TalkingHeadsDiT(
    inner_dim              = 1920,
    audio_input_dim        = AUDIO_DIM,
    audio_tokens_per_frame = 4,
    freeze_backbone        = False,   
    gradient_checkpointing = False,   
).to(device, dtype=dtype)

print(f"\n{model.param_summary()}\n")

video_latents  = torch.randn(B, 16, T, H, W, device=device, dtype=dtype)
ref_latents    = torch.randn(B, 16,  1, H, W, device=device, dtype=dtype)
timestep       = torch.randint(0, 1000, (B,), device=device)
pose_keypoints = torch.rand(B, T, 133, 2, device=device, dtype=dtype)

with torch.no_grad():
    noise_pred = model(
        video_latents  = video_latents,
        ref_latents    = ref_latents,
        timestep       = timestep,
        audio_embeds   = ae,
        pose_keypoints = pose_keypoints,
    )

assert noise_pred.shape == (B, 16, T, H, W)
print(f"[PASS] Full forward    : noise_pred {tuple(noise_pred.shape)}")

# ── Test 4: No pose (APDH stage where pose is dropped) ──────────
with torch.no_grad():
    noise_pred_no_pose = model(
        video_latents  = video_latents,
        ref_latents    = ref_latents,
        timestep       = timestep,
        audio_embeds   = ae,
        pose_keypoints = None,        
    )
assert noise_pred_no_pose.shape == (B, 16, T, H, W)
print(f"[PASS] No-pose forward : noise_pred {tuple(noise_pred_no_pose.shape)}")

# ── Test 5: Loss step ───────────────────────────────────────────
actual_noise = torch.randn_like(video_latents)
loss = F.mse_loss(noise_pred, actual_noise)
print(f"[PASS] Loss computed   : {loss.item():.4f}")

print("\n All tests passed. TalkingHeadsDiT is ready for training.")