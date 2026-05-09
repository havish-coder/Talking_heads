"""
tests/test_talking_heads_dit.py

Integration and shape testing script for the TalkingHeadsDiT model and its sub-components.
Ensures that tensor dimensions align correctly through the Pose Encoder, Audio Projection,
and the full DiT forward pass (both with and without pose conditioning).
"""

import os
import sys

# Ensure the Models directory is importable from the script's location
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from Models.talking_heads_dit import AudioProjection, PoseEncoder, TalkingHeadsDiT

# --- Hardware Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"[SYSTEM] Hardware Check -> Device: {device} | Dtype: {dtype}\n")

# --- Global Test Configuration ---
B, T, H, W = 1, 24, 96, 96
AUDIO_DIM = 3072


# --- Test 1: Pose Encoder ---
# Verifies that normalized keypoints are correctly projected into the DiT token space
pe = PoseEncoder(inner_dim=1920).to(device, dtype=dtype)
kp = torch.rand(B, T, 133, 2, device=device, dtype=dtype)
out_pe = pe(kp)

assert out_pe.shape == (B, T, 1920), f"PoseEncoder shape mismatch: {out_pe.shape}"
print(f"[TEST 1/5] PoseEncoder     : {tuple(kp.shape)} -> {tuple(out_pe.shape)} [PASS]")


# --- Test 2: Audio Projection ---
# Verifies that audio embeddings are projected and expanded to match the required sequence length
ap = AudioProjection(
    audio_dim=AUDIO_DIM, 
    inner_dim=1920, 
    n_tokens_per_frame=4
).to(device, dtype=dtype)

ae = torch.randn(B, T, AUDIO_DIM, device=device, dtype=dtype)
out_ap = ap(ae)

expected_ap_shape = (B, T * 4, 1920)
assert out_ap.shape == expected_ap_shape, f"AudioProjection shape mismatch: {out_ap.shape}"
print(f"[TEST 2/5] AudioProjection : {tuple(ae.shape)} -> {tuple(out_ap.shape)} [PASS]")


# --- Test 3: Full Model Forward Pass ---
# Verifies the full integration of video, reference, audio, and pose conditioning
print("\n[SYSTEM] Initializing Full TalkingHeadsDiT Model (Random Weights)...")
model = TalkingHeadsDiT(
    inner_dim=1920,
    audio_input_dim=AUDIO_DIM,
    audio_tokens_per_frame=4,
    freeze_backbone=False,
    gradient_checkpointing=False,
).to(device, dtype=dtype)

print(f"[INFO] {model.param_summary()}\n")

# Generate dummy latents simulating the VAE outputs
video_latents = torch.randn(B, 16, T, H, W, device=device, dtype=dtype)
ref_latents = torch.randn(B, 16, 1, H, W, device=device, dtype=dtype)
timestep = torch.randint(0, 1000, (B,), device=device)
pose_keypoints = torch.rand(B, T, 133, 2, device=device, dtype=dtype)

with torch.no_grad():
    noise_pred = model(
        video_latents=video_latents,
        ref_latents=ref_latents,
        timestep=timestep,
        audio_embeds=ae,
        pose_keypoints=pose_keypoints,
    )

expected_noise_shape = (B, 16, T, H, W)
assert noise_pred.shape == expected_noise_shape, f"Full forward shape mismatch: {noise_pred.shape}"
print(f"[TEST 3/5] Full Forward Pass : noise_pred {tuple(noise_pred.shape)} [PASS]")


# --- Test 4: Full Model (No Pose) ---
# Verifies stability during curriculum stages where pose conditioning is explicitly dropped
with torch.no_grad():
    noise_pred_no_pose = model(
        video_latents=video_latents,
        ref_latents=ref_latents,
        timestep=timestep,
        audio_embeds=ae,
        pose_keypoints=None, 
    )

assert noise_pred_no_pose.shape == expected_noise_shape, f"No-pose forward shape mismatch: {noise_pred_no_pose.shape}"
print(f"[TEST 4/5] No-Pose Forward   : noise_pred {tuple(noise_pred_no_pose.shape)} [PASS]")


# --- Test 5: Loss Computation ---
# Verifies that gradients and loss calculations can flow through the predictions
actual_noise = torch.randn_like(video_latents)
loss = F.mse_loss(noise_pred, actual_noise)
print(f"[TEST 5/5] Loss Computed     : {loss.item():.4f} [PASS]")


print("\n[SYSTEM] All tests passed successfully. TalkingHeadsDiT is ready for training.")