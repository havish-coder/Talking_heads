"""
Offline smoke test for TalkingHeadsDiT (v2: frame-aligned audio cross-attention).

Builds a TINY CogVideoX backbone (no download) with the CogVideoX-2B family
settings (sinusoidal positional embeddings, no rotary) and checks:
  * forward-pass shapes for pose heatmaps and the full model
  * that audio conditioning actually changes the output once its gate is open
  * that gradients flow into the audio cross-attention adapter

This does NOT load pretrained weights; use train.py for the real model.

Run:  python Src/test_dit.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from Models.talking_heads_dit import (
    ConditionedCogVideoXTransformer, TalkingHeadsDiT, render_pose_heatmaps,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

cfg = dict(
    num_attention_heads=2, attention_head_dim=8, in_channels=32, out_channels=16,
    text_embed_dim=32, num_layers=2, time_embed_dim=32, patch_size=2,
    sample_height=60, sample_width=90, sample_frames=49, max_text_seq_length=226,
    temporal_compression_ratio=4, use_rotary_positional_embeddings=False,
    use_learned_positional_embeddings=False, spatial_interpolation_scale=1.875,
    temporal_interpolation_scale=1.0,
)
AUDIO_DIM = 24
backbone = ConditionedCogVideoXTransformer.from_config(cfg)
backbone.add_audio_conditioning(audio_dim=AUDIO_DIM, audio_layers=[0, 1], audio_window=2)
backbone = backbone.to(device)

model = TalkingHeadsDiT(
    backbone, inner_dim=16, text_embed_dim=32, latent_channels=16,
    use_rotary=False, audio_input_dim=AUDIO_DIM, audio_seq_len=226,
).to(device)
print(model.param_summary())

B, T, H, W = 1, 5, 12, 12
vid = torch.randn(B, 16, T, H, W, device=device)
ref = torch.randn(B, 16, 1, H, W, device=device)
ts  = torch.randint(0, 1000, (B,), device=device)
# audio fed at video-frame resolution; model interpolates to T latent frames
aud = torch.randn(B, (T - 1) * 4 + 1, AUDIO_DIM, device=device)
kp  = torch.rand(B, T, 133, 2, device=device)

hm = render_pose_heatmaps(kp, H, W)
assert hm.shape == (B, 3, T, H, W)
print(f"[PASS] pose heatmap     : {tuple(kp.shape)} -> {tuple(hm.shape)}")

out = model(vid, ref, ts, aud, kp)
assert out.shape == (B, 16, T, H, W), out.shape
print(f"[PASS] forward (pose)   : noise_pred {tuple(out.shape)}")

out2 = model(vid, ref, ts, aud, None)
assert out2.shape == (B, 16, T, H, W)
print(f"[PASS] forward (no pose): {tuple(out2.shape)}")

# Audio gate is zero-init -> audio must NOT change the output yet.
with torch.no_grad():
    base = model(vid, ref, ts, aud, None)
    aud2 = torch.randn_like(aud)
    same = model(vid, ref, ts, aud2, None)
assert torch.allclose(base, same, atol=1e-5), "audio leaked through a closed gate!"
print("[PASS] zero-init gate   : audio has no effect at init (as intended)")

# Open the gates -> audio MUST now change the output.
with torch.no_grad():
    for ad in backbone.audio_adapters.values():
        ad.gate.fill_(1.0)
        torch.nn.init.normal_(ad.audio_out.weight, std=0.1)
    diff_a = model(vid, ref, ts, aud, None)
    diff_b = model(vid, ref, ts, aud2, None)
assert not torch.allclose(diff_a, diff_b, atol=1e-5), "audio still has no effect!"
print("[PASS] open gate        : different audio -> different output (lip-sync path live)")

# Gradients flow into the audio adapter.
model.zero_grad()
loss = model(vid, ref, ts, aud, kp).pow(2).mean()
loss.backward()
g = backbone.audio_adapters["0"].audio_q.weight.grad
assert g is not None and g.abs().sum() > 0, "no gradient into audio adapter!"
print("[PASS] backward         : gradient reaches audio cross-attention")

print("\nAll offline tests passed.")
