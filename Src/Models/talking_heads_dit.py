"""
models/talking_heads_dit.py

TalkingHeadsDiT — CogVideoX-5B Transformer with injected conditioning.

This is the core model for the Talking_Heads project. It wraps the
pretrained CogVideoX-5B Diffusion Transformer and surgically injects
three conditioning pathways without altering any pretrained weights:

  1. Audio Cross-Attention  — per-block, learnable, frozen-then-unfrozen
     following the APDH curriculum in training.
  2. Pose Conditioning      — lightweight MLP projects DWPose keypoints
     into token space, added to video hidden states before the block loop.
  3. Reference Frame        — reference image latent is concatenated to the
     noisy video latent on the channel dimension BEFORE patch embedding,
     matching the CogVideoX-I2V strategy (zero-cost, no extra parameters).

Architecture facts from source inspection (CogVideoX-5B):
  inner_dim  = num_attention_heads × attention_head_dim = 30 × 64 = 1920
  num_layers = 42
  in_channels = 16   (CogVideoX VAE latent channels)
  text_embed_dim = 4096  (T5 text encoder dim — we repurpose this slot
                          as our audio embedding slot, same dim via projection)

The "encoder_hidden_states" slot in every CogVideoXBlock is normally used
for text (T5). We inject audio embeddings here instead, keeping the full
joint attention mechanism (audio tokens attend to video tokens and vice
versa) which is strictly better than one-directional cross-attention.

Data contracts (from your preprocessing scripts):
  video_latents  : (B, 16, T,  96, 96)   — CogVideoX VAE encoded
  ref_latents    : (B, 16, 1,  96, 96)   — same VAE, reference frame
  pose_keypoints : (B, T, 133, 2)        — DWPose wholebody, pixel coords
                                            normalised to [0, 1] before input
  audio_embeds   : (B, T, 1920)          — from AudioEncoder, projected to
                                            inner_dim

CogVideoX forward expects:
  hidden_states         : (B, T, C, H, W)   <-- note T before C
  encoder_hidden_states : (B, seq_len, inner_dim)
  timestep              : (B,)

NOTE on in_channels doubling:
  Reference conditioning concatenates ref_latent on channel dim → 32ch input.
  We adjust patch_embed.proj to accept 32 channels by initialising the extra
  16 channels to zero (standard zero-init trick from ControlNet/InstructPix2Pix).
  The pretrained 16-channel weights are preserved exactly.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel,
)


# ---------------------------------------------------------------------------
# 1.  Pose Encoder  (self-contained, no file dependency)
# ---------------------------------------------------------------------------

class PoseEncoder(nn.Module):
    """
    Projects per-frame DWPose keypoint coordinates into the DiT token space.

    Input  : (B, T, 133, 2)  — normalised [0,1] xy coords
    Output : (B, T, inner_dim) — one conditioning token per video frame,
             broadcast-added to all spatial tokens of that frame.

    Design:
      - Flatten 133×2 → 266-dim per frame
      - 3-layer MLP with SiLU activations and LayerNorm
      - No temporal attention here — the DiT's 3D attention already handles
        temporal context. Keeping pose encoder stateless per-frame is
        intentional: simpler, fewer params, less overfit risk.
    """

    def __init__(self, inner_dim: int = 1920, dropout: float = 0.1):
        super().__init__()
        in_dim = 133 * 2   # 266

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, inner_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(inner_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(inner_dim // 2, inner_dim),
            nn.SiLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim),
        )
        self._init_weights()

    def _init_weights(self):
        # Zero-init the final layer so the model starts training
        # with zero pose influence, warming up gracefully.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        keypoints : (B, T, 133, 2) — normalised to [0,1]
        returns   : (B, T, inner_dim)
        """
        B, T, K, D = keypoints.shape
        x = keypoints.reshape(B, T, K * D)   # (B, T, 266)
        return self.mlp(x)                    # (B, T, inner_dim)


# ---------------------------------------------------------------------------
# 2.  Audio Projection  (maps AudioEncoder output to inner_dim)
# ---------------------------------------------------------------------------

class AudioProjection(nn.Module):
    """
    Two-layer MLP that projects audio embeddings from AudioEncoder's
    output_dim (3072 default) → CogVideoX inner_dim (1920).

    Also expands the sequence: audio has T_frames tokens, but for richer
    cross-attention we tile to N_audio_tokens_per_frame × T_frames tokens.
    Default tiling = 1 (no tiling). Increase if you want the model to attend
    to finer-grained audio context per frame.
    """

    def __init__(
        self,
        audio_dim: int = 3072,
        inner_dim: int = 1920,
        n_tokens_per_frame: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens_per_frame

        self.proj = nn.Sequential(
            nn.Linear(audio_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, inner_dim * n_tokens_per_frame),
        )
        self.norm = nn.LayerNorm(inner_dim)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """
        audio_embeds : (B, T_frames, audio_dim)
        returns      : (B, T_frames × n_tokens_per_frame, inner_dim)
                       Ready to be used as encoder_hidden_states.
        """
        B, T, _ = audio_embeds.shape
        # (B, T, inner_dim * n_tokens)
        x = self.proj(audio_embeds)
        # (B, T, n_tokens, inner_dim)
        x = x.reshape(B, T, self.n_tokens, -1)
        # (B, T*n_tokens, inner_dim)
        x = x.reshape(B, T * self.n_tokens, -1)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 3.  Reference Channel Expander  (zero-init trick for patch_embed)
# ---------------------------------------------------------------------------

def expand_patch_embed_to_32ch(patch_embed_proj: nn.Conv2d) -> nn.Conv2d:
    """
    Expand a 16-channel Conv2d patch embed proj to 32 channels.
    The original 16-channel weights are preserved exactly.
    The new 16 channels (for the ref latent) are zero-initialised.

    This means at the very start of training, the reference frame has
    zero influence — the model can only improve from there.

    Args:
        patch_embed_proj : original nn.Conv2d with in_channels=16

    Returns:
        new nn.Conv2d with in_channels=32, original weights intact
    """
    old = patch_embed_proj
    new = nn.Conv2d(
        in_channels=32,               # 16 video + 16 reference
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        # Copy original weights into first 16 channels
        new.weight[:, :16, :, :] = old.weight.clone()
        # Zero-init the reference frame channels
        new.weight[:, 16:, :, :] = 0.0
        if old.bias is not None:
            new.bias.data = old.bias.data.clone()
    return new


# ---------------------------------------------------------------------------
# 4.  Main Model: TalkingHeadsDiT
# ---------------------------------------------------------------------------

class TalkingHeadsDiT(nn.Module):
    """
    Talking Heads Diffusion Transformer.

    Wraps CogVideoX-5B with three conditioning pathways:
      - Audio (via encoder_hidden_states cross-attention, already in CogVideoX)
      - Pose  (per-frame MLP token, added to video hidden states before blocks)
      - Reference image (channel-concatenated, zero-init expanded patch embed)

    Parameters
    ----------
    inner_dim : int
        CogVideoX-5B inner dimension. Do not change unless you know what
        you're doing. Default: 1920 (30 heads × 64 dim/head).
    audio_input_dim : int
        Output dim of your AudioEncoder. Default: 3072.
    audio_tokens_per_frame : int
        How many audio tokens to generate per video frame for cross-attention.
        More tokens = richer audio context but larger sequence. Default: 4.
    pose_dropout : float
        Dropout in pose MLP. Helps prevent over-reliance on pose during
        the APDH curriculum where pose is progressively dropped. Default: 0.1.
    gradient_checkpointing : bool
        Enable gradient checkpointing on transformer blocks. Mandatory on
        A100 for full fine-tuning to avoid OOM. Default: True.
    freeze_backbone : bool
        If True, all pretrained CogVideoX weights are frozen. Only new
        conditioning modules train. Set False for full fine-tuning.
        Default: True (recommended for first training stage).
    """

    COGVIDEOX_INNER_DIM      = 3072   # 30 × 64
    COGVIDEOX_IN_CHANNELS    = 16
    COGVIDEOX_NUM_LAYERS     = 42
    COGVIDEOX_TEXT_EMBED_DIM = 4096   # T5 slot — we repurpose via projection

    def __init__(
        self,
        inner_dim: int = 3072,
        audio_input_dim: int = 3072,
        audio_tokens_per_frame: int = 4,
        pose_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        assert inner_dim == self.COGVIDEOX_INNER_DIM, (
            f"inner_dim must be 1920 for CogVideoX-5B, got {inner_dim}"
        )

        self.inner_dim = inner_dim
        self.audio_tokens_per_frame = audio_tokens_per_frame

        # ------------------------------------------------------------------
        # 4a. Load pretrained CogVideoX-5B transformer backbone
        #     We load the config and construct the model; weights are loaded
        #     separately in the training script via from_pretrained.
        #     The model is constructed here with the correct expanded in_channels.
        # ------------------------------------------------------------------
        self.backbone = CogVideoXTransformer3DModel(
            num_attention_heads=48,
            attention_head_dim=64,
            in_channels=32,            # 16 video + 16 ref (expanded)
            out_channels=16,           # predict noise only in video channels
            flip_sin_to_cos=True,
            freq_shift=0,
            time_embed_dim=512,
            text_embed_dim=inner_dim,  # audio tokens will be this dim
            num_layers=42,
            dropout=0.0,
            attention_bias=True,
            sample_width=48,           # 96 / patch_size(2) = 48
            sample_height=48,
            sample_frames=24,          # CogVideoX default, overridden at fwd
            patch_size=2,
            patch_size_t=None,         # CogVideoX 1.0 style
            temporal_compression_ratio=4,
            max_text_seq_length=226,   # audio seq length ceiling
            activation_fn="gelu-approximate",
            timestep_activation_fn="silu",
            norm_elementwise_affine=True,
            norm_eps=1e-5,
            spatial_interpolation_scale=1.875,
            temporal_interpolation_scale=1.0,
            use_rotary_positional_embeddings=True,
            use_learned_positional_embeddings=False,
        )

        if gradient_checkpointing:
            self.backbone.enable_gradient_checkpointing()

        # ------------------------------------------------------------------
        # 4b. New conditioning modules (all randomly initialised)
        # ------------------------------------------------------------------
        self.audio_proj = AudioProjection(
            audio_dim=audio_input_dim,
            inner_dim=inner_dim,
            n_tokens_per_frame=audio_tokens_per_frame,
            dropout=0.1,
        )

        self.pose_encoder = PoseEncoder(
            inner_dim=inner_dim,
            dropout=pose_dropout,
        )

        # Learnable scale for pose conditioning — starts at zero
        # so pose has no effect at init, grows as training proceeds.
        self.pose_scale = nn.Parameter(torch.zeros(1))

        # ------------------------------------------------------------------
        # 4c. Freezing strategy
        # ------------------------------------------------------------------
        if freeze_backbone:
            self._freeze_backbone()
        # New modules always train — they have no pretrained weights to protect.

    # ----------------------------------------------------------------------
    # Freezing helpers
    # ----------------------------------------------------------------------

    def _freeze_backbone(self):
        """Freeze all pretrained backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone_top_n_layers(self, n: int):
        """
        Unfreeze the top N transformer blocks of the backbone.
        Call this from the training script as the APDH curriculum progresses.

        Args:
            n : Number of blocks to unfreeze from the end (deepest layers).
        """
        total = len(self.backbone.transformer_blocks)
        for i, block in enumerate(self.backbone.transformer_blocks):
            if i >= total - n:
                for param in block.parameters():
                    param.requires_grad_(True)

    def unfreeze_full_backbone(self):
        """Stage 2: full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    # ----------------------------------------------------------------------
    # Weight loading helper
    # ----------------------------------------------------------------------

    @classmethod
    def from_pretrained_cogvideox(
        cls,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b",
        **kwargs,
    ) -> "TalkingHeadsDiT":
        """
        Construct TalkingHeadsDiT and load pretrained CogVideoX-5B weights
        into the backbone (except patch_embed.proj which we expand to 32ch).

        Usage:
            model = TalkingHeadsDiT.from_pretrained_cogvideox(
                "THUDM/CogVideoX-5b", freeze_backbone=True
            )
        """
        # Build model with expanded 32ch input
        model = cls(**kwargs)

        # Load original pretrained backbone (16ch) from HF
        print(f"[TalkingHeadsDiT] Loading pretrained weights from {pretrained_model_name_or_path}...")
        pretrained = CogVideoXTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # Carefully copy weights: everything except patch_embed.proj
        # (which changed from 16→32 channels)
        # ------------------------------------------------------------------
        pretrained_sd = pretrained.state_dict()
        model_sd = model.backbone.state_dict()

        keys_to_skip = set()
        mismatched = []

        for key in pretrained_sd:
            if key not in model_sd:
                mismatched.append(f"MISSING in model: {key}")
                continue
            if pretrained_sd[key].shape != model_sd[key].shape:
                # This is the patch_embed.proj.weight: (out, 16, kH, kW) vs (out, 32, kH, kW)
                keys_to_skip.add(key)
                print(f"  [SKIP] {key}: pretrained {pretrained_sd[key].shape} "
                      f"→ model {model_sd[key].shape} — using zero-init expansion.")
                continue
            model_sd[key] = pretrained_sd[key]

        if mismatched:
            print(f"  [WARN] {len(mismatched)} keys not found in pretrained model.")

        model.backbone.load_state_dict(model_sd, strict=False)

        # Now apply zero-init expansion for the skipped patch_embed.proj
        # Now apply zero-init expansion ONLY for the skipped patch_embed.proj
        for key in keys_to_skip:
            if key == "patch_embed.proj.weight":
                param_model = dict(model.backbone.named_parameters())[key]
                param_pretrained = pretrained_sd[key]
                with torch.no_grad():
                    param_model[:, :16, ...] = param_pretrained
                    param_model[:, 16:, ...] = 0.0
            else:
                print(f"  [INFO] Expected mismatch for {key}. Leaving randomly initialized.")

        return model

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        video_latents: torch.Tensor,
        ref_latents: torch.Tensor,
        timestep: torch.Tensor,
        audio_embeds: torch.Tensor,
        pose_keypoints: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """
        Forward pass — predicts noise in the video latent.

        Args:
            video_latents  : (B, 16, T, H, W)
                             Noisy video latent at timestep t.
                             H=W=96 for 768×768 input.

            ref_latents    : (B, 16, 1, H, W)
                             Reference image latent. Expanded to (B, 16, T, H, W)
                             by repeating along the time dim, then concatenated
                             with video_latents on channel dim → (B, 32, T, H, W).

            timestep       : (B,) — diffusion timestep, int or float tensor.

            audio_embeds   : (B, T, audio_input_dim)
                             Frame-aligned audio embeddings from AudioEncoder.

            pose_keypoints : (B, T, 133, 2) or None
                             DWPose keypoints normalised to [0, 1].
                             If None, pose conditioning is skipped (for APDH
                             curriculum stages where pose is fully dropped).

            attention_kwargs : passed through to CogVideoX attention blocks.

        Returns:
            noise_pred : (B, 16, T, H, W) — predicted noise in video latent space.
        """
        B, C, T, H, W = video_latents.shape
        assert C == 16, f"Expected 16 video latent channels, got {C}"

        # ------------------------------------------------------------------
        # Step 1: Reference conditioning via channel concatenation
        # Broadcast ref latent across time, concat on channel dim.
        # ------------------------------------------------------------------
        ref_expanded = ref_latents.expand(B, 16, T, H, W)          # (B, 16, T, H, W)
        x = torch.cat([video_latents, ref_expanded], dim=1)         # (B, 32, T, H, W)

        # CogVideoX expects (B, T, C, H, W) — permute
        x = x.permute(0, 2, 1, 3, 4)                               # (B, T, 32, H, W)

        # ------------------------------------------------------------------
        # Step 2: Audio conditioning
        # Project audio embeds to inner_dim, use as encoder_hidden_states.
        # This slot is normally T5 text — we repurpose it for audio.
        # Shape must be (B, seq_len, inner_dim).
        # ------------------------------------------------------------------
        audio_tokens = self.audio_proj(audio_embeds)
        # (B, T * audio_tokens_per_frame, inner_dim)

        # ------------------------------------------------------------------
        # Step 3: Backbone forward (CogVideoX DiT)
        # The backbone does: patch_embed → 42 blocks → unpatchify
        # Each block has joint attention over (video_tokens, audio_tokens).
        # ------------------------------------------------------------------
        backbone_out = self.backbone(
            hidden_states=x,
            encoder_hidden_states=audio_tokens,
            timestep=timestep,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )
        # backbone_out[0]: (B, T, 16, H, W) — note CogVideoX returns T before C
        noise_pred = backbone_out[0]

        # ------------------------------------------------------------------
        # Step 4: Pose conditioning (additive, post-backbone injection)
        #
        # We inject pose AFTER the backbone to keep the injection point clean
        # and controllable. The pose signal is:
        #   pose_tokens: (B, T, inner_dim) → unpatchified back to spatial → add
        #
        # Simpler and equally effective alternative used here:
        # Project pose to (B, T, 16, 1, 1) and broadcast-add to noise_pred.
        # This lets the model learn "shift the prediction based on pose" per frame.
        # ------------------------------------------------------------------
        if pose_keypoints is not None:
            pose_feat = self.pose_encoder(pose_keypoints)     # (B, T, inner_dim)

            # Project inner_dim → 16 output channels for noise space residual
            if not hasattr(self, '_pose_to_noise'):
                # Lazy-init to avoid bloating __init__; created once
                self._pose_to_noise = nn.Linear(
                    self.inner_dim, 16, bias=False
                ).to(pose_feat.device, dtype=pose_feat.dtype)
                nn.init.zeros_(self._pose_to_noise.weight)

            pose_residual = self._pose_to_noise(pose_feat)    # (B, T, 16)
            # Reshape to (B, T, 16, 1, 1) and broadcast
            pose_residual = pose_residual.unsqueeze(-1).unsqueeze(-1)
            # Permute noise_pred to (B, T, 16, H, W) — CogVideoX already returns this
            noise_pred = noise_pred + self.pose_scale * pose_residual

        # CogVideoX returns (B, T, 16, H, W). Convert back to (B, 16, T, H, W)
        # to match the convention of your data pipeline.
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)             # (B, 16, T, H, W)

        return noise_pred

    # ----------------------------------------------------------------------
    # Introspection helpers
    # ----------------------------------------------------------------------

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> str:
        trainable = self.num_trainable_params()
        total = self.num_total_params()
        pct = 100 * trainable / total if total > 0 else 0
        return (
            f"TalkingHeadsDiT | "
            f"trainable: {trainable:,} ({pct:.1f}%) | "
            f"total: {total:,}"
        )


# ---------------------------------------------------------------------------
# Smoke test — run directly: python models/talking_heads_dit.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Smoke test on device={device}, dtype={dtype}\n")

    # --- Instantiate WITHOUT loading pretrained weights (for CI / shape test) ---
    model = TalkingHeadsDiT(
        inner_dim=1920,
        audio_input_dim=3072,
        audio_tokens_per_frame=4,
        pose_dropout=0.1,
        gradient_checkpointing=False,   # off for smoke test speed
        freeze_backbone=True,
    ).to(device)

    if device == "cuda":
        model = model.to(dtype)

    print(model.param_summary())
    print()

    # --- Dummy tensors matching your exact data shapes ---
    B = 1       # batch size
    T = 24      # 1 second at 24 fps
    H = W = 96  # 768 / 8 (VAE spatial compression)
    AUDIO_DIM = 3072

    video_latents  = torch.randn(B, 16, T, H, W, device=device, dtype=dtype)
    ref_latents    = torch.randn(B, 16, 1, H, W, device=device, dtype=dtype)
    timestep       = torch.randint(0, 1000, (B,), device=device)
    audio_embeds   = torch.randn(B, T, AUDIO_DIM, device=device, dtype=dtype)
    pose_keypoints = torch.rand(B, T, 133, 2, device=device, dtype=dtype)   # [0,1]

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device=="cuda")):
            noise_pred = model(
                video_latents  = video_latents,
                ref_latents    = ref_latents,
                timestep       = timestep,
                audio_embeds   = audio_embeds,
                pose_keypoints = pose_keypoints,
            )

    print(f"video_latents  : {tuple(video_latents.shape)}")
    print(f"ref_latents    : {tuple(ref_latents.shape)}")
    print(f"audio_embeds   : {tuple(audio_embeds.shape)}")
    print(f"pose_keypoints : {tuple(pose_keypoints.shape)}")
    print(f"noise_pred     : {tuple(noise_pred.shape)}")

    expected = (B, 16, T, H, W)
    assert noise_pred.shape == expected, \
        f"Shape mismatch: expected {expected}, got {tuple(noise_pred.shape)}"

    print("\n✓ Smoke test passed. TalkingHeadsDiT is ready.")
    print("  Load pretrained weights with:")
    print("  model = TalkingHeadsDiT.from_pretrained_cogvideox('THUDM/CogVideoX-5b')")
