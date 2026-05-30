"""
Models/talking_heads_dit.py

TalkingHeadsDiT — CogVideoX-2B backbone with EchoMimicV2-style conditioning,
ported to the CogVideoX Diffusion Transformer.

This is the v2 re-architecture. The original version routed audio through
CogVideoX's frozen T5 *text* slot after interpolating it to 226 tokens, which
destroyed the frame-level time alignment that lip-sync depends on. This version
injects audio the way EchoMimicV2 does: dedicated, frame-aligned audio
cross-attention.

Conditioning pathways
---------------------
  1. Audio  — frame-aligned WINDOWED cross-attention injected *between*
     CogVideoX transformer blocks (ControlNet-style, zero-init gate). Each
     frame's video tokens attend only to a small temporal window of audio
     features centred on that frame, so lip motion stays time-locked to speech.
     The text slot is fed a learned NULL embedding (CogVideoX still needs an
     encoder sequence for its joint attention).
  2. Pose   — DWPose keypoints rendered to per-frame heatmaps at LATENT
     resolution, passed through a zero-init 3D-conv "pose guider" that produces
     a spatial latent residual added to the video latent BEFORE patch embed.
  3. Reference frame — reference image latent channel-concatenated to the noisy
     video latent (16 -> 32 ch); patch-embed conv expanded 16->32 with the new
     channels zero-initialised and kept trainable.

Resolution
----------
Train/infer at CogVideoX-2B's NATIVE latent grid (480x720 video -> 60x90 or
90x60 latent). CogVideoX-2B uses sinusoidal positional embeddings; off-native
resolutions silently degrade the pretrained spatial prior and cost ~3x more
spatial tokens, so the dataset is re-encoded to native resolution.

Trainable surface
-----------------
LoRA on attention + audio cross-attention adapters + null-text + pose guider +
expanded reference patch-embed channels. The bulk of the backbone stays frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# ---------------------------------------------------------------------------
# DWPose wholebody (133 kp) -> 3-channel heatmap groups
#   0-16  body (17) | 17-22 feet (6) | 23-90 face (68)
#   91-111 left hand (21) | 112-132 right hand (21)
# ---------------------------------------------------------------------------
POSE_CHANNEL_GROUPS: dict[int, list[int]] = {
    0: list(range(23, 91)),                       # face
    1: list(range(0, 23)),                        # body + feet
    2: list(range(91, 133)),                      # hands
}
POSE_NUM_CHANNELS = len(POSE_CHANNEL_GROUPS)


def render_pose_heatmaps(
    keypoints: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Splat normalised keypoints onto a multi-channel heatmap at latent resolution.

    keypoints : (B, T, 133, 2) — normalised (x, y) in [0, 1]
    returns   : (B, POSE_NUM_CHANNELS, T, height, width)
    """
    B, T, K, _ = keypoints.shape
    device, dtype = keypoints.device, keypoints.dtype

    xs = (keypoints[..., 0].clamp(0, 1) * (width - 1)).round().long()    # (B,T,K)
    ys = (keypoints[..., 1].clamp(0, 1) * (height - 1)).round().long()
    lin = ys * width + xs                                                # (B,T,K) flat pixel idx

    hm = torch.zeros(B, POSE_NUM_CHANNELS, T, height * width,
                     device=device, dtype=dtype)
    for ch, kp_ids in POSE_CHANNEL_GROUPS.items():
        idx = lin[:, :, kp_ids]                                         # (B,T,nk)
        ones = torch.ones_like(idx, dtype=dtype)
        hm[:, ch].scatter_add_(2, idx, ones)

    hm = hm.clamp(max=1.0).view(B, POSE_NUM_CHANNELS, T, height, width)
    return hm


# ---------------------------------------------------------------------------
# Pose guider — spatial latent residual (zero-init output)
# ---------------------------------------------------------------------------
class PoseGuider(nn.Module):
    def __init__(self, in_channels: int = POSE_NUM_CHANNELS,
                 latent_channels: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, latent_channels, 3, padding=1),
        )
        # Zero-init the final conv so pose has no effect at init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """heatmaps: (B, C, T, H, W) -> (B, latent_channels, T, H, W)"""
        return self.net(heatmaps)


# ---------------------------------------------------------------------------
# Windowed audio cross-attention (frame-aligned, zero-init gated)
# ---------------------------------------------------------------------------
class WindowedAudioCrossAttention(nn.Module):
    """
    Per-frame windowed cross-attention from video tokens to audio features.

    For each latent frame f, the HW video tokens of that frame (queries) attend
    to a temporal window [f-w, f+w] of per-frame audio features (keys/values).
    This keeps lip/gesture motion locked to the audio timeline instead of a
    globally-pooled token soup. Output is zero-init gated so the layer is a
    no-op at init and ramps up during training (ControlNet-style).

    Projections are named `audio_*` so PEFT's attention LoRA (which targets
    `to_q/to_k/to_v/to_out`) does NOT accidentally wrap them.
    """

    def __init__(
        self,
        dim: int,
        audio_dim: int,
        heads: int = 8,
        head_dim: int = 64,
        window: int = 2,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.window = window
        self.scale = head_dim ** -0.5
        inner = heads * head_dim

        self.norm_q = nn.LayerNorm(dim)
        self.audio_q = nn.Linear(dim, inner, bias=False)
        self.audio_k = nn.Linear(audio_dim, inner, bias=False)
        self.audio_v = nn.Linear(audio_dim, inner, bias=False)
        self.audio_out = nn.Linear(inner, dim)
        self.gate = nn.Parameter(torch.zeros(1))

        # Zero-init output so the residual is exactly 0 at init.
        nn.init.zeros_(self.audio_out.weight)
        nn.init.zeros_(self.audio_out.bias)

    def forward(
        self,
        video_tokens: torch.Tensor,   # (B, T*HW, dim) frame-major
        audio_feats: torch.Tensor,    # (B, T, audio_dim)
        T: int,
        HW: int,
    ) -> torch.Tensor:
        B, N, dim = video_tokens.shape
        assert N == T * HW, f"token count {N} != T*HW {T*HW}"
        w = self.window
        nh, hd = self.heads, self.head_dim

        x = self.norm_q(video_tokens).view(B, T, HW, dim)

        # Build per-frame audio windows: (B, T, 2w+1, audio_dim)
        a = audio_feats.transpose(1, 2)                       # (B, A, T)
        a = F.pad(a, (w, w), mode="replicate")                # (B, A, T+2w)
        a = a.transpose(1, 2)                                 # (B, T+2w, A)
        windows = a.unfold(1, 2 * w + 1, 1)                   # (B, T, A, 2w+1)
        windows = windows.permute(0, 1, 3, 2).contiguous()   # (B, T, 2w+1, A)

        q = self.audio_q(x).view(B, T, HW, nh, hd)
        k = self.audio_k(windows).view(B, T, 2 * w + 1, nh, hd)
        v = self.audio_v(windows).view(B, T, 2 * w + 1, nh, hd)

        attn = torch.einsum("bthnd,btwnd->bthnw", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bthnw,btwnd->bthnd", attn, v)     # (B,T,HW,nh,hd)
        out = out.reshape(B, T, HW, nh * hd)
        out = self.audio_out(out).reshape(B, N, dim)
        return self.gate * out


# ---------------------------------------------------------------------------
# CogVideoX transformer subclass with audio cross-attention injected between
# blocks. Overrides forward to keep full control of the block loop.
# ---------------------------------------------------------------------------
class ConditionedCogVideoXTransformer(CogVideoXTransformer3DModel):
    """CogVideoX-2B transformer + injected frame-aligned audio cross-attention."""

    def add_audio_conditioning(
        self,
        audio_dim: int = 768,
        audio_layers: list[int] | None = None,
        audio_window: int = 2,
        audio_heads: int = 8,
        audio_head_dim: int = 64,
    ) -> None:
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        num_layers = self.config.num_layers
        if audio_layers is None:
            # spread audio attention across the network (every other block)
            audio_layers = list(range(1, num_layers, 2))
        self.audio_layers = set(audio_layers)

        # shared audio pre-net (latent-frame features -> normalised features)
        self.audio_prenet = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, audio_dim),
            nn.SiLU(),
        )
        self.audio_adapters = nn.ModuleDict({
            str(i): WindowedAudioCrossAttention(
                dim=inner_dim, audio_dim=audio_dim,
                heads=audio_heads, head_dim=audio_head_dim, window=audio_window,
            )
            for i in audio_layers
        })

    # ----------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep,
        timestep_cond=None,
        ofs=None,
        image_rotary_emb=None,
        attention_kwargs: dict | None = None,
        audio_feats: torch.Tensor | None = None,   # (B, T, audio_dim)
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape
        p = self.config.patch_size
        HW = (height // p) * (width // p)

        # 1. Time embedding
        t_emb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs).to(dtype=hidden_states.dtype)
            emb = emb + self.ofs_embedding(ofs_emb)

        # 2. Patch embedding (text + video jointly, + positional embeds)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]        # video tokens (B, T*HW, D)

        # audio features at latent-frame rate
        a_feats = None
        if audio_feats is not None and len(self.audio_layers) > 0:
            a = audio_feats
            if a.shape[1] != num_frames:
                a = F.interpolate(a.transpose(1, 2).float(), size=num_frames,
                                  mode="linear", align_corners=False).transpose(1, 2)
                a = a.to(hidden_states.dtype)
            a_feats = self.audio_prenet(a)

        # 3. Transformer blocks (+ audio cross-attention between blocks)
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, emb,
                    image_rotary_emb, attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

            if a_feats is not None and i in self.audio_layers:
                hidden_states = hidden_states + self.audio_adapters[str(i)](
                    hidden_states, a_feats, num_frames, HW
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p_t = self.config.patch_size_t
        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class TalkingHeadsDiT(nn.Module):
    def __init__(
        self,
        backbone: ConditionedCogVideoXTransformer,
        *,
        inner_dim: int,
        text_embed_dim: int,
        latent_channels: int,
        use_rotary: bool,
        audio_input_dim: int = 768,
        audio_seq_len: int = 226,
        pose_hidden: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.inner_dim = inner_dim
        self.text_embed_dim = text_embed_dim
        self.latent_channels = latent_channels
        self.use_rotary = use_rotary
        self.audio_seq_len = audio_seq_len

        # Learned NULL text embedding fed through CogVideoX's encoder slot
        # (joint attention still needs an encoder sequence; audio no longer
        # rides this slot — it goes through the audio cross-attention).
        self.null_text = nn.Parameter(
            torch.randn(1, audio_seq_len, text_embed_dim) * 0.02
        )

        self.pose_guider = PoseGuider(
            in_channels=POSE_NUM_CHANNELS,
            latent_channels=latent_channels,
            hidden=pose_hidden,
        )
        self.pose_scale = nn.Parameter(torch.zeros(1))

    # ----------------------------------------------------------------------
    @classmethod
    def from_pretrained_cogvideox(
        cls,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-2b",
        *,
        freeze_backbone: bool = True,
        gradient_checkpointing: bool = True,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        audio_input_dim: int = 768,
        audio_layers: list[int] | None = None,
        audio_window: int = 2,
        pose_hidden: int = 64,
    ) -> "TalkingHeadsDiT":
        print(f"[TalkingHeadsDiT] Loading pretrained transformer from "
              f"{pretrained_model_name_or_path} ...")
        pretrained = CogVideoXTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )

        cfg = dict(pretrained.config)
        cfg["in_channels"] = 32                       # 16 video + 16 reference
        backbone = ConditionedCogVideoXTransformer.from_config(cfg)

        inner_dim = pretrained.config.num_attention_heads * pretrained.config.attention_head_dim
        text_embed_dim = pretrained.config.text_embed_dim
        latent_channels = pretrained.config.out_channels
        use_rotary = bool(pretrained.config.use_rotary_positional_embeddings)
        audio_seq_len = pretrained.config.max_text_seq_length

        # ---- copy weights (expand patch_embed.proj 16 -> 32) ----
        sd_p = pretrained.state_dict()
        sd_n = backbone.state_dict()
        copied = 0
        for k, v in sd_p.items():
            if k == "patch_embed.proj.weight":
                w = sd_n[k].clone()
                w[:, :16] = v
                w[:, 16:] = 0.0
                sd_n[k] = w
                copied += 1
            elif k in sd_n and sd_n[k].shape == v.shape:
                sd_n[k] = v
                copied += 1
        backbone.load_state_dict(sd_n, strict=False)

        coverage = copied / len(sd_p)
        print(f"[TalkingHeadsDiT] Copied {copied}/{len(sd_p)} backbone tensors "
              f"({coverage:.1%}).")
        assert coverage > 0.95, (
            f"Only {coverage:.1%} of backbone weights copied — config/checkpoint "
            f"mismatch. Refusing to train from a half-loaded backbone."
        )
        del pretrained

        # ---- inject audio cross-attention adapters (before freezing/LoRA) ----
        backbone.add_audio_conditioning(
            audio_dim=audio_input_dim,
            audio_layers=audio_layers,
            audio_window=audio_window,
        )
        print(f"[TalkingHeadsDiT] Audio cross-attention on layers "
              f"{sorted(backbone.audio_layers)}.")

        if gradient_checkpointing:
            backbone.enable_gradient_checkpointing()

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none",
            )
            # get_peft_model injects LoRA in-place; we keep calling the
            # transformer's own (overridden) forward directly, so LoRA stays
            # active without routing through the PeftModel wrapper.
            get_peft_model(backbone, lora_cfg)
            print("[TalkingHeadsDiT] LoRA injected into attention projections.")

        # Re-enable the trainable conditioning surface that PEFT just froze:
        #   * expanded reference channels in patch_embed.proj
        #   * audio cross-attention adapters + audio pre-net
        TRAINABLE_KEYS = ("patch_embed.proj", "audio_adapters", "audio_prenet")
        for name, p in backbone.named_parameters():
            if any(key in name for key in TRAINABLE_KEYS):
                p.requires_grad_(True)

        return cls(
            backbone,
            inner_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            latent_channels=latent_channels,
            use_rotary=use_rotary,
            audio_input_dim=audio_input_dim,
            audio_seq_len=audio_seq_len,
            pose_hidden=pose_hidden,
        )

    # ----------------------------------------------------------------------
    def forward(
        self,
        video_latents: torch.Tensor,        # (B, 16, T, H, W) — noisy
        ref_latents: torch.Tensor,          # (B, 16, 1, H, W)
        timestep: torch.Tensor,             # (B,)
        audio_embeds: torch.Tensor,         # (B, T_audio, audio_input_dim)
        pose_keypoints: torch.Tensor | None = None,   # (B, T, 133, 2) in [0,1]
        image_rotary_emb=None,
        attention_kwargs: dict | None = None,
    ) -> torch.Tensor:
        B, C, T, H, W = video_latents.shape

        x_video = video_latents
        if pose_keypoints is not None:
            heatmaps = render_pose_heatmaps(pose_keypoints, H, W).to(x_video.dtype)
            pose_residual = self.pose_guider(heatmaps)          # (B, 16, T, H, W)
            x_video = x_video + self.pose_scale * pose_residual

        ref_expanded = ref_latents.expand(-1, -1, T, -1, -1)    # (B, 16, T, H, W)
        x = torch.cat([x_video, ref_expanded], dim=1)           # (B, 32, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)                            # (B, T, 32, H, W)

        null_text = self.null_text.expand(B, -1, -1).to(x.dtype)

        out = self.backbone(
            hidden_states=x,
            encoder_hidden_states=null_text,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            audio_feats=audio_embeds,
            return_dict=False,
        )[0]                                                    # (B, T, 16, H, W)

        return out.permute(0, 2, 1, 3, 4)                       # (B, 16, T, H, W)

    # ----------------------------------------------------------------------
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> str:
        tr, tot = self.num_trainable_params(), self.num_total_params()
        pct = 100 * tr / tot if tot else 0
        return f"TalkingHeadsDiT | trainable: {tr:,} ({pct:.2f}%) | total: {tot:,}"
