"""
Models/audio_encoder.py

Wraps facebook/wav2vec2-base to produce frame-aligned audio embeddings
suitable for cross-attention injection into the DiT backbone.

Input:
  audio_values : (B, T_audio)   -- raw 16 kHz waveform

Output:
  embeddings   : (B, T_frames, output_dim)  -- frame-aligned, ready for xattn
                 Default output_dim=1920 matches CogVideoX-2B inner_dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class TemporalAdapter(nn.Module):
    """
    Interpolates Wav2Vec2 token sequence to target frame count.
    Wav2Vec2-base produces ~50 tokens/sec. Video is 24fps.
    Interpolation + conv handles the mismatch cleanly.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim, in_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_dim,
            bias=False,
        )
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        x : (B, T_audio_tokens, in_dim)
        returns : (B, target_frames, out_dim)
        """
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.norm(x)
        return x


class AudioEncoder(nn.Module):
    """
    Wav2Vec2-base backbone with temporal adapter head.

    Args:
        output_dim    : must match DiT inner_dim. CogVideoX-5B = 3072.
        freeze_encoder: freeze Wav2Vec2 weights. Default True.
        model_name    : HuggingFace model ID.
    """

    WAV2VEC2_HIDDEN = 768

    def __init__(
        self,
        output_dim    : int  = 1920,   # CogVideoX-2B inner_dim (was 3072 for 5B)
        freeze_encoder: bool = True,
        model_name    : str  = "facebook/wav2vec2-base",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.wav2vec2   = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad_(False)

        self.temporal_adapter = TemporalAdapter(
            in_dim  = self.WAV2VEC2_HIDDEN,
            out_dim = output_dim,
        )

    def forward(
        self,
        audio_values  : torch.Tensor,
        target_frames : int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        audio_values  : (B, T_audio) raw 16kHz waveform in [-1, 1]
        target_frames : number of video latent frames
        returns       : (B, target_frames, output_dim)
        """
        wav2vec_out = self.wav2vec2(
            audio_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden     = wav2vec_out.last_hidden_state
        embeddings = self.temporal_adapter(hidden, target_frames)
        return embeddings

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())