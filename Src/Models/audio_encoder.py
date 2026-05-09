"""
Models/audio_encoder.py

Wraps facebook/wav2vec2-base to produce frame-aligned audio embeddings
suitable for cross-attention injection into the DiT backbone.

Input:
  audio_values : (B, T_audio) - Raw 16 kHz waveform.

Output:
  embeddings   : (B, T_frames, output_dim) - Frame-aligned embeddings ready for cross-attention.
                 Default output_dim=1920 matches CogVideoX-2B inner_dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class TemporalAdapter(nn.Module):
    """
    Interpolates the Wav2Vec2 token sequence to match the target video frame count.
    Wav2Vec2-base produces approximately 50 tokens/sec, whereas video is typically 24fps.
    A depthwise 1D convolution and linear interpolation resolve this temporal mismatch.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise 1D convolution to smooth temporal features before interpolation
        self.conv = nn.Conv1d(
            in_dim, 
            in_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_dim,
            bias=False,
        )
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, T_audio_tokens, in_dim)
            target_frames: The desired temporal length (number of video frames)
            
        Returns:
            Tensor of shape (Batch, target_frames, out_dim)
        """
        # Permute to (Batch, Channels, Length) for Conv1d and 1D Interpolation
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
        
        # Revert back to (Batch, Length, Channels) for Linear projection and LayerNorm
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.norm(x)
        
        return x


class AudioEncoder(nn.Module):
    """
    Wav2Vec2-base backbone combined with a temporal adapter head.

    Args:
        output_dim: Must match the DiT inner_dim (e.g., 1920 for CogVideoX-2B, 3072 for 5B).
        freeze_encoder: If True, freezes Wav2Vec2 weights to only train the adapter.
        model_name: HuggingFace model ID for the base audio encoder.
    """

    WAV2VEC2_HIDDEN = 768

    def __init__(
        self,
        output_dim: int = 1920,
        freeze_encoder: bool = True,
        model_name: str = "facebook/wav2vec2-base",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze the base audio model to preserve pre-trained representations 
        # and significantly reduce memory overhead during training.
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad_(False)

        self.temporal_adapter = TemporalAdapter(
            in_dim=self.WAV2VEC2_HIDDEN,
            out_dim=output_dim,
        )

    def forward(
        self,
        audio_values: torch.Tensor,
        target_frames: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_values: (Batch, T_audio) raw 16kHz waveform normalized to [-1, 1].
            target_frames: Number of video latent frames to align with.
            attention_mask: Optional mask for padded audio sequences.
            
        Returns:
            Tensor of shape (Batch, target_frames, output_dim).
        """
        wav2vec_out = self.wav2vec2(
            audio_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        
        # Extract the sequence of hidden states from the audio encoder
        hidden = wav2vec_out.last_hidden_state
        
        # Project and interpolate the audio tokens to match the video frame count
        embeddings = self.temporal_adapter(hidden, target_frames)
        
        return embeddings

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Returns a list of parameters that currently require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.trainable_parameters())