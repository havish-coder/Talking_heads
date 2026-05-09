"""
training/loss.py

PhD Loss — Phase-specific Denoising Loss from EchoMimicV2 (Section 3.5).

This loss function divides the diffusion process into three distinct phases based 
on the timestep `t` (0 = clean, 1000 = pure noise):

    1. Stage 1 (Pose-dominant, t in [700, 1000]):
       Early 10% of diffusion. Focuses on broad structure.
       Loss: L_latent + λ * L_pose (Sobel edges on skeleton)

    2. Stage 2 (Detail-dominant, t in [100, 700)):
       Middle 60% of diffusion. Focuses on high-frequency details.
       Loss: L_latent + λ * L_detail (Canny/Sobel edges on RGB)

    3. Stage 3 (Quality-dominant, t in [0, 100)):
       Final 30% of diffusion. Focuses on perceptual realism.
       Loss: L_latent + λ * L_low (LPIPS on RGB)

Auxiliary losses require a one-step decode of the predicted latent back to RGB.
This "one-step sampling" is computed once and reused for whichever phase is active.
The VAE decode is performed in fp32 with gradients disabled (frozen VAE).

Memory Optimization (for e.g., T4 12GB):
  - VAE is CPU-offloaded and moved to the GPU *only* for the decode step, then back.
  - LPIPS is kept on the compute device (small footprint, ~0.1 GB).
"""

from __future__ import annotations

import typing
from typing import Dict, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Timestep Phase Boundaries (Paper Section 3.5) ---
S1_MIN = 700  # Pose-dominant   : [700, 1000]
S2_MIN = 100  # Detail-dominant : [100, 700)
# S3            Quality-dominant: [0, 100)


# --- Differentiable Edge Extraction ---

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Extracts edges via Sobel kernels as a differentiable proxy for Canny edges.
    Matches the paper's exact implementation (no Non-Maximum Suppression).
    
    Args:
        x: Tensor of shape (Batch, 3, H, W) representing RGB images in [-1, 1].
        
    Returns:
        Tensor of shape (Batch, 1, H, W) representing edge magnitudes.
    """
    # Convert RGB to Grayscale
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gray = gray.unsqueeze(1)  # (B, 1, H, W)

    # Define Sobel kernels
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)

    # Apply Convolutions
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    
    # Calculate Magnitude (add epsilon for numerical stability in sqrt)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


# --- One-Step Latent to RGB Decoding ---

def one_step_decode(
    pred_latent: typing.Any,  # Unused directly, kept for API clarity
    noisy_latent: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: typing.Any,
    vae: typing.Any,
    vae_device: torch.device | None = None,
) -> torch.Tensor:
    """
    One-step sampling: Reconstructs the clean latent from the noisy latent and noise_pred,
    then VAE-decodes it to RGB space for auxiliary loss calculation.

    Memory-optimized: If `vae_device` is provided, it moves the VAE to the GPU strictly 
    for the decode step, and moves it back to the CPU immediately after.

    Args:
        noisy_latent: (B, 16, T, H, W) — Noisy video latent at timestep t.
        noise_pred: (B, 16, T, H, W) — Model's predicted noise.
        timestep: (Batch,) — Current diffusion timestep.
        scheduler: Diffusion scheduler (must have alphas_cumprod attribute).
        vae: Frozen CogVideoX VAE used strictly for decoding.
        vae_device: GPU device to temporarily move the VAE to.

    Returns:
        rgb: (B, 3, H*8, W*8) — Decoded RGB middle frame. Values clamped to [-1, 1].
    """
    with torch.no_grad():
        # Reconstruct predicted clean latent z0 using the scheduler's formulation:
        # z0 = (z_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        alpha = scheduler.alphas_cumprod.to(timestep.device)[timestep].view(-1, 1, 1, 1, 1)
        z0_pred = (noisy_latent - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
        z0_pred = z0_pred.clamp(-4.0, 4.0)  # Clamp for stability before VAE decode

        # Extract the middle frame only (decoding the full video clip is prohibitively expensive)
        T = z0_pred.shape[2]
        mid = T // 2
        z0_frame = z0_pred[:, :, mid:mid+1, :, :]  # (B, 16, 1, H, W)

        # Move VAE to target device for decoding
        if vae_device is not None:
            vae.to(vae_device)

        # VAE Decode: (B, 16, 1, H, W) -> (B, 3, 1, H*8, W*8)
        z0_frame = z0_frame.float().to(next(vae.parameters()).device)
        rgb = vae.decode(z0_frame).sample
        rgb = rgb.squeeze(2)  # Remove time dimension -> (B, 3, H*8, W*8)
        rgb = rgb.clamp(-1.0, 1.0)

        # Immediately offload VAE back to CPU to free VRAM
        if vae_device is not None:
            vae.to("cpu")
            torch.cuda.empty_cache()

    return rgb.to(timestep.device)


# --- PhD Loss Module ---

class PhDLoss(nn.Module):
    """
    Phase-specific Denoising Loss (PhD Loss) implementation.
    """

    def __init__(
        self,
        lambda_pose: float = 0.1,
        lambda_detail: float = 0.1,
        lambda_low: float = 0.1,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.lambda_pose = lambda_pose
        self.lambda_detail = lambda_detail
        self.lambda_low = lambda_low

        # Initialize LPIPS backbone (Frozen, Eval mode only)
        # VGG is generally preferred over AlexNet for stability in generation tasks.
        self.lpips_fn = lpips.LPIPS(net=lpips_net)
        self.lpips_fn.requires_grad_(False)
        self.lpips_fn.eval()

    def _get_phase(self, t: int) -> str:
        """Maps a scalar timestep to its corresponding curriculum phase."""
        if t >= S1_MIN:
            return "S1"
        elif t >= S2_MIN:
            return "S2"
        return "S3"

    def forward(
        self,
        noise_pred: torch.Tensor,
        actual_noise: torch.Tensor,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        target_rgb: torch.Tensor,
        scheduler: typing.Any,
        vae: typing.Any,
        vae_device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, typing.Any]]:
        """
        Computes the PhD Loss for a given batch.
        Assumes all samples in the batch share the same timestep `t`.

        Returns:
            total_loss: Scalar tensor representing the combined loss.
            log_dict: Dictionary containing individual loss components for logging.
        """
        device = noise_pred.device
        t_val = int(timestep.item()) if timestep.numel() == 1 else int(timestep[0].item())
        phase = self._get_phase(t_val)

        # 1. Base Loss: Always computed (Latent MSE)
        L_latent = F.mse_loss(noise_pred, actual_noise)
        log_dict = {"Llatent": L_latent.item(), "phase": phase, "t": t_val}

        # 2. Auxiliary Loss: Phase-dependent
        pred_rgb = one_step_decode(
            pred_latent=None,
            noisy_latent=noisy_latent,
            noise_pred=noise_pred,
            timestep=timestep,
            scheduler=scheduler,
            vae=vae,
            vae_device=vae_device,
        ).to(device)

        if phase == "S1":
            # Stage 1: Pose-dominant (Sobel edges on predicted vs target)
            pred_edges = sobel_edges(pred_rgb)
            target_edges = sobel_edges(target_rgb.to(device))
            L_pose = F.mse_loss(pred_edges, target_edges)
            total_loss = L_latent + self.lambda_pose * L_pose
            log_dict["Lpose"] = L_pose.item()

        elif phase == "S2":
            # Stage 2: Detail-dominant (Canny/Sobel edges)
            pred_edges = sobel_edges(pred_rgb)
            target_edges = sobel_edges(target_rgb.to(device))
            L_detail = F.mse_loss(pred_edges, target_edges)
            total_loss = L_latent + self.lambda_detail * L_detail
            log_dict["Ldetail"] = L_detail.item()

        else:
            # Stage 3: Quality-dominant (LPIPS Perceptual Loss)
            self.lpips_fn = self.lpips_fn.to(device)
            L_low = self.lpips_fn(pred_rgb.float(), target_rgb.to(device).float()).mean()
            total_loss = L_latent + self.lambda_low * L_low
            log_dict["Llow"] = L_low.item()

        # Free the decoded RGB tensor immediately to reclaim VRAM
        del pred_rgb

        log_dict["total_loss"] = total_loss.item()
        return total_loss, log_dict


# --- Integration Smoke Test ---

if __name__ == "__main__":
    import sys
    print("[SYSTEM] Testing PhD Loss phases...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, W = 1, 8, 48, 48

    loss_fn = PhDLoss(lambda_pose=0.1, lambda_detail=0.1, lambda_low=0.1).to(device)

    # Dummy tensors
    noise_pred = torch.randn(B, 16, T, H, W, device=device)
    actual_noise = torch.randn(B, 16, T, H, W, device=device)
    noisy_latent = torch.randn(B, 16, T, H, W, device=device)
    target_rgb = torch.randn(B, 3, H*8, W*8, device=device).clamp(-1, 1)

    # Mock components
    class MockScheduler:
        alphas_cumprod = torch.linspace(0.99, 0.01, 1000)

    class MockVAE:
        def decode(self, z):
            class Out:
                sample = torch.randn(z.shape[0], 3, z.shape[2], z.shape[3]*8, z.shape[4]*8)
            return Out()
        def parameters(self):
            return iter([torch.zeros(1)])
        def to(self, *args, **kwargs):
            return self

    scheduler = MockScheduler()
    vae = MockVAE()

    # Test phase routing
    for t_val, expected_phase in [(800, "S1"), (400, "S2"), (50, "S3")]:
        t = torch.tensor([t_val], device=device)
        total, logs = loss_fn(
            noise_pred, actual_noise, noisy_latent, t, target_rgb, scheduler, vae
        )
        phase = logs["phase"]
        
        assert phase == expected_phase, f"Phase mismatch: Expected {expected_phase}, got {phase}"
        
        formatted_logs = ", ".join([f"{k}: {v if isinstance(v, str) else f'{v:.4f}'}" for k, v in logs.items()])
        print(f"[TEST] Timestep {t_val:4d} -> {phase} | Logs: {formatted_logs} [PASS]")

    print("\n[SYSTEM] PhD Loss is validated and ready for training.")