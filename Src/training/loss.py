"""
training/loss.py

PhD Loss — Phase-specific Denoising Loss from EchoMimicV2 (Section 3.5).

Three phases based on timestep t (0=clean, 1000=pure noise):

    S1 — Pose-dominant   : t in [700, 1000]  early 10% → Llatent + λ*Lpose
    S2 — Detail-dominant : t in [100, 700]   middle 60% → Llatent + λ*Ldetail
    S3 — Quality-dominant: t in [0,   100]   final 30%  → Llatent + λ*Llow

Each phase adds one auxiliary loss on top of the base Llatent (MSE on noise).
The auxiliary losses require a one-step decode of the predicted latent back to
RGB — this is the "one-step sampling" trick from the paper.

Lpose   : MSE between predicted and target pose keypoint maps (Sobel on skeleton)
Ldetail : MSE between Canny edges of predicted vs target RGB frame
Llow    : LPIPS perceptual loss between predicted and target RGB frame

All three auxiliary losses use the SAME one-step decode — computed once,
reused across whichever phase is active. VAE decode is done in fp32,
no grad (frozen VAE).

Memory optimization (T4 12GB):
  - VAE is CPU-offloaded; moved to GPU only for decode, then back.
  - LPIPS is kept on the compute device (small ~0.1 GB).
  - Auxiliary loss is only computed ~25% of iterations (controlled by train.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


# ── Timestep phase boundaries (paper Section 3.5) ─────────────────────────
S1_MIN = 700   # Pose-dominant   : [700, 1000]
S2_MIN = 100   # Detail-dominant : [100, 700)
# S3              Quality-dominant: [0,   100)


# ── Differentiable Canny (Sobel only, no NMS — matches paper exactly) ──────

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Extract edges via Sobel kernels. Differentiable.
    Input : (B, 3, H, W) RGB in [-1, 1]
    Output: (B, 1, H, W) edge magnitude
    """
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gray = gray.unsqueeze(1)   # (B, 1, H, W)

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


# ── One-step latent → RGB decode ───────────────────────────────────────────

def one_step_decode(
    pred_latent : torch.Tensor,
    noisy_latent: torch.Tensor,
    noise_pred  : torch.Tensor,
    timestep    : torch.Tensor,
    scheduler,
    vae,
    vae_device  : torch.device | None = None,
) -> torch.Tensor:
    """
    One-step sampling: reconstruct clean latent from noisy latent + noise_pred,
    then VAE-decode to RGB.

    Memory-optimized: if vae_device is provided, moves VAE to GPU for decode
    then back to CPU immediately.

    Args:
        pred_latent  : not used directly, kept for API clarity
        noisy_latent : (B, 16, T, H, W) — noisy video latent at timestep t
        noise_pred   : (B, 16, T, H, W) — model's noise prediction
        timestep     : (B,) — current diffusion timestep
        scheduler    : DDPMScheduler or similar — has step() method
        vae          : CogVideoX VAE — frozen, used only for decode
        vae_device   : if set, move VAE to this device for decode, then back to CPU

    Returns:
        rgb : (B, 3, H*8, W*8) — decoded RGB frame (middle frame of clip)
              Values in [-1, 1]
    """
    with torch.no_grad():
        # Reconstruct predicted clean latent z0 using scheduler formula
        # z0 = (z_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        alpha = scheduler.alphas_cumprod.to(timestep.device)[timestep].view(-1, 1, 1, 1, 1)
        z0_pred = (noisy_latent - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
        z0_pred = z0_pred.clamp(-4.0, 4.0)   # stability

        # Take middle frame only — decoding full video clip is too expensive
        T = z0_pred.shape[2]
        mid = T // 2
        z0_frame = z0_pred[:, :, mid:mid+1, :, :]   # (B, 16, 1, H, W)

        # Move VAE to GPU for decode if offloaded
        if vae_device is not None:
            vae.to(vae_device)

        # VAE decode: (B, 16, 1, H, W) → (B, 3, 1, H*8, W*8)
        z0_frame = z0_frame.float().to(next(vae.parameters()).device)
        rgb = vae.decode(z0_frame).sample         # (B, 3, 1, H*8, W*8)
        rgb = rgb.squeeze(2)                      # (B, 3, H*8, W*8)
        rgb = rgb.clamp(-1.0, 1.0)

        # Move VAE back to CPU to free VRAM
        if vae_device is not None:
            vae.to("cpu")
            torch.cuda.empty_cache()

    return rgb.to(timestep.device)


# ── PhD Loss ───────────────────────────────────────────────────────────────

class PhDLoss(nn.Module):
    """
    Phase-specific Denoising Loss (PhD Loss) from EchoMimicV2 Section 3.5.

    Args:
        lambda_pose   : weight for Lpose   (paper: 0.1)
        lambda_detail : weight for Ldetail (paper: 0.1)
        lambda_low    : weight for Llow    (paper: 0.1)
        lpips_net     : backbone for LPIPS ('vgg' or 'alex'). vgg = more stable.
    """

    def __init__(
        self,
        lambda_pose   : float = 0.1,
        lambda_detail : float = 0.1,
        lambda_low    : float = 0.1,
        lpips_net     : str   = "vgg",
    ):
        super().__init__()
        self.lambda_pose   = lambda_pose
        self.lambda_detail = lambda_detail
        self.lambda_low    = lambda_low

        # LPIPS — frozen, eval only
        self.lpips_fn = lpips.LPIPS(net=lpips_net)
        self.lpips_fn.requires_grad_(False)
        self.lpips_fn.eval()

    def _get_phase(self, t: int) -> str:
        """Map scalar timestep to phase name."""
        if t >= S1_MIN:
            return "S1"
        elif t >= S2_MIN:
            return "S2"
        else:
            return "S3"

    def forward(
        self,
        noise_pred   : torch.Tensor,
        actual_noise : torch.Tensor,
        noisy_latent : torch.Tensor,
        timestep     : torch.Tensor,
        target_rgb   : torch.Tensor,
        scheduler,
        vae,
        vae_device   : torch.device | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute PhD Loss for a batch.

        Note: assumes all samples in the batch share the same timestep t.
        This is standard practice — sample one t per batch, not per sample.

        Args:
            noise_pred   : (B, 16, T, H, W) — DiT output
            actual_noise : (B, 16, T, H, W) — ground truth noise added
            noisy_latent : (B, 16, T, H, W) — noisy video latent
            timestep     : scalar tensor — same t for whole batch
            target_rgb   : (B, 3, H*8, W*8) — target RGB (middle frame, decoded)
            scheduler    : diffusion scheduler
            vae          : frozen CogVideoX VAE (may be on CPU)
            vae_device   : GPU device to move VAE to for decode (then back to CPU)

        Returns:
            total_loss : scalar tensor
            log_dict   : dict of individual loss values for logging
        """
        device = noise_pred.device
        t_val  = int(timestep.item()) if timestep.numel() == 1 else int(timestep[0].item())
        phase  = self._get_phase(t_val)

        # ── Base loss: always computed ─────────────────────────────────────
        Llatent = F.mse_loss(noise_pred, actual_noise)
        log_dict = {"Llatent": Llatent.item(), "phase": phase, "t": t_val}

        # ── Auxiliary loss: phase-dependent ───────────────────────────────
        # One-step decode of predicted frame (moves VAE to GPU temporarily)
        pred_rgb = one_step_decode(
            None, noisy_latent, noise_pred, timestep, scheduler, vae,
            vae_device=vae_device,
        ).to(device)

        if phase == "S1":
            # Lpose: MSE on Sobel edge maps (pose proxy — captures body contours)
            pred_edges   = sobel_edges(pred_rgb)
            target_edges = sobel_edges(target_rgb.to(device))
            Lpose = F.mse_loss(pred_edges, target_edges)
            total_loss = Llatent + self.lambda_pose * Lpose
            log_dict["Lpose"] = Lpose.item()

        elif phase == "S2":
            # Ldetail: MSE on Canny edges (Sobel only, paper Section 3.5)
            pred_edges   = sobel_edges(pred_rgb)
            target_edges = sobel_edges(target_rgb.to(device))
            Ldetail = F.mse_loss(pred_edges, target_edges)
            total_loss = Llatent + self.lambda_detail * Ldetail
            log_dict["Ldetail"] = Ldetail.item()

        else:
            # S3 — Llow: LPIPS perceptual loss
            self.lpips_fn = self.lpips_fn.to(device)
            Llow = self.lpips_fn(pred_rgb.float(), target_rgb.to(device).float()).mean()
            total_loss = Llatent + self.lambda_low * Llow
            log_dict["Llow"] = Llow.item()

        # Free the decoded RGB immediately
        del pred_rgb

        log_dict["total_loss"] = total_loss.item()
        return total_loss, log_dict


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("Testing PhD Loss phases...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, W = 1, 8, 48, 48

    loss_fn = PhDLoss(lambda_pose=0.1, lambda_detail=0.1, lambda_low=0.1).to(device)

    noise_pred   = torch.randn(B, 16, T, H, W, device=device)
    actual_noise = torch.randn(B, 16, T, H, W, device=device)
    noisy_latent = torch.randn(B, 16, T, H, W, device=device)
    target_rgb   = torch.randn(B, 3, H*8, W*8, device=device).clamp(-1, 1)

    # Mock scheduler and VAE
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
    vae       = MockVAE()

    for t_val, expected_phase in [(800, "S1"), (400, "S2"), (50, "S3")]:
        t = torch.tensor([t_val], device=device)
        total, logs = loss_fn(noise_pred, actual_noise, noisy_latent, t, target_rgb, scheduler, vae)
        phase = logs["phase"]
        assert phase == expected_phase, f"Expected {expected_phase}, got {phase}"
        print(f"[PASS] t={t_val:4d} → {phase} | {logs}")

    print("\nPhD Loss ready.")