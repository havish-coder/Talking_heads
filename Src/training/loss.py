"""
training/loss.py

PhD Loss — Phase-specific Denoising Loss from EchoMimicV2 (Section 3.5).

Base term is the diffusion regression loss `Llatent` (MSE between the model
prediction and the scheduler's regression target — *velocity* for CogVideoX's
v-prediction scheduler, or noise for an epsilon scheduler). On top of it, a
phase-dependent auxiliary loss is added, based on the sampled timestep:

    S1 — Pose-dominant   : t in [700, 1000]  -> Llatent + lambda * Lpose (edges)
    S2 — Detail-dominant : t in [100, 700)   -> Llatent + lambda * Ldetail (edges)
    S3 — Quality-dominant: t in [0,   100)   -> Llatent + lambda * Llow (LPIPS)

The auxiliary losses need the predicted clean RGB frame, obtained by a one-step
reconstruction of the clean latent (handling whichever prediction type the
scheduler uses) followed by a single-frame VAE decode (frozen VAE, no grad).

Latent scaling: latents fed to the diffusion process are scaled by the VAE
`scaling_factor`. Reconstructed clean latents are therefore divided by the same
factor before decoding back to RGB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


S1_MIN = 700   # Pose-dominant   : [700, 1000]
S2_MIN = 100   # Detail-dominant : [100, 700)
# S3              Quality-dominant: [0,   100)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Sobel edge magnitude. x: (B,3,H,W) in [-1,1] -> (B,1,H,W)."""
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def reconstruct_x0(
    noisy_latent: torch.Tensor,
    model_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler,
    prediction_type: str,
) -> torch.Tensor:
    """Recover the predicted clean (scaled) latent from the model output."""
    alphas = scheduler.alphas_cumprod.to(noisy_latent.device)[timestep]
    alpha = alphas.view(-1, 1, 1, 1, 1)
    sqrt_a = alpha.sqrt()
    sqrt_1ma = (1 - alpha).sqrt()
    if prediction_type == "v_prediction":
        # x0 = sqrt(a) * x_t - sqrt(1-a) * v
        return sqrt_a * noisy_latent - sqrt_1ma * model_pred
    # epsilon
    return (noisy_latent - sqrt_1ma * model_pred) / sqrt_a


def one_step_decode(
    noisy_latent: torch.Tensor,
    model_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler,
    vae,
    prediction_type: str,
    scaling_factor: float,
) -> torch.Tensor:
    """Reconstruct clean latent, unscale, decode middle frame -> RGB (B,3,H*8,W*8)."""
    with torch.no_grad():
        x0 = reconstruct_x0(noisy_latent, model_pred, timestep, scheduler, prediction_type)
        x0 = (x0 / scaling_factor).clamp(-6.0, 6.0)
        T = x0.shape[2]
        mid = T // 2
        z = x0[:, :, mid:mid + 1, :, :].float()
        rgb = vae.decode(z).sample.squeeze(2).clamp(-1.0, 1.0)
    return rgb


class PhDLoss(nn.Module):
    def __init__(self, lambda_pose=0.1, lambda_detail=0.1, lambda_low=0.1, lpips_net="vgg"):
        super().__init__()
        self.lambda_pose = lambda_pose
        self.lambda_detail = lambda_detail
        self.lambda_low = lambda_low
        self.lpips_fn = lpips.LPIPS(net=lpips_net)
        self.lpips_fn.requires_grad_(False)
        self.lpips_fn.eval()

    @staticmethod
    def _phase(t: int) -> str:
        if t >= S1_MIN:
            return "S1"
        if t >= S2_MIN:
            return "S2"
        return "S3"

    def forward(
        self,
        model_pred: torch.Tensor,    # (B,16,T,H,W) — DiT output
        target: torch.Tensor,        # (B,16,T,H,W) — scheduler regression target
        noisy_latent: torch.Tensor,  # (B,16,T,H,W) — scaled noisy latent
        timestep: torch.Tensor,      # (B,) — same t across batch
        target_rgb: torch.Tensor,    # (B,3,H*8,W*8) — decoded clean middle frame
        scheduler,
        vae,
        prediction_type: str,
        scaling_factor: float,
    ):
        device = model_pred.device
        t_val = int(timestep.reshape(-1)[0].item())
        phase = self._phase(t_val)

        Llatent = F.mse_loss(model_pred, target)
        log_dict = {"Llatent": Llatent.item(), "phase": phase, "t": t_val}

        if phase in ("S1", "S2"):
            pred_rgb = one_step_decode(noisy_latent, model_pred, timestep,
                                       scheduler, vae, prediction_type, scaling_factor).to(device)
            Laux = F.mse_loss(sobel_edges(pred_rgb), sobel_edges(target_rgb.to(device)))
            lam = self.lambda_pose if phase == "S1" else self.lambda_detail
            total_loss = Llatent + lam * Laux
            log_dict["Lpose" if phase == "S1" else "Ldetail"] = Laux.item()
        else:
            pred_rgb = one_step_decode(noisy_latent, model_pred, timestep,
                                       scheduler, vae, prediction_type, scaling_factor).to(device)
            self.lpips_fn = self.lpips_fn.to(device)
            Llow = self.lpips_fn(pred_rgb.float(), target_rgb.to(device).float()).mean()
            total_loss = Llatent + self.lambda_low * Llow
            log_dict["Llow"] = Llow.item()

        log_dict["total_loss"] = total_loss.item()
        return total_loss, log_dict
