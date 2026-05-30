"""
Src/infer.py — generate a talking-head video from a reference image + audio.

Mirrors training exactly so the trained weights behave consistently:
  * Reference image -> 2B VAE latent, scaled by vae.config.scaling_factor.
  * Denoising runs in the SAME scaled-latent space the model was trained in,
    with the CogVideoX v-prediction scheduler (scheduler.step handles the
    parameterization). The final clean latent is unscaled before decoding.
  * Audio is encoded at video-frame resolution; AudioProjection resamples it to
    the fixed encoder length internally.
  * Pose is optional. With no pose the motion is audio-driven only.

Usage (run from Src/):
  python infer.py \
      --checkpoint ../checkpoints/checkpoint_010000/checkpoint.pt \
      --ref_image  path/to/reference.jpg \
      --audio      path/to/speech.wav \
      --output     ../result.mp4 \
      --num_frames 49 \
      --steps 50
  # optional: --pose ../DATASET2/DATASET/pose_data_single/000.npy
  #           --guidance 1.0   (audio CFG; 1.0 = off, matches training)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import librosa

from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler

from Models.talking_heads_dit import TalkingHeadsDiT
from training.train import build_audio_encoder_with_lora

SIZE = 768
FPS = 24
TEMPORAL_COMPRESSION = 4
CANVAS = 768.0


def resize_and_pad(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Aspect-preserving letterbox to size x size (RGB in, RGB out)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def image_to_tensor(img_rgb: np.ndarray, device, dtype) -> torch.Tensor:
    """HxWx3 RGB uint8 -> (1, 3, 1, SIZE, SIZE) in [-1, 1]."""
    img = resize_and_pad(img_rgb)
    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(1).unsqueeze(0).to(device, dtype)   # (1,3,1,H,W)


def load_pose(pose_path: Path, t_lat: int, device, dtype) -> torch.Tensor:
    """pose_data_single .npy -> (1, t_lat, 133, 2) normalised, sampled at i*TC."""
    pose_data = np.load(pose_path, allow_pickle=True)
    n = len(pose_data)
    frames = []
    for j in range(t_lat):
        vf = min(j * TEMPORAL_COMPRESSION, n - 1)
        kp = pose_data[vf]["keypoints"]
        if kp is None or len(kp) == 0:
            kp = np.zeros((133, 2), dtype=np.float32)
        frames.append(np.asarray(kp, dtype=np.float32))
    kp = torch.from_numpy(np.stack(frames)) / CANVAS
    return kp.clamp(0, 1).unsqueeze(0).to(device, dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ref_image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", default="../result.mp4")
    ap.add_argument("--pretrained_model", default="THUDM/CogVideoX-2b")
    ap.add_argument("--wav2vec2_model", default="facebook/wav2vec2-base")
    ap.add_argument("--audio_dim", type=int, default=768)
    ap.add_argument("--num_frames", type=int, default=49, help="video frames to generate")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.0, help="audio CFG; 1.0 = off")
    ap.add_argument("--pose", default=None)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # latent frame count for the requested number of video frames
    t_lat = (args.num_frames - 1) // TEMPORAL_COMPRESSION + 1
    n_video = (t_lat - 1) * TEMPORAL_COMPRESSION + 1
    print(f"[infer] {args.num_frames} req -> {t_lat} latent frames -> {n_video} video frames")

    # ── models ──
    print("[infer] loading DiT...")
    dit = TalkingHeadsDiT.from_pretrained_cogvideox(
        args.pretrained_model, freeze_backbone=True, gradient_checkpointing=False,
        use_lora=True, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        audio_input_dim=args.audio_dim,
    ).to(device, dtype=dtype).eval()

    print("[infer] loading audio encoder...")
    audio_enc = build_audio_encoder_with_lora(args.wav2vec2_model, args.audio_dim).to(device).eval()

    print("[infer] loading VAE + scheduler...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    vae.enable_tiling()
    scaling_factor = float(vae.config.scaling_factor)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # ── load trained weights ──
    print(f"[infer] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    miss_d, _ = dit.load_state_dict(ckpt["dit_trainable"], strict=False)
    audio_enc.load_state_dict(ckpt["audio_trainable"], strict=False)
    print(f"[infer] loaded {len(ckpt['dit_trainable'])} DiT + "
          f"{len(ckpt['audio_trainable'])} audio tensors")

    # ── conditioning ──
    ref_bgr = cv2.imread(args.ref_image)
    if ref_bgr is None:
        raise FileNotFoundError(args.ref_image)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_img = image_to_tensor(ref_rgb, device, torch.float32)            # (1,3,1,H,W)
    ref_latent_raw = vae.encode(ref_img).latent_dist.sample()            # (1,16,1,96,96)
    ref = (ref_latent_raw * scaling_factor).to(dtype)

    waveform, _ = librosa.load(args.audio, sr=16000, mono=True)
    waveform = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
        audio_embeds = audio_enc(waveform, target_frames=n_video)        # (1,n_video,audio_dim)
    # Unconditional branch for audio CFG = NO audio (matches training's muted
    # state), so the adapters are skipped entirely rather than fed zeros.
    audio_uncond = None

    pose = None
    if args.pose:
        pose = load_pose(Path(args.pose), t_lat, device, dtype)
        print(f"[infer] pose conditioning: {tuple(pose.shape)}")

    # ── denoising loop (scaled-latent space, v-prediction) ──
    _, c, _, h, w = ref.shape
    latent = torch.randn((1, c, t_lat, h, w), generator=gen, device=device, dtype=dtype)
    latent = latent * scheduler.init_noise_sigma
    scheduler.set_timesteps(args.steps, device=device)

    print(f"[infer] sampling {args.steps} steps...")
    for t in scheduler.timesteps:
        t_b = t.reshape(1).expand(1).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            pred = dit(video_latents=latent, ref_latents=ref, timestep=t_b,
                       audio_embeds=audio_embeds, pose_keypoints=pose)
            if args.guidance != 1.0:
                pred_u = dit(video_latents=latent, ref_latents=ref, timestep=t_b,
                             audio_embeds=audio_uncond, pose_keypoints=pose)
                pred = pred_u + args.guidance * (pred - pred_u)
        latent = scheduler.step(pred.float(), t, latent.float()).prev_sample.to(dtype)

    # ── decode ──
    print("[infer] decoding...")
    z0 = (latent.float() / scaling_factor)
    frames = vae.decode(z0).sample                                       # (1,3,T_video,H,W)
    frames = frames.squeeze(0).permute(1, 2, 3, 0).clamp(-1, 1)          # (T,H,W,3)
    frames = ((frames + 1.0) * 127.5).round().byte().cpu().numpy()

    # ── write video (+ audio mux if possible) ──
    out_path = Path(args.output)
    tmp = out_path.with_name(out_path.stem + "_noaudio.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (frames.shape[2], frames.shape[1]))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    try:
        from moviepy import VideoFileClip, AudioFileClip
        v = VideoFileClip(str(tmp))
        a = AudioFileClip(args.audio).subclipped(0, v.duration)
        v.with_audio(a).write_videofile(str(out_path), codec="libx264",
                                        audio_codec="aac", fps=FPS, logger=None)
        v.close(); a.close()
        tmp.unlink(missing_ok=True)
    except Exception as e:
        print(f"[infer] audio mux skipped ({e}); video at {tmp}")
        out_path = tmp

    print(f"[infer] done -> {out_path}")


if __name__ == "__main__":
    main()
