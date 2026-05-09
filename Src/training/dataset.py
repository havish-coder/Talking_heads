"""
training/dataset.py

TalkingHeadsDataset — Coordinates the loading of preprocessed .npy files and raw audio,
serving them as synchronized batches to the training loop.

Directory structure expected:
    DATASET/processed_dataset_wholebody/
        audio/                        <- .m4a files
        video_latents_final_videos/   <- .npy  shape (1, 16, T, 96, 96)
        ref_latents/                  <- .npy  shape (1, 16, 1, 96, 96)
        pose_data_single/             <- .npy  list of dicts {frame_index, keypoints(133,2), scores}

APDH Pose Dropout (applied per sample dynamically at runtime):
    Stage 1 -> Full pose          (all 133 keypoints active)
    Stage 2 -> No lips            (zero out indices 0-19)
    Stage 3 -> No head            (zero out indices 0-67)
    Stage 4 -> Hands only         (zero out everything except 91-132)

Pose keypoint index groups (DWPose wholebody 133 format):
    0  - 19  : Face / lips region
    20 - 67  : Head / facial landmarks
    68 - 90  : Body skeleton
    91 - 132 : Hands (left + right)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Suppress librosa warnings for cleaner console output during training
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# --- APDH Curriculum Configurations ---

POSE_GROUPS = {
    "lips": list(range(0, 20)),
    "head": list(range(0, 68)),
    "body": list(range(68, 91)),
    "hands": list(range(91, 133)),
}

# Defines which keypoints are ZEROED OUT for each stage.
APDH_DROPOUT = {
    1: [],                                  # Stage 1: Full pose, nothing dropped
    2: POSE_GROUPS["lips"],                 # Stage 2: Model learns to rely on audio for lip sync
    3: POSE_GROUPS["head"],                 # Stage 3: Drops head to force reliance on reference frame
    4: POSE_GROUPS["lips"] + POSE_GROUPS["head"] + POSE_GROUPS["body"], # Stage 4: Hands only
}


def apply_pose_dropout(
    keypoints: torch.Tensor,
    stage: int,
    iterative_prob: float = 0.0,
) -> torch.Tensor:
    """
    Applies APDH spatial pose dropout to keypoints dynamically.

    Args:
        keypoints: Tensor of shape (Time, 133, 2) containing normalized keypoints.
        stage: Current APDH stage (1-4).
        iterative_prob: Probability of dropping the entire pose sequence.

    Returns:
        Tensor of shape (Time, 133, 2) with targeted indices zeroed out.
    """
    kp = keypoints.clone()

    # Iterative-level dropout: Zero out the entire pose randomly to build robustness
    if iterative_prob > 0.0 and torch.rand(1).item() < iterative_prob:
        return torch.zeros_like(kp)

    # Spatial-level dropout: Zero out specific keypoint regions based on the curriculum stage
    drop_indices = APDH_DROPOUT.get(stage, [])
    if drop_indices:
        kp[:, drop_indices, :] = 0.0

    return kp


class TalkingHeadsDataset(Dataset):
    """
    Loads synchronized preprocessed latents, pose, and audio for a single training sample.
    Dynamically applies temporal cropping, padding, and APDH dropout.
    """

    CANVAS_SIZE = 768.0  # Keypoints were originally mapped to a 768x768 pixel space

    def __init__(
        self,
        root_dir: str,
        clip_frames: int = 24,
        audio_sr: int = 16000,
        apdh_stage: int = 1,
        iterative_prob: float = 0.0,
        normalize_kp: bool = True,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.clip_frames = clip_frames
        self.audio_sr = audio_sr
        self.apdh_stage = apdh_stage
        self.iterative_prob = iterative_prob
        self.normalize_kp = normalize_kp

        # Validate and define paths
        self.video_lat_dir = self.root / "video_latents_final_videos"
        self.ref_lat_dir = self.root / "ref_latents"
        self.pose_dir = self.root / "pose_data_single"
        self.audio_dir = self.root / "audio"

        # Build valid sample list
        self.samples = self._build_sample_list()
        print(f"[SYSTEM] Dataset initialized. Found {len(self.samples)} valid samples.")

    def _build_sample_list(self) -> List[str]:
        """Scans directories and keeps only samples that contain all required modalities."""
        stems = []
        for vid_path in sorted(self.video_lat_dir.glob("*.npy")):
            stem = vid_path.stem
            if (
                (self.ref_lat_dir / f"{stem}.npy").exists() and
                (self.pose_dir / f"{stem}.npy").exists() and
                (self.audio_dir / f"{stem}.m4a").exists()
            ):
                stems.append(stem)
            else:
                print(f"[WARN] Skipping {stem} — Missing one or more modalities.")
        return stems

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.samples[idx]

        # --- 1. Video Latents ---
        vid_lat = np.load(self.video_lat_dir / f"{stem}.npy")
        vid_lat = torch.from_numpy(vid_lat).squeeze(0).float()  # (16, Time, 96, 96)
        total_frames = vid_lat.shape[1]

        # Extract a random temporal crop of length `clip_frames`
        start_frame = torch.randint(0, max(1, total_frames - self.clip_frames), (1,)).item()
        vid_lat = vid_lat[:, start_frame:start_frame + self.clip_frames, :, :]

        # Pad temporally if the video is shorter than `clip_frames`
        if vid_lat.shape[1] < self.clip_frames:
            pad_amount = self.clip_frames - vid_lat.shape[1]
            vid_lat = torch.nn.functional.pad(vid_lat, (0, 0, 0, 0, 0, pad_amount))

        # --- 2. Reference Latents ---
        ref_lat = np.load(self.ref_lat_dir / f"{stem}.npy")
        ref_lat = torch.from_numpy(ref_lat).squeeze(0).float()  # (16, 1, 96, 96)

        # --- 3. Pose Keypoints ---
        pose_data = np.load(self.pose_dir / f"{stem}.npy", allow_pickle=True)
        kp_frames = []
        
        # Load keypoints matching the video's temporal crop
        for frame_dict in pose_data[start_frame:start_frame + self.clip_frames]:
            kp = frame_dict.get("keypoints")
            if kp is None or len(kp) == 0:
                kp = np.zeros((133, 2), dtype=np.float32)
            kp_frames.append(kp.astype(np.float32))

        # Pad keypoints if necessary
        while len(kp_frames) < self.clip_frames:
            kp_frames.append(np.zeros((133, 2), dtype=np.float32))

        keypoints = torch.from_numpy(np.stack(kp_frames))  # (Time, 133, 2)

        if self.normalize_kp:
            keypoints = keypoints / self.CANVAS_SIZE
            keypoints = keypoints.clamp(0.0, 1.0)

        # Apply spatial/iterative dropout based on the current training stage
        keypoints = apply_pose_dropout(keypoints, self.apdh_stage, self.iterative_prob)

        # --- 4. Audio Waveform ---
        audio_path = self.audio_dir / f"{stem}.m4a"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sr = librosa.load(str(audio_path), sr=self.audio_sr, mono=True)

        # Align audio segment with the video temporal crop
        samples_per_frame = self.audio_sr / 24.0
        audio_start = int(start_frame * samples_per_frame)
        audio_len = int(self.clip_frames * samples_per_frame)
        waveform = waveform[audio_start:audio_start + audio_len]

        # Pad audio if necessary
        if len(waveform) < audio_len:
            waveform = np.pad(waveform, (0, audio_len - len(waveform)))

        waveform = torch.from_numpy(waveform).float()  # (T_audio,)

        return {
            "video_latents": vid_lat,      # (16, T, 96, 96)
            "ref_latents": ref_lat,        # (16, 1, 96, 96)
            "pose_keypoints": keypoints,   # (T, 133, 2)
            "audio_waveform": waveform,    # (T_audio,)
            "stem": stem,
        }

    def set_apdh_stage(self, stage: int, iterative_prob: float = 0.0) -> None:
        """Dynamically updates the APDH curriculum stage during the training loop."""
        if stage not in [1, 2, 3, 4]:
            raise ValueError("APDH Stage must be an integer between 1 and 4.")
        self.apdh_stage = stage
        self.iterative_prob = iterative_prob
        print(f"[INFO] Curriculum Updated -> APDH Stage: {stage} | Iterative Dropout Prob: {iterative_prob:.2f}")


def get_dataloader(
    root_dir: str,
    batch_size: int = 4,
    clip_frames: int = 24,
    num_workers: int = 4,
    apdh_stage: int = 1,
    iterative_prob: float = 0.0,
    shuffle: bool = True,
) -> Tuple[TalkingHeadsDataset, DataLoader]:
    """Factory function to initialize the Dataset and DataLoader seamlessly."""
    dataset = TalkingHeadsDataset(
        root_dir=root_dir,
        clip_frames=clip_frames,
        apdh_stage=apdh_stage,
        iterative_prob=iterative_prob,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Ensures consistent batch sizing
    )

    return dataset, loader


# --- Integration Smoke Test ---
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "DATASET/processed_dataset_wholebody"

    print("[SYSTEM] Running Dataset Smoke Test...")
    dataset, loader = get_dataloader(
        root_dir=root,
        batch_size=2,
        clip_frames=24,
        num_workers=0,  # Set to 0 for cross-platform compatibility during tests
        apdh_stage=1,
    )

    batch = next(iter(loader))

    print(f"[INFO] Batch Tensor Shapes:")
    print(f"  -> video_latents  : {tuple(batch['video_latents'].shape)}")
    print(f"  -> ref_latents    : {tuple(batch['ref_latents'].shape)}")
    print(f"  -> pose_keypoints : {tuple(batch['pose_keypoints'].shape)}")
    print(f"  -> audio_waveform : {tuple(batch['audio_waveform'].shape)}")

    # Test APDH curriculum progression
    dataset.set_apdh_stage(4, iterative_prob=0.2)
    batch_stage4 = next(iter(loader))
    
    # Assert that the head region (indices 0-67) has been completely dropped out
    head_region = batch_stage4["pose_keypoints"][:, :, :68, :]
    assert head_region.sum().item() == 0.0, "Head keypoints failed to zero-out in stage 4."
    
    print("[TEST] APDH Stage 4 dropout logic validated successfully.")
    print("[SYSTEM] Dataset is ready for training.")