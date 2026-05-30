"""
training/dataset.py

TalkingHeadsDataset — loads all preprocessed .npy files and raw audio
and serves them as clean batches to the training loop.

Directory structure expected:
    DATASET/processed_dataset_wholebody/
        audio/                  ← .wav files
        video_latents_final_videos/  ← .npy  shape (1, 16, T, 96, 96)
        ref_latents/            ← .npy  shape (1, 16, 1, 96, 96)
        pose_data_single/       ← .npy  list of dicts {frame_index, keypoints(133,2), scores}

APDH Pose Dropout (applied per sample at runtime):
    Stage 1 → full pose          (all 133 keypoints)
    Stage 2 → no lips            (zero out indices 0-19)
    Stage 3 → no head            (zero out indices 0-67)
    Stage 4 → hands only         (zero out everything except 91-132)

Pose keypoint index groups (DWPose wholebody 133):
    0  - 19  : face / lips region
    20 - 67  : head / facial landmarks
    68 - 90  : body skeleton
    91 - 132 : hands (left + right)
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
from typing import Optional


# ── APDH Stage Definitions ─────────────────────────────────────────────────

POSE_GROUPS = {
    "lips"  : list(range(0,  20)),
    "head"  : list(range(0,  68)),
    "body"  : list(range(68, 91)),
    "hands" : list(range(91, 133)),
}

# What keypoints to ZERO OUT per stage
APDH_DROPOUT = {
    1: [],                          # Stage 1: full pose, nothing dropped
    2: POSE_GROUPS["lips"],         # Stage 2: drop lips
    3: POSE_GROUPS["head"],         # Stage 3: drop head (includes lips)
    4: POSE_GROUPS["lips"] +        # Stage 4: hands only — drop everything except hands
       POSE_GROUPS["head"] +
       POSE_GROUPS["body"],
}


def apply_pose_dropout(
    keypoints: torch.Tensor,
    stage: int,
    iterative_prob: float = 0.0,
) -> torch.Tensor:
    """
    Apply APDH spatial pose dropout to keypoints.

    Args:
        keypoints      : (T, 133, 2) normalised keypoints
        stage          : APDH stage 1-4
        iterative_prob : probability of dropping the entire pose
                         (iterative level dropout, increases over training)

    Returns:
        keypoints with dropped indices zeroed out : (T, 133, 2)
    """
    kp = keypoints.clone()

    # Iterative-level dropout: zero entire pose with some probability
    if iterative_prob > 0.0 and torch.rand(1).item() < iterative_prob:
        return torch.zeros_like(kp)

    # Spatial-level dropout: zero specific keypoint groups
    drop_indices = APDH_DROPOUT.get(stage, [])
    if drop_indices:
        kp[:, drop_indices, :] = 0.0

    return kp


# ── Dataset ────────────────────────────────────────────────────────────────

class TalkingHeadsDataset(Dataset):
    """
    Loads preprocessed latents, pose, and audio for one training sample.

    Each sample is matched by stem name:
        audio/001.wav
        video_latents_final_videos/001.npy
        ref_latents/001.npy
        pose_data_single/001.npy

    Args:
        root_dir       : path to processed_dataset_wholebody/
        clip_frames    : number of frames per training clip (default 24 = 1s)
        audio_sr       : sample rate for audio loading (default 16000 for Wav2Vec2)
        apdh_stage     : current APDH stage (1-4), controls pose dropout
        iterative_prob : iterative pose dropout probability (grows during training)
        normalize_kp   : normalise keypoints to [0,1] using 768px canvas
    """

    CANVAS = 768.0   # keypoints were mapped to 768×768 space
    TEMPORAL_COMPRESSION = 4   # CogVideoX VAE: 1 latent frame per 4 video frames
    VIDEO_FPS = 24

    def __init__(
        self,
        root_dir       : str,
        clip_frames    : int   = 13,   # LATENT frames
        audio_sr       : int   = 16000,
        apdh_stage     : int   = 1,
        iterative_prob : float = 0.0,
        normalize_kp   : bool  = True,
    ):
        super().__init__()
        self.root          = Path(root_dir)
        self.clip_frames   = clip_frames
        self.audio_sr      = audio_sr
        self.apdh_stage    = apdh_stage
        self.iterative_prob = iterative_prob
        self.normalize_kp  = normalize_kp
        # Directories
        self.video_lat_dir = self.root / "video_latents_final_videos"
        self.ref_lat_dir   = self.root / "ref_latents"
        self.pose_dir      = self.root / "pose_data_single"
        self.audio_dir     = self.root / "audio"

        # Build sample list — only keep stems that have ALL four modalities
        self.samples = self._build_sample_list()
        print(f"[Dataset] Found {len(self.samples)} valid samples.")

    def _build_sample_list(self) -> list[str]:
        stems = []
        for vid_path in sorted(self.video_lat_dir.glob("*.npy")):
            stem = vid_path.stem
            if (
                (self.ref_lat_dir  / f"{stem}.npy").exists() and
                (self.pose_dir     / f"{stem}.npy").exists() and
                (self.audio_dir    / f"{stem}.m4a").exists()
            ):
                stems.append(stem)
            else:
                print(f"[Dataset] Skipping {stem} - missing modality.")
        return stems

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        stem = self.samples[idx]
        TC = self.TEMPORAL_COMPRESSION

        # ── 1. Video latents (clip in LATENT frames) ───────────────────────
        vid_lat = np.load(self.video_lat_dir / f"{stem}.npy")
        vid_lat = torch.from_numpy(vid_lat).squeeze(0).float()  # (16, T_lat, 96, 96)
        T_total = vid_lat.shape[1]

        start = torch.randint(0, max(1, T_total - self.clip_frames), (1,)).item()
        vid_lat = vid_lat[:, start:start + self.clip_frames, :, :]
        if vid_lat.shape[1] < self.clip_frames:
            pad = self.clip_frames - vid_lat.shape[1]
            vid_lat = torch.nn.functional.pad(vid_lat, (0, 0, 0, 0, 0, pad))

        # ── 2. Reference latents ───────────────────────────────────────────
        ref_lat = np.load(self.ref_lat_dir / f"{stem}.npy")
        ref_lat = torch.from_numpy(ref_lat).squeeze(0).float()  # (16, 1, 96, 96)

        # ── 3. Pose keypoints — one per LATENT frame (sample at video idx i*TC)
        pose_data = np.load(self.pose_dir / f"{stem}.npy", allow_pickle=True)
        n_video_frames = len(pose_data)
        kp_frames = []
        for j in range(self.clip_frames):
            vf = min((start + j) * TC, n_video_frames - 1)
            kp = pose_data[vf]["keypoints"]
            if kp is None or len(kp) == 0:
                kp = np.zeros((133, 2), dtype=np.float32)
            kp_frames.append(np.asarray(kp, dtype=np.float32))
        keypoints = torch.from_numpy(np.stack(kp_frames))  # (clip_frames, 133, 2)

        if self.normalize_kp:
            keypoints = (keypoints / self.CANVAS).clamp(0.0, 1.0)

        keypoints = apply_pose_dropout(keypoints, self.apdh_stage, self.iterative_prob)

        # ── 4. Audio — slice the video-frame span the latent clip covers ───
        audio_path = self.audio_dir / f"{stem}.m4a"
        waveform, _ = librosa.load(str(audio_path), sr=self.audio_sr, mono=True)

        samples_per_frame = self.audio_sr / self.VIDEO_FPS
        video_start = start * TC
        n_video = (self.clip_frames - 1) * TC + 1
        audio_start = int(round(video_start * samples_per_frame))
        audio_len   = int(round(n_video * samples_per_frame))
        waveform    = waveform[audio_start:audio_start + audio_len]
        if len(waveform) < audio_len:
            waveform = np.pad(waveform, (0, audio_len - len(waveform)))

        waveform = torch.from_numpy(waveform).float()  # (T_audio,)

        return {
            "video_latents"  : vid_lat,    # (16, T, 96, 96)
            "ref_latents"    : ref_lat,    # (16, 1, 96, 96)
            "pose_keypoints" : keypoints,  # (T, 133, 2)
            "audio_waveform" : waveform,   # (T_audio,)
            "stem"           : stem,
        }

    def set_apdh_stage(self, stage: int, iterative_prob: float = 0.0):
        """Call this from the training loop when advancing APDH curriculum."""
        assert stage in [1, 2, 3, 4], "Stage must be 1-4"
        self.apdh_stage     = stage
        self.iterative_prob = iterative_prob
        print(f"[Dataset] APDH stage -> {stage} | iterative_prob -> {iterative_prob:.2f}")


# ── DataLoader factory ─────────────────────────────────────────────────────

def get_dataloader(
    root_dir       : str,
    batch_size     : int   = 1,
    clip_frames    : int   = 13,
    num_workers    : int   = 0,
    apdh_stage     : int   = 1,
    iterative_prob : float = 0.0,
    shuffle        : bool  = True,
) -> tuple[TalkingHeadsDataset, DataLoader]:

    dataset = TalkingHeadsDataset(
        root_dir       = root_dir,
        clip_frames    = clip_frames,
        apdh_stage     = apdh_stage,
        iterative_prob = iterative_prob,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,    # keeps batch size consistent
    )

    return dataset, loader


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "../DATASET2/DATASET"

    dataset, loader = get_dataloader(
        root_dir    = root,
        batch_size  = 2,
        clip_frames = 13,
        num_workers = 0,    # 0 for Windows
        apdh_stage  = 1,
    )

    batch = next(iter(loader))

    print(f"video_latents  : {tuple(batch['video_latents'].shape)}")
    print(f"ref_latents    : {tuple(batch['ref_latents'].shape)}")
    print(f"pose_keypoints : {tuple(batch['pose_keypoints'].shape)}")
    print(f"audio_waveform : {tuple(batch['audio_waveform'].shape)}")

    # Test APDH stage progression
    dataset.set_apdh_stage(4, iterative_prob=0.2)
    batch2 = next(iter(loader))
    hands_only = batch2["pose_keypoints"][:, :, :68, :]  # head region
    assert hands_only.sum() == 0.0, "Head keypoints should be zeroed in stage 4"
    print("\n[PASS] APDH stage 4 dropout correct")
    print("\nDataset ready.")