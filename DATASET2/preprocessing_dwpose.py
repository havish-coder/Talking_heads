import cv2
import gc
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from moviepy import VideoFileClip
import torch
from rtmlib import Wholebody

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_VIDEO_DIR = SCRIPT_DIR / "talkvid_sample"
OUTPUT_ROOT   = SCRIPT_DIR / "processed_dataset_wholebody"

PROCESSED_VIDEOS_DIR = OUTPUT_ROOT / "videos_24fps"
POSE_OUTPUT_DIR      = OUTPUT_ROOT / "pose_data_single"
FINAL_VIDEO_DIR      = OUTPUT_ROOT / "final_videos"
MODEL_DIR            = SCRIPT_DIR  / "dwpose_models"

TARGET_SIZE      = 768
TARGET_FPS       = 24
CHECKPOINT_EVERY = 500

# ==========================================
# START INDEX — change this to resume from
# any video. 16 = skip first 16, start 17th
# ==========================================
START_FROM = 99

warnings.filterwarnings("ignore")

# ==========================================
# MODEL LOADING
# ==========================================
device = 'cuda'
print(f"PyTorch CUDA   : {torch.cuda.is_available()}")
print(f"Forcing device : {device} (onnxruntime-gpu handles this)")
print("Loading DWPose model...")

wholebody = Wholebody(
    det=str(MODEL_DIR / "yolox_l.onnx"),
    pose=str(MODEL_DIR / "dw-ll_ucoco_384.onnx"),
    to_openpose=False,
    backend='onnxruntime',
    device=device
)
print("Model loaded.\n")

# ==========================================
# FPS CONVERSION
# ==========================================
def convert_fps(src_path: Path, tgt_path: Path) -> bool:
    if tgt_path.exists():
        return True
    try:
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        clip = VideoFileClip(str(src_path))
        clip.write_videofile(
            str(tgt_path),
            fps=TARGET_FPS,
            codec='libx264',
            audio=False,
            logger=None
        )
        clip.close()
        del clip
        gc.collect()
        return True
    except Exception as e:
        print(f"  FPS conversion failed: {e}")
        return False

# ==========================================
# RESIZE + PAD
# ==========================================
def resize_and_pad(img: np.ndarray):
    """
    Resize keeping aspect ratio, pad to TARGET_SIZE x TARGET_SIZE.
    Returns:
        padded     : (TARGET_SIZE, TARGET_SIZE, 3)
        pad_params : (top, left, new_h, new_w, orig_h, orig_w)
    """
    orig_h, orig_w = img.shape[:2]
    scale  = TARGET_SIZE / max(orig_h, orig_w)
    new_h  = int(orig_h * scale)
    new_w  = int(orig_w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    top    = (TARGET_SIZE - new_h) // 2
    bottom = TARGET_SIZE - new_h - top
    left   = (TARGET_SIZE - new_w) // 2
    right  = TARGET_SIZE - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, (top, left, new_h, new_w, orig_h, orig_w)

# ==========================================
# KEYPOINT REMAPPING
# ==========================================
def map_keypoints_to_padded(
    keypoints: np.ndarray,
    pad_params: tuple
) -> np.ndarray:
    """
    Remap raw rtmlib keypoints from original frame space
    into padded 768x768 coordinate space.

    133-keypoint layout:
        0-16   : body       (17 pts)
        17-22  : feet       (6 pts)
        23-90  : face       (68 pts)
        91-111 : left hand  (21 pts)
        112-132: right hand (21 pts)
    """
    top, left, new_h, new_w, orig_h, orig_w = pad_params

    if keypoints is None or (
        isinstance(keypoints, np.ndarray) and keypoints.size == 0
    ):
        return np.zeros((133, 2), dtype=np.float32)

    kps = keypoints[0].copy().astype(np.float32)
    kps[:, 0] = kps[:, 0] * (new_w / orig_w) + left
    kps[:, 1] = kps[:, 1] * (new_h / orig_h) + top
    return kps

# ==========================================
# EMPTY FRAME FALLBACK
# ==========================================
def empty_frame(frame_idx: int, pad_params: tuple) -> dict:
    return {
        "frame_index": frame_idx,
        "keypoints"  : np.zeros((133, 2), dtype=np.float32),
        "scores"     : np.zeros(133,      dtype=np.float32),
        "pad_params" : pad_params,
    }

# ==========================================
# GET TOTAL FRAME COUNT (without loading)
# ==========================================
def get_frame_count(video_path: Path) -> int:
    cap   = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

# ==========================================
# PROCESS ONE VIDEO — FRAME BY FRAME
# ==========================================
def process_video(video_file: Path) -> str:
    """
    Full pipeline for a single video.
    Returns: 'completed', 'skipped', or 'failed'
    """
    std_path         = PROCESSED_VIDEOS_DIR / f"{video_file.stem}_24fps.mp4"
    final_path       = FINAL_VIDEO_DIR      / f"{video_file.stem}.mp4"
    pose_output_path = POSE_OUTPUT_DIR      / f"{video_file.stem}.npy"
    done_flag        = POSE_OUTPUT_DIR      / f"{video_file.stem}.done"

    FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # skip if already completed
    if done_flag.exists():
        return 'skipped'

    # Step 1: FPS conversion
    if not convert_fps(video_file, std_path):
        return 'failed'

    # Step 2: Get total frame count (no frames loaded yet)
    total_frames = get_frame_count(std_path)
    if total_frames == 0:
        print(f"  No frames found in {video_file.name}")
        return 'failed'

    # Step 3: Check video opens correctly
    cap = cv2.VideoCapture(str(std_path))
    if not cap.isOpened():
        print(f"  Cannot open {std_path}")
        return 'failed'
    cap.release()

    # Step 4: Open for processing
    cap = cv2.VideoCapture(str(std_path))

    # Step 5: Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(
        str(final_path), fourcc, TARGET_FPS,
        (TARGET_SIZE, TARGET_SIZE)
    )

    # Step 6: Frame-by-frame — one frame in RAM at a time
    all_pose_data = []
    failed_frames = 0
    i = 0

    with tqdm(total=total_frames, desc=f"  {video_file.stem}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # resize + pad → write output video
            padded_frame, pad_params = resize_and_pad(frame)
            out.write(padded_frame)

            # pose on original resolution frame
            try:
                keypoints, scores = wholebody(frame)

                mapped_kps = map_keypoints_to_padded(keypoints, pad_params)

                frame_scores = (
                    scores[0].astype(np.float32)
                    if scores is not None and len(scores) > 0
                    else np.zeros(133, dtype=np.float32)
                )

                pose_frame = {
                    "frame_index": i,
                    "keypoints"  : mapped_kps,
                    "scores"     : frame_scores,
                    "pad_params" : pad_params,
                }

            except Exception as e:
                tqdm.write(f"  Pose failed frame {i} [{video_file.stem}]: {e}")
                pose_frame = empty_frame(i, pad_params)
                failed_frames += 1

            all_pose_data.append(pose_frame)

            # free frame memory immediately
            del frame
            del padded_frame

            # checkpoint every N frames
            if (i + 1) % CHECKPOINT_EVERY == 0:
                ckpt_path = POSE_OUTPUT_DIR / f"{video_file.stem}_ckpt_{i+1}.npy"
                np.save(ckpt_path, all_pose_data, allow_pickle=True)

            i += 1
            pbar.update(1)

    cap.release()
    out.release()
    gc.collect()

    # Step 7: Save final pose file
    np.save(pose_output_path, all_pose_data, allow_pickle=True)

    # free pose data from RAM
    del all_pose_data
    gc.collect()

    # mark as done
    done_flag.touch()

    # clean up checkpoint files
    for ckpt in POSE_OUTPUT_DIR.glob(f"{video_file.stem}_ckpt_*.npy"):
        ckpt.unlink()

    tqdm.write(
        f"  Done: {video_file.name} | "
        f"frames: {total_frames} | "
        f"failed: {failed_frames} ({failed_frames/total_frames*100:.1f}%)"
    )
    return 'completed'


# ==========================================
# MAIN — LOOP OVER ALL VIDEOS
# ==========================================
if __name__ == "__main__":

    all_videos   = sorted(RAW_VIDEO_DIR.glob("*.[mM][pP]4"))
    video_files  = all_videos[START_FROM:]   # skip already processed videos

    if not video_files:
        print(f"No MP4 files found in {RAW_VIDEO_DIR}")
        exit()

    total_videos = len(video_files)
    print(f"Total videos found : {len(all_videos)}")
    print(f"Starting from      : #{START_FROM + 1} ({video_files[0].name})")
    print(f"Remaining to process: {total_videos}")
    print(f"Output             : {OUTPUT_ROOT}\n")

    completed = 0
    skipped   = 0
    failed    = 0

    for video_file in tqdm(video_files, desc="Overall progress"):
        status = process_video(video_file)
        if status == 'completed':
            completed += 1
        elif status == 'skipped':
            skipped += 1
        else:
            failed += 1

    print(f"""
==========================================
ALL DONE
==========================================
  Total processed : {total_videos}
  Completed       : {completed}
  Skipped         : {skipped}  (already had .done flag)
  Failed          : {failed}

Output folder:
  {OUTPUT_ROOT}/
  ├── videos_24fps/      re-encoded at 24fps
  ├── final_videos/      padded to {TARGET_SIZE}x{TARGET_SIZE}
  └── pose_data_single/  one .npy per video

To verify any output:
  import numpy as np
  data = np.load('processed_dataset_wholebody/pose_data_single/VIDEO.npy', allow_pickle=True)
  print(len(data), data[0]['keypoints'].shape)   # e.g. (576, (133, 2))
""")