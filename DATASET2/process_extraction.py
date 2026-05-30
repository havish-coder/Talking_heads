import os
import cv2
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip
from PIL import Image
import torch
from dwpose import DwposeDetector

# ==========================================
#      CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_VIDEO_DIR = SCRIPT_DIR / "talkvid_sample"
OUTPUT_ROOT = SCRIPT_DIR / "processed_dataset_wholebody"

PROCESSED_VIDEOS_DIR = OUTPUT_ROOT / "videos_24fps"
POSE_OUTPUT_DIR = OUTPUT_ROOT / "pose_data"
FINAL_VIDEO_DIR = OUTPUT_ROOT / "final_videos"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"

TARGET_SIZE = 768
DETECT_RESOLUTION = 512
TARGET_FPS = 24

warnings.filterwarnings("ignore")

# ==========================================
#      MODEL LOADING
# ==========================================
print("🚀 Loading DWPose model...")
detector = DwposeDetector.from_pretrained_default()
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    detector = detector.cuda()
print(f"✅ Model loaded on {device}")

# ==========================================
#      UTILITIES
# ==========================================
def convert_fps(src_path, tgt_path, tgt_fps=TARGET_FPS):
    if tgt_path.exists():
        return True
    try:
        clip = VideoFileClip(str(src_path))
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        clip.write_videofile(
            str(tgt_path),
            fps=tgt_fps,
            codec='libx264',
            audio_codec='aac',
            logger=None,
            threads=4
        )
        clip.close()
        return True
    except Exception as e:
        print(f"FPS conversion failed: {e}")
        return False

def resize_and_pad_whole(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(0,0,0))
    pad_params = (top, left, new_h, new_w)
    return padded, pad_params

def save_audio(video_path, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        clip = AudioFileClip(str(video_path))
        clip.write_audiofile(str(save_dir / f"{video_path.stem}.wav"), logger=None)
        clip.close()
    except Exception as e:
        print(f"Audio extraction failed: {e}")

def map_coords_to_padded(coords_norm, pad_params, orig_h, orig_w):
    top, left, new_h, new_w = pad_params
    x_orig = coords_norm[:, 0] * orig_w
    y_orig = coords_norm[:, 1] * orig_h
    x_resized = x_orig * (new_w / orig_w)
    y_resized = y_orig * (new_h / orig_h)
    x_padded = x_resized + left
    y_padded = y_resized + top
    return np.stack([x_padded, y_padded], axis=1)

# ==========================================
#      POSE PARSING – ROBUST VERSION
# ==========================================
def parse_dwpose_output(pose_dict, pad_params, orig_h, orig_w):
    people = pose_dict.get('people', [])
    if not people:
        return None, None, None

    person = people[0]

    # ----- BODY -----
    pose_raw = person.get('pose_keypoints_2d', [])
    pose_arr = np.array(pose_raw).flatten()
    if pose_arr.size >= 54:
        pose_kps = pose_arr[:54].reshape(-1, 3)
        pose_kps[:, 0] /= orig_w
        pose_kps[:, 1] /= orig_h
    else:
        pose_kps = np.zeros((18, 3))

    body_xy = map_coords_to_padded(pose_kps[:, :2], pad_params, orig_h, orig_w)
    body_kps_padded = np.hstack([body_xy, pose_kps[:, 2:3]])
    bodies = {
        'candidate': body_kps_padded,
        'score': body_kps_padded[:, 2]
    }

    # ----- FACE -----
    face_raw = person.get('face_keypoints_2d', [])
    face_arr = np.array(face_raw).flatten()
    if face_arr.size == 136:
        face_kps = face_arr.reshape(-1, 2)
        face_kps[:, 0] /= orig_w
        face_kps[:, 1] /= orig_h
        face_xy = map_coords_to_padded(face_kps, pad_params, orig_h, orig_w)
        faces = [face_xy]
    else:
        faces = []

    # ----- HANDS -----
    # Left
    left_raw = person.get('hand_left_keypoints_2d', [])
    left_arr = np.array(left_raw).flatten()
    if left_arr.size == 42:
        left_kps = left_arr.reshape(-1, 2)
        left_kps[:, 0] /= orig_w
        left_kps[:, 1] /= orig_h
        left_xy = map_coords_to_padded(left_kps, pad_params, orig_h, orig_w)
    else:
        left_xy = np.array([])

    # Right
    right_raw = person.get('hand_right_keypoints_2d', [])
    right_arr = np.array(right_raw).flatten()
    if right_arr.size == 42:
        right_kps = right_arr.reshape(-1, 2)
        right_kps[:, 0] /= orig_w
        right_kps[:, 1] /= orig_h
        right_xy = map_coords_to_padded(right_kps, pad_params, orig_h, orig_w)
    else:
        right_xy = np.array([])

    if left_xy.size > 0 and right_xy.size > 0:
        hands = np.stack([left_xy, right_xy], axis=0)
    else:
        hands = np.array([])

    return bodies, faces, hands

# ==========================================
#      MAIN PROCESSING
# ==========================================
if __name__ == "__main__":
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Output will be saved to: {OUTPUT_ROOT}")

    video_files = list(RAW_VIDEO_DIR.glob("*.[mM][pP]4"))
    if not video_files:
        print(f"No MP4 files found in {RAW_VIDEO_DIR}")
        exit()

    for video_file in tqdm(video_files, desc="Processing videos"):
        std_path = PROCESSED_VIDEOS_DIR / f"{video_file.stem}_24fps.mp4"
        final_path = FINAL_VIDEO_DIR / f"{video_file.stem}.mp4"
        pose_dir = POSE_OUTPUT_DIR / video_file.stem

        FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        if not convert_fps(video_file, std_path):
            print(f"Skipping {video_file.name}: FPS conversion failed")
            continue
        save_audio(std_path, AUDIO_OUTPUT_DIR)

        # Read video frames
        cap = cv2.VideoCapture(str(std_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == 0:
            print(f"No frames in {video_file.name}")
            continue

        orig_h, orig_w = frames[0].shape[:2]

        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(final_path), fourcc, TARGET_FPS, (TARGET_SIZE, TARGET_SIZE))

        for i, frame in enumerate(tqdm(frames, desc=f"  {video_file.stem}", leave=False)):
            # Resize+pad whole frame
            final_frame, pad_params = resize_and_pad_whole(frame, TARGET_SIZE)
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

            # Run DWPose
            pil_img = Image.fromarray(frame)
            vis, pose_dict, _ = detector(
                pil_img,
                include_hand=True,
                include_face=True,
                include_body=True,
                image_and_json=True,
                detect_resolution=DETECT_RESOLUTION,
            )

            # Parse and map keypoints
            bodies, faces, hands = parse_dwpose_output(pose_dict, pad_params, orig_h, orig_w)

            # Save data
            pose_data = {
                'num': i,
                'pad_params': pad_params,
                'orig_size': (orig_h, orig_w),
                'bodies': bodies if bodies is not None else {'candidate': np.array([]), 'score': np.array([])},
                'faces': faces if faces is not None else [],
                'hands': hands if hands is not None else np.array([]),
            }
            np.save(str(pose_dir / f"{i:06d}.npy"), pose_data)

        out.release()
        print(f"✅ Saved: {final_path.name}")

    print("\n🎉 All done! Whole-body videos + pose keypoints saved.")