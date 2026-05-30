import cv2
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# CONFIG – change VIDEO_NAME to your video (without .mp4)
# ------------------------------------------------------------
VIDEO_NAME = "videovideo1844_bqbWi7VMraY-scene9-scene5"
POSE_ROOT = Path("processed_dataset_wholebody/pose_data")
VIDEO_PATH = Path("processed_dataset_wholebody/final_videos") / f"{VIDEO_NAME}.mp4"
OUTPUT_VIDEO_PATH = Path("processed_dataset_wholebody/visualizations") / f"{VIDEO_NAME}_skeleton.mp4"

# ------------------------------------------------------------
# COCO SKELETON (17 keypoints, indices 0-16) – matches DWPose
# ------------------------------------------------------------
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # nose, eyes, ears
    (5, 6),                                # shoulders
    (5, 7), (7, 9),                       # left arm
    (6, 8), (8, 10),                     # right arm
    (5, 11), (6, 12),                   # hips
    (11, 13), (13, 15),                # left leg
    (12, 14), (14, 16)                # right leg
]

# Colors (BGR)
BODY_COLOR = (0, 255, 0)      # green
FACE_COLOR = (255, 0, 0)      # blue
HAND_COLOR = (0, 255, 255)    # yellow
SKELETON_COLOR = (0, 255, 0)  # green

# ------------------------------------------------------------
def draw_skeleton(frame, keypoints, conf_thresh=0.3):
    """
    Draw COCO skeleton on frame.
    keypoints: numpy array of shape (17, 3) – pixel coordinates + confidence
    """
    for a, b in COCO_SKELETON:
        if a < len(keypoints) and b < len(keypoints):
            if keypoints[a, 2] > conf_thresh and keypoints[b, 2] > conf_thresh:
                x1, y1 = int(keypoints[a, 0]), int(keypoints[a, 1])
                x2, y2 = int(keypoints[b, 0]), int(keypoints[b, 1])
                cv2.line(frame, (x1, y1), (x2, y2), SKELETON_COLOR, 2)

# ------------------------------------------------------------
def main():
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    pose_dir = POSE_ROOT / VIDEO_NAME
    if not pose_dir.exists():
        print(f"❌ Pose directory not found: {pose_dir}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_file = pose_dir / f"{frame_idx:06d}.npy"
        if not pose_file.exists():
            print(f"⚠️ Pose file missing: {pose_file}")
            frame_idx += 1
            continue

        data = np.load(pose_file, allow_pickle=True).item()

        # --- Draw body keypoints and skeleton ---
        bodies = data['bodies']
        if bodies['candidate'].size > 0:
            body_kps = bodies['candidate']  # should be (17, 3) in pixel coordinates

            # DEBUG: print first few keypoints (run on first frame only)
            if frame_idx == 0:
                print("\n=== BODY KEYPOINTS (first 5) ===")
                for i in range(min(5, len(body_kps))):
                    print(f"  {i}: ({body_kps[i,0]:.1f}, {body_kps[i,1]:.1f}), conf={body_kps[i,2]:.2f}")

            draw_skeleton(frame, body_kps)
            for i, (x, y, conf) in enumerate(body_kps):
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, BODY_COLOR, -1)
                    cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, BODY_COLOR, 1)

        # --- Draw face keypoints ---
        faces = data['faces']
        if faces and len(faces) > 0 and faces[0] is not None:
            face_kps = faces[0]  # (68, 2) or (68, 3)
            for pt in face_kps[:, :2]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, FACE_COLOR, -1)

        # --- Draw hand keypoints ---
        hands = data['hands']
        if isinstance(hands, np.ndarray) and hands.size > 0 and hands.shape[0] >= 2:
            left_hand, right_hand = hands[0], hands[1]
            for pt in left_hand[:, :2]:
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, HAND_COLOR, -1)
            for pt in right_hand[:, :2]:
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, HAND_COLOR, -1)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"✅ Visualization saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()