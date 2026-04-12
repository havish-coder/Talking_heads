import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

POSE_DIR  = "pose_data_single"   # .npy files
IMAGE_DIR = "reference_images_final_768"   # .jpg/.png files

# grab first pose file
pose_path = sorted(Path(POSE_DIR).glob("*.npy"))[0]
stem = pose_path.stem
print(f"Loading: {pose_path}")

# load pose data — each .npy is an object array of dicts (one per frame)
pose_data = np.load(pose_path, allow_pickle=True)

# handle 0-d wrapper
if pose_data.ndim == 0:
    pose_data = pose_data.item()

# if it's a single dict, wrap it in a list
if isinstance(pose_data, dict):
    pose_data = [pose_data]

# pick first frame
frame = pose_data[0]

if isinstance(frame, dict):
    keypoints  = np.array(frame["keypoints"])       # (J, 2)  x, y in padded-768 space
    scores     = np.array(frame.get("scores", np.ones(len(keypoints))))
    pad_params = frame.get("pad_params", None)      # (top, left, new_h, new_w, orig_h, orig_w)
    print(f"Frame 0 — keypoints shape: {keypoints.shape}, scores shape: {scores.shape}")
    if pad_params is not None:
        print(f"pad_params: top={pad_params[0]}, left={pad_params[1]}, "
              f"new_h={pad_params[2]}, new_w={pad_params[3]}, "
              f"orig_h={pad_params[4]}, orig_w={pad_params[5]}")
else:
    # fallback: treat as plain numeric array
    keypoints = np.array(frame)
    if keypoints.ndim == 1 and keypoints.shape[0] % 3 == 0:
        keypoints = keypoints.reshape(-1, 3)
    scores = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(len(keypoints))
    keypoints = keypoints[:, :2]
    pad_params = None
    print(f"Frame 0 — keypoints shape: {keypoints.shape}")

# ----- determine the canvas size the pose was plotted on -----
# Keypoints live in a TARGET_SIZE x TARGET_SIZE padded space (768x768 by default)
if pad_params is not None:
    top, left, new_h, new_w, orig_h, orig_w = pad_params
    # figure out TARGET_SIZE from pad_params
    canvas_size = new_h + 2 * top if new_h + 2 * top == new_w + 2 * left else max(new_h + 2 * top, new_w + 2 * left)
else:
    # fallback: infer from keypoint range
    canvas_size = 768
    top, left, new_h, new_w = 0, 0, canvas_size, canvas_size

print(f"Pose canvas size: {canvas_size}x{canvas_size}")

# ----- load and resize reference image to match pose canvas -----
img_path = None
for ext in [".jpg", ".jpeg", ".png"]:
    candidate = Path(IMAGE_DIR) / (stem + ext)
    if candidate.exists():
        img_path = candidate
        break

if img_path is None:
    raise FileNotFoundError(f"No image found for {stem} in {IMAGE_DIR}")

img_orig = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
h_orig, w_orig = img_orig.shape[:2]
print(f"Original image size: {w_orig}x{h_orig}")

# Resize & pad the reference image the same way the preprocessing did
scale = canvas_size / max(h_orig, w_orig)
rh, rw = int(h_orig * scale), int(w_orig * scale)
resized = cv2.resize(img_orig, (rw, rh))

pad_top    = (canvas_size - rh) // 2
pad_bottom = canvas_size - rh - pad_top
pad_left   = (canvas_size - rw) // 2
pad_right  = canvas_size - rw - pad_left

img_canvas = cv2.copyMakeBorder(
    resized, pad_top, pad_bottom, pad_left, pad_right,
    cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
print(f"Resized image to: {img_canvas.shape[1]}x{img_canvas.shape[0]} (matches pose canvas)")

# ----- plot keypoints on the resized image -----
xs = keypoints[:, 0]
ys = keypoints[:, 1]

plt.figure(figsize=(8, 8))
plt.imshow(img_canvas)
for i, (x, y, c) in enumerate(zip(xs, ys, scores)):
    if c > 0.3:
        plt.plot(x, y, 'ro', markersize=4)
        plt.text(x, y, str(i), color='yellow', fontsize=6)

plt.title(f"{stem} — {len(keypoints)} keypoints on {canvas_size}x{canvas_size} canvas")
plt.axis('off')
plt.tight_layout()
plt.savefig("pose_verify.png", dpi=150)
plt.show()
print("Saved pose_verify.png")
