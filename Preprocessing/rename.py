from pathlib import Path

root = Path("processed_dataset_wholebody")

folders = {
    "audio"                      : "*.m4a",
    "video_latents_final_videos" : "*.npy",
    "ref_latents"                : "*.npy",
    "pose_data_single"           : "*.npy",
    "final_videos"               : "*.mp4",
    "reference_images_final_768" : "*.jpg",
}

for folder, pattern in folders.items():
    path = root / folder
    files = sorted(path.glob(pattern))
    for i, f in enumerate(files):
        new_name = f"{i:03d}{f.suffix}"
        f.rename(path / new_name)
        print(f"[{folder}] {f.name} → {new_name}")

print("\nDone.")