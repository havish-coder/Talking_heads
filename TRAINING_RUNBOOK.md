# Talking_Heads — v2 Training Runbook

Audio-driven half-body human animation on a **CogVideoX-2B** DiT backbone,
porting EchoMimicV2's APDH curriculum + PhD Loss. This runbook is the path from
the current repo to a trained model on a **Lightning A100**.

> Local GPU is a 4 GB RTX 3050 — it cannot hold CogVideoX-2B. **All training and
> the latent re-encode run on the Lightning A100.** Only the offline shape test
> (`Src/test_dit.py`) runs locally.

## What changed in v2 (why the old results were bad)

1. **Audio routing (root cause).** Old code projected audio into CogVideoX's
   frozen T5 *text* slot and interpolated it to 226 tokens — destroying the
   frame-level time alignment lip-sync needs. v2 injects **frame-aligned
   windowed audio cross-attention** between transformer blocks
   (`WindowedAudioCrossAttention`), zero-init gated. The text slot now gets a
   learned **null embedding**.
2. **Resolution.** Old latents were 96×96 (768²), off CogVideoX-2B's native
   sinusoidal-pos-embed scale and ~2.5× the token cost. v2 re-encodes at
   **480² → 60×60** (uniform downscale → existing DWPose keypoints stay valid).
3. **Audio curriculum.** Audio cross-attention is **muted during the Initial
   Pose phase** (APDH stage 1) and phased in from stage 2, per the paper.

## Step 0 — push code

```bash
git add -A && git commit -m "v2: frame-aligned audio xattn + native-res fix"
git push origin main
```

## Step 1 — on the Lightning A100 Studio

```bash
git clone https://github.com/havish-coder/Taliking_heads && cd Taliking_heads
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Step 2 — get the data onto the Studio

You do **not** need the 4.6 GB of old latents. Upload only (~200 MB):

```
DATASET2/processed_dataset_wholebody/final_videos/          # 189 MB source mp4 (re-encode source)
DATASET2/processed_dataset_wholebody/reference_images_final_768/
DATASET2/DATASET/audio/                                     # 97 .m4a
DATASET2/DATASET/pose_data_single/                          # *.npy DWPose
```

(Lightning: drag-drop into the Studio file browser, or `rsync`/cloud drive.)

## Step 3 — re-encode latents at native resolution (A100, one-time)

```bash
python DATASET2/encode_2b.py --size 480
# -> DATASET2/DATASET/video_latents_final_videos/*.npy  (1,16,T,60,60)
# -> DATASET2/DATASET/ref_latents/*.npy                 (1,16,1,60,60)
```

## Step 4 — verify the model builds (downloads CogVideoX-2B once)

```bash
cd Src && python -c "from Models.talking_heads_dit import TalkingHeadsDiT as M; m=M.from_pretrained_cogvideox(); print(m.param_summary())"
```

## Step 5 — train

```bash
cd Src && python training/train.py --config training/config.yaml
# checkpoints -> ../checkpoints/checkpoint_XXXXXX/checkpoint.pt
```

Watch the logs: `loss`, `Llatent`, and the APDH `stage`. Audio engages at stage 2
(iter 10k). Early on, run a quick **overfit sanity check** on ~4 clips
(`total_iters: 2000`, point `data_root` at a 4-sample copy) — if it can't overfit
4 clips, something is still wrong before burning A100 hours on the full run.

## Step 6 — inference

```bash
cd Src && python infer.py \
  --checkpoint ../checkpoints/checkpoint_010000/checkpoint.pt \
  --ref_image  <face.jpg> --audio <speech.wav> \
  --output ../result.mp4 --num_frames 49 --steps 50 --guidance 2.0
```

## Honest expectations / next levers

- **~97 clips is small** for half-body + hands. Expect identity drift and rough
  gestures first. Biggest quality levers, in order:
  1. **More data** (the single highest-impact lever).
  2. **Spatial Audio Diffusion** — mask audio xattn to lips→head→global across
     APDH stages (currently global only). The paper's full §3.2.2.
  3. **Stronger identity** — reference tokens in joint attention (a
     ReferenceNet-lite), beyond the current channel-concat.
- Validate the pipeline by overfitting a handful of clips before scaling.
