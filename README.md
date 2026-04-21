# Talking Heads
### Audio-Driven Half-Body Human Animation
*Based on EchoMimicV2 (CVPR 2025) — adapted for CogVideoX DiT backbone*

---

## What This Project Does

This project generates a **talking half-body video** of a person from three inputs:

- A **reference image** — who the person looks like
- A **driving audio clip** — what they are saying
- A **hand pose sequence** — how their hands move

The output is a realistic, audio-synchronized video where the person's lips, face, and upper body move naturally with the speech.

---

## Our Architecture Pipeline

Our pipeline is adapted from **EchoMimicV2** but uses **CogVideoX DiT** as the backbone instead of the original UNet-based design.

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT PROCESSING                          │
│                                                                  │
│   Reference Image          Audio Clip          Hand Pose .npy    │
│        │                      │                      │           │
│        ▼                      ▼                      ▼           │
│  CogVideoX 3D VAE        Wav2Vec2 Encoder       DWPose Encoder   │
│  (encodes to latent)    (extracts embeddings)  (keypoint maps)   │
└──────────────┬────────────────┬──────────────────────┬───────────┘
               │                │                      │
               ▼                │                      │
  ┌────────────────────┐        │                      │
  │  Token Prepending  │        │                      │
  │  ref latent = T=0  │        │                      │
  └────────────┬───────┘        │                      │
               │                │                      │
               ▼                ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                       CogVideoX DiT                              │
│                                                                  │
│    Unified 3D Transformer — all tokens attend globally           │
│                                                                  │
│    ┌──────────────────────────────────────────────────────────┐  │
│    │  DiT Block × N                                           │  │
│    │   Self-Attention   (video + ref tokens together)         │  │
│    │   Audio Cross-Attn ◄── Wav2Vec2 embeddings               │  │
│    │   [IP-Adapter]     ◄── ref identity features (planned)   │  │
│    └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                   CogVideoX 3D VAE Decoder
                             │
                             ▼
                   ┌───────────────────┐
                   │  Output Video     │
                   │  (.mp4)           │
                   └───────────────────┘
```

### How This Differs from the Original EchoMimicV2

The paper uses a **dual-UNet** design. We swap the entire backbone for CogVideoX DiT:

| Component | EchoMimicV2 paper | Our Implementation |
|---|---|---|
| Backbone | SD-based UNet | CogVideoX DiT |
| Reference mechanism | Separate ReferenceNet (frozen UNet copy) | Token prepending + IP-Adapter (planned) |
| VAE | SD VAE — encodes per frame independently | CogVideoX 3D causal VAE — encodes temporally |
| Audio encoder | Wav2Vec2 via dedicated Audio Cross-Attention | Wav2Vec2 projected to DiT hidden dim |
| Pose injection | Channel concat with noisy latent | Pose encoder output as additional tokens |
| Temporal coherence | Explicit Temporal-Attention blocks | Native in DiT 3D full-sequence attention |

### The ReferenceNet Problem

The biggest architectural gap between our implementation and the paper is the **ReferenceNet**. In the paper it is a frozen duplicate of the denoising UNet that runs in parallel, processes only the reference image, and injects identity features (face shape, clothing, skin tone) into every layer of the denoising UNet through cross-attention. This is what makes generated videos consistently look like the specific reference person.

CogVideoX DiT has no equivalent. We address this in two stages:

- **Current**: Reference image is encoded with the 3D VAE and prepended as `T=0` to the noisy video token sequence. The DiT's global 3D attention can attend to it from every generated frame.
- **Planned**: IP-Adapter style cross-attention adapter inserted into each DiT block, giving a dedicated identity preservation pathway without needing a second full network.

### Audio Injection

The paper trains dedicated Audio Cross-Attention blocks from scratch. In our setup we project **Wav2Vec2 embeddings** (dimension 768) to match the CogVideoX DiT hidden dimension (4096 for the 5B model) and pass them as `encoder_hidden_states`, replacing the T5 text conditioning. This reuses the existing cross-attention infrastructure without modifying the DiT.

### Training Strategy

The paper's Audio-Pose Dynamic Harmonization (APDH) runs through a 60,000+ step progressive curriculum. Our fine-tuning simplifies this for a single GPU:

- **Frozen during training**: 3D VAE, Wav2Vec2, CogVideoX DiT backbone weights
- **Trained**: Audio projection layer, pose encoder adapter
- **Loss**: Standard diffusion MSE on noise prediction
- **VRAM saving**: VAE latents and audio embeddings are pre-computed and saved to disk so neither model occupies GPU memory during training runs

---

## Project Structure

```
Talking_heads/
│
├── DATASET/                     ← raw dataset files
│
├── Testing_models/
│   └── Wav2Vec2_test/           ← audio encoder experiments
│
├── data/                        ← prepared training data (generated by scripts)
│   ├── videos/                  ← .mp4 half-body speaking videos
│   ├── audios/                  ← .wav files (16kHz mono)
│   ├── poses/                   ← .npy DWPose keypoint arrays
│   ├── latents/                 ← pre-encoded VAE latents (generated)
│   └── audio_embs/              ← pre-encoded Wav2Vec2 embeddings (generated)
│
├── pretrained_weights/          ← CogVideoX model weights (downloaded separately)
│
├── prepare_dataset.py           ← organises raw files into expected structure
├── precompute_latents.py        ← pre-encodes dataset, removes VAE from GPU
├── train_finetune.py            ← main training script
├── train.yaml                   ← training configuration
├── requirements.txt
└── README.md
```

---

## Setup and Run Guide

### Requirements

- Python 3.10
- CUDA 11.7 or higher
- Git and Git LFS installed
- ffmpeg installed (for audio extraction)

### Step 1 — Clone the repository

```bash
git clone https://github.com/havish-coder/Talking_heads
cd Talking_heads
```

### Step 2 — Create a conda environment

```bash
conda create -n talking_heads python=3.10
conda activate talking_heads
```

### Step 3 — Install dependencies

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### Step 4 — Download CogVideoX pretrained weights

```bash
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-5b pretrained_weights
```

After download the structure should look like this:

```
pretrained_weights/
├── transformer/        ← DiT weights
├── vae/                ← 3D causal VAE
├── tokenizer/
├── text_encoder/
└── scheduler/
```

### Step 5 — Prepare your dataset

Place your videos, audio files, and DWPose `.npy` keypoint arrays in a folder. Files must share the same stem name — for example `clip001.mp4`, `clip001.wav`, `clip001.npy`.

Then run:

```bash
python prepare_dataset.py \
    --video_dir /path/to/your/videos \
    --pose_dir  /path/to/your/dwpose_npy_files \
    --out_dir   ./data \
    --extract_audio
```

The `--extract_audio` flag uses ffmpeg to extract audio directly from your video files. Skip it if you already have separate `.wav` files.

After this step verify the output structure:

```
data/
  videos/   ← .mp4 files
  audios/   ← .wav files (16kHz mono)
  poses/    ← .npy files
```

### Step 6 — Pre-compute latents and audio embeddings

This step runs the VAE and Wav2Vec2 encoder once over your dataset and saves the outputs to disk. After this the heavy encoders are never loaded during training, saving approximately 2GB of VRAM.

```bash
python precompute_latents.py \
    --data_root   ./data \
    --weights_dir ./pretrained_weights \
    --image_size  384 \
    --n_frames    16
```

This will populate `data/latents/` and `data/audio_embs/`. It only needs to run once. If you change `image_size` or `n_frames` later you need to re-run it.

### Step 7 — Edit the training config

Open `train.yaml`:

```yaml
pretrained_weights_dir: "./pretrained_weights"
data_root:              "./data"
output_dir:             "./checkpoints/run1"

image_size:    384      # lower to 256 if you get out-of-memory errors
sample_frames: 16       # lower to 8 if you get out-of-memory errors
batch_size:    1
gradient_accumulation_steps: 8
max_train_steps: 10000
learning_rate: 1.0e-5
save_steps: 500
```

### Step 8 — Run training

```bash
python train_finetune.py --config train.yaml
```

Checkpoints save every 500 steps to `./checkpoints/run1/`. Loss and learning rate are printed to console at each optimizer step.

To resume from a checkpoint, set `resume_from` in `train.yaml` to the checkpoint path.

---

## DWPose Preprocessing Notes

Your pose `.npy` files should be generated from source videos using DWPose before running any training scripts. The `Testing_models/Wav2Vec2_test/` folder contains early experiments for the audio side of this. Each `.npy` file should contain per-frame keypoint data corresponding to one video clip.

Expected format per frame is a dict or array with body and hand keypoint `(x, y)` coordinates normalised to `[0, 1]`. If your format is different, edit the `_load_pose()` method inside `train_finetune.py` to match your output structure.

---

## Known Issues and Limitations

### GPU Memory — The Primary Bottleneck

This is the central constraint of the project. The original EchoMimicV2 paper trains on **8× A100 80GB GPUs**. Our development hardware is a **single RTX 3050 with 8GB VRAM**. Nearly every design decision in our implementation exists because of this gap.

**Resolution**: The paper trains at 768×768. At 8GB we are limited to 384×384 or lower. Fine detail in lip movement and finger articulation is noticeably softer at this resolution.

**Batch size**: We use batch size 1 with gradient accumulation of 8 steps. The paper uses effective batch 32. This slows convergence significantly — expect training to take much longer per quality checkpoint.

**Frame count**: The paper uses 24-frame clips. We cap at 16 frames due to VRAM. Temporal coherence in longer generated videos is weaker as a result.

**ReferenceNet not feasible**: Running a full second UNet copy in parallel (as the paper does) requires approximately 1.7GB of additional VRAM just for the frozen reference branch. On 8GB total, alongside the DiT, VAE, and activations, this does not fit. This is why we use token prepending as a substitute and plan an IP-Adapter solution instead.

**Training time**: Without multi-GPU setup, training runs that the paper completes in hours take days on an RTX 3050. For any serious training we recommend renting a cloud GPU — RunPod, Lambda Labs, or Google Colab Pro+ with an A100 are viable options.

### Identity Consistency Is Weaker

Without the full ReferenceNet the generated person will resemble the reference image in general appearance — hair colour, clothing, approximate face shape — but fine facial features may drift across frames, especially during speech. The planned IP-Adapter implementation will improve this substantially.

### Audio-to-Body Gesture Correlation

The paper runs a 60,000+ step progressive training curriculum (APDH) to teach the model to translate audio rhythm into corresponding body and hand motion. Our simplified fine-tuning skips the full curriculum. Audio-to-gesture correlation — hand movements that follow speech rhythm naturally — will be noticeably weaker than the paper demonstrates.

### No Headshot Data Augmentation

The paper uses 540 hours of headshot talking-face video with a Head Partial Attention mechanism to boost facial expressiveness during training. We do not have access to this volume of data, which affects how expressive the generated face appears during speech.

---

## Planned Improvements

- [ ] IP-Adapter cross-attention adapter for proper identity preservation
- [ ] PhD Loss — three-phase denoising loss from the paper (pose, detail, quality)
- [ ] Full APDH progressive training curriculum
- [ ] Cloud training guide for RunPod / Colab A100
- [ ] Inference script for generating videos from fine-tuned checkpoints
- [ ] Gradio demo interface

---

## Citation

```bibtex
@article{meng2024echomimicv2,
  title={EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation},
  author={Meng, Rang and Zhang, Xingyu and Li, Yuming and Ma, Chenguang},
  journal={arXiv preprint arXiv:2411.10061},
  year={2024}
}
```

---

## References

- [EchoMimicV2 paper](https://arxiv.org/abs/2411.10061) — CVPR 2025
- [EchoMimicV2 official repo](https://github.com/antgroup/echomimic_v2)
- [CogVideoX](https://github.com/THUDM/CogVideo) — THUDM
- [DWPose](https://github.com/IDEA-Research/DWPose) — pose estimation backbone
