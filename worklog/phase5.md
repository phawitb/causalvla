# Phase 5: Training ทุก Experiment Variants

> **อัปเดตล่าสุด:** 10 สิงหาคม 2026
> **สถานะ:** IN PROGRESS — Preflight/Pilot ผ่านแล้ว, กำลัง Full Train Model A

## Overview

Train 5 models บน LIBERO Spatial dataset เพื่อเปรียบเทียบ:

| Model | Policy Type | Dataset | คำอธิบาย |
|-------|------------|---------|---------|
| **A** | `smolvla` | `lerobot/libero_spatial_image` | Standard SFT baseline |
| **B** | `smolvla` | `phawitbinabik/libero_spatial_augmented` | Domain Randomization (train บน augmented-only) |
| **C** | `causal_vla` | `lerobot/libero_spatial_image` | CausalVLA (ours) — full invariance losses |
| **D** | `causal_vla` | `lerobot/libero_spatial_image` | Ablation: w/o L_latent |
| **E** | `causal_vla` | `lerobot/libero_spatial_image` | Ablation: w/o L_action |

## Training Conditions (ทุก model เหมือนกัน)

| Parameter | Value |
|-----------|-------|
| Base model | SmolVLA (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) |
| Steps | 50,000 |
| Batch size | 8 |
| Seed | 1000 (LeRobot default) |
| Learning rate | 1e-4 |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8, wd=1e-10) |
| Scheduler | Warmup 1000 steps → cosine decay to 2.5e-6 over 30000 steps |
| Grad clip | 10 |
| Save freq | 10,000 steps |
| FPS | 10 |
| chunk_size | 50 |
| freeze_vision_encoder | True |
| train_expert_only | True |

---

## Step 5.0 — Setup & Verification (Jupyter)

### สิ่งที่ต้องมี

| ข้อ | ตรวจสอบ | คำสั่ง |
|-----|---------|--------|
| 1 | Python environment มี lerobot + dependencies | `import lerobot; print(lerobot.__version__)` |
| 2 | GPU available (CUDA or MPS) | `import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())` |
| 3 | `causal_vla` policy registered | ดู cell ด้านล่าง |
| 4 | `causal_aug` package importable | `from causal_aug import CausalAugmenter` |
| 5 | Dataset accessible | `from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset("lerobot/libero_spatial_image", episodes=[0])` |
| 6 | Augmented dataset accessible | `ds = LeRobotDataset("phawitbinabik/libero_spatial_augmented", episodes=[0])` |
| 7 | WandB logged in (optional) | `import wandb; wandb.login()` |
| 8 | HuggingFace logged in | `from huggingface_hub import HfApi; print(HfApi().whoami()['name'])` |

### Jupyter Notebook — Cell-by-Cell

#### Cell 0: Install & Setup (Colab Only)

```python
# One-command setup — clone จาก GitHub แล้ว install ทุกอย่าง
!git clone https://github.com/phawitb/causalvla.git 2>/dev/null || (cd causalvla && git pull)
!bash causalvla/setup_colab.sh
```

หลังรัน Cell 0 แล้ว **Restart runtime** (Runtime > Restart runtime) แล้วข้ามไป Cell 1

ถ้า update code ใหม่จาก GitHub:
```python
!cd causalvla && git pull && bash setup_colab.sh
```

---

#### Cell 1: Environment Check

```python
import sys
import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")
```

#### Cell 2: LeRobot + CausalVLA Import Check

```python
import lerobot
print(f"LeRobot version: {lerobot.__version__}")

# Trigger policy registration
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.causal_vla.configuration_causal_vla import CausalVLAConfig
from lerobot.configs import PreTrainedConfig

reg = PreTrainedConfig._choice_registry
print(f"smolvla registered: {'smolvla' in reg}")
print(f"causal_vla registered: {'causal_vla' in reg}")

# CausalAugmenter
from causal_aug import CausalAugmenter
aug = CausalAugmenter(K=3)
test_img = torch.randn(2, 3, 256, 256)
out = aug(test_img)
print(f"CausalAugmenter output shape: {out.shape}")  # [3, 2, 3, 256, 256]
```

#### Cell 3: Dataset Check

```python
from lerobot.datasets import LeRobotDataset

# Original dataset
ds_orig = LeRobotDataset("lerobot/libero_spatial_image", episodes=[0])
print(f"Original: {ds_orig.num_episodes} ep, {ds_orig.num_frames} frames")
print(f"  Features: {list(ds_orig.meta.features.keys())}")

# Augmented dataset
ds_aug = LeRobotDataset("phawitbinabik/libero_spatial_augmented", episodes=[0])
print(f"Augmented: {ds_aug.num_episodes} ep, {ds_aug.num_frames} frames")

# Verify shapes match
item = ds_orig[0]
print(f"  image shape: {item['observation.images.image'].shape}")
print(f"  state shape: {item['observation.state'].shape}")
print(f"  action shape: {item['action'].shape}")
```

#### Cell 4: HuggingFace & WandB Auth

```python
from huggingface_hub import HfApi
api = HfApi()
HF_USER = api.whoami()['name']
print(f"HuggingFace user: {HF_USER}")

# WandB (optional — ปิดได้ถ้าไม่ต้องการ)
# import wandb
# wandb.login()
```

#### Cell 5: Train — Model A (Standard SFT)

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train

# สร้าง config ด้วย Python (ไม่ต้อง CLI)
cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="lerobot/libero_spatial_image",
    ),
    policy=SmolVLAConfig(
        device=DEVICE,
        push_to_hub=True,
        repo_id=f"{HF_USER}/causalvla-model-a-sft",
    ),
    output_dir="outputs/train/model_a_sft",
    batch_size=8,
    steps=50_000,
    seed=1000,
    save_freq=10_000,
    save_checkpoint_to_hub=True,  # push checkpoint ทุก save_freq steps
    log_freq=200,
    num_workers=4,
)

print(cfg)
train(cfg)
```

#### Cell 6: Train — Model B (Domain Randomization)

```python
cfg_b = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="phawitbinabik/libero_spatial_augmented",
    ),
    policy=SmolVLAConfig(
        device=DEVICE,
        push_to_hub=True,
        repo_id=f"{HF_USER}/causalvla-model-b-dr",
    ),
    output_dir="outputs/train/model_b_dr",
    batch_size=8,
    steps=50_000,
    seed=1000,
    save_freq=10_000,
    save_checkpoint_to_hub=True,
    log_freq=200,
    num_workers=4,
)

train(cfg_b)
```

#### Cell 7: Train — Model C (CausalVLA — Ours)

```python
cfg_c = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="lerobot/libero_spatial_image",
    ),
    policy=CausalVLAConfig(
        device=DEVICE,
        push_to_hub=True,
        repo_id=f"{HF_USER}/causalvla-model-c-ours",
        n_counterfactual=3,
        lambda_latent=0.1,
        lambda_action=0.1,
        lambda_smooth=0.01,
    ),
    output_dir="outputs/train/model_c_causal",
    batch_size=8,
    steps=50_000,
    seed=1000,
    save_freq=10_000,
    save_checkpoint_to_hub=True,
    log_freq=200,
    num_workers=4,
)

train(cfg_c)
```

#### Cell 8: Train — Model D (Ablation w/o L_latent)

```python
cfg_d = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="lerobot/libero_spatial_image",
    ),
    policy=CausalVLAConfig(
        device=DEVICE,
        push_to_hub=True,
        repo_id=f"{HF_USER}/causalvla-model-d-no-latent",
        n_counterfactual=3,
        use_latent_loss=False,   # ← ปิด L_latent
        lambda_action=0.1,
        lambda_smooth=0.01,
    ),
    output_dir="outputs/train/model_d_no_latent",
    batch_size=8,
    steps=50_000,
    seed=1000,
    save_freq=10_000,
    save_checkpoint_to_hub=True,
    log_freq=200,
    num_workers=4,
)

train(cfg_d)
```

#### Cell 9: Train — Model E (Ablation w/o L_action)

```python
cfg_e = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="lerobot/libero_spatial_image",
    ),
    policy=CausalVLAConfig(
        device=DEVICE,
        push_to_hub=True,
        repo_id=f"{HF_USER}/causalvla-model-e-no-action",
        use_action_loss=False,   # ← ปิด L_action
        lambda_latent=0.1,
        lambda_smooth=0.01,
    ),
    output_dir="outputs/train/model_e_no_action",
    batch_size=8,
    steps=50_000,
    seed=1000,
    save_freq=10_000,
    save_checkpoint_to_hub=True,
    log_freq=200,
    num_workers=4,
)

train(cfg_e)
```

---

---

## Smoke Test Results (Mac M2, CPU, 2 steps, batch=2)

```
Model A (SmolVLA):
  step:1  loss:2.244   grdn:30.406  updt_s:29.5s
  step:2  loss:13.652  grdn:92.901  updt_s:29.7s
  Total: ~60s ✓

Model C (CausalVLA, K=2):
  step:1  loss:9.414   loss_task:2.244  loss_latent:71.640  loss_action:0.058  loss_smooth:0.016
  step:2  loss:20.062  loss_task:14.112 loss_latent:59.473  loss_action:0.025  loss_smooth:0.001
  Total: ~180s (3x slower due to K+1=3 forward passes) ✓
```

**สิ่งสำคัญที่ค้นพบ:**
- `push_to_hub` default = **True** ใน `PreTrainedConfig` → ต้อง set `push_to_hub=False` หรือใส่ `repo_id`
- ไม่ต้องใส่ `type="smolvla"` ใน config — draccus ดูจาก class type เอง
- `train(cfg)` bypass CLI parser ได้ถ้าส่ง `TrainPipelineConfig` object ตรงๆ (line 398 ใน parser.py)
- CausalVLA loss breakdown log ครบ: `loss_task`, `loss_latent`, `loss_action`, `loss_smooth`

---

## Alternative: CLI Commands (Terminal)

ถ้าไม่ใช้ Jupyter สามารถรันจาก terminal ได้:

```bash
cd /Users/phawit/Projects/CausalVLA/lerobot

# Model A: Standard SFT
lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_a_sft \
    --policy.device=cuda --batch_size=8 --steps=50000

# Model B: Domain Randomization
lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=phawitbinabik/libero_spatial_augmented \
    --output_dir=outputs/train/model_b_dr \
    --policy.device=cuda --batch_size=8 --steps=50000

# Model C: CausalVLA (Ours)
lerobot-train \
    --policy.type=causal_vla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_c_causal \
    --policy.device=cuda --policy.n_counterfactual=3 \
    --policy.lambda_latent=0.1 --policy.lambda_action=0.1 --policy.lambda_smooth=0.01 \
    --batch_size=8 --steps=50000

# Model D: Ablation w/o L_latent
lerobot-train \
    --policy.type=causal_vla --policy.use_latent_loss=false \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_d_no_latent \
    --policy.device=cuda --batch_size=8 --steps=50000

# Model E: Ablation w/o L_action
lerobot-train \
    --policy.type=causal_vla --policy.use_action_loss=false \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_e_no_action \
    --policy.device=cuda --batch_size=8 --steps=50000
```

---

## หมายเหตุสำคัญ

### Jupyter vs CLI

| | Jupyter | CLI |
|---|---------|-----|
| สร้าง config | Python dataclass | `--policy.type=smolvla --batch_size=8` |
| เรียก train | `train(cfg)` | `lerobot-train --...` |
| ข้อดี | ดู config ก่อน train, ปรับได้ flexible | สั้น, copy-paste ง่าย |
| ข้อเสีย | ต้อง import ถูก, draccus parser bypass | ต้องจำ flag ทั้งหมด |

### ข้อควรระวัง

1. **`@parser.wrap()`** — `train()` ถูก wrap ด้วย draccus parser, ถ้าเรียกจาก Jupyter ตรงๆ อาจต้อง bypass (ส่ง config object ตรงๆ แทน CLI args)
2. **`accelerate`** — training ใช้ HuggingFace Accelerate, ใน Jupyter อาจมีปัญหากับ multiprocessing — ลอง `num_workers=0` ถ้ามี error
3. **GPU memory** — SmolVLA + batch_size=8 ต้องการ ~16GB VRAM, CausalVLA (K=3 counterfactuals) ต้องการ ~24GB+ เพราะ forward K+1 ครั้ง
4. **MPS (Mac)** — รองรับได้แต่ช้ากว่า CUDA มาก, ลด batch_size เหลือ 2-4
5. **`causal_vla` import** — ต้อง import `CausalVLAConfig` ก่อนเรียก `train()` เพื่อให้ draccus registry มีค่า

### VRAM Estimation

| Model | Forward passes/step | Est. VRAM (batch=8) |
|-------|--------------------|--------------------|
| A, B (SmolVLA) | 1 | ~16 GB |
| C (CausalVLA, K=3) | 1 + 3 = 4 | ~24 GB |
| D, E (CausalVLA ablation) | 1 + 3 = 4 | ~24 GB |

ถ้า VRAM ไม่พอ: ลด `batch_size` หรือ ลด `n_counterfactual` (K=1 ก็ได้)

---

## GPU Server Training Log — 10 สิงหาคม 2026

### สรุปสถานะล่าสุด

| งาน | สถานะ | ผลลัพธ์ |
|---|---|---|
| เตรียม GPU environment | **PASS** | RTX 4090 24 GB, CUDA ใช้งานได้ |
| ติดตั้ง LeRobot/CausalVLA | **PASS** | LeRobot 0.6.1, policy registration และ augmentation ผ่าน |
| Dataset preflight | **PASS** | Original และ augmented dataset โหลด/ถอดวิดีโอได้ |
| Model A smoke/pilot | **PASS** | 10 steps และ 500 steps สำเร็จ |
| Model B pilot | **PASS** | 500 steps สำเร็จบน augmented dataset |
| Model C pilot | **PASS** | 500 steps สำเร็จหลังแก้ latent gradient |
| Hyperparameter calibration | **PASS** | เลือก K=3, λ_latent=5.0, λ_action=1.0 |
| Model A full training | **COMPLETED** | 25,000 steps, final loss 0.394, Hub upload ครบ |
| Model B full training | **COMPLETED** | 25,000 steps, final loss 0.334, มี resume 2 รอบ |
| Model C full training | **COMPLETED** | 25,000 steps, final loss 0.825, K=3, λ_latent=5, λ_action=1 |
| Model D full training | **COMPLETED** | 25,000 steps, final loss 0.798, ปิด latent loss สำเร็จ |
| Model E full training | **COMPLETED** | 25,000 steps, final loss 0.433, ปิด action loss สำเร็จ |
| LIBERO ID/OOD evaluation | **PENDING** | ทำหลังได้ checkpoint ครบ A–E |

### 1. GPU Server Environment

| Item | Value |
|---|---|
| Host | `PhyAI4090` |
| OS | Ubuntu 22.04, Linux 6.8 |
| GPU | NVIDIA GeForce RTX 4090 |
| VRAM | 24,564 MiB |
| Driver | 580.173.02 |
| Driver CUDA | 13.0 |
| Conda env | `causalvla` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| LeRobot | 0.6.1 |
| Transformers | 5.5.4 |
| PEFT | 0.20.0 |

ใช้ `PYTHONNOUSERSITE=1` เพื่อป้องกัน package ใน `~/.local` ปนกับ Conda environment และกำหนด Hugging Face cache ที่ `~/hf_cache/causalvla`.

### 2. Environment Issues ที่แก้ไข

1. LeRobot 0.6.1 ต้องใช้ Python ≥3.12 — สร้าง Conda environment ใหม่ด้วย Python 3.12 แทนการ upgrade environment เดิม
2. Environment เดิมมี CUDA dependencies ค้างใต้ Python 3.11 ทำให้ขาด `libnvshmem_host.so.3` — แก้ด้วยการสร้าง environment ใหม่แบบ clean
3. `causal_aug` editable install ถูก namespace folder ชั้นนอกบัง — เปลี่ยนเป็นติดตั้งแบบปกติด้วย `pip install ./causal_aug`
4. ติดตั้ง LeRobot extras ครบด้วย `lerobot[training,smolvla,peft]==0.6.1`
5. DataLoader worker มี warning ตอนปิด short run — ใช้ `persistent_workers=false` ใน pilot; ไม่กระทบ checkpoint

### 3. Dataset Preflight

ทดสอบทั้งสอง dataset ด้วย episode 0 สำเร็จ:

| Dataset | Episode/Frames ที่ทดสอบ | Image | State | Action |
|---|---:|---|---|---|
| `lerobot/libero_spatial_image` | 1 / 110 | 2 กล้อง, `[3,256,256]` | `[8]` | `[7]` |
| `phawitbinabik/libero_spatial_augmented` | 1 / 110 | 2 กล้อง, `[3,256,256]` | `[8]` | `[7]` |

Full original dataset ที่ใช้ train มี 432 episodes / 52,970 frames / 10 FPS.

### 4. Architectural Bug ที่พบและแก้ไข

เดิม `L_latent` คำนวณจาก `prefix_embs` (image + language + state) แต่ config ใช้ `freeze_vision_encoder=True` และ `train_expert_only=True` ทำให้ representation หลักถูก freeze และ latent loss มีค่ามากโดยแทบไม่มี gradient ไปยัง trainable parameters.

หลักฐานจาก smoke test เดิม:

- `loss_latent ≈ 50–75`
- Weighted latent (`0.1 × loss_latent`) ครองประมาณ 70–75% ของ total loss
- Gradient norm ของ Model A และ C เกือบเท่ากัน (`25.657` vs `25.658`)

การแก้ไข: เปลี่ยน latent representation เป็น `suffix_out` ซึ่งเป็น hidden states จาก trainable action expert แล้ว mean-pool เป็น `z` สำหรับ latent invariance loss.

หลังแก้:

- `loss_latent` อยู่ประมาณ 0.01–0.17
- มี gradient path เข้า action expert โดยตรง
- VRAM และเวลา/step แทบไม่เพิ่มจาก implementation เดิม
- Commit: `a021483 fix: compute latent invariance on action expert states`

ปรับ default loss weights จากผล pilot และ push ขึ้น GitHub แล้ว:

- `lambda_latent: 0.1 → 5.0`
- `lambda_action: 0.1 → 1.0`
- Commit: `87cfb54 config: tune causal invariance loss weights`

### 5. Batch-size Benchmark

| Config | Batch | LeRobot mem | GPU total โดยประมาณ | Throughput | ผล |
|---|---:|---:|---:|---:|---|
| Model C, K=3 | 8 | 6.88 GB | — | 15–16 samples/s | PASS |
| Model C, K=3 | 16 | 12.03 GB | 14.1 GB | 17–18 samples/s | **SELECTED** |

ที่ batch 16 GPU utilization ถึง 99% และใช้กำลังประมาณ 399/450 W แล้ว จึงไม่เลือก batch 32 ซึ่งคาดว่าจะใช้ VRAM ชิดขีดจำกัดและแทบไม่เพิ่ม throughput.

### 6. Pilot Training Results (500 Steps)

| Model/Config | Dataset | Loss เริ่ม → จบ | Gradient เริ่ม → จบ | Memory | เวลา | สถานะ |
|---|---|---:|---:|---:|---:|---|
| A — Standard SFT | Original | 2.064 → 0.954 | 13.20 → 3.35 | 4.27 GB | 4m44s | PASS |
| B — Domain Randomization | Augmented | 2.193 → 1.011 | 12.03 → 3.31 | 4.28 GB | 5m33s | PASS |
| C — λ_lat=1, λ_act=1 | Original | task 2.040 → 1.285 | 11.43 → 2.80 | 12.04 GB | 7m23s | PASS |
| C — λ_lat=5, λ_act=1 | Original | task 2.031 → 1.326 | 10.28 → 2.62 | 12.04 GB | 7m24s | **SELECTED** |

ผล calibration ที่ step 500:

| Weight | Task loss | Raw latent loss | Action loss | ข้อสรุป |
|---|---:|---:|---:|---|
| λ_lat=1, λ_act=1 | 1.285 | 0.009 | 0.219 | latent contribution ต่ำเกินไป |
| λ_lat=5, λ_act=1 | 1.326 | 0.003 | 0.209 | latent discrepancy ลด ~3 เท่า, task trade-off ~3% |

### 7. Final Training Configuration (Locked)

| Parameter | Final Value |
|---|---:|
| Batch size | 16 |
| Steps | 25,000 |
| Effective samples | 400,000 |
| Seed | 1000 |
| Learning rate | 1e-4 |
| Warmup | 500 steps |
| Cosine decay | 15,000 steps |
| Checkpoint frequency | 5,000 steps |
| Counterfactuals (K) | 3 |
| Augmentation intensity | 1.0 |
| λ_latent | 5.0 |
| λ_action | 1.0 |
| λ_smooth | 0.01 |

ลด steps จาก 50,000 เป็น 25,000 เพราะเพิ่ม batch 8 → 16 และรักษา sample budget เดิม:

```text
50,000 × 8  = 400,000 samples
25,000 × 16 = 400,000 samples
```

### 8. Full Training Status

Model A full training สำเร็จบน `tmux` session `causalvla_model_a`:

```text
Output: outputs/final/model_a_sft
Hub:    phawitbinabik/causalvla-model-a-sft (private)
Steps:  25,000
Save:   ทุก 5,000 steps + push checkpoint ไป Hugging Face Hub
```

ผลลัพธ์สุดท้าย:

```text
step 25,000: loss=0.394, gradient=1.614, lr=2.5e-06
samples: 400,000
epochs: 7.55
memory: 4.27 GB
throughput: 27–29 samples/s
total time: 3h52m18s
```

Checkpoint ถูกบันทึกและ push ไป Hugging Face Hub ครบที่ steps 5k, 10k, 15k, 20k และ 25k รวมถึง final pretrained model ที่ root ของ repo (69 files). ไม่มี NaN, OOM หรือ runtime error.

Model B full training บน augmented dataset สำเร็จครบ 25,000 steps:

```text
Initial run:      step 0 → 10,000
Failed resume:    เริ่มจาก 10,000 และล้มหลังรันเพิ่ม 3,555 steps (ไม่ใช้ checkpoint ช่วงนี้)
Successful resume: step 10,000 → 25,000
Final output:     outputs/resumed/model_b_dr
Final loss:       0.334
Gradient norm:    1.800
Learning rate:    2.5e-6
Samples:          400,000
Epochs:           7.55
Memory:           4.27 GB
Throughput:       24–25 samples/s ช่วงท้าย
```

Checkpoint local แบ่งอยู่สอง directory: steps 5k/10k ที่ `outputs/final/model_b_dr` และ steps 10k/15k/20k/25k ที่ `outputs/resumed/model_b_dr`. Log รอบแรกและ failed resume อยู่ใน `logs/model_b_dr.log`; successful resume อยู่ใน `logs/model_b_dr_resume.log`. ต้องเก็บทั้งสองไฟล์และสร้าง combined timeline สำหรับวิเคราะห์ย้อนหลัง.

ตรวจ full log แล้วพบว่าสาเหตุของ failed resume ไม่ใช่ CUDA OOM หรือโมเดลพัง แต่เป็นค่า checkpoint path ว่างใน shell ใหม่ ทำให้ path ที่ส่งเข้า `--config_path` เหลือเพียง `/pretrained_model` และ Hugging Face ตีความเป็น repo id ที่ไม่ถูกต้อง. รอบ successful resume โหลด checkpoint 10,000 ถูกต้อง พร้อม optimizer, scheduler และ RNG state; ช่วง 3,555 steps ที่ล้มจึงถูกทิ้งและไม่ปนใน final Model B.

สถิติจาก full log: ช่วงท้าย 2,000 steps มี mean loss ประมาณ `0.332`, mean gradient norm `1.778` และ throughput `24.1 samples/s`. ไม่พบ NaN, CUDA OOM หรือ runtime error ในรอบ successful resume. Loss สุดท้ายของ B (`0.334`) ต่ำกว่า A (`0.394`) ประมาณ 15% แต่ยังสรุปว่า B ดีกว่าไม่ได้ เพราะทั้งสองโมเดล train คนละ data distribution; ต้องเทียบด้วย LIBERO clean/OOD evaluation.

Model C full training สำเร็จครบ 25,000 steps โดยใช้ CausalVLA, K=3, λ_latent=5, λ_action=1 และ λ_smooth=0.01:

```text
Output:           outputs/final/model_c_causal
Hub:              phawitbinabik/causalvla-model-c-ours (private)
Final loss:       0.825
Task loss:        0.600
Latent loss:      0.001  (weighted contribution ≈ 0.005)
Action loss:      0.218  (weighted contribution ≈ 0.218)
Smooth loss:      0.401  (weighted contribution ≈ 0.004)
Gradient norm:    1.742
Learning rate:    2.5e-6
Samples:          400,000
Epochs:           7.55
Memory:           12.03 GB
Throughput:       18 samples/s
Total time:       6h05m54s training; 6h06m30s รวม final save
```

ค่าเฉลี่ย 2,000 steps สุดท้าย: total loss `0.831`, task `0.607`, latent `0.001`, action `0.218`, smooth `0.398`, gradient norm `1.771`. สมการ loss ตรงกับค่าที่ log ภายในความละเอียดการปัดเศษ เช่น final ≈ `0.600 + 5×0.001 + 1×0.218 + 0.01×0.401 = 0.827` เทียบกับ log `0.825`. Latent loss ลดจาก `0.086` ที่ step 100 เหลือประมาณ `0.001` ช่วงท้าย แสดงว่า objective ทำงานและ representation มี invariance สูงขึ้น; action consistency ยังเป็น regularizer หลักช่วงท้าย.

Checkpoint ครบ steps 5k, 10k, 15k, 20k และ 25k และพบ `End of training`. ไม่พบ traceback, NaN, CUDA OOM หรือ runtime error. Total loss ของ C ห้ามเปรียบเทียบตรงกับ A/B เพราะ C รวม weighted causal regularizers และ task forward หลาย counterfactual views; ต้องใช้ success rate จาก clean/OOD evaluation เป็นผลหลัก.

Model D full training (ablation w/o L_latent) สำเร็จครบ 25,000 steps:

```text
Output:           outputs/final/model_d_no_latent
Hub:              phawitbinabik/causalvla-model-d-no-latent (private)
Final loss:       0.798
Task loss:        0.571
Action loss:      0.223
Smooth loss:      0.409
Gradient norm:    1.686
Learning rate:    2.5e-6
Samples:          400,000
Epochs:           7.55
Memory:           12.03 GB
Throughput:       18 samples/s
Total time:       6h06m20s training; 6h06m37s รวม final save
```

ค่าเฉลี่ย 2,000 steps สุดท้ายของ D: total loss `0.805`, task `0.579`, action `0.222`, smooth `0.406`, gradient norm `1.723`. ไม่พบ `loss_latent` ตลอด 250 metric records จึงยืนยันว่า ablation ปิด latent objective จริง. เมื่อเทียบช่วงท้ายกับ C, D มี task loss ต่ำกว่าประมาณ 4.6% (`0.579` vs `0.607`) แต่ action lossสูงกว่าเล็กน้อย (`0.222` vs `0.218`); ความแตกต่างนี้ยังบอก robustness ไม่ได้จนกว่าจะรัน evaluation.

Checkpoint ครบ steps 5k, 10k, 15k, 20k และ 25k และพบ `End of training`. ไม่พบ traceback, NaN, CUDA OOM หรือ runtime error.

Model E full training (ablation w/o L_action) สำเร็จครบ 25,000 steps:

```text
Output:           outputs/final/model_e_no_action
Hub:              phawitbinabik/causalvla-model-e-no-action (private)
Final loss:       0.433
Task loss:        0.415
Latent loss:      0.001  (weighted contribution ≈ 0.005)
Smooth loss:      1.431  (weighted contribution ≈ 0.014)
Gradient norm:    1.739
Learning rate:    2.5e-6
Samples:          400,000
Epochs:           7.55
Memory:           12.02 GB
Throughput:       18 samples/s
Total time:       6h08m10s training; 6h08m28s รวม final save
```

ค่าเฉลี่ย 2,000 steps สุดท้ายของ E: total loss `0.438`, task `0.420`, latent `0.001`, smooth `1.427`, gradient norm `1.740`. ไม่พบ `loss_action` ตลอด 250 metric records จึงยืนยันว่า ablation ปิด action-consistency objective จริง. Smooth loss ของ E สูงกว่า C/D ประมาณ 3.5 เท่า (`1.427` vs `0.398–0.406`) เมื่อปิด action loss ซึ่งเป็นสัญญาณว่าควรตรวจ trajectory smoothness และ action consistency โดยตรงใน Phase 6; ยังไม่ควรสรุปจาก training loss อย่างเดียว.

Checkpoint ครบ steps 5k, 10k, 15k, 20k และ 25k และพบ `End of training`. ไม่พบ traceback, NaN, CUDA OOM หรือ runtime error. ณ จุดนี้ full training ครบทั้ง Model A–E และ Phase 5 เสร็จสมบูรณ์.

### 9. ลำดับงานถัดไป

1. ตรวจ Hub artifacts และเก็บ full logs ของ Model A–E
2. ทำ Phase 6 evaluation preflight ด้วย checkpoint 25,000 ของแต่ละโมเดล
3. รัน LIBERO evaluation บน clean/ID และ OOD mild/medium/heavy
4. วัด trajectory smoothness และ action consistency โดยเฉพาะ C เทียบ E
5. รวม success rate, robustness drop, mean/std และ ablation results ลง Paper Final

---

## Command Log — คำสั่งที่ใช้จริงบน GPU Server

> คำสั่งส่วนนี้บันทึกตามลำดับที่รันจริง และใช้เป็น runbook สำหรับทำซ้ำบนเครื่องใหม่

### A. ตรวจ GPU Server

```bash
hostname
uname -a
nvidia-smi
python3 --version
git --version
df -h .
free -h
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

### B. สร้าง Conda Environment

LeRobot 0.6.1 ต้องใช้ Python ≥3.12:

```bash
conda create -n causalvla python=3.12 pip -y
conda activate causalvla

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

ปิด user-site เพื่อไม่ให้ package จาก `~/.local` ปนเข้ามา:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

printf 'export PYTHONNOUSERSITE=1\n' \
  > "$CONDA_PREFIX/etc/conda/activate.d/causalvla_isolation.sh"

printf 'unset PYTHONNOUSERSITE\n' \
  > "$CONDA_PREFIX/etc/conda/deactivate.d/causalvla_isolation.sh"
```

ตรวจ CUDA:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
print("CUDA calculation:", y.shape)
print("Finite:", torch.isfinite(y).all().item())
PY

python -m pip check
```

### C. Clone และติดตั้ง Dependencies

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/phawitb/causalvla.git
cd causalvla

python -m pip install "lerobot[training,smolvla,peft]==0.6.1"
python -m pip install ./causal_aug
```

ตั้ง Hugging Face cache และ login:

```bash
mkdir -p ~/hf_cache/causalvla
export HF_HOME=~/hf_cache/causalvla

printf 'export HF_HOME="$HOME/hf_cache/causalvla"\n' \
  >> "$CONDA_PREFIX/etc/conda/activate.d/causalvla_isolation.sh"

hf auth login
hf auth whoami
```

### D. ติดตั้ง CausalVLA Policy เข้า LeRobot

```bash
cd ~/projects/causalvla

python - <<'PY'
from pathlib import Path
import inspect
import shutil

import lerobot.policies
import lerobot.policies.smolvla.modeling_smolvla as smolvla

repo = Path.cwd()
policies_dir = Path(inspect.getfile(lerobot.policies)).parent

source_policy = repo / "lerobot_patches" / "causal_vla"
target_policy = policies_dir / "causal_vla"
shutil.copytree(source_policy, target_policy, dirs_exist_ok=True)

init_file = policies_dir / "__init__.py"
source = init_file.read_text()
registration = (
    "from .causal_vla.configuration_causal_vla "
    "import CausalVLAConfig as CausalVLAConfig"
)

if registration not in source:
    anchor = "from .act.configuration_act import ACTConfig as ACTConfig"
    source = source.replace(anchor, anchor + "\n" + registration, 1)
    init_file.write_text(source)

model_file = Path(inspect.getfile(smolvla))
source = model_file.read_text()

if "def forward_with_latent(" not in source:
    method = (repo / "lerobot_patches" / "forward_with_latent.py").read_text().rstrip()
    anchor = "    def sample_actions("
    source = source.replace(anchor, method + "\n\n" + anchor, 1)
    model_file.write_text(source)

print("CausalVLA policy installed")
PY
```

ตรวจ policy registration และ expert latent:

```bash
python - <<'PY'
import inspect

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_policy_config, get_policy_class
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

cfg = make_policy_config("causal_vla")

print("Config:", type(cfg).__name__)
print("Policy:", get_policy_class("causal_vla").__name__)
print("Registered:", "causal_vla" in PreTrainedConfig._choice_registry)
print("forward_with_latent:", hasattr(VLAFlowMatching, "forward_with_latent"))
print(inspect.getsource(VLAFlowMatching.forward_with_latent).splitlines()[-1])
PY
```

บรรทัดสุดท้ายต้องเป็น:

```text
return losses, suffix_out, v_t
```

### E. Dataset Preflight

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

datasets = [
    ("original", "lerobot/libero_spatial_image"),
    ("augmented", "phawitbinabik/libero_spatial_augmented"),
]

for label, repo_id in datasets:
    ds = LeRobotDataset(repo_id, episodes=[0])
    item = ds[0]
    print(label, ds.num_episodes, ds.num_frames, ds.fps)
    print("image:", item["observation.images.image"].shape)
    print("wrist:", item["observation.images.wrist_image"].shape)
    print("state:", item["observation.state"].shape)
    print("action:", item["action"].shape)
PY
```

### F. Pilot Commands (500 Steps)

Model A — Standard SFT:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/pilot/model_a_500steps \
  --job_name=model_a_pilot \
  --batch_size=16 --steps=500 --seed=1000 \
  --save_freq=500 --log_freq=25 \
  --num_workers=4 --persistent_workers=false \
  --env_eval_freq=0 \
  2>&1 | tee logs/model_a_500steps.log
```

Model B — Domain Randomization:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=phawitbinabik/libero_spatial_augmented \
  --output_dir=outputs/pilot/model_b_500steps \
  --job_name=model_b_pilot \
  --batch_size=16 --steps=500 --seed=1000 \
  --save_freq=500 --log_freq=25 \
  --num_workers=4 --persistent_workers=false \
  --env_eval_freq=0 \
  2>&1 | tee logs/model_b_500steps.log
```

Model C — Final Causal Weight Pilot:

```bash
lerobot-train \
  --policy.type=causal_vla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_counterfactual=3 \
  --policy.aug_intensity=1.0 \
  --policy.lambda_latent=5.0 \
  --policy.lambda_action=1.0 \
  --policy.lambda_smooth=0.01 \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/pilot/model_c_l5_a1_500steps \
  --job_name=model_c_l5_a1_pilot \
  --batch_size=16 --steps=500 --seed=1000 \
  --save_freq=500 --log_freq=25 \
  --num_workers=4 --persistent_workers=false \
  --env_eval_freq=0 \
  2>&1 | tee logs/model_c_l5_a1_500steps.log
```

### G. Full Training Model A — คำสั่งที่กำลังรัน

สร้าง tmux session:

```bash
tmux new -s causalvla_model_a
```

ภายใน tmux:

```bash
conda activate causalvla
cd ~/projects/causalvla

export PYTHONNOUSERSITE=1
export HF_HOME=~/hf_cache/causalvla
export CUDA_VISIBLE_DEVICES=0

lerobot-train \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-model-a-sft \
  --policy.private=true \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=15000 \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/final/model_a_sft \
  --job_name=model_a_sft \
  --batch_size=16 \
  --steps=25000 \
  --seed=1000 \
  --save_freq=5000 \
  --save_checkpoint_to_hub=true \
  --log_freq=100 \
  --num_workers=4 \
  --persistent_workers=true \
  --env_eval_freq=0 \
  2>&1 | tee logs/model_a_sft.log
```

Detach tmux: กด `Ctrl+B` แล้วกด `D`.

### H. Full Training Model C — CausalVLA (Next)

ก่อนรัน ตรวจว่า installed patch ใช้ action-expert latent จริง:

```bash
python - <<'PY'
import inspect
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

source = inspect.getsource(VLAFlowMatching.forward_with_latent)
assert "return losses, suffix_out, v_t" in source
print("Expert latent patch: PASS")
PY
```

เริ่ม session และ train โดยบันทึก full log:

```bash
tmux new -s causalvla_model_c

conda activate causalvla
cd ~/projects/causalvla

export PYTHONNOUSERSITE=1
export HF_HOME=~/hf_cache/causalvla
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

lerobot-train \
  --policy.type=causal_vla \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-model-c-ours \
  --policy.private=true \
  --policy.n_counterfactual=3 \
  --policy.aug_intensity=1.0 \
  --policy.lambda_latent=5.0 \
  --policy.lambda_action=1.0 \
  --policy.lambda_smooth=0.01 \
  --policy.use_latent_loss=true \
  --policy.use_action_loss=true \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=15000 \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/final/model_c_causal \
  --job_name=model_c_causal \
  --batch_size=16 \
  --steps=25000 \
  --seed=1000 \
  --save_freq=5000 \
  --save_checkpoint_to_hub=true \
  --log_freq=100 \
  --num_workers=4 \
  --persistent_workers=true \
  --env_eval_freq=0 \
  2>&1 | tee logs/model_c_causal.log
```

หลัง step 100 ตรวจสุขภาพ run:

```bash
grep -E 'step:100|Auto-scaling|Traceback|RuntimeError|CUDA out of memory' \
  logs/model_c_causal.log
```

ถ้าต้อง resume ให้ใช้ `tee -a logs/model_c_causal.log` เพื่อรักษา log เดิมไว้เสมอ.

### I. Monitoring Commands

```bash
tmux ls
tmux attach -t causalvla_model_a

tail -n 30 ~/projects/causalvla/logs/model_a_sft.log
watch -n 1 nvidia-smi
df -h .
du -sh "$HF_HOME"
```

ตรวจ error และ checkpoint:

```bash
grep -E 'step:|Checkpoint|End of training|Traceback|RuntimeError|CUDA out of memory' \
  logs/model_a_sft.log | tail -n 40

find outputs/final/model_a_sft/checkpoints \
  -name training_step.json \
  -exec sh -c 'echo "$1"; sed -n "1,40p" "$1"' _ {} \;
```

ตรวจไฟล์บน Hugging Face Hub:

```bash
python - <<'PY'
from huggingface_hub import HfApi

repo = "phawitbinabik/causalvla-model-a-sft"
files = HfApi().list_repo_files(repo)

print("Repo:", repo)
print("Files:", len(files))
for path in files:
    print(path)
PY
```

### J. Resume Training จาก Checkpoint เดิม

Checkpoint ของ LeRobot มีทั้ง model, optimizer, scheduler, RNG และ training step จึง resume ต่อได้แบบ sample-exact เมื่อใช้ batch size และจำนวน GPU เท่าเดิม.

กรณีงานหยุดกลางทางและต้องการรันต่อให้ครบ 25,000 steps ใช้ checkpoint ล่าสุดที่บันทึกสำเร็จ เช่น step 20,000:

```bash
conda activate causalvla
cd ~/projects/causalvla

export PYTHONNOUSERSITE=1
export HF_HOME=~/hf_cache/causalvla
export CUDA_VISIBLE_DEVICES=0

lerobot-train \
  --resume=true \
  --config_path=outputs/final/model_a_sft/checkpoints/020000/pretrained_model \
  --steps=25000 \
  2>&1 | tee -a logs/model_a_sft.log
```

`--steps` คือจำนวน step ปลายทางทั้งหมด ไม่ใช่จำนวนที่จะเพิ่ม เช่น resume จาก 20,000 ด้วย `--steps=25000` จะรันเพิ่มอีก 5,000 steps.

Resume จาก Hugging Face Hub เมื่อไฟล์ local หาย โดยระบบจะเลือก checkpoint ล่าสุดใน `checkpoints/<step>/`:

```bash
lerobot-train \
  --resume=true \
  --config_path=phawitbinabik/causalvla-model-a-sft \
  --output_dir=outputs/resumed/model_a_sft \
  --steps=25000 \
  2>&1 | tee logs/model_a_sft_resume.log
```

กรณีโมเดลจบ 25,000 แล้วและต้องการต่อถึง 50,000 steps:

```bash
lerobot-train \
  --resume=true \
  --config_path=outputs/final/model_a_sft/checkpoints/025000/pretrained_model \
  --steps=50000 \
  2>&1 | tee -a logs/model_a_sft.log
```

การ extend แบบนี้จะโหลด scheduler เดิมต่อ และ LR จะอยู่ใกล้ minimum (`2.5e-6`) หลัง step 15,000 หากต้องการรอบ learning-rate ใหม่ควรทำเป็น fine-tuning run ใหม่แทน true resume.

### K. Training Log Retention — เก็บ Log ทุก Full Training

ใช้ชื่อไฟล์มาตรฐานแยกแต่ละโมเดล:

| Model | Full Log | Extracted Step Log | Hub Repo |
|---|---|---|---|
| A | `logs/model_a_sft.log` | `logs/model_a_sft_steps.log` | `causalvla-model-a-sft` |
| B | `logs/model_b_dr.log` | `logs/model_b_dr_steps.log` | `causalvla-model-b-dr` |
| C | `logs/model_c_causal.log` | `logs/model_c_causal_steps.log` | `causalvla-model-c-ours` |
| D | `logs/model_d_no_latent.log` | `logs/model_d_no_latent_steps.log` | `causalvla-model-d-no-latent` |
| E | `logs/model_e_no_action.log` | `logs/model_e_no_action_steps.log` | `causalvla-model-e-no-action` |

การรันครั้งแรกต้องใช้ `tee` เพื่อบันทึก stdout และ stderr:

```bash
lerobot-train ... 2>&1 | tee logs/<model>.log
```

การ resume ต้องใช้ `tee -a` เพื่อ append และไม่เขียนทับ log เดิม:

```bash
lerobot-train --resume=true ... 2>&1 | tee -a logs/<model>.log
```

หลัง train จบ ให้สร้างไฟล์ step metrics ที่อ่านง่าย:

```bash
grep -E 'step:|Checkpoint|End of training|Traceback|RuntimeError|CUDA out of memory' \
  logs/<model>.log > logs/<model>_steps.log
```

เก็บสำเนา log ไว้ใน output directory ของ run:

```bash
mkdir -p outputs/final/<model>/training_logs
cp logs/<model>.log outputs/final/<model>/training_logs/
cp logs/<model>_steps.log outputs/final/<model>/training_logs/
```

บีบอัด full log สำหรับส่งหรือ archive โดยยังเก็บไฟล์ต้นฉบับ:

```bash
gzip -c logs/<model>.log > logs/<model>.log.gz
sha256sum logs/<model>.log logs/<model>.log.gz
```

อัปโหลด log ไปยัง private Hugging Face model repo:

```bash
hf upload phawitbinabik/<repo-id> \
  logs/<model>.log training_logs/<model>.log \
  --repo-type model

hf upload phawitbinabik/<repo-id> \
  logs/<model>_steps.log training_logs/<model>_steps.log \
  --repo-type model
```

#### Archive Full Train Model A ที่จบแล้ว

```bash
conda activate causalvla
cd ~/projects/causalvla

test -s logs/model_a_sft.log
wc -l -c logs/model_a_sft.log

grep -E 'step:|Checkpoint|End of training|Traceback|RuntimeError|CUDA out of memory' \
  logs/model_a_sft.log > logs/model_a_sft_steps.log

mkdir -p outputs/final/model_a_sft/training_logs
cp logs/model_a_sft.log outputs/final/model_a_sft/training_logs/
cp logs/model_a_sft_steps.log outputs/final/model_a_sft/training_logs/

gzip -c logs/model_a_sft.log > logs/model_a_sft.log.gz
sha256sum logs/model_a_sft.log logs/model_a_sft.log.gz \
  > logs/model_a_sft_checksums.txt

hf upload phawitbinabik/causalvla-model-a-sft \
  logs/model_a_sft.log training_logs/model_a_sft.log \
  --repo-type model

hf upload phawitbinabik/causalvla-model-a-sft \
  logs/model_a_sft_steps.log training_logs/model_a_sft_steps.log \
  --repo-type model
```

ไฟล์ที่ควรส่งมาวิเคราะห์หลังแต่ละโมเดลจบ:

1. `<model>_steps.log` — ใช้วิเคราะห์ convergence, gradient, LR, memory และ speed
2. `<model>.log` หรือ `<model>.log.gz` — ใช้ตรวจ config, warning และเหตุการณ์ทั้งหมด
3. `training_step.json` ของ checkpoint สุดท้าย
4. `train_config.json` ของ checkpoint สุดท้าย
