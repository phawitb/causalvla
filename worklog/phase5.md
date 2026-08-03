# Phase 5: Training ทุก Experiment Variants

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
