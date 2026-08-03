# Phase 1: Environment Setup & Sanity Check

> **วันที่:** 2 สิงหาคม 2026
> **สถานะ:** COMPLETED
> **เครื่อง:** Mac M2 (local)

---

## Step 1.1 — Verify Installation

### สิ่งที่ทำ
ตรวจสอบ lerobot v0.6.1 และ dependencies ทั้งหมดบน Mac M2

### Commands

```bash
conda activate lerobot2
python --version
# Python 3.12.13

python -c "import lerobot; print('lerobot version:', lerobot.__version__)"
# lerobot version: 0.6.1
```

```python
import torch
print('PyTorch:', torch.__version__)        # 2.11.0
print('MPS available:', torch.backends.mps.is_available())  # True
print('CUDA available:', torch.cuda.is_available())         # False

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
print('SmolVLAConfig loaded OK')

import huggingface_hub
print('HF Hub:', huggingface_hub.__version__)  # 1.26.0

from huggingface_hub import HfApi
api = HfApi()
info = api.whoami()
print('HF logged in as:', info['name'])  # phawitbinabik
```

### ผลลัพธ์
| Item | Value |
|------|-------|
| lerobot | 0.6.1 |
| Python | 3.12.13 |
| PyTorch | 2.11.0 |
| MPS | Available |
| CUDA | Not available (Mac) |
| HF Hub | 1.26.0 |
| HF Account | phawitbinabik |

### Warning (ไม่กระทบ)
```
objc: Class AVFFrameReceiver is implemented in both
  cv2/.dylibs/libavdevice.61.3.100.dylib and
  av/.dylibs/libavdevice.61.3.100.dylib
```
เกิดจาก cv2 กับ PyAV มี libavdevice ซ้ำกัน — ไม่มีผลต่อการทำงาน

---

## Step 1.2 — Smoke Test: Training Pipeline

### สิ่งที่ทำ
รัน SmolVLA training 10 steps บน MPS (Mac M2) เพื่อยืนยัน pipeline ทำงานได้ครบ (data load, forward, backward, checkpoint save)

### Command

```bash
conda run -n lerobot2 python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/debug_smoke_test \
    --job_name=smoke_test \
    --policy.device=mps \
    --batch_size=2 \
    --steps=10 \
    --save_freq=5 \
    --env_eval_freq=0 \
    --policy.push_to_hub=false \
    --num_workers=0
```

### ผลลัพธ์

```
INFO: dataset.num_frames=52970 (53K)
INFO: dataset.num_episodes=432
INFO: Effective batch size: 2 x 1 = 2
INFO: num_learnable_params=99880992 (100M)
INFO: num_total_params=450046176 (450M)

Training:  10%|█         | 1/10 [00:29<04:28, 29.83s/step]
Training:  50%|█████     | 5/10 [02:30] → Checkpoint saved at step 5
Training: 100%|██████████| 10/10 [05:03] → Checkpoint saved at step 10
INFO: End of training
```

| Metric | Value |
|--------|-------|
| Training speed | ~30 sec/step (MPS) |
| Total time | ~5 minutes |
| Total params | 450M |
| Learnable params | 100M (train_expert_only=True) |
| Dataset | 432 episodes, 52,970 frames |
| Checkpoints saved | step 5, step 10 |

### Insight
- `conda run -n lerobot2` ทำให้ working directory เป็น lerobot/ ดังนั้น output ที่ใช้ relative path จะถูก save ที่ `lerobot/outputs/` ไม่ใช่ project root — **ใช้ absolute path แก้ได้**
- `num_workers=0` จำเป็นเพราะ spawn context กับ Mac M2 อาจมีปัญหากับ ffmpeg workers ในบาง setup
- SmolVLA default: `freeze_vision_encoder=True`, `train_expert_only=True` → เทรนแค่ action expert (~100M จาก 450M params)
- LR scheduler auto-scale: warmup 1000→0, decay 30000→10 เพราะ steps น้อยมาก

---

## Step 1.3 — Verify Checkpoint Structure

### สิ่งที่ทำ
ตรวจสอบว่า checkpoint มีไฟล์ครบและ load กลับมาได้

### Checkpoint Location
```
lerobot/outputs/train/debug_smoke_test/checkpoints/
├── 000005/
│   ├── pretrained_model/
│   │   ├── config.json
│   │   ├── model.safetensors          (1.1 GB)
│   │   ├── train_config.json
│   │   ├── policy_preprocessor.json
│   │   ├── policy_postprocessor.json
│   │   ├── policy_preprocessor_step_5_normalizer_processor.safetensors
│   │   └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
│   └── training_state/
│       ├── optimizer_state.safetensors
│       ├── scheduler_state.json
│       ├── training_step.json
│       ├── rng_state.safetensors
│       └── optimizer_param_groups.json
└── 000010/
    ├── pretrained_model/   (same structure)
    └── training_state/     (same structure)
```

### Load Test

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

model = SmolVLAPolicy.from_pretrained(
    '/Users/phawit/Projects/CausalVLA/lerobot/outputs/train/debug_smoke_test/checkpoints/000010/pretrained_model'
)
print('Model loaded OK!')
print('Type:', type(model).__name__)   # SmolVLAPolicy
print('Params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')  # 450.046176 M
```

### ผลลัพธ์
- Checkpoint structure ถูกต้อง: มี `config.json`, `model.safetensors`, processor files ครบ
- Load กลับมาสำเร็จ: SmolVLAPolicy 450M params
- มี `training_state/` แยกต่างหาก สำหรับ resume training

### Insight
- LeRobot v0.6.1 แยก `pretrained_model/` กับ `training_state/` — pretrained_model คือส่วนที่ push ขึ้น Hub ได้เลย
- `model.safetensors` = 1.1 GB สำหรับ 450M params (ใช้ float32 ไม่ได้ quantize)

---

## Step 1.4 — Hub Round-Trip Test

### สิ่งที่ทำ
Push checkpoint ขึ้น HuggingFace Hub (private repo) แล้ว download กลับมาเพื่อทดสอบ transfer pipeline

### Push to Hub

```bash
# หมายเหตุ: huggingface-cli deprecated แล้ว ใช้ hf แทน
conda run -n lerobot2 hf upload phawitbinabik/causalvla-smoke-test \
    /Users/phawit/Projects/CausalVLA/lerobot/outputs/train/debug_smoke_test/checkpoints/000010/pretrained_model \
    --private
```

```
Found 7 files to upload
url=https://huggingface.co/phawitbinabik/causalvla-smoke-test/commit/c82cae39d05d357edcb623ea1b03f24271944e6c
```

### Download from Hub

```python
from huggingface_hub import snapshot_download
path = snapshot_download('phawitbinabik/causalvla-smoke-test')
print('Downloaded to:', path)
# /Users/phawit/.cache/huggingface/hub/models--phawitbinabik--causalvla-smoke-test/snapshots/c82cae39...

import os
print('Files:', os.listdir(path))
# ['model.safetensors', 'config.json', 'policy_postprocessor.json',
#  'policy_preprocessor.json', '.gitattributes',
#  'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
#  'train_config.json',
#  'policy_preprocessor_step_5_normalizer_processor.safetensors']
```

### Load from Hub

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
model = SmolVLAPolicy.from_pretrained('phawitbinabik/causalvla-smoke-test')
print('Download + Load from Hub OK!')
# Params: 450.046176 M
```

### ผลลัพธ์
- Push สำเร็จ → `phawitbinabik/causalvla-smoke-test` (private)
- Download สำเร็จ → cached ที่ `~/.cache/huggingface/hub/`
- Load จาก Hub ตรงๆ ด้วย `from_pretrained('phawitbinabik/causalvla-smoke-test')` ได้เลย

### Insight
- `huggingface-cli` ถูก deprecated แล้วใน HF Hub 1.26.0 → ใช้ `hf` CLI แทน
- `SmolVLAPolicy.from_pretrained()` รับทั้ง local path และ Hub repo_id
- ไฟล์ cache อยู่ที่ `~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{commit}/`

---

## Step 1.5 — Colab Training Smoke Test (CUDA)

> **Colab Notebook:** [colab.research.google.com/drive/1AJtPUmx7zAnwkync2I2FW87wQ0AnlzUG](https://colab.research.google.com/drive/1AJtPUmx7zAnwkync2I2FW87wQ0AnlzUG?usp=sharing)

### สิ่งที่ทำ
ทดสอบ training pipeline บน Google Colab (Tesla T4 GPU) เพื่อยืนยันว่า CUDA training ทำงานได้

### Setup

```bash
# ตรวจสอบ GPU
import torch
print("CUDA Available:", torch.cuda.is_available())      # True
print("GPU Device Name:", torch.cuda.get_device_name(0)) # Tesla T4
print("VRAM Available (GB):", 15.64)

# Clone และติดตั้ง
!git clone https://github.com/huggingface/lerobot.git
%cd lerobot
!pip install -e ".[libero]" -q
!pip install huggingface_hub wandb -q
!pip install num2words inflect sentencepiece timm accelerate einops
```

### Colab Environment

```
lerobot-info:
- LeRobot version: 0.6.1
- Platform: Linux-6.6.122+-x86_64-with-glibc2.35
- Python version: 3.12.13
- PyTorch version: 2.11.0+cu128
- Cuda version: 12.8
- GPU model: Tesla T4
- git commit: adccdea1cfbec83ed98263feb7e59f7d047c5692 (main)
```

### Training Command

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/smolvla_colab_test \
    --job_name=smolvla_test \
    --policy.device=cuda \
    --batch_size=8 \
    --steps=100 \
    --save_freq=50 \
    --env_eval_freq=0 \
    --policy.push_to_hub=false
```

### ผลลัพธ์

```
INFO: dataset.num_frames=52970 (53K)
INFO: dataset.num_episodes=432
INFO: Effective batch size: 8 x 1 = 8
INFO: num_learnable_params=99880992 (100M)
INFO: num_total_params=450046176 (450M)
INFO: Auto-scaling LR scheduler: warmup 1000 → 3, decay 30000 → 100

Training:  50%  50/100 [02:55] → Checkpoint saved at step 50
Training: 100% 100/100 [05:00] → Checkpoint saved at step 100
INFO: End of training
```

| Metric | Value |
|--------|-------|
| Training speed | ~3 sec/step (CUDA T4) |
| Total time | ~5 minutes (100 steps) |
| Batch size | 8 |
| Total params | 450M |
| Learnable params | 100M |
| Checkpoints saved | step 50, step 100 |

### Insight
- **CUDA T4 เร็วกว่า MPS ~10x**: ~3 sec/step (CUDA, batch=8) vs ~30 sec/step (MPS, batch=2)
- Colab แนะนำ `num_workers=2` แต่ default ตั้งไว้ 4 → มี warning แต่ไม่ error
- LR scheduler auto-scale ทำงานเหมือนกัน: warmup 1000→3, decay 30000→100 เพราะ steps=100
- Flax deprecation warning จาก Diffusers — ไม่กระทบ
- ใช้ lerobot จาก `main` branch (commit `adccdea`) ซึ่งเป็น v0.6.1

---

## สรุป Phase 1

| Step | Status | หมายเหตุสำคัญ |
|------|--------|--------------|
| 1.1 Dependencies | **PASS** | lerobot 0.6.1, PyTorch 2.11, MPS, HF Hub OK |
| 1.2 Training Smoke Test (Local) | **PASS** | 10 steps, ~30s/step MPS, checkpoint saved |
| 1.3 Checkpoint Verify | **PASS** | config.json + model.safetensors (1.1GB) + load OK |
| 1.4 Hub Round-Trip | **PASS** | push/pull/load จาก Hub สำเร็จ |
| 1.5 Training Smoke Test (Colab) | **PASS** | 100 steps, ~3s/step CUDA T4, checkpoint saved |

### ปัญหาที่เจอและวิธีแก้

1. **`conda run` ทำให้ cwd เปลี่ยน** → ใช้ absolute path สำหรับ output_dir
2. **`huggingface-cli` deprecated** → ใช้ `hf` CLI แทน
3. **libavdevice duplicate warning** → ไม่กระทบ ไม่ต้องแก้
4. **Colab num_workers warning** → แนะนำ 2 แต่ default 4, ไม่ error

### สิ่งที่ยังไม่ได้ทำ (ต้องทำบน Colab)
- Eval บน LIBERO simulator (ต้อง MuJoCo + GPU render)
