# Phase 2: CausalVLA Architecture

> **วันที่:** 2 สิงหาคม 2026
> **สถานะ:** COMPLETED
> **เครื่อง:** Mac M2 (local)

---

## Step 2.1 — สร้าง `causal_aug` Python Package

### สิ่งที่ทำ
สร้าง local Python package สำหรับ online GPU counterfactual augmentation ใช้ pure PyTorch ops ทั้งหมด (ไม่ใช้ OpenCV)

### โครงสร้าง

```
CausalVLA/causal_aug/
├── pyproject.toml
└── causal_aug/
    ├── __init__.py
    ├── augmentations.py      # Individual GPU aug functions
    └── gpu_augmenter.py      # CausalAugmenter class
```

### ติดตั้ง

```bash
conda run -n lerobot2 pip install -e causal_aug/
```

### CausalAugmenter API

```python
from causal_aug import CausalAugmenter
import torch

augmenter = CausalAugmenter(K=3, intensity=1.0)
images = torch.randn(2, 3, 256, 256)        # [B, C, H, W]
counterfactuals = augmenter(images)           # [K, B, C, H, W] = [3, 2, 3, 256, 256]
```

### Augmentation Functions (pure PyTorch GPU ops)

| Function | Input | คำอธิบาย |
|----------|-------|---------|
| `adjust_brightness` | factor [B] | คูณ brightness ทั้งภาพ |
| `adjust_contrast` | factor [B] | ปรับ contrast รอบค่าเฉลี่ย |
| `adjust_saturation` | factor [B] | ปรับ saturation (ITU-R 601 luma) |
| `adjust_hue` | shift [B] | หมุน hue ใน RGB space (Rodrigues' rotation) |
| `add_gaussian_noise` | sigma [B] | เพิ่ม Gaussian noise |

### Default Ranges

| Augmentation | Range | Neutral |
|-------------|-------|---------|
| Brightness | 0.6 — 1.4 | 1.0 |
| Contrast | 0.6 — 1.4 | 1.0 |
| Saturation | 0.5 — 1.5 | 1.0 |
| Hue | -0.05 — 0.05 | 0.0 |
| Noise sigma | 0 — 0.08 | 0.0 |

### ทดสอบ

```python
from causal_aug import CausalAugmenter
import torch

a = CausalAugmenter(K=3)
out = a(torch.randn(2, 3, 256, 256))
print(out.shape)  # torch.Size([3, 2, 3, 256, 256])
# OK
```

### Insight
- `intensity` parameter ทำหน้าที่ interpolate ระหว่าง neutral (ไม่ augment) กับ full range — ทำให้ค่อยๆ เพิ่ม augmentation strength ได้
- ใช้ Rodrigues' rotation formula สำหรับ hue shift แทนการแปลง RGB→HSV→RGB ซึ่งเร็วกว่าบน GPU
- ทุก function รับ per-sample parameter `[B]` ทำให้แต่ละ sample ใน batch ได้ augmentation ต่างกัน

---

## Step 2.2 — เพิ่ม `forward_with_latent()` ให้ VLAFlowMatching

### สิ่งที่ทำ
เพิ่ม method ใหม่ใน `VLAFlowMatching` (smolvla/modeling_smolvla.py) ที่ return ทั้ง losses, prefix embeddings, และ predicted velocity — เพื่อให้ CausalVLA ดึง latent features ได้โดยไม่ต้องรัน model ซ้ำ

### ไฟล์ที่แก้ไข

```
lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

### Method Signature

```python
def forward_with_latent(
    self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
        losses: [B, chunk_size, max_action_dim] per-element MSE losses
        prefix_embs: [B, L_prefix, D] prefix embeddings (images + language + state)
        v_t: [B, chunk_size, max_action_dim] predicted velocity
    """
```

### Insight
- เหมือน `forward()` ทุกประการ แต่ return เพิ่ม `prefix_embs` และ `v_t`
- ไม่แก้ไข method เดิมใดๆ — เป็นการเพิ่มใหม่ล้วนๆ ไม่กระทบ SmolVLA ปกติ
- `prefix_embs` คือ concatenation ของ image embeddings + language embeddings + state projection → ใช้เป็น latent representation z สำหรับ invariance loss
- `v_t` คือ predicted velocity จาก action expert → ใช้เป็น action proxy สำหรับ action consistency loss

---

## Step 2.3 — สร้าง CausalVLA Policy

### สิ่งที่ทำ
สร้าง CausalVLA policy ที่ inherit จาก SmolVLAPolicy โดยเพิ่มเฉพาะ training forward pass (counterfactual augmentation + dual invariance losses)

### ไฟล์ที่สร้าง

```
lerobot/src/lerobot/policies/causal_vla/
├── __init__.py
├── configuration_causal_vla.py
├── modeling_causal_vla.py
└── processor_causal_vla.py
```

### CausalVLAConfig

```python
@PreTrainedConfig.register_subclass("causal_vla")
@dataclass
class CausalVLAConfig(SmolVLAConfig):
    n_counterfactual: int = 3        # K counterfactual images
    aug_intensity: float = 1.0

    lambda_latent: float = 0.1      # น้ำหนัก L_latent
    lambda_action: float = 0.1      # น้ำหนัก L_action
    lambda_smooth: float = 0.01     # น้ำหนัก L_smooth

    use_latent_loss: bool = True    # False = Model D (ablation)
    use_action_loss: bool = True    # False = Model E (ablation)
```

### CausalVLAPolicy Forward Pass

```
Training forward():
1. forward_with_latent(original_images) → losses_0, prefix_embs_0 (z_0), v_t_0 (a_0)
2. L_task = flow matching MSE (เหมือน SmolVLA)
3. augmenter(images) → K counterfactual image sets
4. for k in 1..K:
     forward_with_latent(cf_images[k]) → prefix_embs_k (z_k), v_t_k (a_k)
     L_latent += MSE(z_0, z_k)
     L_action += MSE(a_0, a_k)
5. L_smooth = MSE(a_0[t], a_0[t+1])  (temporal smoothing)
6. L_total = L_task + λ_lat·L_lat + λ_act·L_act + λ_smooth·L_smooth
```

### Architecture Decision: Inherit vs Compose

เลือก **inherit จาก SmolVLAPolicy** เพราะ:
- CausalVLA ใช้ inference path เหมือน SmolVLA 100% (select_action, predict_action_chunk, sample_actions)
- Override เฉพาะ `forward()` สำหรับ training
- ไม่ต้อง duplicate โค้ด ~800 บรรทัด
- Checkpoint structure เหมือนกัน — load/save ผ่าน HuggingFace Hub ได้เลย

### Latent Representation Design

```python
def _pool_prefix_embs(self, prefix_embs):
    """Mean-pool prefix embeddings → fixed-size latent vector [B, D]"""
    return prefix_embs.mean(dim=1)
```

- `prefix_embs` = concat(image_embs, language_embs, state_emb) จาก `embed_prefix()`
- Mean pooling ให้ได้ vector `[B, D]` ที่ represent ทั้ง scene
- ใช้ MSE ระหว่าง z_0 กับ z_k เพื่อบังคับให้ latent representation ไม่เปลี่ยนเมื่อ visual input เปลี่ยน

### Processor

- ใช้ processor เดียวกับ SmolVLA ทุกประการ (tokenizer, normalizer, etc.)
- Factory function: `make_causal_vla_pre_post_processors()`

---

## Step 2.4 — ลงทะเบียน Policy ใน LeRobot

### ไฟล์ที่แก้ไข

```
lerobot/src/lerobot/policies/__init__.py
```

### สิ่งที่เพิ่ม

```python
from .causal_vla.configuration_causal_vla import CausalVLAConfig as CausalVLAConfig
# + เพิ่ม "CausalVLAConfig" ใน __all__
```

### ทดสอบ Registration

```python
from lerobot.policies.factory import make_policy_config, get_policy_class

cfg = make_policy_config('causal_vla')
# <class 'CausalVLAConfig'>

cls = get_policy_class('causal_vla')
# <class 'CausalVLAPolicy'>
# Name: causal_vla
# Config class: CausalVLAConfig
```

### Insight
- LeRobot v0.6.1 ใช้ convention-based auto-discovery:
  - `configuration_causal_vla.py` → `modeling_causal_vla.py` (เปลี่ยน prefix `configuration_` เป็น `modeling_`)
  - `CausalVLAConfig` → `CausalVLAPolicy` (เปลี่ยน suffix `Config` เป็น `Policy`)
  - `processor_causal_vla.py` → `make_causal_vla_pre_post_processors()` (ตาม pattern `make_{type}_pre_post_processors`)
- ต้อง import config ใน `policies/__init__.py` เพื่อ trigger `@register_subclass` decorator

---

## Step 2.5 — Smoke Test Training

### Command

```bash
conda run -n lerobot2 python -m lerobot.scripts.lerobot_train \
    --policy.type=causal_vla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/causal_vla_smoke_test \
    --job_name=causal_vla_smoke \
    --policy.device=mps \
    --batch_size=2 \
    --steps=3 \
    --save_freq=3 \
    --env_eval_freq=0 \
    --policy.push_to_hub=false \
    --num_workers=0
```

### ผลลัพธ์

```
INFO: Creating policy
INFO: num_learnable_params=99880992 (100M)
INFO: num_total_params=450046176 (450M)

Training:  33%|███▎      | 1/3 [00:07]
Training:  67%|██████▋   | 2/3 [00:09]
Training: 100%|██████████| 3/3 [00:10] → Checkpoint saved at step 3
INFO: End of training
```

| Metric | Value |
|--------|-------|
| Policy type | causal_vla |
| Total params | 450M |
| Learnable params | 100M |
| Training speed | ~4 sec/step (MPS, batch=2) |
| Total time | ~12 sec (3 steps) |
| Checkpoint | saved at step 3 |
| CausalVLA fields | n_counterfactual=3, lambda_latent=0.1, lambda_action=0.1, lambda_smooth=0.01 |

### Config ที่แสดงใน Training Output

```
'type': 'causal_vla',
'n_counterfactual': 3,
'aug_intensity': 1.0,
'lambda_action': 0.1,
'lambda_latent': 0.1,
'lambda_smooth': 0.01,
'use_action_loss': True,
'use_latent_loss': True,
```

ยืนยันว่า CausalVLA-specific config fields ถูกส่งผ่านไปถูกต้องทั้งหมด

---

## สรุป Phase 2

| Step | Status | หมายเหตุสำคัญ |
|------|--------|--------------|
| 2.1 causal_aug package | **PASS** | CausalAugmenter [K,B,C,H,W] output, pure PyTorch GPU ops |
| 2.2 forward_with_latent() | **PASS** | เพิ่ม method ใหม่ใน VLAFlowMatching, ไม่กระทบ SmolVLA |
| 2.3 CausalVLA policy | **PASS** | CausalVLAPolicy(SmolVLAPolicy) + dual invariance losses |
| 2.4 Registration | **PASS** | `make_policy_config('causal_vla')` สำเร็จ |
| 2.5 Smoke test | **PASS** | 3 steps training สำเร็จ, checkpoint saved |

### ไฟล์ที่สร้าง/แก้ไข

| Action | ไฟล์ |
|--------|------|
| CREATE | `causal_aug/pyproject.toml` |
| CREATE | `causal_aug/causal_aug/__init__.py` |
| CREATE | `causal_aug/causal_aug/augmentations.py` |
| CREATE | `causal_aug/causal_aug/gpu_augmenter.py` |
| CREATE | `lerobot/src/lerobot/policies/causal_vla/__init__.py` |
| CREATE | `lerobot/src/lerobot/policies/causal_vla/configuration_causal_vla.py` |
| CREATE | `lerobot/src/lerobot/policies/causal_vla/modeling_causal_vla.py` |
| CREATE | `lerobot/src/lerobot/policies/causal_vla/processor_causal_vla.py` |
| MODIFY | `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` (เพิ่ม forward_with_latent) |
| MODIFY | `lerobot/src/lerobot/policies/__init__.py` (register CausalVLAConfig) |

### สิ่งที่ยังไม่ได้ทำ (Phase 3)
- Unit test: ทดสอบ loss components แยกแต่ละตัว (L_task, L_latent, L_action, L_smooth)
- ทดสอบ gradient flow ผ่านทุก trainable parameter
- ทดสอบ ablation configs (use_latent_loss=False, use_action_loss=False)
- ทดสอบว่า loss ไม่มี NaN
