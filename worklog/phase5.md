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
# ── 0a. Install LeRobot ──
!pip install "lerobot[training]" -q

# ── 0b. สร้าง causal_aug package ──
import os, site

# หา site-packages path
sp = site.getsitepackages()[0]
print(f"site-packages: {sp}")

# สร้าง causal_aug/
os.makedirs(f"{sp}/causal_aug", exist_ok=True)
```

```python
%%writefile {sp}/causal_aug/augmentations.py
"""Individual GPU-based augmentation functions (pure PyTorch, no OpenCV)."""

import torch
from torch import Tensor


def adjust_brightness(images: Tensor, factor: Tensor) -> Tensor:
    if factor.ndim == 1:
        factor = factor[:, None, None, None]
    return images * factor


def adjust_contrast(images: Tensor, factor: Tensor) -> Tensor:
    if factor.ndim == 1:
        factor = factor[:, None, None, None]
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return (images - mean) * factor + mean


def adjust_saturation(images: Tensor, factor: Tensor) -> Tensor:
    if factor.ndim == 1:
        factor = factor[:, None, None, None]
    gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
    return (images - gray) * factor + gray


def adjust_hue(images: Tensor, shift: Tensor) -> Tensor:
    angle = shift * 2.0 * 3.14159265
    cos_a = torch.cos(angle)[:, None, None, None]
    sin_a = torch.sin(angle)[:, None, None, None]
    one_third = 1.0 / 3.0
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    new_r = cos_a * r + (1 - cos_a) * (r + g + b) * one_third + sin_a * (b - g) / 1.7320508
    new_g = cos_a * g + (1 - cos_a) * (r + g + b) * one_third + sin_a * (r - b) / 1.7320508
    new_b = cos_a * b + (1 - cos_a) * (r + g + b) * one_third + sin_a * (g - r) / 1.7320508
    return torch.cat([new_r, new_g, new_b], dim=1)


def add_gaussian_noise(images: Tensor, sigma: Tensor) -> Tensor:
    if sigma.ndim == 1:
        sigma = sigma[:, None, None, None]
    return images + torch.randn_like(images) * sigma
```

```python
%%writefile {sp}/causal_aug/gpu_augmenter.py
"""CausalAugmenter: Online GPU counterfactual augmentation engine."""

import torch
import torch.nn as nn
from torch import Tensor
from .augmentations import add_gaussian_noise, adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation


class CausalAugmenter(nn.Module):
    def __init__(self, K=3, intensity=1.0,
                 brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4),
                 saturation_range=(0.5, 1.5), hue_range=(-0.05, 0.05), noise_sigma=0.08):
        super().__init__()
        self.K, self.intensity = K, intensity
        self.brightness_range, self.contrast_range = brightness_range, contrast_range
        self.saturation_range, self.hue_range, self.noise_sigma = saturation_range, hue_range, noise_sigma

    def _sample_uniform(self, low, high, shape, device):
        raw = torch.empty(shape, device=device).uniform_(low, high)
        neutral = (low + high) / 2.0
        return neutral + (raw - neutral) * self.intensity

    def forward(self, images: Tensor) -> Tensor:
        B, device, results = images.shape[0], images.device, []
        for _ in range(self.K):
            aug = images.clone()
            aug = adjust_brightness(aug, self._sample_uniform(*self.brightness_range, (B,), device))
            aug = adjust_contrast(aug, self._sample_uniform(*self.contrast_range, (B,), device))
            if images.shape[1] >= 3:
                aug = adjust_saturation(aug, self._sample_uniform(*self.saturation_range, (B,), device))
                aug = adjust_hue(aug, self._sample_uniform(*self.hue_range, (B,), device))
            sigma = torch.empty(B, device=device).uniform_(0, self.noise_sigma * self.intensity)
            aug = add_gaussian_noise(aug, sigma)
            results.append(aug)
        return torch.stack(results, dim=0)
```

```python
%%writefile {sp}/causal_aug/__init__.py
from .gpu_augmenter import CausalAugmenter
__all__ = ["CausalAugmenter"]
```

```python
# ── 0c. สร้าง causal_vla policy ──
import lerobot
lerobot_policies = os.path.dirname(lerobot.policies.__file__)
causal_dir = os.path.join(lerobot_policies, "causal_vla")
os.makedirs(causal_dir, exist_ok=True)
print(f"causal_vla dir: {causal_dir}")
```

```python
%%writefile {causal_dir}/__init__.py
from .configuration_causal_vla import CausalVLAConfig
from .modeling_causal_vla import CausalVLAPolicy
from .processor_causal_vla import make_causal_vla_pre_post_processors
__all__ = ["CausalVLAConfig", "CausalVLAPolicy", "make_causal_vla_pre_post_processors"]
```

```python
%%writefile {causal_dir}/configuration_causal_vla.py
from dataclasses import dataclass
from lerobot.configs import PreTrainedConfig
from ..smolvla.configuration_smolvla import SmolVLAConfig

@PreTrainedConfig.register_subclass("causal_vla")
@dataclass
class CausalVLAConfig(SmolVLAConfig):
    n_counterfactual: int = 3
    aug_intensity: float = 1.0
    lambda_latent: float = 0.1
    lambda_action: float = 0.1
    lambda_smooth: float = 0.01
    use_latent_loss: bool = True
    use_action_loss: bool = True
```

```python
%%writefile {causal_dir}/modeling_causal_vla.py
"""CausalVLA: SmolVLA + Counterfactual Augmentation + Dual Invariance Losses."""
import torch
import torch.nn.functional as F
from torch import Tensor
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_causal_vla import CausalVLAConfig

class CausalVLAPolicy(SmolVLAPolicy):
    config_class = CausalVLAConfig
    name = "causal_vla"

    def __init__(self, config: CausalVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import CausalAugmenter
        self.augmenter = CausalAugmenter(K=config.n_counterfactual, intensity=config.aug_intensity)

    def _pool_prefix_embs(self, prefix_embs: Tensor) -> Tensor:
        return prefix_embs.mean(dim=1)

    def _augment_images(self, images: list[Tensor]) -> list[list[Tensor]]:
        K = self.config.n_counterfactual
        cf_images = [[] for _ in range(K)]
        for img in images:
            aug = self.augmenter(img.detach())
            for k in range(K):
                cf_images[k].append(aug[k])
        return cf_images

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses_0, prefix_embs_0, v_t_0 = self.model.forward_with_latent(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        original_action_dim = self.config.action_feature.shape[0]
        task_losses = losses_0[:, :, :original_action_dim]
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            task_losses = task_losses * in_episode_bound.unsqueeze(-1)
        task_losses = task_losses[:, :, :self.config.max_action_dim]

        if actions_is_pad is None:
            loss_task = task_losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * task_losses.shape[-1]).clamp_min(1)
            loss_task = task_losses.sum() / num_valid

        loss_dict = {"loss_task": loss_task.item()}
        z_0 = self._pool_prefix_embs(prefix_embs_0)
        a_0 = v_t_0[:, :, :original_action_dim]

        cf_images = self._augment_images(images)
        latent_losses, action_losses = [], []

        for k in range(self.config.n_counterfactual):
            _, prefix_embs_k, v_t_k = self.model.forward_with_latent(
                cf_images[k], img_masks, lang_tokens, lang_masks, state, actions, noise, time)
            z_k = self._pool_prefix_embs(prefix_embs_k)
            a_k = v_t_k[:, :, :original_action_dim]
            if self.config.use_latent_loss:
                latent_losses.append(F.mse_loss(z_0, z_k))
            if self.config.use_action_loss:
                action_losses.append(F.mse_loss(a_0, a_k))

        total_loss = loss_task
        if self.config.use_latent_loss and latent_losses:
            loss_latent = sum(latent_losses) / len(latent_losses)
            total_loss = total_loss + self.config.lambda_latent * loss_latent
            loss_dict["loss_latent"] = loss_latent.item()
        if self.config.use_action_loss and action_losses:
            loss_action = sum(action_losses) / len(action_losses)
            total_loss = total_loss + self.config.lambda_action * loss_action
            loss_dict["loss_action"] = loss_action.item()

        loss_smooth = F.mse_loss(a_0[:, :-1, :], a_0[:, 1:, :])
        total_loss = total_loss + self.config.lambda_smooth * loss_smooth
        loss_dict["loss_smooth"] = loss_smooth.item()
        loss_dict["loss"] = total_loss.item()

        if reduction == "none":
            if actions_is_pad is None:
                per_sample_loss = task_losses.mean(dim=(1, 2))
            else:
                num_valid_per_sample = ((~actions_is_pad).sum(dim=1) * task_losses.shape[-1]).clamp_min(1)
                per_sample_loss = task_losses.sum(dim=(1, 2)) / num_valid_per_sample
            return per_sample_loss, loss_dict
        return total_loss, loss_dict
```

```python
%%writefile {causal_dir}/processor_causal_vla.py
"""CausalVLA uses the same processor pipeline as SmolVLA."""
from typing import Any
import torch
from lerobot.processor import (
    NewLineTaskProcessorStep, PolicyAction, PolicyProcessorPipeline,
    TokenizerProcessorStep, make_default_policy_processor_steps, make_policy_processor_pipelines,
)
from .configuration_causal_vla import CausalVLAConfig

def make_causal_vla_pre_post_processors(config: CausalVLAConfig, dataset_stats=None):
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps = [
        steps.rename_observations, steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to, padding_side="right", max_length=config.tokenizer_max_length),
        steps.to_device, steps.normalize,
    ]
    output_steps = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
```

```python
# ── 0d. Register causal_vla ใน policies/__init__.py ──
init_path = os.path.join(lerobot_policies, "__init__.py")
with open(init_path, "r") as f:
    content = f.read()
if "causal_vla" not in content:
    with open(init_path, "a") as f:
        f.write("\nfrom .causal_vla.configuration_causal_vla import CausalVLAConfig as CausalVLAConfig\n")
    print("✓ Registered causal_vla in policies/__init__.py")
else:
    print("✓ causal_vla already registered")
```

```python
# ── 0e. Patch SmolVLA: เพิ่ม forward_with_latent() ──
import lerobot.policies.smolvla.modeling_smolvla as smolvla_mod
import inspect

if not hasattr(smolvla_mod.VLAFlowMatching, 'forward_with_latent'):
    smolvla_path = inspect.getfile(smolvla_mod)
    
    patch_code = '''

    def forward_with_latent(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        """Like forward() but also returns prefix embeddings and predicted velocity."""
        import torch
        import torch.nn.functional as F
        from .modeling_smolvla import make_att_2d_masks
        
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, prefix_embs, v_t
'''
    
    with open(smolvla_path, "r") as f:
        src = f.read()
    
    # Insert before the last method or at the end of the class
    marker = "    def sample_actions("
    if marker in src:
        src = src.replace(marker, patch_code + "\n" + marker)
        with open(smolvla_path, "w") as f:
            f.write(src)
        print("✓ Patched forward_with_latent() into VLAFlowMatching")
    else:
        print("✗ Could not find insertion point — patch manually")
else:
    print("✓ forward_with_latent() already exists")
```

```python
# ── 0f. Verify ──
# Restart runtime if needed (Colab menu: Runtime > Restart runtime) แล้วรัน cell นี้
from causal_aug import CausalAugmenter
from lerobot.policies.causal_vla.configuration_causal_vla import CausalVLAConfig
import torch

aug = CausalAugmenter(K=3)
out = aug(torch.randn(2, 3, 256, 256))
print(f"✓ CausalAugmenter: {out.shape}")  # [3, 2, 3, 256, 256]
print(f"✓ CausalVLAConfig: {CausalVLAConfig}")
print("Setup complete!")
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
