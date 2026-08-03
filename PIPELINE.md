# CausalVLA — Experimental Development Pipeline

> **LeRobot v0.6.1** | **LIBERO Benchmark** | **SmolVLA Base Model**
>
> แผนการพัฒนาและทดลองเชิงรุก (Step-by-Step Execution Pipeline) สำหรับงานวิจัย CausalVLA
> พัฒนาบน Mac M2 (debug/eval) + Google Colab (training) + HuggingFace Hub (model/data transfer)

---

## สารบัญ

- [ภาพรวม Experimental Design](#ภาพรวม-experimental-design)
- [Phase 1: Environment Setup & Sanity Check](#phase-1-environment-setup--sanity-check)
- [Phase 2: พัฒนาสถาปัตยกรรม CausalVLA Policy](#phase-2-พัฒนาสถาปัตยกรรม-causalvla-policy)
- [Phase 3: Unit Test — Loss & Gradient Flow](#phase-3-unit-test--loss--gradient-flow)
- [Phase 4: OOD Perturbation Engine สำหรับ Evaluation](#phase-4-ood-perturbation-engine-สำหรับ-evaluation)
- [Phase 5: Training ทุก Experiment Variants](#phase-5-training-ทุก-experiment-variants)
- [Phase 6: Evaluation & Results Collection](#phase-6-evaluation--results-collection)
- [โครงสร้างไฟล์โปรเจกต์](#โครงสร้างไฟล์โปรเจกต์)
- [Workflow Diagram: Local ↔ Hub ↔ Colab](#workflow-diagram-local--hub--colab)

---

## ภาพรวม Experimental Design

### Models ที่ต้องเปรียบเทียบ (ทั้งหมดใช้ SmolVLA เป็น Base)

| Model ID | ชื่อ | คำอธิบาย |
|----------|------|----------|
| **A** | Standard SFT (Baseline) | SmolVLA fine-tune ปกติบน clean data ไม่มี augmentation/causal loss |
| **B** | Domain Randomization | SmolVLA fine-tune บน augmented data (จำลอง poor env) ไม่มี causal loss |
| **C** | CausalVLA (Ours) | SmolVLA + **Online GPU Counterfactual Augmentation** + **Dual Invariance Loss** ครบทุกตัว |
| **D** | Ablation w/o L_latent | CausalVLA ตัด Counterfactual Latent Invariance Loss ออก |
| **E** | Ablation w/o L_action | CausalVLA ตัด Action Consistency Loss ออก |

### Loss Function ของ CausalVLA

```
L_total = L_task + λ_latent · L_latent + λ_action · L_action + λ_smooth · L_smooth
```

| Loss | สูตร | ความหมาย |
|------|------|----------|
| `L_task` | Flow matching loss (จาก SmolVLA เดิม) | เรียนรู้ action จาก demonstration |
| `L_latent` | `MSE(z_0, z_k)` สำหรับ k=1..K | Latent representation ต้องคงที่แม้ภาพเปลี่ยน (causal invariance) |
| `L_action` | `MSE(a_0, a_k)` สำหรับ k=1..K | Action output ต้องเหมือนกันแม้ visual input ต่างกัน |
| `L_smooth` | `MSE(a_t, a_{t+1})` | ลด action jitter ให้การเคลื่อนไหวนุ่มนวล |

### หลักการสำคัญ: แยก Augmentation ออกเป็น 2 ระบบ

| ระบบ | ใช้ตอนไหน | ทำอะไร | ที่มา |
|------|----------|--------|------|
| **Offline Dataset Augmentation** | สร้าง dataset สำหรับ Model B (Domain Randomization) | สร้าง dataset ใหม่ที่มี camera shift, light change, robot noise | `Lerobot-Dataset-Manager` |
| **Online GPU Counterfactual Augmentation** | ระหว่าง training Model C/D/E (CausalVLA) | สร้าง K counterfactual images แบบ on-the-fly บน GPU | `causal_aug` package (สร้างใหม่) |

---

## Phase 1: Environment Setup & Sanity Check

> **เป้าหมาย:** ยืนยันว่า LeRobot + LIBERO + HuggingFace Hub ทำงานร่วมกันได้ทั้งบน local (Mac M2) และ Colab (GPU)

### Step 1.1 — ติดตั้ง Dependencies

**Mac M2 (Local):**
```bash
conda activate lerobot2
# lerobot v0.6.1 ติดตั้งแล้ว ตรวจสอบ:
lerobot-info
```

**Google Colab:**
```bash
pip install -e ".[libero,smolvla]"
# ตรวจสอบ CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 1.2 — Smoke Test: Training Pipeline

รัน SmolVLA training ~10 steps บน local เพื่อยืนยัน data load + forward/backward pass:

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/debug_smoke_test \
    --job_name=smoke_test \
    --policy.device=cpu \
    --batch_size=2 \
    --steps=10 \
    --save_freq=5 \
    --env_eval_freq=0 \
    --policy.push_to_hub=false
```

**ตรวจสอบ:**
- [ ] Data download + load สำเร็จ
- [ ] Forward pass ไม่มี error
- [ ] Backward pass + optimizer step สำเร็จ
- [ ] Checkpoint saved ที่ `outputs/train/debug_smoke_test/checkpoints/`
- [ ] Checkpoint มีไฟล์ `config.json`, `model.safetensors`

### Step 1.3 — Smoke Test: Evaluation Pipeline

รัน evaluation บน LIBERO simulator (ต้องทำบน **Colab** เพราะ LIBERO ต้องใช้ MuJoCo + GPU render):

```bash
lerobot-eval \
    --policy.path=outputs/train/debug_smoke_test/checkpoints/000010/pretrained_model \
    --env.type=libero \
    --eval.batch_size=1 \
    --eval.n_episodes=2 \
    --policy.device=cuda
```

**ตรวจสอบ:**
- [ ] LIBERO simulator สร้าง environment ได้
- [ ] Policy inference ทำงาน (แม้ผลจะไม่ดีเพราะ train แค่ 10 steps)
- [ ] ได้ success rate กลับมา (คาดว่า 0% ตอนนี้)
- [ ] Video render ได้ (ถ้าเปิด)

### Step 1.4 — Hub Round-Trip Test

ทดสอบ push/pull model ระหว่าง local ↔ Hub ↔ Colab:

```bash
# Push (local → Hub)
huggingface-cli upload <username>/causalvla-smoke-test \
    outputs/train/debug_smoke_test/checkpoints/000010/pretrained_model \
    --private

# Pull (Colab ← Hub)
huggingface-cli download <username>/causalvla-smoke-test
```

**ตรวจสอบ:**
- [ ] Upload สำเร็จ
- [ ] Download สำเร็จ
- [ ] Model load ได้หลัง download

---

## Phase 2: พัฒนาสถาปัตยกรรม CausalVLA Policy

> **เป้าหมาย:** สร้าง CausalVLA policy + Counterfactual Augmentation Engine

### Step 2.1 — สร้าง `causal_aug` Python Package (Modular Augmentation)

แยก augmentation engine ออกเป็น local package ที่ pip install ได้:

```
CausalVLA/
└── causal_aug/                          # ← Local Python Package
    ├── pyproject.toml
    ├── causal_aug/
    │   ├── __init__.py
    │   ├── gpu_augmenter.py             # CausalAugmenter (GPU-based, online)
    │   ├── augmentations.py             # แต่ละ augmentation function
    │   └── ood_wrapper.py               # OOD Gym Wrapper (ใช้ตอน eval, Phase 4)
    └── tests/
        └── test_augmenter.py
```

**ติดตั้ง:**
```bash
pip install -e /path/to/CausalVLA/causal_aug
```

**การใช้งาน:**
```python
from causal_aug import CausalAugmenter

augmenter = CausalAugmenter(K=3, intensity=1.0).to("cuda")
# input: original observation tensor [B, C, H, W]
# output: K counterfactual observations [K, B, C, H, W]
counterfactual_obs = augmenter(original_obs)
```

#### `gpu_augmenter.py` — CausalAugmenter Class

```python
class CausalAugmenter(nn.Module):
    """
    Online GPU Counterfactual Augmentation Engine.
    สร้าง K ภาพสมมติ (counterfactual) จากภาพต้นฉบับ
    เพื่อใช้คำนวณ Dual Invariance Loss

    Augmentation techniques (ทั้งหมดทำบน GPU tensor):
    - Color Jitter: brightness, contrast, saturation, hue
    - Lighting Shift: global brightness scale
    - Gaussian Noise: additive noise
    - (Optional) Perspective/Affine transform
    """

    def __init__(self, K=3, intensity=1.0):
        self.K = K                  # จำนวน counterfactual images
        self.intensity = intensity  # ความรุนแรงของ augmentation

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: [B, C, H, W] original observation
        Returns:
            augmented: [K, B, C, H, W] counterfactual observations
        """
        ...
```

> **อ้างอิง:** เทคนิค augmentation อ้างอิงจาก `Lerobot-Dataset-Manager/main.py`
> (`_augp_camera`, `_augp_light`, `_aug_frame` functions) แต่แปลงเป็น **pure PyTorch GPU ops**
> แทน OpenCV+NumPy เพื่อความเร็ว

### Step 2.2 — สร้าง CausalVLA Policy ใน LeRobot

สร้างไฟล์ policy ใหม่ใน lerobot:

```
lerobot/src/lerobot/policies/causal_vla/
├── __init__.py
├── configuration_causal_vla.py    # CausalVLAConfig
└── modeling_causal_vla.py         # CausalVLAPolicy
```

#### `configuration_causal_vla.py`

```python
@PreTrainedConfig.register_subclass("causal_vla")
@dataclass
class CausalVLAConfig(SmolVLAConfig):
    """CausalVLA = SmolVLA + Counterfactual Augmentation + Dual Invariance Loss"""

    # Counterfactual augmentation
    n_counterfactual: int = 3           # K: จำนวนภาพสมมติ
    aug_intensity: float = 1.0          # ความรุนแรงของ augmentation

    # Loss weights
    lambda_latent: float = 0.1          # น้ำหนัก L_latent
    lambda_action: float = 0.1          # น้ำหนัก L_action
    lambda_smooth: float = 0.01         # น้ำหนัก L_smooth

    # Ablation flags
    use_latent_loss: bool = True        # False = Model D (ablation)
    use_action_loss: bool = True        # False = Model E (ablation)
```

#### `modeling_causal_vla.py`

```python
class CausalVLAPolicy(SmolVLAPolicy):
    """
    CausalVLA extends SmolVLA with:
    1. Online counterfactual augmentation (GPU)
    2. Dual invariance loss (latent + action consistency)
    3. Action smoothness loss
    """

    config_class = CausalVLAConfig
    name = "causal_vla"

    def __init__(self, config: CausalVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import CausalAugmenter
        self.augmenter = CausalAugmenter(
            K=config.n_counterfactual,
            intensity=config.aug_intensity,
        )

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        # 1. Original forward → L_task (flow matching loss จาก SmolVLA)
        loss_task, loss_dict = super().forward(batch, noise, time, reduction)

        # 2. Extract original latent z_0 และ action a_0
        z_0 = self._extract_latent(batch)           # [B, D]
        a_0 = self._extract_action_prediction(batch) # [B, T, A]

        # 3. Generate K counterfactual observations
        images = self._get_images_from_batch(batch)  # [B, C, H, W]
        cf_images = self.augmenter(images)            # [K, B, C, H, W]

        # 4. Forward counterfactual → z_k, a_k
        z_ks, a_ks = [], []
        for k in range(self.config.n_counterfactual):
            cf_batch = self._replace_images(batch, cf_images[k])
            z_k = self._extract_latent(cf_batch)
            a_k = self._extract_action_prediction(cf_batch)
            z_ks.append(z_k)
            a_ks.append(a_k)

        # 5. Compute invariance losses
        L_latent = sum(F.mse_loss(z_0, z_k) for z_k in z_ks) / len(z_ks)
        L_action = sum(F.mse_loss(a_0, a_k) for a_k in a_ks) / len(a_ks)

        # 6. Action smoothness loss
        L_smooth = F.mse_loss(a_0[:, :-1, :], a_0[:, 1:, :])

        # 7. Combine
        total_loss = loss_task
        if self.config.use_latent_loss:
            total_loss = total_loss + self.config.lambda_latent * L_latent
        if self.config.use_action_loss:
            total_loss = total_loss + self.config.lambda_action * L_action
        total_loss = total_loss + self.config.lambda_smooth * L_smooth

        loss_dict.update({
            "loss_task": loss_task.item(),
            "loss_latent": L_latent.item(),
            "loss_action": L_action.item(),
            "loss_smooth": L_smooth.item(),
            "loss": total_loss.item(),
        })
        return total_loss, loss_dict
```

### Step 2.3 — ลงทะเบียน Policy ใน LeRobot

ไม่ต้องแก้ `factory.py` โดยตรง เพราะ v0.6.1 ใช้ `@PreTrainedConfig.register_subclass("causal_vla")` decorator — draccus จะค้นหา config อัตโนมัติ

แต่ต้องเพิ่ม import path ใน `__init__.py`:

```python
# lerobot/src/lerobot/policies/causal_vla/__init__.py
from .configuration_causal_vla import CausalVLAConfig
```

**ทดสอบ registration:**
```bash
python -c "from lerobot.configs import PreTrainedConfig; print('causal_vla' in PreTrainedConfig._registry)"
```

---

## Phase 3: Unit Test — Loss & Gradient Flow

> **เป้าหมาย:** ยืนยันความถูกต้องทางคณิตศาสตร์, tensor shapes, และ gradient flow

### Step 3.1 — สร้าง Test File

```
CausalVLA/tests/test_causal_vla.py
```

### Step 3.2 — Test Cases

```python
def test_causal_augmenter_output_shape():
    """ตรวจ CausalAugmenter ให้ output [K, B, C, H, W] ถูกต้อง"""
    augmenter = CausalAugmenter(K=3).to(device)
    obs = torch.randn(2, 3, 256, 256, device=device)
    cf = augmenter(obs)
    assert cf.shape == (3, 2, 3, 256, 256)

def test_forward_loss_components():
    """ตรวจว่า forward() คืน loss ทุกตัวและไม่มี NaN"""
    # สร้าง mock batch ที่มี shape ถูกต้อง
    loss, loss_dict = policy.forward(mock_batch)
    assert not torch.isnan(loss)
    assert "loss_task" in loss_dict
    assert "loss_latent" in loss_dict
    assert "loss_action" in loss_dict
    assert "loss_smooth" in loss_dict

def test_latent_shape_consistency():
    """ตรวจว่า z_0 และ z_k มี shape ตรงกัน"""
    z_0 = policy._extract_latent(batch)
    cf_batch = ...  # augmented batch
    z_k = policy._extract_latent(cf_batch)
    assert z_0.shape == z_k.shape

def test_gradient_flow():
    """ตรวจว่า gradient backprop ถึง vision encoder (ไม่มี detached tensor)"""
    loss, _ = policy.forward(mock_batch)
    loss.backward()
    # ตรวจว่า parameter ที่ควร train มี gradient
    for name, param in policy.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

def test_ablation_no_latent():
    """Model D: ปิด L_latent แล้ว loss ต้องไม่รวม latent term"""
    config = CausalVLAConfig(use_latent_loss=False)
    ...

def test_ablation_no_action():
    """Model E: ปิด L_action แล้ว loss ต้องไม่รวม action term"""
    config = CausalVLAConfig(use_action_loss=False)
    ...
```

### Step 3.3 — รัน Test

```bash
# Mac M2 (cpu)
pytest tests/test_causal_vla.py -v

# Colab (cuda)
pytest tests/test_causal_vla.py -v --device=cuda
```

**ตรวจสอบ:**
- [ ] ทุก test ผ่าน
- [ ] ไม่มี NaN ใน loss
- [ ] Gradient flow ถึงทุก trainable parameter
- [ ] Ablation configs ทำงานถูกต้อง

---

## Phase 4: OOD Perturbation Engine สำหรับ Evaluation

> **เป้าหมาย:** สร้าง Gym Wrapper ที่ฉีด perturbation เข้า observation ระหว่าง evaluation
> เพื่อวัด robustness ของแต่ละ model

### Step 4.1 — สร้าง OOD Wrapper (ใน `causal_aug` package)

```python
# causal_aug/ood_wrapper.py

class OODPerturbationWrapper(gym.ObservationWrapper):
    """
    Gym Observation Wrapper ที่เพิ่ม visual perturbation
    ใช้ตอน evaluation เพื่อจำลอง poor/OOD environment
    """

    OOD_LEVELS = {
        "level_0": {                    # In-Distribution (ไม่ปรับแต่ง)
            "brightness_range": (1.0, 1.0),
            "noise_sigma": 0.0,
            "cutout": False,
        },
        "level_1": {                    # Mild OOD
            "brightness_range": (0.7, 1.3),
            "noise_sigma": 0.05,
            "cutout": False,
        },
        "level_2": {                    # Extreme OOD
            "brightness_range": (0.1, 3.0),
            "noise_sigma": 0.20,
            "cutout": True,
            "cutout_ratio": 0.15,
        },
    }

    def __init__(self, env, ood_level="level_0"):
        super().__init__(env)
        self.params = self.OOD_LEVELS[ood_level]

    def observation(self, obs):
        # Apply perturbation to image observations
        ...
```

### Step 4.2 — Integrate กับ LeRobot Eval

ปรับ evaluation script ให้รับ OOD level:

**วิธี A (แนะนำ): Wrapper script แยก**
```bash
python scripts/eval_ood.py \
    --policy.path=<model_path> \
    --env.type=libero \
    --ood_level=level_2 \
    --eval.n_episodes=50
```

**วิธี B: แก้ lerobot eval config เพิ่ม flag**
```bash
lerobot-eval \
    --policy.path=<model_path> \
    --env.type=libero \
    --env.ood_level=level_2
```

> เลือกวิธี A เพื่อไม่ต้องแก้ lerobot source code มาก

---

## Phase 5: Training ทุก Experiment Variants

> **เป้าหมาย:** เทรนทุก model ด้วย controlled conditions เดียวกัน

### Training Conditions (เหมือนกันทุก model)

| Parameter | Value |
|-----------|-------|
| Base model | SmolVLA (`lerobot/smolvla_base` pretrained) |
| Dataset | `lerobot/libero_spatial_image` (+ augmented สำหรับ Model B) |
| Steps | 50,000 |
| Batch size | 8 (ปรับตาม GPU memory) |
| Seed | 42 (เหมือนกันทุก model) |
| Save frequency | 10,000 steps |
| Eval frequency | 10,000 steps |
| LIBERO suites | `libero_spatial`, `libero_goal`, `libero_object` |

### Step 5.1 — Model A: Standard SFT (Baseline)

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_a_sft \
    --job_name=model_a_sft \
    --policy.device=cuda \
    --batch_size=8 \
    --steps=50000 \
    --seed=42 \
    --save_freq=10000 \
    --env_eval_freq=10000 \
    --policy.push_to_hub=true \
    --policy.repo_id=<username>/causalvla-model-a-sft
```

### Step 5.2 — Model B: Domain Randomization

**ก่อน train — สร้าง augmented dataset:**

ใช้ `Lerobot-Dataset-Manager` สร้าง dataset ที่มี poor env simulation:
- Camera shift (perspective, affine, rotation)
- Light quality (brightness, contrast, saturation, color jitter)
- Robot noise (random start, joint offset, trajectory jitter)

Push augmented dataset ขึ้น Hub:
```bash
huggingface-cli upload <username>/libero_spatial_augmented ./augmented_dataset --private
```

**Train:**
```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --dataset.repo_id=<username>/libero_spatial_augmented \
    --output_dir=outputs/train/model_b_dr \
    --job_name=model_b_dr \
    --policy.device=cuda \
    --batch_size=8 \
    --steps=50000 \
    --seed=42 \
    --save_freq=10000 \
    --env_eval_freq=10000 \
    --policy.push_to_hub=true \
    --policy.repo_id=<username>/causalvla-model-b-dr
```

### Step 5.3 — Model C: CausalVLA (Ours)

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=causal_vla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_c_causal \
    --job_name=model_c_causal \
    --policy.device=cuda \
    --policy.n_counterfactual=3 \
    --policy.lambda_latent=0.1 \
    --policy.lambda_action=0.1 \
    --policy.lambda_smooth=0.01 \
    --batch_size=8 \
    --steps=50000 \
    --seed=42 \
    --save_freq=10000 \
    --env_eval_freq=10000 \
    --policy.push_to_hub=true \
    --policy.repo_id=<username>/causalvla-model-c-ours
```

### Step 5.4 — Model D: Ablation w/o L_latent

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=causal_vla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_d_no_latent \
    --job_name=model_d_no_latent \
    --policy.device=cuda \
    --policy.use_latent_loss=false \
    --batch_size=8 \
    --steps=50000 \
    --seed=42 \
    --save_freq=10000 \
    --policy.push_to_hub=true \
    --policy.repo_id=<username>/causalvla-model-d-ablation
```

### Step 5.5 — Model E: Ablation w/o L_action

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=causal_vla \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --output_dir=outputs/train/model_e_no_action \
    --job_name=model_e_no_action \
    --policy.device=cuda \
    --policy.use_action_loss=false \
    --batch_size=8 \
    --steps=50000 \
    --seed=42 \
    --save_freq=10000 \
    --policy.push_to_hub=true \
    --policy.repo_id=<username>/causalvla-model-e-ablation
```

---

## Phase 6: Evaluation & Results Collection

> **เป้าหมาย:** ประเมินทุก model ในทุก OOD level และสร้างตารางเปรียบเทียบ

### Step 6.1 — Evaluation Matrix

รัน eval สำหรับทุก model × ทุก OOD level × ทุก LIBERO suite:

```bash
# สำหรับแต่ละ MODEL in [a, b, c, d, e]
# สำหรับแต่ละ OOD_LEVEL in [level_0, level_1, level_2]
python scripts/eval_ood.py \
    --policy.path=<username>/causalvla-model-{MODEL} \
    --env.type=libero \
    --env.task_suite=libero_spatial \
    --ood_level={OOD_LEVEL} \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda \
    --output_dir=outputs/eval/model_{MODEL}_{OOD_LEVEL}
```

**Total eval runs:** 5 models × 3 OOD levels = **15 runs** (ต่อ LIBERO suite)

### Step 6.2 — Metrics Collection

| Metric | วิธีเก็บ |
|--------|---------|
| **Success Rate (%)** | จาก eval result (50 episodes ต่อ run) |
| **Action Jitter** | คำนวณ `mean(‖a_t - a_{t+1}‖)` จาก rollout |
| **Latent Drift (Δz)** | สกัด `z_0, z_k` แล้วคำนวณ `mean(‖z_0 - z_k‖)` |
| **t-SNE Plot** | Plot latent features ของ clean vs perturbed obs |
| **Video** | บันทึก rollout video เพื่อเปรียบเทียบ visual |

### Step 6.3 — Results Table (Template)

#### Success Rate (%) — LIBERO Spatial

| Model | Level 0 (Clean) | Level 1 (Mild) | Level 2 (Extreme) | Δ (L0→L2) |
|-------|----------------|----------------|-------------------|------------|
| A: Standard SFT | — | — | — | — |
| B: Domain Rand. | — | — | — | — |
| C: CausalVLA (Ours) | — | — | — | — |
| D: Ablation w/o L_lat | — | — | — | — |
| E: Ablation w/o L_act | — | — | — | — |

> **Δ (L0→L2):** ผลต่าง success rate ระหว่าง clean กับ extreme OOD (ยิ่งน้อยยิ่ง robust)

#### Latent Drift & Action Jitter

| Model | Latent Drift (Δz) | Action Jitter |
|-------|-------------------|---------------|
| A: Standard SFT | — | — |
| B: Domain Rand. | — | — |
| C: CausalVLA (Ours) | — | — |

---

## โครงสร้างไฟล์โปรเจกต์

```
CausalVLA/
├── PIPELINE.md                              # ← ไฟล์นี้
├── paper-draf.md                            # Research paper draft
├── mynote.txt                               # Notes
│
├── causal_aug/                              # ← Local Python Package (pip install -e)
│   ├── pyproject.toml
│   ├── causal_aug/
│   │   ├── __init__.py                      # exports: CausalAugmenter, OODPerturbationWrapper
│   │   ├── gpu_augmenter.py                 # CausalAugmenter class (online GPU augmentation)
│   │   ├── augmentations.py                 # Individual augmentation functions (PyTorch ops)
│   │   └── ood_wrapper.py                   # OODPerturbationWrapper (Gym wrapper)
│   └── tests/
│       └── test_augmenter.py
│
├── lerobot/                                 # ← LeRobot v0.6.1 (cloned, editable install)
│   └── src/lerobot/
│       ├── policies/
│       │   ├── smolvla/                     # Base SmolVLA (ไม่แก้ไข)
│       │   └── causal_vla/                  # ← NEW: CausalVLA Policy
│       │       ├── __init__.py
│       │       ├── configuration_causal_vla.py
│       │       └── modeling_causal_vla.py
│       ├── envs/
│       │   └── libero.py                    # LIBERO env (ไม่แก้ไข)
│       └── scripts/
│           ├── lerobot_train.py             # Training (ไม่แก้ไข)
│           └── lerobot_eval.py              # Evaluation (ไม่แก้ไข)
│
├── scripts/                                 # ← Custom scripts
│   ├── eval_ood.py                          # Evaluation with OOD wrapper
│   ├── collect_results.py                   # Aggregate eval results → tables
│   └── plot_tsne.py                         # t-SNE visualization
│
├── tests/                                   # ← Tests
│   └── test_causal_vla.py                   # Unit tests for CausalVLA
│
└── outputs/                                 # ← Training & eval outputs
    ├── train/
    │   ├── model_a_sft/
    │   ├── model_b_dr/
    │   ├── model_c_causal/
    │   ├── model_d_no_latent/
    │   └── model_e_no_action/
    └── eval/
        ├── model_a_level_0/
        ├── model_a_level_1/
        └── ...
```

---

## Workflow Diagram: Local ↔ Hub ↔ Colab

```
Mac M2 (Local)                    HuggingFace Hub (Private)         Google Colab (GPU)
──────────────────               ─────────────────────────          ──────────────────

Phase 1: Smoke test (cpu)
Phase 2: Dev causal_aug pkg
Phase 2: Dev causal_vla policy
Phase 3: Unit tests (cpu)
                                                                    Phase 1: Smoke test (cuda)
                                                                    Phase 1: LIBERO eval test

Augment dataset ──────push───→  Augmented Dataset  ←───pull──────  Phase 5: Model B training
                                Clean Dataset      ←───pull──────  Phase 5: Model A training
                                                                    Phase 5: Model C/D/E training

                                Model Checkpoints:
                                ├─ model-a-sft     ←───push──────  ← Train output
                                ├─ model-b-dr      ←───push──────  ← Train output
                                ├─ model-c-ours    ←───push──────  ← Train output
                                ├─ model-d-abl     ←───push──────  ← Train output
                                └─ model-e-abl     ←───push──────  ← Train output

Download models  ←────pull────  All checkpoints                     Phase 6: LIBERO eval
                                                                    (Level 0, 1, 2)
Phase 5+ (future):
Real robot eval (SO-101)        Eval Results       ←───push──────  ← Eval output
```

---

## Experiment Checklist

| Phase | Deliverables | เครื่องมือ/ไฟล์หลัก | สถานะ |
|-------|-------------|---------------------|-------|
| **Phase 1** | Verified environment & baseline run | conda, lerobot-train, lerobot-eval | ☐ |
| **Phase 2** | `causal_aug` package + `causal_vla` policy | `causal_aug/`, `policies/causal_vla/` | ☐ |
| **Phase 3** | Passed unit tests | pytest, `tests/test_causal_vla.py` | ☐ |
| **Phase 4** | OOD Gym Wrapper | `causal_aug/ood_wrapper.py`, `scripts/eval_ood.py` | ☐ |
| **Phase 5** | 5 trained model checkpoints | WandB, Colab GPU, HuggingFace Hub | ☐ |
| **Phase 6** | Success rate matrix, t-SNE plots, videos | `scripts/collect_results.py`, matplotlib | ☐ |

---

## หมายเหตุเกี่ยวกับ LeRobot v0.6.1

- **Policy path:** `src/lerobot/policies/` (ไม่ใช่ `lerobot/common/policies/`)
- **Env path:** `src/lerobot/envs/` (ไม่ใช่ `lerobot/common/envs/`)
- **Policy registration:** ใช้ `@PreTrainedConfig.register_subclass("causal_vla")` decorator
- **Training CLI:** `python -m lerobot.scripts.lerobot_train` หรือ `lerobot-train`
- **Eval CLI:** `lerobot-eval`
- **LIBERO suites:** `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
- **SmolVLA architecture:** VLM backbone (`SmolVLM2-500M-Video-Instruct`) + Action Expert (flow matching)
- **SmolVLA forward:** คืน `(loss, loss_dict)` — CausalVLA ต้อง extend pattern นี้
