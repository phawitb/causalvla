# Phase 7: CausalVLA-v2 Development & MPS Training Smoke Test

> **วันที่:** 13 สิงหาคม 2026
>
> **สถานะ:** IMPLEMENTATION + LOCAL SMOKE TEST COMPLETED
>
> **เครื่องทดสอบ:** Mac M2, Apple MPS

## เป้าหมาย

ปรับ CausalVLA จากผล Phase 6 เพื่อแก้ปัญหาที่ Model C แพ้ Domain Randomization baseline (Model B) โดยเน้นให้ counterfactual image ได้รับ ground-truth action supervision โดยตรง แทนการพึ่ง invariance loss ที่แรงเกินไป

ผล Phase 6 ที่ใช้ตัดสินใจ:

| Model | Clean | Mild OOD | Extreme OOD |
|---|---:|---:|---:|
| A — Standard SFT | 67.4% | 14.2% | 0.4% |
| B — Domain Randomization | 61.8% | **53.2%** | **30.2%** |
| C — CausalVLA v1 | 44.4% | 3.2% | 1.0% |
| E — CausalVLA w/o Action Loss | 67.0% | 29.0% | 4.0% |

Model E ฟื้นทั้ง clean และ mild OOD เมื่อปิด action-consistency loss จึงเป็นหลักฐานว่า `L_action` ของ v1 เป็น regularizer ที่แรงและกด policy ผิดทิศ ขณะที่ Model B แสดงว่า supervised training บน augmented observations ให้ robustness สูงที่สุด

## CausalVLA-v2 Objective

เปลี่ยน objective หลักเป็น paired supervised flow matching:

```text
L_total = 0.5 × L_task_clean + 0.5 × L_task_augmented
```

แต่ละ training sample มีสอง views:

1. Clean observation
2. Counterfactual/augmented observation
3. ทั้งสอง views ใช้ ground-truth action, flow noise และ flow time ชุดเดียวกัน

Default configuration:

```text
n_counterfactual       = 1
clean_task_weight      = 0.5
augmented_task_weight  = 0.5
use_latent_loss        = false
use_action_loss        = false
lambda_latent          = 0.0
lambda_action          = 0.0
lambda_smooth          = 0.0
```

Optional losses ยังเก็บไว้สำหรับ ablation ภายหลัง:

- Action consistency ใช้ augmented prediction เทียบกับ `clean_prediction.detach()`
- Latent consistency ใช้ cosine loss บน action-expert latent และ stop-gradient clean teacher
- Smoothness loss ไม่ทำงานเมื่อ `lambda_smooth=0`

## Implementation Changes

ไฟล์ที่แก้:

- `lerobot_patches/causal_vla/modeling_causal_vla.py`
- `lerobot_patches/causal_vla/configuration_causal_vla.py`
- `lerobot_patches/forward_with_latent.py`
- `lerobot_patches/lerobot_causalvla.patch`
- `causal_aug/causal_aug/gpu_augmenter.py`

การเปลี่ยนสำคัญ:

1. เพิ่ม supervised flow-matching loss ให้ clean และ augmented branch
2. Sample `noise` และ `time` เพียงครั้งเดียวก่อนเรียกทั้งสอง branch
3. เปลี่ยน `forward_with_latent()` ให้คืน `suffix_out` จาก trainable action expert
4. ตั้ง `K=1` เพื่อลด compute จาก v1 ที่ใช้ `K=3`
5. ปิด latent, action-consistency และ velocity-smoothness losses โดย default
6. เพิ่ม config validation ป้องกัน `K<1`, task weights ติดลบ หรือ weights รวมเป็นศูนย์
7. ทำ augmentation ใน pixel range `[0,1]`, clamp แล้ว normalize กลับเป็น `[-1,1]`
8. ใช้ scene-level nuisance parameters ร่วมกันระหว่าง main และ wrist cameras
9. Online augmenter ครอบคลุม brightness, contrast, saturation, hue, noise, blur, shadow, rotation, affine และ perspective
10. ปรับ augmentation ranges ให้ใกล้ training distribution ของ Model B

## Mac M2 / MPS Training Smoke Test

รัน training จริงด้วย LeRobot dataset และ optimizer pipeline ครบสาย โดยใช้ LeRobot runtime ชั่วคราวเพื่อไม่แก้ installed environment หรือ nested source tree

Environment:

```text
Python environment: causalvla
PyTorch:            2.11.0
LeRobot:            0.6.1
Device:             mps
Dataset:            lerobot/libero_spatial_image
Frames:             52,970
Episodes:           432
Batch size:         1
Steps:              2
Seed:               1000
Push to Hub:        false
```

Model size:

```text
Total parameters:     450,046,176 (450M)
Learnable parameters:  99,880,992 (100M)
```

### Training Results

```text
step 1
  loss_task_clean:       2.263
  loss_task_augmented:   2.625
  loss:                  2.444
  gradient norm:        30.588
  update time:           3.039 s

step 2
  loss_task_clean:      12.041
  loss_task_augmented:   2.803
  loss:                  7.422
  gradient norm:        56.102
  update time:           0.615 s
```

ตรวจสมการ loss ผ่านทั้งสอง steps:

```text
step 1: 0.5 × 2.263 + 0.5 × 2.625 = 2.444
step 2: 0.5 × 12.041 + 0.5 × 2.803 = 7.422
```

### Verification Results

| Check | Result |
|---|---|
| Policy/config registration | PASS |
| MPS model construction | PASS |
| Dataset loading | PASS |
| Paired clean + augmented forward | PASS |
| Shared flow noise/time | PASS |
| Backward and gradient computation | PASS |
| Optimizer and scheduler update | PASS |
| No NaN/Inf | PASS |
| No MPS OOM | PASS |
| Checkpoint save | PASS |
| Optimizer/scheduler/RNG state save | PASS |
| Serialized v2 config verification | PASS |
| End of training reached | PASS |

Smoke checkpoint ถูกสร้างครบที่ step 2 โดยมี policy weights, pre/post processors, optimizer state, scheduler state และ RNG state

## GPU Server Training Configuration

ค่าที่ควรใช้สำหรับ full training:

```bash
lerobot-train \
  --policy.type=causal_vla \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-model-v2 \
  --policy.private=true \
  --policy.n_counterfactual=1 \
  --policy.aug_intensity=1.0 \
  --policy.clean_task_weight=0.5 \
  --policy.augmented_task_weight=0.5 \
  --policy.use_latent_loss=false \
  --policy.use_action_loss=false \
  --policy.lambda_latent=0.0 \
  --policy.lambda_action=0.0 \
  --policy.lambda_smooth=0.0 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=15000 \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/final/model_v2 \
  --job_name=causalvla_v2 \
  --batch_size=16 \
  --steps=25000 \
  --seed=1000 \
  --save_freq=5000 \
  --save_checkpoint_to_hub=true \
  --log_freq=100 \
  --num_workers=4 \
  --persistent_workers=true \
  --env_eval_freq=0
```

ก่อน full training ต้อง sync commit ล่าสุดและยืนยันว่า installed patch มี:

```python
assert "return losses, suffix_out, v_t" in inspect.getsource(
    VLAFlowMatching.forward_with_latent
)
```

หลังเริ่ม training ให้ตรวจ log ว่ามี metrics ต่อไปนี้:

```text
loss_task_clean
loss_task_augmented
loss_task
```

และต้องไม่มี `loss_action`, `loss_latent` หรือ `loss_smooth` ใน default v2 run

## Next Steps

1. Commit และ push CausalVLA-v2 implementation
2. Sync repo และติดตั้ง patch บน GPU server
3. รัน GPU preflight 2–10 steps ก่อน full training
4. Train 25,000 steps ด้วย sample budget เดิม 400,000 samples
5. Evaluate clean/mild/extreme OOD ด้วย seed และ episode budgetเดียวกับ Model B
6. เป้าหมาย: clean >61.8%, mild >53.2%, extreme >30.2%
