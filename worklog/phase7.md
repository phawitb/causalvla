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

## Pilot Evaluation — 10 Episodes per Task

วันที่ 2026-08-14 ประเมิน checkpoint เต็มจาก
`phawitbinabik/causalvla-model-v2` revision
`6fc4104176b08ba7f9592583a8431c2e30b035ab` บน Mac M2/MPS โดยใช้
LIBERO Spatial 10 tasks, 10 episodes/task, seed 1000 รวม 100 episodes ต่อระดับ

| OOD level | CausalVLA-v2 | Model B (Phase 6) | Difference |
|---|---:|---:|---:|
| Clean (`level_0`) | 63.0% (63/100) | 61.8% | +1.2 pp |
| Mild (`level_1`) | 60.0% (60/100) | 53.2% | +6.8 pp |
| Extreme (`level_2`) | 45.0% (45/100) | 30.2% | +14.8 pp |

Per-task success rates (task 0–9):

| OOD level | Task success rates |
|---|---|
| Clean | 90, 50, 70, 70, 80, 20, 90, 90, 40, 30% |
| Mild | 90, 50, 80, 70, 60, 30, 60, 80, 40, 40% |
| Extreme | 50, 50, 60, 60, 20, 0, 60, 40, 60, 50% |

Evaluation time: clean 848.1 s, mild 1209.4 s, extreme 1567.4 s.

ผล pilot ชนะ Model B ทั้งสามระดับ แต่ clean ต่างเพียง 1.2 percentage points
และการทดลองนี้มีเพียง 100 episodes/level จึงยังไม่ควรถือเป็นข้อสรุปสุดท้าย
ขั้นต่อไปคือรัน protocol เต็ม 50 episodes/task ด้วย seed เดียวกับ Phase 6
เพื่อเปรียบเทียบแบบ 500 episodes/level

ระหว่างเตรียม eval ได้แก้ compatibility ของ evaluator ให้
`OODProcessorStep` ใช้ `ObservationProcessorStep` API ของ LeRobot ปัจจุบัน
และเพิ่ม model id `v2` พร้อม pin revision ใน `scripts/run_eval_mps.sh`

## Model F — Online Domain Randomization Baseline

เพิ่ม baseline สำหรับแยกผลของ online augmentation ออกจากผลของ
counterfactual pairing ใน V2:

```text
Original clean dataset
        ↓
สุ่มต่อ sample: clean 50% / online-augmented 50%
        ↓
SmolVLA forward 1 ครั้ง
        ↓
Supervised task loss
```

Model F ใช้ `CausalAugmenter` และ augmentation ranges ชุดเดียวกับ V2 แต่ไม่มี
clean/counterfactual pair, shared flow target หรือ consistency loss และใช้เพียง
หนึ่ง policy forward ต่อ sample. ตอน inference ไม่มี augmentationและเหมือน
SmolVLA ปกติ

ไฟล์ implementation:

- `lerobot_patches/online_dr/configuration_online_dr.py`
- `lerobot_patches/online_dr/modeling_online_dr.py`
- `lerobot_patches/online_dr/processor_online_dr.py`

Default configuration:

```text
type             = online_dr
aug_probability  = 0.5
aug_intensity    = 1.0
```

### MPS Smoke Test

รันด้วย original `lerobot/libero_spatial_image`, batch size 2, seed 1000 และ
2 optimizer steps บน Mac M2/MPS สำเร็จครบ:

```text
step 1: loss=2.072, grad_norm=28.632, augmented_fraction=0.500
step 2: loss=13.101, grad_norm=91.488, augmented_fraction=0.500
End of training
```

ตรวจผ่าน policy registration, dataset loading, online augmentation, single
forward loss, backward, optimizer/scheduler update และ checkpoint serialization
โดย checkpoint มี `type=online_dr`, `aug_probability=0.5` และ
`aug_intensity=1.0`

### GPU Full Training Command

หลัง `git pull origin main` ให้ติดตั้ง Model F เข้า active conda environment และ
ตรวจ registration:

```bash
python scripts/install_policy_patches.py online_dr

python - <<'PY'
from lerobot.policies.factory import get_policy_class, make_policy_config

cfg = make_policy_config("online_dr")
assert cfg.type == "online_dr"
assert cfg.aug_probability == 0.5
assert get_policy_class("online_dr").__name__ == "OnlineDRPolicy"
print("Model F Online DR install: PASS")
PY
```

```bash
lerobot-train \
  --policy.type=online_dr \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-model-f-online-dr \
  --policy.private=true \
  --policy.aug_probability=0.5 \
  --policy.aug_intensity=1.0 \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/final/model_f_online_dr \
  --job_name=model_f_online_dr \
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

หลัง full training ให้ eval ด้วย 10 episodes/task เป็น protocol หลักทั้ง clean,
mild และ extreme OOD. ถ้า Model F ใกล้ V2 แสดงว่าผลหลักมาจาก online
augmentation; ถ้า V2 ชนะ F จะสนับสนุนประโยชน์ของ paired supervision

### Model F Full Training and Evaluation Results

Full training จบครบ 25,000 steps และ upload ไปที่
`phawitbinabik/causalvla-model-f-online-dr` revision
`997d94a9325bc359422cd3cf54bd74b0a4c9be98`. Final checkpoint ผ่าน config
validation: `type=online_dr`, `aug_probability=0.5`, `aug_intensity=1.0`

ประเมินบน LIBERO Spatial ด้วย seed 1000, 10 tasks และ 10 episodes/task:

| Model | Clean | Mild OOD | Extreme OOD |
|---|---:|---:|---:|
| B — Offline DR | 61.8% | 53.2% | 30.2% |
| V2 — Paired supervision | 63.0% | 60.0% | 45.0% |
| **F — Online DR** | **64.0%** | **62.0%** | **49.0%** |

Model F ต่างจาก V2 `+1 pp` clean, `+2 pp` mild และ `+4 pp` extreme โดยใช้
policy forward เพียงครั้งเดียวต่อ sample เทียบกับ V2 ที่ใช้สองครั้ง ผลนี้ไม่
สนับสนุนสมมติฐานว่า paired supervision ของ V2 ให้ประโยชน์เหนือ online domain
randomization ใน protocol ปัจจุบัน และชี้ว่า improvement จาก B ไป V2 น่าจะ
อธิบายได้มากจาก online augmentation และ augmentation diversity

ข้อสรุปเชิง paper ต้องรายงานผลนี้ตรงไปตรงมา: Model F เป็น current best model
และเป็น stronger, simpler baseline. ขั้นต่อไปไม่ควร claim ว่า V2 pairing ชนะ
จนกว่าจะมี controlled experiment อื่นรองรับ เช่น fixed visual-sample budget,
unseen-intervention holdout หรือ multi-seed evaluation

### Multi-seed Validation — Seed 2000

รันเพิ่มด้วย protocol เดิม 10 tasks × 10 episodes/task ทั้ง V2 และ Model F:

| Seed | Model | Clean | Mild OOD | Extreme OOD |
|---:|---|---:|---:|---:|
| 1000 | V2 | 63.0% | 60.0% | 45.0% |
| 1000 | F | 64.0% | 62.0% | 49.0% |
| 2000 | V2 | 57.0% | 57.0% | 45.0% |
| 2000 | F | 68.0% | 55.0% | 48.0% |

ผลรวมสอง eval seeds (mean ± sample standard deviation):

| Model | Clean | Mild OOD | Extreme OOD |
|---|---:|---:|---:|
| V2 | 60.0 ± 4.2% | 58.5 ± 2.1% | 45.0 ± 0.0% |
| F — Online DR | **66.0 ± 2.8%** | **58.5 ± 4.9%** | **48.5 ± 0.7%** |

Model F ชนะค่าเฉลี่ย clean `+6.0 pp`, เสมอ mild และชนะ extreme `+3.5 pp`.
ผล seed 2000 ยืนยันว่า F ไม่ได้ชนะทุก level ทุก seed (V2 ชนะ mild 2 pp)
แต่ภาพรวมยังไม่พบหลักฐานว่า paired supervision ให้ประโยชน์เหนือ single-forward
online DR. หากต้องการสรุปเชิงสถิติควรรัน seed 3000 เพิ่มตามแผน

### Final Multi-seed Results — Seeds 1000, 2000, 3000

| Seed | Model | Clean | Mild OOD | Extreme OOD |
|---:|---|---:|---:|---:|
| 1000 | V2 | 63.0% | 60.0% | 45.0% |
| 1000 | F | 64.0% | 62.0% | 49.0% |
| 2000 | V2 | 57.0% | 57.0% | 45.0% |
| 2000 | F | 68.0% | 55.0% | 48.0% |
| 3000 | V2 | 60.0% | 56.0% | 43.0% |
| 3000 | F | 64.0% | 59.0% | 51.0% |

Aggregate mean ± sample standard deviation:

| Model | Clean | Mild OOD | Extreme OOD |
|---|---:|---:|---:|
| V2 | 60.0 ± 3.0% | 57.7 ± 2.1% | 44.3 ± 1.2% |
| **F — Online DR** | **65.3 ± 2.3%** | **58.7 ± 3.5%** | **49.3 ± 1.5%** |

จาก 300 episodes/level ต่อโมเดล Model F ต่างจาก V2 โดยเฉลี่ย `+5.3 pp`
clean, `+1.0 pp` mild และ `+5.0 pp` extreme. F ชนะ clean และ extreme ครบ
ทั้งสาม eval seeds และชนะ mild สองในสาม seeds. อย่างไรก็ตามจำนวน training seed
ยังมีเพียงหนึ่ง seed และการเปรียบเทียบ success proportions แบบไม่ใช้ paired
episode outcomes ยังไม่เพียงพอสำหรับ claim statistical significance

ข้อสรุป Phase 7 ปัจจุบัน: online augmentation เป็นคำอธิบายที่แข็งแรงกว่า
counterfactual pairing สำหรับ performance gain ที่สังเกตได้ และ Model F เป็น
โมเดลหลักที่เรียบง่ายกว่า เร็วกว่าในการ train ต่อ step และให้ผลเฉลี่ยสูงกว่า V2
