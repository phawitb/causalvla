# Phase 3: Unit Testing

> **วันที่:** 2 สิงหาคม 2026
> **สถานะ:** COMPLETED
> **เครื่อง:** Mac M2 (local)

---

## สิ่งที่ทำ

เขียน unit tests ครอบคลุม CausalVLA ทุก component — CausalAugmenter, policy registration, forward pass loss components, gradient flow, และ ablation configs

## ไฟล์ที่สร้าง

```
lerobot/tests/policies/causal_vla/
├── __init__.py
└── test_causal_vla.py
```

## Test Classes & Results

### Test 1: CausalAugmenter (5/5 PASSED)

| Test | คำอธิบาย |
|------|---------|
| `test_output_shape` | K=3, input [4,3,256,256] → output [3,4,3,256,256] |
| `test_output_shape_k1` | K=1, input [2,3,128,128] → output [1,2,3,128,128] |
| `test_intensity_zero_is_near_identity` | intensity=0 → output ≈ input (atol=1e-5) |
| `test_different_k_values` | K=1,2,5 ทุกค่าให้ output.shape[0]==K |
| `test_each_counterfactual_is_different` | K=3 แต่ละ counterfactual ต่างกัน |

### Test 2: Policy Registration (4/4 PASSED)

| Test | คำอธิบาย |
|------|---------|
| `test_make_policy_config` | `make_policy_config("causal_vla")` → CausalVLAConfig |
| `test_get_policy_class` | `get_policy_class("causal_vla")` → CausalVLAPolicy |
| `test_config_inherits_smolvla` | chunk_size=50, n_obs_steps=1, freeze_vision_encoder=True |
| `test_config_has_causal_fields` | n_counterfactual, lambda_latent, lambda_action, lambda_smooth, use_latent_loss, use_action_loss, aug_intensity |

### Test 3: Forward Pass — Loss Components (5/5 PASSED)

| Test | คำอธิบาย |
|------|---------|
| `test_forward_returns_loss_and_dict` | forward() returns (scalar tensor, dict) |
| `test_loss_dict_has_all_components` | loss_task, loss_latent, loss_action, loss_smooth, loss ครบ |
| `test_no_nan_in_loss` | ไม่มี NaN ในทุก loss component |
| `test_total_loss_greater_than_task_loss` | L_total ≥ L_task (invariance losses เป็นบวก) |
| `test_reduction_none` | reduction="none" → per-sample loss shape [B] |

### Test 4: Gradient Flow (1/1 PASSED)

| Test | คำอธิบาย |
|------|---------|
| `test_gradients_reach_trainable_params` | loss.backward() → 155 trainable params ได้รับ gradient, 0 params ไม่ได้รับ |

### Test 5: Ablation Configs (3/3 PASSED)

| Test | คำอธิบาย |
|------|---------|
| `test_ablation_no_latent_loss` | use_latent_loss=False → loss_latent ไม่อยู่ใน loss_dict |
| `test_ablation_no_action_loss` | use_action_loss=False → loss_action ไม่อยู่ใน loss_dict |
| `test_ablation_both_off_still_has_task_and_smooth` | ทั้งคู่ off → ยังมี loss_task + loss_smooth |

---

## ผลรวม

```
=================== 18 passed, 1 warning in 74.98s (0:01:14) ===================
```

| Metric | Value |
|--------|-------|
| Total tests | 18 |
| Passed | 18 |
| Failed | 0 |
| Time | ~75 sec |
| Device | MPS (Mac M2) |

## Bug ที่แก้ไข

### Mock batch image shape

**ปัญหา:** `_make_batch()` สร้าง image tensor เป็น `[B, T, H, W, C]` = `[2, 1, 480, 640, 3]` แต่ SmolVLA คาดหวัง `[B, T, C, H, W]`

**Error:**
```
RuntimeError: expected input[2, 480, 512, 512] to have 3 channels,
but got 480 channels instead
```

**แก้ไข:** เปลี่ยนเป็น `torch.rand(BATCH_SIZE, 1, 3, 480, 640)` — format `[B, T, C, H, W]`

### causal_aug import path

**ปัญหา:** รัน pytest จาก `/Users/phawit/Projects/CausalVLA/` ทำให้ outer `causal_aug/` directory ถูก resolve เป็น namespace package แทน installed package

**แก้ไข:** รัน pytest จาก `lerobot/` directory แทน — `sys.path` ไม่รวม parent directory ที่มี outer `causal_aug/`

---

## Command ที่ใช้รัน

```bash
cd /Users/phawit/Projects/CausalVLA/lerobot
conda run -n lerobot2 python -m pytest tests/policies/causal_vla/test_causal_vla.py -svv --tb=short
```

---

## Insight

- SmolVLA image format คือ `[B, T, C, H, W]` — ต่างจาก LeRobot dataset ที่เก็บเป็น `[B, T, H, W, C]` → ต้องระวังเรื่อง channel dimension เมื่อสร้าง mock data
- `scope="class"` fixture ช่วยให้ policy ถูกสร้างครั้งเดียวต่อ test class → ประหยัดเวลา model loading
- Gradient flow test ยืนยันว่า 155 trainable parameters ทั้งหมดได้รับ gradient — ไม่มี dead parameter
- Ablation tests ยืนยันว่า flag `use_latent_loss` / `use_action_loss` ทำงานถูกต้อง — ปิดได้จริงโดยไม่กระทบ loss อื่น
