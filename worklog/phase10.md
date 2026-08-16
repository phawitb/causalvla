# Phase 10 — COVER-VLA

> Started: 2026-08-16  
> Status: IMPLEMENTATION IN PROGRESS

## Goal

ต่อยอดจาก Model F ด้วย coverage-constrained group-robust supervised training
โดยใช้หนึ่ง forward/sample และรักษา clean/augmentation coverage อย่างชัดเจน
แทน policy-disagreement หรือ static risk ranking ที่ไม่ผ่าน Phase 8–9 gates.

## Approved Sequence

1. Model L1 — COVER-Base, pilot 5K steps
2. Model L2 — COVER-Safe, pilot 5K steps
3. Eval seed 1000, 5 episodes/task, Clean/Mild/Extreme
4. เลือกหนึ่ง variant ตาม preregistered pilot rule
5. Full train 25K เฉพาะผู้ชนะ
6. Eval seed 1000, 10 episodes/task และขยาย seeds เมื่อผ่าน full-run gate

สเปกหลัก: `docs/superpowers/specs/2026-08-16-cover-vla-design.md`

## Status Checklist

- [x] Analyze Phase 6–9 evidence
- [x] Select Model F as the parent baseline
- [x] Approve two-pilot, one-full-run experiment budget
- [x] Write COVER-VLA design
- [x] Write implementation plan
- [ ] Implement with tests (controller/policies complete; workflow verification pending)
- [ ] Mac MPS smoke
- [ ] CUDA smoke
- [ ] L1/L2 5K pilots
- [ ] Pilot evaluation and variant selection
- [ ] Selected 25K full training
- [ ] Preregistered final evaluation

## GPU Pilot Workflow

ติดตั้ง policy patches ก่อนทุก smoke/pilot หลัง `git pull`:

```bash
conda activate causalvla
python -m pip install -e causal_aug
python scripts/install_policy_patches.py cover_base cover_safe
```

CUDA smoke ใช้คำสั่ง training เดียวกับ pilot ด้านล่าง แต่เปลี่ยนเป็น `--steps=2`,
`--batch_size=2`, `--save_freq=2`, `--log_freq=1`, และ output ใต้
`outputs/smoke/`. ต้องเห็น `cover/forward_count:1.000` และไม่มี error/NaN.

### L1 — COVER-Base 5K pilot

```bash
PYTHONPATH="$PWD/causal_aug${PYTHONPATH:+:$PYTHONPATH}" lerobot-train \
  --policy.type=cover_base --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-cover-base-pilot \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/phase10/cover_base_pilot \
  --job_name=cover_base_pilot --batch_size=16 --steps=5000 --seed=1000 \
  --save_freq=5000 --save_checkpoint_to_hub=true --log_freq=100 \
  --num_workers=4 --persistent_workers=true --env_eval_freq=0 \
  2>&1 | tee logs/cover_base_pilot.log
```

### L2 — COVER-Safe 5K pilot

```bash
PYTHONPATH="$PWD/causal_aug${PYTHONPATH:+:$PYTHONPATH}" lerobot-train \
  --policy.type=cover_safe --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-cover-safe-pilot \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/phase10/cover_safe_pilot \
  --job_name=cover_safe_pilot --batch_size=16 --steps=5000 --seed=1000 \
  --save_freq=5000 --save_checkpoint_to_hub=true --log_freq=100 \
  --num_workers=4 --persistent_workers=true --env_eval_freq=0 \
  2>&1 | tee logs/cover_safe_pilot.log
```

Pilot evaluation ใช้ seed 1000 และ **5 episodes/task** ทั้ง Clean/Mild/Extreme.
เลือก variant ตาม gate ใน design และห้ามนำ pilot episodes ไปรวมกับ final result.

## Selected Full Training

แทน `<selected>` ด้วย `cover_base` หรือ `cover_safe` หลังประกาศผล pilot เท่านั้น:

```bash
PYTHONPATH="$PWD/causal_aug${PYTHONPATH:+:$PYTHONPATH}" lerobot-train \
  --policy.type=<selected> --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-<selected-with-hyphens> \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/final/<selected> \
  --job_name=<selected> --batch_size=16 --steps=25000 --seed=1000 \
  --save_freq=5000 --save_checkpoint_to_hub=true --log_freq=100 \
  --num_workers=4 --persistent_workers=true --env_eval_freq=0 \
  2>&1 | tee logs/<selected>.log
```

หลัง upload ให้บันทึก exact 40-character Hub SHA ที่
`outputs/phase10/<selected>_revision.txt`. Final evaluation ใช้
**10 episodes/task**:

```bash
./scripts/run_eval_cover.sh <selected> level_0 1000 10
./scripts/run_eval_cover.sh <selected> level_1 1000 10
./scripts/run_eval_cover.sh <selected> level_2 1000 10
```
