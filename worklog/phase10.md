# Phase 10 — COVER-VLA

> Started: 2026-08-16  
> Status: DESIGN APPROVED; IMPLEMENTATION PLAN PENDING

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
- [ ] Write implementation plan
- [ ] Implement with tests
- [ ] Mac MPS smoke
- [ ] CUDA smoke
- [ ] L1/L2 5K pilots
- [ ] Pilot evaluation and variant selection
- [ ] Selected 25K full training
- [ ] Preregistered final evaluation
