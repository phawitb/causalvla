# Phase 9 — PACER-VLA

> Started: 2026-08-16  
> Status: DESIGN REVIEW  
> Primary baseline: Model F — Online DR

## Goal

พัฒนาเทคนิคใหม่สำหรับ robust Vision-Language-Action learning ที่มีโอกาสเป็น
แกนหลักของ paper Q1 โดยใช้หลักฐานจาก Phase 6--8 เป็นตัวกำหนด design ไม่สร้าง
fixed augmentation mixture เพิ่มโดยไม่มี hypothesis ใหม่

Phase 9 ใช้ลำดับดังนี้:

1. **Model J — PACER-Lite:** two-forward policy-adaptive counterfactual training
2. **Model K — PACER-Full:** consistency และ worst-group extension เฉพาะเมื่อ J
   ผ่าน preregistered gate

Design ฉบับเต็มอยู่ที่
`docs/superpowers/specs/2026-08-16-pacer-vla-design.md`.

## Final Phase 8 Evidence

Residual RAPID evaluation เสร็จครบ 3 eval seeds หลัง Phase 8 gate เดิม:

| Seed | Clean | Mild | Extreme |
|---:|---:|---:|---:|
| 1000 | 74% | 50% | 56% |
| 2000 | 58% | 51% | 45% |
| 3000 | 55% | 52% | 32% |
| Mean +/- SD | 62.3 +/- 10.2% | 51.0 +/- 1.0% | 44.3 +/- 12.0% |

เทียบกับ Model F:

| Model | Clean | Mild | Extreme | Three-mode mean |
|---|---:|---:|---:|---:|
| Model F | **65.3 +/- 2.3%** | **58.7 +/- 3.5%** | **49.3 +/- 1.5%** | **57.8%** |
| Residual RAPID | 62.3 +/- 10.2% | 51.0 +/- 1.0% | 44.3 +/- 12.0% | 52.6% |

ผล seed 1000 ของ Residual RAPID ไม่ทำซ้ำใน seeds 2000/3000. Static risk
overlay จึงเป็น negative result สำคัญ: action sensitivity แบบ global สามารถหา
case ที่ยากได้ แต่ไม่ได้รับประกัน generalization และเพิ่ม variance ของ Clean กับ
Extreme อย่างมาก

## Research Decision

เลือกแนวทาง **Policy-Adaptive Counterfactual Exposure with clean-Risk
constraints (PACER-VLA)** แทนการสร้าง RAPID fixed mixture รุ่นใหม่

### Approaches considered

| Approach | Decision | Reason |
|---|---|---|
| 2-forward contextual bandit | **Selected for Model J** | compute เท่า V2 โดยประมาณ, adaptive และ ablate ได้ชัด |
| 3--5-forward exhaustive search | Reserve as Model K oracle | แพงและเสี่ยง hardest-example bias |
| Static curriculum/mixture | Rejected | Phase 8 ทดสอบแล้วและแพ้ F |

ผู้ใช้เลือกให้ทำ Model J ก่อนแล้วค่อย Model K และมอบอำนาจให้ agent ตัดสินใจ
technical trade-offs ต่อจากนี้ โดยมีข้อจำกัดว่า Model J ใช้ 2 forwards/sample.

## Model J Summary

แต่ละ batch ใช้ clean forward และ augmented forward โดย share flow noise/time.
Clean loss แบ่ง sample เป็น easy/medium/hard context. Contextual bandit เลือก
intervention หนึ่ง arm ต่อ sample จาก brightness, color, noise, blur, shadow,
geometry และ composed. Reward ใช้ action disagreement ที่ถูกลงโทษเมื่อ augmented
task loss สูงกว่า clean task loss มากเกินไป จึงหา productive difficulty ไม่ใช่
maximum difficulty

Clean-safety controller ใช้ fast/slow EMA ของ clean loss และปรับ augmented loss
weight ในช่วง 0.10--0.50. ถ้า clean loss เสื่อมเร็วกว่าค่าอ้างอิง controller จะลด
augmented weight; เมื่อเสถียรจะค่อย ๆ คืนสู่ 0.50. Inference เหมือน SmolVLA ปกติ
และไม่มี extra cost

## Preregistered Seed-1000 Gate

หลัง train 25,000 steps ประเมิน LIBERO Spatial 10 episodes/task:

- three-mode mean `>=58.3%`
- Clean `>=61%`
- Mild `>=59%`
- Extreme `>=46%`

ต้องผ่านทุกข้อก่อนขยาย eval seeds 2000/3000. Model K เริ่มได้ต่อเมื่อ Model J ผ่าน
gate นี้

## Paper Direction

Paper จะไม่ claim ว่า causal training ชนะ Domain Randomization จากผลปัจจุบัน แต่
จะวาง Model F เป็น strong baseline และทดสอบ claim ที่เฉพาะกว่า:

> Static risk curricula fail across seeds; sample-conditioned closed-loop
> interventions with a clean-retention constraint can target current policy
> vulnerabilities while preserving task competence.

งานที่ทับซ้อนและต้องแยก novelty ให้ชัด ได้แก่ RoCoDA, RoVLA, STRONG-VLA,
CofactVLA, LIBERO-Plus, LIBERO-PRO และ LIBERO-CF. จุดต่างหลักของ Phase 9 คือ
online contextual selection จาก current-policy feedback ร่วมกับ explicit clean
safety controller ภายใต้งบสอง forwards

## Status Checklist

- [x] Aggregate Residual RAPID 3-seed result
- [x] Select Model J then conditional Model K sequence
- [x] Lock Model J to two forwards/sample
- [x] Write PACER-VLA design and preregister gates
- [ ] Review and approve written design
- [ ] Write implementation plan
- [ ] Implement Model J with TDD
- [ ] Run Mac unit tests and MPS smoke
- [ ] Commit and push GPU-server workflow

