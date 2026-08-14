# Phase 8: RAPID-VLA — Policy-Guided Intervention Curriculum

## Motivation

ผล Phase 7 จาก 3 evaluation seeds แสดงว่า Model F (Online DR) ชนะ CausalVLA-v2
เฉลี่ย `+5.3 pp` บน clean, `+1.0 pp` บน mild OOD และ `+5.0 pp` บน extreme OOD.
ดังนั้น Phase 8 จะไม่เพิ่ม paired forward หรือ auxiliary loss โดยยังไม่มีหลักฐาน
แต่ต่อยอดข้อค้นพบที่แข็งแรงกว่า: **online intervention ช่วย และควรเลือก intervention
จากความเปราะบางของ policy แทนการสุ่มแบบคงที่**

ชื่อทำงานของวิธีคือ **RAPID-VLA** (Risk-Adaptive Policy-guided Intervention
Distribution for VLA). ขั้นแรกของ Phase 8 คือ Intervention Profiler; ยังไม่ถือว่า
RAPID-VLA ตัวเต็มหรือเป็นผลยืนยันเชิง causal

## Research Hypothesis

เมื่อควบคุม flow noise และ time ให้เหมือนกัน ภาพ clean และภาพ intervention
จาก scene เดียวกันจะเผยให้เห็นว่า policy ไวต่อ nuisance family ใดมากที่สุด.
การให้น้ำหนัก online augmentation ตาม risk นี้ โดยมี semantic guard ป้องกัน
label-breaking transforms น่าจะใช้ training budget ได้ดีกว่า uniform Online DR
(Model F)

## Intervention Profiler

Profiler ใช้ Model F checkpoint ที่ revision
`997d94a9325bc359422cd3cf54bd74b0a4c9be98` และ LIBERO Spatial training
observations โดยไม่มีการแก้ checkpoint

สำหรับแต่ละภาพ `x` และ intervention `T_k(x; s)`:

1. ใช้ action target, flow noise และ diffusion/flow time ชุดเดียวกัน
2. วัด `action_sensitivity = mean(abs(v_aug - v_clean))`
3. วัด `loss_increase = L_aug - L_clean`
4. วัด cosine similarity ของ pooled frozen SmolVLA visual embeddings
5. อนุญาต intervention เข้า curriculum เมื่อ semantic similarity ผ่าน threshold

Intervention bank รุ่นแรกมี 7 families:

- brightness
- color
- noise
- blur
- shadow
- geometry
- composed

แต่ละ family รองรับ intensity `0.0–1.0`; transformation เดียวกันถูกใช้กับทุก
camera view ใน sample เพื่อคงความสอดคล้องข้ามกล้อง

## Reproducible Pilot Command (Apple M2 / MPS)

```bash
cd /Users/phawit/Projects/CausalVLA
PYTHONPATH="$PWD/lerobot/src:$PWD/causal_aug" PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /opt/miniconda3/envs/causalvla/bin/python scripts/profile_interventions.py \
  --samples 8 --batch-size 2 \
  --intensities 0.25 0.5 0.75 1.0 \
  --output-dir outputs/phase8/pilot_profile
```

ผลลัพธ์ประกอบด้วย `profile.json`, `profile.csv`, `summary.json` และ
`summary.md`. Pilot 8 samples ใช้ตรวจ pipeline และหา signal เบื้องต้นเท่านั้น
ยังไม่ใช้เป็นผลทดลองใน paper

## MPS Smoke Test and Pilot Result

Implementation ผ่าน MPS smoke test และ pilot ครบ `8 samples × 7 families × 4
intensities`. ค่าเด่นจาก pilot:

| Intervention | Intensity | Action sensitivity | Semantic similarity | Guard pass rate |
|---|---:|---:|---:|---:|
| composed | 1.00 | 0.070779 | 0.5818 | 0% |
| shadow | 1.00 | 0.047282 | 0.8465 | 25% |
| brightness | 1.00 | 0.046594 | 0.8270 | 25% |
| shadow | 0.75 | 0.031007 | 0.9365 | 100% |
| composed | 0.25 | 0.025695 | 0.9571 | 100% |
| geometry | 1.00 | 0.023548 | 0.9992 | 100% |

ผลนี้ตรวจพบสิ่งที่ profiler ถูกออกแบบมาให้แยก: intervention ที่แรงที่สุดให้
sensitivity สูง แต่ composed/brightness/shadow ระดับแรงมักไม่ผ่าน semantic guard.
candidate เบื้องต้นสำหรับ RAPID-Lite จึงควรมาจากจุดที่ risk สูงและยังผ่าน guard
เช่น shadow 0.75, composed 0.25 และ geometry 1.0 ไม่ใช่เลือกค่าที่ sensitivity
สูงที่สุดอย่างเดียว. ตัวเลขนี้เป็น engineering pilot ขนาดเล็กและต้อง rerun ด้วย
sample/seed มากขึ้นก่อนตรึง sampling distribution

## Phase 8 Decision Ladder

1. **Profiler pilot:** ตรวจ shared-noise measurement, intensity monotonicity และ
   semantic guard
2. **RAPID-Lite:** สร้าง static sampling weights จาก normalized sensitivity เฉพาะ
   transforms ที่ผ่าน guard แล้วเปรียบเทียบกับ Model F ภายใต้ compute budget เท่ากัน
3. **Adaptive RAPID:** อัปเดต exponential-moving-average risk ระหว่าง training พร้อม
   exploration floor เพื่อไม่ให้ sampler collapse
4. **Held-out generalization:** แยก intervention families สำหรับ train/test เพื่อพิสูจน์
   ว่าไม่ได้จำ augmentation bank
5. **Paper-grade validation:** อย่างน้อย 3 training seeds, 10 episodes/task เป็น
   primary protocol และเพิ่ม episodes เฉพาะ comparison ที่มีแนวโน้มชนะ

## Stop/Go Criteria

- GO จาก profiler เมื่อ sensitivity มี reproducible ranking และ transform ที่ risk สูง
  ยังผ่าน semantic guard
- GO จาก RAPID-Lite เมื่อชนะ Model F บน extreme OOD โดยไม่ลด clean อย่างมีนัยสำคัญ
  และมีประสิทธิภาพต่อ training compute ดีกว่า
- STOP หรือ redesign ถ้า ranking เปลี่ยนมากตาม seed, semantic guard ตัด high-risk
  transforms เกือบทั้งหมด หรือ static sampler ไม่ชนะ uniform Online DR

## Scientific Caveats

- action sensitivity เป็น diagnostic ของ policy ภายใต้ shared stochastic variables
  ไม่ใช่ causal effect estimate
- visual-embedding cosine similarity เป็น proxy ของ semantic preservation; ต้องเสริม
  ด้วย task-aware checks และ qualitative audit
- การใช้ข้อมูลฝึกทำ profiling อาจ overfit intervention bank; held-out families และ
  independently generated OOD suites จึงเป็น evaluation บังคับ
- Phase 7 มีหลาย eval seeds แต่ยังมี training seed เดียว จึงยังไม่ควร claim
  statistical significance หรือ Q1 readiness
