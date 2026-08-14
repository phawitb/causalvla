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

## RAPID-Lite Implementation

RAPID-Lite เป็น static risk-weighted ablation ที่ยังคงข้อดีของ Model F:

- ใช้ VLA forward เพียงครั้งเดียวต่อ batch
- augmentation ทำเฉพาะ training path; inference เหมือน SmolVLA
- `aug_probability=0.5` เท่ากับ Model F เพื่อควบคุม augmentation exposure
- candidate เริ่มต้นจาก pilot ถูกแทนที่ด้วยผล profiling 3 seeds × 256 samples
  หลังผ่าน profiling gate
- sampling probability สร้างจาก softmax ของ log risk พร้อม
  `exploration_floor=0.10`; ไม่มี candidate ใดถูกตัดเป็นศูนย์
- checkpoint บันทึก `profile_revision`, `risk_temperature` และ
  `exploration_floor` เพื่อ audit curriculum ได้

ไฟล์หลัก:

- `causal_aug/causal_aug/risk_sampler.py`
- `lerobot_patches/rapid_lite/configuration_rapid_lite.py`
- `lerobot_patches/rapid_lite/modeling_rapid_lite.py`
- `lerobot_patches/rapid_lite/processor_rapid_lite.py`

### MPS Training Smoke Test

รัน training จริง 2 steps บน Apple M2/MPS สำเร็จ และ checkpoint ที่ step 2 มี
model weights, optimizer, scheduler และ RNG state ครบ. Serialized config ผ่าน:

```text
type=rapid_lite
aug_probability=0.5
risk_temperature=1.0
exploration_floor=0.1
profile_revision=phase8-3seed-256samples-robust-risk-v1
```

Unit tests ของ intervention bank, sampler และ multi-seed aggregator ผ่าน `23 tests`.

## GPU Profiling Gate Before Full Training

ห้ามใช้ pilot 8 samples เป็น final curriculum สำหรับ paper. บน GPU server ให้ดึง
commit ล่าสุด ติดตั้ง policy และสร้าง profiles อย่างน้อย 3 seeds ก่อน:

```bash
cd ~/projects/causalvla
git pull origin main

python scripts/install_policy_patches.py rapid_lite

for SEED in 1000 2000 3000; do
  PYTHONPATH="$PWD/lerobot/src:$PWD/causal_aug" python scripts/profile_interventions.py \
    --device cuda \
    --samples 256 \
    --batch-size 8 \
    --seed "$SEED" \
    --intensities 0.25 0.5 0.75 1.0 \
    --output-dir "outputs/phase8/profiles/seed_${SEED}"
done

python scripts/aggregate_intervention_profiles.py \
  outputs/phase8/profiles/seed_1000/summary.json \
  outputs/phase8/profiles/seed_2000/summary.json \
  outputs/phase8/profiles/seed_3000/summary.json \
  --min-guard-rate 0.95 \
  --uncertainty-penalty 1.0 \
  --top-k 3 \
  --output outputs/phase8/rapid_lite_profile.json
```

หลังรวม ranking ข้าม seeds แล้วจึงอัปเดต `RAPID_LITE_CANDIDATES` และ
`profile_revision` ก่อน full training. เกณฑ์ขั้นต่ำคือ candidate ต้องผ่าน semantic
guard อย่างสม่ำเสมอ และ ranking ต้องไม่กลับทิศอย่างรุนแรงระหว่าง seeds

aggregator ใช้ robust risk `mean sensitivity − 1×standard deviation` เพื่อไม่ให้
intervention ที่คะแนนสูงจาก seed เดียวครอง curriculum และคัด guard ก่อนจัดอันดับ

### Multi-Seed Profiling Result

GPU profiling ครบ seeds `1000, 2000, 3000`, seed ละ 256 samples แล้ว. Candidates
ที่ผ่าน `mean guard ≥ 95%` และมี robust risk สูงสุดคือ:

| Rank | Family | Intensity | Robust risk | Raw risk weight | Mean guard |
|---:|---|---:|---:|---:|---:|
| 1 | shadow | 0.75 | 0.023396 | 0.379 | 100.0% |
| 2 | brightness | 0.50 | 0.021158 | 0.342 | 99.0% |
| 3 | geometry | 1.00 | 0.017234 | 0.279 | 100.0% |

ค่าชุดนี้ถูกตรึงเป็น `RAPID_LITE_CANDIDATES` และ revision
`phase8-3seed-256samples-robust-risk-v1`. ระหว่าง training sampler จะผสม raw
risk distribution กับ uniform distribution 10% ตาม `exploration_floor=0.10`;
ดังนั้น effective probabilities โดยประมาณคือ `0.374, 0.341, 0.285` ตามลำดับ

### Provisional Full-Training Command

คำสั่งนี้ใช้หลังผ่าน profiling gate และ commit candidate revision ใหม่แล้ว:

```bash
lerobot-train \
  --policy.type=rapid_lite \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=phawitbinabik/causalvla-rapid-lite \
  --policy.private=true \
  --policy.aug_probability=0.5 \
  --policy.risk_temperature=1.0 \
  --policy.exploration_floor=0.1 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=15000 \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir=outputs/final/rapid_lite \
  --job_name=rapid_lite \
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

## RAPID-Lite Full Training Result

Full training seed 1000 เสร็จครบ 25,000 steps บน RTX 4090 ใน `4:11:54`.
Final metrics ที่ step 25K:

```text
loss=0.405
gradient_norm=1.637
augmented_fraction=0.506
curriculum/shadow:0.75=0.194
curriculum/brightness:0.5=0.166
curriculum/geometry:1.0=0.146
GPU memory=4.28 GB
```

ผลรวม curriculum fractions เท่ากับ augmented fraction โดยประมาณ และสัดส่วน
conditional เมื่อหารด้วย `0.506` คือ `38.3%, 32.8%, 28.9%` ซึ่งใกล้กับ effective
target `37.4%, 34.2%, 28.4%`. Training จบด้วย `End of training`, checkpoint step
25,000 ผ่าน config validation และ push ไปที่:

```text
Repo: phawitbinabik/causalvla-rapid-lite
Revision: bad76c163d35e3254d976985f1f8a1f148672a2c
Files: 69
```

### Primary Evaluation Protocol

ประเมิน clean/mild/extreme ที่ 10 episodes/task และ eval seeds 1000/2000/3000
ก่อน. เทียบ RAPID-Lite กับ Model F และ V2 ภายใต้ protocol เดียวกัน. ใช้ revision
ที่ pin ไว้และไม่เลือก checkpoint จากผล eval

## Scientific Caveats

- action sensitivity เป็น diagnostic ของ policy ภายใต้ shared stochastic variables
  ไม่ใช่ causal effect estimate
- visual-embedding cosine similarity เป็น proxy ของ semantic preservation; ต้องเสริม
  ด้วย task-aware checks และ qualitative audit
- การใช้ข้อมูลฝึกทำ profiling อาจ overfit intervention bank; held-out families และ
  independently generated OOD suites จึงเป็น evaluation บังคับ
- Phase 7 มีหลาย eval seeds แต่ยังมี training seed เดียว จึงยังไม่ควร claim
  statistical significance หรือ Q1 readiness
