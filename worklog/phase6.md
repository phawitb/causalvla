# Phase 6: Evaluation & Results

## สถานะปัจจุบัน

| งาน | สถานะ | หมายเหตุ |
|---|---|---|
| Full training A–E | **COMPLETED** | ทุกโมเดลครบ 25,000 steps |
| ตรวจ OOD evaluation wrapper | **FIXED LOCALLY** | แก้ processor ให้รับ LeRobot `EnvTransition` และ perturb observation images จริง |
| Mac MPS environment | **PASS** | PyTorch 2.11, MPS, hf-libero 0.1.4, MuJoCo 3.8.1, robosuite 1.4.0 |
| OOD processor MPS test | **PASS** | level 0 คงรูปเดิม, level 1 เปลี่ยน image tensor จริง |
| Model A clean smoke test | **PASS** | task 0 สำเร็จ 1/1, 73 simulator steps, eval 97.95s รวม asset download |
| Model A mild OOD smoke test | **PASS (pipeline)** | task 0 จบครบแต่ policy ไม่สำเร็จ 0/1; eval 19.57s |
| MPS batch calibration | **PASS** | batch size 4, task 0 clean 4 episodes จบใน 47.95s, success 2/4 |
| Model A full clean | **COMPLETED — 67.4%** | 337/500 episodes สำเร็จ, batch size 4, seed 1000 |
| Evaluation matrix | **COMPLETED — 15/15 RUNS** | Model A–E ครบทั้ง 3 levels; ทุก run มี `eval_info.json`, metrics ครบ 10 tasks และมี rollout videos 100 ไฟล์/run |
| Remaining-run episode budget | **COMPLETED — 10 episodes/task** | Model C level 2 และ Model D/E ทุก levelใช้ 10 episodes/task; ผลที่จบก่อนหน้าคงไว้ที่ 50 episodes/task |

## จุดแก้สำคัญก่อน Evaluation

`scripts/eval_ood.py` เดิมเพิ่ม callable ธรรมดาเข้า processor pipeline แต่ LeRobot 0.6.1 ส่ง `EnvTransition` เข้าแต่ละ step ทำให้มีโอกาสรันผ่านโดยไม่ได้แก้ image observation จริง. แก้ `OODProcessorStep` ให้สืบทอด `ObservationProcessorStep`, implement `observation()` และรักษา feature schema ผ่าน `transform_features()` แล้ว.

ต้อง sync script ที่แก้แล้วไป GPU Server ก่อน smoke test และยืนยันว่า level 1 ทำให้ tensor รูปเปลี่ยน แต่ level 0 ไม่เปลี่ยน.

## Evaluation Matrix

| Model | Clean (`level_0`) | Mild (`level_1`) | Extreme (`level_2`) |
|---|---:|---:|---:|
| A — Standard SFT | **67.4% (337/500)** | **14.2% (71/500)** | **0.4% (2/500)** |
| B — Domain Randomization | **61.8% (309/500)** | **53.2% (266/500)** | **30.2% (151/500)** |
| C — CausalVLA | **44.4% (222/500)** | **3.2% (16/500)** | **1.0% (1/100)** |
| D — w/o Latent Loss | **47.0% (47/100)** | **2.0% (2/100)** | **1.0% (1/100)** |
| E — w/o Action Loss | **67.0% (67/100)** | **29.0% (29/100)** | **4.0% (4/100)** |

### ผลลัพธ์แยกตาม Task

แต่ละช่องแสดง **success rate (จำนวนสำเร็จ/50 episodes)** ของ LIBERO Spatial task นั้น ๆ

#### Model A — Standard SFT

| Task | Clean (`level_0`) | Mild (`level_1`) | Extreme (`level_2`) |
|---:|---:|---:|---:|
| 0 | **74% (37/50)** | **22% (11/50)** | **0% (0/50)** |
| 1 | **74% (37/50)** | **14% (7/50)** | **0% (0/50)** |
| 2 | **74% (37/50)** | **14% (7/50)** | **0% (0/50)** |
| 3 | **70% (35/50)** | **14% (7/50)** | **0% (0/50)** |
| 4 | **62% (31/50)** | **12% (6/50)** | **0% (0/50)** |
| 5 | **30% (15/50)** | **8% (4/50)** | **0% (0/50)** |
| 6 | **74% (37/50)** | **24% (12/50)** | **0% (0/50)** |
| 7 | **76% (38/50)** | **10% (5/50)** | **4% (2/50)** |
| 8 | **76% (38/50)** | **18% (9/50)** | **0% (0/50)** |
| 9 | **64% (32/50)** | **6% (3/50)** | **0% (0/50)** |
| **Overall** | **67.4% (337/500)** | **14.2% (71/500)** | **0.4% (2/500)** |

#### Model B — Domain Randomization

| Task | Clean (`level_0`) | Mild (`level_1`) | Extreme (`level_2`) |
|---:|---:|---:|---:|
| 0 | **76% (38/50)** | **76% (38/50)** | **40% (20/50)** |
| 1 | **68% (34/50)** | **62% (31/50)** | **16% (8/50)** |
| 2 | **74% (37/50)** | **68% (34/50)** | **38% (19/50)** |
| 3 | **68% (34/50)** | **56% (28/50)** | **42% (21/50)** |
| 4 | **68% (34/50)** | **56% (28/50)** | **14% (7/50)** |
| 5 | **16% (8/50)** | **22% (11/50)** | **2% (1/50)** |
| 6 | **46% (23/50)** | **44% (22/50)** | **44% (22/50)** |
| 7 | **82% (41/50)** | **60% (30/50)** | **38% (19/50)** |
| 8 | **62% (31/50)** | **34% (17/50)** | **34% (17/50)** |
| 9 | **58% (29/50)** | **54% (27/50)** | **34% (17/50)** |
| **Overall** | **61.8% (309/500)** | **53.2% (266/500)** | **30.2% (151/500)** |

Model C–E ครบทั้ง 3 OOD levels แล้ว. ตารางราย task และจำนวน episode ของทุก run เก็บใน `outputs/eval/reports/eval_summary.csv`; ผลรวมอ่านได้จาก `outputs/eval/reports/eval_summary.md` และ `eval_info.json` ของแต่ละ run.

## Final Evaluation Completion — 13 August 2026

Evaluation ครบ **15/15 runs** เวลา 17:08:52 (Europe/Helsinki) โดยคิวจบด้วยข้อความ `ALL 15 EVALUATION RUNS COMPLETE` และไม่พบ `Traceback`, `RuntimeError` หรือ MPS out-of-memory ใน run ที่เหลือ. ทุก run มีวิดีโอ 100 ไฟล์ (10 ต่อ task); สำหรับ run ขนาด 50 episodes/task evaluator จำกัดการบันทึกไว้ 10 วิดีโอต่อ task แต่ metrics ยังคงรวมครบ 500 episodes.

| Model | Clean | Mild OOD | Strong OOD |
|---|---:|---:|---:|
| A — Standard SFT | 67.4% (337/500) | 14.2% (71/500) | 0.4% (2/500) |
| B — Domain Randomization | 61.8% (309/500) | 53.2% (266/500) | 30.2% (151/500) |
| C — CausalVLA | 44.4% (222/500) | 3.2% (16/500) | 1.0% (1/100) |
| D — w/o Latent Loss | 47.0% (47/100) | 2.0% (2/100) | 1.0% (1/100) |
| E — w/o Action Loss | 67.0% (67/100) | 29.0% (29/100) | 4.0% (4/100) |

หมายเหตุด้านการเปรียบเทียบ: A/B ทุก level และ C Clean/Mild ใช้ 50 episodes/task; C Strong และ D/E ทุก levelใช้ 10 episodes/task ตามคำสั่งลดงบ evaluation. ดังนั้น uncertainty ของผล 10 episodes/task สูงกว่า และควรรายงาน denominator ควบคู่กับเปอร์เซ็นต์เสมอ.

## Step 1 — Environment Preflight (Mac M2 + MPS)

```bash
conda activate causalvla
cd ~/projects/causalvla

export PYTHONNOUSERSITE=1
export HF_HOME=~/hf_cache/causalvla
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MUJOCO_GL=glfw

python - <<'PY'
modules = ["torch", "lerobot", "libero", "robosuite", "mujoco"]
for name in modules:
    try:
        module = __import__(name)
        print(name, "PASS", getattr(module, "__version__", "unknown"))
    except Exception as error:
        print(name, "FAIL", repr(error))
PY
```

## Step 2 — Model A Clean Smoke Test

รัน task 0 เพียง 1 episode และ batch size 1 ก่อน เพื่อพิสูจน์ว่า simulator, policy loading, observation mapping และ result serialization ทำงานครบ:

```bash
mkdir -p logs/eval outputs/eval/preflight

python scripts/eval_ood.py \
  --policy.path=phawitbinabik/causalvla-model-a-sft \
  --policy.device=mps \
  --env.type=libero \
  --env.task=libero_spatial \
  '--env.task_ids=[0]' \
  --ood_level=level_0 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/preflight/model_a_level0_task0_1ep \
  --seed=1000 \
  2>&1 | tee logs/eval/model_a_level0_task0_1ep.log
```

Smoke test ผ่านเมื่อมี `End of OOD eval`, ไม่มี traceback และไฟล์นี้ถูกสร้าง:

```bash
cat outputs/eval/preflight/model_a_level0_task0_1ep/eval_info.json
```

หลัง clean smoke test ผ่าน ต้องทำ level 1 smoke test และตรวจใน log ว่า `OODProcessorStep(level=level_1)` ถูกเพิ่มใน pipeline ก่อนเริ่ม 15 full runs.

## Smoke Test Results — Mac MPS

| Model | Task | Level | Result | เวลา | Artifact |
|---|---:|---|---:|---:|---|
| A | 0 | Clean | 1/1 (100%) | 97.95s | JSON + MP4 |
| A | 0 | Mild | 0/1 (0%) | 19.57s | JSON + MP4 |

Clean episode สำเร็จประมาณ simulator step 72/280. Mild episode รันครบ 280 steps แต่ไม่สำเร็จ ซึ่งเป็นเพียง smoke test 1 episode จึงยังใช้เป็นผลเชิงสถิติไม่ได้. ทั้งสองรอบสร้าง `eval_info.json` และ rollout video สำเร็จ ยืนยันว่า policy loading, MPS inference, LIBERO simulator, camera rename, OOD injection และ result serialization ทำงานครบสาย.

Camera mapping ที่จำเป็นสำหรับทุก evaluation command:

```bash
'--rename_map={"observation.images.image2":"observation.images.wrist_image"}'
```

## Full Evaluation Runner — Mac MPS

สร้าง runner แบบ resume-safe ที่ `scripts/run_eval_mps.sh`. ถ้ามี `eval_info.json` ของ run นั้นแล้ว script จะข้ามโดยไม่เขียนทับผลเดิม.

```bash
./scripts/run_eval_mps.sh a level_0 1000
./scripts/run_eval_mps.sh a level_1 1000
./scripts/run_eval_mps.sh a level_2 1000
```

ผล A/B ทุก level และ C level 0–1 ประเมิน LIBERO Spatial ทั้ง 10 tasks, 50 episodes ต่อ task รวม 500 episodes. ตั้งแต่ C level 2 และ D/E ทุก level ปรับเป็น 10 episodes ต่อ task รวม 100 episodes ตามแผนลดเวลา โดยยังใช้ batch size 4, seed 1000 และบันทึก log/result แยกตาม model, OOD level และ seed.

คำสั่ง runner สำหรับ 10 episodes/task:

```bash
./scripts/run_eval_mps.sh c level_2 1000 10
./scripts/run_eval_mps.sh d level_0 1000 10
```

คิวอัตโนมัติสำหรับประเมิน A–E ครบทุก level ใช้ `scripts/run_all_evals_mps.sh`. คิวข้าม run ที่มี `eval_info.json` สมบูรณ์, retry ได้สูงสุด 3 ครั้งเมื่อ run ล้มเหลว และอัปเดตรายงานหลังจบแต่ละ run:

```bash
caffeinate -dimsu ./scripts/run_all_evals_mps.sh
```

รายงานสะสมถูกสร้างที่:

- `outputs/eval/reports/eval_summary.md`
- `outputs/eval/reports/eval_summary.csv`
- `logs/eval/all_models_queue.log`

## Metrics หลัก

- Success rate ต่อ model/level พร้อมจำนวน successes และจำนวน episodes
- Robustness drop: clean success rate − OOD success rate
- Action jitter: mean `||a_t − a_{t-1}||`
- Mean/std หรือ confidence interval ข้าม seeds/episodes
- Latent drift และ trajectory smoothness โดยเน้นเปรียบเทียบ C กับ D/E
