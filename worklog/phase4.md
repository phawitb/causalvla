# Phase 4: OOD Perturbation Engine

> **วันที่:** 2 สิงหาคม 2026
> **สถานะ:** COMPLETED
> **เครื่อง:** Mac M2 (local)

---

## สิ่งที่ทำ

สร้าง OOD (Out-of-Distribution) perturbation engine สำหรับทดสอบ robustness ของ policy ภายใต้ visual distribution shift — ครอบคลุม perturbation ครบทุกประเภทจาก `Lerobot-Dataset-Manager` (`_augp_camera`, `_augp_light`) แปลงเป็น pure PyTorch GPU ops + evaluation script ที่ inject perturbation เข้า LeRobot eval pipeline + script สร้าง augmented dataset สำหรับ Model B (Domain Randomization)

---

## Step 4.1 — Augmented Dataset for Model B (Domain Randomization)

### แนวคิด

Model B ต้อง train บน augmented dataset เพื่อเปรียบเทียบกับ Model C (CausalVLA) อย่าง **fair** — จึงใช้ **perturbation functions ชุดเดียวกัน** กับที่ใช้ใน OOD evaluation (`ood_wrapper.py`) เพื่อให้:
- ถ้า Model C ชนะ Model B = เพราะ **causal learning approach** จริงๆ
- ไม่ใช่เพราะ augmentation types ต่างกัน

### ไฟล์ที่สร้าง/แก้ไข

| Action | ไฟล์ | หน้าที่ |
|--------|------|--------|
| CREATE | `scripts/augment_dataset.py` | สร้าง augmented dataset จาก source dataset |
| CREATE | `scripts/generate_preview_assets.py` | สร้าง MP4 videos + state/action JSON สำหรับ preview |
| CREATE | `augment_preview/index.html` | Visualization website (video + graph) |
| CREATE | `augment_preview/aug_params.json` | Per-episode augmentation params |
| CREATE | `augment_preview/ep_tasks.json` | Task instruction text ต่อ episode |
| CREATE | `augment_preview/ep_N.json` | Per-episode state/action timeseries (432 files, lazy loaded) |
| CREATE | `augment_preview/ep*_{orig,aug}.mp4` | 1,728 MP4 videos (432 ep x 2 cam x {orig,aug}) |
| CREATE | `index.html` | Home page (link ไป Document + Libero Dataset) |
| MODIFY | `pipeline.html` | เพิ่มปุ่ม "Augment Preview" + hash-based tab switch (`#tab=mywork`) |

### Architecture

```
augment_dataset.py
  ├── AUG_LEVELS                  ← 3 ระดับ: training (default), mild, heavy
  ├── sample_episode_params()     ← สุ่ม params ครั้งเดียวต่อ episode (seed = 42 + ep_idx)
  ├── _perspective_fixed()        ← perspective ด้วย pre-sampled offsets (ไม่สั่นต่อ frame)
  ├── augment_image()             ← ใช้ functions จาก ood_wrapper.py
  └── main()                      ← อ่าน dataset → augment → เขียน dataset ใหม่ → push to hub

generate_preview_assets.py
  ├── frames_to_mp4_pipe()        ← pipe raw RGB bytes → ffmpeg stdin (ไม่ต้อง temp PNG)
  ├── process_episode()           ← extract frames + state/action สำหรับ 1 episode
  ├── generate_aug_params()       ← สร้าง aug_params.json (same logic as augment_dataset.py)
  └── main()                      ← load ทั้ง 2 datasets → สร้าง MP4 + per-episode JSON
```

### Per-Episode Consistent Augmentation

เหมือน `Lerobot-Dataset-Manager` — แต่ละ episode สุ่ม params **ครั้งเดียว** แล้วใช้กับ **ทุก frame** ใน episode นั้น:

| ประเภท | Per-Episode (คงที่ทั้ง episode) | Per-Frame (สุ่มใหม่ทุก frame) |
|--------|-------------------------------|------------------------------|
| Geometric | rotation, perspective, affine | - |
| Photometric | brightness, contrast, saturation, hue, shadow | noise, blur |

เหตุผล: ในสถานการณ์จริง กล้องและแสงไม่เปลี่ยนระหว่าง episode แต่ noise/blur เปลี่ยนได้ทุก frame

**สำคัญ:** `_perspective` ใน `ood_wrapper.py` สุ่ม corner offsets **ทุกครั้งที่เรียก** (ออกแบบสำหรับ per-batch evaluation) — แต่ใน augmented dataset ต้อง consistent ภายใน episode ดังนั้นจึงสร้าง `_perspective_fixed()` ที่รับ pre-sampled offsets แทน เพื่อไม่ให้วิดีโอสั่น

### Training Augmentation Level (Default)

Geometric ตั้งไว้ **very mild** เพื่อไม่ให้สั่นเกินไป — เน้น photometric diversity แทน:

| Category | Perturbation | Range | หมายเหตุ |
|---|---|---|---|
| Geometric | Rotation | **±2°** | mild — แทบไม่เห็น |
| Geometric | Perspective | **mag 0.008** | very mild |
| Geometric | Affine translate | **±1%** | very mild |
| Geometric | Affine shear | **±0.5%** | very mild |
| Geometric | Affine scale | **0.98–1.02x** | very mild |
| Photometric | Brightness | 0.6–1.4x | moderate |
| Photometric | Contrast | 0.7–1.3x | moderate |
| Photometric | Saturation | 0.6–1.5x | moderate |
| Photometric | Hue shift | ±0.05 | moderate |
| Photometric | Shadow | 30% prob, α 0.1–0.35 | directional |
| Photometric | Noise (per-frame) | σ = 0.03 | slight per-frame variation |
| Photometric | Blur (per-frame) | 20% prob, k=3, σ=0.6 | occasional |

### Augmentation Levels Summary

| Level | ใช้ทำอะไร | Geometric | Photometric |
|-------|----------|-----------|-------------|
| **training** | สร้าง augmented dataset สำหรับ train Model B | very mild (rotation ±2°) | moderate (brightness 0.6-1.4x) |
| **mild** | augment เบาๆ (สำรอง) | rotation ±3° เท่านั้น | mild (brightness 0.8-1.2x) |
| **heavy** | augment หนัก + cutout (สำรอง) | rotation ±15°, perspective 0.05 | extreme (brightness 0.3-2.5x) + cutout 10% |

### Usage

```bash
# Full augmentation (432 episodes, ~52k frames)
cd /Users/phawit/Projects/CausalVLA/lerobot
/opt/miniconda3/envs/lerobot2/bin/python ../scripts/augment_dataset.py \
    --src_repo_id lerobot/libero_spatial_image \
    --dst_repo_id phawitbinabik/libero_spatial_augmented \
    --aug_level training

# Dry run (10 episodes)
/opt/miniconda3/envs/lerobot2/bin/python ../scripts/augment_dataset.py \
    --src_repo_id lerobot/libero_spatial_image \
    --dst_repo_id causalvla/libero_spatial_augmented \
    --aug_level training \
    --max_episodes 10

# Generate preview assets (MP4 + per-episode JSON)
/opt/miniconda3/envs/lerobot2/bin/python ../scripts/generate_preview_assets.py

# Generate preview for first N episodes only
/opt/miniconda3/envs/lerobot2/bin/python ../scripts/generate_preview_assets.py --max 20

# Push existing dataset to Hub (without re-augmenting)
/opt/miniconda3/envs/lerobot2/bin/python -c "
import sys; sys.path.insert(0, 'src')
from lerobot.datasets import LeRobotDataset; from pathlib import Path
ds = LeRobotDataset('phawitbinabik/libero_spatial_augmented',
    root=Path('../lerobot/datasets/libero_spatial_augmented'))
ds.push_to_hub(tags=['augmented','domain-randomization','libero','causalvla'],
    license='apache-2.0', private=False)
"
```

### Verification

```
432-episode augmented dataset (52,970 frames):
  Shape preserved: [3, 256, 256] ✓
  Values clamped [0, 1] ✓
  Speed: ~4.5 sec/episode on Mac M2 (~31 min total)
  Per-episode consistency: geometric+photometric fixed, noise/blur per-frame ✓
  Videos stable (no shaking from perspective) ✓
  All 1,728 MP4 files generated (885 MB total) ✓
  Pushed to HuggingFace Hub ✓
```

### Visualization

Interactive preview website เปรียบเทียบ original vs augmented:

```
augment_preview/
├── index.html                         ← Visualization (video player + Chart.js graphs)
├── aug_params.json                      ← Per-episode augmentation params + aug_level
├── ep_tasks.json                        ← Task instruction text per episode
├── ep_0.json ... ep_431.json            ← Per-episode state/action timeseries (lazy loaded)
├── ep{0-431}_image_{orig,aug}.mp4       ← 864 MP4: Top camera
├── ep{0-431}_wrist_image_{orig,aug}.mp4 ← 864 MP4: Wrist camera
└── (2,166 files total)
```

```bash
cd /Users/phawit/Projects/CausalVLA
python -m http.server 8000
# http://localhost:8000                → Home page
# http://localhost:8000/pipeline.html  → Document (มีปุ่ม Augment Preview)
# http://localhost:8000/augment_preview/ → Augmented Dataset Preview
```

Features:
- **Navigation tabs:** Planning / My Work / Augment Preview — สลับไปมาระหว่าง pipeline.html กับ augment_preview ได้เลย (ใช้ `#tab=` hash)
- **Left sidebar:** Episode 0-431 พร้อม frame count, duration, aug level — scroll ได้
- **Task instruction:** แสดง text instruction ของ task นั้นๆ ในกล่องสีฟ้า (10 unique LIBERO tasks)
- **Augmentation params:** แสดงทุก parameter ที่ใช้ (rotation, brightness, contrast, shadow, etc.) + ระบุ **aug level**
- **Video player:** 4 วิดิโอแถวเดียวกัน (Top Orig, Top Aug, Wrist Orig, Wrist Aug) — ซิงค์ทุกวิดิโอ
- **Seekbar:** ลากได้ + Play All / Pause / Reset + แสดง frame/total + เวลา
- **State/Action graphs:** Chart.js line chart — State 8-dim + Action 7-dim — มี **playhead แท่งแดง** วิ่งตาม video playback (ใช้ chartjs-plugin-annotation)
- **Lazy loading:** โหลดเฉพาะ ep ที่กดดู (ep_N.json ~32KB) + cache ใน `epDataCache` — ไม่โหลด 16MB JSON ทีเดียว

### Dataset Paths

| Dataset | Path |
|---------|------|
| Original | [`lerobot/libero_spatial_image`](https://huggingface.co/datasets/lerobot/libero_spatial_image) (HuggingFace Hub) |
| Augmented (local) | `/Users/phawit/Projects/CausalVLA/lerobot/datasets/libero_spatial_augmented/` |
| Augmented (Hub) | [`phawitbinabik/libero_spatial_augmented`](https://huggingface.co/datasets/phawitbinabik/libero_spatial_augmented) |
| Preview assets | `/Users/phawit/Projects/CausalVLA/augment_preview/` |
| Augment script | `/Users/phawit/Projects/CausalVLA/scripts/augment_dataset.py` |
| Preview script | `/Users/phawit/Projects/CausalVLA/scripts/generate_preview_assets.py` |

### Dataset Info

| Item | Value |
|------|-------|
| Source | `lerobot/libero_spatial_image` |
| Total episodes | 432 |
| Total frames | 52,970 |
| Resolution | 256 x 256 |
| Cameras | 2 (image + wrist_image) |
| State dim | 8 (x, y, z, rx, ry, rz, grip_l, grip_r) |
| Action dim | 7 (dx, dy, dz, drx, dry, drz, gripper) |
| FPS | 10 |
| Robot | panda |
| Tasks | 10 unique LIBERO spatial instructions |
| Local size | ~13 GB |
| Aug level | training (default) |

### Push to HuggingFace Hub

Dataset ถูก push ไปที่ [`phawitbinabik/libero_spatial_augmented`](https://huggingface.co/datasets/phawitbinabik/libero_spatial_augmented) เพื่อให้:
- **ใช้ train ได้จากทุกเครื่อง** — ไม่ต้อง augment ใหม่ (ประหยัดเวลา ~30 นาที/ครั้ง)
- **Reproducibility** — dataset version ถูก tag ตาม codebase version
- **Collaboration** — แชร์กับคนอื่นได้ง่าย

```bash
# Push existing local dataset to Hub
cd lerobot
python -c "
import sys; sys.path.insert(0, 'src')
from lerobot.datasets import LeRobotDataset
from pathlib import Path

ds = LeRobotDataset(
    repo_id='phawitbinabik/libero_spatial_augmented',
    root=Path('../lerobot/datasets/libero_spatial_augmented'),
)
ds.push_to_hub(
    tags=['augmented', 'domain-randomization', 'libero', 'causalvla'],
    license='apache-2.0',
    private=False,
)
"

# Or push during augmentation (one step)
python ../scripts/augment_dataset.py \
    --src_repo_id lerobot/libero_spatial_image \
    --dst_repo_id phawitbinabik/libero_spatial_augmented \
    --aug_level training \
    --push_to_hub

# Load from Hub (on any machine)
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset("phawitbinabik/libero_spatial_augmented")
```

| Detail | Value |
|--------|-------|
| Hub repo | `phawitbinabik/libero_spatial_augmented` |
| Visibility | public |
| License | Apache 2.0 |
| Tags | `augmented`, `domain-randomization`, `libero`, `causalvla` |
| Upload size | ~13 GB (images + parquet) |

### Augmentation Run Time

| เครื่อง | 432 episodes | หมายเหตุ |
|--------|-------------|---------|
| Mac M2 (local) | ~31 นาที | 4.5 sec/ep × 432 (bottleneck = I/O save) |
| Colab (GPU) | ~15 นาที | disk เร็วกว่า |

### คำสั่งที่รันจริง

```bash
# 1. สร้าง augmented dataset (ทั้ง 432 episodes)
cd /Users/phawit/Projects/CausalVLA/lerobot
/opt/miniconda3/envs/lerobot2/bin/python ../scripts/augment_dataset.py \
    --src_repo_id lerobot/libero_spatial_image \
    --dst_repo_id causalvla/libero_spatial_augmented \
    --aug_level training
# ผลลัพธ์: datasets/libero_spatial_augmented/ (~13 GB, 31 นาที)

# 2. สร้าง preview assets (MP4 + JSON)
/opt/miniconda3/envs/lerobot2/bin/python ../scripts/generate_preview_assets.py
# ผลลัพธ์: 1,728 MP4 files (885 MB), aug_params.json, ep_0.json - ep_431.json

# 3. สร้าง ep_tasks.json (task instruction ต่อ episode)
/opt/miniconda3/envs/lerobot2/bin/python -c "
import sys, json; sys.path.insert(0, 'src')
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset('lerobot/libero_spatial_image', episodes=list(range(432)))
tasks = {}
for ep in range(432):
    fi = ds.meta.episodes['dataset_from_index'][ep]
    item = ds[fi]
    tasks[f'ep_{ep}'] = item.get('task', '')
with open('../augment_preview/ep_tasks.json', 'w') as f:
    json.dump(tasks, f)
"
# ผลลัพธ์: ep_tasks.json (36 KB, 10 unique task instructions)

# 4. Push dataset to HuggingFace Hub
/opt/miniconda3/envs/lerobot2/bin/python -c "
import sys; sys.path.insert(0, 'src')
from lerobot.datasets import LeRobotDataset
from pathlib import Path
ds = LeRobotDataset(
    repo_id='phawitbinabik/libero_spatial_augmented',
    root=Path('../lerobot/datasets/libero_spatial_augmented'),
)
ds.push_to_hub(
    tags=['augmented', 'domain-randomization', 'libero', 'causalvla'],
    license='apache-2.0', private=False,
)
"

# 5. เปิด preview website
cd /Users/phawit/Projects/CausalVLA
python -m http.server 8000
# http://localhost:8000/augment_preview/
```

**หมายเหตุ:** ต้องใช้ `/opt/miniconda3/envs/lerobot2/bin/python` ไม่ใช่ `python3` ของ system เพราะ default python ไม่มี `draccus`, `lerobot` dependencies

### ปัญหาที่เจอและแก้ไข

| ปัญหา | สาเหตุ | วิธีแก้ |
|-------|--------|--------|
| วิดีโอ augmented สั่นหนัก | `_perspective()` ใน `ood_wrapper.py` สุ่ม corner offsets ใหม่ทุก frame | สร้าง `_perspective_fixed()` ที่รับ pre-sampled offsets, สุ่มครั้งเดียวต่อ episode |
| Geometric augment เห็นชัดเกินไป | Range ตั้งไว้สูง (rotation ±15°) | ลดเหลือ very mild: rotation ±2°, perspective 0.008, translate ±1% |
| Dataset path ผิด | `augment_dataset.py` รันจาก `lerobot/` dir เขียนไปที่ `lerobot/datasets/` แต่ `generate_preview_assets.py` หา path ผิด | แก้ `AUG_ROOT` ให้ชี้ `parent.parent / "lerobot" / "datasets" / "libero_spatial_augmented"` |
| `episode_data.json` 16 MB ทำให้ browser ค้าง | JSON ใหญ่เกินโหลดทีเดียว | แยกเป็น 432 ไฟล์ `ep_N.json` (~32 KB/file) + lazy loading ใน JS |
| `hasData` used before declaration | ใน HTML ใช้ตัวแปรก่อนประกาศ | ย้าย `const hasData = !!epData` ขึ้นไปก่อน playback bar |
| `draccus` module not found | `python3` default เป็น homebrew ไม่ใช่ conda env | ใช้ `/opt/miniconda3/envs/lerobot2/bin/python` |
| Push to Hub 403 Forbidden | `ds.repo_id` เป็น `causalvla/...` ไม่ใช่ `phawitbinabik/...` | Load dataset ด้วย `repo_id='phawitbinabik/...'` ตั้งแต่แรก |
| Task instruction ไม่แสดง | `ds.meta.episodes['task_index']` ใช้ไม่ได้ | ใช้ `ds[first_frame_idx]['task']` แทน (อ่านจาก per-frame data) |

### การแก้ไข pipeline.html

| สิ่งที่เพิ่ม | รายละเอียด |
|------------|-----------|
| ปุ่ม "Augment Preview" | เพิ่มใน tab bar ของ pipeline.html, link ไป `augment_preview/` |
| Hash-based tab switch | เพิ่ม `window.onload` handler อ่าน `location.hash` เช่น `pipeline.html#tab=mywork` แล้วเรียก `switchTab()` อัตโนมัติ — เพื่อให้กดจาก augment_preview กลับมาเปิดถูก tab |

### การแก้ไข augment_preview/index.html (Timeline)

| Version | การเปลี่ยนแปลง |
|---------|---------------|
| v1 | 10 episodes, sidebar + video player (2 วิดิโอ) + Chart.js graphs |
| v2 | แก้วิดีโอสั่น (ลด geometric ranges), เพิ่ม aug_level label |
| v3 | 4 วิดิโอในแถวเดียว (Top Orig, Top Aug, Wrist Orig, Wrist Aug), playhead แท่งแดงบน chart |
| v4 | ขยายเป็น 432 episodes, แก้ 16MB JSON → lazy loading per-episode |
| v5 | เพิ่ม task instruction text, seekbar ลากได้ |
| v6 | เพิ่ม navigation tabs (Planning / My Work / Augment Preview) + hash-based tab switch |

---

## Step 4.2 — OOD Perturbation Module (ood_wrapper.py)

### ไฟล์ที่สร้าง

```
causal_aug/causal_aug/ood_wrapper.py
```

### Perturbation Types (อ้างอิง Lerobot-Dataset-Manager)

#### Camera (Geometric)

| Perturbation | Function | คำอธิบาย |
|---|---|---|
| **Perspective** | `_perspective()` | Random perspective warp ด้วย corner offsets, ใช้ `grid_sample` |
| **Affine** | `_affine()` | Translate + shear + scale, ใช้ `affine_grid` + `grid_sample` |
| **Rotation** | `_rotation()` | หมุนภาพตาม angle (degrees), ใช้ `grid_sample` + `border` padding |

#### Light (Photometric)

| Perturbation | Function | คำอธิบาย |
|---|---|---|
| **Brightness** | `_brightness()` | Multiplicative factor ต่อ pixel values |
| **Contrast** | `_contrast()` | ปรับ contrast รอบ per-image mean |
| **Saturation** | `_saturation()` | ITU-R 601 luma-based saturation |
| **Hue shift** | `_hue_shift()` | Rodrigues' rotation ใน RGB space |
| **Shadow** | `_shadow()` | Directional gradient (left/right/top/bottom) |
| **Gaussian noise** | `torch.randn_like()` | Additive noise ด้วย σ ที่กำหนด |
| **Gaussian blur** | `_gaussian_blur()` | Separable 2D convolution ด้วย Gaussian kernel |

#### Occlusion

| Perturbation | Function | คำอธิบาย |
|---|---|---|
| **Cutout** | `_cutout()` | Random rectangular black patch |

### OOD Levels

#### Level 0 — In-Distribution (Clean)

ไม่มี perturbation ใดๆ — return input โดยตรง

#### Level 1 — Mild OOD

| Category | Perturbation | Range/Value |
|---|---|---|
| Geometric | Rotation | ±5° |
| Photometric | Brightness | 0.7–1.3x |
| Photometric | Contrast | 0.8–1.2x |
| Photometric | Saturation | 0.7–1.3x |
| Photometric | Hue shift | ±0.03 |
| Photometric | Noise | σ = 0.05 |
| Photometric | Blur | 20% prob, k=3, σ=0.5 |
| Photometric | Shadow | 20% prob, α 0.1–0.25 |

#### Level 2 — Extreme OOD

| Category | Perturbation | Range/Value |
|---|---|---|
| Geometric | Perspective | mag 0.05 |
| Geometric | Affine translate | ±4% |
| Geometric | Affine shear | ±3% |
| Geometric | Affine scale | 0.9–1.1x |
| Geometric | Rotation | ±15° |
| Photometric | Brightness | 0.1–3.0x |
| Photometric | Contrast | 0.5–1.8x |
| Photometric | Saturation | 0.3–2.0x |
| Photometric | Hue shift | ±0.08 |
| Photometric | Noise | σ = 0.20 |
| Photometric | Blur | 40% prob, k=5, σ=1.2 |
| Photometric | Shadow | 40% prob, α 0.2–0.5 |
| Occlusion | Cutout | 15% black patch |

### API

```python
from causal_aug import OODPerturbation

perturb = OODPerturbation("level_1", seed=42)
perturbed = perturb(images)  # [B, C, H, W] float32 [0, 1] → [B, C, H, W]
```

### Perturbation Pipeline Order

```
Input [B, C, H, W]
  → Perspective (geometric)
  → Affine: translate + shear + scale (geometric)
  → Rotation (geometric)
  → Brightness (photometric)
  → Contrast (photometric)
  → Saturation (photometric)
  → Hue shift (photometric)
  → Shadow (photometric)
  → Gaussian noise (photometric)
  → Gaussian blur (photometric)
  → Cutout (occlusion)
  → Clamp [0, 1]
Output [B, C, H, W]
```

### Design Decisions

- **Pure PyTorch** — ไม่ใช้ OpenCV หรือ PIL, ทำงานบน GPU ได้โดยตรง
- **Per-sample randomization** — แต่ละ sample ใน batch ได้ perturbation ต่างกัน
- **Deterministic option** — รับ `seed` parameter สำหรับ reproducibility
- **Level 0 = identity** — return input โดยตรง, ไม่มี overhead
- **Geometric ก่อน Photometric** — ทำ spatial transform ก่อนแล้วค่อยเปลี่ยนสี/แสง ตาม convention ของ augmentation pipeline
- **Probabilistic perturbations** — Shadow และ Blur ใช้ probability เพื่อไม่ให้ทุก sample ได้ผลเหมือนกัน

---

## Step 4.3 — OOD Evaluation Script

### ไฟล์ที่สร้าง

```
scripts/eval_ood.py
```

### Architecture

```
eval_ood.py
  ├── OODEvalConfig(EvalPipelineConfig)     ← เพิ่ม --ood_level flag
  ├── OODProcessorStep                       ← inject เข้า env_preprocessor pipeline
  └── eval_ood_main()                        ← reuse lerobot eval_policy_all()
```

### Integration Strategy

แทนที่จะ fork/copy eval loop ทั้งหมด — inject `OODProcessorStep` เข้าไปใน `env_preprocessor` pipeline:

```
Env output (raw) → preprocess_observation() → env_preprocessor → OODProcessorStep → policy preprocessor → policy
```

`OODProcessorStep` ทำงานกับ observation dict ที่ผ่าน `preprocess_observation()` แล้ว:
- Images เป็น `[B, C, H, W]` float32 [0, 1]
- เลือก perturb เฉพาะ key ที่มี "image" ใน key name
- ไม่แตะ state / language observations

### Usage

```bash
python scripts/eval_ood.py \
    --policy.path=<hub_id_or_local_path> \
    --env.type=libero --env.task=libero_spatial \
    --ood_level=level_1 \
    --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
```

### Output

- `eval_info.json` — ผลลัพธ์ evaluation + metadata (ood_level, ood_params)
- Videos — recorded rollouts (optional)

---

## Step 4.4 — Update causal_aug Exports

### ไฟล์ที่แก้ไข

```
causal_aug/causal_aug/__init__.py
```

เพิ่ม export: `OODPerturbation`, `OOD_LEVELS`

---

## Step 4.5 — Verification

### Individual Perturbation Tests

| Function | Test | Result |
|---|---|---|
| `_brightness` | factor=0.5 darker, factor=1.5 brighter | **PASS** |
| `_contrast` | factor=0.5 → values closer to mean | **PASS** |
| `_saturation` | factor=0 → grayscale (all channels equal) | **PASS** |
| `_hue_shift` | shift=0 → no change, shift=0.1 → changed | **PASS** |
| `_shadow` | left → left side darker, top → top darker | **PASS** |
| `_gaussian_blur` | blurred image smoother (orig=0.3326 > blur=0.0472) | **PASS** |
| `_rotation` | 0° → no change, 30° → changed | **PASS** |
| `_perspective` | magnitude=0.1 → shape preserved | **PASS** |
| `_affine` | identity → no change, translate+shear → changed | **PASS** |
| `_cutout` | ratio=0.3 → zero pixels present | **PASS** |

### OODPerturbation Level Tests

```
Level 0: PASS (identity — torch.equal)
Level 1: PASS (stochastic, clamped to [0,1])
Level 2: PASS (more extreme — avg diff 0.42 vs Level 1 avg diff 0.20)
Level 2 (360x360): PASS (real eval resolution)
Invalid level: PASS (ValueError raised)
```

### OODProcessorStep Tests

```
OODProcessorStep(level_1): PASS
  - Images perturbed: YES
  - State unchanged: YES
  - Shape preserved: YES
```

### eval_ood.py CLI

```bash
python scripts/eval_ood.py --help
# --ood_level flag present
# --policy.type supports causal_vla
# PASS
```

---

## สรุป Phase 4

| Step | Status | หมายเหตุสำคัญ |
|------|--------|--------------|
| 4.1 Augmented Dataset Script | **PASS** | Per-episode consistent augmentation, ใช้ perturbation เดียวกับ OOD eval |
| 4.2 OOD Perturbation Module | **PASS** | 11 perturbation types, 3 levels, pure PyTorch |
| 4.3 eval_ood.py Script | **PASS** | Inject via env_preprocessor, reuse lerobot eval pipeline |
| 4.4 Package Exports | **PASS** | OODPerturbation, OOD_LEVELS exported |
| 4.5 Verification | **PASS** | ทุก function ทดสอบแยก + ทุก level + dataset augmentation + ProcessorStep + CLI |

### ไฟล์ที่สร้าง/แก้ไข

| Action | ไฟล์ |
|--------|------|
| CREATE | `scripts/augment_dataset.py` — สร้าง augmented dataset |
| CREATE | `scripts/generate_preview_assets.py` — สร้าง MP4 + JSON สำหรับ preview |
| CREATE | `scripts/eval_ood.py` — OOD evaluation wrapper |
| CREATE | `augment_preview/index.html` — visualization website (video + graph) |
| CREATE | `augment_preview/aug_params.json` — per-episode params + level |
| CREATE | `augment_preview/episode_data.json` — state/action timeseries |
| CREATE | `augment_preview/ep*.mp4` — 40 MP4 videos |
| CREATE | `index.html` — home page (Document + Libero Dataset) |
| CREATE | `causal_aug/causal_aug/ood_wrapper.py` — OOD perturbation engine |
| CREATE | `lerobot/datasets/libero_spatial_augmented/` — augmented dataset (10 ep) |
| MODIFY | `causal_aug/causal_aug/__init__.py` — เพิ่ม exports |
| MODIFY | `pipeline.html` — เพิ่มปุ่ม Augment Preview |

### Perturbation Coverage เทียบกับ Lerobot-Dataset-Manager

| LDM Function | Perturbation | OOD Wrapper |
|---|---|---|
| `_augp_camera` → perspective | Perspective warp | `_perspective()` |
| `_augp_camera` → affine | Translate + shear + scale | `_affine()` |
| `_augp_camera` → rotation | Rotation | `_rotation()` |
| `_augp_light` → brightness | Brightness | `_brightness()` |
| `_augp_light` → contrast | Contrast | `_contrast()` |
| `_augp_light` → saturation | Saturation | `_saturation()` |
| `_augp_light` → color_jitter | Hue shift | `_hue_shift()` |
| `_augp_light` → shadow | Directional shadow | `_shadow()` |
| `_augp_light` → noise | Gaussian noise | `torch.randn_like()` |
| `_augp_light` → blur | Gaussian blur | `_gaussian_blur()` |
| (เพิ่มใหม่) | Cutout | `_cutout()` |
| `_augp_robot` | Robot noise | N/A (ไม่ใช่ visual perturbation) |

### Insight

- **ครบทุก visual perturbation จาก LDM** — Camera 3 ประเภท + Light 7 ประเภท + Cutout 1 ประเภท = **11 perturbation types**
- **Inject ไม่ Fork** — เพิ่มแค่ ProcessorStep เข้า pipeline ที่มีอยู่ → maintenance ง่าย
- **Geometric ก่อน Photometric** — spatial transform ก่อนค่อยเปลี่ยนสี/แสง ลดการ interpolate ซ้ำ
- **Level 0 = No-op** — ใช้ script เดียวกัน eval ทั้ง clean และ perturbed conditions
- **Robot noise (`_augp_robot`)** ไม่รวมใน OOD wrapper เพราะเป็น perturbation ระดับ action/state ไม่ใช่ visual — ใช้ทดสอบ robustness ด้านอื่น ไม่เกี่ยวกับ visual distribution shift ที่ CausalVLA มุ่งแก้

### สิ่งที่ใช้ใน Phase 6 (Evaluation)

```bash
# 5 models × 3 OOD levels = 15 eval runs ต่อ LIBERO suite
for MODEL in a b c d e; do
  for LEVEL in level_0 level_1 level_2; do
    python scripts/eval_ood.py \
        --policy.path=<username>/causalvla-model-${MODEL} \
        --env.type=libero --env.task=libero_spatial \
        --ood_level=${LEVEL} \
        --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
  done
done
```
