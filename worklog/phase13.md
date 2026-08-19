# Phase 13: CausalVLA-v2 Warm Consistency

## Question

Can a weak, continuously warmed action-consistency objective improve CausalVLA-v2
without reproducing the early-training collapse observed with the stronger Phase-5
objective?

## Model K — V2-Warm

V2-Warm preserves the paired supervised objective and shared flow target from V2:

\[
L_{task}=0.5L_{clean}+0.5L_{augmented}
\]

It adds stop-gradient action consistency between the augmented and clean flow
predictions:

\[
L_{action}=\lVert v_{aug}-\operatorname{stopgrad}(v_{clean})\rVert_2^2
\]

The weight increases continuously rather than in discrete stages:

\[
\lambda_{action}(s)=0.05\min(s/10000,1)
\]

| Step | Action-consistency weight |
|---:|---:|
| 0 | 0.0000 |
| 1,000 | 0.0050 |
| 2,500 | 0.0125 |
| 5,000 | 0.0250 |
| 7,500 | 0.0375 |
| 10,000–25,000 | 0.0500 |

Latent consistency and smoothness remain disabled. The scheduler step is a
serialized model buffer, so checkpoint resume continues from the saved weight.
Training logs expose `lambda_action_current` and `loss_action`.

## Workflow

Install the policy and run a two-step CUDA smoke test:

```bash
python scripts/install_policy_patches.py causal_vla causal_vla_warm
V2_WARM_STEPS=2 V2_WARM_BATCH_SIZE=2 ./scripts/train_v2_warm.sh
```

After the smoke checkpoint confirms policy type, schedule state, finite loss and
two forwards per sample, run the fixed 25K configuration:

```bash
./scripts/train_v2_warm.sh
```

The full run uses LIBERO Spatial, batch size 16, seed 1000, one augmented view,
augmentation intensity 1.0 and Hub repository
`phawitbinabik/causalvla-v2-warm`.

## Status

- [x] Continuous linear scheduler implemented
- [x] Scheduler checkpoint/resume state tested
- [x] V2-Warm policy registered without changing V2 defaults
- [x] Reproducible smoke/full training launcher added
- [x] Local MPS smoke: 2/2 steps, batch 2, checkpoint finite, schedule step 2
- [x] CUDA smoke
- [x] Full 25K training
- [x] Seed-1000 Clean/Mild/Extreme evaluation
- [x] Three-seed Clean/Mild/Extreme evaluation

## Local MPS Smoke Result

The local smoke completed two optimizer steps with batch size 2 and ended with
`End of training`. The step-2 checkpoint serialized `type=causal_vla_warm`,
`lambda_action=0.05`, `action_warmup_steps=10000`, and
`consistency_schedule.step=2`. All 501 tensors were finite. Hub upload was
disabled, and the run saved optimizer, scheduler, RNG and training-step state.

## Full Training Result

The CUDA run completed 25,000 optimizer steps in `4:16:23` with 400,000
sampled examples. Final metrics were total loss `0.349`, clean task loss
`0.345`, augmented task loss `0.352`, action-consistency loss `0.006`, gradient
norm `1.626`, throughput about 27 samples/s and GPU memory 6.86 GB. The final
checkpoint serialized `consistency_schedule.step=25000`; all 501 tensors were
finite. No traceback, runtime error, CUDA OOM or NaN was found.

Pinned Hub checkpoint:

```text
Repo: phawitbinabik/causalvla-v2-warm
Revision: 119ee2e25cb1e190e89561287dad8c2ffc967d4f
```

## Seed-1000 Evaluation Result

The pinned revision was evaluated locally on MPS using LIBERO Spatial, 10
tasks, 10 episodes/task, synchronous environments and seed 1000.

| Model | Clean | Mild OOD | Extreme OOD | Three-mode mean |
|---|---:|---:|---:|---:|
| CausalVLA-v2 | 63% | 60% | 45% | 56.0% |
| Model F — Online DR | 64% | 62% | 49% | 58.3% |
| **V2-Warm** | **72%** | **67%** | **53%** | **64.0%** |

V2-Warm exceeded V2 by `+9`, `+7` and `+8` percentage points on Clean, Mild
and Extreme. It exceeded Model F by `+8`, `+5` and `+4` points. Evaluation
times were 756.7 s, 1095.0 s and 1392.3 s respectively. This is a positive
seed-1000 result; the evaluation-initial-state check is completed below.

## Three-Seed Evaluation Result

The same pinned checkpoint was evaluated with seeds 1000, 2000 and 3000. Each
cell contains 100 episodes (10 tasks, 10 episodes/task).

| Evaluation seed | Clean | Mild OOD | Extreme OOD | Three-mode mean |
|---:|---:|---:|---:|---:|
| 1000 | 72% | 67% | 53% | 64.0% |
| 2000 | 68% | 63% | 60% | 63.7% |
| 3000 | 59% | 63% | 53% | 58.3% |
| **Mean +/- sample SD** | **66.3 +/- 6.7%** | **64.3 +/- 2.3%** | **55.3 +/- 4.0%** | **62.0%** |

Comparison against the existing three-seed evaluations:

| Model | Clean | Mild OOD | Extreme OOD | Mean across all modes/seeds |
|---|---:|---:|---:|---:|
| CausalVLA-v2 | 60.0 +/- 3.0% | 57.7 +/- 2.1% | 44.3 +/- 1.2% | 54.0% |
| Model F — Online DR | 65.3 +/- 2.3% | 58.7 +/- 3.5% | 49.3 +/- 1.5% | 57.8% |
| **V2-Warm** | **66.3 +/- 6.7%** | **64.3 +/- 2.3%** | **55.3 +/- 4.0%** | **62.0%** |

V2-Warm exceeded Model F by `+1.0`, `+5.6` and `+6.0` percentage points on
the three per-level means, and exceeded CausalVLA-v2 by `+6.3`, `+6.6` and
`+11.0` points. Its Clean result varied more across evaluation seeds than the
baselines. These runs vary only the evaluation initial-state seed: V2-Warm was
trained once with seed 1000, so this is not a multi-training-seed estimate.
