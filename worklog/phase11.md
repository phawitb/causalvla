# Phase 11 — Matched-Budget Learning-Curve Audit

> Started: 2026-08-16  
> Status: MODEL F 5K EVALUATION RUNNING

## Question

Phase 10 rejected both COVER variants at 5K using a Clean >=60% survival gate,
but that threshold was informed by 25K models. Phase 11 tests whether
COVER-Base failed because its method is worse than Online DR or because 5K
optimization is insufficient for both methods.

## Fixed Protocol

- Model F checkpoint: optimizer step 5,000
- Model F Hub revision: `05a56ee5ec79d2879ab1d0cc877946074d151904`
- COVER checkpoints: step 5,000, training seed 1000
- Evaluation seed: 1000
- LIBERO Spatial: 10 tasks
- Budget: 5 episodes/task/mode
- Modes: Clean, Mild, Extreme
- Batch size: 2, synchronous environments

## Existing COVER Results

| Model | Clean | Mild | Extreme | Mean |
|---|---:|---:|---:|---:|
| COVER-Base 5K | 46% | 40% | 18% | 34.7% |
| COVER-Safe 5K | 44% | 36% | 12% | 30.7% |

## Preregistered Diagnostic Rule

Resume COVER-Base beyond 5K only if its three-mode mean is no more than 2
percentage points below Model F 5K and no individual mode is more than 5 points
below Model F 5K. If COVER-Base exceeds the F-5K mean, it becomes the sole
Phase-11 candidate. Otherwise stop COVER permanently. COVER-Safe is not resumed
under any outcome because it lost to COVER-Base in every pilot mode.

Any continuation is a new learning-curve experiment and does not override the
recorded Phase-10 NO-GO decision.
