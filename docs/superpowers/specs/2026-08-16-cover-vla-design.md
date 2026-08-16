# COVER-VLA Design

> Phase 10 design approved on 2026-08-16

## 1. Objective

COVER-VLA tests whether coverage-constrained group-robust training can improve
the strongest current baseline, Model F (Online DR), without repeating the
coverage collapse observed in RAPID-Lite and PACER-Lite. The method must use one
VLA forward per sample, preserve normal SmolVLA inference, and make its training
adaptation auditable by augmentation group.

The paper hypothesis is:

> Fixed broad online randomization is strong, but it under-allocates learning
> pressure to persistently weak nuisance groups. Loss-adaptive group weighting
> can improve worst-group robustness when every group retains a fixed exposure
> floor and clean competence is explicitly protected.

## 2. Evidence and Scope

Under the three-seed, 10 episodes/task protocol, Model F obtains
`65.3/58.7/49.3%` on Clean/Mild/Extreme (57.8% overall). CausalVLA-v2 obtains
54.0% overall with two forwards. RAPID-Lite, RAPID-Mix, and Residual RAPID show
that static risk targeting either loses broad coverage or increases variance.
PACER-Lite obtains `67/57/41%` at seed 1000: its clean controller works, but
action disagreement is not a reliable robustness objective.

Phase 10 therefore excludes latent consistency, action consistency, policy
disagreement rewards, static sensitivity rankings, and multi-forward candidate
search. It changes only training-time augmentation assignment and supervised
task-loss aggregation.

## 3. Experiment Sequence

Phase 10 contains two 5,000-step pilots followed by at most one 25,000-step
training run:

1. **L1 — COVER-Base:** coverage-constrained group DRO.
2. **L2 — COVER-Safe:** L1 plus a clean-retention controller.
3. Evaluate both pilots at seed 1000 with 5 episodes/task in Clean, Mild, and
   Extreme.
4. Select one variant before any 25K run.
5. Train the selected variant for 25K steps from the same SmolVLA initialization
   and recipe as Model F.
6. Evaluate seed 1000 with 10 episodes/task. Expand to seeds 2000 and 3000 only
   if the preregistered gate passes.

The pilots are screening experiments, not headline paper results.

## 4. Augmentation Groups and Coverage

Each training sample is assigned exactly one group before its single model
forward:

1. `clean`
2. `brightness`
3. `color`
4. `noise`
5. `blur`
6. `shadow`
7. `geometry`
8. `composed`

Clean exposure is fixed at 0.50, matching Model F's clean probability. The
remaining 0.50 is allocated among the seven augmented groups. `composed`
retains the complete Model F broad transform and receives an augmented-branch
floor of 0.30. Each of the other six groups receives an augmented-branch floor
of 0.05. These floors consume 0.60 of augmented probability; the remaining
0.40 is adaptive. Consequently, total-sample probabilities are never below
0.15 for composed and 0.025 for each atomic group.

All atomic transforms use the same ranges and intensity as Model F. Camera
views share sampled scene-level nuisance parameters. Augmentation is detached
and applied only during training. Group assignment uses a categorical draw per
sample; it is not based on the current sample's loss, so no extra forward is
required.

## 5. Group Statistics and Robust Objective

The policy requests unreduced SmolVLA task losses and reduces them to one scalar
per sample after action-padding masking. For group `g`, it computes the current
batch mean only when that group is present and updates a registered-buffer EMA:

`loss_ema[g] <- beta * loss_ema[g] + (1 - beta) * stopgrad(batch_loss[g])`

Defaults:

- EMA decay `beta = 0.95`
- warm-up `1,000` optimizer steps
- robust temperature `0.5`
- adaptive-mass update interval `100` steps

After warm-up, a softmax over normalized augmented-group EMA losses allocates
the adaptive 0.40 augmented mass. EMA losses are divided by their detached
augmented-group mean before the softmax, preventing raw flow-loss scale from
changing the effective temperature.

The training objective is the mean supervised task loss after group weights are
applied. Group weights are derived from desired group mass divided by observed
batch frequency, clipped to `[0.5, 2.0]`, detached, and renormalized to mean one.
This importance correction prevents small stochastic batches from silently
changing the target objective. If a group is absent, it contributes neither a
loss nor an EMA update in that step.

No auxiliary latent or action loss is used.

## 6. L1 — COVER-Base

L1 uses the coverage and robust weighting rules above. Clean probability stays
exactly 0.50 and clean samples retain unit weight before final normalization.
Only the augmented adaptive mass changes. This isolates whether
coverage-constrained group DRO improves Model F.

## 7. L2 — COVER-Safe

L2 adds a clean-retention controller. It tracks fast and slow EMAs of the clean
task loss from clean samples:

- fast decay `0.90`
- slow decay `0.99`
- tolerance `0.05`
- minimum robust strength `0.25`
- recovery per update `0.01`

Robust strength `alpha` starts at 1.0. When
`clean_fast > clean_slow * (1 + tolerance)`, `alpha` is multiplied by 0.90.
Otherwise it recovers toward 1.0 by 0.01. `alpha` interpolates only the adaptive
augmented mass between the uniform residual allocation and Group-DRO
allocation; it never reduces clean exposure or any group floor. This differs
from PACER's controller, which changed paired augmented-loss weight: COVER-Safe
always trains one view per sample and always retains the full coverage budget.

## 8. State, Checkpointing, and Metrics

The following are registered buffers and must survive checkpoint save/load:

- per-group loss EMA
- per-group EMA initialization flags/counts
- per-group selection counts
- optimizer-step counter used for warm-up/update cadence
- L2 fast/slow clean EMAs and robust strength

Training logs must expose:

- `cover/group/<name>_fraction`
- `cover/group/<name>_loss`
- `cover/group/<name>_ema`
- `cover/group/<name>_target_mass`
- `cover/weight_min`, `cover/weight_max`
- `cover/robust_strength`
- `cover/clean_fast_ema`, `cover/clean_slow_ema`
- `loss_task`

NaN/non-finite group statistics do not update controller state. If all adaptive
scores are invalid, allocation falls back to the floor plus uniform residual
mass and logs `cover/fallback=1`.

## 9. Policy Interface and Compute Contract

Two policy types are exposed for the pilots: `cover_base` and `cover_safe`.
Both subclass SmolVLA configuration/policy patterns already used by
`online_dr`. They override only the training forward path.

Hard requirements:

- exactly one call to the underlying VLA training forward per batch
- no inference override and no inference augmentation
- same dataset, initialization, optimizer, scheduler, 25K steps, seed, and
  effective batch size as Model F for the selected full run
- no pretrained-revision drift during evaluation

## 10. Pilot Selection Rule

Each 5K pilot is evaluated with 5 episodes/task at seed 1000. Selection uses
this order:

1. Reject a variant if Clean is below 60% or any logged group has total exposure
   below its configured floor minus 1 percentage point.
2. Among survivors, select the highest three-mode mean.
3. If means differ by less than 2 percentage points, select L1 because it is
   simpler and provides the cleaner ablation.
4. If neither survives, stop Phase 10 full training and report both pilots as
   negative results; do not tune on the 10-episode test protocol.

The 5-episode pilot result is used only for selection and is not combined with
the final evaluation.

## 11. Full-Run Gate

The selected 25K model is first evaluated at seed 1000 with 10 episodes/task.
It advances only if all conditions hold:

- Clean `>= 64%`
- Mild `>= 62%`
- Extreme `>= 50%`
- three-mode mean `>= 59%`
- one training forward per sample
- no group violates its exposure floor by more than 1 percentage point over
  the full training log

If the gate passes, evaluate seeds 2000 and 3000 with the same 10 episodes/task
protocol. It becomes the paper candidate only if its three-seed overall mean is
strictly above Model F's 57.8%, no mode mean is more than 2 points below Model
F, and training-seed replication is scheduled before a final Q1 claim.

## 12. Tests and Verification

Unit tests must verify probability conservation, all coverage floors, softmax
monotonicity, clipped/normalized importance weights, warm-up behavior, clean
controller decay/recovery, absent-group handling, non-finite fallback, and
checkpoint state round-trip. Policy contract tests must count exactly one
underlying forward and verify inference equivalence with SmolVLA.

Run Python unit tests and a two-step Mac MPS smoke before CUDA smoke. CUDA smoke
uses two steps and validates group metrics, finite gradients, serialized config,
registered buffers, and checkpoint weights. Full training is authorized only
after both smoke tests pass.

## 13. Interpretation

A positive result supports the claim that loss-aware allocation needs explicit
coverage constraints to outperform broad randomization. If L1 wins, the clean
controller is unnecessary. If L2 wins, clean retention is a useful constraint
when it modulates adaptation without reducing exposure. If neither wins, the
evidence favors Model F and indicates that augmentation selection is not the
main bottleneck under the present data and evaluation regime.
