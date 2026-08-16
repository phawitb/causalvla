# PACER-VLA Phase 9 Design

## Objective

Phase 9 develops PACER-VLA (Policy-Adaptive Counterfactual Exposure with
clean-Risk constraints), a two-stage research program derived from the Phase
6--8 results. Model J, PACER-Lite, is implemented and evaluated first. Model K,
PACER-Full, is permitted only if Model J passes its preregistered seed-1000
gate.

The primary claim is not that counterfactual training is automatically better
than domain randomization. The evidence shows the opposite: simple Online DR
(Model F) is the strongest and most stable current baseline. Phase 9 tests the
more specific hypothesis that policy-conditioned, sample-level intervention
selection can improve on Online DR when paired with an explicit clean-retention
constraint.

## Evidence Behind the Design

All modern results use LIBERO Spatial, 10 episodes per task, 10 tasks, and eval
start seeds 1000, 2000, and 3000.

| Model | Clean | Mild | Extreme | Three-mode mean |
|---|---:|---:|---:|---:|
| Model F | 65.3 +/- 2.3 | 58.7 +/- 3.5 | 49.3 +/- 1.5 | 57.8 |
| CausalVLA-v2 | 60.0 +/- 3.0 | 57.7 +/- 2.1 | 44.3 +/- 1.2 | 54.0 |
| RAPID-Lite | 61.3 +/- 4.0 | 41.3 +/- 6.8 | 12.3 +/- 4.2 | 38.3 |
| RAPID-Mix | 57.3 +/- 3.1 | 53.0 +/- 1.0 | 46.3 +/- 4.2 | 52.2 |
| Residual RAPID | 62.3 +/- 10.2 | 51.0 +/- 1.0 | 44.3 +/- 12.0 | 52.6 |

These results support four design constraints:

1. Preserve online broad augmentation because Model F is the strongest model.
2. Preserve clean--augmented pairing because V2 is stable across eval seeds.
3. Do not use a static global risk ranking; RAPID-Lite loses distributional
   coverage and Residual RAPID is highly seed-sensitive.
4. Do not optimize maximum disagreement. Excessive intervention hardness is a
   plausible cause of the Mild regression and high variance in Phase 8.

## Alternatives Considered

### A. Two-forward contextual bandit (selected)

Run one clean and one augmented branch with shared flow noise and time. The
clean branch supplies sample context; a bandit selects one intervention for the
augmented branch; the resulting paired disagreement updates online arm
statistics. This retains V2-like compute, permits a practical RTX 4090 batch,
and provides a direct ablation against V2 and Model F.

### B. Exhaustive candidate search

Run clean plus three or more augmented candidates and train on the selected
candidate. This gives immediate within-sample ranking but costs 3--5 forwards,
encourages hardest-example bias, and makes a 25,000-step experiment more
expensive. It is reserved as an oracle ablation after Model J succeeds.

### C. Static curriculum or fixed mixture

Use a precomputed risk profile or fixed branch probabilities. Phase 8 already
tests this family through RAPID-Lite, RAPID-Mix, and Residual RAPID. It is not a
new Phase 9 model.

## Model J: PACER-Lite

### Architecture

PACER-Lite subclasses SmolVLA and changes training only. Inference calls the
unchanged SmolVLA action-generation path and has no additional forward,
augmentation, selector, or latency.

Training performs exactly two model forwards per batch:

1. Prepare the clean images, state, language, and action target.
2. Sample flow noise and time once.
3. Run the clean branch with `forward_with_latent`.
4. Derive a detached per-sample context from clean task loss. Samples are split
   into `easy`, `medium`, and `hard` buckets by within-batch rank, avoiding
   dataset-specific absolute thresholds.
5. The contextual bandit samples one intervention arm per sample.
6. Apply that intervention coherently to every camera view of the sample.
7. Run the augmented branch with the same flow noise and time.
8. Combine supervised clean and augmented task losses through the clean-safety
   controller.
9. Update detached bandit statistics from the observed paired reward.

The intervention arms are `brightness`, `color`, `noise`, `blur`, `shadow`,
`geometry`, and `composed`. All arms use the existing label-preserving
`InterventionBank`; the exploration floor guarantees continuing coverage of
every family.

### Productive-difficulty reward

For sample `i`, action disagreement is the mean squared difference between the
clean and augmented flow-velocity predictions over real action dimensions.
The bandit must seek useful rather than maximum difficulty. Its detached reward
is:

```text
ratio_i = augmented_task_loss_i / max(clean_task_loss_i, epsilon)
reward_i = clipped_disagreement_i * exp(-overhard_penalty * relu(ratio_i - max_loss_ratio))
```

Disagreement is clipped before the penalty. An intervention that changes the
policy while remaining learnable receives positive reward; an intervention
whose supervised loss is disproportionately larger than the clean loss is
down-weighted. No reward gradient flows into SmolVLA.

For each context and arm, PACER maintains registered buffers containing an EMA
reward and observation count. Selection probabilities are a temperature-scaled
softmax over normalized EMA reward mixed with a uniform exploration floor.
During the warm-up window, selection is uniform so all context--arm cells
receive observations before exploitation.

### Clean-safety controller

The base objective starts from equal paired supervision:

```text
L = (1 - w_aug) * L_clean + w_aug * L_aug
```

`w_aug` is bounded by `min_augmented_weight <= w_aug <= 0.5`, so the clean
branch always has at least half of the task-loss weight. A fast and a slow EMA
track detached clean task loss. When the fast EMA exceeds the slow EMA by more
than `clean_tolerance`, the controller multiplicatively reduces `w_aug`.
Otherwise it gradually restores `w_aug` toward 0.5. Both EMAs and the current
weight are checkpointed registered buffers.

The controller is inactive during warm-up and never changes inference.

### Default configuration

| Parameter | Default | Purpose |
|---|---:|---|
| `aug_intensity` | 1.0 | Match broad Online DR strength |
| `bandit_temperature` | 1.0 | Avoid early selector collapse |
| `exploration_floor` | 0.20 | Preserve all-family coverage |
| `bandit_ema_decay` | 0.95 | Track non-stationary policy risk |
| `bandit_warmup_steps` | 1,000 | Uniform initial exploration |
| `max_loss_ratio` | 2.0 | Reject destructive hardness |
| `overhard_penalty` | 2.0 | Strength of learnability gate |
| `max_augmented_weight` | 0.50 | Equal paired supervision ceiling |
| `min_augmented_weight` | 0.10 | Retain counterfactual exposure |
| `clean_tolerance` | 0.05 | Allowed fast/slow clean-loss gap |
| `clean_weight_decay` | 0.90 | Safety response when clean degrades |
| `clean_weight_recovery` | 0.01 | Slow restoration toward 0.50 |

Defaults are preregistered. They may be changed only after a documented smoke
failure, not after viewing full evaluation outcomes.

### Required metrics

Training logs must expose:

- `loss`, `loss_task_clean`, and `loss_task_augmented`
- `pacer/augmented_weight`
- `pacer/action_disagreement`
- `pacer/loss_ratio`
- `pacer/context_easy`, `pacer/context_medium`, `pacer/context_hard`
- selection fraction and EMA reward for every intervention arm
- clean fast EMA, clean slow EMA, and safety-trigger fraction

The final checkpoint must serialize every configuration value and all adaptive
buffers required to reproduce training state.

## Model J Tests and Failure Handling

Unit tests must cover configuration validation, context assignment, probability
normalization, exploration-floor bounds, warm-up uniformity, per-context bandit
updates, coherent multi-camera intervention application, productive-difficulty
penalty, safety-controller decay and recovery, buffer checkpoint round-trip,
two-forward source contract, shared noise/time, and unchanged inference.

Non-finite clean loss, augmented loss, disagreement, reward, or EMA input must
not update adaptive state. The affected reward is treated as zero and a metric
counts rejected updates. Empty camera lists, inconsistent camera batch sizes,
unknown arms, and invalid probabilities raise explicit `ValueError`s.

## Preregistered Evaluation Gates

### Local and CUDA gates

1. All existing and Phase 9 unit tests pass on Mac.
2. A two-step MPS smoke test produces finite loss and exactly two forwards.
3. A two-step CUDA smoke test passes on the GPU server.
4. A 25,000-step training run completes without traceback, NaN, or OOM and its
   empirical arm coverage respects the exploration floor.

### Seed-1000 go/no-go gate

Evaluate 10 episodes per task on Clean, Mild, and Extreme using the same
protocol and batch size as Model F. Expand to seeds 2000 and 3000 only if all
conditions hold:

- three-mode mean is at least `58.3%` (Model F seed-1000 mean),
- Clean is at least `61%`,
- Mild is at least `59%`, and
- Extreme is at least `46%`.

This gate allows at most a 3-point regression in an individual mode while
requiring no regression in the overall mean.

### Three-eval-seed success gate

Model J becomes a finalist only if its three-seed mean exceeds Model F's 57.8%
and no mode is more than 2 points below Model F. Report mean, sample standard
deviation, pooled successes, per-task results, and paired bootstrap intervals.

Evaluation seeds measure rollout variance, not training variance. A paper claim
requires at least three independent training seeds for Model F, V2, and the
final PACER model. Ten episodes per task are the screening protocol; finalists
must use 50 episodes per task for the final table.

## Model K: PACER-Full Admission Rule

Model K is not implemented until Model J passes the seed-1000 gate. It adds one
research component at a time:

1. asymmetric clean-teacher action consistency,
2. worst-family CVaR/group-DRO weighting, and
3. an optional multi-candidate oracle used as an ablation rather than the
   default training path.

Each component receives an independent ablation. Model K must retain the same
clean-safety controller and unchanged SmolVLA inference.

## Paper Scope

The intended paper story is that static global risk selection fails across
seeds, while closed-loop sample-conditioned intervention exposure can target
current policy vulnerabilities without sacrificing clean competence. Model F
is the strong baseline, not a weak comparator.

The final study must include fair A/B reruns, unseen and composed visual shifts,
camera and initial-state shifts, language paraphrases, counterfactual
instructions, a second VLA architecture, compute and latency accounting, and
preferably real-robot evaluation. Relevant adjacent work includes RoCoDA,
RoVLA, STRONG-VLA, CofactVLA, LIBERO-Plus, LIBERO-PRO, and LIBERO-CF; therefore
generic paired consistency alone is not a sufficient novelty claim.

## Out of Scope for Model J

- inference-time guidance or multiple inference forwards
- learned neural augmentation generator
- exhaustive candidate evaluation during default training
- language or proprioception interventions
- architecture changes inside SmolVLA
- tuning thresholds after inspecting full seed-1000 evaluation

