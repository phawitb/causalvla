# Model I — Residual RAPID Design

## Objective

Build a single-forward SmolVLA training policy that preserves the complete
online domain-randomization coverage of Model F and uses policy-risk information
only as a residual visual intervention. The experiment tests whether risk
targeting adds value when it does not replace any of Model F's broad augmented
exposure.

## Evidence and Motivation

Under the primary LIBERO Spatial protocol (three evaluation seeds, ten
episodes/task), Model F achieved `65.3/58.7/49.3%` on clean/mild/extreme.
RAPID-Lite achieved `61.3/41.3/12.3%`, showing that action-sensitivity-only
selection destroys OOD coverage. RAPID-Mix restored part of that coverage and
achieved `57.3/53.0/46.3%`, but remained below Model F because its broad branch
covered only 25% of samples.

Residual RAPID therefore retains Model F's total broad exposure at 50% and
applies a risk intervention on top of a subset of those already augmented
samples.

## Training Distribution

Each sample belongs to exactly one observable logging branch:

| Branch | Batch probability | Image supplied to SmolVLA |
|---|---:|---|
| Clean | 0.500 | Original clean observation |
| Broad | 0.375 | Model F broad domain randomization |
| Residual | 0.125 | Model F broad DR followed by one risk overlay |

The implementation expresses this as:

- `augmentation_probability = 0.50`
- `risk_overlay_probability = 0.25`, conditional on augmentation
- total broad coverage = `broad + residual = 0.50`
- total risk-overlay exposure = `0.50 × 0.25 = 0.125`

The broad transformation distribution and intensity are identical to Model F.
Risk overlay candidates and robust-risk scores are pinned to profile revision
`phase8-3seed-256samples-robust-risk-v1`:

| Candidate | Strength | Robust risk | Effective sampling probability |
|---|---:|---:|---:|
| Shadow | 0.75 | 0.023396 | 0.3741 |
| Brightness | 0.50 | 0.021158 | 0.3415 |
| Geometry | 1.00 | 0.017234 | 0.2844 |

The risk sampler keeps `exploration_floor = 0.10` and
`risk_temperature = 1.0`.

## Architecture and Data Flow

The new LeRobot policy type is `residual_rapid` and subclasses
`SmolVLAPolicy`.

For each training batch:

1. Prepare and detach the camera tensors for augmentation.
2. Sample an augmented mask with probability 0.50.
3. Generate Model F broad views for the batch.
4. Sample an overlay mask only within augmented samples with conditional
   probability 0.25.
5. Apply one risk-weighted intervention to broad views selected by the overlay
   mask. Clean samples never receive an overlay.
6. Select clean, broad, or broad-plus-risk pixels with tensor masks.
7. Run one ordinary SmolVLA task-loss forward.

The policy adds no trainable modules, auxiliary loss, paired branch, latent
loss, action consistency loss, or second VLA forward. Inference remains the
standard SmolVLA path and has no augmentation or additional cost.

The compositor method is named `compose(images, broad_images)`; it deliberately
does not use `apply`, which is reserved by `torch.nn.Module` for recursive module
transforms.

## Configuration

`ResidualRapidConfig` exposes and serializes:

- `augmentation_probability: float = 0.50`
- `risk_overlay_probability: float = 0.25`
- `broad_intensity: float = 1.0`
- `risk_temperature: float = 1.0`
- `exploration_floor: float = 0.10`
- `profile_revision: str = "phase8-3seed-256samples-robust-risk-v1"`

Probabilities must be in `[0, 1]`, broad intensity must be non-negative, risk
temperature must be positive, and exploration floor must be in `[0, 1]`.

## Metrics

Every reduced training forward reports:

- `loss` and `loss_task`
- `branch/clean`
- `branch/broad`
- `branch/residual`
- `residual/shadow:0.75`
- `residual/brightness:0.5`
- `residual/geometry:1.0`

The three branch fractions must sum to one. The three residual-arm fractions
must sum to `branch/residual` up to floating-point error.

## Error Handling and Compatibility

- An empty camera-view list is rejected by the augmentation components.
- Invalid configuration values fail during config construction.
- Multi-camera views use the same branch and scene-level nuisance parameters
  for a given sample.
- `reduction="none"` preserves per-sample task losses for LeRobot compatibility.
- Pre/post-processing is identical to SmolVLA, Online DR, RAPID-Lite, and
  RAPID-Mix.
- The local policy installer registers `residual_rapid` without altering
  existing policy behavior.

## Verification Strategy

Development follows red-green-refactor:

1. Write tests for branch-mask invariants and expected probabilities; verify
   they fail before the branch sampler exists.
2. Implement the minimal branch sampler and verify tests pass.
3. Write failing config-validation and registration tests.
4. Implement config, policy, processor, and installer registration.
5. Run the complete `causal_aug/tests` suite and Python compilation checks.
6. Run a CPU image-mixing smoke test for shape, finite values, normalized range,
   branch totals, and residual-arm totals.
7. Run a two-step CUDA training smoke on the GPU server before full training.

## Experiment Protocol and Decision Gate

Full training initially uses only training seed 1000, 25,000 steps, batch size
16, and the same optimizer/scheduler/data recipe as Model F.

Primary evaluation uses LIBERO Spatial, 10 episodes/task, with clean, mild OOD,
and extreme OOD. The first gate uses evaluation seed 1000:

| Mode | GO threshold |
|---|---:|
| Clean | at least 64% |
| Mild OOD | at least 62% |
| Extreme OOD | at least 49% |

Only if the model meets all three thresholds will evaluation expand to seeds
2000 and 3000. If it fails, risk-based augmentation selection is stopped and
the next research direction becomes held-out-feedback adaptive intensity over
the complete Model F distribution.

## Scope Boundaries

This implementation does not include adaptive online weights, held-out
validation rollouts during training, task rebalancing, new demonstrations,
architecture changes, additional losses, or multiple training seeds. Those are
separate experiments contingent on this gate.
