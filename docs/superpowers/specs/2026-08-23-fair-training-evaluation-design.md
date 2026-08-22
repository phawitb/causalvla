# Fair Training and Evaluation Protocol v1

Date: 2026-08-23

## Objective

Create a reproducible comparison of four SmolVLA training treatments while holding optimizer updates and source-sample exposure constant. The first phase validates all four workflows locally, trains LIBERO-Spatial models on a GPU server through Hugging Face, and evaluates one paired seed on the Mac. Later phases extend the same protocol to LIBERO-Object, LIBERO-Goal, and evaluation seeds 5000 and 6000.

The primary fairness definition is fixed optimizer steps plus fixed source-sample exposure. It is not a fixed-compute comparison. M3 processes paired clean and augmented views and therefore uses more image forwards and compute; the experiment must report that overhead.

## Experiment Matrix

| ID | Treatment | Dataset input | Per-batch treatment | Policy behavior |
|---|---|---|---|---|
| `M0-clean` | Clean baseline | Original LIBERO-Spatial | 16 clean | Standard SmolVLA |
| `M1-offline-dr` | Offline domain randomization | Original plus deterministic materialized augmentation | 8 clean + 8 augmented | Standard SmolVLA |
| `M2-online-dr` | Online domain randomization | Original LIBERO-Spatial | Exactly 8 clean + 8 augmented | One standard policy forward per sample |
| `M3-v2-warm` | Paired consistency training | Original LIBERO-Spatial | 16 source samples expanded to 16 clean + 16 augmented views | Weighted task losses plus warmed action-consistency loss |

The model IDs must not use `L0` through `L3`, because `level_0` through `level_2` already identify evaluation-time OOD conditions.

## Locked Training Contract

All full training runs use:

- the same pinned SmolVLA base-model commit;
- the same pinned `lerobot/libero_spatial_image` source-dataset commit;
- 25,000 optimizer steps;
- batch size 16, interpreted as 16 source samples before M3 view expansion;
- training seed 1000;
- identical optimizer, learning-rate scheduler, preprocessing, normalization, tokenizer, action chunking, shuffle, and worker configuration;
- checkpoints at steps 5,000, 10,000, 15,000, 20,000, and 25,000;
- the step-25,000 checkpoint as the primary evaluation checkpoint.

Expected view exposure is:

| Model | Clean views | Augmented views | Total image views |
|---|---:|---:|---:|
| M0 | 400,000 | 0 | 400,000 |
| M1 | 200,000 | 200,000 | 400,000 |
| M2 | 200,000 | 200,000 | 400,000 |
| M3 | 400,000 | 400,000 | 800,000 |

M3 uses clean-task weight 0.5, augmented-task weight 0.5, one counterfactual view, no latent or smoothness loss, and an action-consistency weight that warms linearly from 0 to 0.05 during the first 10,000 steps.

The primary comparison holds optimizer steps and source samples constant. Every run records GPU-hours, wall-clock duration, samples per second, and peak device memory so M3's additional compute remains explicit. A future optional `M3-v2-warm-12.5k` ablation may compare approximately equal image-view counts, but it is outside the primary v1 matrix.

## Matched Offline and Online Augmentation

M1 and M2 must differ only in when augmentation is applied.

A shared augmentation manifest defines the augmentation families, parameter distributions, intensities, implementation version, and deterministic parameter derivation. Parameters derive from training seed, source episode ID, frame index, and exposure index. Augmentation never changes language instructions or action labels.

M1 materializes exactly one augmented counterpart for every clean source item and trains through a balanced sampler that emits exactly eight clean and eight augmented samples per batch. The materialized dataset is published as `libero-spatial-offline-dr-fair-v1` with its source revision and augmentation-manifest hash.

M2 reads the original dataset and applies the same parameter record online. Its sampler selects exactly eight augmented positions in every 16-sample batch rather than relying on an unconstrained Bernoulli draw. A contract test compares M1 materialized images with M2 online output from the same parameter record using a documented numeric tolerance.

## Architecture

The implementation uses one versioned experiment manifest as the source of truth and thin runners for training, smoke testing, and evaluation.

The manifest contains:

- model ID and display name;
- policy type and model-specific parameters;
- source dataset and pinned revision;
- offline dataset and pinned revision where applicable;
- pinned base-model revision;
- training steps, batch size, seed, checkpoint cadence, optimizer, and scheduler;
- augmentation-manifest path and hash;
- Hugging Face repository name;
- evaluation suites, OOD levels, seeds, and episode counts.

The training runner accepts a model ID and supports dry-run, smoke, full training, and safe resume. It resolves the manifest into the LeRobot command and writes a run manifest before starting. The evaluation runner resolves a pinned Hugging Face model commit into an evaluation command and validates output completeness.

The runners must reject configuration drift instead of silently overriding the contract. Model-specific code remains isolated behind the existing policy interfaces: standard SmolVLA for M0/M1, online DR for M2, and V2-Warm for M3.

## Local Smoke Test

The Mac smoke workflow runs one real optimizer step for every model using MPS with CPU fallback enabled. It uses a deterministic tiny subset, batch size 2, one step, local outputs, no Hugging Face push, and minimal workers. These are the only permitted differences from the full training contract.

Smoke outputs live under `outputs/smoke/fair-v1/` and never qualify as experimental checkpoints.

Each model passes only if the workflow:

1. loads the pinned base model and source data;
2. produces the expected batch keys and tensor shapes;
3. enforces the treatment's clean/augmented ratio;
4. produces a finite forward loss;
5. completes backward and an optimizer update;
6. verifies that at least one trainable parameter changed;
7. saves a checkpoint;
8. reloads the checkpoint and performs inference;
9. satisfies offline/online augmentation equivalence for M1/M2;
10. verifies paired views and a zero initial consistency weight for M3.

## Hugging Face Handoff

Full training publishes to separate repositories:

- `causalvla-fair-v1-m0-clean-spatial`;
- `causalvla-fair-v1-m1-offline-dr-spatial`;
- `causalvla-fair-v1-m2-online-dr-spatial`;
- `causalvla-fair-v1-m3-v2-warm-spatial`.

Each repository includes checkpoints at the locked cadence, the resolved configuration, dataset and base-model revisions, source Git commit, augmentation manifest and hash, logs, runtime metrics, and a model card marking the run as training seed 1000. Visibility must be consistent across all four repositories.

The GPU-server interface is one model-selecting command. Resume is permitted only when the saved protocol hash equals the current resolved protocol hash. A downloaded model is evaluated only after resolving and pinning its Hugging Face commit SHA.

An optional HF round-trip smoke test may upload to a temporary private repository when explicitly enabled. Local smoke testing never uploads by default.

## Evaluation Protocol

The first evaluation phase uses:

- suite: LIBERO-Spatial;
- models: M0, M1, M2, and M3;
- primary checkpoint: step 25,000;
- evaluation seed: 4000;
- OOD conditions: `level_0`, `level_1`, and `level_2`;
- 10 episodes per task across 10 tasks;
- MPS device with CPU fallback;
- a pinned Hugging Face commit for every model;
- the same episode seeds for every model and condition.

A preflight evaluates one episode per task at `level_0`. Full pilot evaluation starts only after preflight output passes validation. The full phase runs 300 episodes per model and 1,200 episodes total.

Outputs include aggregate and per-task metrics, per-episode results, clean-view and policy-view videos, resolved model revision, OOD provenance, and runtime. Primary reporting includes overall and per-task success rates, absolute OOD success, degradation from level 0 to levels 1 and 2, paired differences against M0, and inference wall-clock time.

Seed 4000 is a pilot feasibility result and cannot support a claim of statistical superiority. The future confirmatory phase adds evaluation seeds 5000 and 6000, followed by LIBERO-Object and LIBERO-Goal under the same protocol.

## Provenance and Failure Handling

Every train, smoke, and evaluation run writes `run_manifest.json` containing the resolved configuration, protocol hash, source Git commit, dataset revision, model revision, augmentation hash, environment summary, start/end timestamps, runtime metrics, and status (`started`, `completed`, or `failed`). Failed runs retain an error reason.

The runner stops before doing work when:

- a dataset or base model is not pinned to an immutable commit;
- locked cross-model parameters differ;
- M1/M2 cannot enforce their batch ratio;
- M1 and M2 augmentation hashes differ;
- an existing output directory has a different protocol hash;
- a resume checkpoint belongs to another protocol;
- a loss is non-finite;
- checkpoint save/reload fails;
- an HF upload is incomplete or resolves to the wrong revision;
- evaluation produces an incomplete task or episode matrix.

Completed outputs are not overwritten. Safe resume and idempotent evaluation checks use the run manifest and expected artifact set.

## Verification Strategy

Verification has five layers:

1. Unit tests cover manifest parsing, invariant validation, exact balanced sampling, deterministic augmentation parameters, hashing, and model routing.
2. Contract tests inspect dry-run commands and prove that shared parameters are identical across M0–M3.
3. Augmentation tests compare M1 materialized pixels with M2 online output from identical parameter records.
4. Local integration tests run one optimizer step and checkpoint save/reload for all four models.
5. Evaluation validation checks task count, episode count, seed mapping, pinned model revision, OOD provenance, metrics, and video paths.

Implementation is complete only after all automated tests pass, all four Mac smoke runs meet their acceptance criteria, and dry-run output for the four full GPU commands matches the locked contract.

## Scope Boundaries

Version 1 includes the LIBERO-Spatial training matrix, local smoke tests, Hugging Face handoff, seed-4000 Mac evaluation, provenance, and validation tooling.

Version 1 does not include full GPU training performed by the local agent, seeds 5000/6000, LIBERO-Object, LIBERO-Goal, fixed-compute training, hyperparameter tuning from evaluation outcomes, or statistical superiority claims.
