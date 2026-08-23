# Fixed-Per-Episode M-Model Evaluation Design

## Goal

Add a second fair-v1 evaluation track in which each episode receives one deterministic OOD augmentation record. Every frame and every camera view in that episode reuses the same record. Preserve the existing frame-randomized results and expose the new results under a separate `M-Models (Fix)` Results subtab.

The first execution covers evaluation seed 4000 only and every trained model that already has a complete fair-v1 result: `M0-clean`, `M2-online-dr`, and `M3-v2-warm`. `M1-offline-dr` is skipped until a trained checkpoint and complete evaluation workflow are available.

## Evaluation Contract

For each `(evaluation seed, task ID, episode index, OOD level)` tuple, derive one deterministic augmentation record. The record contains all sampled geometric and photometric parameters, including pixel-noise seed. The same record is applied to:

- every observation frame in the episode;
- every image/camera key in the observation;
- every environment step, including the initial reset observation.

Different episodes receive different records. Repeating the same tuple produces the same record and pixels. Level 0 remains an identity transform but still records the fixed-episode provenance.

The record derivation must not depend on call order, vectorized environment scheduling, device, or the global PyTorch RNG. It must use an explicit stable hash of the tuple plus an algorithm version. Camera views share sampled transform parameters, while pixel-space operations are applied independently to each camera tensor using the same recorded noise seed and shape-aware generation.

## Architecture

### Fixed augmentation records

Extend the OOD augmentation package with a serializable record type and two focused operations:

1. derive a record from the evaluation tuple and level configuration;
2. apply that record without sampling new values.

The existing frame-randomized `OODPerturbation` behavior remains unchanged for backward compatibility. Fixed evaluation uses an explicit mode rather than silently changing existing runs.

### Episode lifecycle

The evaluator must explicitly notify the OOD processor when a rollout batch starts. It provides the task ID and the actual episode indices/seeds assigned to each vector environment. The processor creates one record per vector slot and retains it until that episode ends. A new rollout replaces the retained records before processing its reset observations.

The evaluator must fail closed if fixed mode is enabled but episode identity is unavailable, batch sizes disagree, or a frame arrives before records are initialized. It must never fall back to frame-randomized augmentation.

### Workflow and outputs

Add a fixed evaluation mode to the fair-v1 workflow. Results are written under:

```text
outputs/eval/fair-v1-fixed/full/<model>/<level>/seed4000/
```

Existing results under `outputs/eval/fair-v1/full/` are never overwritten. Each `eval_info.json` includes:

- `augmentation_scope: "episode"`;
- fixed augmentation algorithm and schema versions;
- evaluation seed and record-derivation fields;
- model revision and protocol digest;
- the existing per-task metrics and clean/policy-view videos.

The first matrix contains nine runs: three models (`M0`, `M2`, `M3`) by three levels (`0`, `1`, `2`), with ten tasks and ten episodes per task. Only seed 4000 runs in this iteration. The workflow remains capable of adding seeds 5000 and 6000 later without changing the contract.

## Results Dashboard

Extend the results manifest with a separate fixed-result collection sourced only from `outputs/eval/fair-v1-fixed/full/`. The dashboard adds a third subtab, `M-Models (Fix)`, alongside `All Models` and `M-Models`.

`M-Models (Fix)` shows only fixed-evaluation models, summary rates, level rates, filters, provenance, and rollout videos. It labels the evaluation scope as fixed per episode and seed 4000. Results from the original and fixed tracks must never be aggregated together.

If no fixed results exist, the subtab displays a dedicated empty state. Partially completed matrices display only completed runs and clearly show their run/episode counts.

## Validation and Tests

Automated tests must prove:

1. repeated frames in one episode use identical sampled parameters;
2. all camera views in one episode share the record;
3. different episode identities produce different records;
4. the same tuple reproduces the same record and output across fresh objects;
5. vector slots retain independent records without call-order dependence;
6. fixed mode rejects missing or mismatched lifecycle data;
7. level 0 is pixel-identical;
8. existing frame-randomized evaluation behavior remains available;
9. workflow commands target only seed 4000 and the trained model set for this iteration;
10. outputs cannot overwrite the original fair-v1 tree;
11. manifest generation keeps original and fixed collections separate;
12. the dashboard filters `M-Models (Fix)` independently.

Before full evaluation, run unit and workflow tests plus one preflight episode for each selected model at level 0 and level 2. Validate provenance, video generation, deterministic replay, and expected episode counts before launching the nine full runs.

## Execution and Failure Handling

The runner validates the checkpoint revision before each run and validates the resulting artifact before proceeding. Existing valid fixed artifacts may be resumed; malformed, incomplete, wrong-scope, or wrong-seed artifacts stop the matrix and are not treated as complete.

Full evaluation is compute-intensive and may require the configured GPU environment. Implementation completion and evaluation completion are reported separately. If the local environment cannot access the trained checkpoints or required simulator/GPU, the code and preflight validation may complete locally, but the full matrix remains pending rather than fabricating results.

After successful runs, rebuild `results-data.json`, run dashboard tests, and visually verify the `M-Models (Fix)` subtab without browser console errors.
