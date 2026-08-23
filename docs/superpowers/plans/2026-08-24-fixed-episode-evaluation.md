# Fixed-Per-Episode M-Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible fixed-per-episode OOD evaluation track and expose seed-4000 results for trained M0, M2, and M3 models in a separate `M-Models (Fix)` dashboard subtab.

**Architecture:** A stateless record builder derives one serializable augmentation record from `(seed, task_id, episode_index, level, version)`, and a stateful processor retains one record per vector slot for a rollout. A narrow LeRobot eval hook initializes those records before reset observations. Fixed results use a separate output tree and manifest collection so they cannot mix with frame-randomized results.

**Tech Stack:** Python 3.12+, PyTorch, LeRobot evaluation pipeline, pytest/unittest, static HTML/JavaScript, Node.js test runner.

**Spec:** `docs/superpowers/specs/2026-08-24-fixed-episode-evaluation-design.md`

## Global Constraints

- One deterministic record is shared by every frame and camera view in an episode.
- Identity is exactly `(evaluation seed, task ID, episode index, OOD level, schema version)` and cannot depend on call order, device, vector scheduling, or global RNG.
- Existing `outputs/eval/fair-v1/full/` data and frame-randomized behavior remain unchanged.
- Fixed outputs live only under `outputs/eval/fair-v1-fixed/full/`.
- First full matrix: seed 4000; `M0-clean`, `M2-online-dr`, `M3-v2-warm`; levels 0–2.
- `M1-offline-dr` is excluded until its trained checkpoint is available.
- Level 0 is pixel-identical and still records fixed provenance.
- Full runs require passing unit tests and M0/M2/M3 preflights at levels 0 and 2.

---

### Task 1: Deterministic Fixed OOD Records

**Files:**
- Create: `causal_aug/causal_aug/fixed_ood.py`
- Modify: `causal_aug/causal_aug/__init__.py`
- Create: `causal_aug/tests/test_fixed_ood.py`

**Interfaces:**
- Consumes: `OOD_LEVELS` and pure transforms from `causal_aug.ood_wrapper`.
- Produces: `FixedOODIdentity`, `derive_fixed_ood_record(identity) -> dict`, `apply_fixed_ood_record(images, record) -> Tensor`.

- [ ] **Step 1: Write failing identity tests**

```python
def test_record_is_reproducible_and_independent_of_global_rng():
    identity = FixedOODIdentity(4000, 3, 7, "level_2")
    torch.manual_seed(1)
    first = derive_fixed_ood_record(identity)
    torch.manual_seed(999)
    assert derive_fixed_ood_record(identity) == first

def test_different_episode_identity_changes_record():
    left = derive_fixed_ood_record(FixedOODIdentity(4000, 3, 7, "level_2"))
    right = derive_fixed_ood_record(FixedOODIdentity(4000, 3, 8, "level_2"))
    assert left != right
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest causal_aug/tests/test_fixed_ood.py -v`

Expected: FAIL with `ModuleNotFoundError: causal_aug.fixed_ood`.

- [ ] **Step 3: Implement stable derivation**

```python
@dataclass(frozen=True)
class FixedOODIdentity:
    evaluation_seed: int
    task_id: int
    episode_index: int
    level: str
    schema_version: int = 1

def _generator(identity):
    encoded = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":")).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)
```

Use this generator for affine, rotation, brightness, contrast, saturation, hue, shadow enable/alpha/direction, noise seed, blur enable, and normalized cutout coordinates.

- [ ] **Step 4: Add failing application tests**

```python
def test_same_record_repeats_pixels():
    record = derive_fixed_ood_record(FixedOODIdentity(4000, 1, 2, "level_2"))
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    assert torch.equal(apply_fixed_ood_record(image, record), apply_fixed_ood_record(image, record))

def test_level_zero_is_identity():
    image = torch.rand(2, 3, 16, 16)
    record = derive_fixed_ood_record(FixedOODIdentity(4000, 0, 0, "level_0"))
    assert torch.equal(apply_fixed_ood_record(image, record), image)
```

- [ ] **Step 5: Implement application without sampling**

Apply recorded values through pure transforms. Generate noise using a fresh CPU generator seeded by `record["noise_seed"]`, then move it to the image device. Never call `torch.rand`, `torch.randint`, `uniform_`, or `randn_like` in this function.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest causal_aug/tests/test_fixed_ood.py causal_aug/tests/test_ood_policy_view.py -v
git add causal_aug/causal_aug/fixed_ood.py causal_aug/causal_aug/__init__.py causal_aug/tests/test_fixed_ood.py
git commit -m "feat: add deterministic fixed OOD records"
```

---

### Task 2: Fixed Episode Processor Lifecycle

**Files:**
- Modify: `scripts/eval_ood.py`
- Create: `causal_aug/tests/test_fixed_ood_processor.py`

**Interfaces:**
- Consumes: Task 1 record APIs.
- Produces: `FixedEpisodeOODProcessorStep.begin_rollout(task_id: int, episode_indices: Sequence[int])`, `.observation(observation)`, `.provenance()`.

- [ ] **Step 1: Write lifecycle failure tests**

```python
def test_rejects_frame_before_begin_rollout():
    step = FixedEpisodeOODProcessorStep("level_2", evaluation_seed=4000)
    with pytest.raises(RuntimeError, match="begin_rollout"):
        step.observation({"observation.images.image": torch.rand(1, 3, 8, 8)})

def test_rejects_batch_mismatch():
    step = FixedEpisodeOODProcessorStep("level_2", evaluation_seed=4000)
    step.begin_rollout(task_id=2, episode_indices=[0, 1])
    with pytest.raises(ValueError, match="batch size"):
        step.observation({"observation.images.image": torch.rand(1, 3, 8, 8)})
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest causal_aug/tests/test_fixed_ood_processor.py -v`

Expected: FAIL because the processor is absent.

- [ ] **Step 3: Implement retained record state**

The constructor has no active records. `begin_rollout` replaces records for every vector slot. `observation` validates batch sizes and applies slot record `i` to every camera tensor slice `i:i+1`.

Provenance is exactly:

```python
{
  "algorithm": "causal_aug.FixedEpisodeOOD",
  "version": 1,
  "augmentation_scope": "episode",
  "record_identity": ["evaluation_seed", "task_id", "episode_index", "level", "schema_version"],
  "evaluation_seed": 4000,
}
```

- [ ] **Step 4: Test repeated frames, two cameras, two slots, and a second rollout**

Assert repeated calls are byte-identical, camera keys share records, slots differ, and a new episode identity changes output.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest causal_aug/tests/test_fixed_ood_processor.py -v
git add scripts/eval_ood.py causal_aug/tests/test_fixed_ood_processor.py
git commit -m "feat: retain OOD records for each episode"
```

---

### Task 3: LeRobot Rollout Initialization Hook

**Files:**
- Create: `lerobot_patches/lerobot_eval_fixed_episode.patch`
- Modify: `scripts/install_policy_patches.py`
- Modify: `causal_aug/tests/test_ood_policy_view.py`
- Modify: `causal_aug/tests/test_eval_workflows.py`

**Interfaces:**
- Consumes: processor steps with `begin_rollout(task_id, episode_indices)`.
- Produces: backward-compatible evaluator lifecycle initialization before the first processed reset observation.

- [ ] **Step 1: Write failing integration test**

Add a spy step that raises when `observation()` precedes `begin_rollout()`. Run a fake two-slot rollout and assert task ID and episode indices are received first.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest causal_aug/tests/test_ood_policy_view.py -k fixed_episode -v`

Expected: FAIL because no lifecycle hook exists.

- [ ] **Step 3: Patch the rollout boundary**

Before reset observations enter `env_preprocessor`:

```python
for step in env_preprocessor.steps:
    begin_rollout = getattr(step, "begin_rollout", None)
    if begin_rollout is not None:
        begin_rollout(task_id=task_id, episode_indices=episode_indices)
```

Use the episode sequence that also derives environment seeds. Never infer identity from frame count. Steps without the hook remain unchanged.

- [ ] **Step 4: Register idempotent installation**

Install this patch after `lerobot_eval_policy_view.patch`. Accept already-applied state and reject partial application.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest causal_aug/tests/test_ood_policy_view.py causal_aug/tests/test_eval_workflows.py -v
git add lerobot_patches/lerobot_eval_fixed_episode.patch scripts/install_policy_patches.py causal_aug/tests/test_ood_policy_view.py causal_aug/tests/test_eval_workflows.py
git commit -m "feat: initialize episode-aware eval processors"
```

---

### Task 4: Seed-4000 Fixed Fair-v1 Workflow

**Files:**
- Create: `configs/fair_v1_fixed.json`
- Create: `scripts/eval_fair_v1_fixed.py`
- Create: `scripts/run_fair_eval_fixed_v1.sh`
- Create: `causal_aug/tests/test_fixed_fair_eval_workflow.py`

**Interfaces:**
- Consumes: Tasks 2–3 and model metadata from `configs/fair_v1.json`.
- Produces: `build_fixed_matrix(mode: str) -> list[EvalRun]`, `build_fixed_eval_command(protocol: dict, run: EvalRun, mode: str, model_revision: str, output_dir: Path) -> list[str]`, `validate_fixed_result(path: Path, expected: FixedEvalExpectation) -> None`, and resumable fixed execution.

- [ ] **Step 1: Write failing matrix tests**

```python
def test_full_matrix_is_seed_4000_for_trained_models_only():
    matrix = build_fixed_matrix("full")
    assert {run.model_id for run in matrix} == {"M0-clean", "M2-online-dr", "M3-v2-warm"}
    assert {run.seed for run in matrix} == {4000}
    assert len(matrix) == 9

def test_command_cannot_target_original_tree():
    run = EvalRun("M0-clean", "level_0", 4000, 10)
    rendered = " ".join(build_fixed_eval_command(protocol, run, "full", "a" * 40, fixed_output))
    assert "outputs/eval/fair-v1-fixed/full" in rendered
    assert "--augmentation_scope=episode" in rendered
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest causal_aug/tests/test_fixed_fair_eval_workflow.py -v`

Expected: FAIL because the workflow is absent.

- [ ] **Step 3: Implement config and commands**

Config values: schema 1, algorithm 1, seeds `[4000]`, models M0/M2/M3, levels 0–2, 10 tasks, 10 episodes per task, 1 preflight episode. Commands include pinned revision, protocol digest, fixed scope, and fixed output root.

- [ ] **Step 4: Implement strict validation/resume**

Require 10 tasks, expected episodes, fixed scope/version, seed, model revision, protocol digest, clean videos, and policy videos. Skip valid artifacts; stop on invalid artifacts.

- [ ] **Step 5: Add wrapper tests**

The wrapper accepts model or `all`, `--mode preflight|full`, and `--dry-run`. Reject M1 and seeds other than 4000.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest causal_aug/tests/test_fixed_fair_eval_workflow.py -v
git add configs/fair_v1_fixed.json scripts/eval_fair_v1_fixed.py scripts/run_fair_eval_fixed_v1.sh causal_aug/tests/test_fixed_fair_eval_workflow.py
git commit -m "feat: add fixed fair evaluation workflow"
```

---

### Task 5: Fixed Manifest and Dashboard Subtab

**Files:**
- Modify: `scripts/build_results_data.py`
- Modify: `tests/test_build_results_data.py`
- Modify: `scripts/results_dashboard.js`
- Modify: `tests/results_dashboard.test.mjs`
- Modify: `pipeline.html`

**Interfaces:**
- Consumes: `outputs/eval/fair-v1-fixed/full/<model>/<level>/seed4000/eval_info.json`.
- Produces: `fixedModels`, `fixedRuns`, `fixedEpisodes`; dashboard view `m-models-fixed`.

- [ ] **Step 1: Write failing separation test**

Create original and fixed M0 fixtures with different success values. Assert original data appears only in `models/runs/episodes`, fixed data only in `fixedModels/fixedRuns/fixedEpisodes`, and fixed scope equals `episode`.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_build_results_data.BuildResultsDataTest.test_separates_fixed_episode_results`

Expected: FAIL because fixed collections are absent.

- [ ] **Step 3: Implement independent fixed-tree ingestion**

Extract focused aggregation helpers, scan original and fixed roots separately, validate fixed scope, and never concatenate collections.

- [ ] **Step 4: Write failing JS collection tests**

```javascript
test('M-Models Fix selects only fixed models', () => {
  assert.deepEqual(resultCollectionForView(data, 'm-models-fixed').models.map(x => x.id), ['M0-clean']);
});
```

Also prove `m-models` uses original data and `all` excludes fixed duplicates.

- [ ] **Step 5: Implement the UI**

Add `M-Models (Fix)`, reuse cards/table/filters/videos with the selected collection, show `Fixed per episode · Seed 4000`, and add a dedicated empty state.

- [ ] **Step 6: Verify and commit**

```bash
python -m unittest discover -s tests -v
node --test tests/results_dashboard.test.mjs
git add scripts/build_results_data.py tests/test_build_results_data.py scripts/results_dashboard.js tests/results_dashboard.test.mjs pipeline.html
git commit -m "feat: show fixed episode model results"
```

---

### Task 6: Install and Preflight

**Files:**
- Generate only: `outputs/eval/fair-v1-fixed/preflight/<model>/<level>/seed4000/` (not committed).
- Modify Task 1–4 files only when a new failing regression test demonstrates a defect.

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: six valid preflights: M0/M2/M3 at levels 0 and 2, seed 4000.

- [ ] **Step 1: Run the full available suite**

```bash
python -m pytest causal_aug/tests -v
python -m unittest discover -s tests -v
node --test tests/results_dashboard.test.mjs
git diff --check
```

- [ ] **Step 2: Install patches twice**

```bash
python scripts/install_policy_patches.py
python scripts/install_policy_patches.py
```

Expected: first applies missing patches; second reports already applied without changes.

- [ ] **Step 3: Dry-run preflights**

```bash
for model in M0-clean M2-online-dr M3-v2-warm; do
  ./scripts/run_fair_eval_fixed_v1.sh "$model" --mode preflight --dry-run
done
```

Inspect seed 4000, levels 0/2, pinned revisions, fixed scope, and fixed roots.

- [ ] **Step 4: Run all preflights**

```bash
for model in M0-clean M2-online-dr M3-v2-warm; do
  ./scripts/run_fair_eval_fixed_v1.sh "$model" --mode preflight
done
```

Expected: six artifacts, ten tasks each, one episode per task, videos and fixed provenance.

- [ ] **Step 5: Prove replay determinism**

Save one M0 level-2 manifest and first policy-view frame hash, rerun into a temporary fixed output root, and compare record and frame hashes. Expected: identical.

- [ ] **Step 6: Handle defects with TDD**

If a defect appears, first add a focused failing test, implement one fix, rerun Steps 1–5, and commit with `fix: preserve fixed episode evaluation contract`. If no defect appears, make no commit.

---

### Task 7: Full Seed-4000 Matrix and Published Results

**Files:**
- Generate only: `outputs/eval/fair-v1-fixed/full/<model>/<level>/seed4000/` (not committed).
- Modify: `results-data.json`
- Modify: `docs/fair-v1-results.md`

**Interfaces:**
- Consumes: six validated preflights.
- Produces: nine full runs, dashboard data, and fixed-vs-original seed-4000 report.

- [ ] **Step 1: Run one model at a time**

```bash
for model in M0-clean M2-online-dr M3-v2-warm; do
  ./scripts/run_fair_eval_fixed_v1.sh "$model" --mode full
done
```

Do not run M1, seed 5000, or seed 6000.

- [ ] **Step 2: Validate the matrix**

Require exactly nine runs, 900 episodes per model, levels 0–2, seed 4000 only, fixed provenance, and complete clean/policy video lists.

- [ ] **Step 3: Rebuild static data**

Run: `python scripts/build_results_data.py`

Assert `fixedModels` is exactly M0/M2/M3 and `fixedRuns` has nine entries.

- [ ] **Step 4: Update the report**

Add `Fixed-per-episode OOD · seed 4000` to `docs/fair-v1-results.md`: overall/level rates, counts, absolute delta from original seed-4000, and a single-seed/no-significance disclaimer.

- [ ] **Step 5: Final verification**

```bash
python -m pytest causal_aug/tests -v
python -m unittest discover -s tests -v
node --test tests/results_dashboard.test.mjs
git diff --check
```

Serve the project and verify M0/M2/M3 cards, seed label, filters, videos, provenance, and zero browser errors in `M-Models (Fix)`.

- [ ] **Step 6: Commit dashboard/report data**

```bash
git add results-data.json docs/fair-v1-results.md
git commit -m "results: add fixed episode seed 4000 evaluation"
```

- [ ] **Step 7: Report and request push approval**

Report exact metrics, full-run artifacts, test output, environment limitations, and commit list. Do not push implementation/results until the user explicitly approves.
