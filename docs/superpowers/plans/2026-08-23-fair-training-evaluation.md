# Fair Training and Evaluation Protocol v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validated M0–M3 LIBERO-Spatial workflow with matched exposure, four real one-step MPS smoke tests, reproducible Hugging Face handoff, and paired seed-4000 evaluation.

**Architecture:** A versioned JSON manifest is the source of truth. Focused Python modules validate it, create deterministic augmentations, materialize M1 data, construct train/eval commands, and validate artifacts; thin shell wrappers only configure the environment. Existing LeRobot policies remain the backend, with tracked patches for exact-balanced sampling and smoke instrumentation.

**Tech Stack:** Python 3.12, PyTorch, LeRobot/SmolVLA, Hugging Face Hub, Bash, pytest, Apple MPS, CUDA.

**Spec:** `docs/superpowers/specs/2026-08-23-fair-training-evaluation-design.md`

## Global Constraints

- Model IDs are `M0-clean`, `M1-offline-dr`, `M2-online-dr`, and `M3-v2-warm`.
- Full runs use 25,000 optimizer steps, batch size 16, seed 1000, and checkpoints every 5,000 steps.
- Pin the SmolVLA base model and `lerobot/libero_spatial_image` to immutable Hugging Face commit SHAs.
- M1 and M2 emit exactly eight clean and eight augmented samples per full batch and share one augmentation manifest/hash.
- M3 expands 16 sources to 16 clean plus 16 augmented views; action consistency warms from 0 to 0.05 over 10,000 steps.
- Smoke runs use one step, batch size 2, MPS with CPU fallback, no HF push, and `outputs/smoke/fair-v1/`.
- Primary eval uses LIBERO-Spatial, step 25,000, seed 4000, all three OOD levels, and 10 episodes/task after a level-0 one-episode/task preflight.
- Never overwrite a completed run or resume when its protocol hash differs.
- Never stage `.firecrawl/`.

## File Map

- `configs/fair_v1.json`: experiment matrix and immutable revisions.
- `configs/fair_v1_augmentation.json`: shared M1/M2 augmentation definition.
- `scripts/fair_protocol.py`: loading, invariants, hashing, run manifests, and command builders.
- `causal_aug/causal_aug/balanced_sampler.py`: exact-half masks and M1 paired batches.
- `causal_aug/causal_aug/fair_augmentation.py`: stateless records and tensor transforms.
- `scripts/materialize_fair_offline.py`: deterministic paired M1 dataset.
- `scripts/train_fair_v1.py`, `scripts/run_fair_v1.sh`: dry-run/smoke/full/resume training.
- `scripts/smoke_fair_v1.py`: four-model MPS smoke orchestration.
- `scripts/eval_fair_v1.py`, `scripts/run_fair_eval_v1.sh`: pinned preflight/full evaluation.
- `scripts/summarize_fair_v1.py`: paired pilot report.
- `lerobot_patches/lerobot_fair_sampler.patch`: optional M1 batch sampler and smoke hooks.
- `causal_aug/tests/test_fair_*.py`: protocol, augmentation, training, smoke, HF, and eval contracts.

---

### Task 1: Manifest and Protocol Invariants

**Files:**
- Create: `configs/fair_v1.json`
- Create: `configs/fair_v1_augmentation.json`
- Create: `scripts/fair_protocol.py`
- Test: `causal_aug/tests/test_fair_protocol.py`

**Interfaces:**
- Produces: `load_protocol(path: Path) -> dict`, `validate_protocol(protocol: dict) -> None`, `canonical_json(value: object) -> str`, `protocol_hash(protocol: dict) -> str`, `model_config(protocol: dict, model_id: str) -> dict`.
- Consumes: no earlier task.

- [ ] **Step 1: Write failing contract tests**

```python
def test_manifest_locks_the_shared_contract():
    protocol = load_protocol(ROOT / "configs/fair_v1.json")
    validate_protocol(protocol)
    assert protocol["training"]["steps"] == 25000
    assert protocol["training"]["batch_size"] == 16
    assert protocol["training"]["seed"] == 1000
    assert protocol["training"]["save_freq"] == 5000
    assert list(protocol["models"]) == list(MODEL_IDS)

def test_mutable_revision_is_rejected():
    protocol = load_protocol(ROOT / "configs/fair_v1.json")
    protocol["dataset"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable 40-character commit"):
        validate_protocol(protocol)
```

Add negative tests for the wrong model order, M1/M2 ratios other than 8:8, mismatched augmentation hashes, M3 weights other than 0.5/0.5, lambda other than 0.05, and warmup other than 10,000.

- [ ] **Step 2: Run and observe import failure**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_protocol.py`

Expected: FAIL because `scripts.fair_protocol` is absent.

- [ ] **Step 3: Create explicit manifests**

Resolve real SHAs with `HfApi.model_info(repo_id).sha` and `HfApi.dataset_info(repo_id).sha`. Store top-level `protocol_version`, `base_model`, `dataset`, `training`, `augmentation_manifest`, `models`, and `evaluation`. Store the four HF repo names from the spec, evaluation seed 4000, levels 0/1/2, and 10 episodes/task. The augmentation manifest lists only transforms already supported by `CausalAugmenter`, with explicit numeric distributions.

- [ ] **Step 4: Implement strict validation and canonical hashing**

```python
MODEL_IDS = ("M0-clean", "M1-offline-dr", "M2-online-dr", "M3-v2-warm")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def protocol_hash(protocol: dict) -> str:
    return hashlib.sha256(canonical_json(protocol).encode()).hexdigest()
```

`validate_protocol` must validate every global constraint and verify the augmentation file SHA-256.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_protocol.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/fair_v1.json configs/fair_v1_augmentation.json scripts/fair_protocol.py causal_aug/tests/test_fair_protocol.py
git commit -m "feat: define fair v1 experiment protocol"
```

### Task 2: Exact-Balanced M1/M2 Sampling

**Files:**
- Create: `causal_aug/causal_aug/balanced_sampler.py`
- Modify: `causal_aug/causal_aug/__init__.py`
- Modify: `lerobot_patches/online_dr/configuration_online_dr.py`
- Modify: `lerobot_patches/online_dr/modeling_online_dr.py`
- Create: `lerobot_patches/lerobot_fair_sampler.patch`
- Modify: `scripts/install_policy_patches.py`
- Test: `causal_aug/tests/test_balanced_sampler.py`
- Modify: `causal_aug/tests/test_residual_policy_contract.py`

**Interfaces:**
- Produces: `exact_half_mask(batch_size: int, device: torch.device | str, generator: torch.Generator | None = None) -> Tensor`; `PairedBatchSampler(clean_indices: Sequence[int], augmented_indices: Sequence[int], batch_size: int, seed: int, drop_last: bool = True)`.
- Consumes: M1/M2 ratios from Task 1.

- [ ] **Step 1: Write failing balance tests**

```python
def test_exact_half_mask_is_reproducible():
    a = exact_half_mask(16, "cpu", torch.Generator().manual_seed(1000))
    b = exact_half_mask(16, "cpu", torch.Generator().manual_seed(1000))
    assert a.sum().item() == 8
    assert torch.equal(a, b)

def test_paired_batches_have_equal_domains():
    sampler = PairedBatchSampler(range(8), range(8, 16), 4, 1000)
    for batch in sampler:
        assert sum(index < 8 for index in batch) == 2
        assert sum(index >= 8 for index in batch) == 2
```

Also reject odd/non-positive batches and unequal pools.

- [ ] **Step 2: Run and observe missing module**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_balanced_sampler.py`

Expected: FAIL during import.

- [ ] **Step 3: Implement exact-half selection**

```python
def exact_half_mask(batch_size, device, generator=None):
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("batch_size must be a positive even integer")
    order = torch.randperm(batch_size, generator=generator, device=device)
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask[order[: batch_size // 2]] = True
    return mask
```

Implement `PairedBatchSampler` using separate seeded permutations for each pool, interleave half-batches, and return the correct full-batch count from `__len__`.

- [ ] **Step 4: Route M2 through exact balance**

Add `exact_balance: bool = True` to `OnlineDRConfig`. When true, require `aug_probability == 0.5` and call `exact_half_mask`; retain the existing Bernoulli path only when explicitly false.

- [ ] **Step 5: Patch LeRobot DataLoader for M1**

Add optional configuration fields `paired_clean_count`, `paired_augmented_count`, and `paired_batch_seed`. When present, construct `PairedBatchSampler` and pass it as `batch_sampler`, omitting conflicting DataLoader arguments. Make installer application idempotent by checking for the marker `PairedBatchSampler`.

- [ ] **Step 6: Run focused regression tests**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_balanced_sampler.py causal_aug/tests/test_residual_policy_contract.py`

Expected: PASS with M2 augmented fraction exactly 0.5.

- [ ] **Step 7: Commit**

```bash
git add causal_aug/causal_aug/balanced_sampler.py causal_aug/causal_aug/__init__.py causal_aug/tests/test_balanced_sampler.py causal_aug/tests/test_residual_policy_contract.py lerobot_patches/online_dr lerobot_patches/lerobot_fair_sampler.patch scripts/install_policy_patches.py
git commit -m "feat: enforce balanced clean augmented batches"
```

### Task 3: Shared Augmentation Records and M1 Dataset

**Files:**
- Create: `causal_aug/causal_aug/fair_augmentation.py`
- Modify: `causal_aug/causal_aug/__init__.py`
- Create: `scripts/materialize_fair_offline.py`
- Modify: `scripts/augment_dataset.py`
- Test: `causal_aug/tests/test_fair_augmentation.py`

**Interfaces:**
- Produces: `derive_record(manifest: dict, seed: int, episode_id: int, frame_index: int, exposure_index: int) -> dict`; `apply_record(images: list[Tensor], record: dict) -> list[Tensor]`; `augmentation_records.jsonl`.
- Consumes: Task 1 manifests and hashing.

- [ ] **Step 1: Write failing determinism/equivalence tests**

```python
def test_record_identity_is_stable():
    first = derive_record(MANIFEST, 1000, 7, 42, 0)
    second = derive_record(MANIFEST, 1000, 7, 42, 0)
    assert first == second
    assert first != derive_record(MANIFEST, 1000, 7, 43, 0)

def test_offline_online_pixels_match():
    image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    record = derive_record(MANIFEST, 1000, 1, 2, 0)
    torch.testing.assert_close(
        apply_record([image.clone()], record)[0],
        apply_record([image.clone()], record)[0], rtol=0, atol=1 / 255,
    )
```

- [ ] **Step 2: Run and observe import failure**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_augmentation.py`

Expected: FAIL during import.

- [ ] **Step 3: Implement stateless records**

Hash `seed:episode_id:frame_index:exposure_index` with SHA-256, seed a local Torch generator from the first eight bytes, and sample every parameter from that generator. Return source identifiers, schema version, transform parameters, and manifest SHA. Never use global Python/Torch RNG state.

- [ ] **Step 4: Implement paired materialization**

Require `--protocol`, `--output-root`, and `--records-out`; support `--max-episodes` and `--push-to-hub`. Write clean items first and augmented items second, attach `domain` and `source_index`, and write one JSONL record per augmented frame. Legacy `augment_dataset.py` behavior remains unchanged unless `--fair-protocol` is passed.

- [ ] **Step 5: Verify two one-episode materializations**

Run materialization twice into temporary directories; compare record-file SHA-256 and decoded tensors from fixed indices.

Expected: identical records and tensors.

- [ ] **Step 6: Run augmentation regression tests**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_augmentation.py causal_aug/tests/test_gpu_augmenter.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add causal_aug/causal_aug/fair_augmentation.py causal_aug/causal_aug/__init__.py causal_aug/tests/test_fair_augmentation.py scripts/materialize_fair_offline.py scripts/augment_dataset.py
git commit -m "feat: match offline and online augmentation records"
```

### Task 4: Training Runner and Safe Run Manifests

**Files:**
- Modify: `scripts/fair_protocol.py`
- Create: `scripts/train_fair_v1.py`
- Create: `scripts/run_fair_v1.sh`
- Test: `causal_aug/tests/test_fair_train_workflow.py`

**Interfaces:**
- Produces: `build_train_command(protocol: dict, model_id: str, mode: Literal["smoke", "full"], output_dir: Path) -> list[str]`; `start_run_manifest(output_dir: Path, protocol: dict, model_id: str) -> Path`; `finish_run_manifest(path: Path, status: str, error: str | None = None) -> None`.
- Consumes: Tasks 1–3.

- [ ] **Step 1: Write failing four-model command tests**

```python
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_full_commands_share_locked_values(model_id, tmp_path):
    command = shlex.join(build_train_command(PROTOCOL, model_id, "full", tmp_path / model_id))
    assert "--steps=25000" in command
    assert "--batch_size=16" in command
    assert "--seed=1000" in command
    assert "--save_freq=5000" in command
```

Assert M0/M1 use SmolVLA, M1 uses paired counts, M2 uses online DR exact balance, M3 has lambda 0.05/warmup 10,000, and smoke commands cannot push.

- [ ] **Step 2: Run and observe missing builder**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_train_workflow.py`

Expected: FAIL because builder APIs are absent.

- [ ] **Step 3: Build argument lists without shell interpolation**

Full M0/M1 use `smolvla`; M2 uses `online_dr`; M3 uses `causal_vla_warm`. Always include pinned base/dataset revisions, output, repo, cadence, optimizer, and scheduler. Smoke overrides only steps=1, batch=2, device=mps, workers=0, push=false, and output root.

- [ ] **Step 4: Implement atomic manifests and safe resume**

Write JSON through a sibling temporary file and `Path.replace`. Record resolved config, protocol SHA, model, Git commit, revisions, environment, timestamps, metrics, and status. Reject completed runs and mismatched resume hashes. Record exception type/message before re-raising failures.

- [ ] **Step 5: Add CLI and wrapper**

The CLI accepts model ID, `--protocol`, `--mode`, `--dry-run`, `--resume`, and `--output-dir`. The shell wrapper sets `PYTHONNOUSERSITE`, `PYTHONPATH`, and unbuffered logs without duplicating experiment values.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_protocol.py causal_aug/tests/test_fair_train_workflow.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/fair_protocol.py scripts/train_fair_v1.py scripts/run_fair_v1.sh causal_aug/tests/test_fair_train_workflow.py
git commit -m "feat: add safe fair v1 training runner"
```

### Task 5: Real One-Step MPS Smoke Suite

**Files:**
- Create: `scripts/smoke_fair_v1.py`
- Modify: `scripts/train_fair_v1.py`
- Modify: `lerobot_patches/lerobot_fair_sampler.patch`
- Modify: `causal_aug/tests/test_fair_train_workflow.py`

**Interfaces:**
- Produces: `run_smoke(model_id: str, protocol_path: Path, output_root: Path) -> dict`; `validate_smoke_artifacts(output_dir: Path) -> dict`.
- Consumes: training commands and manifests from Task 4.

- [ ] **Step 1: Write failing smoke acceptance tests**

Use a fake trainer to write `training_metrics.json`, checkpoint files, and `reload_metrics.json`. Assert rejection of missing files, non-finite loss, unchanged parameters, failed reload, wrong M1/M2 ratio, and nonzero initial M3 consistency weight.

```python
def test_smoke_validator_accepts_real_update_and_reload(tmp_path):
    write_smoke_artifacts(tmp_path, loss=0.25, changed=True, reload_ok=True)
    assert validate_smoke_artifacts(tmp_path)["loss"] == 0.25
```

- [ ] **Step 2: Run and observe missing validation**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_train_workflow.py -k smoke`

Expected: FAIL.

- [ ] **Step 3: Add tracked smoke instrumentation**

Patch the trainer to record initial/final parameter checksums, finite loss, checkpoint path, reload inference, M1/M2 augmented fraction, and M3 consistency weight. Validate all ten spec acceptance criteria.

- [ ] **Step 4: Orchestrate all four models sequentially**

Install patches, prepare deterministic tiny original/M1 datasets, run one model at a time to limit MPS memory, release objects between runs, and write `outputs/smoke/fair-v1/summary.json`.

- [ ] **Step 5: Run contract tests and real smoke**

```bash
PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_train_workflow.py
PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/miniconda3/envs/causalvla/bin/python scripts/smoke_fair_v1.py --protocol configs/fair_v1.json
```

Expected: four completed models; finite losses, changed parameters, save/reload success, M1/M2 fraction 0.5, and M3 initial consistency weight 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_fair_v1.py scripts/train_fair_v1.py causal_aug/tests/test_fair_train_workflow.py lerobot_patches/lerobot_fair_sampler.patch
git commit -m "test: add four-model MPS smoke workflow"
```

### Task 6: Hugging Face Handoff

**Files:**
- Modify: `scripts/fair_protocol.py`
- Modify: `scripts/train_fair_v1.py`
- Modify: `causal_aug/tests/test_fair_train_workflow.py`

**Interfaces:**
- Produces: `resolve_hf_revision(repo_type: Literal["model", "dataset"], repo_id: str, revision: str | None = None) -> str`; `validate_hf_checkpoint(repo_id: str, revision: str, expected_protocol_hash: str) -> dict`.
- Consumes: Task 4 completed run metadata.

- [ ] **Step 1: Write failing mocked-HF tests**

```python
def test_model_revision_resolves_to_sha(fake_api):
    fake_api.model_info.return_value.sha = "a" * 40
    assert resolve_hf_revision("model", "owner/model", api=fake_api) == "a" * 40

def test_downloaded_metadata_rejects_wrong_hash():
    with pytest.raises(ValueError, match="protocol hash"):
        validate_downloaded_metadata({"protocol_sha256": "bad"}, "good")
```

- [ ] **Step 2: Run and observe missing APIs**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_train_workflow.py -k 'hf or revision'`

Expected: FAIL.

- [ ] **Step 3: Implement immutable upload verification**

Upload resolved config, run manifest, augmentation hash, Git commit, runtime metrics, and model card. Resolve the resulting SHA, download metadata at that SHA, verify protocol hash, and save the SHA locally. Normal smoke performs no HF mutation.

- [ ] **Step 4: Add guarded round-trip smoke mode**

Permit `--hf-round-trip-repo` only in smoke mode and require `fair-v1-smoke` in its name. Upload and verify but never delete automatically.

- [ ] **Step 5: Run mocked HF tests**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_train_workflow.py -k 'hf or revision'`

Expected: PASS without network mutations.

- [ ] **Step 6: Commit**

```bash
git add scripts/fair_protocol.py scripts/train_fair_v1.py causal_aug/tests/test_fair_train_workflow.py
git commit -m "feat: verify fair model handoff through Hugging Face"
```

### Task 7: Pinned Seed-4000 Evaluation

**Files:**
- Create: `scripts/eval_fair_v1.py`
- Create: `scripts/run_fair_eval_v1.sh`
- Create: `causal_aug/tests/test_fair_eval_workflow.py`
- Modify: `scripts/build_results_data.py`
- Modify: `tests/test_build_results_data.py`

**Interfaces:**
- Produces: `evaluation_matrix(protocol: dict, phase: Literal["preflight", "full"]) -> list[EvalRun]`; `build_eval_command(protocol: dict, model_id: str, level: str, episodes: int, revision: str, output_dir: Path) -> list[str]`; `validate_eval_result(path: Path, expected: EvalExpectation) -> dict`.
- Consumes: immutable HF resolution and existing `scripts/eval_ood.py`.

- [ ] **Step 1: Write failing matrix/result tests**

```python
def test_primary_matrix_has_twelve_runs():
    matrix = evaluation_matrix(PROTOCOL, "full")
    assert len(matrix) == 12
    assert {run.seed for run in matrix} == {4000}
    assert {run.level for run in matrix} == {"level_0", "level_1", "level_2"}
    assert {run.episodes_per_task for run in matrix} == {10}

def test_preflight_has_four_clean_runs():
    matrix = evaluation_matrix(PROTOCOL, "preflight")
    assert len(matrix) == 4
    assert {run.level for run in matrix} == {"level_0"}
    assert {run.episodes_per_task for run in matrix} == {1}
```

Also reject incomplete tasks/episodes, wrong seeds, mutable revisions, missing provenance, and missing clean/policy video paths.

- [ ] **Step 2: Run and observe missing evaluator**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_eval_workflow.py`

Expected: FAIL.

- [ ] **Step 3: Build pinned MPS commands**

Use `eval_ood.py`, LIBERO-Spatial, MPS fallback, batch 2, synchronous envs, seed 4000, and the existing rename map. Resolve each HF SHA before command construction. Store runs under `outputs/eval/fair-v1/<phase>/<model>/<level>/seed4000`.

- [ ] **Step 4: Validate completeness and idempotence**

Require ten tasks and the exact episode count, matching seed/OOD provenance, policy revision, metrics, and video paths. Skip only a fully validated completed run; reject partial or mismatched output.

- [ ] **Step 5: Extend dashboard parsing without breaking legacy IDs**

Parse fair-v1 directories in addition to legacy output. Include model revision and protocol hash in run metadata; add four-model fixtures.

- [ ] **Step 6: Run tests and dry runs**

```bash
PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_eval_workflow.py tests/test_build_results_data.py
./scripts/run_fair_eval_v1.sh --phase preflight --dry-run
./scripts/run_fair_eval_v1.sh --phase full --dry-run
```

Expected: tests PASS; 4 preflight commands and 12 full commands, all seed 4000 with pinned SHAs.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_fair_v1.py scripts/run_fair_eval_v1.sh causal_aug/tests/test_fair_eval_workflow.py scripts/build_results_data.py tests/test_build_results_data.py
git commit -m "feat: add paired fair v1 evaluation matrix"
```

### Task 8: Pilot Report and Compute Disclosure

**Files:**
- Create: `scripts/summarize_fair_v1.py`
- Modify: `causal_aug/tests/test_fair_eval_workflow.py`
- Create: `docs/fair-v1-results.md`

**Interfaces:**
- Produces: `summarize_runs(paths: Sequence[Path]) -> dict`; Markdown report.
- Consumes: validated Task 7 results and training runtime metrics.

- [ ] **Step 1: Write failing summary tests**

```python
def test_summary_reports_degradation_and_m0_delta(tmp_path):
    report = summarize_runs(write_complete_fixture_matrix(tmp_path))
    cell = report["M1-offline-dr"]["level_2"]
    assert cell["success_rate"] == pytest.approx(0.4)
    assert cell["degradation_from_level_0"] == pytest.approx(-0.3)
    assert cell["delta_vs_m0"] == pytest.approx(0.1)
```

- [ ] **Step 2: Run and observe missing summarizer**

Run: `PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_eval_workflow.py -k summary`

Expected: FAIL.

- [ ] **Step 3: Implement paired metrics**

Report overall/per-task success, absolute OOD scores, degradation from level 0, episode-paired delta against M0, inference time, GPU-hours, wall time, samples/second, and peak memory. Label the output `Pilot feasibility result: evaluation seed 4000 only`; make no significance claim.

- [ ] **Step 4: Handle empty and partial matrices explicitly**

With no results, render the protocol table plus `Results pending`. Reject incomplete matrices unless `--allow-partial`; partial reports list every missing cell.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=causal_aug /opt/miniconda3/envs/causalvla/bin/python -m pytest -q causal_aug/tests/test_fair_eval_workflow.py
git add scripts/summarize_fair_v1.py causal_aug/tests/test_fair_eval_workflow.py docs/fair-v1-results.md
git commit -m "feat: report fair v1 paired pilot results"
```

### Task 9: End-to-End Verification and Operator Handoff

**Files:**
- Modify: `PIPELINE.md`
- Modify: `docs/fair-v1-results.md`
- Test: all files created above.

**Interfaces:**
- Produces: exact Mac smoke, GPU training, HF verification, Mac evaluation, and report commands.
- Consumes: Tasks 1–8.

- [ ] **Step 1: Add exact operator commands to `PIPELINE.md`**

```bash
/opt/miniconda3/envs/causalvla/bin/python scripts/smoke_fair_v1.py --protocol configs/fair_v1.json
./scripts/run_fair_v1.sh M0-clean --mode full
./scripts/run_fair_v1.sh M1-offline-dr --mode full
./scripts/run_fair_v1.sh M2-online-dr --mode full
./scripts/run_fair_v1.sh M3-v2-warm --mode full
./scripts/run_fair_eval_v1.sh --phase preflight
./scripts/run_fair_eval_v1.sh --phase full
/opt/miniconda3/envs/causalvla/bin/python scripts/summarize_fair_v1.py
```

Document authentication, repository names, output paths, resume rules, and the single-seed interpretation limit.

- [ ] **Step 2: Run the full suite from `/tmp` to avoid package shadowing**

Run: `cd /tmp && PYTHONPATH=/Users/phawit/Projects/CausalVLA/causal_aug:/Users/phawit/Projects/CausalVLA /opt/miniconda3/envs/causalvla/bin/python -m pytest -q /Users/phawit/Projects/CausalVLA/tests /Users/phawit/Projects/CausalVLA/causal_aug/tests`

Expected: zero failures.

- [ ] **Step 3: Validate patches and whitespace**

```bash
git diff --check
/opt/miniconda3/envs/causalvla/bin/python scripts/install_policy_patches.py online_dr causal_vla causal_vla_warm
```

Expected: no whitespace error; installer succeeds or reports already installed.

- [ ] **Step 4: Run the real four-model MPS smoke**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/miniconda3/envs/causalvla/bin/python scripts/smoke_fair_v1.py --protocol configs/fair_v1.json`

Expected: four completed models and every acceptance field true in `outputs/smoke/fair-v1/summary.json`.

- [ ] **Step 5: Verify all full GPU commands without training**

```bash
for model in M0-clean M1-offline-dr M2-online-dr M3-v2-warm; do
  ./scripts/run_fair_v1.sh "$model" --mode full --dry-run
done
```

Expected: four commands with 25,000 steps, batch 16, seed 1000, pinned revisions, distinct HF repos, and correct policies.

- [ ] **Step 6: Verify evaluation commands without simulation**

```bash
./scripts/run_fair_eval_v1.sh --phase preflight --dry-run
./scripts/run_fair_eval_v1.sh --phase full --dry-run
```

Expected: four preflight and twelve full commands, all seed 4000 and pinned revisions.

- [ ] **Step 7: Commit handoff docs**

Check `git status --short`, keep `.firecrawl/` unstaged, then run:

```bash
git add PIPELINE.md docs/fair-v1-results.md
git commit -m "docs: add fair v1 operator runbook"
```

- [ ] **Step 8: Stop before expensive work**

Report smoke evidence, automated test count, dry-run matrices, HF repo names, and the first GPU-server command. Do not start full GPU training or the 1,200-episode evaluation.
