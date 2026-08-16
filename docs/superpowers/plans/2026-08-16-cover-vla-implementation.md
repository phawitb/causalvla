# COVER-VLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build COVER-Base and COVER-Safe as single-forward SmolVLA training policies that adapt supervised loss pressure across augmentation groups while preserving fixed clean and per-group coverage floors.

**Architecture:** A focused `causal_aug.cover_control` module owns group sampling, EMA statistics, constrained target masses, importance weights, and clean-retention state. Two thin LeRobot policies share one implementation: `cover_base` uses full robust strength and `cover_safe` enables the clean controller. Both generate exactly one observation view per sample, call the underlying VLA forward once, and inherit SmolVLA inference unchanged.

**Tech Stack:** Python 3.12, PyTorch, LeRobot SmolVLA, pytest, Hugging Face Hub, LIBERO, MPS/CUDA.

## Global Constraints

- Groups are exactly `clean`, `brightness`, `color`, `noise`, `blur`, `shadow`, `geometry`, and `composed`.
- Clean target mass is exactly `0.50`; composed has total-sample floor `0.15`; each atomic augmented group has total-sample floor `0.025`.
- Augmentation ranges and intensity are identical to Model F; camera views share scene-level nuisance parameters.
- The underlying VLA training model is called exactly once per batch; no latent/action consistency or policy-disagreement loss is permitted.
- EMA decay is `0.95`, warm-up is `1,000` steps, robust temperature is `0.5`, and allocation update interval is `100` steps.
- Importance weights are detached, clipped to `[0.5, 2.0]`, and normalized to mean one.
- COVER-Safe defaults: fast decay `0.90`, slow decay `0.99`, tolerance `0.05`, minimum robust strength `0.25`, decay multiplier `0.90`, recovery `0.01`.
- Training state must round-trip through `state_dict`; inference must remain identical to SmolVLA.
- Pilot budget is 5K steps per variant and 5 episodes/task; only one selected variant may receive a 25K full run.

---

## File Structure

- Create `causal_aug/causal_aug/cover_control.py`: group constants, constrained controller, clean-safety controller, and pure loss-weight helpers.
- Modify `causal_aug/causal_aug/intervention_bank.py`: expose one named augmentation family per sample while reusing current transforms/ranges.
- Modify `causal_aug/causal_aug/__init__.py`: export Phase 10 interfaces.
- Create `causal_aug/tests/test_cover_control.py`: mathematical/controller behavior tests.
- Create `causal_aug/tests/test_cover_policy_contract.py`: one-forward, masking, metrics, state, and inference contracts.
- Create `lerobot_patches/cover_base/{__init__,configuration_cover_base,modeling_cover_base,processor_cover_base}.py`: shared COVER policy implementation and L1 registration.
- Create `lerobot_patches/cover_safe/{__init__,configuration_cover_safe,modeling_cover_safe,processor_cover_safe}.py`: L2 configuration and thin subclass.
- Modify `scripts/install_policy_patches.py`: install/register both policies.
- Create `scripts/smoke_cover_mps.sh`: deterministic two-step MPS smoke for both variants.
- Create `scripts/run_eval_cover.sh`: revision-pinned evaluation entry point.
- Modify `worklog/phase10.md`: verification commands and measured status only.

---

### Task 1: Coverage-Constrained Group Controller

**Files:**
- Create: `causal_aug/causal_aug/cover_control.py`
- Create: `causal_aug/tests/test_cover_control.py`
- Modify: `causal_aug/causal_aug/__init__.py`

**Interfaces:**
- Produces: `COVER_GROUPS: tuple[str, ...]`
- Produces: `CoverageController(nn.Module)`
- Produces: `CoverageController.sample(batch_size: int, device: torch.device | str) -> Tensor`
- Produces: `CoverageController.update(losses: Tensor, group_ids: Tensor) -> None`
- Produces: `CoverageController.target_mass() -> Tensor`
- Produces: `CoverageController.importance_weights(group_ids: Tensor) -> Tensor`
- Produces: `CoverageController.metrics() -> dict[str, float]`

- [ ] **Step 1: Write failing probability and monotonicity tests**

```python
def test_target_mass_preserves_exact_floors_and_probability():
    c = CoverageController(warmup_steps=0, update_interval=1)
    c.loss_ema.copy_(torch.tensor([1., 1., 1., 1., 1., 1., 1., 4.]))
    c.initialized.fill_(True)
    mass = c.target_mass()
    assert mass.sum().item() == pytest.approx(1.0)
    assert mass[0].item() == pytest.approx(0.50)
    assert torch.all(mass[1:7] >= 0.025)
    assert mass[7].item() >= 0.15
    assert mass[7] > mass[1]

def test_importance_weights_are_detached_clipped_and_normalized():
    c = CoverageController(warmup_steps=0, update_interval=1)
    ids = torch.tensor([0, 0, 0, 1, 7])
    weights = c.importance_weights(ids)
    assert not weights.requires_grad
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights.min() >= 0.5 and weights.max() <= 2.0
```

- [ ] **Step 2: Run tests and verify collection failure**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_control.py`

Expected: FAIL because `CoverageController` and `COVER_GROUPS` do not exist.

- [ ] **Step 3: Implement constants, validation, target mass, and sampling**

```python
COVER_GROUPS = (
    "clean", "brightness", "color", "noise", "blur", "shadow",
    "geometry", "composed",
)

class CoverageController(nn.Module):
    def __init__(self, ema_decay=0.95, warmup_steps=1000,
                 temperature=0.5, update_interval=100,
                 weight_min=0.5, weight_max=2.0):
        super().__init__()
        self.register_buffer("loss_ema", torch.zeros(8))
        self.register_buffer("initialized", torch.zeros(8, dtype=torch.bool))
        self.register_buffer("selection_counts", torch.zeros(8, dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("cached_mass", torch.tensor(
            [0.50] + [0.025 + 0.20 / 7] * 6 + [0.15 + 0.20 / 7]
        ))
```

Implement augmented floors as `[0.025] * 6 + [0.15]`, distribute the remaining
`0.20` total probability through a temperature softmax, and use uniform scores
during warm-up or when no finite initialized scores exist. Validate all ranges,
floor sum, positive temperature/update interval, and `weight_min <= weight_max`.

- [ ] **Step 4: Add EMA, absent-group, non-finite, cadence, and state tests**

```python
def test_update_skips_absent_and_nonfinite_groups():
    c = CoverageController(ema_decay=0.5, warmup_steps=0, update_interval=1)
    c.update(torch.tensor([2.0, float("nan"), 4.0]), torch.tensor([1, 2, 1]))
    assert c.initialized[1]
    assert c.loss_ema[1].item() == pytest.approx(3.0)
    assert not c.initialized[2]

def test_state_dict_round_trip():
    source = CoverageController(warmup_steps=0, update_interval=1)
    source.update(torch.tensor([1., 2.]), torch.tensor([0, 7]))
    target = CoverageController(warmup_steps=0, update_interval=1)
    target.load_state_dict(source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
```

- [ ] **Step 5: Implement update, importance correction, metrics, and exports**

Use `scatter_add_` for batch group sums/counts, update only finite present group
means under `torch.no_grad()`, refresh `cached_mass` only on the configured
cadence, and compute observed frequency with `torch.bincount(..., minlength=8)`.
When an observed frequency is zero it receives no sample weight. Use a bounded
mean-one projection after the raw ratio so the final weights simultaneously
satisfy the clip bounds and mean-one constraint; a plain clip-then-divide is
insufficient because division can leave the bounds. Export the controller and
constants from `causal_aug/__init__.py`.

- [ ] **Step 6: Run focused tests**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_control.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add causal_aug/causal_aug/cover_control.py causal_aug/causal_aug/__init__.py causal_aug/tests/test_cover_control.py
git commit -m "feat: add coverage-constrained group controller"
```

---

### Task 2: Named Single-View Intervention Application

**Files:**
- Modify: `causal_aug/causal_aug/intervention_bank.py`
- Modify: `causal_aug/causal_aug/__init__.py`
- Create: `causal_aug/tests/test_cover_interventions.py`

**Interfaces:**
- Consumes: `COVER_GROUPS`
- Produces: `apply_cover_groups(images: list[Tensor], group_ids: Tensor, intensity: float = 1.0) -> list[Tensor]`

- [ ] **Step 1: Write failing identity, shape, determinism, and camera-sharing tests**

```python
def test_clean_group_is_exact_identity():
    images = [torch.rand(8, 3, 32, 32) * 2 - 1 for _ in range(2)]
    out = apply_cover_groups(images, torch.zeros(8, dtype=torch.long))
    assert all(torch.equal(a, b) for a, b in zip(images, out))

@pytest.mark.parametrize("group_id", range(1, 8))
def test_each_augmented_group_changes_pixels_and_preserves_shape(group_id):
    torch.manual_seed(1000)
    images = [torch.linspace(-1, 1, 8*3*32*32).reshape(8, 3, 32, 32)]
    out = apply_cover_groups(images, torch.full((8,), group_id), 1.0)
    assert out[0].shape == images[0].shape
    assert torch.isfinite(out[0]).all()
    assert not torch.equal(out[0], images[0])
```

- [ ] **Step 2: Run tests and verify missing-interface failure**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_interventions.py`

Expected: FAIL importing `apply_cover_groups`.

- [ ] **Step 3: Implement one named family per group**

Reuse the exact functions and parameter ranges already used by
`CausalAugmenter`: brightness; contrast/saturation/hue as `color`; Gaussian
noise; Gaussian blur; shadow; perspective/affine/rotation as `geometry`; and
`CausalAugmenter(K=1).augment_camera_views` as `composed`. Generate parameters
once per sample and reuse them across every camera view. Apply each transform
only where `group_ids` matches and use `torch.where` to preserve other samples.

- [ ] **Step 4: Add invalid-input tests and validation**

Test empty images, mismatched batch sizes, group IDs outside `[0, 7]`, negative
intensity, and non-floating images. Raise `ValueError` with the offending field
in each message.

- [ ] **Step 5: Run focused and existing augmentation tests**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_interventions.py causal_aug/tests/test_intervention_bank.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add causal_aug/causal_aug/intervention_bank.py causal_aug/causal_aug/__init__.py causal_aug/tests/test_cover_interventions.py
git commit -m "feat: add named COVER intervention groups"
```

---

### Task 3: COVER-Safe Clean Controller

**Files:**
- Modify: `causal_aug/causal_aug/cover_control.py`
- Modify: `causal_aug/tests/test_cover_control.py`

**Interfaces:**
- Produces: `CoverCleanController(nn.Module)`
- Produces: `CoverCleanController.update(clean_loss: Tensor) -> tuple[Tensor, Tensor]`
- Consumed by: `CoverageController.target_mass(robust_strength: Tensor | float = 1.0)`

- [ ] **Step 1: Write failing decay, recovery, bounds, warm-up, and NaN tests**

```python
def test_cover_clean_controller_decays_and_recovers_strength():
    c = CoverCleanController(warmup_steps=0, strength_decay=0.9, recovery=0.01)
    c.fast_ema.fill_(2.0); c.slow_ema.fill_(1.0); c.initialized.fill_(True)
    strength, triggered = c.update(torch.tensor(2.0))
    assert triggered.item() is True
    assert strength.item() == pytest.approx(0.9)
    c.fast_ema.fill_(1.0); c.slow_ema.fill_(1.0)
    strength, triggered = c.update(torch.tensor(1.0))
    assert triggered.item() is False
    assert strength.item() == pytest.approx(0.91)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_control.py -k clean_controller`

Expected: FAIL because `CoverCleanController` is undefined.

- [ ] **Step 3: Implement registered-buffer controller and interpolation**

Track `fast_ema`, `slow_ema`, `initialized`, `step`, and `robust_strength` as
buffers. During warm-up return strength 1.0. On a finite post-warm-up clean
loss, update EMAs, decay when fast exceeds slow by tolerance, otherwise recover;
clamp to `[0.25, 1.0]`. Change `CoverageController.target_mass` so strength zero
means uniform allocation of the adaptive `0.20` mass and strength one means the
full loss-softmax allocation; group floors and clean 0.50 never change.

- [ ] **Step 4: Run controller tests**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_control.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add causal_aug/causal_aug/cover_control.py causal_aug/tests/test_cover_control.py
git commit -m "feat: add COVER clean-retention controller"
```

---

### Task 4: Configurations and Single-Forward Policies

**Files:**
- Create: `lerobot_patches/cover_base/__init__.py`
- Create: `lerobot_patches/cover_base/configuration_cover_base.py`
- Create: `lerobot_patches/cover_base/modeling_cover_base.py`
- Create: `lerobot_patches/cover_base/processor_cover_base.py`
- Create: `lerobot_patches/cover_safe/__init__.py`
- Create: `lerobot_patches/cover_safe/configuration_cover_safe.py`
- Create: `lerobot_patches/cover_safe/modeling_cover_safe.py`
- Create: `lerobot_patches/cover_safe/processor_cover_safe.py`
- Create: `causal_aug/tests/test_cover_policy_contract.py`

**Interfaces:**
- Produces: `CoverBaseConfig`, registered type `cover_base`
- Produces: `CoverSafeConfig`, registered type `cover_safe`
- Produces: `CoverBasePolicy.forward(batch, noise=None, time=None, reduction="mean")`
- Produces: `CoverSafePolicy`, which enables clean safety without duplicating forward logic

- [ ] **Step 1: Write failing config-default and validation tests**

Assert every Global Constraint default serializes, invalid probability/floor
sums and controller ranges raise `ValueError`, and `CoverSafeConfig` adds only
the clean-controller parameters. Assert both default `device="mps"` configs can
round-trip through `to_dict()`.

- [ ] **Step 2: Implement dataclass configurations and processors**

Subclass `SmolVLAConfig`, register through `PreTrainedConfig.register_subclass`,
and copy the processor factory pattern from `online_dr` with only policy/config
names changed. `CoverSafeConfig` subclasses `CoverBaseConfig` and sets
`enable_clean_safety=True`; `CoverBaseConfig` sets it false.

- [ ] **Step 3: Write failing one-forward policy contract**

```python
def test_cover_policy_calls_model_once(monkeypatch, policy, batch):
    calls = 0
    original = policy.model.forward
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(policy.model, "forward", counted)
    loss, metrics = policy.forward(batch)
    assert calls == 1
    assert torch.isfinite(loss)
    assert metrics["cover/forward_count"] == 1.0
```

Also assert padded actions do not enter per-sample losses, all eight group metric
keys exist, group fractions sum to one, gradients reach policy parameters, and
neither policy overrides `select_action` or `predict_action_chunk`.

- [ ] **Step 4: Implement shared policy forward**

Follow `OnlineDRPolicy.forward`: prepare images/state/language/actions once,
sample group IDs, apply named interventions, call `self.model.forward` once,
mask padded actions, reduce to per-sample losses, update detached controllers,
compute detached importance weights, and return their normalized weighted mean.
For COVER-Safe, compute clean mean only from present finite clean samples and
update its controller before requesting the next cached allocation. Log all
metrics required by the design; absent group loss metrics use `0.0` plus a zero
fraction, never NaN.

- [ ] **Step 5: Verify state and inference contracts**

Add a checkpoint round-trip test for both controllers and assert two identical
policies in eval mode produce identical action chunks when controller buffers
differ. This proves controller state cannot affect inference.

- [ ] **Step 6: Run policy tests**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_cover_policy_contract.py causal_aug/tests/test_cover_control.py causal_aug/tests/test_cover_interventions.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lerobot_patches/cover_base lerobot_patches/cover_safe causal_aug/tests/test_cover_policy_contract.py
git commit -m "feat: implement single-forward COVER policies"
```

---

### Task 5: Installation and Registration

**Files:**
- Modify: `scripts/install_policy_patches.py`
- Create: `causal_aug/tests/test_cover_registration.py`

**Interfaces:**
- Consumes: `CoverBaseConfig`, `CoverSafeConfig`
- Produces CLI: `python scripts/install_policy_patches.py cover_base cover_safe`

- [ ] **Step 1: Write failing installer registration test**

Parse `POLICIES` and assert exact mappings:

```python
assert POLICIES["cover_base"] == "CoverBaseConfig"
assert POLICIES["cover_safe"] == "CoverSafeConfig"
```

- [ ] **Step 2: Add both installer mappings**

Do not add either policy to the `forward_with_latent` patch condition because
COVER uses standard SmolVLA forward only.

- [ ] **Step 3: Install into the active environment and test imports**

Run:

```bash
python -m pip install -e causal_aug
python scripts/install_policy_patches.py cover_base cover_safe
python - <<'PY'
from lerobot.policies.cover_base.configuration_cover_base import CoverBaseConfig
from lerobot.policies.cover_safe.configuration_cover_safe import CoverSafeConfig
assert CoverBaseConfig(device="mps", push_to_hub=False).type == "cover_base"
assert CoverSafeConfig(device="mps", push_to_hub=False).type == "cover_safe"
print("COVER registration: PASS")
PY
```

Expected: `COVER registration: PASS`.

- [ ] **Step 4: Run registration and regression tests**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/install_policy_patches.py causal_aug/tests/test_cover_registration.py
git commit -m "feat: register COVER policy variants"
```

---

### Task 6: Smoke, Pilot, and Revision-Pinned Evaluation Workflows

**Files:**
- Create: `scripts/smoke_cover_mps.sh`
- Create: `scripts/run_eval_cover.sh`
- Modify: `worklog/phase10.md`

**Interfaces:**
- Produces CLI: `./scripts/smoke_cover_mps.sh cover_base|cover_safe`
- Produces CLI: `./scripts/run_eval_cover.sh <cover_base|cover_safe> <level_0|level_1|level_2> <seed> [episodes]`

- [ ] **Step 1: Create two-step MPS smoke script**

Use dataset `lerobot/libero_spatial_image`, seed 1000, batch size 8, two steps,
`num_workers=0`, `env_eval_freq=0`, and separate output directories under
`outputs/smoke/<policy>_mps`. After training, assert `model.safetensors` is
non-empty, config type matches, and the log contains all eight group fractions,
`cover/forward_count:1.000`, finite loss, and no traceback/runtime error/NaN.

- [ ] **Step 2: Run both MPS smokes**

Run:

```bash
./scripts/smoke_cover_mps.sh cover_base
./scripts/smoke_cover_mps.sh cover_safe
```

Expected: both scripts print `PASS` and create checkpoint `000002`.

- [ ] **Step 3: Create revision-pinned evaluator**

Require `outputs/phase10/<policy>_revision.txt` containing an exact lowercase
40-character SHA. Use policy repos `phawitbinabik/causalvla-cover-base` and
`phawitbinabik/causalvla-cover-safe`, MPS, synchronous envs, batch size 2,
LIBERO Spatial, and output name
`outputs/eval/full/model_<policy>_<level>_<episodes>ep_seed<seed>`. Refuse to
overwrite a completed `eval_info.json`.

- [ ] **Step 4: Add exact 5K pilot and 25K full commands to Phase 10**

Both variants use Model F optimizer/scheduler/data settings and seed 1000. Pilot
repos end in `-pilot`; full training uses only the selected policy's final repo.
Document post-training config assertions, Hub SHA pinning, exposure-floor audit,
and the exact pilot/final evaluation commands.

- [ ] **Step 5: Run static verification**

Run:

```bash
bash -n scripts/smoke_cover_mps.sh scripts/run_eval_cover.sh
python -m compileall -q causal_aug/causal_aug lerobot_patches/cover_base lerobot_patches/cover_safe scripts
PYTHONPATH=causal_aug pytest -q causal_aug/tests
git diff --check
```

Expected: syntax, compile, tests, and whitespace checks pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_cover_mps.sh scripts/run_eval_cover.sh worklog/phase10.md
git commit -m "docs: add COVER pilot and evaluation workflow"
```

---

### Task 7: Final Verification and Handoff

**Files:**
- Modify: `worklog/phase10.md`

**Interfaces:**
- Produces: GPU-server install, CUDA smoke, L1/L2 pilot, selection, full-train, and evaluation handoff with exact commands.

- [ ] **Step 1: Run the complete local verification suite**

```bash
PYTHONPATH=causal_aug pytest -q causal_aug/tests
python -m compileall -q causal_aug/causal_aug lerobot_patches/cover_base lerobot_patches/cover_safe scripts
./scripts/smoke_cover_mps.sh cover_base
./scripts/smoke_cover_mps.sh cover_safe
git diff --check
```

Expected: all unit/contract tests pass, both smokes end after two steps with
finite losses and checkpoints, and no traceback, RuntimeError, OOM, or NaN is
present.

- [ ] **Step 2: Audit hard contracts from source and checkpoints**

Assert the policy source contains one `self.model.forward(` call, neither policy
defines inference methods, serialized configs contain every preregistered
default, controller buffers exist in `model.safetensors`, group floors sum to
one with clean fixed at 0.50, and both policies report one forward.

- [ ] **Step 3: Record measured evidence in Phase 10**

Write exact test count, smoke loss/metrics, elapsed time, output paths, and any
platform fallback warnings. Do not mark CUDA smoke or pilots complete until the
GPU-server output is supplied and verified.

- [ ] **Step 4: Commit final local evidence**

```bash
git add worklog/phase10.md
git commit -m "docs: mark COVER ready for CUDA smoke"
```

- [ ] **Step 5: Push only after user authorization**

Run `git status --short --branch` and `git log -5 --oneline`; preserve unrelated
files such as `.firecrawl/`. Push `main` only when the user explicitly requests
it.
