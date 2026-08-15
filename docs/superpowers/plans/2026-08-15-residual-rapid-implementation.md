# Residual RAPID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Model I (`residual_rapid`) so every augmented sample retains Model F broad domain randomization and a guarded 25% subset receives an additional risk-weighted overlay.

**Architecture:** Add a small, independently tested branch sampler to `causal_aug`, then add a LeRobot policy package that combines `CausalAugmenter`, `InterventionBank`, and the existing risk probabilities before one standard SmolVLA forward. Keep preprocessing and inference identical to SmolVLA and expose auditable branch metrics/configuration.

**Tech Stack:** Python 3.12, PyTorch, pytest, LeRobot/SmolVLA, Apple MPS smoke checks, CUDA training smoke on the remote GPU server.

## Global Constraints

- Training distribution is exactly 50% clean, 37.5% broad, and 12.5% broad-plus-risk in expectation.
- Broad coverage is exactly 50% and uses the same `CausalAugmenter(K=1, intensity=1.0)` distribution as Model F.
- Risk overlay probability is 0.25 conditional on broad augmentation.
- Risk profile revision is `phase8-3seed-256samples-robust-risk-v1`.
- Risk temperature is 1.0 and exploration floor is 0.10.
- The policy runs one SmolVLA forward and adds no loss or inference path.
- Existing `.firecrawl/` content is unrelated and must not be staged, edited, or removed.

---

### Task 1: Residual branch sampler

**Files:**
- Create: `causal_aug/tests/test_residual_sampler.py`
- Create: `causal_aug/causal_aug/residual_sampler.py`
- Modify: `causal_aug/causal_aug/__init__.py`

**Interfaces:**
- Produces: `ResidualBranchSampler(augmentation_probability: float, overlay_probability: float, risk_temperature: float = 1.0, exploration_floor: float = 0.10)`
- Produces: `ResidualBranchSampler.sample(batch_size: int, device: torch.device | str) -> tuple[Tensor, Tensor]`
- Return tensors: `branch` uses `0=clean, 1=broad, 2=residual`; `overlay_choices` contains the selected risk arm for every sample.

- [ ] **Step 1: Write failing probability-validation tests**

```python
import pytest

from causal_aug import ResidualBranchSampler


@pytest.mark.parametrize(
    ("augmentation_probability", "overlay_probability"),
    [(-0.1, 0.25), (1.1, 0.25), (0.5, -0.1), (0.5, 1.1)],
)
def test_rejects_invalid_probabilities(augmentation_probability, overlay_probability):
    with pytest.raises(ValueError):
        ResidualBranchSampler(augmentation_probability, overlay_probability)
```

- [ ] **Step 2: Run the validation test and verify RED**

Run:

```bash
PYTHONPATH="$PWD/causal_aug" /opt/miniconda3/envs/causalvla/bin/python \
  -m pytest causal_aug/tests/test_residual_sampler.py -q
```

Expected: collection/import failure because `ResidualBranchSampler` does not exist.

- [ ] **Step 3: Add failing branch-invariant and distribution tests**

```python
import torch


def test_residual_is_subset_of_augmented_and_distribution_is_correct():
    torch.manual_seed(1000)
    sampler = ResidualBranchSampler(0.5, 0.25)
    branch, choices = sampler.sample(100_000, "cpu")

    clean = branch == 0
    broad = branch == 1
    residual = branch == 2
    assert torch.equal(clean | broad | residual, torch.ones_like(clean))
    assert not torch.any((branch == 0) & residual)
    assert clean.float().mean().item() == pytest.approx(0.50, abs=0.01)
    assert broad.float().mean().item() == pytest.approx(0.375, abs=0.01)
    assert residual.float().mean().item() == pytest.approx(0.125, abs=0.01)
    assert choices.shape == (100_000,)
```

- [ ] **Step 4: Implement the minimal sampler**

Create `residual_sampler.py` with a PyTorch module that validates both
probabilities, owns `RiskWeightedInterventionSampler`, samples an augmented mask,
samples a conditional overlay mask, encodes the three branches, and returns
risk-arm choices from the existing risk probabilities.

- [ ] **Step 5: Export the sampler and verify GREEN**

Export `ResidualBranchSampler` from `causal_aug/__init__.py`, rerun the Task 1
test, and expect all tests to pass.

- [ ] **Step 6: Run the full augmentation test suite**

```bash
PYTHONPATH="$PWD/causal_aug" /opt/miniconda3/envs/causalvla/bin/python \
  -m pytest causal_aug/tests -q
```

Expected: all existing tests plus residual sampler tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add causal_aug/causal_aug/__init__.py \
  causal_aug/causal_aug/residual_sampler.py \
  causal_aug/tests/test_residual_sampler.py
git commit -m "feat: add Residual RAPID branch sampler"
```

---

### Task 2: Residual image compositor

**Files:**
- Modify: `causal_aug/tests/test_residual_sampler.py`
- Modify: `causal_aug/causal_aug/residual_sampler.py`

**Interfaces:**
- Produces: `ResidualBranchSampler.apply(images: list[Tensor], broad_images: list[Tensor]) -> tuple[list[Tensor], Tensor, Tensor]`
- Consumes: normalized camera tensors and broad-augmented camera tensors with matching shapes.

- [ ] **Step 1: Write a failing real-image behavior test**

Create two camera views and exercise two real sampler configurations: `(0.0,
0.0)` must preserve every clean pixel, while `(1.0, 1.0)` must produce residual
views for every sample. Assert:

```python
assert output[0].shape == images[0].shape
assert torch.equal(clean_output[0], images[0])
assert not torch.equal(residual_output[0], images[0])
assert torch.isfinite(output[0]).all()
assert output[0].min() >= -1 and output[0].max() <= 1
```

Use the real sampler and `InterventionBank`; do not mock randomness or image
transformations.

- [ ] **Step 2: Verify RED**

Run the new test directly. Expected failure: `ResidualBranchSampler` has no
`apply` method.

- [ ] **Step 3: Implement minimal composition**

The method must:

1. reject empty/mismatched camera lists;
2. call `sample` once;
3. use clean images for branch 0;
4. use broad images for branch 1;
5. apply the selected risk intervention to broad images for branch 2;
6. merge camera tensors with `torch.where` masks;
7. return mixed images, branch IDs, and risk choices.

- [ ] **Step 4: Verify GREEN and full suite**

Run the direct test, then all `causal_aug/tests`. Both must pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add causal_aug/causal_aug/residual_sampler.py \
  causal_aug/tests/test_residual_sampler.py
git commit -m "feat: compose broad and residual interventions"
```

---

### Task 3: LeRobot policy and configuration

**Files:**
- Create: `lerobot_patches/residual_rapid/__init__.py`
- Create: `lerobot_patches/residual_rapid/configuration_residual_rapid.py`
- Create: `lerobot_patches/residual_rapid/modeling_residual_rapid.py`
- Create: `lerobot_patches/residual_rapid/processor_residual_rapid.py`
- Create: `causal_aug/tests/test_residual_policy_contract.py`
- Modify: `scripts/install_policy_patches.py`
- Modify: `lerobot_patches/lerobot_causalvla.patch`

**Interfaces:**
- Produces: `ResidualRapidConfig(SmolVLAConfig)` registered as `residual_rapid`.
- Produces: `ResidualRapidPolicy(SmolVLAPolicy)` with one training forward.
- Produces: `make_residual_rapid_pre_post_processors(...)` matching SmolVLA.

- [ ] **Step 1: Write failing config contract tests**

The test imports the installed policy through LeRobot and asserts exact defaults
and invalid-value behavior:

```python
cfg = ResidualRapidConfig(device="mps", push_to_hub=False)
assert cfg.type == "residual_rapid"
assert cfg.augmentation_probability == 0.5
assert cfg.risk_overlay_probability == 0.25
assert cfg.broad_intensity == 1.0
assert cfg.profile_revision == "phase8-3seed-256samples-robust-risk-v1"
```

- [ ] **Step 2: Verify RED**

First confirm that `residual_rapid` is absent from `POLICIES`, then run the
contract test with `PYTHONPATH="$PWD/lerobot/src:$PWD/causal_aug"`. Expected
failure because the policy package/registration does not exist.

- [ ] **Step 3: Implement configuration and processor**

Add the exact fields from the design, validate their ranges in `__post_init__`,
and copy the standard SmolVLA processor structure used by `rapid_mix` while
renaming types/functions to `ResidualRapidConfig` and
`make_residual_rapid_pre_post_processors`.

- [ ] **Step 4: Implement the policy**

The policy must:

- create `CausalAugmenter(K=1, intensity=config.broad_intensity)`;
- create `ResidualBranchSampler(config.augmentation_probability,
  config.risk_overlay_probability, config.risk_temperature,
  config.exploration_floor)`;
- generate broad views before residual composition;
- run exactly one `self.model.forward(...)`;
- preserve padding-aware reductions from `OnlineDRPolicy`;
- report branch and residual-arm fractions only for reduced forwards.

- [ ] **Step 5: Register and install the policy**

Add `"residual_rapid": "ResidualRapidConfig"` to `POLICIES`, add the config
import/export to `lerobot_causalvla.patch`, and run:

```bash
/opt/miniconda3/envs/causalvla/bin/python scripts/install_policy_patches.py residual_rapid
```

- [ ] **Step 6: Verify config tests GREEN**

Run the direct policy contract test. Expected: all assertions pass.

- [ ] **Step 7: Add and run a source contract test**

Assert with `inspect.getsource(ResidualRapidPolicy.forward)` that the source
contains one occurrence of `self.model.forward(` and contains no
`forward_with_latent`, `loss_latent`, or `loss_action`.

- [ ] **Step 8: Run compilation and full tests**

```bash
/opt/miniconda3/envs/causalvla/bin/python -m py_compile \
  lerobot_patches/residual_rapid/*.py
PYTHONPATH="$PWD/causal_aug" /opt/miniconda3/envs/causalvla/bin/python \
  -m pytest causal_aug/tests -q
```

- [ ] **Step 9: Commit Task 3**

```bash
git add lerobot_patches/residual_rapid causal_aug/tests/test_residual_policy_contract.py \
  scripts/install_policy_patches.py lerobot_patches/lerobot_causalvla.patch
git commit -m "feat: add Residual RAPID policy"
```

---

### Task 4: Worklog and GPU handoff

**Files:**
- Modify: `worklog/phase8.md`

**Interfaces:**
- Worklog provides exact CUDA smoke, config verification, and full training commands.

- [ ] **Step 1: Install and verify local registration**

```bash
/opt/miniconda3/envs/causalvla/bin/python scripts/install_policy_patches.py residual_rapid
PYTHONPATH="$PWD/lerobot/src:$PWD/causal_aug" \
  /opt/miniconda3/envs/causalvla/bin/python - <<'PY'
from lerobot.policies import get_policy_class
from lerobot.policies.residual_rapid.configuration_residual_rapid import ResidualRapidConfig
cfg = ResidualRapidConfig(device="mps", push_to_hub=False)
assert get_policy_class("residual_rapid").__name__ == "ResidualRapidPolicy"
assert cfg.type == "residual_rapid"
print("Residual RAPID registration: PASS")
PY
```

- [ ] **Step 2: Add Phase 8 commands and decision gate**

Document a two-step CUDA smoke using batch size 2, then a 25K seed-1000 run with
repository `phawitbinabik/causalvla-residual-rapid`. Include expected long-run
metrics: clean `~0.50`, broad `~0.375`, residual `~0.125`, and residual-arm
fractions summing to `~0.125`.

- [ ] **Step 3: Run final verification**

```bash
PYTHONPATH="$PWD/causal_aug" /opt/miniconda3/envs/causalvla/bin/python \
  -m pytest causal_aug/tests -q
/opt/miniconda3/envs/causalvla/bin/python -m py_compile \
  causal_aug/causal_aug/residual_sampler.py \
  lerobot_patches/residual_rapid/*.py scripts/install_policy_patches.py
bash -n scripts/run_eval_mps.sh scripts/run_eval_gpu.sh
git diff --check
```

Expected: zero test failures, compilation success, shell syntax success, and no
whitespace errors.

- [ ] **Step 4: Commit and push Task 4**

```bash
git add worklog/phase8.md
git commit -m "docs: add Residual RAPID GPU workflow"
git push origin main
```

The GPU server then pulls `main`, installs `residual_rapid`, runs the documented
two-step CUDA smoke, verifies the serialized checkpoint, and starts full
training only after the smoke passes.
