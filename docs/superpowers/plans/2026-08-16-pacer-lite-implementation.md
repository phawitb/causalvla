# PACER-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Model J, a two-forward SmolVLA training policy that selects one counterfactual intervention per sample from online policy feedback and automatically protects clean-task learning.

**Architecture:** `causal_aug` owns two stateful, independently tested modules: a contextual intervention bandit and a clean-safety controller. The `pacer_lite` LeRobot policy performs a clean and augmented `forward_with_latent` using shared noise/time, derives detached rewards, updates both modules, and exposes reproducible metrics; inference remains inherited SmolVLA behavior.

**Tech Stack:** Python 3.12, PyTorch, pytest, LeRobot/SmolVLA, MPS smoke testing, CUDA server training.

## Global Constraints

- Model J performs exactly two VLA forwards per training batch.
- Clean and augmented branches share the identical flow noise and time tensors.
- Adaptive state uses registered buffers and receives no gradient.
- Augmented loss weight remains within `[0.10, 0.50]` and clean weight is `1 - augmented_weight`.
- Intervention selection uses seven existing `InterventionBank` families and exploration floor `0.20`.
- Inference is unchanged SmolVLA and has no augmentation or extra forward.
- Implement behavior through strict RED-GREEN-REFACTOR cycles.
- Do not modify or commit the pre-existing untracked `.firecrawl/` directory.

---

### Task 1: Contextual Intervention Bandit

**Files:**
- Create: `causal_aug/causal_aug/adaptive_sampler.py`
- Modify: `causal_aug/causal_aug/__init__.py`
- Create: `causal_aug/tests/test_adaptive_sampler.py`

**Interfaces:**
- Produces: `PacerContextualBandit(temperature, exploration_floor, ema_decay, warmup_steps, families=INTERVENTION_FAMILIES)`.
- Produces: `assign_context(clean_losses: Tensor) -> Tensor`, returning integer context IDs `0=easy`, `1=medium`, `2=hard` by stable within-batch rank.
- Produces: `probabilities(contexts: Tensor) -> Tensor` with shape `[B, n_arms]`.
- Produces: `sample(clean_losses: Tensor) -> tuple[Tensor, Tensor]`, returning `(contexts, choices)`.
- Produces: `apply(images: list[Tensor], choices: Tensor, intensity: float) -> list[Tensor]`.
- Produces: `update(contexts: Tensor, choices: Tensor, rewards: Tensor) -> Tensor`, returning the number of rejected non-finite rewards.
- Registered buffers: `reward_ema[3,n_arms]`, `counts[3,n_arms]`, and scalar `steps`.

- [ ] **Step 1: Write failing validation and context tests**

```python
def test_context_assignment_is_balanced_by_rank():
    bandit = PacerContextualBandit(warmup_steps=0)
    contexts = bandit.assign_context(torch.tensor([9.0, 1.0, 5.0, 2.0, 8.0, 4.0]))
    assert contexts.tolist() == [2, 0, 1, 0, 2, 1]

@pytest.mark.parametrize("kwargs", [
    {"temperature": 0.0}, {"exploration_floor": -0.1},
    {"exploration_floor": 1.1}, {"ema_decay": 1.0},
    {"warmup_steps": -1},
])
def test_bandit_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PacerContextualBandit(**kwargs)
```

- [ ] **Step 2: Run RED test**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_adaptive_sampler.py`

Expected: FAIL because `PacerContextualBandit` does not exist.

- [ ] **Step 3: Implement configuration validation and rank contexts**

Use `torch.argsort(..., stable=True)` and assign bucket `min(2, rank * 3 // batch_size)`. Reject empty/non-1D loss tensors and non-finite losses.

- [ ] **Step 4: Add failing probability and warm-up tests**

```python
def test_warmup_probabilities_are_uniform():
    bandit = PacerContextualBandit(exploration_floor=0.2, warmup_steps=10)
    probs = bandit.probabilities(torch.tensor([0, 1, 2]))
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.shape[1]))

def test_exploration_floor_bounds_every_arm():
    bandit = PacerContextualBandit(exploration_floor=0.2, warmup_steps=0)
    bandit.reward_ema[0, 0] = 100.0
    probs = bandit.probabilities(torch.tensor([0]))[0]
    assert probs.sum().item() == pytest.approx(1.0)
    assert torch.all(probs >= 0.2 / probs.numel())
```

- [ ] **Step 5: Run RED test and implement probability calculation**

Exploit with `softmax(zscore(reward_ema[context]) / temperature)`, treating rows with no counts as uniform, then mix `(1-floor)*exploit + floor*uniform`. Warm-up is uniform while `steps < warmup_steps`.

- [ ] **Step 6: Add failing update, finite guard, checkpoint, and application tests**

```python
def test_update_is_context_and_arm_specific():
    bandit = PacerContextualBandit(ema_decay=0.5, warmup_steps=0)
    rejected = bandit.update(
        torch.tensor([0, 0, 2]), torch.tensor([1, 1, 3]),
        torch.tensor([2.0, 4.0, 8.0]),
    )
    assert rejected.item() == 0
    assert bandit.reward_ema[0, 1].item() == pytest.approx(3.0)
    assert bandit.reward_ema[2, 3].item() == pytest.approx(8.0)
    assert bandit.counts[0, 1].item() == 2

def test_nonfinite_reward_is_rejected():
    bandit = PacerContextualBandit(warmup_steps=0)
    rejected = bandit.update(torch.tensor([1]), torch.tensor([2]), torch.tensor([float("nan")]))
    assert rejected.item() == 1
    assert bandit.counts.sum().item() == 0

def test_state_dict_round_trip_preserves_adaptation():
    source = PacerContextualBandit(warmup_steps=0)
    source.update(torch.tensor([2]), torch.tensor([4]), torch.tensor([3.0]))
    target = PacerContextualBandit(warmup_steps=0)
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.reward_ema, source.reward_ema)
    assert torch.equal(target.counts, source.counts)
    assert torch.equal(target.steps, source.steps)
```

Application test uses two camera tensors and explicit choices covering at least two arms; assert shape, finite range `[-1,1]`, camera coherence, and unchanged source tensors. Invalid camera batches and out-of-range choices must raise `ValueError`.

- [ ] **Step 7: Run RED test and implement update/application**

Aggregate rewards per `(context, arm)` before a single EMA update. On the first observation set EMA to the group mean; afterward use `decay*old + (1-decay)*mean`. Increment `steps` once per batch update. Apply each arm only if present and merge with a broadcast sample mask.

- [ ] **Step 8: Run GREEN tests and commit**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_adaptive_sampler.py causal_aug/tests/test_intervention_bank.py`

Expected: PASS.

Commit: `feat: add PACER contextual intervention bandit`

---

### Task 2: Productive-Difficulty Reward and Clean-Safety Controller

**Files:**
- Create: `causal_aug/causal_aug/pacer_control.py`
- Modify: `causal_aug/causal_aug/__init__.py`
- Create: `causal_aug/tests/test_pacer_control.py`

**Interfaces:**
- Produces: `productive_difficulty_reward(clean_loss, augmented_loss, disagreement, max_loss_ratio, overhard_penalty, disagreement_clip) -> tuple[Tensor, Tensor]`; the second tensor is the per-sample loss ratio.
- Produces: `CleanSafetyController(max_weight, min_weight, tolerance, weight_decay, weight_recovery, fast_decay=0.9, slow_decay=0.99, warmup_steps=1000)`.
- Produces: `update(clean_loss: Tensor) -> tuple[Tensor, Tensor]`, returning detached `(augmented_weight, safety_trigger)`.
- Registered buffers: scalar `fast_ema`, `slow_ema`, `augmented_weight`, `steps`, and `initialized`.

- [ ] **Step 1: Write failing productive-difficulty tests**

```python
def test_reward_prefers_learnable_disagreement():
    clean = torch.tensor([1.0, 1.0])
    augmented = torch.tensor([1.5, 4.0])
    disagreement = torch.tensor([0.4, 0.4])
    reward, ratio = productive_difficulty_reward(clean, augmented, disagreement, 2.0, 2.0, 1.0)
    assert ratio.tolist() == pytest.approx([1.5, 4.0])
    assert reward[0] > reward[1]

def test_reward_is_finite_with_zero_clean_loss():
    reward, ratio = productive_difficulty_reward(
        torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.5]), 2.0, 2.0, 1.0
    )
    assert torch.isfinite(reward).all() and torch.isfinite(ratio).all()
```

- [ ] **Step 2: Run RED and implement detached reward**

Clamp clean denominator to `torch.finfo(dtype).eps`, clamp disagreement to `[0, disagreement_clip]`, apply the exponential over-hard penalty, and replace non-finite reward values with zero.

- [ ] **Step 3: Write failing safety decay, recovery, warm-up, and round-trip tests**

```python
def test_safety_controller_decays_and_recovers_weight():
    ctl = CleanSafetyController(warmup_steps=0, tolerance=0.05, weight_decay=0.5, weight_recovery=0.1)
    ctl.fast_ema.fill_(2.0); ctl.slow_ema.fill_(1.0); ctl.initialized.fill_(True)
    weight, triggered = ctl.update(torch.tensor(2.0))
    assert triggered.item() is True
    assert weight.item() == pytest.approx(0.25)
    ctl.fast_ema.fill_(1.0); ctl.slow_ema.fill_(1.0)
    weight, triggered = ctl.update(torch.tensor(1.0))
    assert triggered.item() is False
    assert weight.item() == pytest.approx(0.35)

def test_safety_weight_never_leaves_bounds():
    ctl = CleanSafetyController(min_weight=0.1, max_weight=0.5, warmup_steps=0)
    for _ in range(100):
        ctl.fast_ema.fill_(10.0); ctl.slow_ema.fill_(1.0); ctl.initialized.fill_(True)
        ctl.update(torch.tensor(10.0))
    assert ctl.augmented_weight.item() == pytest.approx(0.1)
```

- [ ] **Step 4: Run RED and implement controller**

Initialize both EMAs from the first finite scalar clean loss. Update EMAs before testing `fast > slow*(1+tolerance)`. Hold weight at `max_weight` during warm-up. Ignore non-finite input without changing buffers and return a false trigger.

- [ ] **Step 5: Run GREEN tests and commit**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_pacer_control.py`

Expected: PASS.

Commit: `feat: add PACER productive difficulty control`

---

### Task 3: PACER-Lite Policy Configuration and Registration

**Files:**
- Create: `lerobot_patches/pacer_lite/__init__.py`
- Create: `lerobot_patches/pacer_lite/configuration_pacer_lite.py`
- Create: `lerobot_patches/pacer_lite/processor_pacer_lite.py`
- Modify: `scripts/install_policy_patches.py`
- Create: `causal_aug/tests/test_pacer_policy_contract.py`

**Interfaces:**
- Produces: registered LeRobot policy type `pacer_lite` and `PacerLiteConfig`.
- Config fields match every default in the approved design, plus `disagreement_clip=1.0`, `fast_ema_decay=0.9`, and `slow_ema_decay=0.99`.
- Installer accepts `python scripts/install_policy_patches.py pacer_lite` and installs `forward_with_latent` for either `causal_vla` or `pacer_lite`.

- [ ] **Step 1: Write failing config and installer contract tests**

```python
def test_pacer_config_serializes_preregistered_defaults():
    cfg = PacerLiteConfig(device="mps", push_to_hub=False)
    assert cfg.type == "pacer_lite"
    assert cfg.aug_intensity == 1.0
    assert cfg.bandit_temperature == 1.0
    assert cfg.exploration_floor == 0.2
    assert cfg.bandit_warmup_steps == 1000
    assert cfg.max_augmented_weight == 0.5
    assert cfg.min_augmented_weight == 0.1

def test_installer_registers_pacer_and_latent_patch():
    source = Path("scripts/install_policy_patches.py").read_text()
    assert '"pacer_lite": "PacerLiteConfig"' in source
    assert 'if {"causal_vla", "pacer_lite"}' in source
```

Parameterize invalid values: negative intensity, non-positive temperature, exploration outside `[0,1]`, EMA decay outside `[0,1)`, negative warm-up, non-positive loss ratio/penalty/clip, weight bounds outside `[0,1]`, `min > max`, max weight above `0.5`, negative tolerance/recovery, and invalid fast/slow EMA decay.

- [ ] **Step 2: Run RED test**

Install location imports are expected to fail until files and registration exist.

- [ ] **Step 3: Implement config, standard processor, exports, and installer registration**

Copy the standard SmolVLA processor structure used by `rapid_lite`, changing only names and types. Update the latent-patch condition with set intersection:

```python
if {"causal_vla", "pacer_lite"}.intersection(args.policies):
```

- [ ] **Step 4: Install locally and run GREEN tests**

Run:

```bash
python scripts/install_policy_patches.py pacer_lite
PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_pacer_policy_contract.py
```

Expected: registration and config tests PASS.

- [ ] **Step 5: Commit**

Commit: `feat: register PACER-Lite policy configuration`

---

### Task 4: Two-Forward PACER-Lite Training Policy

**Files:**
- Create: `lerobot_patches/pacer_lite/modeling_pacer_lite.py`
- Modify: `lerobot_patches/pacer_lite/__init__.py`
- Modify: `causal_aug/tests/test_pacer_policy_contract.py`

**Interfaces:**
- Produces: `PacerLitePolicy(SmolVLAPolicy)` with `name="pacer_lite"`.
- Private helper `_per_sample_task_loss(losses, action_is_pad) -> Tensor[B]`.
- Private helper `_action_disagreement(clean_velocity, augmented_velocity, action_is_pad) -> Tensor[B]`.
- `forward(batch, noise=None, time=None, reduction="mean")` retains LeRobot's `(loss, info)` contract.

- [ ] **Step 1: Add failing source and helper contract tests**

```python
def test_policy_has_exactly_two_shared_target_forwards():
    source = inspect.getsource(PacerLitePolicy.forward)
    assert source.count("self.model.forward_with_latent(") == 2
    assert source.count("noise, time") >= 2
    assert "self.bandit.update" in source
    assert "self.safety.update" in source
    assert "loss_latent" not in source

def test_policy_inference_is_inherited():
    assert "select_action" not in PacerLitePolicy.__dict__
    assert "predict_action_chunk" not in PacerLitePolicy.__dict__
```

Helper tests use small tensors with padding to verify per-sample denominators and padded action steps excluded from disagreement.

- [ ] **Step 2: Run RED test**

Run: `PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_pacer_policy_contract.py`

Expected: FAIL because `PacerLitePolicy` does not exist.

- [ ] **Step 3: Implement minimal two-forward policy**

Implementation order inside `forward`:

```python
images, masks = self.prepare_images(batch)
state, actions = self.prepare_state(batch), self.prepare_action(batch)
noise = supplied_or_sampled_noise
time = supplied_or_sampled_time
clean_losses, _, clean_velocity = self.model.forward_with_latent(..., noise, time)
clean_per_sample = self._per_sample_task_loss(clean_losses, action_is_pad)
contexts, choices = self.bandit.sample(clean_per_sample.detach())
augmented_images = self.bandit.apply([x.detach() for x in images], choices, self.config.aug_intensity)
aug_losses, _, aug_velocity = self.model.forward_with_latent(..., noise, time)
aug_per_sample = self._per_sample_task_loss(aug_losses, action_is_pad)
disagreement = self._action_disagreement(clean_velocity, aug_velocity, action_is_pad)
reward, ratio = productive_difficulty_reward(...)
self.bandit.update(contexts, choices, reward)
aug_weight, safety_trigger = self.safety.update(clean_per_sample.mean())
loss = (1 - aug_weight) * clean_per_sample.mean() + aug_weight * aug_per_sample.mean()
```

Adaptive updates execute under `torch.no_grad()` and only when `self.training`.
For `reduction="none"`, return the weighted per-sample paired loss and the same
metrics without changing the LeRobot API.

- [ ] **Step 4: Add failing metric coverage test**

Assert source/info contract contains all required keys, including per-arm
`pacer/select/<family>` and `pacer/reward/<family>`. Use the seven
`INTERVENTION_FAMILIES` names rather than duplicating a second arm list.

- [ ] **Step 5: Implement metrics and reinstall policy**

Log scalar detached values only. Selection fractions are computed from current
choices; EMA rewards are the mean across three contexts per arm; rejected
updates are reported as `pacer/rejected_updates`.

- [ ] **Step 6: Run focused and full GREEN tests**

Run:

```bash
python scripts/install_policy_patches.py pacer_lite
PYTHONPATH=causal_aug pytest -q causal_aug/tests/test_adaptive_sampler.py \
  causal_aug/tests/test_pacer_control.py causal_aug/tests/test_pacer_policy_contract.py
PYTHONPATH=causal_aug pytest -q causal_aug/tests
python -m compileall -q causal_aug/causal_aug lerobot_patches/pacer_lite
```

Expected: all tests and compilation PASS.

- [ ] **Step 7: Commit**

Commit: `feat: implement two-forward PACER-Lite policy`

---

### Task 5: Mac MPS Smoke Test and Checkpoint Verification

**Files:**
- Create: `scripts/smoke_pacer_lite_mps.sh`
- Modify: `worklog/phase9.md`

**Interfaces:**
- Produces resumable local smoke output at `outputs/smoke/pacer_lite_mps`.
- Produces config verification for policy type and all preregistered defaults.

- [ ] **Step 1: Write smoke script with fail-fast checks**

The script installs editable `causal_aug`, installs the policy patch, runs two
training steps with batch size 2, MPS, seed 1000, no environment eval, and
checks `eval`-independent checkpoint serialization. It searches the log for
`Traceback|RuntimeError|nan` and fails if found.

- [ ] **Step 2: Run shell syntax check**

Run: `bash -n scripts/smoke_pacer_lite_mps.sh`

Expected: PASS.

- [ ] **Step 3: Run smoke test**

Run: `./scripts/smoke_pacer_lite_mps.sh`

Expected: two completed steps, finite paired losses, PACER metrics, checkpoint
with non-empty `model.safetensors`, and `End of training`.

- [ ] **Step 4: Record actual evidence and commit**

Append exact runtime, loss, arm selections, safety weight, checkpoint path, and
PASS/FAIL status to `worklog/phase9.md`.

Commit: `test: verify PACER-Lite MPS smoke`

---

### Task 6: GPU Workflow, Evaluation Runner, and Phase 9 Handoff

**Files:**
- Create: `scripts/run_eval_pacer.sh`
- Modify: `worklog/phase9.md`
- Modify: `pipeline.html`

**Interfaces:**
- GPU install command: `python scripts/install_policy_patches.py pacer_lite`.
- GPU smoke and full-train commands use the exact approved defaults.
- Eval runner accepts `<level_0|level_1|level_2> <seed> [episodes_per_task]`, pins the eventual Hugging Face revision, defaults to 10 episodes/task, and remains resume-safe.

- [ ] **Step 1: Add CUDA smoke and full-training workflow to Phase 9 notes**

Use batch size 2/steps 2 for smoke and batch size 8 initially for the
two-forward 25K run, increasing to 16 only if CUDA smoke proves memory headroom.
Training seed remains 1000. The log command must verify two forwards through
PACER metrics, not source inspection alone.

- [ ] **Step 2: Add resume-safe evaluation runner**

Mirror the established MPS/GPU runner contracts: fixed checkpoint revision,
same `eval_ood.py`, LIBERO Spatial, camera rename map, synchronous envs, explicit
seed, output named `model_pacer_lite_<level>_<episodes>ep_seed<seed>`, and skip
only when non-empty `eval_info.json` exists.

- [ ] **Step 3: Update website planning section**

Add Phase 9 status, the final Residual RAPID three-seed table, Model J data
flow, preregistered gates, and conditional Model K admission rule without
changing unrelated website sections.

- [ ] **Step 4: Verify documentation and scripts**

Run:

```bash
bash -n scripts/smoke_pacer_lite_mps.sh scripts/run_eval_pacer.sh
rg -n "PACER|Model J|Phase 9|58.3|two forwards" worklog/phase9.md pipeline.html
git diff --check
```

Expected: scripts parse, required documentation exists, and no whitespace errors.

- [ ] **Step 5: Run final regression verification**

Run:

```bash
PYTHONPATH=causal_aug pytest -q causal_aug/tests
python -m compileall -q causal_aug/causal_aug lerobot_patches/pacer_lite scripts
git status --short
```

Expected: all tests pass; only intended Phase 9 files are modified; `.firecrawl/`
remains untracked and untouched.

- [ ] **Step 6: Commit and push**

Commit: `docs: add PACER-Lite GPU and evaluation workflow`

Push `main` only after all verification evidence is recorded. The handoff must
include the pushed commit SHA, GPU commands, expected config assertion, and the
seed-1000 gate.

