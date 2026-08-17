# LIBERO Object F vs V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build reproducible training and evaluation workflows comparing Model F and CausalVLA-v2 on LIBERO Object.

**Architecture:** Training uses the same pinned dataset, backbone defaults, optimizer-step budget, batch size, and seed for both policies. Evaluation loads immutable Hugging Face revisions and runs the same LIBERO Object Clean/Mild/Extreme protocol for seeds 1000, 2000, and 3000.

**Tech Stack:** Bash, LeRobot, Hugging Face Hub, PyTorch/MPS, pytest.

## Global Constraints

- Dataset is \`lerobot/libero_object_image\` revision \`e1e080d7df1d0a359dff5c86c222e047549f447f\`.
- Both models train for 25,000 optimizer steps with batch size 16 and seed 1000.
- Evaluation uses 10 episodes/task, seeds 1000/2000/3000, synchronous environments, and batch size 2.
- Model revisions must be exact 40-character lowercase Git SHAs.

### Task 1: Training workflow

- [ ] Write a failing contract test.
- [ ] Implement F/V2 training dispatch.
- [ ] Run the test.

### Task 2: Evaluation workflow

- [ ] Write failing tests for exact revisions and the 3-seed matrix.
- [ ] Implement single-run and matrix entry points.
- [ ] Run focused and full tests.

### Task 3: Documentation and delivery

- [ ] Record the protocol in \`worklog/phase12.md\`.
- [ ] Verify shell syntax and tests.
- [ ] Commit and push.
