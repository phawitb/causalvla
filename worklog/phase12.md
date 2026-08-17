# Phase 12: LIBERO Object — Online DR vs Paired Supervision

## Research question

Does CausalVLA-v2 counterfactual pairing outperform the strong Model F online
domain-randomization baseline when both policies are retrained on an
object-centric benchmark?

This is a cross-suite retraining experiment, not zero-shot transfer from
LIBERO Spatial.

## Preregistered fairness contract

| Dimension | Model F | CausalVLA-v2 |
|---|---|---|
| Dataset | \`lerobot/libero_object_image\` | same |
| Dataset revision | \`e1e080d7df1d0a359dff5c86c222e047549f447f\` | same |
| Training seed | 1000 | 1000 |
| Optimizer steps | 25,000 | 25,000 |
| Batch size | 16 | 16 |
| Backbone defaults | SmolVLA | SmolVLA |
| Online augmentation | 50% single branch | clean/augmented paired branches |
| Policy forwards/sample | 1 | 2 |

The comparison controls optimizer steps and sampled batch size as requested.
It does not claim equal FLOPs: V2 intentionally uses two policy forwards per
sample.

## Training

Install both policy patches on the GPU server:

\`\`\`bash
cd ~/projects/causalvla
conda activate causalvla
git pull origin main
python scripts/install_policy_patches.py online_dr causal_vla
\`\`\`

Run two-step CUDA smoke tests:

\`\`\`bash
OBJECT_STEPS=2 OBJECT_BATCH_SIZE=2 ./scripts/train_libero_object.sh f
OBJECT_STEPS=2 OBJECT_BATCH_SIZE=2 ./scripts/train_libero_object.sh v2
\`\`\`

Smoke outputs are isolated under \`outputs/phase12/smoke/\` and Hub upload is
disabled automatically. Full runs use \`outputs/phase12/final/\`:

\`\`\`bash
./scripts/train_libero_object.sh f
./scripts/train_libero_object.sh v2
\`\`\`

Public Hub repositories:

- \`phawitbinabik/causalvla-object-f-online-dr\`
- \`phawitbinabik/causalvla-object-v2\`

After upload, write the exact 40-character revisions locally:

\`\`\`text
outputs/phase12/f_revision.txt
outputs/phase12/v2_revision.txt
\`\`\`

## Evaluation

Primary protocol:

- suite: \`libero_object\`
- modes: Clean, Mild, Extreme
- seeds: 1000, 2000, 3000
- 10 tasks
- 10 episodes/task
- synchronous environments
- batch size 2
- 900 episodes/model and 1,800 episodes total

\`\`\`bash
./scripts/run_eval_object_matrix.sh f
./scripts/run_eval_object_matrix.sh v2
\`\`\`

## Decision rule

V2 is the primary winner only if its three-seed three-mode mean exceeds Model
F. Supporting requirements are Extreme OOD no lower than F and Clean no more
than 3 percentage points below F. Report mean and standard deviation across
seeds plus paired per-task differences; do not select a winner from seed 1000
alone.

## Status

- [x] Official dataset and immutable revision verified
- [x] Training workflow implemented
- [x] Evaluation workflow implemented
- [x] Three-seed matrix implemented
- [ ] CUDA smoke: Model F
- [ ] CUDA smoke: CausalVLA-v2
- [ ] Full training: Model F
- [ ] Full training: CausalVLA-v2
- [ ] Evaluation: 18 runs
