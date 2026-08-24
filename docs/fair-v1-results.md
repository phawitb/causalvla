# Fair Protocol v1 Results

> Pilot feasibility result: evaluation seed 4000 only. This report does not establish statistical superiority.

## Fixed-episode augmentation results

Evaluation seed 4000 is complete for every trained Fair v1 model. Each episode
uses one deterministic augmentation record shared by every frame and camera
view. The record changes between episodes and is keyed by evaluation seed,
task, episode index, OOD level, and schema version.

| Model | Level 0 | Level 1 | Level 2 | Overall |
| --- | ---: | ---: | ---: | ---: |
| M0 — Clean | 57% | 48% | 26% | 43.7% |
| M2 — Online DR | 70% | 67% | 49% | 62.0% |
| M3 — V2-Warm | 66% | 61% | 58% | 61.7% |

Absolute change from the earlier frame-randomized seed-4000 run:

| Model | Level 0 | Level 1 | Level 2 | Overall |
| --- | ---: | ---: | ---: | ---: |
| M0 — Clean | -1 pp | +3 pp | -8 pp | -2.0 pp |
| M2 — Online DR | 0 pp | -5 pp | -2 pp | -2.3 pp |
| M3 — V2-Warm | +1 pp | -5 pp | +1 pp | -1.0 pp |

Each cell contains 100 episodes (10 tasks × 10 episodes), for 300 episodes per
model and 900 episodes total. Results are stored separately under
`outputs/eval/fair-v1-fixed/full` and do not replace the earlier frame-randomized
evaluation.

At seed 4000, M2 has the highest aggregate success rate, while M3 is strongest
at OOD level 2. Additional seeds 5000 and 6000 are still required before making
statistical claims.
