#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <cover_base|cover_safe|cover_base_pilot|cover_safe_pilot|cover_base_20k> <level_0|level_1|level_2> <seed> [episodes_per_task]" >&2
  exit 2
fi

policy="$1"
ood_level="$2"
seed="$3"
episodes="${4:-10}"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
revision_file="$project_dir/outputs/phase10/${policy}_revision.txt"

case "$policy" in
  cover_base) repo="phawitbinabik/causalvla-cover-base" ;;
  cover_safe) repo="phawitbinabik/causalvla-cover-safe" ;;
  cover_base_pilot) repo="phawitbinabik/causalvla-cover-base-pilot" ;;
  cover_safe_pilot) repo="phawitbinabik/causalvla-cover-safe-pilot" ;;
  cover_base_20k) repo="phawitbinabik/causalvla-cover-base-20k" ;;
  *) echo "Unknown policy: $policy" >&2; exit 2 ;;
esac
case "$ood_level" in
  level_0|level_1|level_2) ;;
  *) echo "Unknown OOD level: $ood_level" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "seed must be non-negative" >&2; exit 2; }
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || { echo "episodes must be positive" >&2; exit 2; }
[[ -s "$revision_file" ]] || { echo "Missing revision: $revision_file" >&2; exit 2; }
revision="$(tr -d '[:space:]' < "$revision_file")"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "Revision must match ^[0-9a-f]{40}$" >&2; exit 2; }

run_name="model_${policy}_${ood_level}_${episodes}ep_seed${seed}"
output_dir="$project_dir/outputs/eval/full/$run_name"
log_file="$project_dir/logs/eval/$run_name.log"
if [[ -s "$output_dir/eval_info.json" ]]; then
  echo "Already complete: $output_dir/eval_info.json"
  exit 0
fi

mkdir -p "$project_dir/logs/eval" "$output_dir"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/causal_aug${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MUJOCO_GL=glfw

/opt/miniconda3/envs/causalvla/bin/python "$project_dir/scripts/eval_ood.py" \
  --policy.path="$repo" \
  --policy.pretrained_revision="$revision" \
  --policy.device=mps \
  --env.type=libero \
  --env.task=libero_spatial \
  '--rename_map={"observation.images.image2":"observation.images.wrist_image"}' \
  --ood_level="$ood_level" \
  --eval.n_episodes="$episodes" \
  --eval.batch_size=2 \
  --eval.use_async_envs=false \
  --output_dir="$output_dir" \
  --seed="$seed" \
  2>&1 | tee "$log_file"

test -s "$output_dir/eval_info.json"
echo "Evaluation complete: $output_dir/eval_info.json"
