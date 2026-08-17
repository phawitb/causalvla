#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <f|v2> <level_0|level_1|level_2> <seed> [episodes_per_task]" >&2
  exit 2
fi

model="$1"
ood_level="$2"
seed="$3"
episodes="${4:-10}"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
revision_file="$project_dir/outputs/phase12/${model}_revision.txt"

case "$model" in
  f) repo="phawitbinabik/causalvla-object-f-online-dr" ;;
  v2) repo="phawitbinabik/causalvla-object-v2" ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
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

run_name="object_${model}_${ood_level}_${episodes}ep_seed${seed}"
output_dir="$project_dir/outputs/phase12/eval/$run_name"
log_file="$project_dir/logs/phase12/eval_${run_name}.log"
if [[ -s "$output_dir/eval_info.json" ]]; then
  echo "Already complete: $output_dir/eval_info.json"
  exit 0
fi

mkdir -p "$project_dir/logs/phase12" "$output_dir"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/lerobot/src:$project_dir/causal_aug"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MUJOCO_GL=glfw
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/opt/miniconda3/envs/causalvla/bin/python "$project_dir/scripts/eval_ood.py" \
  --policy.path="$repo" \
  --policy.pretrained_revision="$revision" \
  --policy.device=mps \
  --env.type=libero \
  --env.task=libero_object \
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
