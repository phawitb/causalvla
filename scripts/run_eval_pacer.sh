#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <level_0|level_1|level_2> <seed> [episodes_per_task]" >&2
  exit 2
fi

ood_level="$1"
seed="$2"
episodes="${3:-10}"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
revision_file="$project_dir/outputs/phase9/pacer_lite_revision.txt"

case "$ood_level" in
  level_0|level_1|level_2) ;;
  *) echo "Unknown OOD level: $ood_level" >&2; exit 2 ;;
esac
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "episodes_per_task must be a positive integer" >&2
  exit 2
fi
if [[ ! -s "$revision_file" ]]; then
  echo "Missing pinned model revision: $revision_file" >&2
  echo "Write the exact Hugging Face commit SHA after training and upload." >&2
  exit 2
fi

revision="$(tr -d '[:space:]' < "$revision_file")"
if ! [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Pinned revision must be a 40-character lowercase Git SHA" >&2
  exit 2
fi

run_name="model_pacer_lite_${ood_level}_${episodes}ep_seed${seed}"
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/opt/miniconda3/envs/causalvla/bin/python "$project_dir/scripts/eval_ood.py" \
  --policy.path=phawitbinabik/causalvla-pacer-lite \
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
