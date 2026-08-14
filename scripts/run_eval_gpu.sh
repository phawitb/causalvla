#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <model_id:f|v2|rapid> <level_0|level_1|level_2> <seed> [episodes_per_task]"
  exit 2
fi

model_id="$1"
ood_level="$2"
seed="$3"
episodes="${4:-10}"

if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "episodes_per_task must be a positive integer"
  exit 2
fi

case "$model_id" in
  f) repo_id="phawitbinabik/causalvla-model-f-online-dr"; revision="997d94a9325bc359422cd3cf54bd74b0a4c9be98" ;;
  v2) repo_id="phawitbinabik/causalvla-model-v2"; revision="6fc4104176b08ba7f9592583a8431c2e30b035ab" ;;
  rapid) repo_id="phawitbinabik/causalvla-rapid-lite"; revision="bad76c163d35e3254d976985f1f8a1f148672a2c" ;;
  *) echo "Unknown model: $model_id"; exit 2 ;;
esac

case "$ood_level" in
  level_0|level_1|level_2) ;;
  *) echo "Unknown OOD level: $ood_level"; exit 2 ;;
esac

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
run_name="model_${model_id}_${ood_level}_${episodes}ep_seed${seed}"
output_dir="$project_dir/outputs/eval/full/$run_name"
log_file="$project_dir/logs/eval/$run_name.log"

if [[ -s "$output_dir/eval_info.json" ]]; then
  echo "Already complete: $output_dir/eval_info.json"
  exit 0
fi

mkdir -p "$project_dir/logs/eval" "$output_dir"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/lerobot/src:$project_dir/causal_aug${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python "$project_dir/scripts/eval_ood.py" \
  --policy.path="$repo_id" \
  --policy.pretrained_revision="$revision" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  '--rename_map={"observation.images.image2":"observation.images.wrist_image"}' \
  --ood_level="$ood_level" \
  --eval.n_episodes="$episodes" \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --output_dir="$output_dir" \
  --seed="$seed" \
  2>&1 | tee "$log_file"

test -s "$output_dir/eval_info.json"
echo "Evaluation complete: $output_dir/eval_info.json"
