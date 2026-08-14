#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <model_id:a|b|c|d|e|f|v2> <level_0|level_1|level_2> <seed> [episodes_per_task]"
  exit 2
fi

model_id="$1"
ood_level="$2"
seed="$3"
episodes="${4:-50}"

if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "episodes_per_task must be a positive integer"
  exit 2
fi

case "$model_id" in
  a) repo_id="phawitbinabik/causalvla-model-a-sft"; revision="7a34ed04c4f8b2b43514cfbe1b4468c1c87c7a7d" ;;
  b) repo_id="phawitbinabik/causalvla-model-b-dr"; revision="9161f4ee7c9276bdd18f4c6214201324c4c1400f" ;;
  c) repo_id="phawitbinabik/causalvla-model-c-ours"; revision="82d1d7338d2b150061dd0a82a016c652c94ec45b" ;;
  d) repo_id="phawitbinabik/causalvla-model-d-no-latent"; revision="ec27cb968af93239da74188ffce7c6ebeec0b05c" ;;
  e) repo_id="phawitbinabik/causalvla-model-e-no-action"; revision="f506bfa3a5d9b678b49497f0af95d15855d8614d" ;;
  f) repo_id="phawitbinabik/causalvla-model-f-online-dr"; revision="997d94a9325bc359422cd3cf54bd74b0a4c9be98" ;;
  v2) repo_id="phawitbinabik/causalvla-model-v2"; revision="6fc4104176b08ba7f9592583a8431c2e30b035ab" ;;
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
export PYTHONPATH="$project_dir/lerobot/src:$project_dir/causal_aug"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MUJOCO_GL=glfw
export CUBLAS_WORKSPACE_CONFIG=:4096:8

/opt/miniconda3/envs/causalvla/bin/python "$project_dir/scripts/eval_ood.py" \
  --policy.path="$repo_id" \
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
