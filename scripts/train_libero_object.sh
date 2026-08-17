#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <f|v2>" >&2
  exit 2
fi

model="$1"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
dataset_repo="lerobot/libero_object_image"
dataset_revision="e1e080d7df1d0a359dff5c86c222e047549f447f"
batch_size="${OBJECT_BATCH_SIZE:-16}"
steps="${OBJECT_STEPS:-25000}"
push_to_hub="true"
save_checkpoint_to_hub="true"
run_root="final"
if [[ "$steps" != "25000" ]]; then
  push_to_hub="false"
  save_checkpoint_to_hub="false"
  run_root="smoke"
fi

case "$model" in
  f)
    policy_type="online_dr"
    repo_id="phawitbinabik/causalvla-object-f-online-dr"
    output_dir="$project_dir/outputs/phase12/$run_root/object_f"
    extra_args=(--policy.aug_probability=0.5 --policy.aug_intensity=1.0)
    ;;
  v2)
    policy_type="causal_vla"
    repo_id="phawitbinabik/causalvla-object-v2"
    output_dir="$project_dir/outputs/phase12/$run_root/object_v2"
    extra_args=(
      --policy.n_counterfactual=1
      --policy.aug_intensity=1.0
      --policy.clean_task_weight=0.5
      --policy.augmented_task_weight=0.5
      --policy.use_latent_loss=false
      --policy.use_action_loss=false
      --policy.lambda_latent=0.0
      --policy.lambda_action=0.0
      --policy.lambda_smooth=0.0
    )
    ;;
  *)
    echo "Unknown model: $model" >&2
    exit 2
    ;;
esac

mkdir -p "$project_dir/logs/phase12"
cd "$project_dir"

lerobot-train \
  --policy.type="$policy_type" \
  --policy.device=cuda \
  --policy.push_to_hub="$push_to_hub" \
  --policy.repo_id="$repo_id" \
  --policy.private=false \
  --dataset.repo_id="$dataset_repo" \
  --dataset.revision="$dataset_revision" \
  --output_dir="$output_dir" \
  --job_name="object_${model}" \
  --batch_size="$batch_size" \
  --steps="$steps" \
  --seed=1000 \
  --save_freq=5000 \
  --save_checkpoint_to_hub="$save_checkpoint_to_hub" \
  --log_freq=100 \
  --num_workers=4 \
  --persistent_workers=true \
  --env_eval_freq=0 \
  "${extra_args[@]}" \
  2>&1 | tee "$project_dir/logs/phase12/object_${model}.log"
