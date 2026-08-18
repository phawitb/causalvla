#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
steps="${V2_WARM_STEPS:-25000}"
batch_size="${V2_WARM_BATCH_SIZE:-16}"
output_dir="${V2_WARM_OUTPUT_DIR:-$project_dir/outputs/phase13/v2_warm}"
device="${V2_WARM_DEVICE:-cuda}"
push_to_hub="true"
save_to_hub="true"

if [[ "$steps" != "25000" ]]; then
  push_to_hub="false"
  save_to_hub="false"
fi

command=(
  lerobot-train
  --policy.type=causal_vla_warm
  --policy.device="$device"
  --policy.push_to_hub="$push_to_hub"
  --policy.repo_id=phawitbinabik/causalvla-v2-warm
  --policy.n_counterfactual=1
  --policy.aug_intensity=1.0
  --policy.clean_task_weight=0.5
  --policy.augmented_task_weight=0.5
  --policy.use_action_loss=true
  --policy.lambda_action=0.05
  --policy.action_warmup_steps=10000
  --policy.use_latent_loss=false
  --policy.lambda_latent=0.0
  --policy.lambda_smooth=0.0
  --dataset.repo_id=lerobot/libero_spatial_image
  --output_dir="$output_dir"
  --job_name=causalvla_v2_warm
  --batch_size="$batch_size"
  --steps="$steps"
  --seed=1000
  --save_freq=5000
  --save_checkpoint_to_hub="$save_to_hub"
  --log_freq=100
  --num_workers=4
  --persistent_workers=true
  --env_eval_freq=0
)

if [[ "${V2_WARM_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

export PYTHONPATH="$project_dir/causal_aug${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$output_dir")" "$project_dir/logs"
"${command[@]}" 2>&1 | tee "$project_dir/logs/causalvla_v2_warm.log"
