#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/opt/miniconda3/envs/causalvla/bin/python"
train_bin="/opt/miniconda3/envs/causalvla/bin/lerobot-train"
output_dir="$project_dir/outputs/smoke/pacer_lite_mps"
log_file="$project_dir/logs/pacer_lite_mps_smoke.log"

mkdir -p "$project_dir/logs"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/causal_aug${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

"$python_bin" "$project_dir/scripts/install_policy_patches.py" pacer_lite

"$train_bin" \
  --policy.type=pacer_lite \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir="$output_dir" \
  --job_name=pacer_lite_mps_smoke \
  --batch_size=2 \
  --steps=2 \
  --seed=1000 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --env_eval_freq=0 \
  2>&1 | tee "$log_file"

if grep -Eiq 'Traceback|RuntimeError|CUDA out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$log_file"; then
  echo "PACER-Lite MPS smoke log contains an error or NaN" >&2
  exit 1
fi

"$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "checkpoints/000002/pretrained_model"
config = json.loads((root / "config.json").read_text())
expected = {
    "type": "pacer_lite",
    "aug_intensity": 1.0,
    "bandit_temperature": 1.0,
    "exploration_floor": 0.2,
    "bandit_ema_decay": 0.95,
    "bandit_warmup_steps": 1000,
    "max_loss_ratio": 2.0,
    "overhard_penalty": 2.0,
    "disagreement_clip": 1.0,
    "max_augmented_weight": 0.5,
    "min_augmented_weight": 0.1,
    "clean_tolerance": 0.05,
    "clean_weight_decay": 0.9,
    "clean_weight_recovery": 0.01,
    "fast_ema_decay": 0.9,
    "slow_ema_decay": 0.99,
}
assert root.joinpath("model.safetensors").stat().st_size > 0
assert all(config.get(key) == value for key, value in expected.items())
print("PACER-Lite MPS smoke: PASS")
PY
