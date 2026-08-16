#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cover_base|cover_safe>" >&2
  exit 2
fi
policy="$1"
case "$policy" in
  cover_base|cover_safe) ;;
  *) echo "Unknown policy: $policy (expected cover_base|cover_safe)" >&2; exit 2 ;;
esac

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/opt/miniconda3/envs/causalvla/bin/python"
train_bin="/opt/miniconda3/envs/causalvla/bin/lerobot-train"
output_dir="$project_dir/outputs/smoke/${policy}_mps"
log_file="$project_dir/logs/${policy}_mps_smoke.log"

mkdir -p "$project_dir/logs"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_dir/causal_aug${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

"$python_bin" "$project_dir/scripts/install_policy_patches.py" "$policy"

"$train_bin" \
  --policy.type="$policy" \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --output_dir="$output_dir" \
  --job_name="${policy}_mps_smoke" \
  --batch_size=8 \
  --steps=2 \
  --seed=1000 \
  --save_freq=2 \
  --log_freq=1 \
  --num_workers=0 \
  --env_eval_freq=0 \
  2>&1 | tee "$log_file"

if grep -Eiq 'Traceback|RuntimeError|CUDA out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$log_file"; then
  echo "COVER MPS smoke contains Traceback|RuntimeError|CUDA out of memory|nan" >&2
  exit 1
fi
grep -q 'cover/forward_count:1.000' "$log_file"

"$python_bin" - "$output_dir" "$policy" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "checkpoints/000002/pretrained_model"
policy = sys.argv[2]
config = json.loads((root / "config.json").read_text())
expected = {
    "type": policy,
    "aug_intensity": 1.0,
    "cover_ema_decay": 0.95,
    "cover_warmup_steps": 1000,
    "cover_temperature": 0.5,
    "cover_update_interval": 100,
    "cover_weight_min": 0.5,
    "cover_weight_max": 2.0,
    "enable_clean_safety": policy == "cover_safe",
}
assert root.joinpath("model.safetensors").stat().st_size > 0
assert all(config.get(key) == value for key, value in expected.items())
print(f"{policy} MPS smoke: PASS")
PY
