#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_dir/causal_aug:$project_dir/lerobot/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

exec /opt/miniconda3/envs/causalvla/bin/python "$project_dir/scripts/train_fair_v1.py" \
  --protocol "$project_dir/configs/fair_v1.json" "$@"
