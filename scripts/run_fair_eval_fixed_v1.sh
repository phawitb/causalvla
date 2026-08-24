#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:?model required: M0-clean, M1-offline-dr, M2-online-dr, M3-v2-warm, or all}"; shift
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 MUJOCO_GL=glfw
export PYTHONPATH="$project_dir/causal_aug:$project_dir/lerobot/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
args=(--protocol "$project_dir/configs/fair_v1.json")
if [[ "$model" != all ]]; then args+=(--model "$model"); fi
exec /opt/miniconda3/envs/causalvla/bin/python "$project_dir/scripts/eval_fair_v1_fixed.py" "${args[@]}" "$@"
