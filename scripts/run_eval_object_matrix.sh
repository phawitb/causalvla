#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <f|v2>" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
model="$1"

for seed in 1000 2000 3000; do
  for level in level_0 level_1 level_2; do
    "$project_dir/scripts/run_eval_object.sh" "$model" "$level" "$seed" 10
  done
done
