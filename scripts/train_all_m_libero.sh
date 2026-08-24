#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CAUSALVLA_PYTHON:-python}"

cd "$project_dir"
exec "$python_bin" "$project_dir/scripts/train_all_m_libero.py" "$@"
