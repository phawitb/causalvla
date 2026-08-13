#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
queue_log="$project_dir/logs/eval/all_models_queue.log"
status_file="$project_dir/outputs/eval/reports/queue_status.txt"
seed=1000
remaining_episodes=10

mkdir -p "$project_dir/logs/eval" "$project_dir/outputs/eval/reports"
cd "$project_dir"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
status() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$queue_log"
  printf '[%s] %s\n' "$(timestamp)" "$*" > "$status_file"
}

status "Evaluation queue started"

for model in a b c d e; do
  model_label="$(printf '%s' "$model" | tr '[:lower:]' '[:upper:]')"
  model_failed=0
  for level in level_0 level_1 level_2; do
    result_50="$project_dir/outputs/eval/full/model_${model}_${level}_50ep_seed${seed}/eval_info.json"
    result_10="$project_dir/outputs/eval/full/model_${model}_${level}_10ep_seed${seed}/eval_info.json"
    if [[ -s "$result_50" || -s "$result_10" ]]; then
      status "SKIP complete: model ${model_label} ${level}"
      /opt/miniconda3/envs/causalvla/bin/python scripts/summarize_eval.py >> "$queue_log" 2>&1
      continue
    fi

    succeeded=0
    for attempt in 1 2 3; do
      status "RUN model ${model_label} ${level}, attempt ${attempt}/3"
      if ./scripts/run_eval_mps.sh "$model" "$level" "$seed" "$remaining_episodes"; then
        succeeded=1
        status "DONE model ${model_label} ${level}"
        /opt/miniconda3/envs/causalvla/bin/python scripts/summarize_eval.py 2>&1 | tee -a "$queue_log"
        break
      fi
      status "FAILED model ${model_label} ${level}, attempt ${attempt}/3"
      sleep 10
    done
    if [[ "$succeeded" -ne 1 ]]; then
      model_failed=1
      status "GAVE UP after retries: model ${model_label} ${level}; continuing queue"
    fi
  done

  /opt/miniconda3/envs/causalvla/bin/python scripts/summarize_eval.py 2>&1 | tee -a "$queue_log"
  if [[ "$model_failed" -eq 0 ]]; then
    status "MODEL ${model_label} ALL LEVELS COMPLETE"
  else
    status "MODEL ${model_label} incomplete because at least one level failed"
  fi
done

/opt/miniconda3/envs/causalvla/bin/python scripts/summarize_eval.py 2>&1 | tee -a "$queue_log"
completed="$(find outputs/eval/full -mindepth 2 -maxdepth 2 -name eval_info.json -type f | wc -l | tr -d ' ')"
if [[ "$completed" == "15" ]]; then
  status "ALL 15 EVALUATION RUNS COMPLETE"
  exit 0
fi

status "QUEUE FINISHED WITH ${completed}/15 COMPLETE"
exit 1
