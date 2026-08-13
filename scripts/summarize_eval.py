#!/usr/bin/env python3
"""Build compact, reproducible summaries from completed full evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "outputs" / "eval" / "full"
REPORT_DIR = PROJECT_DIR / "outputs" / "eval" / "reports"
MODELS = "abcde"
LEVELS = ("level_0", "level_1", "level_2")
SEED = 1000


def collect_rows() -> list[dict[str, object]]:
    rows = []
    for model in MODELS:
        for level in LEVELS:
            candidates = [
                RESULTS_DIR / f"model_{model}_{level}_50ep_seed{SEED}" / "eval_info.json",
                RESULTS_DIR / f"model_{model}_{level}_10ep_seed{SEED}" / "eval_info.json",
            ]
            result_path = next((path for path in candidates if path.is_file()), None)
            if result_path is None:
                continue
            result = json.loads(result_path.read_text())
            overall = result["overall"]
            per_task = result.get("per_task", [])
            task_rates = {
                str(item["task_id"]): 100.0
                * sum(item["metrics"]["successes"])
                / len(item["metrics"]["successes"])
                for item in per_task
            }
            rows.append(
                {
                    "model": model.upper(),
                    "level": level,
                    "success_rate_pct": float(overall["pc_success"]),
                    "successful_episodes": round(
                        float(overall["pc_success"]) * int(overall["n_episodes"]) / 100
                    ),
                    "episodes": int(overall["n_episodes"]),
                    "eval_seconds": float(overall["eval_s"]),
                    "task_success_rates_pct": json.dumps(task_rates, sort_keys=True),
                    "result_path": str(result_path),
                }
            )
    return rows


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    fieldnames = list(rows[0]) if rows else [
        "model", "level", "success_rate_pct", "successful_episodes",
        "episodes", "eval_seconds", "task_success_rates_pct", "result_path",
    ]
    csv_path = REPORT_DIR / "eval_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["model"], row["level"]): row for row in rows}
    lines = [
        "# Full Evaluation Summary",
        "",
        "LIBERO Spatial, 10 tasks, seed 1000. Completed 50-episode/task results are retained; remaining runs use 10 episodes/task.",
        "",
        "| Model | Clean (level 0) | Mild OOD (level 1) | Strong OOD (level 2) | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for model in MODELS.upper():
        values = []
        complete = True
        for level in LEVELS:
            row = by_key.get((model, level))
            if row:
                values.append(f'{row["success_rate_pct"]:.1f}%')
            else:
                values.append("—")
                complete = False
        lines.append(f"| {model} | {' | '.join(values)} | {'COMPLETE' if complete else 'RUNNING/PENDING'} |")
    lines.extend(["", f"Completed runs: **{len(rows)}/15**", ""])
    (REPORT_DIR / "eval_summary.md").write_text("\n".join(lines))

    print(f"Completed runs: {len(rows)}/15")
    for row in rows:
        print(
            f'Model {row["model"]} {row["level"]}: '
            f'{row["success_rate_pct"]:.1f}% '
            f'({row["successful_episodes"]}/{row["episodes"]})'
        )


if __name__ == "__main__":
    main()
