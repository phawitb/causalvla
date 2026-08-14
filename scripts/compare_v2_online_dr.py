#!/usr/bin/env python3
"""Compare CausalVLA-v2 and Online DR across completed evaluation seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "outputs" / "eval" / "full"
REPORT_PATH = PROJECT_DIR / "outputs" / "eval" / "reports" / "v2_vs_online_dr.md"
MODELS = {"v2": "CausalVLA-v2", "f": "Model F — Online DR"}
LEVELS = ("level_0", "level_1", "level_2")


def result_path(model: str, level: str, seed: int) -> Path:
    return RESULTS_DIR / f"model_{model}_{level}_10ep_seed{seed}" / "eval_info.json"


def load_score(model: str, level: str, seed: int) -> float | None:
    path = result_path(model, level, seed)
    if not path.is_file():
        return None
    return float(json.loads(path.read_text())["overall"]["pc_success"])


def format_summary(scores: list[float]) -> str:
    if not scores:
        return "—"
    if len(scores) == 1:
        return f"{scores[0]:.1f}% (n=1 seed)"
    return f"{statistics.mean(scores):.1f} ± {statistics.stdev(scores):.1f}% (n={len(scores)} seeds)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000, 2000, 3000])
    args = parser.parse_args()

    scores: dict[tuple[str, str], list[float]] = {}
    lines = [
        "# CausalVLA-v2 vs Online DR — Multi-seed Evaluation",
        "",
        "LIBERO Spatial, 10 tasks, 10 episodes/task/seed.",
        "",
        "| Seed | OOD level | CausalVLA-v2 | Model F — Online DR | F − V2 |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed in args.seeds:
        for level in LEVELS:
            v2 = load_score("v2", level, seed)
            online = load_score("f", level, seed)
            if v2 is not None:
                scores.setdefault(("v2", level), []).append(v2)
            if online is not None:
                scores.setdefault(("f", level), []).append(online)
            delta = None if v2 is None or online is None else online - v2
            lines.append(
                f"| {seed} | {level} | "
                f"{'—' if v2 is None else f'{v2:.1f}%'} | "
                f"{'—' if online is None else f'{online:.1f}%'} | "
                f"{'—' if delta is None else f'{delta:+.1f} pp'} |"
            )

    lines.extend(
        [
            "",
            "## Aggregate across completed seeds",
            "",
            "| Model | Clean | Mild OOD | Extreme OOD |",
            "|---|---:|---:|---:|",
        ]
    )
    for model, name in MODELS.items():
        values = [format_summary(scores.get((model, level), [])) for level in LEVELS]
        lines.append(f"| {name} | {' | '.join(values)} |")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
