#!/usr/bin/env python3
"""Summarize Fair v1 paired pilot evaluations without overstating one seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def summarize_runs(paths: Sequence[Path]) -> dict:
    raw: dict[str, dict[str, dict]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        model_id = path.parents[2].name
        level = path.parents[1].name
        payload = json.loads(path.read_text())
        successes = [
            bool(value)
            for task in payload.get("per_task", [])
            for value in task.get("metrics", {}).get("successes", [])
        ]
        raw.setdefault(model_id, {})[level] = {
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "episodes": len(successes),
            "inference_wall_seconds": payload.get("runtime", {}).get("wall_seconds"),
        }
    baseline = raw.get("M0-clean", {})
    for model_id, levels in raw.items():
        clean_rate = levels.get("level_0", {}).get("success_rate")
        for level, cell in levels.items():
            cell["degradation_from_level_0"] = (
                cell["success_rate"] - clean_rate if clean_rate is not None else None
            )
            base_rate = baseline.get(level, {}).get("success_rate")
            cell["delta_vs_m0"] = cell["success_rate"] - base_rate if base_rate is not None else None
    return raw


def render_markdown(summary: dict, missing: list[str]) -> str:
    lines = [
        "# Fair Protocol v1 Results",
        "",
        "> Pilot feasibility result: evaluation seed 4000 only. This report does not establish statistical superiority.",
        "",
    ]
    if not summary:
        lines.extend(["## Results pending", "", "GPU training and pinned seed-4000 evaluation have not completed."])
        return "\n".join(lines) + "\n"
    lines.extend(["| Model | Level | Success | Δ from clean | Δ vs M0 | Episodes |", "|---|---|---:|---:|---:|---:|"])
    for model_id, levels in summary.items():
        for level, cell in sorted(levels.items()):
            degradation = cell["degradation_from_level_0"]
            delta = cell["delta_vs_m0"]
            lines.append(
                f"| {model_id} | {level} | {cell['success_rate']:.1%} | "
                f"{'—' if degradation is None else f'{degradation:+.1%}'} | "
                f"{'—' if delta is None else f'{delta:+.1%}'} | {cell['episodes']} |"
            )
    if missing:
        lines.extend(["", "## Missing cells", "", *[f"- {item}" for item in missing]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path("outputs/eval/fair-v1/full"))
    parser.add_argument("--output", type=Path, default=Path("docs/fair-v1-results.md"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.eval_root.glob("*/level_*/seed4000/eval_info.json"))
    summary = summarize_runs(paths)
    expected = {f"{model}/{level}" for model in ("M0-clean", "M1-offline-dr", "M2-online-dr", "M3-v2-warm") for level in ("level_0", "level_1", "level_2")}
    actual = {f"{path.parents[2].name}/{path.parents[1].name}" for path in paths}
    missing = sorted(expected - actual)
    if paths and missing and not args.allow_partial:
        raise SystemExit("incomplete evaluation matrix: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary, missing if paths else []))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
