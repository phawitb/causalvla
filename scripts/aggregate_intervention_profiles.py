#!/usr/bin/env python3
"""Aggregate multi-seed profiler outputs into a robust RAPID-Lite curriculum."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate(paths: list[Path], min_guard_rate: float, uncertainty_penalty: float, top_k: int):
    grouped = defaultdict(list)
    for path in paths:
        for row in json.loads(path.read_text()):
            grouped[(row["family"], float(row["intensity"]))].append(row)

    expected_seeds = len(paths)
    rows = []
    for (family, intensity), items in grouped.items():
        if len(items) != expected_seeds:
            raise ValueError(f"{family}:{intensity} appears in {len(items)}/{expected_seeds} profiles")
        sensitivities = [float(item["action_sensitivity"]) for item in items]
        mean = statistics.fmean(sensitivities)
        std = statistics.stdev(sensitivities) if len(sensitivities) > 1 else 0.0
        guard = statistics.fmean(float(item["guard_pass_rate"]) for item in items)
        rows.append({
            "family": family,
            "intensity": intensity,
            "mean_action_sensitivity": mean,
            "std_action_sensitivity": std,
            "mean_guard_pass_rate": guard,
            "robust_risk": mean - uncertainty_penalty * std,
            "eligible": guard >= min_guard_rate,
        })
    rows.sort(key=lambda row: row["robust_risk"], reverse=True)
    candidates = [row for row in rows if row["eligible"]][:top_k]
    if len(candidates) < top_k:
        raise ValueError(f"Only {len(candidates)} candidates pass guard; requested top_k={top_k}")
    total = sum(max(row["robust_risk"], 0.0) for row in candidates)
    if total <= 0:
        raise ValueError("Eligible robust-risk scores are not positive")
    for row in candidates:
        row["risk_weight"] = max(row["robust_risk"], 0.0) / total
    return {"profiles": [str(path) for path in paths], "ranking": rows, "candidates": candidates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path, help="Per-seed summary.json files")
    parser.add_argument("--min-guard-rate", type=float, default=0.95)
    parser.add_argument("--uncertainty-penalty", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase8/rapid_lite_profile.json"))
    args = parser.parse_args()
    if not 0 <= args.min_guard_rate <= 1:
        parser.error("--min-guard-rate must be in [0, 1]")
    if args.uncertainty_penalty < 0 or args.top_k <= 0:
        parser.error("uncertainty penalty must be non-negative and top-k positive")
    result = aggregate(args.profiles, args.min_guard_rate, args.uncertainty_penalty, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["candidates"], indent=2))


if __name__ == "__main__":
    main()
