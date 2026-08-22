#!/usr/bin/env python3
"""Run and validate the four Fair v1 one-step MPS smoke jobs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from scripts.fair_protocol import MODEL_IDS


REQUIRED_TRUE = ("parameter_changed", "checkpoint_saved", "reload_inference", "batch_contract")


def validate_smoke_artifacts(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    run_path = output_dir / "run_manifest.json"
    metrics_path = output_dir / "smoke_metrics.json"
    if not run_path.is_file() or not metrics_path.is_file():
        raise ValueError("smoke acceptance failed: required artifact is missing")
    run = json.loads(run_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    failures = []
    if run.get("status") != "completed":
        failures.append("run status")
    loss = metrics.get("loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        failures.append("finite loss")
    failures.extend(field for field in REQUIRED_TRUE if metrics.get(field) is not True)
    if failures:
        raise ValueError(f"smoke acceptance failed: {', '.join(failures)}")
    return metrics


def run_smoke(model_id: str, protocol_path: Path, output_root: Path) -> dict:
    if model_id == "M1-offline-dr":
        dataset_root = output_root / "m1-dataset"
        if not (dataset_root / "fair_v1_metadata.json").is_file():
            subprocess.run(
                [
                    "/opt/miniconda3/envs/causalvla/bin/python",
                    str(protocol_path.parents[1] / "scripts/materialize_fair_offline.py"),
                    "--protocol", str(protocol_path),
                    "--output-root", str(dataset_root),
                    "--records-out", str(output_root / "m1-records.jsonl"),
                    "--max-episodes", "1",
                ],
                check=True,
            )
    output_dir = output_root / model_id
    if (output_dir / "run_manifest.json").is_file():
        return validate_smoke_artifacts(output_dir)
    command = [
        str(protocol_path.parents[1] / "scripts" / "run_fair_v1.sh"),
        model_id,
        "--mode",
        "smoke",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)
    return validate_smoke_artifacts(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/fair_v1.json"))
    parser.add_argument("--model", action="append", choices=MODEL_IDS)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/smoke/fair-v1"))
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    models = args.model or list(MODEL_IDS)
    summary = {model: run_smoke(model, protocol_path, output_root) for model in models}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
