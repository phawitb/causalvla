#!/usr/bin/env python3
"""Train one Fair Protocol v1 model with drift and overwrite protection."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from scripts.fair_protocol import (
    MODEL_IDS,
    build_train_command,
    finish_run_manifest,
    load_protocol,
    start_run_manifest,
    validate_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=MODEL_IDS)
    parser.add_argument("--protocol", type=Path, default=Path("configs/fair_v1.json"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    validate_protocol(protocol, protocol_path)
    if args.mode == "full" and args.model_id == "M1-offline-dr":
        revision = os.environ.get("FAIR_V1_OFFLINE_REVISION")
        if revision:
            protocol["offline_dataset"]["revision"] = revision
        if not protocol["offline_dataset"].get("revision"):
            raise SystemExit(
                "M1 full training requires FAIR_V1_OFFLINE_REVISION=<40-character HF commit> "
                "after materializing and uploading the paired dataset"
            )
    default_root = Path("outputs") / ("smoke/fair-v1" if args.mode == "smoke" else "train/fair-v1")
    output_dir = (args.output_dir or default_root / args.model_id).resolve()
    command = build_train_command(protocol, args.model_id, args.mode, output_dir, protocol_path)
    if args.resume:
        command.append("--resume=true")
    if args.dry_run:
        print(shlex.join(command))
        return

    manifest = start_run_manifest(output_dir, protocol, args.model_id)
    try:
        subprocess.run(command, check=True)
    except Exception as error:
        finish_run_manifest(manifest, "failed", f"{type(error).__name__}: {error}")
        raise
    finish_run_manifest(manifest, "completed")


if __name__ == "__main__":
    main()
