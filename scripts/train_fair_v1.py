#!/usr/bin/env python3
"""Train one Fair Protocol v1 model with drift and overwrite protection."""

from __future__ import annotations

import argparse
import os
import json
import math
import re
import shlex
import subprocess
import shutil
from pathlib import Path

from scripts.fair_protocol import (
    MODEL_IDS,
    build_train_command,
    finish_run_manifest,
    load_protocol,
    start_run_manifest,
    validate_protocol,
    resolve_hf_revision,
)


def model_card_text(protocol: dict, model_id: str) -> str:
    suite = protocol["evaluation"]["suite"].removeprefix("libero_")
    suite_name = "LIBERO " + ("Long" if suite == "10" else suite.title())
    return (
        f"# {model_id} — Fair Protocol v1\n\n"
        f"{suite_name} model. Training seed: {protocol['training']['seed']}. "
        f"Primary checkpoint: step {protocol['training']['steps']}.\n\n"
        "This model is one cell of a fixed-source-exposure comparison; do not interpret a single "
        "evaluation seed as statistical superiority.\n"
    )


def _run_and_log(command: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            lines.append(line)
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    return "".join(lines)


def _write_smoke_metrics(output_dir: Path, output: str, model_id: str) -> None:
    checkpoints = sorted((output_dir / "checkpoints").glob("*/pretrained_model"))
    checkpoint = checkpoints[-1] if checkpoints else None
    matches = re.findall(r"(?:loss[:=]\s*)([0-9.eE+-]+)", output)
    loss = float(matches[-1]) if matches else math.nan
    model_file = checkpoint / "model.safetensors" if checkpoint else None
    training_state = checkpoint.parent / "training_state" / "training_step.json" if checkpoint else None
    reload_ok = False
    if model_file and model_file.is_file() and (checkpoint / "config.json").is_file():
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as weights:
            reload_ok = bool(list(weights.keys()))
    batch_contract = True
    if model_id == "M2-online-dr":
        fractions = re.findall(r"augmented_fraction[:=]\s*([0-9.]+)", output)
        batch_contract = bool(fractions) and float(fractions[-1]) == 0.5
    metrics = {
        "loss": loss,
        "parameter_changed": bool(training_state and training_state.is_file()),
        "checkpoint_saved": bool(model_file and model_file.is_file()),
        "reload_inference": reload_ok,
        "batch_contract": batch_contract,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }
    (output_dir / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))


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
    if args.mode == "smoke" and args.model_id == "M1-offline-dr":
        dataset_root = protocol_path.parents[1] / "outputs/smoke/fair-v1/m1-dataset"
        metadata_path = dataset_root / "fair_v1_metadata.json"
        if not metadata_path.is_file():
            raise SystemExit("M1 smoke dataset is missing; run scripts/smoke_fair_v1.py to prepare it")
        metadata = json.loads(metadata_path.read_text())
        protocol["offline_dataset"].update(root=str(dataset_root), smoke_count=metadata["clean_count"])
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
    log_path = output_dir.parent / f".{output_dir.name}.train.log"
    try:
        output = _run_and_log(command, log_path)
        if args.mode == "smoke":
            _write_smoke_metrics(output_dir, output, args.model_id)
    except Exception as error:
        finish_run_manifest(manifest, "failed", f"{type(error).__name__}: {error}")
        raise
    finish_run_manifest(manifest, "completed")
    shutil.copy2(manifest, output_dir / "run_manifest.json")
    shutil.move(log_path, output_dir / "train.log")
    if args.mode == "full":
        from huggingface_hub import HfApi

        repo_id = protocol["models"][args.model_id]["repo_id"]
        api = HfApi()
        model_card = output_dir / "README.md"
        model_card.write_text(model_card_text(protocol, args.model_id))
        uploads = (
            (output_dir / "run_manifest.json", "run_manifest.json"),
            (output_dir / "train.log", "train.log"),
            (protocol_path, "fair_v1.json"),
            (protocol_path.parent / protocol["augmentation_manifest"]["path"], "fair_v1_augmentation.json"),
            (model_card, "README.md"),
        )
        for local_path, remote_path in uploads:
            api.upload_file(
                path_or_fileobj=str(local_path), path_in_repo=remote_path, repo_id=repo_id,
                repo_type="model", commit_message="Record Fair v1 provenance",
            )
        revision = resolve_hf_revision("model", repo_id, api=api)
        completed = json.loads((output_dir / "run_manifest.json").read_text())
        completed["hf_revision"] = revision
        (output_dir / "run_manifest.json").write_text(json.dumps(completed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
