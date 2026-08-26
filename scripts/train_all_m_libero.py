#!/usr/bin/env python3
"""Prepare offline datasets and train the Fair-v1 matrix on LIBERO suites."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.fair_protocol import MODEL_IDS, load_protocol, resolve_hf_revision, validate_protocol


SUITES = {
    "object": "configs/fair_object.json",
    "goal": "configs/fair_goal.json",
    "long": "configs/fair_long.json",
}

POLICY_MODULES = (
    "lerobot.policies.smolvla",
    "lerobot.policies.online_dr",
    "lerobot.policies.causal_vla_warm",
)


@dataclass(frozen=True)
class TrainingRun:
    suite: str
    model_id: str
    protocol_path: Path
    dataset_repo: str
    dataset_revision: str
    offline_repo: str
    output_dir: Path

    @property
    def needs_offline_dataset(self) -> bool:
        return self.model_id == "M1-offline-dr"


def _parse_selection(raw: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    if not selected:
        raise ValueError(f"at least one {label} is required")
    return selected


def build_run_matrix(root: Path, suites: tuple[str, ...], models: tuple[str, ...]) -> list[TrainingRun]:
    runs: list[TrainingRun] = []
    for suite in suites:
        if suite not in SUITES:
            raise ValueError(f"unknown suite: {suite}")
        protocol_path = (root / SUITES[suite]).resolve()
        protocol = load_protocol(protocol_path)
        validate_protocol(protocol, protocol_path)
        for model_id in models:
            if model_id not in MODEL_IDS:
                raise ValueError(f"unknown model: {model_id}")
            runs.append(
                TrainingRun(
                    suite=suite,
                    model_id=model_id,
                    protocol_path=protocol_path,
                    dataset_repo=protocol["dataset"]["repo_id"],
                    dataset_revision=protocol["dataset"]["revision"],
                    offline_repo=protocol["offline_dataset"]["repo_id"],
                    output_dir=(root / "outputs" / "train" / "fair-v1-multisuite" / suite / model_id).resolve(),
                )
            )
    return runs


def missing_policy_modules(find_spec=importlib.util.find_spec) -> list[str]:
    missing = []
    for module in POLICY_MODULES:
        try:
            found = find_spec(module)
        except (ImportError, ModuleNotFoundError):
            found = None
        if found is None:
            missing.append(module)
    return missing


def _materialize_command(root: Path, run: TrainingRun) -> list[str]:
    dataset_root = root / "outputs" / "datasets" / "fair-v1" / run.suite
    return [
        sys.executable,
        str(root / "scripts" / "materialize_fair_offline.py"),
        "--protocol",
        str(run.protocol_path),
        "--output-root",
        str(dataset_root),
        "--records-out",
        str(dataset_root / "augmentation_records.jsonl"),
        "--push-to-hub",
    ]


def _train_command(root: Path, run: TrainingRun, resume: bool) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "train_fair_v1.py"),
        run.model_id,
        "--protocol",
        str(run.protocol_path),
        "--mode",
        "full",
        "--output-dir",
        str(run.output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def _preflight(runs: list[TrainingRun]) -> None:
    missing = missing_policy_modules()
    if missing:
        policies = " ".join(module.rsplit(".", 1)[-1] for module in missing if not module.endswith("smolvla"))
        raise RuntimeError(
            f"missing policy modules: {', '.join(missing)}; "
            f"run: python scripts/install_policy_patches.py {policies}".rstrip()
        )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed in the active Python environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the active Python environment")

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.whoami()
    except Exception as error:
        raise RuntimeError("Hugging Face login is required because all full runs push checkpoints") from error
    for run in runs:
        resolved = resolve_hf_revision("dataset", run.dataset_repo, run.dataset_revision, api=api)
        if resolved != run.dataset_revision:
            raise RuntimeError(f"dataset revision mismatch for {run.dataset_repo}: {resolved}")


def _existing_offline_revision(repo_id: str) -> str | None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return resolve_hf_revision("dataset", repo_id, api=HfApi())
    except RepositoryNotFoundError:
        return None


def _prepare_offline_datasets(root: Path, runs: list[TrainingRun], env: dict[str, str]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for run in runs:
        if not run.needs_offline_dataset or run.suite in revisions:
            continue
        revision = _existing_offline_revision(run.offline_repo)
        if revision is None:
            dataset_root = root / "outputs" / "datasets" / "fair-v1" / run.suite
            if dataset_root.exists():
                raise RuntimeError(
                    f"offline dataset output already exists but {run.offline_repo} does not; "
                    f"move or remove {dataset_root} after inspecting it"
                )
            subprocess.run(_materialize_command(root, run), cwd=root, env=env, check=True)
            revision = _existing_offline_revision(run.offline_repo)
            if revision is None:
                raise RuntimeError(f"offline dataset upload did not create {run.offline_repo}")
        revisions[run.suite] = revision
    return revisions


def _is_completed(output_dir: Path) -> bool:
    manifest = output_dir / "run_manifest.json"
    return manifest.is_file() and json.loads(manifest.read_text()).get("status") == "completed"


def should_resume_run(run: TrainingRun, requested: bool) -> bool:
    return requested and run.output_dir.is_dir()


def _print_dry_run(root: Path, runs: list[TrainingRun], resume: bool) -> None:
    prepared: set[str] = set()
    for run in runs:
        if run.needs_offline_dataset and run.suite not in prepared:
            print("# if offline dataset is absent:")
            print(shlex.join(_materialize_command(root, run)))
            prepared.add(run.suite)
        prefix = f"FAIR_V1_OFFLINE_REVISION=<{run.suite}-offline-revision> " if run.needs_offline_dataset else ""
        print(prefix + shlex.join(_train_command(root, run, resume)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--models", default=",".join(MODEL_IDS))
    parser.add_argument("--device", default="cuda:0", help="CUDA device, for example cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    try:
        suites = _parse_selection(args.suites, tuple(SUITES), "suite")
        models = _parse_selection(args.models, MODEL_IDS, "model")
        runs = build_run_matrix(root, suites, models)
        if not args.device.startswith("cuda:") or not args.device.removeprefix("cuda:").isdigit():
            raise ValueError("device must use the form cuda:<index>")
    except ValueError as error:
        parser.error(str(error))

    if args.dry_run:
        _print_dry_run(root, runs, args.resume)
        return

    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{root / 'causal_aug'}:{root / 'lerobot' / 'src'}:{root}"
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""),
        CUDA_VISIBLE_DEVICES=args.device.removeprefix("cuda:"),
    )
    _preflight(runs)
    offline_revisions = _prepare_offline_datasets(root, runs, env)
    for index, run in enumerate(runs, start=1):
        if _is_completed(run.output_dir):
            print(f"[{index}/{len(runs)}] skip completed {run.suite}/{run.model_id}", flush=True)
            continue
        print(f"[{index}/{len(runs)}] train {run.suite}/{run.model_id}", flush=True)
        run_env = env.copy()
        if run.needs_offline_dataset:
            run_env["FAIR_V1_OFFLINE_REVISION"] = offline_revisions[run.suite]
        subprocess.run(
            _train_command(root, run, should_resume_run(run, args.resume)),
            cwd=root,
            env=run_env,
            check=True,
        )


if __name__ == "__main__":
    main()
