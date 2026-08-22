#!/usr/bin/env python3
"""Run the paired Fair v1 LIBERO-Spatial evaluation matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.fair_protocol import MODEL_IDS, SHA_PATTERN, load_protocol, protocol_hash, resolve_hf_revision, validate_protocol


@dataclass(frozen=True)
class EvalRun:
    model_id: str
    level: str
    seed: int
    episodes_per_task: int


@dataclass(frozen=True)
class EvalExpectation:
    level: str
    seed: int
    task_count: int
    episodes_per_task: int
    model_revision: str
    protocol_sha256: str


def evaluation_matrix(protocol: dict, phase: Literal["preflight", "full"]) -> list[EvalRun]:
    if phase == "preflight":
        levels = ("level_0",)
        episodes = protocol["evaluation"]["preflight_episodes_per_task"]
    elif phase == "full":
        levels = tuple(protocol["evaluation"]["levels"])
        episodes = protocol["evaluation"]["episodes_per_task"]
    else:
        raise ValueError(f"unknown evaluation phase: {phase}")
    return [
        EvalRun(model, level, protocol["evaluation"]["seed"], episodes)
        for model in MODEL_IDS
        for level in levels
    ]


def build_eval_command(
    protocol: dict,
    model_id: str,
    level: str,
    episodes: int,
    revision: str,
    output_dir: Path,
) -> list[str]:
    if not SHA_PATTERN.fullmatch(revision):
        raise ValueError("evaluation requires an immutable model revision")
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_ood.py"),
        f"--policy.path={protocol['models'][model_id]['repo_id']}",
        f"--policy.pretrained_revision={revision}",
        "--policy.device=mps",
        "--env.type=libero",
        "--env.task=libero_spatial",
        '--rename_map={"observation.images.image2":"observation.images.wrist_image"}',
        f"--ood_level={level}",
        f"--eval.n_episodes={episodes}",
        "--eval.batch_size=2",
        "--eval.use_async_envs=false",
        f"--output_dir={output_dir}",
        f"--seed={protocol['evaluation']['seed']}",
    ]


def validate_eval_result(path: Path, expected: EvalExpectation) -> dict:
    payload = json.loads(Path(path).read_text())
    if len(payload.get("per_task", [])) != expected.task_count:
        raise ValueError("evaluation task count is incomplete")
    if payload.get("ood_level") != expected.level:
        raise ValueError("evaluation OOD level mismatch")
    if payload.get("ood_provenance", {}).get("seed") != expected.seed:
        raise ValueError("evaluation seed mismatch")
    if payload.get("model_revision") != expected.model_revision:
        raise ValueError("evaluation model revision mismatch")
    if payload.get("protocol_sha256") != expected.protocol_sha256:
        raise ValueError("evaluation protocol hash mismatch")
    for task in payload["per_task"]:
        metrics = task.get("metrics", {})
        for key in ("successes", "video_paths", "policy_video_paths"):
            if len(metrics.get(key, [])) != expected.episodes_per_task:
                raise ValueError(f"evaluation {key} count is incomplete")
    return payload


def _record_provenance(result_path: Path, revision: str, digest: str) -> None:
    payload = json.loads(result_path.read_text())
    payload["model_revision"] = revision
    payload["protocol_sha256"] = digest
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/fair_v1.json"))
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revision", action="append", default=[], metavar="MODEL=SHA")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    validate_protocol(protocol, protocol_path)
    supplied = dict(item.split("=", 1) for item in args.revision)
    digest = protocol_hash(protocol)
    for run in evaluation_matrix(protocol, args.phase):
        revision = supplied.get(run.model_id)
        if revision is None:
            try:
                revision = resolve_hf_revision("model", protocol["models"][run.model_id]["repo_id"])
            except Exception:
                if args.dry_run:
                    print(f"PENDING {run.model_id}: train and push before revision can be pinned")
                    continue
                raise
        output_dir = (
            protocol_path.parents[1]
            / "outputs/eval/fair-v1"
            / args.phase
            / run.model_id
            / run.level
            / f"seed{run.seed}"
        )
        result_path = output_dir / "eval_info.json"
        expectation = EvalExpectation(run.level, run.seed, 10, run.episodes_per_task, revision, digest)
        if result_path.is_file():
            validate_eval_result(result_path, expectation)
            continue
        command = build_eval_command(protocol, run.model_id, run.level, run.episodes_per_task, revision, output_dir)
        if args.dry_run:
            print(shlex.join(command))
            continue
        subprocess.run(command, check=True)
        _record_provenance(result_path, revision, digest)
        validate_eval_result(result_path, expectation)


if __name__ == "__main__":
    main()
