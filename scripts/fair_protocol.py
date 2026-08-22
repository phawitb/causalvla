"""Shared configuration and safety helpers for Fair Protocol v1."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


MODEL_IDS = ("M0-clean", "M1-offline-dr", "M2-online-dr", "M3-v2-warm")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def load_protocol(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def protocol_hash(protocol: dict) -> str:
    return hashlib.sha256(canonical_json(protocol).encode()).hexdigest()


def model_config(protocol: dict, model_id: str) -> dict:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model ID: {model_id}")
    return protocol["models"][model_id]


def validate_protocol(protocol: dict, protocol_path: Path | None = None) -> None:
    if tuple(protocol.get("models", {})) != MODEL_IDS:
        raise ValueError(f"models must be ordered as {MODEL_IDS}")
    for item in (protocol["base_model"], protocol["dataset"]):
        if not SHA_PATTERN.fullmatch(item["revision"]):
            raise ValueError("revision must be an immutable 40-character commit")

    locked = {"steps": 25000, "batch_size": 16, "seed": 1000, "save_freq": 5000}
    for key, expected in locked.items():
        if protocol["training"].get(key) != expected:
            raise ValueError(f"training.{key} must equal {expected}")
    if protocol["training"].get("action_warmup_steps") != 10000:
        raise ValueError("training.action_warmup_steps must equal 10000")

    for model_id in ("M1-offline-dr", "M2-online-dr"):
        config = protocol["models"][model_id]
        if (config.get("clean_per_batch"), config.get("augmented_per_batch")) != (8, 8):
            raise ValueError(f"{model_id} must use exactly 8 clean and 8 augmented samples")

    warm = protocol["models"]["M3-v2-warm"]
    if (warm.get("clean_task_weight"), warm.get("augmented_task_weight")) != (0.5, 0.5):
        raise ValueError("M3-v2-warm task weights must equal 0.5/0.5")
    if warm.get("lambda_action") != 0.05:
        raise ValueError("M3-v2-warm lambda_action must equal 0.05")

    if protocol_path is not None:
        manifest = Path(protocol_path).parent / protocol["augmentation_manifest"]["path"]
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual != protocol["augmentation_manifest"]["sha256"]:
            raise ValueError("augmentation manifest hash does not match file contents")


def _arg(name: str, value: object) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{name}={value}"


def build_train_command(
    protocol: dict,
    model_id: str,
    mode: Literal["smoke", "full"],
    output_dir: Path,
    protocol_path: Path,
) -> list[str]:
    validate_protocol(protocol, protocol_path)
    config = model_config(protocol, model_id)
    smoke = mode == "smoke"
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unknown training mode: {mode}")
    dataset_id = protocol["dataset"]["repo_id"]
    dataset_revision = protocol["dataset"]["revision"]
    if model_id == "M1-offline-dr":
        dataset_id = protocol["offline_dataset"]["repo_id"]
        dataset_revision = protocol["offline_dataset"].get("revision")
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        _arg("policy.type", config["policy_type"]),
        _arg("policy.pretrained_path", protocol["base_model"]["repo_id"]),
        _arg("policy.pretrained_revision", protocol["base_model"]["revision"]),
        _arg("policy.device", "mps" if smoke else "cuda"),
        _arg("policy.push_to_hub", not smoke),
        _arg("policy.repo_id", config["repo_id"]),
        _arg("dataset.repo_id", dataset_id),
        _arg("output_dir", output_dir),
        _arg("job_name", f"fair_v1_{model_id.lower()}"),
        _arg("batch_size", 2 if smoke else protocol["training"]["batch_size"]),
        _arg("steps", 1 if smoke else protocol["training"]["steps"]),
        _arg("seed", protocol["training"]["seed"]),
        _arg("save_freq", 1 if smoke else protocol["training"]["save_freq"]),
        _arg("save_checkpoint_to_hub", not smoke),
        _arg("log_freq", 1 if smoke else 100),
        _arg("num_workers", 0 if smoke else 4),
        _arg("persistent_workers", not smoke),
        _arg("env_eval_freq", 0),
    ]
    if dataset_revision:
        command.append(_arg("dataset.revision", dataset_revision))
    dataset_root = protocol.get("offline_dataset", {}).get("root") if model_id == "M1-offline-dr" else None
    if dataset_root:
        command.append(_arg("dataset.root", dataset_root))
    if model_id == "M1-offline-dr":
        count = protocol["offline_dataset"].get("smoke_count", 2) if smoke else protocol["dataset"]["total_frames"]
        command.extend(
            [_arg("paired_clean_count", count), _arg("paired_augmented_count", count), _arg("paired_batch_seed", 1000)]
        )
    elif model_id == "M2-online-dr":
        manifest = (protocol_path.parent / protocol["augmentation_manifest"]["path"]).resolve()
        command.extend(
            [
                _arg("policy.aug_probability", 0.5),
                _arg("policy.aug_intensity", 1.0),
                _arg("policy.exact_balance", True),
                _arg("policy.fair_augmentation_manifest", manifest),
                _arg("policy.fair_seed", protocol["training"]["seed"]),
            ]
        )
    elif model_id == "M3-v2-warm":
        command.extend(
            [
                _arg("policy.n_counterfactual", 1),
                _arg("policy.aug_intensity", 1.0),
                _arg("policy.clean_task_weight", 0.5),
                _arg("policy.augmented_task_weight", 0.5),
                _arg("policy.use_action_loss", True),
                _arg("policy.lambda_action", 0.05),
                _arg("policy.action_warmup_steps", 10000),
                _arg("policy.use_latent_loss", False),
                _arg("policy.lambda_latent", 0.0),
                _arg("policy.lambda_smooth", 0.0),
            ]
        )
    return command


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def start_run_manifest(output_dir: Path, protocol: dict, model_id: str) -> Path:
    output_dir = Path(output_dir)
    path = output_dir.parent / f".{output_dir.name}.run_manifest.json"
    digest = protocol_hash(protocol)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("protocol_sha256") != digest:
            raise ValueError("existing run manifest has a different protocol hash")
        if existing.get("status") == "completed":
            raise FileExistsError("completed run will not be overwritten")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    payload = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": digest,
        "model_id": model_id,
        "git_commit": git_commit,
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "base_model_revision": protocol["base_model"]["revision"],
        "dataset_revision": protocol["dataset"]["revision"],
    }
    _atomic_json(path, payload)
    return path


def finish_run_manifest(path: Path, status: str, error: str | None = None) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("status must be completed or failed")
    payload = json.loads(Path(path).read_text())
    payload.update(status=status, finished_at=datetime.now(UTC).isoformat())
    if error is not None:
        payload["error"] = error
    _atomic_json(Path(path), payload)


def resolve_hf_revision(
    repo_type: Literal["model", "dataset"],
    repo_id: str,
    revision: str | None = None,
    api=None,
) -> str:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.model_info(repo_id, revision=revision) if repo_type == "model" else api.dataset_info(
        repo_id, revision=revision
    )
    if not SHA_PATTERN.fullmatch(info.sha):
        raise ValueError(f"Hugging Face did not resolve {repo_id} to an immutable commit")
    return info.sha


def validate_downloaded_metadata(metadata: dict, expected_protocol_hash: str) -> dict:
    if metadata.get("protocol_sha256") != expected_protocol_hash:
        raise ValueError("downloaded checkpoint protocol hash does not match")
    return metadata


def validate_hf_checkpoint(repo_id: str, revision: str, expected_protocol_hash: str) -> dict:
    from huggingface_hub import hf_hub_download

    resolved = resolve_hf_revision("model", repo_id, revision)
    path = hf_hub_download(repo_id, "run_manifest.json", revision=resolved)
    metadata = validate_downloaded_metadata(json.loads(Path(path).read_text()), expected_protocol_hash)
    return {**metadata, "resolved_revision": resolved}
