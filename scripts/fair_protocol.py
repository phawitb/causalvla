"""Shared configuration and safety helpers for Fair Protocol v1."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


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

