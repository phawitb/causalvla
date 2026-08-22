import json
import shlex
from pathlib import Path

import pytest

from scripts.fair_protocol import (
    MODEL_IDS,
    build_train_command,
    finish_run_manifest,
    load_protocol,
    protocol_hash,
    start_run_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "fair_v1.json"
PROTOCOL = load_protocol(PROTOCOL_PATH)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_full_commands_share_locked_values(model_id, tmp_path):
    rendered = shlex.join(build_train_command(PROTOCOL, model_id, "full", tmp_path / model_id, PROTOCOL_PATH))
    assert "--steps=25000" in rendered
    assert "--batch_size=16" in rendered
    assert "--seed=1000" in rendered
    assert "--save_freq=5000" in rendered
    assert f"--policy.pretrained_path={PROTOCOL['base_model']['repo_id']}" in rendered
    assert f"--policy.pretrained_revision={PROTOCOL['base_model']['revision']}" in rendered


def test_commands_route_model_specific_contracts(tmp_path):
    commands = {
        model: shlex.join(build_train_command(PROTOCOL, model, "full", tmp_path / model, PROTOCOL_PATH))
        for model in MODEL_IDS
    }
    assert "--policy.type=smolvla" in commands["M0-clean"]
    assert "--paired_clean_count=" in commands["M1-offline-dr"]
    assert "--policy.type=online_dr" in commands["M2-online-dr"]
    assert "--policy.exact_balance=true" in commands["M2-online-dr"]
    assert "--policy.type=causal_vla_warm" in commands["M3-v2-warm"]
    assert "--policy.lambda_action=0.05" in commands["M3-v2-warm"]
    assert "--policy.action_warmup_steps=10000" in commands["M3-v2-warm"]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_smoke_command_never_pushes(model_id, tmp_path):
    rendered = shlex.join(build_train_command(PROTOCOL, model_id, "smoke", tmp_path / model_id, PROTOCOL_PATH))
    assert "--steps=1" in rendered
    assert "--batch_size=2" in rendered
    assert "--policy.device=mps" in rendered
    assert "--policy.push_to_hub=false" in rendered
    assert "--save_checkpoint_to_hub=false" in rendered


def test_run_manifest_rejects_protocol_drift_and_completed_runs(tmp_path):
    path = start_run_manifest(tmp_path, PROTOCOL, "M0-clean")
    finish_run_manifest(path, "completed")
    with pytest.raises(FileExistsError, match="completed"):
        start_run_manifest(tmp_path, PROTOCOL, "M0-clean")
    payload = json.loads(path.read_text())
    payload["status"] = "failed"
    payload["protocol_sha256"] = "bad"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protocol hash"):
        start_run_manifest(tmp_path, PROTOCOL, "M0-clean")


def test_protocol_hash_is_written_to_run_manifest(tmp_path):
    path = start_run_manifest(tmp_path, PROTOCOL, "M2-online-dr")
    assert json.loads(path.read_text())["protocol_sha256"] == protocol_hash(PROTOCOL)
