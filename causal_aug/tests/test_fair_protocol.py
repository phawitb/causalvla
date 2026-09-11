import copy
from pathlib import Path

import pytest

from scripts.fair_protocol import MODEL_IDS, load_protocol, protocol_hash, validate_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "fair_v1.json"


def test_manifest_locks_shared_contract():
    protocol = load_protocol(PROTOCOL_PATH)
    validate_protocol(protocol, PROTOCOL_PATH)
    assert protocol["training"] == {
        "steps": 25000,
        "batch_size": 16,
        "seed": 1000,
        "save_freq": 5000,
        "action_warmup_steps": 10000,
    }
    assert tuple(protocol["models"]) == MODEL_IDS
    assert protocol["evaluation"]["seed"] == 4000


def test_protocol_hash_is_order_independent():
    protocol = load_protocol(PROTOCOL_PATH)
    reordered = {key: protocol[key] for key in reversed(protocol)}
    assert protocol_hash(protocol) == protocol_hash(reordered)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["dataset"].update(revision="main"), "immutable 40-character commit"),
        (lambda p: p["training"].update(steps=20000), "training.steps must equal 25000"),
        (lambda p: p["models"]["M1-offline-dr"].update(clean_per_batch=7), "8 clean and 8 augmented"),
        (lambda p: p["models"]["M3-v2-warm"].update(lambda_action=0.1), "lambda_action must equal 0.05"),
        (lambda p: p["models"]["M4-v2-warm-030"].update(lambda_action=0.05), "lambda_action must equal 0.03"),
        (lambda p: p["models"]["M5-v2-warm-070"].update(lambda_action=0.05), "lambda_action must equal 0.07"),
    ],
)
def test_protocol_rejects_drift(mutate, message):
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    mutate(protocol)
    with pytest.raises(ValueError, match=message):
        validate_protocol(protocol, PROTOCOL_PATH)


def test_protocol_rejects_augmentation_hash_mismatch():
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["augmentation_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="augmentation manifest hash"):
        validate_protocol(protocol, PROTOCOL_PATH)


def test_m4_differs_from_m3_only_by_identity_and_action_weight():
    protocol = load_protocol(PROTOCOL_PATH)
    m3 = protocol["models"]["M3-v2-warm"]
    m4 = protocol["models"]["M4-v2-warm-030"]
    comparable_m3 = {**m3, "repo_id": None, "lambda_action": None}
    comparable_m4 = {**m4, "repo_id": None, "lambda_action": None}
    assert comparable_m4 == comparable_m3
    assert m3["lambda_action"] == 0.05
    assert m4["lambda_action"] == 0.03


def test_m5_differs_from_m3_only_by_identity_and_action_weight():
    protocol = load_protocol(PROTOCOL_PATH)
    m3 = protocol["models"]["M3-v2-warm"]
    m5 = protocol["models"]["M5-v2-warm-070"]
    comparable_m3 = {**m3, "repo_id": None, "lambda_action": None}
    comparable_m5 = {**m5, "repo_id": None, "lambda_action": None}
    assert comparable_m5 == comparable_m3
    assert m3["lambda_action"] == 0.05
    assert m5["lambda_action"] == 0.07
