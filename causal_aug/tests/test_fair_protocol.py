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
