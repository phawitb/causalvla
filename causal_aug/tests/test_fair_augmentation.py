import json
from pathlib import Path

import torch

from causal_aug import apply_record, derive_record


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "configs" / "fair_v1_augmentation.json").read_text())


def test_record_identity_is_stable_and_stateless():
    torch.manual_seed(99)
    first = derive_record(MANIFEST, 1000, 7, 42, 0)
    torch.manual_seed(2)
    second = derive_record(MANIFEST, 1000, 7, 42, 0)
    assert first == second
    assert first != derive_record(MANIFEST, 1000, 7, 43, 0)
    assert len(first["manifest_sha256"]) == 64


def test_offline_and_online_pixels_match_without_global_rng_effects():
    image = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    record = derive_record(MANIFEST, 1000, 1, 2, 0)
    torch.manual_seed(1)
    offline = apply_record([image.clone()], record)[0]
    torch.manual_seed(999)
    online = apply_record([image.clone()], record)[0]
    torch.testing.assert_close(offline, online, rtol=0, atol=1 / 255)
    assert offline.min() >= -1
    assert offline.max() <= 1


def test_record_keeps_source_identity():
    record = derive_record(MANIFEST, 1000, 3, 17, 2)
    assert record["source"] == {"episode_id": 3, "frame_index": 17, "exposure_index": 2}


def test_online_policy_uses_shared_record_for_selected_source(tmp_path):
    from lerobot.policies.online_dr.configuration_online_dr import OnlineDRConfig
    from lerobot.policies.online_dr.modeling_online_dr import OnlineDRPolicy

    manifest_path = tmp_path / "augmentation.json"
    manifest_path.write_text(json.dumps(MANIFEST))
    policy = object.__new__(OnlineDRPolicy)
    policy.config = OnlineDRConfig(
        exact_balance=True,
        aug_probability=0.5,
        fair_augmentation_manifest=str(manifest_path),
        fair_seed=1000,
    )
    policy.fair_manifest = MANIFEST
    images = [torch.zeros(2, 3, 32, 32)]
    torch.manual_seed(1000)
    mixed, mask = policy._randomize_images(
        images, {"episode_index": torch.tensor([1, 1]), "frame_index": torch.tensor([2, 3])}
    )
    selected = int(mask.nonzero()[0])
    expected = apply_record(
        [images[0][selected : selected + 1]], derive_record(MANIFEST, 1000, 1, 2 + selected, 0)
    )[0]
    torch.testing.assert_close(mixed[0][selected : selected + 1], expected)
