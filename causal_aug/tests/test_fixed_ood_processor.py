import pytest
import torch

from causal_aug.fixed_ood import FixedEpisodeOODProcessor


def test_rejects_frame_before_begin_rollout():
    step = FixedEpisodeOODProcessor("level_2", evaluation_seed=4000)
    with pytest.raises(RuntimeError, match="begin_rollout"):
        step.apply({"observation.images.image": torch.rand(1, 3, 8, 8)})


def test_rejects_batch_mismatch():
    step = FixedEpisodeOODProcessor("level_2", evaluation_seed=4000)
    step.begin_rollout(task_id=2, episode_indices=[0, 1])
    with pytest.raises(ValueError, match="batch size"):
        step.apply({"observation.images.image": torch.rand(1, 3, 8, 8)})


def test_reuses_records_across_frames_and_camera_views():
    step = FixedEpisodeOODProcessor("level_2", evaluation_seed=4000)
    step.begin_rollout(task_id=2, episode_indices=[0, 1])
    image = torch.linspace(0, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    first = step.apply({"image": image.clone(), "wrist_image": image.clone()})
    second = step.apply({"image": image.clone(), "wrist_image": image.clone()})
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["image"], first["wrist_image"])
    assert not torch.equal(first["image"][0], first["image"][1])


def test_new_rollout_replaces_records():
    step = FixedEpisodeOODProcessor("level_2", evaluation_seed=4000)
    image = torch.rand(1, 3, 8, 8)
    step.begin_rollout(task_id=2, episode_indices=[0])
    first = step.apply({"image": image.clone()})["image"]
    step.begin_rollout(task_id=2, episode_indices=[1])
    second = step.apply({"image": image.clone()})["image"]
    assert not torch.equal(first, second)
