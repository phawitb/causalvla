from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _OneStepEnv:
    num_envs = 1

    def reset(self, **_kwargs):
        return {"camera": np.zeros((1, 2, 2, 3), dtype=np.uint8)}, {}

    def step(self, _action):
        observation = {"camera": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
        return observation, np.array([0.0]), np.array([True]), np.array([False]), {"is_success": [False]}

    def call(self, name):
        if name == "_max_episode_steps":
            return [1]
        if name in ("task_description", "task"):
            return ["test task"]
        raise AttributeError(name)


class _Policy(nn.Module):
    def reset(self):
        pass

    def select_action(self, observation):
        assert torch.all(observation["observation.images.image"] == 0.25)
        return torch.zeros((1, 1))


class _OODPreprocessor:
    def __call__(self, observation):
        observation["observation.images.image"] = torch.full((1, 3, 2, 2), 0.25)
        return observation


class _Identity:
    def __call__(self, value):
        return value


def test_rollout_policy_view_callback_receives_post_ood_observation(monkeypatch):
    import lerobot.scripts.lerobot_eval as eval_module

    monkeypatch.setattr(
        eval_module,
        "preprocess_observation",
        lambda _observation: {"observation.images.image": torch.zeros((1, 3, 2, 2))},
    )
    monkeypatch.setattr(eval_module, "check_env_attributes_and_types", lambda _env: None)
    captured = []

    eval_module.rollout(
        env=_OneStepEnv(),
        policy=_Policy(),
        env_preprocessor=_OODPreprocessor(),
        env_postprocessor=_Identity(),
        preprocessor=_Identity(),
        postprocessor=_Identity(),
        processed_render_callback=lambda observation: captured.append(
            observation["observation.images.image"].clone()
        ),
    )

    assert len(captured) == 1
    assert torch.all(captured[0] == 0.25)


def test_policy_view_frames_convert_post_ood_tensor_to_video_pixels():
    from lerobot.scripts.lerobot_eval import observation_to_video_frames

    observation = {
        "observation.state": torch.zeros((1, 4)),
        "observation.images.image2": torch.ones((1, 3, 2, 2)),
        "observation.images.image": torch.tensor(
            [[[[0.0, 1.0], [0.5, 0.25]], [[0.0, 1.0], [0.5, 0.25]], [[0.0, 1.0], [0.5, 0.25]]]]
        ),
    }

    frames = observation_to_video_frames(observation)

    assert frames.shape == (1, 2, 2, 3)
    assert frames.dtype == np.uint8
    assert frames[0, 0, 0].tolist() == [0, 0, 0]
    assert frames[0, 0, 1].tolist() == [255, 255, 255]
    assert frames[0, 1, 0].tolist() == [128, 128, 128]


def test_eval_one_returns_clean_and_policy_view_paths(monkeypatch, tmp_path):
    import lerobot.scripts.lerobot_eval as eval_module

    def fake_eval_policy(**kwargs):
        assert kwargs["save_policy_view"] is True
        return {
            "per_episode": [{"sum_reward": 1.0, "max_reward": 1.0, "success": True}],
            "video_paths": [str(tmp_path / "eval_episode_0.mp4")],
            "policy_video_paths": [str(tmp_path / "policy_view" / "eval_episode_0.mp4")],
        }

    monkeypatch.setattr(eval_module, "eval_policy", fake_eval_policy)
    metrics = eval_module.eval_one(
        object(),
        policy=object(),
        env_preprocessor=object(),
        env_postprocessor=object(),
        preprocessor=object(),
        postprocessor=object(),
        n_episodes=1,
        max_episodes_rendered=1,
        videos_dir=tmp_path,
        return_episode_data=False,
        start_seed=1000,
        save_policy_view=True,
    )

    assert metrics["video_paths"] == [str(tmp_path / "eval_episode_0.mp4")]
    assert metrics["policy_video_paths"] == [str(tmp_path / "policy_view" / "eval_episode_0.mp4")]


def test_ood_provenance_records_reproduction_parameters():
    from scripts.eval_ood import build_ood_provenance

    provenance = build_ood_provenance("level_2", seed=3000)

    assert provenance == {
        "algorithm": "causal_aug.OODPerturbation",
        "version": 1,
        "processor_position": "post-env-preprocessing",
        "level": "level_2",
        "seed": 3000,
    }
