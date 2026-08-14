#!/usr/bin/env python
"""Evaluate a policy under OOD visual perturbations.

Wraps LeRobot's standard eval pipeline and injects OOD image perturbations
(brightness, noise, cutout) between env observation and policy inference.

Usage:
    python scripts/eval_ood.py \
        --policy.path=<hub_id_or_local_path> \
        --env.type=libero --env.task=libero_spatial \
        --ood_level=level_1 \
        --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda

OOD Levels:
    level_0: Clean (no perturbation)
    level_1: Mild (brightness ±30%, noise σ=0.05)
    level_2: Extreme (brightness 0.1-3.0x, noise σ=0.20, cutout 15%)
"""

import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

# Add lerobot to path if running from CausalVLA root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

from causal_aug import OOD_LEVELS, OODPerturbation

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.scripts.lerobot_eval import close_envs, eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.device_utils import get_safe_torch_device

logger = logging.getLogger(__name__)


class OODProcessorStep(ObservationProcessorStep):
    """Processor step that applies OOD perturbation to image observations.

    Inserted into the env_preprocessor pipeline so perturbations happen
    after preprocess_observation() (images are [B, C, H, W] float32 [0,1])
    but before policy-specific preprocessing.
    """

    name = "ood_perturbation"

    def __init__(self, ood_level: str = "level_0", seed: int | None = None):
        self.perturbation = OODPerturbation(level=ood_level, seed=seed)
        self.ood_level = ood_level

    def observation(self, observation: dict) -> dict:
        for key in list(observation.keys()):
            if "image" in key and isinstance(observation[key], torch.Tensor):
                observation[key] = self.perturbation(observation[key])
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """OOD perturbations preserve observation keys, shapes and feature types."""
        return {feature_type: bucket.copy() for feature_type, bucket in features.items()}

    def __repr__(self) -> str:
        return f"OODProcessorStep(level={self.ood_level})"


@dataclass
class OODEvalConfig(EvalPipelineConfig):
    ood_level: str = "level_0"


@parser.wrap()
def eval_ood_main(cfg: OODEvalConfig):
    logging.basicConfig(level=logging.INFO)
    logger.info(pformat({"ood_level": cfg.ood_level}))

    if cfg.ood_level not in OOD_LEVELS:
        raise ValueError(f"Unknown OOD level '{cfg.ood_level}'. Choose from {list(OOD_LEVELS.keys())}")

    logger.info(f"OOD Perturbation: {cfg.ood_level} -> {OOD_LEVELS[cfg.ood_level]}")

    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # Create environments
    logger.info(f"Making environment (batch_size={cfg.eval.batch_size}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Create policy
    logger.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.eval()

    # Create preprocessors
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create env preprocessor with OOD perturbation injected
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    # Inject OOD perturbation step at the end of env_preprocessor
    ood_step = OODProcessorStep(ood_level=cfg.ood_level, seed=cfg.seed)
    env_preprocessor.steps.append(ood_step)
    logger.info(f"Env preprocessor pipeline: {env_preprocessor.steps}")

    # Run evaluation
    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            recording_dir=recording_dir,
            env_features=cfg.env.features if cfg.eval.recording else None,
            recording_repo_id=cfg.eval.recording_repo_id,
            recording_private=cfg.eval.recording_private,
        )

    # Add OOD metadata to results
    info["ood_level"] = cfg.ood_level
    info["ood_params"] = OOD_LEVELS[cfg.ood_level]

    logger.info(f"\n{'='*60}")
    logger.info(f"OOD Level: {cfg.ood_level}")
    logger.info(f"Overall Aggregated Metrics:")
    logger.info(info["overall"])

    for task_group, task_group_info in info.items():
        if task_group in ("overall", "ood_level", "ood_params"):
            continue
        logger.info(f"\nAggregated Metrics for {task_group}:")
        logger.info(task_group_info)

    close_envs(envs)

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Results saved to {output_dir / 'eval_info.json'}")
    logger.info("End of OOD eval")


if __name__ == "__main__":
    eval_ood_main()
