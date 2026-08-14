"""RAPID-Mix uses the standard SmolVLA processor pipeline."""

from typing import Any

import torch
from lerobot.processor import (
    NewLineTaskProcessorStep, PolicyAction, PolicyProcessorPipeline,
    TokenizerProcessorStep, make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_rapid_mix import RapidMixConfig


def make_rapid_mix_pre_post_processors(
    config: RapidMixConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps = [
        steps.rename_observations, steps.add_batch_dim, NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name, padding=config.pad_language_to,
            padding_side="right", max_length=config.tokenizer_max_length,
        ),
        steps.to_device, steps.normalize,
    ]
    return make_policy_processor_pipelines(
        input_steps=input_steps, output_steps=[steps.unnormalize, steps.to_cpu]
    )
