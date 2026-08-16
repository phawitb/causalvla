"""COVER-Base: single-forward coverage-constrained robust training."""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_cover_base import CoverBaseConfig


class CoverBasePolicy(SmolVLAPolicy):
    config_class = CoverBaseConfig
    name = "cover_base"

    def __init__(self, config: CoverBaseConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import CoverCleanController, CoverageController

        self.coverage = CoverageController(
            ema_decay=config.cover_ema_decay,
            warmup_steps=config.cover_warmup_steps,
            temperature=config.cover_temperature,
            update_interval=config.cover_update_interval,
            weight_min=config.cover_weight_min,
            weight_max=config.cover_weight_max,
        )
        self.clean_controller = None
        if config.enable_clean_safety:
            self.clean_controller = CoverCleanController(
                fast_decay=config.clean_fast_decay,
                slow_decay=config.clean_slow_decay,
                tolerance=config.clean_tolerance,
                minimum_strength=config.minimum_robust_strength,
                strength_decay=config.robust_strength_decay,
                recovery=config.robust_strength_recovery,
                warmup_steps=config.cover_warmup_steps,
            )

    def _per_sample_task_loss(self, losses: Tensor, action_is_pad: Tensor | None) -> Tensor:
        action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :action_dim]
        if action_is_pad is None:
            return losses.mean(dim=(1, 2))
        valid = ~action_is_pad
        denominator = (valid.sum(dim=1) * action_dim).clamp_min(1)
        return (losses * valid.unsqueeze(-1)).sum(dim=(1, 2)) / denominator

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        from causal_aug import COVER_GROUPS, apply_cover_groups

        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE

            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")
        group_ids = self.coverage.sample(actions.shape[0], actions.device)
        training_images = apply_cover_groups(
            [image.detach() for image in images], group_ids, self.config.aug_intensity
        )

        losses = self.model.forward(
            training_images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            actions,
            noise,
            time,
        )
        per_sample = CoverBasePolicy._per_sample_task_loss(self, losses, action_is_pad)
        robust_strength = torch.ones((), device=per_sample.device)
        safety_trigger = torch.zeros((), dtype=torch.bool, device=per_sample.device)
        clean_mask = group_ids == 0
        if self.training and self.clean_controller is not None and bool(clean_mask.any()):
            robust_strength, safety_trigger = self.clean_controller.update(per_sample[clean_mask].mean())
        if self.training:
            self.coverage.update(per_sample.detach(), group_ids, robust_strength)
        weights = self.coverage.importance_weights(group_ids).to(per_sample.device)
        weighted = per_sample * weights
        total_loss = weighted.mean()

        info = self.coverage.metrics()
        info.update(
            {
                "loss": total_loss.item(),
                "loss_task": total_loss.item(),
                "cover/forward_count": 1.0,
                "cover/weight_min": weights.min().item(),
                "cover/weight_max": weights.max().item(),
                "cover/robust_strength": robust_strength.item(),
                "cover/safety_trigger": float(safety_trigger.item()),
            }
        )
        if self.clean_controller is None:
            info["cover/clean_fast_ema"] = 0.0
            info["cover/clean_slow_ema"] = 0.0
        else:
            info["cover/clean_fast_ema"] = self.clean_controller.fast_ema.item()
            info["cover/clean_slow_ema"] = self.clean_controller.slow_ema.item()
        for index, name in enumerate(COVER_GROUPS):
            mask = group_ids == index
            info[f"cover/group/{name}_fraction"] = mask.float().mean().item()
            info[f"cover/group/{name}_loss"] = per_sample[mask].mean().item() if bool(mask.any()) else 0.0

        if reduction == "none":
            return weighted, info
        return total_loss, info
