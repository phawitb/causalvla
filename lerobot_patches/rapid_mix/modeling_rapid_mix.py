"""Model H: coverage-preserving RAPID-Mix training policy."""

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_rapid_mix import RapidMixConfig


class RapidMixPolicy(SmolVLAPolicy):
    """Mix clean, broad DR and risk-guided samples before one VLA forward."""

    config_class = RapidMixConfig
    name = "rapid_mix"

    def __init__(self, config: RapidMixConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import CausalAugmenter, RAPID_LITE_CANDIDATES, RiskWeightedInterventionSampler

        self.broad_augmenter = CausalAugmenter(K=1, intensity=config.broad_intensity)
        self.risk_sampler = RiskWeightedInterventionSampler(
            temperature=config.risk_temperature,
            exploration_floor=config.exploration_floor,
        )
        self.candidate_names = tuple(f"{family}:{strength}" for family, strength, _ in RAPID_LITE_CANDIDATES)

    def _mix_images(self, images: list[Tensor]) -> tuple[list[Tensor], Tensor, Tensor]:
        detached = [image.detach() for image in images]
        broad = self.broad_augmenter.augment_camera_views(detached)[0]
        risk, risk_choices = self.risk_sampler(detached)

        draw = torch.rand(images[0].shape[0], device=images[0].device)
        broad_mask = draw < self.config.broad_probability
        risk_mask = (draw >= self.config.broad_probability) & (
            draw < self.config.broad_probability + self.config.risk_probability
        )
        broad_broadcast = broad_mask[:, None, None, None]
        risk_broadcast = risk_mask[:, None, None, None]
        mixed = [
            torch.where(risk_broadcast, risk_view, torch.where(broad_broadcast, broad_view, clean))
            for clean, broad_view, risk_view in zip(images, broad, risk, strict=True)
        ]
        branch = torch.zeros_like(risk_choices)
        branch[broad_mask] = 1
        branch[risk_mask] = 2
        return mixed, branch, risk_choices

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, image_masks = self.prepare_images(batch)
        images, branch, risk_choices = self._mix_images(images)
        losses = self.model.forward(
            images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch), self.prepare_action(batch), noise, time,
        )
        losses = losses[:, :, : self.config.action_feature.shape[0]]
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = ~action_is_pad
            losses = losses * valid.unsqueeze(-1)

        if reduction == "none":
            if action_is_pad is None:
                per_sample = losses.mean(dim=(1, 2))
            else:
                denominator = (valid.sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample = losses.sum(dim=(1, 2)) / denominator
            return per_sample, {"loss": per_sample.mean().item()}

        if action_is_pad is None:
            loss = losses.mean()
        else:
            denominator = (valid.sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / denominator
        info = {
            "loss": loss.item(),
            "loss_task": loss.item(),
            "branch/clean": (branch == 0).float().mean().item(),
            "branch/broad": (branch == 1).float().mean().item(),
            "branch/risk": (branch == 2).float().mean().item(),
        }
        risk_mask = branch == 2
        for index, name in enumerate(self.candidate_names):
            info[f"risk/{name}"] = (risk_mask & (risk_choices == index)).float().mean().item()
        return loss, info
