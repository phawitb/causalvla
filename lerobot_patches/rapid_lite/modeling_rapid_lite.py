"""RAPID-Lite: single-forward risk-weighted online interventions."""

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_rapid_lite import RapidLiteConfig


class RapidLitePolicy(SmolVLAPolicy):
    config_class = RapidLiteConfig
    name = "rapid_lite"

    def __init__(self, config: RapidLiteConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import RAPID_LITE_CANDIDATES, RiskWeightedInterventionSampler

        self.curriculum = RiskWeightedInterventionSampler(
            temperature=config.risk_temperature,
            exploration_floor=config.exploration_floor,
        )
        self.candidate_names = tuple(f"{f}:{s}" for f, s, _ in RAPID_LITE_CANDIDATES)

    def _randomize_images(self, images: list[Tensor]) -> tuple[list[Tensor], Tensor, Tensor]:
        augmented, choices = self.curriculum([image.detach() for image in images])
        batch_size = images[0].shape[0]
        mask = torch.rand(batch_size, device=images[0].device) < self.config.aug_probability
        broadcast = mask[:, None, None, None]
        mixed = [torch.where(broadcast, aug, clean) for clean, aug in zip(images, augmented, strict=True)]
        return mixed, mask, choices

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        images, augmented_mask, choices = self._randomize_images(images)
        losses = self.model.forward(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch), self.prepare_action(batch), noise, time,
        )
        losses = losses[:, :, : self.config.action_feature.shape[0]]
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            valid = ~actions_is_pad
            losses = losses * valid.unsqueeze(-1)

        if reduction == "none":
            if actions_is_pad is None:
                per_sample = losses.mean(dim=(1, 2))
            else:
                denominator = (valid.sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample = losses.sum(dim=(1, 2)) / denominator
            return per_sample, {"loss": per_sample.mean().item(), "augmented_fraction": augmented_mask.float().mean().item()}

        if actions_is_pad is None:
            loss = losses.mean()
        else:
            denominator = (valid.sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / denominator
        info = {
            "loss": loss.item(),
            "loss_task": loss.item(),
            "augmented_fraction": augmented_mask.float().mean().item(),
        }
        for index, name in enumerate(self.candidate_names):
            info[f"curriculum/{name}"] = ((choices == index) & augmented_mask).float().mean().item()
        return loss, info
