"""Model I: Model F coverage with policy-risk residual overlays."""

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_residual_rapid import ResidualRapidConfig


class ResidualRapidPolicy(SmolVLAPolicy):
    """Apply risk only on top of broad DR, then run one supervised forward."""

    config_class = ResidualRapidConfig
    name = "residual_rapid"

    def __init__(self, config: ResidualRapidConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import CausalAugmenter, RAPID_LITE_CANDIDATES, ResidualBranchSampler

        self.broad_augmenter = CausalAugmenter(K=1, intensity=config.broad_intensity)
        self.residual_sampler = ResidualBranchSampler(
            augmentation_probability=config.augmentation_probability,
            overlay_probability=config.risk_overlay_probability,
            risk_temperature=config.risk_temperature,
            exploration_floor=config.exploration_floor,
        )
        self.candidate_names = tuple(
            f"{family}:{strength}" for family, strength, _ in RAPID_LITE_CANDIDATES
        )

    def _compose_images(self, images: list[Tensor]) -> tuple[list[Tensor], Tensor, Tensor]:
        detached = [image.detach() for image in images]
        broad = self.broad_augmenter.augment_camera_views(detached)[0]
        return self.residual_sampler.compose(images, broad)

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE

            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, image_masks = self.prepare_images(batch)
        images, branch, risk_choices = self._compose_images(images)
        losses = self.model.forward(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch),
            self.prepare_action(batch),
            noise,
            time,
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
            "branch/residual": (branch == 2).float().mean().item(),
        }
        residual_mask = branch == 2
        for index, name in enumerate(self.candidate_names):
            info[f"residual/{name}"] = (
                residual_mask & (risk_choices == index)
            ).float().mean().item()
        return loss, info
