"""Model F: single-forward online domain-randomization baseline."""

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_online_dr import OnlineDRConfig


class OnlineDRPolicy(SmolVLAPolicy):
    """Randomly augment samples online, then run one standard supervised forward.

    Unlike CausalVLA-v2, this baseline has no paired clean/counterfactual branches,
    shared flow target, or extra loss. Each sample is either clean or augmented.
    Inference remains identical to SmolVLA because augmentation is applied only in
    ``forward()``, which is the training path.
    """

    config_class = OnlineDRConfig
    name = "online_dr"

    def __init__(self, config: OnlineDRConfig, **kwargs):
        super().__init__(config, **kwargs)

        from causal_aug import CausalAugmenter

        self.augmenter = CausalAugmenter(K=1, intensity=config.aug_intensity)

    def _randomize_images(self, images: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        augmented = self.augmenter.augment_camera_views([image.detach() for image in images])[0]
        batch_size = images[0].shape[0]
        if self.config.exact_balance:
            from causal_aug import exact_half_mask

            mask = exact_half_mask(batch_size, images[0].device)
        else:
            mask = torch.rand(batch_size, device=images[0].device) < self.config.aug_probability
        broadcast_mask = mask[:, None, None, None]
        mixed = [torch.where(broadcast_mask, aug, clean) for clean, aug in zip(images, augmented)]
        return mixed, mask

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE

            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        images, augmented_mask = self._randomize_images(images)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is not None:
            valid = ~actions_is_pad
            losses = losses * valid.unsqueeze(-1)

        if reduction == "none":
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                denominator = (valid.sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / denominator
            return per_sample_loss, {
                "loss": per_sample_loss.mean().item(),
                "augmented_fraction": augmented_mask.float().mean().item(),
            }

        if actions_is_pad is None:
            loss = losses.mean()
        else:
            denominator = (valid.sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / denominator
        return loss, {
            "loss": loss.item(),
            "loss_task": loss.item(),
            "augmented_fraction": augmented_mask.float().mean().item(),
        }
