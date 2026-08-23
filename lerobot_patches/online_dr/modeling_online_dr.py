"""Model F: single-forward online domain-randomization baseline."""

import json
from pathlib import Path

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
        self.fair_manifest = None

    def _load_fair_manifest(self) -> None:
        if self.fair_manifest is None and self.config.fair_augmentation_manifest:
            self.fair_manifest = json.loads(Path(self.config.fair_augmentation_manifest).read_text())

    def _randomize_images(
        self, images: list[Tensor], batch: dict[str, Tensor] | None = None
    ) -> tuple[list[Tensor], Tensor]:
        batch_size = images[0].shape[0]
        if self.config.exact_balance:
            from causal_aug import exact_half_mask

            mask = exact_half_mask(batch_size, images[0].device)
        else:
            mask = torch.rand(batch_size, device=images[0].device) < self.config.aug_probability
        self._load_fair_manifest()
        if self.fair_manifest is None:
            augmented = self.augmenter.augment_camera_views([image.detach() for image in images])[0]
        else:
            from causal_aug import apply_record, derive_record

            if batch is None or "episode_index" not in batch or not ({"frame_index", "index"} & batch.keys()):
                raise ValueError("fair online augmentation requires episode_index and frame identity")
            frame_key = "frame_index" if "frame_index" in batch else "index"
            augmented = [image.detach().clone() for image in images]
            for index in mask.nonzero(as_tuple=False).flatten().tolist():
                record = derive_record(
                    self.fair_manifest,
                    self.config.fair_seed,
                    int(batch["episode_index"][index]),
                    int(batch[frame_key][index]),
                    0,
                )
                views = apply_record([image[index : index + 1] for image in images], record)
                for camera, view in zip(augmented, views, strict=True):
                    camera[index : index + 1] = view
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
        images, augmented_mask = self._randomize_images(images, batch)
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
