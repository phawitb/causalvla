"""Model J: two-forward policy-adaptive counterfactual training."""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_pacer_lite import PacerLiteConfig


class PacerLitePolicy(SmolVLAPolicy):
    """Pair clean and policy-selected augmented supervision during training."""

    config_class = PacerLiteConfig
    name = "pacer_lite"

    def __init__(self, config: PacerLiteConfig, **kwargs):
        super().__init__(config, **kwargs)
        from causal_aug import CleanSafetyController, PacerContextualBandit

        self.bandit = PacerContextualBandit(
            temperature=config.bandit_temperature,
            exploration_floor=config.exploration_floor,
            ema_decay=config.bandit_ema_decay,
            warmup_steps=config.bandit_warmup_steps,
        )
        self.safety = CleanSafetyController(
            max_weight=config.max_augmented_weight,
            min_weight=config.min_augmented_weight,
            tolerance=config.clean_tolerance,
            weight_decay=config.clean_weight_decay,
            weight_recovery=config.clean_weight_recovery,
            fast_decay=config.fast_ema_decay,
            slow_decay=config.slow_ema_decay,
            warmup_steps=config.bandit_warmup_steps,
        )

    def _per_sample_task_loss(self, losses: Tensor, action_is_pad: Tensor | None) -> Tensor:
        action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :action_dim]
        if action_is_pad is None:
            return losses.mean(dim=(1, 2))
        valid = ~action_is_pad
        denominator = (valid.sum(dim=1) * action_dim).clamp_min(1)
        return (losses * valid.unsqueeze(-1)).sum(dim=(1, 2)) / denominator

    def _action_disagreement(
        self,
        clean_velocity: Tensor,
        augmented_velocity: Tensor,
        action_is_pad: Tensor | None,
    ) -> Tensor:
        action_dim = self.config.action_feature.shape[0]
        squared = (
            augmented_velocity[:, :, :action_dim] - clean_velocity[:, :, :action_dim]
        ).square()
        if action_is_pad is None:
            return squared.mean(dim=(1, 2))
        valid = ~action_is_pad
        denominator = (valid.sum(dim=1) * action_dim).clamp_min(1)
        return (squared * valid.unsqueeze(-1)).sum(dim=(1, 2)) / denominator

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        from causal_aug import productive_difficulty_reward

        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE

            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")
        language_tokens = batch[OBS_LANGUAGE_TOKENS]
        language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        clean_losses, _, clean_velocity = self.model.forward_with_latent(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            actions,
            noise,
            time,
        )
        clean_per_sample = PacerLitePolicy._per_sample_task_loss(
            self, clean_losses, action_is_pad
        )

        contexts, choices = self.bandit.sample(clean_per_sample.detach())
        augmented_images = self.bandit.apply(
            [image.detach() for image in images], choices, self.config.aug_intensity
        )
        augmented_losses, _, augmented_velocity = self.model.forward_with_latent(
            augmented_images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            actions,
            noise,
            time,
        )
        augmented_per_sample = PacerLitePolicy._per_sample_task_loss(
            self, augmented_losses, action_is_pad
        )
        disagreement = PacerLitePolicy._action_disagreement(
            self, clean_velocity, augmented_velocity, action_is_pad
        )
        reward, loss_ratio = productive_difficulty_reward(
            clean_per_sample,
            augmented_per_sample,
            disagreement,
            self.config.max_loss_ratio,
            self.config.overhard_penalty,
            self.config.disagreement_clip,
        )

        with torch.no_grad():
            if self.training:
                rejected = self.bandit.update(contexts, choices, reward)
                augmented_weight, safety_trigger = self.safety.update(clean_per_sample.mean())
            else:
                rejected = torch.zeros((), device=clean_per_sample.device, dtype=torch.long)
                augmented_weight = self.safety.augmented_weight.detach().clone()
                safety_trigger = torch.tensor(False, device=clean_per_sample.device)

        paired_per_sample = (
            (1 - augmented_weight) * clean_per_sample
            + augmented_weight * augmented_per_sample
        )
        total_loss = paired_per_sample.mean()
        info = {
            "loss": total_loss.item(),
            "loss_task": total_loss.item(),
            "loss_task_clean": clean_per_sample.mean().item(),
            "loss_task_augmented": augmented_per_sample.mean().item(),
            "pacer/augmented_weight": augmented_weight.item(),
            "pacer/action_disagreement": disagreement.mean().item(),
            "pacer/loss_ratio": loss_ratio.mean().item(),
            "pacer/context_easy": (contexts == 0).float().mean().item(),
            "pacer/context_medium": (contexts == 1).float().mean().item(),
            "pacer/context_hard": (contexts == 2).float().mean().item(),
            "pacer/clean_fast_ema": self.safety.fast_ema.item(),
            "pacer/clean_slow_ema": self.safety.slow_ema.item(),
            "pacer/safety_trigger": float(safety_trigger.item()),
            "pacer/rejected_updates": float(rejected.item()),
        }
        for index, family in enumerate(self.bandit.families):
            info[f"pacer/select/{family}"] = (choices == index).float().mean().item()
            info[f"pacer/reward/{family}"] = self.bandit.reward_ema[:, index].mean().item()

        if reduction == "none":
            return paired_per_sample, info
        return total_loss, info
