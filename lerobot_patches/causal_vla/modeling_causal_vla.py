"""CausalVLA-v2: paired clean/augmented supervised flow matching."""

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_causal_vla import CausalVLAConfig


class CausalVLAPolicy(SmolVLAPolicy):
    """CausalVLA-v2 supervises both clean and counterfactual observations.

    Training forward pass:
        1. Sample flow noise/time once for the paired views.
        2. Compute supervised flow-matching loss on the clean observation.
        3. Generate one counterfactual observation.
        4. Compute the same supervised loss on the counterfactual observation.
        5. Combine the two task losses. Optional invariance losses remain available
           for controlled ablations, but are disabled by default.

    Inference is identical to SmolVLA (no augmentation, no extra losses).
    """

    config_class = CausalVLAConfig
    name = "causal_vla"

    def __init__(self, config: CausalVLAConfig, **kwargs):
        super().__init__(config, **kwargs)

        from causal_aug import CausalAugmenter

        self.augmenter = CausalAugmenter(
            K=config.n_counterfactual,
            intensity=config.aug_intensity,
        )

    def _pool_expert_latent(self, expert_latent: Tensor) -> Tensor:
        """Mean-pool trainable action-expert hidden states into [B, D]."""
        return expert_latent.mean(dim=1)

    def _augment_images(self, images: list[Tensor]) -> list[list[Tensor]]:
        """Generate K counterfactual variants for each image in the list.

        Args:
            images: List of [B, C, H, W] tensors (one per camera).

        Returns:
            List of K lists, each containing augmented versions of all cameras.
            cf_images[k][cam_idx] = [B, C, H, W]
        """
        K = self.config.n_counterfactual
        cf_images = [[] for _ in range(K)]
        augmented_views = self.augmenter.augment_camera_views([img.detach() for img in images])
        for k in range(K):
            cf_images[k].extend(augmented_views[k])
        return cf_images

    def _task_loss(self, losses: Tensor, actions_is_pad: Tensor | None) -> Tensor:
        """Reduce per-element flow loss over real action dimensions and valid steps."""
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is None:
            return losses.mean()
        valid = ~actions_is_pad
        denominator = (valid.sum() * losses.shape[-1]).clamp_min(1)
        return (losses * valid.unsqueeze(-1)).sum() / denominator

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if self.config.adapt_to_pi_aloha:
            from lerobot.utils.constants import OBS_STATE

            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # Prepare inputs (same as SmolVLA)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        # Sample once so clean and augmented branches solve the same flow target.
        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        # --- 1. Supervised clean branch ---
        losses_0, expert_latent_0, v_t_0 = self.model.forward_with_latent(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        original_action_dim = self.config.action_feature.shape[0]
        loss_task_clean = self._task_loss(losses_0, actions_is_pad)
        loss_dict = {"loss_task_clean": loss_task_clean.item()}

        # --- 2. Extract latent z_0 and action proxy a_0 ---
        z_0 = self._pool_expert_latent(expert_latent_0)
        a_0 = v_t_0[:, :, :original_action_dim]       # [B, chunk_size, action_dim]

        # --- 3. Generate counterfactual images ---
        cf_images = self._augment_images(images)

        # --- 4. Forward counterfactuals and compute invariance losses ---
        augmented_task_losses = []
        latent_losses = []
        action_losses = []

        for k in range(self.config.n_counterfactual):
            # Forward counterfactual k (reuse same noise/time for consistency)
            losses_k, expert_latent_k, v_t_k = self.model.forward_with_latent(
                cf_images[k], img_masks, lang_tokens, lang_masks, state, actions, noise, time
            )
            augmented_task_losses.append(self._task_loss(losses_k, actions_is_pad))
            z_k = self._pool_expert_latent(expert_latent_k)
            a_k = v_t_k[:, :, :original_action_dim]

            if self.config.use_latent_loss:
                latent_losses.append((1 - F.cosine_similarity(z_k, z_0.detach(), dim=-1)).mean())

            if self.config.use_action_loss:
                action_losses.append(F.mse_loss(a_k, a_0.detach()))

        # --- 5. Aggregate invariance losses ---
        loss_task_augmented = sum(augmented_task_losses) / len(augmented_task_losses)
        total_loss = (
            self.config.clean_task_weight * loss_task_clean
            + self.config.augmented_task_weight * loss_task_augmented
        )
        loss_dict["loss_task_augmented"] = loss_task_augmented.item()
        loss_dict["loss_task"] = total_loss.item()

        if self.config.use_latent_loss and latent_losses:
            loss_latent = sum(latent_losses) / len(latent_losses)
            total_loss = total_loss + self.config.lambda_latent * loss_latent
            loss_dict["loss_latent"] = loss_latent.item()

        if self.config.use_action_loss and action_losses:
            loss_action = sum(action_losses) / len(action_losses)
            total_loss = total_loss + self.config.lambda_action * loss_action
            loss_dict["loss_action"] = loss_action.item()

        # --- 6. Temporal smoothness loss ---
        if self.config.lambda_smooth > 0:
            loss_smooth = F.mse_loss(a_0[:, :-1, :], a_0[:, 1:, :])
            total_loss = total_loss + self.config.lambda_smooth * loss_smooth
            loss_dict["loss_smooth"] = loss_smooth.item()

        loss_dict["loss"] = total_loss.item()

        if reduction == "none":
            # Per-sample loss for RA-BC weighting (use task loss component)
            if actions_is_pad is None:
                per_sample_loss = losses_0[:, :, :original_action_dim].mean(dim=(1, 2))
            else:
                valid = ~actions_is_pad
                clean = losses_0[:, :, :original_action_dim] * valid.unsqueeze(-1)
                num_valid_per_sample = (valid.sum(dim=1) * original_action_dim).clamp_min(1)
                per_sample_loss = clean.sum(dim=(1, 2)) / num_valid_per_sample
            return per_sample_loss, loss_dict

        return total_loss, loss_dict
