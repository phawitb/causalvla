"""CausalVLA: SmolVLA + Counterfactual Augmentation + Dual Invariance Losses."""

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy
from .configuration_causal_vla import CausalVLAConfig


class CausalVLAPolicy(SmolVLAPolicy):
    """CausalVLA extends SmolVLA with counterfactual augmentation and causal invariance losses.

    Training forward pass:
        1. Compute flow matching task loss (L_task) via SmolVLA
        2. Extract latent representations (z_0) and predicted velocity (v_t_0) from original images
        3. Generate K counterfactual images via CausalAugmenter
        4. Extract z_k and v_t_k from each counterfactual
        5. Compute invariance losses: L_latent, L_action, L_smooth
        6. L_total = L_task + λ_lat·L_lat + λ_act·L_act + λ_smooth·L_smooth

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

    def _pool_prefix_embs(self, prefix_embs: Tensor) -> Tensor:
        """Mean-pool action-expert hidden states into latent vector [B, D]."""
        return prefix_embs.mean(dim=1)

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
        for img in images:
            # augmenter returns [K, B, C, H, W]
            aug = self.augmenter(img.detach())
            for k in range(K):
                cf_images[k].append(aug[k])
        return cf_images

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

        # --- 1. Forward with latent extraction (original images) ---
        losses_0, prefix_embs_0, v_t_0 = self.model.forward_with_latent(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )

        # Task loss (same reduction as SmolVLA)
        original_action_dim = self.config.action_feature.shape[0]
        task_losses = losses_0[:, :, :original_action_dim]
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            task_losses = task_losses * in_episode_bound.unsqueeze(-1)
        task_losses = task_losses[:, :, : self.config.max_action_dim]

        if actions_is_pad is None:
            loss_task = task_losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * task_losses.shape[-1]).clamp_min(1)
            loss_task = task_losses.sum() / num_valid

        loss_dict = {"loss_task": loss_task.item()}

        # --- 2. Extract latent z_0 and action proxy a_0 ---
        z_0 = self._pool_prefix_embs(prefix_embs_0)  # [B, D]
        a_0 = v_t_0[:, :, :original_action_dim]       # [B, chunk_size, action_dim]

        # --- 3. Generate counterfactual images ---
        cf_images = self._augment_images(images)

        # --- 4. Forward counterfactuals and compute invariance losses ---
        latent_losses = []
        action_losses = []

        for k in range(self.config.n_counterfactual):
            # Forward counterfactual k (reuse same noise/time for consistency)
            _, prefix_embs_k, v_t_k = self.model.forward_with_latent(
                cf_images[k], img_masks, lang_tokens, lang_masks, state, actions, noise, time
            )

            z_k = self._pool_prefix_embs(prefix_embs_k)
            a_k = v_t_k[:, :, :original_action_dim]

            if self.config.use_latent_loss:
                latent_losses.append(F.mse_loss(z_0, z_k))

            if self.config.use_action_loss:
                action_losses.append(F.mse_loss(a_0, a_k))

        # --- 5. Aggregate invariance losses ---
        total_loss = loss_task

        if self.config.use_latent_loss and latent_losses:
            loss_latent = sum(latent_losses) / len(latent_losses)
            total_loss = total_loss + self.config.lambda_latent * loss_latent
            loss_dict["loss_latent"] = loss_latent.item()

        if self.config.use_action_loss and action_losses:
            loss_action = sum(action_losses) / len(action_losses)
            total_loss = total_loss + self.config.lambda_action * loss_action
            loss_dict["loss_action"] = loss_action.item()

        # --- 6. Temporal smoothness loss ---
        loss_smooth = F.mse_loss(a_0[:, :-1, :], a_0[:, 1:, :])
        total_loss = total_loss + self.config.lambda_smooth * loss_smooth
        loss_dict["loss_smooth"] = loss_smooth.item()

        loss_dict["loss"] = total_loss.item()

        if reduction == "none":
            # Per-sample loss for RA-BC weighting (use task loss component)
            if actions_is_pad is None:
                per_sample_loss = task_losses.mean(dim=(1, 2))
            else:
                num_valid_per_sample = ((~actions_is_pad).sum(dim=1) * task_losses.shape[-1]).clamp_min(1)
                per_sample_loss = task_losses.sum(dim=(1, 2)) / num_valid_per_sample
            return per_sample_loss, loss_dict

        return total_loss, loss_dict
