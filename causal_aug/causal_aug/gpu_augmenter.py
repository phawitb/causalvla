"""CausalAugmenter: Online GPU counterfactual augmentation engine."""

import torch
import torch.nn as nn
from torch import Tensor

from .augmentations import (
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)


class CausalAugmenter(nn.Module):
    """Generate K counterfactual image variants using pure PyTorch GPU ops.

    All augmentations are differentiable-friendly (though gradients through
    augmentation are not needed — we detach counterfactual inputs).

    Args:
        K: Number of counterfactual variants to generate per image.
        intensity: Scale factor for augmentation strength (0.0 = no aug, 1.0 = full).
        brightness_range: (min, max) brightness multiplier range around 1.0.
        contrast_range: (min, max) contrast multiplier range around 1.0.
        saturation_range: (min, max) saturation multiplier range around 1.0.
        hue_range: (min, max) hue shift range (fraction of full rotation).
        noise_sigma: Maximum Gaussian noise standard deviation.
    """

    def __init__(
        self,
        K: int = 3,
        intensity: float = 1.0,
        brightness_range: tuple[float, float] = (0.6, 1.4),
        contrast_range: tuple[float, float] = (0.6, 1.4),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        hue_range: tuple[float, float] = (-0.05, 0.05),
        noise_sigma: float = 0.08,
    ):
        super().__init__()
        self.K = K
        self.intensity = intensity
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.noise_sigma = noise_sigma

    def _sample_uniform(self, low: float, high: float, shape: tuple, device: torch.device) -> Tensor:
        """Sample uniform values, scaled by intensity toward the neutral point (1.0 or 0.0)."""
        raw = torch.empty(shape, device=device).uniform_(low, high)
        # Intensity interpolation: at intensity=0, return neutral (midpoint)
        neutral = (low + high) / 2.0
        return neutral + (raw - neutral) * self.intensity

    def forward(self, images: Tensor) -> Tensor:
        """Generate K counterfactual variants.

        Args:
            images: [B, C, H, W] normalized images (e.g., [-1, 1] from SmolVLA).

        Returns:
            [K, B, C, H, W] counterfactual image variants.
        """
        B = images.shape[0]
        device = images.device
        results = []

        for _ in range(self.K):
            aug = images.clone()

            # Brightness
            brightness_factor = self._sample_uniform(
                *self.brightness_range, (B,), device
            )
            aug = adjust_brightness(aug, brightness_factor)

            # Contrast
            contrast_factor = self._sample_uniform(
                *self.contrast_range, (B,), device
            )
            aug = adjust_contrast(aug, contrast_factor)

            # Saturation (only for RGB)
            if images.shape[1] >= 3:
                sat_factor = self._sample_uniform(
                    *self.saturation_range, (B,), device
                )
                aug = adjust_saturation(aug, sat_factor)

            # Hue shift (only for RGB)
            if images.shape[1] >= 3:
                hue_shift = self._sample_uniform(
                    *self.hue_range, (B,), device
                )
                aug = adjust_hue(aug, hue_shift)

            # Gaussian noise
            sigma = torch.empty(B, device=device).uniform_(0, self.noise_sigma * self.intensity)
            aug = add_gaussian_noise(aug, sigma)

            results.append(aug)

        return torch.stack(results, dim=0)  # [K, B, C, H, W]
