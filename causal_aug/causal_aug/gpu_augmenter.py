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
from .ood_wrapper import _affine, _gaussian_blur, _perspective, _rotation, _shadow


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
        contrast_range: tuple[float, float] = (0.7, 1.3),
        saturation_range: tuple[float, float] = (0.6, 1.5),
        hue_range: tuple[float, float] = (-0.05, 0.05),
        noise_sigma: float = 0.03,
        rotation_range: tuple[float, float] = (-2.0, 2.0),
        perspective_mag: float = 0.008,
        affine_translate: float = 0.01,
        affine_shear: float = 0.005,
        affine_scale_range: tuple[float, float] = (0.98, 1.02),
        shadow_prob: float = 0.3,
        shadow_alpha_range: tuple[float, float] = (0.1, 0.35),
        blur_prob: float = 0.2,
    ):
        super().__init__()
        self.K = K
        self.intensity = intensity
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.noise_sigma = noise_sigma
        self.rotation_range = rotation_range
        self.perspective_mag = perspective_mag
        self.affine_translate = affine_translate
        self.affine_shear = affine_shear
        self.affine_scale_range = affine_scale_range
        self.shadow_prob = shadow_prob
        self.shadow_alpha_range = shadow_alpha_range
        self.blur_prob = blur_prob

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

    def augment_camera_views(self, images: list[Tensor]) -> list[list[Tensor]]:
        """Augment multiple cameras with shared scene-level nuisance parameters.

        SmolVLA images arrive normalized to [-1, 1]. Transformations are applied
        in pixel space [0, 1], clamped, and normalized again. Lighting, color,
        camera-jitter, shadow and blur decisions are shared across cameras for a
        sample so a counterfactual remains a coherent observation of one scene.
        """
        if not images:
            return [[] for _ in range(self.K)]
        if self.intensity == 0:
            return [[image.clone() for image in images] for _ in range(self.K)]

        batch_size = images[0].shape[0]
        device = images[0].device
        results: list[list[Tensor]] = []

        for _ in range(self.K):
            brightness = self._sample_uniform(*self.brightness_range, (batch_size,), device)
            contrast = self._sample_uniform(*self.contrast_range, (batch_size,), device)
            saturation = self._sample_uniform(*self.saturation_range, (batch_size,), device)
            hue = self._sample_uniform(*self.hue_range, (batch_size,), device)
            sigma = torch.empty(batch_size, device=device).uniform_(0, self.noise_sigma * self.intensity)

            rotation = self._sample_uniform(*self.rotation_range, (batch_size,), device)
            tx = torch.empty(batch_size, device=device).uniform_(
                -self.affine_translate * self.intensity, self.affine_translate * self.intensity
            )
            ty = torch.empty(batch_size, device=device).uniform_(
                -self.affine_translate * self.intensity, self.affine_translate * self.intensity
            )
            shear = torch.empty(batch_size, device=device).uniform_(
                -self.affine_shear * self.intensity, self.affine_shear * self.intensity
            )
            scale = self._sample_uniform(*self.affine_scale_range, (batch_size,), device)
            shadow_mask = torch.rand(batch_size, device=device) < self.shadow_prob * self.intensity
            shadow_alpha = self._sample_uniform(*self.shadow_alpha_range, (batch_size,), device)
            directions = ["left", "right", "top", "bottom"]
            shadow_directions = [directions[i] for i in torch.randint(0, 4, (batch_size,)).tolist()]
            blur_mask = torch.rand(batch_size, device=device) < self.blur_prob * self.intensity

            camera_results = []
            for normalized in images:
                pixels = ((normalized + 1.0) / 2.0).clamp(0.0, 1.0)
                augmented = _perspective(pixels, self.perspective_mag * self.intensity)
                augmented = _affine(augmented, tx, ty, shear, scale)
                augmented = _rotation(augmented, rotation)
                augmented = adjust_brightness(augmented, brightness)
                augmented = adjust_contrast(augmented, contrast)
                if augmented.shape[1] >= 3:
                    augmented = adjust_saturation(augmented, saturation)
                    augmented = adjust_hue(augmented, hue)

                shadowed = _shadow(augmented, shadow_alpha, shadow_directions)
                augmented = torch.where(shadow_mask[:, None, None, None], shadowed, augmented)
                blurred = _gaussian_blur(augmented, kernel_size=3, sigma=0.6)
                augmented = torch.where(blur_mask[:, None, None, None], blurred, augmented)
                augmented = add_gaussian_noise(augmented, sigma).clamp(0.0, 1.0)
                camera_results.append(augmented * 2.0 - 1.0)

            results.append(camera_results)

        return results
