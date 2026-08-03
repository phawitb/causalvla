"""Individual GPU-based augmentation functions (pure PyTorch, no OpenCV)."""

import torch
from torch import Tensor


def adjust_brightness(images: Tensor, factor: Tensor) -> Tensor:
    """Adjust brightness by multiplicative factor.

    Args:
        images: [B, C, H, W] in any range
        factor: [B] or [B, 1, 1, 1] brightness multiplier (1.0 = no change)
    """
    if factor.ndim == 1:
        factor = factor[:, None, None, None]
    return images * factor


def adjust_contrast(images: Tensor, factor: Tensor) -> Tensor:
    """Adjust contrast around the per-image mean.

    Args:
        images: [B, C, H, W]
        factor: [B] or [B, 1, 1, 1] contrast multiplier (1.0 = no change)
    """
    if factor.ndim == 1:
        factor = factor[:, None, None, None]
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return (images - mean) * factor + mean


def adjust_saturation(images: Tensor, factor: Tensor) -> Tensor:
    """Adjust saturation. Assumes C=3 (RGB).

    Args:
        images: [B, 3, H, W]
        factor: [B] or [B, 1, 1, 1] saturation multiplier (0.0 = grayscale, 1.0 = no change)
    """
    if factor.ndim == 1:
        factor = factor[:, None, None, None]
    # ITU-R 601 luma
    gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
    return (images - gray) * factor + gray


def adjust_hue(images: Tensor, shift: Tensor) -> Tensor:
    """Shift hue by rotating in RGB space (approximate, avoids RGB→HSV conversion).

    Args:
        images: [B, 3, H, W]
        shift: [B] hue shift in [-0.5, 0.5] (fraction of full rotation)
    """
    # Rotation matrix around (1,1,1) axis in RGB space
    angle = shift * 2.0 * 3.14159265  # [B]
    cos_a = torch.cos(angle)[:, None, None, None]  # [B,1,1,1]
    sin_a = torch.sin(angle)[:, None, None, None]

    # Rodrigues' rotation: R = I*cos + (1-cos)*kkT + sin*K
    # k = (1,1,1)/sqrt(3), simplified:
    one_third = 1.0 / 3.0
    r = images[:, 0:1]
    g = images[:, 1:2]
    b = images[:, 2:3]

    new_r = cos_a * r + (1 - cos_a) * (r + g + b) * one_third + sin_a * (b - g) / 1.7320508
    new_g = cos_a * g + (1 - cos_a) * (r + g + b) * one_third + sin_a * (r - b) / 1.7320508
    new_b = cos_a * b + (1 - cos_a) * (r + g + b) * one_third + sin_a * (g - r) / 1.7320508

    return torch.cat([new_r, new_g, new_b], dim=1)


def add_gaussian_noise(images: Tensor, sigma: Tensor) -> Tensor:
    """Add Gaussian noise.

    Args:
        images: [B, C, H, W]
        sigma: [B] or [B, 1, 1, 1] noise standard deviation
    """
    if sigma.ndim == 1:
        sigma = sigma[:, None, None, None]
    noise = torch.randn_like(images) * sigma
    return images + noise
