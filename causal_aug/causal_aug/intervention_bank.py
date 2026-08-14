"""Structured visual intervention families for RAPID-VLA profiling."""

from __future__ import annotations

import torch
from torch import Tensor

from .augmentations import add_gaussian_noise, adjust_brightness, adjust_hue, adjust_saturation
from .ood_wrapper import _affine, _gaussian_blur, _perspective, _rotation, _shadow


INTERVENTION_FAMILIES = ("brightness", "color", "noise", "blur", "shadow", "geometry", "composed")


class InterventionBank:
    """Apply one coherent intervention family to all camera views in a sample."""

    def apply(self, images: list[Tensor], family: str, intensity: float) -> list[Tensor]:
        if family not in INTERVENTION_FAMILIES:
            raise ValueError(f"Unknown intervention family: {family}")
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("intensity must be in [0, 1]")
        if intensity == 0.0:
            return [image.clone() for image in images]

        batch = images[0].shape[0]
        device = images[0].device
        brightness = torch.empty(batch, device=device).uniform_(1 - 0.4 * intensity, 1 + 0.4 * intensity)
        saturation = torch.empty(batch, device=device).uniform_(1 - 0.4 * intensity, 1 + 0.5 * intensity)
        hue = torch.empty(batch, device=device).uniform_(-0.05 * intensity, 0.05 * intensity)
        sigma = torch.empty(batch, device=device).uniform_(0.01 * intensity, 0.04 * intensity)
        angle = torch.empty(batch, device=device).uniform_(-2 * intensity, 2 * intensity)
        tx = torch.empty(batch, device=device).uniform_(-0.01 * intensity, 0.01 * intensity)
        ty = torch.empty(batch, device=device).uniform_(-0.01 * intensity, 0.01 * intensity)
        shear = torch.empty(batch, device=device).uniform_(-0.005 * intensity, 0.005 * intensity)
        scale = torch.empty(batch, device=device).uniform_(1 - 0.02 * intensity, 1 + 0.02 * intensity)
        shadow_alpha = torch.empty(batch, device=device).uniform_(0.1 * intensity, 0.35 * intensity)
        directions = ["left", "right", "top", "bottom"]
        shadow_dirs = [directions[i] for i in torch.randint(0, 4, (batch,)).tolist()]

        outputs = []
        for normalized in images:
            pixels = ((normalized + 1) / 2).clamp(0, 1)
            out = pixels
            if family in ("geometry", "composed"):
                out = _perspective(out, 0.008 * intensity)
                out = _affine(out, tx, ty, shear, scale)
                out = _rotation(out, angle)
            if family in ("brightness", "composed"):
                out = adjust_brightness(out, brightness)
            if family in ("color", "composed") and out.shape[1] >= 3:
                out = adjust_saturation(out, saturation)
                out = adjust_hue(out, hue)
            if family in ("shadow", "composed"):
                out = _shadow(out, shadow_alpha, shadow_dirs)
            if family in ("blur", "composed"):
                out = _gaussian_blur(out, kernel_size=3, sigma=0.3 + 0.7 * intensity)
            if family in ("noise", "composed"):
                out = add_gaussian_noise(out, sigma)
            outputs.append(out.clamp(0, 1) * 2 - 1)
        return outputs
