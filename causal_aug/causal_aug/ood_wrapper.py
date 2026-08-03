"""OOD (Out-of-Distribution) perturbation engine for evaluation.

Perturbation types match Lerobot-Dataset-Manager augmentations, reimplemented
as pure PyTorch GPU ops:

  Camera (geometric): perspective, affine, rotation
  Light (photometric): brightness, contrast, saturation, hue/color_jitter,
                       shadow, gaussian noise, gaussian blur, cutout

Provides 3 severity levels:
  - Level 0: Clean (no perturbation)
  - Level 1: Mild OOD
  - Level 2: Extreme OOD

All operations work on batched tensor images [B, C, H, W] in [0, 1].
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Perturbation functions (pure PyTorch) ────────────────────────────

def _brightness(images: Tensor, factor: Tensor) -> Tensor:
    """Multiplicative brightness. factor [B,1,1,1]."""
    return images * factor


def _contrast(images: Tensor, factor: Tensor) -> Tensor:
    """Contrast around per-image mean. factor [B,1,1,1]."""
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return (images - mean) * factor + mean


def _saturation(images: Tensor, factor: Tensor) -> Tensor:
    """Saturation (ITU-R 601 luma). factor [B,1,1,1]."""
    gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
    return (images - gray) * factor + gray


def _hue_shift(images: Tensor, shift: Tensor) -> Tensor:
    """Hue rotation in RGB space (Rodrigues). shift [B] in [-0.5, 0.5]."""
    angle = shift * 2.0 * math.pi
    cos_a = angle.cos()[:, None, None, None]
    sin_a = angle.sin()[:, None, None, None]
    one_third = 1.0 / 3.0
    sqrt3_inv = 1.0 / math.sqrt(3.0)
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    s = (r + g + b) * one_third
    new_r = cos_a * r + (1 - cos_a) * s + sin_a * (b - g) * sqrt3_inv
    new_g = cos_a * g + (1 - cos_a) * s + sin_a * (r - b) * sqrt3_inv
    new_b = cos_a * b + (1 - cos_a) * s + sin_a * (g - r) * sqrt3_inv
    return torch.cat([new_r, new_g, new_b], dim=1)


def _shadow(images: Tensor, alpha: Tensor, direction: list[str]) -> Tensor:
    """Directional shadow gradient. alpha [B], direction list of 'left'|'right'|'top'|'bottom'."""
    B, C, H, W = images.shape
    out = images.clone()
    for i in range(B):
        a = alpha[i].item()
        d = direction[i]
        if d in ("left", "right"):
            g = torch.linspace(1 - a, 1, W, device=images.device) if d == "left" \
                else torch.linspace(1, 1 - a, W, device=images.device)
            mask = g.unsqueeze(0).unsqueeze(0)  # [1, 1, W]
        else:
            g = torch.linspace(1 - a, 1, H, device=images.device) if d == "top" \
                else torch.linspace(1, 1 - a, H, device=images.device)
            mask = g.unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
        out[i] = out[i] * mask
    return out


def _gaussian_blur(images: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Gaussian blur with given kernel size and sigma."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=images.dtype, device=images.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    # Separable 2D convolution
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  # [K, K]
    kernel_2d = kernel_2d.expand(images.shape[1], 1, -1, -1)  # [C, 1, K, K]
    pad = kernel_size // 2
    return F.conv2d(images, kernel_2d, padding=pad, groups=images.shape[1])


def _rotation(images: Tensor, angles_deg: Tensor) -> Tensor:
    """Rotate images by per-sample angle (degrees). Uses grid_sample."""
    B, C, H, W = images.shape
    angles_rad = angles_deg * (math.pi / 180.0)
    cos_a = angles_rad.cos()
    sin_a = angles_rad.sin()
    # Rotation matrix (no translation)
    theta = torch.zeros(B, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a * H / W
    theta[:, 1, 0] = sin_a * W / H
    theta[:, 1, 1] = cos_a
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _perspective(images: Tensor, magnitude: float) -> Tensor:
    """Random perspective warp. Uses grid_sample with approximate perspective."""
    B, C, H, W = images.shape
    # Random corner offsets
    offsets = torch.empty(B, 4, 2, device=images.device).uniform_(-magnitude, magnitude)
    out = images.clone()
    for i in range(B):
        # Source corners (normalized [-1, 1])
        src = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=images.dtype, device=images.device)
        dst = src + offsets[i] * 2  # scale offsets to [-1, 1] space
        # Approximate with affine (use first 3 point pairs)
        # Solve for affine: dst[:3] = src[:3] @ A.T + t
        ones = torch.ones(3, 1, dtype=images.dtype, device=images.device)
        src3 = torch.cat([src[:3], ones], dim=1)  # [3, 3]
        dst3 = dst[:3]  # [3, 2]
        try:
            M = torch.linalg.solve(src3, dst3)  # [3, 2]
            theta = M.T.unsqueeze(0)  # [1, 2, 3]
        except Exception:
            continue
        grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
        out[i:i+1] = F.grid_sample(images[i:i+1], grid, mode="bilinear", padding_mode="border",
                                    align_corners=False)
    return out


def _affine(images: Tensor, tx: Tensor, ty: Tensor, shear: Tensor, scale: Tensor) -> Tensor:
    """Affine transform: translate + shear + scale. All params [B]."""
    B, C, H, W = images.shape
    theta = torch.zeros(B, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = scale
    theta[:, 0, 1] = shear
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = 0
    theta[:, 1, 1] = scale
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _cutout(images: Tensor, ratio: float) -> Tensor:
    """Random rectangular occlusion (black patch)."""
    B, C, H, W = images.shape
    out = images.clone()
    cut_h = int(H * ratio)
    cut_w = int(W * ratio)
    for i in range(B):
        y = torch.randint(0, H - cut_h + 1, (1,)).item()
        x = torch.randint(0, W - cut_w + 1, (1,)).item()
        out[i, :, y:y + cut_h, x:x + cut_w] = 0.0
    return out


# ── OOD Level Definitions ───────────────────────────────────────────

OOD_LEVELS: dict[str, dict] = {
    "level_0": {
        # In-Distribution — no perturbation
    },
    "level_1": {
        # Mild OOD
        "brightness_range": (0.7, 1.3),
        "contrast_range": (0.8, 1.2),
        "saturation_range": (0.7, 1.3),
        "hue_range": (-0.03, 0.03),
        "noise_sigma": 0.05,
        "blur_prob": 0.2,
        "blur_kernel": 3,
        "blur_sigma": 0.5,
        "shadow_prob": 0.2,
        "shadow_alpha_range": (0.1, 0.25),
        "rotation_range": (-5, 5),
        "perspective_mag": 0.0,
        "affine_translate": 0.0,
        "affine_shear": 0.0,
        "affine_scale_range": (1.0, 1.0),
        "cutout": False,
    },
    "level_2": {
        # Extreme OOD
        "brightness_range": (0.1, 3.0),
        "contrast_range": (0.5, 1.8),
        "saturation_range": (0.3, 2.0),
        "hue_range": (-0.08, 0.08),
        "noise_sigma": 0.20,
        "blur_prob": 0.4,
        "blur_kernel": 5,
        "blur_sigma": 1.2,
        "shadow_prob": 0.4,
        "shadow_alpha_range": (0.2, 0.5),
        "rotation_range": (-15, 15),
        "perspective_mag": 0.05,
        "affine_translate": 0.04,
        "affine_shear": 0.03,
        "affine_scale_range": (0.9, 1.1),
        "cutout": True,
        "cutout_ratio": 0.15,
    },
}


class OODPerturbation:
    """Apply OOD visual perturbations to batched images.

    Covers all augmentation types from Lerobot-Dataset-Manager:
      - Camera: perspective, affine (translate/shear/scale), rotation
      - Light:  brightness, contrast, saturation, hue, shadow, noise, blur
      - Cutout: random rectangular occlusion

    Args:
        level: "level_0", "level_1", "level_2", or a custom params dict.
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, level: str | dict = "level_0", seed: int | None = None):
        if isinstance(level, str):
            if level not in OOD_LEVELS:
                raise ValueError(f"Unknown OOD level '{level}'. Choose from {list(OOD_LEVELS.keys())}")
            self.params = OOD_LEVELS[level]
            self.level_name = level
        else:
            self.params = level
            self.level_name = "custom"

        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def _rand(self, lo: float, hi: float, shape: tuple, device: torch.device) -> Tensor:
        return torch.empty(shape, device=device).uniform_(lo, hi)

    def __call__(self, images: Tensor) -> Tensor:
        """Apply OOD perturbation.

        Args:
            images: [B, C, H, W] float32 in [0, 1]
        Returns:
            Perturbed images, same shape, clamped to [0, 1].
        """
        if not self.params:
            return images

        p = self.params
        out = images.clone()
        B = out.shape[0]
        dev = out.device

        # ── Geometric (camera) perturbations ──

        # Perspective
        mag = p.get("perspective_mag", 0.0)
        if mag > 0:
            out = _perspective(out, mag)

        # Affine (translate + shear + scale)
        t_range = p.get("affine_translate", 0.0)
        s_range = p.get("affine_shear", 0.0)
        sc_lo, sc_hi = p.get("affine_scale_range", (1.0, 1.0))
        if t_range > 0 or s_range > 0 or sc_lo != 1.0 or sc_hi != 1.0:
            tx = self._rand(-t_range, t_range, (B,), dev)
            ty = self._rand(-t_range, t_range, (B,), dev)
            shear = self._rand(-s_range, s_range, (B,), dev)
            scale = self._rand(sc_lo, sc_hi, (B,), dev)
            out = _affine(out, tx, ty, shear, scale)

        # Rotation
        r_lo, r_hi = p.get("rotation_range", (0, 0))
        if r_lo != 0 or r_hi != 0:
            angles = self._rand(r_lo, r_hi, (B,), dev)
            out = _rotation(out, angles)

        # ── Photometric (light) perturbations ──

        # Brightness
        b_lo, b_hi = p.get("brightness_range", (1.0, 1.0))
        if b_lo != 1.0 or b_hi != 1.0:
            factor = self._rand(b_lo, b_hi, (B, 1, 1, 1), dev)
            out = _brightness(out, factor)

        # Contrast
        c_lo, c_hi = p.get("contrast_range", (1.0, 1.0))
        if c_lo != 1.0 or c_hi != 1.0:
            factor = self._rand(c_lo, c_hi, (B, 1, 1, 1), dev)
            out = _contrast(out, factor)

        # Saturation
        s_lo, s_hi = p.get("saturation_range", (1.0, 1.0))
        if s_lo != 1.0 or s_hi != 1.0:
            factor = self._rand(s_lo, s_hi, (B, 1, 1, 1), dev)
            out = _saturation(out, factor)

        # Hue shift
        h_lo, h_hi = p.get("hue_range", (0.0, 0.0))
        if h_lo != 0.0 or h_hi != 0.0:
            shift = self._rand(h_lo, h_hi, (B,), dev)
            out = _hue_shift(out, shift)

        # Shadow
        shadow_prob = p.get("shadow_prob", 0.0)
        if shadow_prob > 0:
            mask = torch.rand(B) < shadow_prob
            if mask.any():
                sa_lo, sa_hi = p.get("shadow_alpha_range", (0.1, 0.3))
                alpha = self._rand(sa_lo, sa_hi, (B,), dev)
                dirs = ["left", "right", "top", "bottom"]
                direction = [dirs[torch.randint(0, 4, (1,)).item()] for _ in range(B)]
                shadowed = _shadow(out, alpha, direction)
                # Apply only to selected samples
                for i in range(B):
                    if mask[i]:
                        out[i] = shadowed[i]

        # Gaussian noise
        sigma = p.get("noise_sigma", 0.0)
        if sigma > 0:
            out = out + torch.randn_like(out) * sigma

        # Gaussian blur
        blur_prob = p.get("blur_prob", 0.0)
        if blur_prob > 0 and torch.rand(1).item() < blur_prob:
            k = p.get("blur_kernel", 3)
            s = p.get("blur_sigma", 0.5)
            out = _gaussian_blur(out, k, s)

        # ── Cutout ──

        if p.get("cutout", False):
            ratio = p.get("cutout_ratio", 0.15)
            out = _cutout(out, ratio)

        return out.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"OODPerturbation(level={self.level_name})"
