"""Deterministic OOD records that remain fixed for one evaluation episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn.functional as F
from torch import Tensor

from .ood_wrapper import (
    OOD_LEVELS, _affine, _brightness, _contrast, _cutout, _gaussian_blur,
    _hue_shift, _perspective, _rotation, _saturation, _shadow,
)


@dataclass(frozen=True)
class FixedOODIdentity:
    evaluation_seed: int
    task_id: int
    episode_index: int
    level: str
    schema_version: int = 1


def _generator(identity: FixedOODIdentity) -> torch.Generator:
    encoded = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":")).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


def _uniform(gen: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * torch.rand((), generator=gen).item()


def derive_fixed_ood_record(identity: FixedOODIdentity) -> dict:
    if identity.level not in OOD_LEVELS:
        raise ValueError(f"Unknown OOD level: {identity.level}")
    params = OOD_LEVELS[identity.level]
    gen = _generator(identity)
    sample = lambda name, default: _uniform(gen, *params.get(name, default))
    shadow = torch.rand((), generator=gen).item() < params.get("shadow_prob", 0.0)
    blur = torch.rand((), generator=gen).item() < params.get("blur_prob", 0.0)
    return {
        "schema_version": identity.schema_version,
        "identity": asdict(identity),
        "level": identity.level,
        "perspective_offsets": [[_uniform(gen, -params.get("perspective_mag", 0.0), params.get("perspective_mag", 0.0)) for _ in range(2)] for _ in range(4)],
        "translate_x": _uniform(gen, -params.get("affine_translate", 0.0), params.get("affine_translate", 0.0)),
        "translate_y": _uniform(gen, -params.get("affine_translate", 0.0), params.get("affine_translate", 0.0)),
        "shear": _uniform(gen, -params.get("affine_shear", 0.0), params.get("affine_shear", 0.0)),
        "scale": sample("affine_scale_range", (1.0, 1.0)),
        "rotation": sample("rotation_range", (0.0, 0.0)),
        "brightness": sample("brightness_range", (1.0, 1.0)),
        "contrast": sample("contrast_range", (1.0, 1.0)),
        "saturation": sample("saturation_range", (1.0, 1.0)),
        "hue": sample("hue_range", (0.0, 0.0)),
        "shadow_enabled": shadow,
        "shadow_alpha": sample("shadow_alpha_range", (0.0, 0.0)),
        "shadow_direction": ("left", "right", "top", "bottom")[int(torch.randint(0, 4, (), generator=gen).item())],
        "noise_sigma": params.get("noise_sigma", 0.0),
        "noise_seed": int(torch.randint(0, 2**31, (), generator=gen).item()),
        "blur_enabled": blur,
        "blur_kernel": params.get("blur_kernel", 3),
        "blur_sigma": params.get("blur_sigma", 0.5),
        "cutout": params.get("cutout", False),
        "cutout_ratio": params.get("cutout_ratio", 0.15),
        "cutout_x": _uniform(gen, 0.0, 1.0),
        "cutout_y": _uniform(gen, 0.0, 1.0),
        "perspective_mag": params.get("perspective_mag", 0.0),
}


def _recorded_perspective(images: Tensor, offsets: list[list[float]]) -> Tensor:
    batch, channels, height, width = images.shape
    source = torch.tensor([[-1, -1], [1, -1], [1, 1]], dtype=images.dtype, device=images.device)
    destination = source + torch.tensor(offsets[:3], dtype=images.dtype, device=images.device) * 2
    matrix = torch.linalg.solve(
        torch.cat([source, torch.ones(3, 1, dtype=images.dtype, device=images.device)], dim=1), destination
    ).T
    theta = matrix.unsqueeze(0).expand(batch, -1, -1)
    grid = F.affine_grid(theta, (batch, channels, height, width), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_fixed_ood_record(images: Tensor, record: dict) -> Tensor:
    if record["level"] == "level_0":
        return images
    out = images.clone()
    batch, _, height, width = out.shape
    device, dtype = out.device, out.dtype
    scalar = lambda value: torch.full((batch,), float(value), device=device, dtype=dtype)
    if record["perspective_mag"]:
        out = _recorded_perspective(out, record["perspective_offsets"])
    out = _affine(out, scalar(record["translate_x"]), scalar(record["translate_y"]), scalar(record["shear"]), scalar(record["scale"]))
    out = _rotation(out, scalar(record["rotation"]))
    out = _brightness(out, scalar(record["brightness"])[:, None, None, None])
    out = _contrast(out, scalar(record["contrast"])[:, None, None, None])
    if out.shape[1] >= 3:
        out = _saturation(out, scalar(record["saturation"])[:, None, None, None])
        out = _hue_shift(out, scalar(record["hue"]))
    if record["shadow_enabled"]:
        out = _shadow(out, scalar(record["shadow_alpha"]), [record["shadow_direction"]] * batch)
    if record["noise_sigma"]:
        gen = torch.Generator(device="cpu").manual_seed(record["noise_seed"])
        noise = torch.randn(out.shape, generator=gen, dtype=dtype).to(device)
        out = out + noise * record["noise_sigma"]
    if record["blur_enabled"]:
        out = _gaussian_blur(out, record["blur_kernel"], record["blur_sigma"])
    if record["cutout"]:
        cut_h, cut_w = int(height * record["cutout_ratio"]), int(width * record["cutout_ratio"])
        top = int(record["cutout_y"] * max(height - cut_h, 0))
        left = int(record["cutout_x"] * max(width - cut_w, 0))
        out[:, :, top:top + cut_h, left:left + cut_w] = 0.0
    return out.clamp(0.0, 1.0)
