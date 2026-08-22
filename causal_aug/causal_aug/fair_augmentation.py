"""Stateless augmentation records shared by offline and online Fair v1 runs."""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F
from torch import Tensor

from .augmentations import add_gaussian_noise, adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
from .ood_wrapper import _affine, _gaussian_blur, _rotation, _shadow


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * torch.rand((), generator=generator).item()


def derive_record(
    manifest: dict,
    seed: int,
    episode_id: int,
    frame_index: int,
    exposure_index: int,
) -> dict:
    identity = f"{seed}:{episode_id}:{frame_index}:{exposure_index}"
    digest = hashlib.sha256(identity.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
    transforms = manifest["transforms"]

    def sample(name: str) -> float:
        config = transforms[name]
        return _uniform(generator, config["min"], config["max"])

    shadow = transforms["shadow"]
    blur = transforms["blur"]
    return {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": hashlib.sha256(_canonical(manifest).encode()).hexdigest(),
        "source": {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "exposure_index": exposure_index,
        },
        "brightness": sample("brightness"),
        "contrast": sample("contrast"),
        "saturation": sample("saturation"),
        "hue": sample("hue"),
        "noise_sigma": sample("noise_sigma"),
        "rotation_degrees": sample("rotation_degrees"),
        "translate_x": sample("translate_x"),
        "translate_y": sample("translate_y"),
        "shear": sample("shear"),
        "scale": sample("scale"),
        "perspective_offsets": [
            [_uniform(generator, -transforms["perspective_magnitude"]["value"], transforms["perspective_magnitude"]["value"]) for _ in range(2)]
            for _ in range(4)
        ],
        "shadow_enabled": torch.rand((), generator=generator).item() < shadow["probability"],
        "shadow_alpha": _uniform(generator, shadow["alpha_min"], shadow["alpha_max"]),
        "shadow_direction": ("left", "right", "top", "bottom")[
            int(torch.randint(0, 4, (), generator=generator).item())
        ],
        "blur_enabled": torch.rand((), generator=generator).item() < blur["probability"],
        "blur_kernel_size": blur["kernel_size"],
        "blur_sigma": blur["sigma"],
        "noise_seed": int(torch.randint(0, 2**31, (), generator=generator).item()),
    }


def _recorded_perspective(images: Tensor, offsets: list[list[float]]) -> Tensor:
    batch, channels, height, width = images.shape
    source = torch.tensor([[-1, -1], [1, -1], [1, 1]], dtype=images.dtype, device=images.device)
    destination = source + torch.tensor(offsets[:3], dtype=images.dtype, device=images.device) * 2
    matrix = torch.linalg.solve(
        torch.cat([source, torch.ones(3, 1, dtype=images.dtype, device=images.device)], dim=1),
        destination,
    ).T
    theta = matrix.unsqueeze(0).expand(batch, -1, -1)
    grid = F.affine_grid(theta, (batch, channels, height, width), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_record(images: list[Tensor], record: dict) -> list[Tensor]:
    output: list[Tensor] = []
    for normalized in images:
        batch = normalized.shape[0]
        device = normalized.device
        pixels = ((normalized + 1.0) / 2.0).clamp(0.0, 1.0)
        pixels = _recorded_perspective(pixels, record["perspective_offsets"])
        scalar = lambda value: torch.full((batch,), value, device=device, dtype=pixels.dtype)
        pixels = _affine(
            pixels,
            scalar(record["translate_x"]),
            scalar(record["translate_y"]),
            scalar(record["shear"]),
            scalar(record["scale"]),
        )
        pixels = _rotation(pixels, scalar(record["rotation_degrees"]))
        pixels = adjust_brightness(pixels, scalar(record["brightness"]))
        pixels = adjust_contrast(pixels, scalar(record["contrast"]))
        if pixels.shape[1] >= 3:
            pixels = adjust_saturation(pixels, scalar(record["saturation"]))
            pixels = adjust_hue(pixels, scalar(record["hue"]))
        if record["shadow_enabled"]:
            pixels = _shadow(pixels, scalar(record["shadow_alpha"]), [record["shadow_direction"]] * batch)
        if record["blur_enabled"]:
            pixels = _gaussian_blur(pixels, record["blur_kernel_size"], record["blur_sigma"])
        noise_generator = torch.Generator().manual_seed(record["noise_seed"])
        noise = torch.randn(pixels.shape, generator=noise_generator, dtype=pixels.dtype).to(device)
        pixels = (pixels + noise * record["noise_sigma"]).clamp(0.0, 1.0)
        output.append(pixels * 2.0 - 1.0)
    return output
