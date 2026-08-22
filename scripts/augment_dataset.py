#!/usr/bin/env python
"""Create an augmented copy of a LeRobot dataset for Domain Randomization (Model B).

Uses the SAME perturbation functions as ood_wrapper.py for experimental fairness.
Applies per-episode consistent augmentation (same params for all frames in an episode).

Usage:
    # Full augmentation (all episodes)
    python scripts/augment_dataset.py \
        --src_repo_id lerobot/libero_spatial_image \
        --dst_repo_id <username>/libero_spatial_augmented \
        --aug_level training

    # Dry run (first 5 episodes only)
    python scripts/augment_dataset.py \
        --src_repo_id lerobot/libero_spatial_image \
        --dst_repo_id test/augmented_dry_run \
        --aug_level training \
        --max_episodes 5

    # Push to Hub after creation
    python scripts/augment_dataset.py \
        --src_repo_id lerobot/libero_spatial_image \
        --dst_repo_id <username>/libero_spatial_augmented \
        --aug_level training \
        --push_to_hub
"""

import argparse
import logging
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

from causal_aug.ood_wrapper import (
    _affine,
    _brightness,
    _contrast,
    _cutout,
    _gaussian_blur,
    _hue_shift,
    _perspective,
    _rotation,
    _saturation,
    _shadow,
)

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Augmentation level for training (Domain Randomization).
# Uses same perturbation types as OOD evaluation but with moderate ranges.
# Per-episode: sample params ONCE, apply to ALL frames in that episode.
AUG_LEVELS = {
    "training": {
        # Camera (geometric) — per-episode consistent, very mild
        "rotation_range": (-2, 2),
        "perspective_mag": 0.008,
        "affine_translate": 0.01,
        "affine_shear": 0.005,
        "affine_scale_range": (0.98, 1.02),
        # Light (photometric) — per-episode consistent
        "brightness_range": (0.6, 1.4),
        "contrast_range": (0.7, 1.3),
        "saturation_range": (0.6, 1.5),
        "hue_range": (-0.05, 0.05),
        "shadow_prob": 0.3,
        "shadow_alpha_range": (0.1, 0.35),
        # Per-frame random (slight variation within episode)
        "noise_sigma": 0.03,
        "blur_prob": 0.2,
        "blur_kernel": 3,
        "blur_sigma": 0.6,
    },
    "mild": {
        "rotation_range": (-3, 3),
        "brightness_range": (0.8, 1.2),
        "contrast_range": (0.85, 1.15),
        "saturation_range": (0.8, 1.2),
        "hue_range": (-0.02, 0.02),
        "noise_sigma": 0.02,
    },
    "heavy": {
        "rotation_range": (-15, 15),
        "perspective_mag": 0.05,
        "affine_translate": 0.04,
        "affine_shear": 0.03,
        "affine_scale_range": (0.9, 1.1),
        "brightness_range": (0.3, 2.5),
        "contrast_range": (0.5, 1.8),
        "saturation_range": (0.3, 2.0),
        "hue_range": (-0.08, 0.08),
        "shadow_prob": 0.5,
        "shadow_alpha_range": (0.15, 0.5),
        "noise_sigma": 0.05,
        "blur_prob": 0.3,
        "blur_kernel": 5,
        "blur_sigma": 1.0,
        "cutout": True,
        "cutout_ratio": 0.10,
    },
}


def sample_episode_params(aug_params: dict, seed: int | None = None) -> dict:
    """Sample per-episode augmentation parameters (fixed for all frames in episode).

    Returns a dict of concrete values (not ranges) for each perturbation.
    """
    if seed is not None:
        random.seed(seed)

    p = {}

    # Geometric
    r_lo, r_hi = aug_params.get("rotation_range", (0, 0))
    if r_lo != 0 or r_hi != 0:
        p["angle_deg"] = random.uniform(r_lo, r_hi)

    mag = aug_params.get("perspective_mag", 0.0)
    if mag > 0:
        # Pre-sample perspective corner offsets (fixed per episode, not per frame)
        p["perspective_offsets"] = [random.uniform(-mag, mag) for _ in range(8)]

    t = aug_params.get("affine_translate", 0.0)
    s = aug_params.get("affine_shear", 0.0)
    sc_lo, sc_hi = aug_params.get("affine_scale_range", (1.0, 1.0))
    if t > 0 or s > 0 or sc_lo != 1.0:
        p["affine_tx"] = random.uniform(-t, t)
        p["affine_ty"] = random.uniform(-t, t)
        p["affine_shear"] = random.uniform(-s, s)
        p["affine_scale"] = random.uniform(sc_lo, sc_hi)

    # Photometric (per-episode fixed)
    b_lo, b_hi = aug_params.get("brightness_range", (1.0, 1.0))
    if b_lo != 1.0 or b_hi != 1.0:
        p["brightness"] = random.uniform(b_lo, b_hi)

    c_lo, c_hi = aug_params.get("contrast_range", (1.0, 1.0))
    if c_lo != 1.0 or c_hi != 1.0:
        p["contrast"] = random.uniform(c_lo, c_hi)

    s_lo, s_hi = aug_params.get("saturation_range", (1.0, 1.0))
    if s_lo != 1.0 or s_hi != 1.0:
        p["saturation"] = random.uniform(s_lo, s_hi)

    h_lo, h_hi = aug_params.get("hue_range", (0.0, 0.0))
    if h_lo != 0.0 or h_hi != 0.0:
        p["hue_shift"] = random.uniform(h_lo, h_hi)

    shadow_prob = aug_params.get("shadow_prob", 0.0)
    if shadow_prob > 0 and random.random() < shadow_prob:
        sa_lo, sa_hi = aug_params.get("shadow_alpha_range", (0.1, 0.3))
        p["shadow_alpha"] = random.uniform(sa_lo, sa_hi)
        p["shadow_dir"] = random.choice(["left", "right", "top", "bottom"])

    # Per-frame params (stored as ranges, applied per-frame)
    p["noise_sigma"] = aug_params.get("noise_sigma", 0.0)
    p["blur_prob"] = aug_params.get("blur_prob", 0.0)
    p["blur_kernel"] = aug_params.get("blur_kernel", 3)
    p["blur_sigma"] = aug_params.get("blur_sigma", 0.5)
    p["cutout"] = aug_params.get("cutout", False)
    p["cutout_ratio"] = aug_params.get("cutout_ratio", 0.15)

    return p


def _perspective_fixed(images: torch.Tensor, offsets: list[float]) -> torch.Tensor:
    """Apply perspective with pre-sampled corner offsets (episode-consistent)."""
    B, C, H, W = images.shape
    src = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=images.dtype, device=images.device)
    offs = torch.tensor(offsets, dtype=images.dtype, device=images.device).reshape(4, 2)
    dst = src + offs * 2
    ones = torch.ones(3, 1, dtype=images.dtype, device=images.device)
    src3 = torch.cat([src[:3], ones], dim=1)
    dst3 = dst[:3]
    try:
        M = torch.linalg.solve(src3, dst3)
        theta = M.T.unsqueeze(0)
    except Exception:
        return images
    grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
    out = images.clone()
    for i in range(B):
        out[i:i+1] = F.grid_sample(images[i:i+1], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out


def augment_image(image: torch.Tensor, ep_params: dict) -> torch.Tensor:
    """Apply augmentation to a single image using per-episode params.

    Args:
        image: [C, H, W] float32 in [0, 1]
        ep_params: dict from sample_episode_params()

    Returns:
        Augmented image [C, H, W] float32 clamped to [0, 1].
    """
    # Add batch dim: [1, C, H, W]
    img = image.unsqueeze(0)

    # ── Geometric (per-episode fixed) ──

    if "perspective_offsets" in ep_params:
        img = _perspective_fixed(img, ep_params["perspective_offsets"])

    if "affine_tx" in ep_params:
        img = _affine(
            img,
            tx=torch.tensor([ep_params["affine_tx"]]),
            ty=torch.tensor([ep_params["affine_ty"]]),
            shear=torch.tensor([ep_params["affine_shear"]]),
            scale=torch.tensor([ep_params["affine_scale"]]),
        )

    if "angle_deg" in ep_params:
        img = _rotation(img, torch.tensor([ep_params["angle_deg"]]))

    # ── Photometric (per-episode fixed) ──

    if "brightness" in ep_params:
        img = _brightness(img, torch.tensor([[[[ep_params["brightness"]]]]]))

    if "contrast" in ep_params:
        img = _contrast(img, torch.tensor([[[[ep_params["contrast"]]]]]))

    if "saturation" in ep_params:
        img = _saturation(img, torch.tensor([[[[ep_params["saturation"]]]]]))

    if "hue_shift" in ep_params:
        img = _hue_shift(img, torch.tensor([ep_params["hue_shift"]]))

    if "shadow_alpha" in ep_params:
        img = _shadow(
            img,
            alpha=torch.tensor([ep_params["shadow_alpha"]]),
            direction=[ep_params["shadow_dir"]],
        )

    # ── Per-frame random ──

    sigma = ep_params.get("noise_sigma", 0.0)
    if sigma > 0:
        img = img + torch.randn_like(img) * sigma

    blur_prob = ep_params.get("blur_prob", 0.0)
    if blur_prob > 0 and random.random() < blur_prob:
        img = _gaussian_blur(img, ep_params["blur_kernel"], ep_params["blur_sigma"])

    if ep_params.get("cutout", False):
        img = _cutout(img, ep_params["cutout_ratio"])

    return img.squeeze(0).clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Create augmented LeRobot dataset")
    parser.add_argument("--fair-protocol", type=Path, default=None)
    parser.add_argument("--src_repo_id", type=str, default="lerobot/libero_spatial_image")
    parser.add_argument("--dst_repo_id", type=str, required=True)
    parser.add_argument("--dst_root", type=str, default=None, help="Local dir for output dataset")
    parser.add_argument("--aug_level", type=str, default="training",
                        choices=list(AUG_LEVELS.keys()), help="Augmentation intensity level")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes (for dry run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.fair_protocol is not None:
        raise SystemExit(
            "Fair v1 paired materialization uses scripts/materialize_fair_offline.py "
            "so clean/augmented ordering and provenance cannot drift."
        )

    aug_params = AUG_LEVELS[args.aug_level]
    logger.info(f"Source: {args.src_repo_id}")
    logger.info(f"Destination: {args.dst_repo_id}")
    logger.info(f"Augmentation level: {args.aug_level}")
    logger.info(f"Aug params: {aug_params}")

    # Load source dataset metadata
    logger.info("Loading source dataset metadata...")
    src_meta = LeRobotDatasetMetadata(args.src_repo_id)
    total_episodes = src_meta.total_episodes
    if args.max_episodes:
        total_episodes = min(total_episodes, args.max_episodes)
    logger.info(f"Total episodes to process: {total_episodes} / {src_meta.total_episodes}")

    camera_keys = src_meta.camera_keys
    logger.info(f"Camera keys: {camera_keys}")

    # Determine features for new dataset (same as source but use 'image' dtype)
    features = {}
    for key, feat in src_meta.features.items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue  # auto-managed
        features[key] = dict(feat)
    logger.info(f"Features: {list(features.keys())}")

    # Create destination dataset
    dst_root = args.dst_root or f"./datasets/{args.dst_repo_id.split('/')[-1]}"
    logger.info(f"Creating destination dataset at: {dst_root}")
    dst_ds = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=src_meta.fps,
        features=features,
        root=Path(dst_root),
        robot_type=src_meta.robot_type,
        use_videos=False,  # store as images for simplicity
    )

    # Load ALL source episodes at once (much faster than per-episode loading)
    episode_list = list(range(total_episodes))
    logger.info(f"Loading source dataset ({total_episodes} episodes)...")
    src_ds = LeRobotDataset(args.src_repo_id, episodes=episode_list)
    logger.info(f"Loaded {src_ds.num_frames} frames")

    # Process episodes
    random.seed(args.seed)
    for ep_idx in trange(total_episodes, desc="Augmenting episodes"):
        from_idx = src_ds.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = src_ds.meta.episodes["dataset_to_index"][ep_idx]

        # Sample per-episode augmentation params
        ep_seed = args.seed + ep_idx
        ep_params = sample_episode_params(aug_params, seed=ep_seed)

        # Process frames
        for frame_global_idx in range(from_idx, to_idx):
            item = src_ds[frame_global_idx]

            frame = {}
            for key in features:
                if key in camera_keys:
                    # Augment image: [C, H, W] float32 [0, 1]
                    img = item[key]
                    aug_img = augment_image(img, ep_params)
                    # Convert to uint8 HWC for dataset storage
                    aug_img_uint8 = (aug_img * 255).to(torch.uint8)
                    aug_img_hwc = aug_img_uint8.permute(1, 2, 0)  # CHW → HWC
                    frame[key] = aug_img_hwc
                else:
                    frame[key] = item[key]

            frame["task"] = item.get("task", "")
            dst_ds.add_frame(frame)

        dst_ds.save_episode()

    # Finalize
    logger.info("Finalizing dataset...")
    dst_ds.finalize()
    logger.info(f"Dataset saved to: {dst_root}")
    logger.info(f"Total episodes: {total_episodes}")

    # Push to Hub
    if args.push_to_hub:
        logger.info(f"Pushing to Hub: {args.dst_repo_id}")
        dst_ds.push_to_hub(
            tags=["augmented", "domain-randomization", "libero", "causalvla"],
            license="apache-2.0",
            private=args.private,
        )
        logger.info("Push complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
