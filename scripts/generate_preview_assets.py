#!/usr/bin/env python
"""Generate MP4 videos and state/action JSON for augment_preview visualization.

Reads both original and augmented datasets, creates:
  - MP4 videos for each episode x camera x {orig, aug}
  - JSON file with state/action timeseries per episode
  - aug_params.json with per-episode augmentation parameters

Usage:
    cd /Users/phawit/Projects/CausalVLA/lerobot
    python ../scripts/generate_preview_assets.py            # all episodes
    python ../scripts/generate_preview_assets.py --max 20   # first 20 only
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

from lerobot.datasets import LeRobotDataset

PREVIEW_DIR = Path(__file__).resolve().parent.parent / "augment_preview"
SRC_REPO = "lerobot/libero_spatial_image"
AUG_ROOT = Path(__file__).resolve().parent.parent / "lerobot" / "datasets" / "libero_spatial_augmented"
AUG_REPO = "causalvla/libero_spatial_augmented"
FPS = 10
CAMERAS = ["observation.images.image", "observation.images.wrist_image"]
CAM_SHORT = {"observation.images.image": "image", "observation.images.wrist_image": "wrist_image"}


def frames_to_mp4_pipe(frames_iter, num_frames: int, h: int, w: int, output_path: Path, fps: int = 10):
    """Encode frames to MP4 by piping raw RGB bytes directly to ffmpeg (no temp PNGs)."""
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo",
        "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "26", "-preset", "ultrafast",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame_tensor in frames_iter:
        # [C, H, W] float32 -> [H, W, C] uint8 bytes
        raw = (frame_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().numpy().tobytes()
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()


def process_episode(ds, ep_idx: int, tag: str, h: int, w: int):
    """Extract frames -> MP4 and state/action for one episode. Returns (states, actions, timestamps)."""
    from_idx = ds.meta.episodes["dataset_from_index"][ep_idx]
    to_idx = ds.meta.episodes["dataset_to_index"][ep_idx]
    n = to_idx - from_idx

    # Collect all frames and data in one pass
    cam_frames = {cam: [] for cam in CAMERAS}
    states, actions, timestamps = [], [], []

    for idx in range(from_idx, to_idx):
        item = ds[idx]
        for cam in CAMERAS:
            cam_frames[cam].append(item[cam])
        if "observation.state" in item:
            s = item["observation.state"]
            states.append(s.tolist() if isinstance(s, torch.Tensor) else list(s))
        if "action" in item:
            a = item["action"]
            actions.append(a.tolist() if isinstance(a, torch.Tensor) else list(a))
        if "timestamp" in item:
            t = item["timestamp"]
            timestamps.append(t.item() if isinstance(t, torch.Tensor) else float(t))

    # Encode MP4s (pipe raw frames, much faster than PNG)
    for cam in CAMERAS:
        short = CAM_SHORT[cam]
        mp4_path = PREVIEW_DIR / f"ep{ep_idx}_{short}_{tag}.mp4"
        frames_to_mp4_pipe(iter(cam_frames[cam]), n, h, w, mp4_path, fps=FPS)

    return states, actions, timestamps


def generate_aug_params(num_episodes: int, seed: int = 42):
    """Generate aug_params.json for all episodes (same logic as augment_dataset.py)."""
    AUG = {
        "rotation_range": (-2, 2),
        "perspective_mag": 0.008,
        "affine_translate": 0.01,
        "affine_shear": 0.005,
        "affine_scale_range": (0.98, 1.02),
        "brightness_range": (0.6, 1.4),
        "contrast_range": (0.7, 1.3),
        "saturation_range": (0.6, 1.5),
        "hue_range": (-0.05, 0.05),
        "shadow_prob": 0.3,
        "shadow_alpha_range": (0.1, 0.35),
    }

    result = {}
    for ep in range(num_episodes):
        random.seed(seed + ep)
        p = {}
        r_lo, r_hi = AUG["rotation_range"]
        p["angle_deg"] = round(random.uniform(r_lo, r_hi), 4)

        mag = AUG["perspective_mag"]
        offsets = [random.uniform(-mag, mag) for _ in range(8)]
        p["perspective_mag"] = round(max(abs(x) for x in offsets), 4)

        t = AUG["affine_translate"]
        s = AUG["affine_shear"]
        sc_lo, sc_hi = AUG["affine_scale_range"]
        p["affine_tx"] = round(random.uniform(-t, t), 4)
        p["affine_ty"] = round(random.uniform(-t, t), 4)
        p["affine_shear"] = round(random.uniform(-s, s), 4)
        p["affine_scale"] = round(random.uniform(sc_lo, sc_hi), 4)

        p["brightness"] = round(random.uniform(*AUG["brightness_range"]), 4)
        p["contrast"] = round(random.uniform(*AUG["contrast_range"]), 4)
        p["saturation"] = round(random.uniform(*AUG["saturation_range"]), 4)
        p["hue_shift"] = round(random.uniform(*AUG["hue_range"]), 4)

        if random.random() < AUG["shadow_prob"]:
            p["shadow_alpha"] = round(random.uniform(*AUG["shadow_alpha_range"]), 4)
            p["shadow_dir"] = random.choice(["left", "right", "top", "bottom"])

        p["aug_level"] = "training"
        result[f"ep_{ep}"] = p

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max episodes (default: all)")
    args = parser.parse_args()

    PREVIEW_DIR.mkdir(exist_ok=True)

    # Load augmented dataset metadata to get total episodes
    aug_ds_meta = LeRobotDataset(AUG_REPO, root=AUG_ROOT, episodes=[0]).meta
    total_eps = aug_ds_meta.total_episodes
    num_eps = min(args.max, total_eps) if args.max else total_eps
    print(f"Will process {num_eps} / {total_eps} episodes")

    # Generate aug_params.json for ALL episodes (fast, no dataset needed)
    print(f"Generating aug_params.json for {total_eps} episodes...")
    aug_params = generate_aug_params(total_eps)
    with open(PREVIEW_DIR / "aug_params.json", "w") as f:
        json.dump(aug_params, f)

    # Load datasets
    ep_list = list(range(num_eps))
    print(f"Loading source dataset: {SRC_REPO} ({num_eps} episodes)...")
    src_ds = LeRobotDataset(SRC_REPO, episodes=ep_list)
    print(f"  Loaded {src_ds.num_frames} frames")

    print(f"Loading augmented dataset: {AUG_ROOT} ({num_eps} episodes)...")
    aug_ds = LeRobotDataset(AUG_REPO, root=AUG_ROOT, episodes=ep_list)
    print(f"  Loaded {aug_ds.num_frames} frames")

    all_episode_data = {}
    h, w = 256, 256

    for ep in tqdm(range(num_eps), desc="Generating previews"):
        # Original
        states, actions, timestamps = process_episode(src_ds, ep, "orig", h, w)
        # Augmented
        process_episode(aug_ds, ep, "aug", h, w)

        all_episode_data[f"ep_{ep}"] = {
            "num_frames": len(states),
            "fps": FPS,
            "timestamps": timestamps,
            "state": states,
            "action": actions,
        }

    # Save episode data JSON
    json_path = PREVIEW_DIR / "episode_data.json"
    with open(json_path, "w") as f:
        json.dump(all_episode_data, f)

    mp4s = list(PREVIEW_DIR.glob("*.mp4"))
    total_mb = sum(f.stat().st_size for f in mp4s) / (1024 * 1024)
    print(f"\nDone! {len(mp4s)} MP4 files ({total_mb:.1f} MB), {num_eps} episodes")


if __name__ == "__main__":
    main()
