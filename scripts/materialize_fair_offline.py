#!/usr/bin/env python3
"""Materialize the paired clean/augmented Fair v1 LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from causal_aug import apply_record, derive_record
from scripts.fair_protocol import load_protocol, validate_protocol


def _copy_frame(item: dict, features: dict, camera_keys: list[str], record: dict | None) -> dict:
    frame = {}
    for key in features:
        value = item[key]
        if key in camera_keys and record is not None:
            value = apply_record([value.unsqueeze(0) * 2.0 - 1.0], record)[0][0]
            value = (((value + 1.0) / 2.0) * 255).round().to(torch.uint8).permute(1, 2, 0)
        frame[key] = value
    frame["task"] = item.get("task", "")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records-out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    validate_protocol(protocol, args.protocol)
    augmentation = json.loads((args.protocol.parent / protocol["augmentation_manifest"]["path"]).read_text())

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    source_meta = LeRobotDatasetMetadata(
        protocol["dataset"]["repo_id"], revision=protocol["dataset"]["revision"]
    )
    episode_count = source_meta.total_episodes
    if args.max_episodes is not None:
        episode_count = min(episode_count, args.max_episodes)
    episodes = list(range(episode_count))
    source = LeRobotDataset(
        protocol["dataset"]["repo_id"], revision=protocol["dataset"]["revision"], episodes=episodes
    )
    features = {
        key: dict(feature)
        for key, feature in source_meta.features.items()
        if key not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    }
    destination = LeRobotDataset.create(
        repo_id=protocol["offline_dataset"]["repo_id"],
        fps=source_meta.fps,
        features=features,
        root=args.output_root,
        robot_type=source_meta.robot_type,
        use_videos=False,
    )

    records: list[dict] = []
    for domain in ("clean", "augmented"):
        for episode_id in episodes:
            start = source.meta.episodes["dataset_from_index"][episode_id]
            stop = source.meta.episodes["dataset_to_index"][episode_id]
            for source_index in range(start, stop):
                item = source[source_index]
                record = None
                if domain == "augmented":
                    record = derive_record(augmentation, protocol["training"]["seed"], episode_id, source_index - start, 0)
                    records.append({"source_index": source_index, **record})
                destination.add_frame(_copy_frame(item, features, source_meta.camera_keys, record))
            destination.save_episode()
    destination.finalize()

    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    args.records_out.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    metadata = {
        "clean_count": source.num_frames,
        "augmented_count": source.num_frames,
        "source_revision": protocol["dataset"]["revision"],
        "augmentation_sha256": protocol["augmentation_manifest"]["sha256"],
    }
    (args.output_root / "fair_v1_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if args.push_to_hub:
        destination.push_to_hub(tags=["fair-v1", "offline-dr", "libero"], license="apache-2.0")


if __name__ == "__main__":
    main()
