#!/usr/bin/env python3
"""Build the static data manifest used by the Results dashboard."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


MODEL_META = {
    "a": {"name": "Model A", "description": "Supervised fine-tuning baseline"},
    "b": {"name": "Model B", "description": "Domain randomization baseline"},
    "v2_warm": {"name": "V2 Warm", "description": "CausalVLA v2 warm-start"},
    "f": {"name": "Model F", "description": "Online domain randomization"},
}
RUN_PATTERN = re.compile(r"^model_(a|b|v2_warm|f)_level_(\d+)_(\d+)ep_seed(\d+)$")


def rate(successes: int, episodes: int) -> float:
    return round(successes * 100 / episodes, 1) if episodes else 0.0


def build_manifest(repo_root: Path) -> dict:
    eval_root = repo_root / "outputs" / "eval" / "full"
    stats = {
        model_id: {
            "id": model_id,
            **meta,
            "runs": 0,
            "episodes": 0,
            "successes": 0,
            "videos": 0,
            "levels": defaultdict(lambda: {"episodes": 0, "successes": 0}),
        }
        for model_id, meta in MODEL_META.items()
    }
    episodes = []

    for info_path in sorted(eval_root.glob("*/eval_info.json")):
        match = RUN_PATTERN.match(info_path.parent.name)
        if not match:
            continue
        model_id, level, _declared_episodes, seed = match.groups()
        model = stats[model_id]
        model["runs"] += 1
        payload = json.loads(info_path.read_text())

        for task in payload.get("per_task", []):
            metrics = task.get("metrics", {})
            task_successes = metrics.get("successes", [])
            video_paths = metrics.get("video_paths", [])
            task_group = task.get("task_group", "unknown")
            task_id = task.get("task_id", "?")

            for episode_index, succeeded in enumerate(task_successes):
                model["episodes"] += 1
                model["successes"] += int(bool(succeeded))
                model["levels"][level]["episodes"] += 1
                model["levels"][level]["successes"] += int(bool(succeeded))

                raw_video = video_paths[episode_index] if episode_index < len(video_paths) else ""
                video_path = Path(raw_video) if raw_video else None
                if not video_path or not video_path.is_file():
                    continue
                try:
                    relative_video = video_path.resolve().relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    continue

                model["videos"] += 1
                episodes.append(
                    {
                        "model": model_id,
                        "level": int(level),
                        "seed": int(seed),
                        "task": f"{task_group} · Task {task_id}",
                        "taskId": task_id,
                        "episode": episode_index,
                        "success": bool(succeeded),
                        "video": relative_video,
                    }
                )

    models = []
    for model_id in MODEL_META:
        model = stats[model_id]
        level_rates = {
            level: rate(values["successes"], values["episodes"])
            for level, values in sorted(model.pop("levels").items())
        }
        model["successRate"] = rate(model["successes"], model["episodes"])
        model["levelRates"] = level_rates
        models.append(model)

    return {"models": models, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.repo_root / "results-data.json"
    output.write_text(json.dumps(build_manifest(args.repo_root), indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
