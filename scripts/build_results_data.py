#!/usr/bin/env python3
"""Build the static data manifest used by the Results dashboard."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


MODEL_META = {
    "a": {
        "name": "Model A — Standard",
        "description": "Trained only on clean images",
    },
    "b": {
        "name": "Model B — Offline Domain Randomization",
        "description": "Trained on a pre-generated augmented dataset",
    },
    "f": {
        "name": "Model F — Online Domain Randomization",
        "description": (
            "Uses 50% clean and 50% augmented samples during training; "
            "one forward pass per sample"
        ),
    },
    "v2_warm": {
        "name": "V2-Warm (ours)",
        "description": (
            "The new method uses online augmentation, so it trains directly on the original "
            "dataset. Image augmentation is built into the training pipeline. For each clean "
            "image, the model creates an augmented version and processes both images. A new "
            "consistency loss encourages the model to predict similar robot actions for the "
            "clean and augmented images. The action-consistency weight gradually increases "
            "from 0 to 0.05 over the first 10K steps."
        ),
    },
}
RUN_PATTERN = re.compile(r"^model_(a|b|v2_warm|f)_level_(\d+)_(\d+)ep_seed(\d+)$")
FAIR_META = {
    "M0-clean": {"name": "M0 — Clean", "description": "Fair v1 clean baseline"},
    "M1-offline-dr": {"name": "M1 — Offline DR", "description": "Fair v1 deterministic offline augmentation"},
    "M2-online-dr": {"name": "M2 — Online DR", "description": "Fair v1 matched online augmentation"},
    "M3-v2-warm": {"name": "M3 — V2-Warm", "description": "Fair v1 paired consistency training"},
}


def rate(successes: int, episodes: int) -> float:
    return round(successes * 100 / episodes, 1) if episodes else 0.0


def repo_relative_existing(raw_path: str, repo_root: Path) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    if "outputs" in path.parts:
        output_suffix = Path(*path.parts[path.parts.index("outputs"):])
        if (repo_root / output_suffix).is_file():
            return output_suffix.as_posix()
    return None


def build_fixed_collection(repo_root: Path) -> tuple[list, list, list]:
    stats, runs, episodes = {}, [], []
    root = repo_root / "outputs/eval/fair-v1-fixed/full"
    for info_path in sorted(root.glob("*/level_*/seed*/eval_info.json")):
        model_id, level, seed = info_path.parents[2].name, info_path.parents[1].name.removeprefix("level_"), info_path.parent.name.removeprefix("seed")
        if model_id not in FAIR_META:
            continue
        payload = json.loads(info_path.read_text())
        if payload.get("augmentation_scope") != "episode":
            continue
        model = stats.setdefault(model_id, {"id":model_id, **FAIR_META[model_id], "runs":0, "episodes":0, "successes":0, "videos":0, "cleanVideos":0, "policyVideos":0, "levels":defaultdict(lambda:{"episodes":0,"successes":0})})
        model["runs"] += 1
        provenance = payload.get("ood_provenance", {})
        run_id = f"fixed:{model_id}:{level}:seed{seed}"
        runs.append({"id":run_id,"model":model_id,"level":int(level),"seed":int(seed),"ood":{"level":payload.get("ood_level",f"level_{level}"),"params":payload.get("ood_params",{}),"algorithm":provenance.get("algorithm"),"version":provenance.get("version",1),"augmentationScope":"episode"},"modelRevision":payload.get("model_revision"),"protocolSha256":payload.get("protocol_sha256")})
        for task in payload.get("per_task", []):
            metrics=task.get("metrics",{}); successes=metrics.get("successes",[]); clean_paths=metrics.get("video_paths",[]); policy_paths=metrics.get("policy_video_paths",[])
            for index, succeeded in enumerate(successes):
                model["episodes"]+=1; model["successes"]+=int(bool(succeeded)); model["levels"][level]["episodes"]+=1; model["levels"][level]["successes"]+=int(bool(succeeded))
                def relative(paths):
                    if index >= len(paths): return None
                    return repo_relative_existing(paths[index], repo_root)
                clean, policy = relative(clean_paths), relative(policy_paths)
                if not clean and not policy: continue
                model["videos"]+=1; model["cleanVideos"]+=int(clean is not None); model["policyVideos"]+=int(policy is not None)
                episodes.append({"run":run_id,"model":model_id,"level":int(level),"seed":int(seed),"task":f"{task.get('task_group','unknown')} · Task {task.get('task_id','?')}","taskId":task.get("task_id","?"),"episode":index,"success":bool(succeeded),"video":policy or clean,"cleanVideo":clean,"policyVideo":policy})
    models=[]
    for model in stats.values():
        model["levelRates"]={key:rate(value["successes"],value["episodes"]) for key,value in sorted(model.pop("levels").items())}; model["successRate"]=rate(model["successes"],model["episodes"]); models.append(model)
    return models, runs, episodes


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
            "cleanVideos": 0,
            "policyVideos": 0,
            "levels": defaultdict(lambda: {"episodes": 0, "successes": 0}),
        }
        for model_id, meta in MODEL_META.items()
    }
    episodes = []
    runs = []

    def relative_existing(raw_path: str) -> str | None:
        return repo_relative_existing(raw_path, repo_root)

    records = []
    for info_path in sorted(eval_root.glob("*/eval_info.json")):
        match = RUN_PATTERN.match(info_path.parent.name)
        if not match:
            continue
        model_id, level, _declared_episodes, seed = match.groups()
        records.append((info_path, model_id, level, seed))
    fair_root = repo_root / "outputs" / "eval" / "fair-v1" / "full"
    for info_path in sorted(fair_root.glob("*/level_*/seed*/eval_info.json")):
        model_id = info_path.parents[2].name
        if model_id not in FAIR_META:
            continue
        level = info_path.parents[1].name.removeprefix("level_")
        seed = info_path.parent.name.removeprefix("seed")
        if model_id not in stats:
            stats[model_id] = {
                "id": model_id, **FAIR_META[model_id], "runs": 0, "episodes": 0,
                "successes": 0, "videos": 0, "cleanVideos": 0, "policyVideos": 0,
                "levels": defaultdict(lambda: {"episodes": 0, "successes": 0}),
            }
        records.append((info_path, model_id, level, seed))

    for info_path, model_id, level, seed in records:
        model = stats[model_id]
        model["runs"] += 1
        payload = json.loads(info_path.read_text())
        run_id = info_path.parent.name
        provenance = payload.get("ood_provenance", {})
        runs.append(
            {
                "id": run_id,
                "model": model_id,
                "level": int(level),
                "seed": int(seed),
                "ood": {
                    "level": payload.get("ood_level", f"level_{level}"),
                    "params": payload.get("ood_params", {}),
                    "algorithm": provenance.get("algorithm", "causal_aug.OODPerturbation"),
                    "version": provenance.get("version", 1),
                    "processorPosition": provenance.get(
                        "processor_position", "post-env-preprocessing"
                    ),
                },
                "modelRevision": payload.get("model_revision"),
                "protocolSha256": payload.get("protocol_sha256"),
            }
        )

        for task in payload.get("per_task", []):
            metrics = task.get("metrics", {})
            task_successes = metrics.get("successes", [])
            video_paths = metrics.get("video_paths", [])
            policy_video_paths = metrics.get("policy_video_paths", [])
            task_group = task.get("task_group", "unknown")
            task_id = task.get("task_id", "?")

            for episode_index, succeeded in enumerate(task_successes):
                model["episodes"] += 1
                model["successes"] += int(bool(succeeded))
                model["levels"][level]["episodes"] += 1
                model["levels"][level]["successes"] += int(bool(succeeded))

                clean_video = relative_existing(
                    video_paths[episode_index] if episode_index < len(video_paths) else ""
                )
                policy_video = relative_existing(
                    policy_video_paths[episode_index]
                    if episode_index < len(policy_video_paths)
                    else ""
                )
                if not clean_video and not policy_video:
                    continue

                model["videos"] += 1
                model["cleanVideos"] += int(clean_video is not None)
                model["policyVideos"] += int(policy_video is not None)
                episodes.append(
                    {
                        "run": run_id,
                        "model": model_id,
                        "level": int(level),
                        "seed": int(seed),
                        "task": f"{task_group} · Task {task_id}",
                        "taskId": task_id,
                        "episode": episode_index,
                        "success": bool(succeeded),
                        "video": policy_video or clean_video,
                        "cleanVideo": clean_video,
                        "policyVideo": policy_video,
                    }
                )

    models = []
    for model_id in stats:
        model = stats[model_id]
        level_rates = {
            level: rate(values["successes"], values["episodes"])
            for level, values in sorted(model.pop("levels").items())
        }
        model["successRate"] = rate(model["successes"], model["episodes"])
        model["levelRates"] = level_rates
        models.append(model)

    fixed_models, fixed_runs, fixed_episodes = build_fixed_collection(repo_root)
    return {"models": models, "runs": runs, "episodes": episodes, "fixedModels": fixed_models, "fixedRuns": fixed_runs, "fixedEpisodes": fixed_episodes}


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
