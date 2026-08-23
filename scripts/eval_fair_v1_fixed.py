#!/usr/bin/env python3
"""Run fixed-per-episode fair-v1 evaluations without touching original results."""

from __future__ import annotations
import argparse, json, shlex, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

from scripts.fair_protocol import SHA_PATTERN, load_protocol, protocol_hash, resolve_hf_revision, validate_protocol

FIXED_MODELS = ("M0-clean", "M2-online-dr", "M3-v2-warm")

@dataclass(frozen=True)
class EvalRun:
    model_id: str
    level: str
    seed: int
    episodes_per_task: int

def build_fixed_matrix(protocol: dict, mode: str) -> list[EvalRun]:
    if mode == "preflight":
        levels, episodes = ("level_0", "level_2"), protocol["evaluation"]["preflight_episodes_per_task"]
    elif mode == "full":
        levels, episodes = tuple(protocol["evaluation"]["levels"]), protocol["evaluation"]["episodes_per_task"]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [EvalRun(model, level, 4000, episodes) for model in FIXED_MODELS for level in levels]

def build_fixed_eval_command(protocol: dict, run: EvalRun, revision: str, output_dir: Path) -> list[str]:
    if not SHA_PATTERN.fullmatch(revision):
        raise ValueError("evaluation requires an immutable model revision")
    return [sys.executable, str(Path(__file__).with_name("eval_ood.py")),
        f"--policy.path={protocol['models'][run.model_id]['repo_id']}", f"--policy.pretrained_revision={revision}",
        "--policy.device=mps", "--env.type=libero", "--env.task=libero_spatial",
        '--rename_map={"observation.images.image2":"observation.images.wrist_image"}',
        f"--ood_level={run.level}", "--augmentation_scope=episode", f"--eval.n_episodes={run.episodes_per_task}",
        "--eval.batch_size=2", "--eval.use_async_envs=false", f"--output_dir={output_dir}", "--seed=4000"]

def validate_fixed_result(path: Path, run: EvalRun, revision: str, digest: str) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("augmentation_scope") != "episode" or payload.get("ood_provenance", {}).get("algorithm") != "causal_aug.FixedEpisodeOOD": raise ValueError("fixed augmentation provenance mismatch")
    if payload.get("model_revision") != revision or payload.get("protocol_sha256") != digest: raise ValueError("pinned provenance mismatch")
    if len(payload.get("per_task", [])) != 10: raise ValueError("evaluation task count is incomplete")
    for task in payload["per_task"]:
        for key in ("successes", "video_paths", "policy_video_paths"):
            if len(task.get("metrics", {}).get(key, [])) != run.episodes_per_task: raise ValueError(f"evaluation {key} count is incomplete")
    return payload

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", type=Path, default=Path("configs/fair_v1.json")); parser.add_argument("--mode", choices=("preflight","full"), required=True); parser.add_argument("--model", choices=FIXED_MODELS); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--revision", action="append", default=[]); args = parser.parse_args()
    protocol_path=args.protocol.resolve(); protocol=load_protocol(protocol_path); validate_protocol(protocol, protocol_path); supplied=dict(item.split("=",1) for item in args.revision); digest=protocol_hash(protocol)
    for run in build_fixed_matrix(protocol,args.mode):
        if args.model and run.model_id != args.model: continue
        revision=supplied.get(run.model_id)
        if revision is None:
            try: revision=resolve_hf_revision("model",protocol["models"][run.model_id]["repo_id"])
            except Exception:
                if args.dry_run: print(f"PENDING {run.model_id}: immutable revision unavailable"); continue
                raise
        output=protocol_path.parents[1]/"outputs/eval/fair-v1-fixed"/args.mode/run.model_id/run.level/"seed4000"; result=output/"eval_info.json"
        if result.is_file(): validate_fixed_result(result,run,revision,digest); continue
        command=build_fixed_eval_command(protocol,run,revision,output)
        if args.dry_run: print(shlex.join(command)); continue
        subprocess.run(command,check=True); payload=json.loads(result.read_text()); payload.update(model_revision=revision,protocol_sha256=digest); result.write_text(json.dumps(payload,indent=2,sort_keys=True)); validate_fixed_result(result,run,revision,digest)

if __name__ == "__main__": main()
