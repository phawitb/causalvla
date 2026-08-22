#!/usr/bin/env python3
"""Install local CausalVLA policy patches into the active LeRobot environment."""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
from pathlib import Path

import lerobot.policies
import lerobot.policies.smolvla.modeling_smolvla as smolvla


POLICIES = {
    "causal_vla": "CausalVLAConfig",
    "causal_vla_warm": "CausalVLAWarmConfig",
    "cover_base": "CoverBaseConfig",
    "cover_safe": "CoverSafeConfig",
    "online_dr": "OnlineDRConfig",
    "pacer_lite": "PacerLiteConfig",
    "rapid_lite": "RapidLiteConfig",
    "rapid_mix": "RapidMixConfig",
    "residual_rapid": "ResidualRapidConfig",
}


def insert_after(source: str, anchor: str, line: str) -> str:
    if line in source:
        return source
    if anchor not in source:
        raise RuntimeError(f"Registration anchor not found: {anchor}")
    return source.replace(anchor, f"{anchor}\n{line}", 1)


def install_eval_policy_view_patch(repo: Path, policies_dir: Path) -> None:
    lerobot_src = policies_dir.parents[1]
    eval_file = lerobot_src / "lerobot" / "scripts" / "lerobot_eval.py"
    if "policy_video_paths" in eval_file.read_text():
        print("Policy-view eval patch already installed")
        return
    patch_file = repo / "lerobot_patches" / "lerobot_eval_policy_view.patch"
    result = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch_file)],
        cwd=lerobot_src.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install policy-view eval patch:\n{result.stdout}\n{result.stderr}")
    print(f"Installed policy-view eval patch: {eval_file}")


def install_fair_sampler_patch(repo: Path, policies_dir: Path) -> None:
    lerobot_src = policies_dir.parents[1]
    train_file = lerobot_src / "lerobot" / "scripts" / "lerobot_train.py"
    if "PairedBatchSampler" in train_file.read_text():
        print("Fair sampler patch already installed")
        return
    patch_file = repo / "lerobot_patches" / "lerobot_fair_sampler.patch"
    result = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch_file)],
        cwd=lerobot_src.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install fair sampler patch:\n{result.stdout}\n{result.stderr}")
    print(f"Installed fair sampler patch: {train_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "policies",
        nargs="*",
        choices=sorted(POLICIES),
        default=list(POLICIES),
        help="Policies to install (default: all)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    policies_dir = Path(inspect.getfile(lerobot.policies)).resolve().parent
    if policies_dir.name != "policies" or not (policies_dir / "smolvla").is_dir():
        raise RuntimeError(f"Unexpected LeRobot policies directory: {policies_dir}")

    init_file = policies_dir / "__init__.py"
    init_source = init_file.read_text()
    import_anchor = "from .act.configuration_act import ACTConfig as ACTConfig"

    for policy_name in args.policies:
        config_name = POLICIES[policy_name]
        source_dir = repo / "lerobot_patches" / policy_name
        target_dir = policies_dir / policy_name
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        registration = (
            f"from .{policy_name}.configuration_{policy_name} "
            f"import {config_name} as {config_name}"
        )
        init_source = insert_after(init_source, import_anchor, registration)
        init_source = insert_after(init_source, '    "ACTConfig",', f'    "{config_name}",')
        print(f"Installed {policy_name}: {target_dir}")

    init_file.write_text(init_source)

    if {"causal_vla", "causal_vla_warm", "pacer_lite"}.intersection(args.policies):
        model_file = Path(inspect.getfile(smolvla)).resolve()
        model_source = model_file.read_text()
        if "def forward_with_latent(" not in model_source:
            method = (repo / "lerobot_patches" / "forward_with_latent.py").read_text().rstrip()
            anchor = "    def sample_actions("
            if anchor not in model_source:
                raise RuntimeError(f"SmolVLA insertion anchor not found: {model_file}")
            model_file.write_text(model_source.replace(anchor, f"{method}\n\n{anchor}", 1))
            print(f"Installed forward_with_latent: {model_file}")
        else:
            print("forward_with_latent already installed")

    install_eval_policy_view_patch(repo, policies_dir)
    install_fair_sampler_patch(repo, policies_dir)

    print(f"LeRobot policies directory: {policies_dir}")


if __name__ == "__main__":
    main()
