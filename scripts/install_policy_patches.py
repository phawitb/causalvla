#!/usr/bin/env python3
"""Install local CausalVLA policy patches into the active LeRobot environment."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import lerobot.policies
import lerobot.policies.smolvla.modeling_smolvla as smolvla


POLICIES = {
    "causal_vla": "CausalVLAConfig",
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

    if {"causal_vla", "pacer_lite"}.intersection(args.policies):
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

    print(f"LeRobot policies directory: {policies_dir}")


if __name__ == "__main__":
    main()
