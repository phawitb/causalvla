#!/usr/bin/env python3
"""Profile action sensitivity and visual-semantic preservation by intervention family."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from causal_aug import INTERVENTION_FAMILIES, InterventionBank
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.common.vla_utils import make_att_2d_masks


def pooled_visual_embedding(policy, images: list[torch.Tensor]) -> torch.Tensor:
    embeddings = [policy.model.vlm_with_expert.embed_image(image).float().mean(dim=1) for image in images]
    return torch.stack(embeddings, dim=1).mean(dim=1)


def task_loss(losses: torch.Tensor, action_dim: int, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    losses = losses[:, :, :action_dim]
    if action_is_pad is None:
        return losses.mean()
    valid = ~action_is_pad
    denominator = (valid.sum() * action_dim).clamp_min(1)
    return (losses * valid.unsqueeze(-1)).sum() / denominator


def flow_forward_with_velocity(model, images, masks, tokens, token_masks, state, actions, noise, time):
    """Training forward that tolerates legacy serialized language-mask lengths."""
    expanded_time = time[:, None, None]
    noisy_actions = expanded_time * noise + (1 - expanded_time) * actions
    target_velocity = noise - actions
    prefix, prefix_pad, prefix_att = model.embed_prefix(images, masks, tokens, token_masks, state=state)
    suffix, suffix_pad, suffix_att = model.embed_suffix(noisy_actions, time)
    if prefix_pad.shape[1] != prefix.shape[1]:
        missing = prefix.shape[1] - prefix_pad.shape[1]
        if missing < 0:
            raise RuntimeError("prefix padding mask is longer than prefix embeddings")
        prefix_pad = torch.cat(
            [prefix_pad, torch.ones(prefix_pad.shape[0], missing, dtype=torch.bool, device=prefix_pad.device)],
            dim=1,
        )
    if prefix_att.shape[1] != prefix.shape[1]:
        if prefix_att.shape[1] > prefix.shape[1]:
            prefix_att = prefix_att[:, : prefix.shape[1]]
        else:
            missing = prefix.shape[1] - prefix_att.shape[1]
            prefix_att = torch.cat(
                [prefix_att, torch.zeros(prefix_att.shape[0], missing, dtype=prefix_att.dtype, device=prefix_att.device)],
                dim=1,
            )
    if suffix_pad.shape[1] != suffix.shape[1]:
        suffix_pad = suffix_pad[:, : suffix.shape[1]]
    if suffix_att.shape[1] != suffix.shape[1]:
        suffix_att = suffix_att[:, : suffix.shape[1]]
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    att = torch.cat([prefix_att, suffix_att], dim=1)
    att_2d = make_att_2d_masks(pad, att)
    positions = torch.cumsum(pad, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].float()
    velocity = model.action_out_proj(suffix_out)
    return F.mse_loss(target_velocity, velocity, reduction="none"), velocity


@torch.inference_mode()
def profile_batch(policy, batch, bank, families, intensities, semantic_threshold):
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    # Some serialized LeRobot 0.6 processors restore an empty language mask
    # while retaining padded token ids. Training-style flow forwards require
    # the two sequence lengths to agree. The profiler uses every restored token
    # when the saved mask is structurally invalid; normal masks are untouched.
    if token_masks.ndim != 2 or token_masks.shape != tokens.shape:
        token_masks = torch.ones_like(tokens, dtype=torch.bool)
    actions = policy.prepare_action(batch)
    action_is_pad = batch.get("action_is_pad")
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = policy.model.sample_time(actions.shape[0], actions.device)
    clean_losses, clean_velocity = flow_forward_with_velocity(
        policy.model,
        images, masks, tokens, token_masks, state, actions, noise, time
    )
    clean_embedding = pooled_visual_embedding(policy, images)
    action_dim = policy.config.action_feature.shape[0]
    clean_loss = task_loss(clean_losses, action_dim, action_is_pad).item()

    rows = []
    for family in families:
        for intensity in intensities:
            augmented = bank.apply(images, family, intensity)
            losses, velocity = flow_forward_with_velocity(
                policy.model,
                augmented, masks, tokens, token_masks, state, actions, noise, time
            )
            aug_embedding = pooled_visual_embedding(policy, augmented)
            semantic_similarity = F.cosine_similarity(clean_embedding, aug_embedding, dim=-1).mean().item()
            sensitivity = (velocity[:, :, :action_dim] - clean_velocity[:, :, :action_dim]).abs().mean().item()
            aug_loss = task_loss(losses, action_dim, action_is_pad).item()
            rows.append(
                {
                    "family": family,
                    "intensity": intensity,
                    "action_sensitivity": sensitivity,
                    "clean_task_loss": clean_loss,
                    "augmented_task_loss": aug_loss,
                    "loss_increase": aug_loss - clean_loss,
                    "semantic_similarity": semantic_similarity,
                    "guard_pass": semantic_similarity >= semantic_threshold,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default="phawitbinabik/causalvla-model-f-online-dr")
    parser.add_argument("--revision", default="997d94a9325bc359422cd3cf54bd74b0a4c9be98")
    parser.add_argument("--dataset", default="lerobot/libero_spatial_image")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--families", nargs="+", choices=INTERVENTION_FAMILIES, default=list(INTERVENTION_FAMILIES))
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--semantic-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase8/intervention_profile"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = PreTrainedConfig.from_pretrained(
        args.policy_path, revision=args.revision, device=args.device
    )
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=config, revision=args.revision)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_path,
        pretrained_revision=args.revision,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    metadata = LeRobotDatasetMetadata(args.dataset)
    dataset = LeRobotDataset(
        args.dataset,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.samples].tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, collate_fn=lerobot_collate_fn, num_workers=0
    )
    bank = InterventionBank()
    rows = []
    for batch_idx, raw_batch in enumerate(loader):
        for key in dataset.meta.camera_keys:
            if key in raw_batch and raw_batch[key].dtype == torch.uint8:
                raw_batch[key] = raw_batch[key].float() / 255
        batch = preprocessor(raw_batch)
        batch_rows = profile_batch(
            policy, batch, bank, args.families, args.intensities, args.semantic_threshold
        )
        task_ids = raw_batch.get("task_index")
        for row in batch_rows:
            row["batch"] = batch_idx
            row["task_ids"] = [] if task_ids is None else task_ids.tolist()
        rows.extend(batch_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "profile.json").write_text(json.dumps(rows, indent=2))
    with (args.output_dir / "profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["intensity"])].append(row)
    summary = []
    for (family, intensity), items in grouped.items():
        summary.append({
            "family": family,
            "intensity": intensity,
            "action_sensitivity": sum(x["action_sensitivity"] for x in items) / len(items),
            "loss_increase": sum(x["loss_increase"] for x in items) / len(items),
            "semantic_similarity": sum(x["semantic_similarity"] for x in items) / len(items),
            "guard_pass_rate": sum(x["guard_pass"] for x in items) / len(items),
        })
    summary.sort(key=lambda x: x["action_sensitivity"], reverse=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    markdown = [
        "# Phase 8 Intervention Profile",
        "",
        f"Samples: {args.samples} | seed: {args.seed} | semantic guard: >= {args.semantic_threshold:.2f}",
        "",
        "| Rank | Family | Intensity | Action sensitivity | Loss increase | Semantic similarity | Guard pass |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(summary, start=1):
        markdown.append(
            f"| {rank} | {item['family']} | {item['intensity']:.2f} | "
            f"{item['action_sensitivity']:.6f} | {item['loss_increase']:+.6f} | "
            f"{item['semantic_similarity']:.4f} | {item['guard_pass_rate']:.0%} |"
        )
    markdown.extend([
        "",
        "> Action sensitivity is a shared-noise diagnostic, not a causal-effect estimate or an evaluation success rate.",
        "",
    ])
    (args.output_dir / "summary.md").write_text("\n".join(markdown))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
