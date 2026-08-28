import ast
from pathlib import Path


def _policy_mapping() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "scripts" / "install_policy_patches.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "POLICIES" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("POLICIES mapping not found")


def test_installer_registers_both_cover_variants():
    policies = _policy_mapping()

    assert policies["cover_base"] == "CoverBaseConfig"
    assert policies["cover_safe"] == "CoverSafeConfig"


def test_cover_does_not_request_forward_with_latent_patch():
    source = (Path(__file__).resolve().parents[2] / "scripts" / "install_policy_patches.py").read_text()

    patch_condition = source.split(
        'if {"causal_vla", "causal_vla_warm", "pacer_lite"}', 1
    )[1].split(":", 1)[0]
    assert "cover_base" not in patch_condition
    assert "cover_safe" not in patch_condition


def test_installer_selects_patch_root_for_source_and_site_packages():
    from scripts.install_policy_patches import patch_invocation

    assert patch_invocation(Path("/repo/lerobot/src")) == (Path("/repo/lerobot"), "-p1")
    assert patch_invocation(Path("/env/lib/python3.12/site-packages")) == (
        Path("/env/lib/python3.12/site-packages"),
        "-p2",
    )


def test_installer_applies_required_sampler_before_optional_eval(monkeypatch, tmp_path):
    import scripts.install_policy_patches as installer

    calls = []
    monkeypatch.setattr(installer, "install_fair_sampler_patch", lambda repo, policies: calls.append("sampler"))
    monkeypatch.setattr(installer, "install_eval_policy_view_patch", lambda repo, policies: calls.append("eval"))

    installer.install_shared_patches(tmp_path, tmp_path)

    assert calls == ["sampler", "eval"]


def test_installer_skips_optional_eval_patch_when_eval_module_is_absent(tmp_path, capsys):
    from scripts.install_policy_patches import install_eval_policy_view_patch

    policies_dir = tmp_path / "site-packages/lerobot/policies"
    policies_dir.mkdir(parents=True)

    install_eval_policy_view_patch(tmp_path, policies_dir)

    assert "Skipping policy-view eval patch" in capsys.readouterr().out


def test_installer_upgrades_existing_sampler_to_drop_odd_online_dr_batch(tmp_path):
    from scripts.install_policy_patches import install_fair_sampler_patch

    policies_dir = tmp_path / "site-packages/lerobot/policies"
    train_file = tmp_path / "site-packages/lerobot/scripts/lerobot_train.py"
    policies_dir.mkdir(parents=True)
    train_file.parent.mkdir(parents=True)
    train_file.write_text(
        "from causal_aug import PairedBatchSampler\n"
        "pin_memory=device.type == 'cuda', drop_last=False, collate_fn=collate_fn,\n"
    )

    install_fair_sampler_patch(tmp_path, policies_dir)

    assert "drop_last=bool(getattr(cfg.policy, 'exact_balance', False))" in train_file.read_text()
