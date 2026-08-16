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

    patch_condition = source.split('if {"causal_vla", "pacer_lite"}', 1)[1].split(":", 1)[0]
    assert "cover_base" not in patch_condition
    assert "cover_safe" not in patch_condition
