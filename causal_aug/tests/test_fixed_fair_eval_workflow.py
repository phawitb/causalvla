from pathlib import Path

from scripts.eval_fair_v1_fixed import FIXED_MODELS, build_fixed_eval_command, build_fixed_matrix


def _protocol():
    return {"evaluation": {"seed": 4000, "levels": ["level_0", "level_1", "level_2"], "episodes_per_task": 10, "preflight_episodes_per_task": 1}, "models": {model: {"repo_id": model} for model in FIXED_MODELS}}


def test_full_matrix_is_seed_4000_for_trained_models_only():
    matrix = build_fixed_matrix(_protocol(), "full")
    assert {run.model_id for run in matrix} == {
        "M0-clean",
        "M1-offline-dr",
        "M2-online-dr",
        "M3-v2-warm",
    }
    assert {run.seed for run in matrix} == {4000}
    assert len(matrix) == 12


def test_command_targets_fixed_tree_and_episode_scope():
    run = build_fixed_matrix(_protocol(), "full")[0]
    rendered = " ".join(build_fixed_eval_command(_protocol(), run, "a" * 40, Path("outputs/eval/fair-v1-fixed/full/M0")))
    assert "fair-v1-fixed/full" in rendered
    assert "--augmentation_scope=episode" in rendered
