from pathlib import Path

from scripts.eval_fair_v1_fixed import FIXED_MODELS, build_fixed_eval_command, build_fixed_matrix, fixed_output_root


def _protocol():
    return {"evaluation": {"seed": 4000, "levels": ["level_0", "level_1", "level_2"], "episodes_per_task": 10, "preflight_episodes_per_task": 1}, "models": {model: {"repo_id": model} for model in FIXED_MODELS}}


def test_full_matrix_is_seed_4000_for_trained_models_only():
    matrix = build_fixed_matrix(_protocol(), "full", seeds=(4000,))
    assert {run.model_id for run in matrix} == {
        "M0-clean",
        "M1-offline-dr",
        "M2-online-dr",
        "M3-v2-warm",
        "M4-v2-warm-030",
        "M5-v2-warm-070",
    }
    assert {run.seed for run in matrix} == {4000}
    assert len(matrix) == 18


def test_full_matrix_expands_requested_fixed_seeds():
    matrix = build_fixed_matrix(_protocol(), "full", seeds=(5000, 6000))
    assert {run.seed for run in matrix} == {5000, 6000}
    assert len(matrix) == 36


def test_full_matrix_orders_all_models_within_each_seed():
    matrix = build_fixed_matrix(_protocol(), "full", seeds=(4000, 5000))
    first_run_for_model = [run for run in matrix if run.level == "level_0"]
    assert [(run.seed, run.model_id) for run in first_run_for_model[:6]] == [
        (4000, model) for model in FIXED_MODELS
    ]
    assert first_run_for_model[6].seed == 5000


def test_command_targets_fixed_tree_and_episode_scope():
    run = build_fixed_matrix(_protocol(), "full", seeds=(5000,))[0]
    rendered = " ".join(build_fixed_eval_command(_protocol(), run, "a" * 40, Path("outputs/eval/fair-v1-fixed/full/M0")))
    assert "fair-v1-fixed/full" in rendered
    assert "--augmentation_scope=episode" in rendered
    assert "--seed=5000" in rendered


def test_object_protocol_targets_object_suite_and_separate_output_tree(tmp_path):
    protocol = _protocol()
    protocol["protocol_version"] = "fair-v1-object"
    protocol["evaluation"]["suite"] = "libero_object"
    run = build_fixed_matrix(protocol, "full", seeds=(4000,))[0]
    rendered = " ".join(build_fixed_eval_command(protocol, run, "a" * 40, Path("out")))

    assert "--env.task=libero_object" in rendered
    assert fixed_output_root(protocol, tmp_path) == tmp_path / "outputs/eval/fair-v1-fixed-object"


def test_validation_rejects_result_from_wrong_evaluation_seed(tmp_path):
    run = build_fixed_matrix(_protocol(), "full", seeds=(5000,))[0]
    payload = {
        "augmentation_scope": "episode",
        "ood_provenance": {"algorithm": "causal_aug.FixedEpisodeOOD", "evaluation_seed": 4000},
        "model_revision": "a" * 40,
        "protocol_sha256": "digest",
        "per_task": [
            {"metrics": {"successes": [True] * 10, "video_paths": ["x"] * 10, "policy_video_paths": ["y"] * 10}}
            for _ in range(10)
        ],
    }
    result = tmp_path / "eval_info.json"
    result.write_text(__import__("json").dumps(payload))

    import pytest
    with pytest.raises(ValueError, match="evaluation seed"):
        from scripts.eval_fair_v1_fixed import validate_fixed_result
        validate_fixed_result(result, run, "a" * 40, "digest")
