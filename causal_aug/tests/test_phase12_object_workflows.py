from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_object_training_uses_pinned_dataset_and_matched_budget():
    source = (ROOT / "scripts" / "train_libero_object.sh").read_text()

    assert "lerobot/libero_object_image" in source
    assert "e1e080d7df1d0a359dff5c86c222e047549f447f" in source
    assert 'batch_size="${OBJECT_BATCH_SIZE:-16}"' in source
    assert 'steps="${OBJECT_STEPS:-25000}"' in source
    assert "--seed=1000" in source
    assert "online_dr" in source
    assert "causal_vla" in source


def test_object_smoke_does_not_push_or_share_full_output():
    source = (ROOT / "scripts" / "train_libero_object.sh").read_text()

    assert 'push_to_hub="false"' in source
    assert 'save_checkpoint_to_hub="false"' in source
    assert 'run_root="smoke"' in source


def test_object_eval_enforces_fair_protocol_and_exact_revision():
    source = (ROOT / "scripts" / "run_eval_object.sh").read_text()

    assert "^[0-9a-f]{40}$" in source
    assert "--env.task=libero_object" in source
    assert "--eval.batch_size=2" in source
    assert "--eval.use_async_envs=false" in source
    assert 'episodes="${4:-10}"' in source


def test_object_matrix_runs_all_levels_and_preregistered_seeds():
    source = (ROOT / "scripts" / "run_eval_object_matrix.sh").read_text()

    assert "1000 2000 3000" in source
    assert "level_0 level_1 level_2" in source
