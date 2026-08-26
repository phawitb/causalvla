import subprocess
import sys
from pathlib import Path

import pytest

from scripts.train_all_m_libero import MODEL_IDS, SUITES, build_run_matrix, missing_policy_modules, should_resume_run


ROOT = Path(__file__).resolve().parents[2]


def test_default_matrix_covers_four_models_across_three_suites():
    runs = build_run_matrix(ROOT, tuple(SUITES), MODEL_IDS)

    assert [(run.suite, run.model_id) for run in runs] == [
        (suite, model) for suite in ("object", "goal", "long") for model in MODEL_IDS
    ]
    assert len(runs) == 12


@pytest.mark.parametrize(
    ("suite", "dataset_repo", "revision"),
    [
        ("object", "lerobot/libero_object_image", "e1e080d7df1d0a359dff5c86c222e047549f447f"),
        ("goal", "lerobot/libero_goal_image", "91a97115558b5b611200a432d9c82e4f30991b60"),
        ("long", "lerobot/libero_10_image", "7e324b526699f444044952c82ce3f438e8d300d0"),
    ],
)
def test_suite_protocols_lock_official_dataset_revisions(suite, dataset_repo, revision):
    runs = build_run_matrix(ROOT, (suite,), ("M0-clean",))

    assert runs[0].dataset_repo == dataset_repo
    assert runs[0].dataset_revision == revision


def test_m1_runs_require_suite_specific_offline_revision():
    runs = build_run_matrix(ROOT, ("object", "goal", "long"), ("M1-offline-dr",))

    assert [run.offline_repo for run in runs] == [
        "phawitbinabik/libero-object-offline-dr-fair-v1",
        "phawitbinabik/libero-goal-offline-dr-fair-v1",
        "phawitbinabik/libero-long-offline-dr-fair-v1",
    ]
    assert all(run.needs_offline_dataset for run in runs)


def test_dry_run_prints_all_training_commands_without_starting_training():
    result = subprocess.run(
        [sys.executable, "scripts/train_all_m_libero.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("scripts/train_fair_v1.py") == 12
    assert "materialize_fair_offline.py" in result.stdout
    assert "fair_object.json" in result.stdout
    assert "fair_goal.json" in result.stdout
    assert "fair_long.json" in result.stdout


def test_unknown_suite_is_rejected_before_any_run_starts():
    result = subprocess.run(
        [sys.executable, "scripts/train_all_m_libero.py", "--suites", "object,unknown", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unknown suite" in result.stderr.lower()


def test_preflight_reports_each_policy_patch_that_is_not_importable():
    available = {"lerobot.policies.smolvla"}

    assert missing_policy_modules(lambda name: object() if name in available else None) == [
        "lerobot.policies.online_dr",
        "lerobot.policies.causal_vla_warm",
    ]


def test_resume_only_applies_to_runs_that_already_have_output(tmp_path):
    runs = build_run_matrix(ROOT, ("object",), ("M0-clean", "M1-offline-dr"))
    started = runs[0]
    not_started = runs[1]
    object.__setattr__(started, "output_dir", tmp_path / "started")
    object.__setattr__(not_started, "output_dir", tmp_path / "not-started")
    started.output_dir.mkdir()

    assert should_resume_run(started, requested=True)
    assert not should_resume_run(not_started, requested=True)
    assert not should_resume_run(started, requested=False)
