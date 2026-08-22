import json
import shlex
from pathlib import Path

import pytest

from scripts.eval_fair_v1 import EvalExpectation, build_eval_command, evaluation_matrix, validate_eval_result
from scripts.fair_protocol import load_protocol
from scripts.summarize_fair_v1 import summarize_runs


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_protocol(ROOT / "configs/fair_v1.json")


def test_primary_matrix_has_twelve_paired_runs():
    matrix = evaluation_matrix(PROTOCOL, "full")
    assert len(matrix) == 12
    assert {run.seed for run in matrix} == {4000}
    assert {run.level for run in matrix} == {"level_0", "level_1", "level_2"}
    assert {run.episodes_per_task for run in matrix} == {10}


def test_preflight_has_four_level_zero_runs():
    matrix = evaluation_matrix(PROTOCOL, "preflight")
    assert len(matrix) == 4
    assert {run.level for run in matrix} == {"level_0"}
    assert {run.episodes_per_task for run in matrix} == {1}


def test_eval_command_is_pinned_and_uses_seed_4000(tmp_path):
    command = shlex.join(
        build_eval_command(PROTOCOL, "M0-clean", "level_0", 1, "a" * 40, tmp_path)
    )
    assert "--policy.pretrained_revision=" + "a" * 40 in command
    assert "--seed=4000" in command
    assert "--policy.device=mps" in command
    assert "--env.task=libero_spatial" in command


def _write_result(path: Path, episodes: int = 1):
    payload = {
        "ood_level": "level_0",
        "ood_provenance": {"seed": 4000},
        "model_revision": "a" * 40,
        "protocol_sha256": "p",
        "per_task": [],
    }
    for task in range(10):
        payload["per_task"].append(
            {
                "task_id": task,
                "metrics": {
                    "successes": [True] * episodes,
                    "video_paths": [f"clean-{task}-{i}.mp4" for i in range(episodes)],
                    "policy_video_paths": [f"policy-{task}-{i}.mp4" for i in range(episodes)],
                },
            }
        )
    path.write_text(json.dumps(payload))


def test_result_validator_accepts_complete_matrix(tmp_path):
    path = tmp_path / "eval_info.json"
    _write_result(path)
    expected = EvalExpectation("level_0", 4000, 10, 1, "a" * 40, "p")
    assert validate_eval_result(path, expected)["ood_level"] == "level_0"


def test_result_validator_rejects_incomplete_tasks(tmp_path):
    path = tmp_path / "eval_info.json"
    _write_result(path)
    payload = json.loads(path.read_text())
    payload["per_task"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task count"):
        validate_eval_result(path, EvalExpectation("level_0", 4000, 10, 1, "a" * 40, "p"))


def test_summary_reports_degradation_and_delta_vs_m0(tmp_path):
    paths = []
    rates = {
        ("M0-clean", "level_0"): 0.8,
        ("M0-clean", "level_2"): 0.3,
        ("M1-offline-dr", "level_0"): 0.7,
        ("M1-offline-dr", "level_2"): 0.4,
    }
    for (model, level), rate in rates.items():
        path = tmp_path / model / level / "seed4000" / "eval_info.json"
        path.parent.mkdir(parents=True)
        successes = [True] * int(rate * 10) + [False] * (10 - int(rate * 10))
        path.write_text(json.dumps({"per_task": [{"task_id": 0, "metrics": {"successes": successes}}]}))
        paths.append(path)
    report = summarize_runs(paths)
    cell = report["M1-offline-dr"]["level_2"]
    assert cell["success_rate"] == pytest.approx(0.4)
    assert cell["degradation_from_level_0"] == pytest.approx(-0.3)
    assert cell["delta_vs_m0"] == pytest.approx(0.1)
