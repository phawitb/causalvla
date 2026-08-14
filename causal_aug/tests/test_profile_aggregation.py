import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "aggregate_intervention_profiles.py"
SPEC = importlib.util.spec_from_file_location("aggregate_intervention_profiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_uses_guard_and_uncertainty_penalty(tmp_path):
    profiles = []
    seed_rows = [
        [
            {"family": "stable", "intensity": 0.5, "action_sensitivity": 0.4, "guard_pass_rate": 1.0},
            {"family": "unstable", "intensity": 1.0, "action_sensitivity": 0.8, "guard_pass_rate": 1.0},
            {"family": "unsafe", "intensity": 1.0, "action_sensitivity": 1.0, "guard_pass_rate": 0.5},
        ],
        [
            {"family": "stable", "intensity": 0.5, "action_sensitivity": 0.4, "guard_pass_rate": 1.0},
            {"family": "unstable", "intensity": 1.0, "action_sensitivity": 0.1, "guard_pass_rate": 1.0},
            {"family": "unsafe", "intensity": 1.0, "action_sensitivity": 1.0, "guard_pass_rate": 0.5},
        ],
    ]
    for index, rows in enumerate(seed_rows):
        path = tmp_path / f"seed_{index}.json"
        path.write_text(json.dumps(rows))
        profiles.append(path)

    result = MODULE.aggregate(profiles, min_guard_rate=0.95, uncertainty_penalty=1.0, top_k=1)

    assert result["candidates"][0]["family"] == "stable"
    unsafe = next(row for row in result["ranking"] if row["family"] == "unsafe")
    assert not unsafe["eligible"]
    assert result["candidates"][0]["risk_weight"] == pytest.approx(1.0)
