import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mps_evaluator_routes_f5k_to_pinned_checkpoint():
    source = (ROOT / "scripts" / "run_eval_mps.sh").read_text()

    assert "f5k)" in source
    assert "phawitbinabik/causalvla-model-f-online-dr-5k" in source
    assert "05a56ee5ec79d2879ab1d0cc877946074d151904" in source


def test_mps_evaluator_dry_run_routes_v2_warm_to_pinned_checkpoint():
    env = os.environ.copy()
    env["EVAL_DRY_RUN"] = "1"

    result = subprocess.run(
        [str(ROOT / "scripts" / "run_eval_mps.sh"), "v2_warm", "level_0", "1000", "10"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy.path=phawitbinabik/causalvla-v2-warm" in result.stdout
    assert "--policy.pretrained_revision=119ee2e25cb1e190e89561287dad8c2ffc967d4f" in result.stdout
    assert "--policy.device=mps" in result.stdout
    assert "--ood_level=level_0" in result.stdout
    assert "--seed=1000" in result.stdout
    assert "--eval.n_episodes=10" in result.stdout
