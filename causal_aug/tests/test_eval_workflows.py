from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mps_evaluator_routes_f5k_to_pinned_checkpoint():
    source = (ROOT / "scripts" / "run_eval_mps.sh").read_text()

    assert "f5k)" in source
    assert "phawitbinabik/causalvla-model-f-online-dr-5k" in source
    assert "05a56ee5ec79d2879ab1d0cc877946074d151904" in source
