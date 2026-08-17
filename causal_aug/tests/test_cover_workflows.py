from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mps_smoke_covers_both_policy_types_and_hard_contracts():
    source = (ROOT / "scripts" / "smoke_cover_mps.sh").read_text()

    assert "cover_base|cover_safe" in source
    assert "--steps=2" in source
    assert "--batch_size=8" in source
    assert "--policy.device=mps" in source
    assert "cover/forward_count:1.000" in source
    assert "model.safetensors" in source
    assert "Traceback|RuntimeError|CUDA out of memory|nan" in source


def test_eval_requires_exact_revision_and_uses_fair_protocol():
    source = (ROOT / "scripts" / "run_eval_cover.sh").read_text()

    assert "^[0-9a-f]{40}$" in source
    assert "--eval.batch_size=2" in source
    assert "--eval.use_async_envs=false" in source
    assert "--env.task=libero_spatial" in source
    assert "--policy.pretrained_revision" in source
    assert 'episodes="${4:-10}"' in source
    assert "eval_info.json" in source


def test_eval_routes_pilot_names_to_pilot_repositories():
    source = (ROOT / "scripts" / "run_eval_cover.sh").read_text()

    assert "cover_base_pilot" in source
    assert "phawitbinabik/causalvla-cover-base-pilot" in source
    assert "cover_safe_pilot" in source
    assert "phawitbinabik/causalvla-cover-safe-pilot" in source


def test_eval_routes_20k_name_to_public_checkpoint_repository():
    source = (ROOT / "scripts" / "run_eval_cover.sh").read_text()

    assert "cover_base_20k" in source
    assert "phawitbinabik/causalvla-cover-base-20k" in source


def test_phase10_documents_pilot_and_full_training_commands():
    source = (ROOT / "worklog" / "phase10.md").read_text()

    assert "--steps=5000" in source
    assert "--steps=25000" in source
    assert "--policy.type=cover_base" in source
    assert "--policy.type=cover_safe" in source
    assert "5 episodes/task" in source
    assert "10 episodes/task" in source
