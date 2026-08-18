import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v2_warm_dry_run_emits_preregistered_training_contract(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "V2_WARM_DRY_RUN": "1",
            "V2_WARM_OUTPUT_DIR": str(tmp_path / "output"),
            "V2_WARM_STEPS": "25000",
            "V2_WARM_BATCH_SIZE": "16",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_v2_warm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy.type=causal_vla_warm" in result.stdout
    assert "--policy.lambda_action=0.05" in result.stdout
    assert "--policy.action_warmup_steps=10000" in result.stdout
    assert "--policy.use_latent_loss=false" in result.stdout
    assert "--policy.clean_task_weight=0.5" in result.stdout
    assert "--policy.augmented_task_weight=0.5" in result.stdout
    assert "--steps=25000" in result.stdout
    assert "--batch_size=16" in result.stdout
    assert not (tmp_path / "output").exists()


def test_v2_warm_leaves_new_output_directory_for_lerobot_to_create(tmp_path):
    output_dir = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_train = fake_bin / "lerobot-train"
    fake_train.write_text(
        "#!/usr/bin/env bash\n"
        f'[[ ! -e "{output_dir}" ]] || exit 42\n'
        "exit 0\n"
    )
    fake_train.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "V2_WARM_OUTPUT_DIR": str(output_dir),
            "V2_WARM_STEPS": "2",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_v2_warm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
