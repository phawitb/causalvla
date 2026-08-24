# Train M0–M3 on LIBERO Object, Goal, and Long

This workflow runs 12 full Fair Protocol v1 jobs sequentially on one GPU:

- Suites: LIBERO Object, Goal, and Long (`libero_10`)
- Models: M0-clean, M1-offline-dr, M2-online-dr, and M3-v2-warm
- Training contract: 25,000 steps, batch size 16, seed 1000, checkpoint every 5,000 steps

The runner validates CUDA, Hugging Face authentication, policy patches, and immutable source-dataset revisions before training. When an M1 offline dataset does not exist in the `phawitbinabik` namespace, it materializes and uploads that suite's deterministic clean/augmented dataset before starting any model training. Existing completed runs are skipped.

## GPU server setup

```bash
cd ~/projects/causalvla
conda activate causalvla
git pull origin main
python scripts/install_policy_patches.py online_dr causal_vla_warm
huggingface-cli whoami
```

Inspect all generated preparation and training commands without downloading, uploading, or training:

```bash
./scripts/train_all_m_libero.sh --dry-run
```

Run the full matrix on physical GPU 0:

```bash
mkdir -p logs
nohup ./scripts/train_all_m_libero.sh --device cuda:0 \
  > logs/train_all_m_libero.log 2>&1 &
echo $! > logs/train_all_m_libero.pid
```

Monitor it with:

```bash
tail -f logs/train_all_m_libero.log
```

Resume interrupted runs while continuing to skip completed runs:

```bash
nohup ./scripts/train_all_m_libero.sh --device cuda:0 --resume \
  >> logs/train_all_m_libero.log 2>&1 &
```

Use comma-separated subsets when needed:

```bash
./scripts/train_all_m_libero.sh \
  --suites object,goal \
  --models M0-clean,M2-online-dr \
  --device cuda:1
```

Training outputs are stored under `outputs/train/fair-v1-multisuite/<suite>/<model>`. Materialized M1 datasets are stored under `outputs/datasets/fair-v1/<suite>` and uploaded to their suite-specific Hugging Face dataset repositories.
