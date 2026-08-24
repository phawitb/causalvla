import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_results_data.py"


class BuildResultsDataTest(unittest.TestCase):
    def test_fixed_videos_survive_worktree_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as source_dir:
            root = Path(temp_dir)
            relative_video = Path("outputs/eval/fair-v1-fixed/full/M0-clean/level_0/seed4000/videos/task0/eval_episode_0.mp4")
            actual_video = root / relative_video
            actual_video.parent.mkdir(parents=True)
            actual_video.touch()
            recorded_video = Path(source_dir) / relative_video
            run_dir = root / relative_video.parents[2]
            payload = {
                "augmentation_scope": "episode",
                "ood_level": "level_0",
                "per_task": [{"task_id": 0, "metrics": {
                    "successes": [True], "video_paths": [str(recorded_video)]
                }}],
            }
            (run_dir / "eval_info.json").write_text(json.dumps(payload))
            output = root / "results-data.json"

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(root), "--output", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text())
            self.assertEqual(data["fixedEpisodes"][0]["video"], relative_video.as_posix())

    def test_separates_fixed_episode_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "outputs/eval/fair-v1-fixed/full/M0-clean/level_1/seed4000"
            video_dir = run_dir / "videos/task0"
            policy_dir = video_dir / "policy_view"
            policy_dir.mkdir(parents=True)
            clean = video_dir / "eval_episode_0.mp4"; clean.touch()
            policy = policy_dir / "eval_episode_0.mp4"; policy.touch()
            payload = {"augmentation_scope":"episode", "ood_level":"level_1", "ood_provenance":{"algorithm":"causal_aug.FixedEpisodeOOD","version":1,"evaluation_seed":4000}, "per_task":[{"task_group":"libero_spatial","task_id":0,"metrics":{"successes":[True],"video_paths":[str(clean)],"policy_video_paths":[str(policy)]}}]}
            (run_dir / "eval_info.json").write_text(json.dumps(payload))
            output = root / "results-data.json"
            completed = subprocess.run([sys.executable, str(SCRIPT), "--repo-root", str(root), "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text())
            self.assertEqual([model["id"] for model in data["fixedModels"]], ["M0-clean"])
            self.assertEqual(data["fixedRuns"][0]["ood"]["augmentationScope"], "episode")
            self.assertEqual(data["fixedEpisodes"][0]["model"], "M0-clean")
    def test_includes_fair_v1_provenance_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "outputs/eval/fair-v1/full/M0-clean/level_0/seed4000"
            run_dir.mkdir(parents=True)
            payload = {
                "ood_level": "level_0",
                "ood_provenance": {"seed": 4000},
                "model_revision": "a" * 40,
                "protocol_sha256": "p",
                "per_task": [{"task_id": 0, "metrics": {"successes": [True]}}],
            }
            (run_dir / "eval_info.json").write_text(json.dumps(payload))
            output = root / "results-data.json"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(root), "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text())
            fair = next(model for model in data["models"] if model["id"] == "M0-clean")
            self.assertEqual(fair["successRate"], 100.0)
            fair_run = next(run for run in data["runs"] if run["model"] == "M0-clean")
            self.assertEqual(fair_run["modelRevision"], "a" * 40)
            self.assertEqual(fair_run["protocolSha256"], "p")

    def test_builds_dashboard_data_for_only_requested_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_root = root / "outputs" / "eval" / "full"
            fixtures = [
                ("model_a_level_0_2ep_seed1000", [True, False]),
                ("model_b_level_1_2ep_seed2000", [False, False]),
                ("model_v2_warm_level_2_2ep_seed3000", [True, True]),
                ("model_f_level_1_2ep_seed1000", [True, False]),
                ("model_v2_level_0_2ep_seed1000", [True, True]),
            ]

            for run_name, successes in fixtures:
                run_dir = eval_root / run_name
                video_dir = run_dir / "videos" / "libero_spatial_3"
                policy_video_dir = video_dir / "policy_view"
                video_dir.mkdir(parents=True)
                policy_video_dir.mkdir(parents=True)
                paths = []
                policy_paths = []
                for episode in range(2):
                    video = video_dir / f"eval_episode_{episode}.mp4"
                    policy_video = policy_video_dir / f"eval_episode_{episode}.mp4"
                    video.touch()
                    policy_video.touch()
                    paths.append(str(video))
                    policy_paths.append(str(policy_video))
                payload = {
                    "per_task": [{
                        "task_group": "libero_spatial",
                        "task_id": 3,
                        "metrics": {
                            "successes": successes,
                            "video_paths": paths,
                            "policy_video_paths": policy_paths,
                        },
                    }],
                    "ood_level": run_name.split("_level_")[1].split("_")[0],
                    "ood_params": {"noise_sigma": 0.2},
                    "ood_provenance": {
                        "algorithm": "causal_aug.OODPerturbation",
                        "version": 1,
                        "processor_position": "post-env-preprocessing",
                    },
                }
                (run_dir / "eval_info.json").write_text(json.dumps(payload))

            output = root / "results-data.json"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(root), "--output", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text())
            self.assertEqual([model["id"] for model in data["models"]], ["a", "b", "f", "v2_warm"])
            self.assertEqual(
                [(model["name"], model["description"]) for model in data["models"]],
                [
                    ("Model A — Standard", "Trained only on clean images"),
                    ("Model B — Offline Domain Randomization", "Trained on a pre-generated augmented dataset"),
                    (
                        "Model F — Online Domain Randomization",
                        "Uses 50% clean and 50% augmented samples during training; one forward pass per sample",
                    ),
                    (
                        "V2-Warm (ours)",
                        "The new method uses online augmentation, so it trains directly on the original dataset. Image augmentation is built into the training pipeline. For each clean image, the model creates an augmented version and processes both images. A new consistency loss encourages the model to predict similar robot actions for the clean and augmented images. The action-consistency weight gradually increases from 0 to 0.05 over the first 10K steps.",
                    ),
                ],
            )
            self.assertEqual(data["models"][0]["episodes"], 2)
            self.assertEqual(data["models"][0]["successRate"], 50.0)
            self.assertEqual(data["models"][3]["levelRates"]["2"], 100.0)
            self.assertEqual(len(data["episodes"]), 8)
            self.assertEqual(data["episodes"][0]["task"], "libero_spatial · Task 3")
            self.assertEqual(
                data["episodes"][0].get("cleanVideo"),
                "outputs/eval/full/model_a_level_0_2ep_seed1000/videos/libero_spatial_3/eval_episode_0.mp4",
            )
            self.assertEqual(
                data["episodes"][0].get("policyVideo"),
                "outputs/eval/full/model_a_level_0_2ep_seed1000/videos/libero_spatial_3/policy_view/eval_episode_0.mp4",
            )
            first_run = data.get("runs", [{}])[0]
            self.assertEqual(first_run.get("ood", {}).get("params"), {"noise_sigma": 0.2})
            self.assertEqual(first_run.get("ood", {}).get("algorithm"), "causal_aug.OODPerturbation")

    def test_ignores_missing_video_files_without_losing_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "outputs" / "eval" / "full" / "model_a_level_0_1ep_seed1000"
            run_dir.mkdir(parents=True)
            payload = {
                "per_task": [{
                    "task_group": "libero_spatial",
                    "task_id": 0,
                    "metrics": {"successes": [True], "video_paths": [str(run_dir / "missing.mp4")]},
                }]
            }
            (run_dir / "eval_info.json").write_text(json.dumps(payload))
            output = root / "results-data.json"

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(root), "--output", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text())
            self.assertEqual(data["models"][0]["episodes"], 1)
            self.assertEqual(data["models"][0]["videos"], 0)
            self.assertEqual(data["episodes"], [])


if __name__ == "__main__":
    unittest.main()
