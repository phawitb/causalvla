import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_results_data.py"


class BuildResultsDataTest(unittest.TestCase):
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
                video_dir.mkdir(parents=True)
                paths = []
                for episode in range(2):
                    video = video_dir / f"eval_episode_{episode}.mp4"
                    video.touch()
                    paths.append(str(video))
                payload = {
                    "per_task": [{
                        "task_group": "libero_spatial",
                        "task_id": 3,
                        "metrics": {"successes": successes, "video_paths": paths},
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
            self.assertEqual([model["id"] for model in data["models"]], ["a", "b", "v2_warm", "f"])
            self.assertEqual(data["models"][0]["episodes"], 2)
            self.assertEqual(data["models"][0]["successRate"], 50.0)
            self.assertEqual(data["models"][2]["levelRates"]["2"], 100.0)
            self.assertEqual(len(data["episodes"]), 8)
            self.assertEqual(data["episodes"][0]["task"], "libero_spatial · Task 3")
            self.assertEqual(
                data["episodes"][0]["video"],
                "outputs/eval/full/model_a_level_0_2ep_seed1000/videos/libero_spatial_3/eval_episode_0.mp4",
            )

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
