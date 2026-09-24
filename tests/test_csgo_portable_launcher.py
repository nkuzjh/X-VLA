"""Exercise script routing in another user's directory without model execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class PortableLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "another user" / "task" / "X-VLA"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "configs").mkdir()
        shutil.copy(PROJECT / "scripts/run_csgo_seen10.sh", self.project / "scripts")
        for path in (PROJECT / "configs").glob("csgo_seen10*.json"):
            shutil.copy(path, self.project / "configs")
        self.python = self.project / ".venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] == '-':\n"
            "    sys.argv = sys.argv[1:]\n"
            "    exec(compile(sys.stdin.read(), '<config reader>', 'exec'))\n"
            "else:\n"
            "    Path(os.environ['TEST_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        self.python.chmod(0o755)
        self.capture = self.project / "capture.json"
        self.env = dict(os.environ, TEST_CAPTURE=str(self.capture))
        for key in ("DATA_ROOT", "CSGO_DATA_ROOT", "CSGO_PYTHON", "UNILIP_PYTHON", "SHARED_EVAL_DIR"):
            self.env.pop(key, None)
        bundled = self.project / "csgo_benchmark_v2_eval"
        bundled.mkdir()
        (bundled / "run_eval.py").touch()

    def run_script(self, mode, *args):
        result = subprocess.run(
            ["bash", str(self.project / "scripts/run_csgo_seen10.sh"), mode, *args],
            cwd=self.temp.name, env=self.env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.capture.read_text())

    def test_train_infer_all_configs_use_sibling_data_and_project_environment(self):
        expected_data = str(self.project.parent / "UniLIP/data/csgo_benchmark_v2")
        for path in (self.project / "configs").glob("*.json"):
            for mode in ("train", "infer"):
                with self.subTest(config=path.name, mode=mode):
                    args = self.run_script(mode, "--config", str(path))
                    self.assertEqual(args[0], f"{mode}_seen10.py")
                    self.assertEqual(args[args.index("--data-root") + 1], expected_data)
                    self.assertTrue(args[args.index("--output-root") + 1].startswith(str(self.project)))

    def test_data_override_precedence(self):
        self.env["CSGO_DATA_ROOT"] = "/csgo_data"
        args = self.run_script("train")
        self.assertEqual(args[args.index("--data-root") + 1], "/csgo_data")
        self.env["DATA_ROOT"] = "/env_data"
        args = self.run_script("train")
        self.assertEqual(args[args.index("--data-root") + 1], "/env_data")
        args = self.run_script("train", "--data-root", "/cli_data")
        self.assertEqual(args[args.index("--data-root") + 1], "/cli_data")

    def test_eval_falls_back_to_bundled_evaluator_and_prefers_shared_if_present(self):
        args = self.run_script("eval")
        self.assertEqual(args[0], str(self.project / "csgo_benchmark_v2_eval/run_eval.py"))
        shared = self.project.parent / "csgo_benchmark_v2_eval_general"
        shared.mkdir()
        (shared / "run_eval.py").touch()
        args = self.run_script("eval")
        self.assertEqual(Path(args[0]).resolve(), shared / "run_eval.py")
        self.env["SHARED_EVAL_DIR"] = str(self.project / "csgo_benchmark_v2_eval")
        args = self.run_script("eval")
        self.assertEqual(args[0], str(self.project / "csgo_benchmark_v2_eval/run_eval.py"))

    def test_download_missing_environment_exits_before_network_or_output_creation(self):
        shutil.copy(PROJECT / "scripts/download_csgo_checkpoint.sh", self.project / "scripts")
        self.python.unlink()
        result = subprocess.run(
            ["bash", str(self.project / "scripts/download_csgo_checkpoint.sh")],
            cwd=self.temp.name, env=self.env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run scripts/setup_csgo_seen10.sh first", result.stderr)
        self.assertFalse((self.project / "pretrained").exists())


if __name__ == "__main__":
    unittest.main()
