"""Fast, isolated checks for the CSGO environment setup preflight."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts/setup_csgo_seen10.sh"
WHEEL_SELECTOR = ROOT / "scripts/select_compatible_csgo_wheels.py"
PYTHON312 = Path("/usr/bin/python3.12")


class SetupPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(SETUP, self.root / "scripts/setup_csgo_seen10.sh")
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin")
        self.env.pop("PYTHON_BIN", None)

    def fake_python(self, name, version):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n[ \"$1\" = -c ] || exit 3\nprintf '%s\\n' '{version}'\n")
        path.chmod(0o755)
        return path

    def run_setup(self, *args, **env):
        return subprocess.run(
            ["/bin/bash", str(self.root / "scripts/setup_csgo_seen10.sh"), *args],
            env=dict(self.env, **env), text=True, capture_output=True,
        )

    def hide_system_pythons(self):
        for name in ("python3.12", "python3.11", "python3.10", "python3", "python"):
            self.fake_python(name, "")

    def test_help_has_no_effects(self):
        result = self.run_setup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-only", result.stdout)
        self.assertFalse((self.root / ".venv").exists())
        self.assertFalse((self.root / ".cache").exists())

    def test_prefers_python312_then_falls_back_to_311_or_310(self):
        self.fake_python("python3.12", "3.12")
        self.fake_python("python3.11", "3.11")
        self.fake_python("python3.10", "3.10")
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected 3.12 at python3.12", result.stdout)
        self.fake_python("python3.12", "")
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected 3.11 at python3.11", result.stdout)
        self.fake_python("python3.11", "")
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected 3.10 at python3.10", result.stdout)
        self.assertFalse((self.root / ".venv").exists())

    def test_explicit_invalid_python_is_rejected_even_with_other_options(self):
        self.fake_python("python3.12", "3.12")
        bad = self.fake_python("python3.13", "")
        result = self.run_setup("--check-only", PYTHON_BIN=str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a working Python 3.10", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def test_generic_conda_base_python_is_discovered(self):
        self.hide_system_pythons()
        self.fake_python("python3", "3.10")
        self.fake_python("python", "3.10")
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected 3.10 at python3", result.stdout)
        self.fake_python("python3", "")
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selected 3.10 at python", result.stdout)

    def test_missing_python_reports_conda_fallback_or_actionable_error(self):
        self.hide_system_pythons()
        result = self.run_setup("--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Install one or Conda", result.stderr)
        conda = self.bin / "conda"
        conda.write_text("#!/bin/sh\nexit 99\n")
        conda.chmod(0o755)
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Would provision Python 3.12 with Conda", result.stdout)
        self.assertFalse((self.root / ".cache").exists())

        prefix = self.root / ".cache/csgo-python"
        prefix.mkdir(parents=True)
        marker = prefix / "keep.txt"
        marker.write_text("keep")
        result = self.run_setup("--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing Conda prefix was preserved", result.stderr)
        self.assertEqual(marker.read_text(), "keep")

    @unittest.skipUnless(PYTHON312.exists(), "system Python 3.12 unavailable")
    def test_existing_venv_reused_without_mutation(self):
        venv = self.root / ".venv"
        subprocess.run([str(PYTHON312), "-m", "venv", "--without-pip", str(venv)], check=True)
        self.hide_system_pythons()
        result = self.run_setup("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Using isolated Python 3.12", result.stdout)
        self.assertFalse((venv / "bin/pip").exists())
        self.assertFalse((self.root / ".cache").exists())

        selected = self.fake_python("python3.11", "3.11")
        mismatch = self.run_setup("--check-only", PYTHON_BIN=str(selected))
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("different Python version", mismatch.stderr)
        self.assertFalse((venv / "bin/pip").exists())

    def test_incomplete_existing_venv_is_preserved(self):
        venv = self.root / ".venv"
        venv.mkdir()
        marker = venv / "keep.txt"
        marker.write_text("keep")
        result = self.run_setup("--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("was preserved", result.stderr)
        self.assertEqual(marker.read_text(), "keep")


class WheelSelectionTests(unittest.TestCase):
    def test_skips_incompatible_python_tags_and_prefers_compatible_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel_dir = Path(directory)
            incompatible = wheel_dir / "torch-2.7.1+cu128-cp310-cp310-manylinux_2_17_x86_64.whl"
            universal = wheel_dir / "torch-2.7.1+cu128-py3-none-any.whl"
            other_version = wheel_dir / "torch-2.7.0+cu128-py3-none-any.whl"
            for wheel in (incompatible, universal, other_version):
                wheel.touch()
            result = subprocess.run(
                [shutil.which("python3"), str(WHEEL_SELECTOR), directory,
                 "torch==2.7.1+cu128", "torchvision==0.22.1+cu128"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(universal)])


if __name__ == "__main__":
    unittest.main()
