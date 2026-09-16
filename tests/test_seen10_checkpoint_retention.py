import os
import json
import tempfile
import unittest
from pathlib import Path

from train_seen10 import _prune_periodic_checkpoints, _set_best_checkpoint, _set_last_checkpoint


class _SingleProcessAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        pass


class CheckpointRetentionTest(unittest.TestCase):
    def test_last_and_best_pointers_match_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            checkpoint = output_root / "checkpoints" / "step_00010000"
            checkpoint.mkdir(parents=True)
            progress = {
                "checkpoint_id": "checkpoint-id",
                "global_step": 10000,
                "seed": 0,
                "validation_normalized_mse": 0.25,
            }
            accelerator = _SingleProcessAccelerator()

            _set_last_checkpoint(accelerator, output_root, checkpoint, progress)
            _set_best_checkpoint(accelerator, output_root, checkpoint, progress)

            for name in ("last", "best"):
                pointer = output_root / "checkpoints" / name
                self.assertTrue(pointer.is_symlink())
                self.assertEqual(os.readlink(pointer), checkpoint.name)
                metadata = json.loads((pointer.parent / f"{name}.json").read_text())
                self.assertEqual(metadata["checkpoint"], checkpoint.name)
                self.assertEqual(metadata["checkpoint_id"], "checkpoint-id")
                self.assertEqual(metadata["global_step"], 10000)

    def test_keeps_recent_five_and_older_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            checkpoints = output_root / "checkpoints"
            checkpoints.mkdir()
            for step in range(1, 8):
                (checkpoints / f"step_{step:08d}").mkdir()
            (checkpoints / ".step_00000008.tmp").mkdir()
            (checkpoints / "notes").mkdir()
            os.symlink("step_00000001", checkpoints / "best")
            os.symlink("step_00000007", checkpoints / "last")

            _prune_periodic_checkpoints(_SingleProcessAccelerator(), output_root, 5)

            kept = {
                path.name
                for path in checkpoints.iterdir()
                if path.is_dir() and path.name.startswith("step_")
            }
            self.assertEqual(
                kept,
                {
                    "step_00000001",
                    "step_00000003",
                    "step_00000004",
                    "step_00000005",
                    "step_00000006",
                    "step_00000007",
                },
            )
            self.assertTrue((checkpoints / ".step_00000008.tmp").is_dir())
            self.assertTrue((checkpoints / "notes").is_dir())


if __name__ == "__main__":
    unittest.main()
