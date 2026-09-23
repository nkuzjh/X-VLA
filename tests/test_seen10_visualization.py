import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from csgo_seen10.dataset import SEEN_MAPS
from infer_seen10 import _expected_metadata, _validate_checkpoint_action_contract, _write_prediction
from csgo_seen10.visualization import (
    map_pixel_from_normalized,
    render_map_visualizations,
    select_visualization_identities,
)
from train_seen10 import evaluate_validation


class VisualizationTest(unittest.TestCase):
    def test_checkpoint_without_action_metadata_is_inferred_as_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract = _validate_checkpoint_action_contract(Path(temporary), {})
        self.assertEqual(contract["mode"], "legacy_reset_dummy")
        self.assertEqual(contract["source"], "legacy-inferred")

    def test_standalone_official_action_contract_is_strictly_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            contract = {
                "mode": "official_auto",
                "external_action_dim": 5,
                "model_action_dim": 20,
                "num_actions": 1,
                "use_proprio": False,
                "padding": "pad_before_noise",
                "noise_width": 20,
                "prediction_width": 20,
                "loss": "valid_action_mse",
                "loss_dimensions": 5,
                "loss_scale": 100.0,
                "dummy_channels_in_loss": False,
                "inference_reset": False,
                "target_pad_before_noise": True,
                "final_output_dim": 5,
                "objective": "x0_clean_action_denoising_regression",
                "inference_steps": 10,
            }
            (checkpoint / "action_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            progress = {
                "normalization": {"epsilon": None, "clamp": False, "qnorm": False},
                "state": {"use_proprio": False, "dim": 20, "values": "all zeros"},
            }
            observed = _validate_checkpoint_action_contract(checkpoint, progress)
            self.assertEqual(observed["mode"], "official_auto")

            contract["noise_width"] = 5
            (checkpoint / "action_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noise_width"):
                _validate_checkpoint_action_contract(checkpoint, progress)

            contract["noise_width"] = 20
            del contract["loss_dimensions"]
            (checkpoint / "action_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _validate_checkpoint_action_contract(checkpoint, progress)

    def test_external_prediction_contract_is_exactly_five_dimensions(self):
        stream = io.StringIO()
        metadata = {"sample_id": "sample", "map_name": SEEN_MAPS[0]}
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            _write_prediction(stream, metadata, [0.0] * 4)
        _write_prediction(stream, metadata, [0.0] * 5)
        row = json.loads(stream.getvalue())
        self.assertEqual(
            {key for key in row if key.startswith("pred_")},
            {"pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw"},
        )

    def test_inference_metadata_declares_native_adaptation_and_state(self):
        class Dataset:
            limit_per_map = None

            def __len__(self):
                return 20

        metadata = _expected_metadata(
            args=SimpleNamespace(seed=0, smoke=False),
            checkpoint=Path("/tmp/seen10-checkpoint"),
            progress={
                "checkpoint_id": "checkpoint-id",
                "global_step": 10,
                "seed": 0,
                "smoke": False,
                "data_root": "/data",
            },
            data_root=Path("/data"),
            dataset=Dataset(),
            batch_size=4,
            steps=10,
            world_size=1,
            inference_seed=42,
            action_contract={
                "mode": "official_auto",
                "real_action_dim": 5,
                "max_action_dim": 20,
                "num_actions": 1,
                "use_proprio": False,
            },
        )
        self.assertEqual(metadata["action_adaptation"]["native_action_dim"], 20)
        self.assertEqual(metadata["action_adaptation"]["trimmed_output_dim"], 5)
        self.assertEqual(metadata["state"]["values"], "all zeros")
        self.assertIsNone(metadata["normalization"]["epsilon"])
        self.assertFalse(metadata["normalization"]["clamp"])
        self.assertFalse(metadata["normalization"]["qnorm"])
        self.assertEqual(metadata["inference"]["steps"], 10)
        self.assertTrue(metadata["inference"]["native_width_preserved_through_all_steps"])
        self.assertFalse(metadata["inference"]["model_only_channels_reset_each_step"])
        self.assertEqual(metadata["action_adaptation"]["training_noise_width"], 20)
        self.assertEqual(metadata["action_adaptation"]["supervised_loss_width"], 5)

    def test_legacy_inference_metadata_discloses_dummy_channel_reset(self):
        class Dataset:
            limit_per_map = None

            def __len__(self):
                return 20

        metadata = _expected_metadata(
            args=SimpleNamespace(seed=0, smoke=False),
            checkpoint=Path("/tmp/legacy-seen10-checkpoint"),
            progress={"seed": 0, "smoke": False, "data_root": "/data"},
            data_root=Path("/data"),
            dataset=Dataset(),
            batch_size=4,
            steps=10,
            world_size=1,
            inference_seed=42,
            action_contract={
                "mode": "legacy_reset_dummy",
                "real_action_dim": 5,
                "max_action_dim": 20,
                "num_actions": 1,
                "use_proprio": False,
            },
        )
        self.assertFalse(metadata["inference"]["native_width_preserved_through_all_steps"])
        self.assertTrue(metadata["inference"]["model_only_channels_reset_each_step"])
        self.assertEqual(metadata["action_adaptation"]["training_noise_width"], 5)

    def test_validation_reuses_generated_prediction_for_selected_row(self):
        class Model:
            def eval(self):
                return self

            def generate_actions(self, input_ids, **kwargs):
                return torch.zeros((input_ids.shape[0], 1, 5), dtype=torch.float32)

        class Accelerator:
            device = torch.device("cpu")
            process_index = 0
            num_processes = 1
            is_main_process = True

            def reduce(self, value, reduction):
                return value

        batch = {
            "inputs": {
                "input_ids": torch.ones((2, 3), dtype=torch.long),
                "image_input": torch.zeros((2, 2, 3, 4, 4)),
                "image_mask": torch.ones((2, 2), dtype=torch.bool),
                "domain_id": torch.zeros(2, dtype=torch.long),
                "proprio": torch.zeros((2, 20)),
                "action": torch.tensor([
                    [[0.1, 0.2, 0.3, 0.4, 0.5]],
                    [[0.6, 0.7, 0.8, 0.9, 1.0]],
                ]),
            },
            "metadata": [
                {"map_name": SEEN_MAPS[0], "sample_id": "first"},
                {"map_name": SEEN_MAPS[0], "sample_id": "second"},
            ],
        }
        with (
            patch("train_seen10._capture_rng_state", return_value={}),
            patch("train_seen10._restore_rng_state"),
            patch("train_seen10.set_seed"),
        ):
            _, _, count, rows = evaluate_validation(
                Model(),
                [batch],
                Accelerator(),
                seed=7,
                steps=10,
                visualization_identities={(SEEN_MAPS[0], "second")},
            )
        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "second")
        for actual, expected in zip(rows[0]["gt_normalized"], [0.6, 0.7, 0.8, 0.9, 1.0]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(rows[0]["pred_normalized"], [0.0] * 5)

    def test_selection_is_reproducible_and_per_map(self):
        records = [
            {"map_name": map_name, "sample_id": f"sample_{index}"}
            for map_name in SEEN_MAPS
            for index in range(20)
        ]
        first = select_visualization_identities(records, 10, 1234)
        second = select_visualization_identities(records, 10, 1234)
        other = select_visualization_identities(records, 10, 1235)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 100)
        for map_name in SEEN_MAPS:
            self.assertEqual(sum(identity[0] == map_name for identity in first), 10)

    def test_out_of_range_prediction_is_kept_visible(self):
        self.assertEqual(map_pixel_from_normalized(-0.5, 1.5, 1024, 1024, 20), (20, 1003))

    def test_render_places_radar_left_and_ten_fpvs_right(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            radar_path = root / "radar.png"
            Image.new("RGB", (1024, 1024), (180, 10, 10)).save(radar_path)
            records = []
            rows = []
            for index in range(10):
                fpv_path = root / f"fpv_{index}.png"
                Image.new("RGB", (448, 448), (10, 10, 180)).save(fpv_path)
                records.append({
                    "map_name": SEEN_MAPS[0],
                    "sample_id": f"sample_{index}",
                    "image_path": str(fpv_path),
                    "radar_path": str(radar_path),
                })
                rows.append({
                    "map_name": SEEN_MAPS[0],
                    "sample_id": f"sample_{index}",
                    "gt_normalized": [0.2 + index * 0.05, 0.2 + index * 0.05, 0.5, 0.25, 0.75],
                    "pred_normalized": [0.22 + index * 0.05, 0.18 + index * 0.05, 0.4, 0.2, 0.7],
                })

            class Dataset:
                pass

            dataset = Dataset()
            dataset.records = records
            dataset.counts = {name: (10 if name == SEEN_MAPS[0] else 0) for name in SEEN_MAPS}
            dataset.z_ranges = {name: (0.0, 100.0) for name in SEEN_MAPS}
            output = root / "output"
            paths = render_map_visualizations(
                dataset,
                rows,
                output,
                samples_per_map=10,
                seed=99,
                radar_size=1000,
                fpv_width=120,
            )
            self.assertEqual(paths, [output / f"{SEEN_MAPS[0]}.png"])
            with Image.open(paths[0]) as rendered:
                self.assertEqual(rendered.size, (1120, 1000))
                self.assertEqual(rendered.getpixel((500, 990)), (180, 10, 10))
                self.assertEqual(rendered.getpixel((1110, 990)), (10, 10, 180))
            manifest = json.loads((output / "visualization_manifest.json").read_text())
            self.assertEqual(manifest["layout"], "radar left, FPV column right; GT solid, prediction hollow")
            self.assertEqual(len(manifest["maps"][SEEN_MAPS[0]]), 10)


if __name__ == "__main__":
    unittest.main()
