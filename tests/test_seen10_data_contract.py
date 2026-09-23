import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from csgo_seen10.dataset import SEEN_MAPS, seen10_data_contract


class Seen10DataContractTest(unittest.TestCase):
    def _write_release(self, root):
        manifest = {
            "benchmark_id": "csgo_benchmark_v2",
            "protocol": {"seen_maps": list(SEEN_MAPS)},
            "calibration": {"file": "calibration/z_calibration.json"},
        }
        report = {"benchmark_id": "csgo_benchmark_v2"}
        (root / "benchmark_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "minimal_dataset_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (root / "calibration").mkdir()
        (root / "calibration/z_calibration.json").write_text(
            json.dumps({"z_ranges": {}}), encoding="utf-8"
        )
        for split in ("train", "validation", "discrete_test"):
            for index, map_name in enumerate(SEEN_MAPS):
                split_root = root / "splits" / "seen" / map_name
                split_root.mkdir(parents=True, exist_ok=True)
                suffix = ".jsonl" if split == "discrete_test" and index == 0 else ".json"
                split_root.joinpath(split + suffix).write_text(
                    json.dumps([{"map": map_name, "file_frame": "frame_0"}])
                    + ("\n" if suffix == ".jsonl" else ""),
                    encoding="utf-8",
                )

    def test_contract_records_only_protocol_and_split_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_release(root)
            contract = seen10_data_contract(root)

            self.assertEqual(contract["benchmark_id"], "csgo_benchmark_v2")
            self.assertEqual(contract["seen_maps"], list(SEEN_MAPS))
            self.assertEqual(len(contract["files"]), 33)  # 3 roots + 30 split files
            paths = {entry["path"] for entry in contract["files"]}
            self.assertIn("calibration/z_calibration.json", paths)
            self.assertIn("splits/seen/cs_agency/discrete_test.jsonl", paths)
            self.assertTrue(all("images/" not in path and "radars/" not in path for path in paths))
            for entry in contract["files"]:
                self.assertEqual(entry["size"], (root / entry["path"]).stat().st_size)
                self.assertEqual(
                    entry["sha256"],
                    hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest(),
                )

    def test_contract_hash_is_canonical_and_changes_with_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_release(root)
            first = seen10_data_contract(root)
            second = seen10_data_contract(root)
            self.assertEqual(first, second)

            payload = dict(first)
            expected = dict(payload)
            expected.pop("contract_sha256")
            canonical = json.dumps(
                expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            self.assertEqual(first["contract_sha256"], hashlib.sha256(canonical).hexdigest())

            manifest = root / "benchmark_manifest.json"
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed = seen10_data_contract(root)
            self.assertNotEqual(first["contract_sha256"], changed["contract_sha256"])

    def test_contract_rejects_wrong_benchmark_or_seen_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_release(root)
            manifest_path = root / "benchmark_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["benchmark_id"] = "other_benchmark"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "csgo_benchmark_v2"):
                seen10_data_contract(root)

            manifest["benchmark_id"] = "csgo_benchmark_v2"
            manifest["protocol"]["seen_maps"] = list(SEEN_MAPS[:-1])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Seen-10 map order"):
                seen10_data_contract(root)

    def test_declared_manifest_and_calibration_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_release(root)
            manifest_path = root / "benchmark_manifest.json"
            report_path = root / "minimal_dataset_report.json"
            calibration_path = root / "calibration/z_calibration.json"

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["manifest"] = {"sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            seen10_data_contract(root)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["calibration"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report["manifest"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Published calibration"):
                seen10_data_contract(root)

            manifest["calibration"]["sha256"] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report["manifest"]["sha256"] = "f" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "minimal_dataset_report"):
                seen10_data_contract(root)


if __name__ == "__main__":
    unittest.main()
