"""Read the published Seen-10 splits without discovering or repartitioning images."""

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


SEEN_MAPS = (
    "cs_agency", "cs_italy", "de_ancient", "de_anubis", "de_dust2",
    "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train",
)
SPLIT_COUNTS = {"train": 5000, "validation": 500, "discrete_test": 2000}
INSTRUCTION = (
    "Localize the player in {map_name} using the first-person image and radar map. "
    "Predict the absolute normalized x, y, z, pitch, and yaw."
)
COORDINATE_DESCRIPTION = {
    "pose": "normalized absolute [x, y, z, pitch, yaw]",
    "x_y": "world coordinate / 1024",
    "z": "(world z - published map z_min) / (published map z_max - published map z_min)",
    "pitch_yaw": "radians / (2*pi), with circular period 1",
    "proprio": "all-zero vector with 20 dimensions",
}


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_split(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a published JSON array in {path}")
    return rows


class Seen10Dataset(Dataset):
    """Two RGB views and horizon-one 5DoF targets from the release metadata.

    ``records`` contains only input identity and paths, including for test data.
    Ground truth is retained separately and only for train/validation targets.
    ``limit_per_map`` is a deterministic prefix for explicitly nonformal smoke runs.
    """

    maps = SEEN_MAPS

    def __init__(self, data_root, split, include_targets=True, limit_per_map=None):
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split.removeprefix("seen_")
        if self.split not in SPLIT_COUNTS:
            raise ValueError(f"Unsupported localization split: {split}")
        if self.split == "discrete_test" and include_targets:
            raise ValueError("discrete_test is input-only; pass include_targets=False")
        if limit_per_map is not None and limit_per_map <= 0:
            raise ValueError("limit_per_map must be positive when supplied")
        self.include_targets = include_targets
        self.limit_per_map = limit_per_map
        report = _read_json(self.data_root / "minimal_dataset_report.json")
        manifest = _read_json(self.data_root / "benchmark_manifest.json")
        if report["benchmark_id"] != "csgo_benchmark_v2" or manifest["benchmark_id"] != "csgo_benchmark_v2":
            raise ValueError("Expected a CSGO Benchmark v2 release")
        if tuple(manifest["protocol"]["seen_maps"]) != SEEN_MAPS:
            raise ValueError("The manifest Seen-10 map order differs from the protocol")

        calibration = _read_json(self.data_root / manifest["calibration"]["file"])
        self.z_ranges = {}
        for map_name in SEEN_MAPS:
            declared = manifest["calibration"]["z_ranges"][map_name]
            published = calibration["z_ranges"][map_name]
            bounds = (float(published["z_min"]), float(published["z_max"]))
            if bounds != (float(declared["z_min"]), float(declared["z_max"])):
                raise ValueError(f"Calibration and manifest Z range disagree for {map_name}")
            if not all(math.isfinite(value) for value in bounds) or bounds[1] <= bounds[0]:
                raise ValueError(f"Invalid published Z range for {map_name}: {bounds}")
            self.z_ranges[map_name] = bounds

        radar_entries = {
            (entry["map"], entry["source"]): entry
            for entry in report["radars"]["entries"]
        }
        image_template = report["images"]["target_template"]
        self.records = []
        self._targets = [] if include_targets else None
        self.counts = {}
        identities = set()
        for map_name in SEEN_MAPS:
            expected_count = SPLIT_COUNTS[self.split]
            if manifest["counts"]["seen"][map_name][self.split] != expected_count:
                raise ValueError(f"Manifest count differs from Seen-10 for {map_name}/{self.split}")
            split_path = self.data_root / "splits" / "seen" / map_name / f"{self.split}.json"
            if not split_path.is_file():
                split_path = split_path.with_suffix(".jsonl")
            rows = _read_split(split_path)
            if len(rows) != expected_count:
                raise ValueError(f"Expected {expected_count} rows in {split_path}, found {len(rows)}")
            source_radar = manifest["source"]["radar_files"][map_name]
            entry = radar_entries[(map_name, source_radar)]
            radar_path = self.data_root / report["radars"]["root"] / entry["target"]
            if not radar_path.is_file():
                raise FileNotFoundError(radar_path)
            selected_rows = rows if limit_per_map is None else rows[:limit_per_map]
            self.counts[map_name] = len(selected_rows)
            for row in selected_rows:
                if row["map"] != map_name:
                    raise ValueError(f"Map mismatch in {split_path}: {row['map']}")
                file_frame = str(row["file_frame"])
                # This release has no separate sample_id: file_frame is its identity.
                sample_id = str(row.get("sample_id", file_frame))
                identity = (map_name, sample_id)
                if identity in identities:
                    raise ValueError(f"Duplicate sample identity in {split_path}: {identity}")
                identities.add(identity)
                image_path = self.data_root / image_template.format(map=map_name, file_frame=file_frame)
                self.records.append({
                    "sample_id": sample_id,
                    "map_name": map_name,
                    "file_frame": file_frame,
                    "image_path": str(image_path),
                    "radar_path": str(radar_path),
                })
                if include_targets:
                    low, high = self.z_ranges[map_name]
                    target = [
                        float(row["x"]) / 1024.0,
                        float(row["y"]) / 1024.0,
                        (float(row["z"]) - low) / (high - low),
                        float(row["angle_v"]) / math.tau,
                        float(row["angle_h"]) / math.tau,
                    ]
                    if not all(math.isfinite(value) for value in target):
                        raise ValueError(f"Nonfinite pose target: {identity}")
                    self._targets.append(target)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            fpv = image.convert("RGB")
        with Image.open(record["radar_path"]) as image:
            radar = image.convert("RGB")
        sample = {key: record[key] for key in ("sample_id", "map_name", "file_frame")}
        sample.update({
            "images": [fpv, radar],
            "language_instruction": INSTRUCTION.format(map_name=record["map_name"]),
        })
        if self.include_targets:
            sample["action"] = torch.tensor([self._targets[index]], dtype=torch.float32)
        return sample


class Seen10Collator:
    """Use the native processor with two valid views and no robot state."""

    def __init__(self, processor, include_targets=True):
        self.processor = processor
        self.processor.num_views = 2
        self.include_targets = include_targets

    def __call__(self, samples):
        inputs = self.processor(
            images=[sample["images"] for sample in samples],
            language_instruction=[sample["language_instruction"] for sample in samples],
        )
        batch_size = len(samples)
        inputs["domain_id"] = torch.zeros(batch_size, dtype=torch.long)
        inputs["proprio"] = torch.zeros(batch_size, 20, dtype=torch.float32)
        if self.include_targets:
            inputs["action"] = torch.stack([sample["action"] for sample in samples])
        return {
            "inputs": inputs,
            "metadata": [
                {key: sample[key] for key in ("sample_id", "map_name", "file_frame")}
                for sample in samples
            ],
        }
