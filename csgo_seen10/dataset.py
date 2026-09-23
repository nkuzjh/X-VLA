"""Read the published Seen-10 splits without discovering or repartitioning images."""

import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
    "pitch_yaw": "radians / (2*pi)",
    "pitch_metric": "linear error in normalized coordinates",
    "yaw_metric": "circular shortest-arc error with normalized period 1",
    "proprio": "all-zero vector with 20 dimensions",
}

# Seen-10 has a five-dimensional external target, while the native XVLA
# denoiser keeps its pretrained twenty-dimensional width.  Keep these values
# in one place so data, inference, and provenance metadata cannot silently
# drift apart.
SEEN10_EXTERNAL_ACTION_DIM = 5
SEEN10_NATIVE_ACTION_DIM = 20
SEEN10_NUM_ACTIONS = 1
SEEN10_STATE_DIM = 20
SEEN10_INFERENCE_STEPS = 10

SEEN10_ACTION_ADAPTATION = {
    "mode": "official_auto",
    "external_action_dim": SEEN10_EXTERNAL_ACTION_DIM,
    "native_action_dim": SEEN10_NATIVE_ACTION_DIM,
    "num_actions": SEEN10_NUM_ACTIONS,
    "external_order": "xyzhw = x, y, z, pitch, yaw",
    "objective": "x0_clean_action_denoising_regression",
    "dataset_target_width": SEEN10_EXTERNAL_ACTION_DIM,
    "target_padding": "pad once to 20D immediately before XVLA.forward and before noise sampling",
    "training_noise_width": SEEN10_NATIVE_ACTION_DIM,
    "training_model_input_width": SEEN10_NATIVE_ACTION_DIM,
    "training_prediction_width": SEEN10_NATIVE_ACTION_DIM,
    "supervised_loss_width": SEEN10_EXTERNAL_ACTION_DIM,
    "loss_scale": 100.0,
    "model_only_channels_directly_supervised": False,
    "inference_state_width": SEEN10_NATIVE_ACTION_DIM,
    "model_only_channels_reset_each_step": False,
    "native_denoising": "20D for every denoising step; trim to 5D only after the final step",
}
SEEN10_STATE_DESCRIPTION = {
    "use_proprio": False,
    "dim": SEEN10_STATE_DIM,
    "values": "all zeros",
    "robot_state_information": "absent",
    "proprio_tensor": "zeros[20]",
    "proprio_structure": "preserved_for_pretrained_72d_encoder",
    "action_encoder_input": "action20 + zero_proprio20 + sinusoidal_time32 = 72D",
}
SEEN10_NORMALIZATION_DESCRIPTION = {
    "space": "normalized absolute xyzhw",
    "x_y": "world coordinate / 1024",
    "z": "(world z - published map z_min) / (published map z_max - published map z_min)",
    "pitch_yaw": "radians / (2*pi)",
    "pitch_metric": "linear error in normalized coordinates",
    "yaw_metric": "circular shortest-arc error with normalized period 1",
    "epsilon": None,
    "clamp": False,
    "qnorm": False,
}

# This is the exact augmentation used by X-VLA's native training reader.  It
# is applied to both views before the native processor and is intentionally
# absent for validation and test samples.
TRAIN_COLOR_JITTER = transforms.ColorJitter(0.2, 0.2, 0.2, 0.0)


def resolve_seen10_augmentation(split: str, augmentation=None) -> bool:
    """Return whether color jitter should run for a dataset split.

    ``augmentation`` may be a bool or a policy name such as ``train_only``
    and ``native_color_jitter``.  The training entry point always passes its
    explicit config value.  ``None`` retains the pre-adapter direct-call
    behavior (disabled), while validation and test are deterministic for every
    policy value.
    """
    split = split.removeprefix("seen_")
    if split != "train":
        return False
    if augmentation is None:
        return False
    if isinstance(augmentation, dict):
        return bool(augmentation.get("enabled", False))
    if isinstance(augmentation, str):
        return augmentation.lower() in {"train_only", "native_color_jitter", "color_jitter"}
    return bool(augmentation)


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


def _resolve_seen10_split_path(data_root, map_name, split):
    """Resolve one published Seen-10 split using the dataset reader's policy.

    The release currently contains JSON split files, but the reader has
    historically accepted JSONL as a fallback.  Keeping this lookup in one
    helper makes the data-contract fingerprint and the dataset consume the
    same physical file.
    """
    split = split.removeprefix("seen_")
    if split not in SPLIT_COUNTS:
        raise ValueError(f"Unsupported localization split: {split}")
    data_root = Path(data_root).expanduser().resolve()
    split_path = data_root / "splits" / "seen" / map_name / f"{split}.json"
    if not split_path.is_file():
        split_path = split_path.with_suffix(".jsonl")
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    return split_path


def _sha256_file(path, *, chunk_size=1024 * 1024):
    """Hash a small release metadata file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_file_entry(data_root, path, kind, **fields):
    """Return deterministic identity metadata for one release metadata file."""
    root = Path(data_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Release metadata path escapes data root: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    entry = {
        "kind": kind,
        "path": relative,
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    entry.update(fields)
    return entry


def seen10_data_contract(data_root):
    """Fingerprint the published Seen-10 metadata used by X-VLA.

    Only small protocol and split metadata files are hashed.  Image and radar
    payloads are intentionally excluded so this check remains cheap and does
    not turn a provenance audit into a multi-gigabyte data scan.  The returned
    ``contract_sha256`` is SHA-256 over the rest of the returned object using
    canonical JSON (sorted keys, compact separators, UTF-8).
    """
    root = Path(data_root).expanduser().resolve()
    manifest_path = root / "benchmark_manifest.json"
    report_path = root / "minimal_dataset_report.json"
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    if manifest.get("benchmark_id") != "csgo_benchmark_v2":
        raise ValueError("benchmark_manifest.json is not csgo_benchmark_v2")
    if report.get("benchmark_id") != "csgo_benchmark_v2":
        raise ValueError("minimal_dataset_report.json is not csgo_benchmark_v2")
    manifest_maps = tuple(manifest.get("protocol", {}).get("seen_maps", ()))
    if manifest_maps != SEEN_MAPS:
        raise ValueError(f"The manifest Seen-10 map order differs from the protocol: {manifest_maps!r}")

    manifest_entry = _contract_file_entry(root, manifest_path, "benchmark_manifest")
    declared_manifest_sha = report.get("manifest", {}).get("sha256")
    if declared_manifest_sha and manifest_entry["sha256"] != declared_manifest_sha:
        raise ValueError(
            "benchmark_manifest.json does not match minimal_dataset_report.json: "
            f"actual={manifest_entry['sha256']}, declared={declared_manifest_sha}"
        )
    files = [
        manifest_entry,
        _contract_file_entry(root, report_path, "minimal_dataset_report"),
    ]
    calibration_rel = manifest.get("calibration", {}).get("file")
    if not calibration_rel:
        raise ValueError("benchmark_manifest.json does not declare a calibration file")
    calibration_entry = _contract_file_entry(root, root / calibration_rel, "calibration")
    declared_calibration_sha = manifest.get("calibration", {}).get("sha256")
    if declared_calibration_sha and calibration_entry["sha256"] != declared_calibration_sha:
        raise ValueError(
            "Published calibration does not match benchmark_manifest.json: "
            f"actual={calibration_entry['sha256']}, declared={declared_calibration_sha}"
        )
    files.append(calibration_entry)

    for split in ("train", "validation", "discrete_test"):
        for map_name in SEEN_MAPS:
            split_path = _resolve_seen10_split_path(root, map_name, split)
            files.append(
                _contract_file_entry(
                    root,
                    split_path,
                    "split",
                    map=map_name,
                    split=split,
                )
            )

    contract = {
        "schema_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "seen_maps": list(SEEN_MAPS),
        "splits": ["train", "validation", "discrete_test"],
        "files": files,
    }
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    contract["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


# Descriptive alias for callers that prefer a verb-style name.
build_seen10_data_contract = seen10_data_contract


class Seen10Dataset(Dataset):
    """Two RGB views and horizon-one 5DoF targets from the release metadata.

    ``records`` contains only input identity and paths, including for test data.
    Ground truth is retained separately and only for train/validation targets.
    ``limit_per_map`` is a deterministic prefix for explicitly nonformal smoke runs.
    Training color jitter is controlled by the launcher's augmentation policy;
    validation and test are always deterministic.
    """

    maps = SEEN_MAPS

    def __init__(self, data_root, split, include_targets=True, limit_per_map=None, augmentation=None):
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
        self.training = self.split == "train"
        self.augment_images = resolve_seen10_augmentation(self.split, augmentation)
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
            split_path = _resolve_seen10_split_path(self.data_root, map_name, self.split)
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
        if self.augment_images:
            # Color-only augmentation is applied independently to FPV and
            # radar, matching the native X-VLA reader.  Geometry, pose labels,
            # and sample identities remain untouched.  Validation and test
            # splits never enter this branch and are therefore deterministic.
            fpv = TRAIN_COLOR_JITTER(fpv)
            radar = TRAIN_COLOR_JITTER(radar)
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
