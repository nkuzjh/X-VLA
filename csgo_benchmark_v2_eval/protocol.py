"""Manifest-driven CSGO Benchmark v2 Seen-10 data reader.

This module intentionally depends only on the Python standard library. It can
be used by model training/inference code without importing evaluator metrics or
any UniLIP/ControlAR model implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)

_SPLIT_FILES = {
    "seen_train": ("train", "train.json"),
    "train": ("train", "train.json"),
    "seen_validation": ("validation", "validation.json"),
    "validation": ("validation", "validation.json"),
    "seen_discrete_test": ("discrete_test", "discrete_test.json"),
    "discrete_test": ("discrete_test", "discrete_test.json"),
    "seen_continuous": ("continuous", "continuous_clips.json"),
    "continuous": ("continuous", "continuous_clips.json"),
}

_FILE_FRAME_RE = re.compile(r"^file_num(?P<file_num>\d+)_frame_(?P<frame_id>\d+)$")
_TAU = 2.0 * math.pi


class BenchmarkDataError(ValueError):
    """Raised when a Benchmark v2 runtime bundle does not match its contract."""


class BenchmarkData:
    """Read Seen-10 rows from the v2 manifest, splits, calibration and flat bundle.

    ``rows(split)`` returns rows in manifest map order and split-file order.
    ``max_samples`` is a reader/smoke convenience, counted globally after map
    ordering; official evaluator calls omit it and always read every row.
    """

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()
        self.manifest_path = self.data_root / "benchmark_manifest.json"
        self.report_path = self.data_root / "minimal_dataset_report.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Benchmark manifest not found: {self.manifest_path}")
        if not self.report_path.is_file():
            raise FileNotFoundError(f"Minimal dataset report not found: {self.report_path}")

        self.manifest = _read_json(self.manifest_path)
        self.report = _read_json(self.report_path)
        if self.manifest.get("benchmark_id") != "csgo_benchmark_v2":
            raise BenchmarkDataError("benchmark_manifest.json is not csgo_benchmark_v2")
        if self.report.get("benchmark_id") != "csgo_benchmark_v2":
            raise BenchmarkDataError("minimal_dataset_report.json is not csgo_benchmark_v2")
        if self.report.get("status") != "verified":
            raise BenchmarkDataError(
                f"minimal dataset report status must be verified, got {self.report.get('status')!r}"
            )

        protocol = self.manifest.get("protocol", {})
        manifest_maps = tuple(protocol.get("seen_maps", ()))
        if manifest_maps != SEEN_MAPS:
            raise BenchmarkDataError(
                f"Seen-10 map order differs from the published contract: {manifest_maps!r}"
            )
        self.maps = SEEN_MAPS

        self._image_target_template = self.report.get("images", {}).get("target_template")
        self._image_root = self.report.get("images", {}).get("root", "images")
        if not self._image_target_template or "{map}" not in self._image_target_template or "{file_frame}" not in self._image_target_template:
            raise BenchmarkDataError("minimal report has no usable images.target_template")
        self._radar_root = self.report.get("radars", {}).get("root", "radars")
        entries = self.report.get("radars", {}).get("entries", [])
        self._radar_targets: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            map_name, target = entry.get("map"), entry.get("target")
            if map_name in self._radar_targets:
                raise BenchmarkDataError(f"Duplicate radar mapping for map={map_name!r}")
            if map_name and target:
                self._radar_targets[str(map_name)] = str(target)
        missing_radars = [name for name in self.maps if name not in self._radar_targets]
        if missing_radars:
            raise BenchmarkDataError(f"Minimal report is missing Seen-10 radar mappings: {missing_radars}")

        self.z_ranges = self._load_z_ranges()
        self._counts = self.manifest.get("counts", {}).get("seen", {})

    def _load_z_ranges(self) -> dict[str, dict[str, float]]:
        calibration_meta = self.manifest.get("calibration", {})
        calibration_rel = calibration_meta.get("file", "calibration/z_calibration.json")
        calibration_path = (self.data_root / calibration_rel).resolve()
        if self.data_root not in calibration_path.parents:
            raise BenchmarkDataError(f"Calibration path escapes data root: {calibration_rel}")
        calibration = _read_json(calibration_path)
        manifest_fp = calibration_meta.get("fingerprint")
        if manifest_fp and calibration.get("calibration_sha256") != manifest_fp:
            raise BenchmarkDataError("Manifest and z_calibration.json fingerprints differ")

        manifest_ranges = calibration_meta.get("z_ranges", {})
        calibration_ranges = calibration.get("z_ranges", {})
        ranges: dict[str, dict[str, float]] = {}
        for map_name in self.maps:
            values = manifest_ranges.get(map_name) or calibration_ranges.get(map_name)
            other = calibration_ranges.get(map_name)
            if not isinstance(values, Mapping):
                raise BenchmarkDataError(f"Published z calibration is missing map={map_name}")
            z_min, z_max = float(values["z_min"]), float(values["z_max"])
            if not z_max > z_min:
                raise BenchmarkDataError(f"Invalid z range for {map_name}: {z_min}, {z_max}")
            if isinstance(other, Mapping) and (
                z_min != float(other["z_min"]) or z_max != float(other["z_max"])
            ):
                raise BenchmarkDataError(f"Manifest and calibration z range differ for {map_name}")
            ranges[map_name] = {"z_min": z_min, "z_max": z_max}
        return ranges

    def rows(
        self,
        split: str,
        *,
        maps: Sequence[str] | None = None,
        max_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized samples. ``max_samples`` limits the global prefix."""

        split_spec = _SPLIT_FILES.get(split)
        if split_spec is None:
            raise ValueError(f"Unsupported Seen-10 split {split!r}; choose from {sorted(_SPLIT_FILES)}")
        count_key, filename = split_spec
        selected_maps = self._selected_maps(maps)
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        if max_samples == 0:
            return []

        if count_key == "continuous":
            all_rows = [
                row
                for clip in self.clips("seen_continuous", maps=selected_maps)
                for row in clip["rows"]
            ]
            return all_rows if max_samples is None else all_rows[:max_samples]

        result: list[dict[str, Any]] = []
        for map_name in selected_maps:
            path = self.data_root / "splits" / "seen" / map_name / filename
            payload = _read_json(path)
            if not isinstance(payload, list):
                raise BenchmarkDataError(f"Expected JSON array in {path}")
            map_rows = [self._normalise_row(map_name, item) for item in payload]

            expected = self._expected_count(map_name, count_key)
            if len(map_rows) != expected:
                raise BenchmarkDataError(
                    f"Manifest count mismatch for {map_name}/{count_key}: expected {expected}, got {len(map_rows)}"
                )
            result.extend(map_rows)
            if max_samples is not None and len(result) >= max_samples:
                return result[:max_samples]
        return result

    def clips(
        self,
        split: str = "seen_continuous",
        *,
        maps: Sequence[str] | None = None,
        max_clips: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return ordered whole continuous clips, preserving manifest frame order."""

        if split not in ("seen_continuous", "continuous"):
            raise ValueError("clips() only supports seen_continuous")
        if max_clips is not None and max_clips < 0:
            raise ValueError("max_clips must be non-negative")
        if max_clips == 0:
            return []
        selected_maps = self._selected_maps(maps)
        clip_result: list[dict[str, Any]] = []
        expected_clips = self.manifest.get("counts", {}).get("seen", {})
        frames_per_clip = int(self.manifest.get("continuous_protocol", {}).get("frames_per_clip", 64))
        for map_name in selected_maps:
            path = self.data_root / "splits" / "seen" / map_name / "continuous_clips.json"
            payload = _read_json(path)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("clips"), list):
                raise BenchmarkDataError(f"Invalid continuous clip file: {path}")
            raw_clips = payload["clips"]
            expected_map = expected_clips.get(map_name, {})
            expected_clip_count = int(expected_map.get("continuous_clips", -1))
            expected_frame_count = int(expected_map.get("continuous_frames", -1))
            if len(raw_clips) != expected_clip_count:
                raise BenchmarkDataError(
                    f"Manifest clip count mismatch for {map_name}: expected {expected_clip_count}, got {len(raw_clips)}"
                )
            map_frame_count = 0
            for clip in raw_clips:
                if not isinstance(clip, Mapping):
                    raise BenchmarkDataError(f"Invalid clip entry for {map_name}: {clip!r}")
                clip_id = str(clip.get("clip_id", ""))
                frames = clip.get("frames")
                if not clip_id or not isinstance(frames, list):
                    raise BenchmarkDataError(f"Clip needs clip_id and frames for {map_name}: {clip!r}")
                if len(frames) != frames_per_clip:
                    raise BenchmarkDataError(
                        f"Clip length mismatch for {clip_id}: expected {frames_per_clip}, got {len(frames)}"
                    )
                rows = [
                    self._normalise_row(map_name, frame, clip_id=clip_id, frame_index=index)
                    for index, frame in enumerate(frames)
                ]
                self._validate_frame_order(clip_id, rows)
                map_frame_count += len(rows)
                clip_result.append({"clip_id": clip_id, "map_name": map_name, "rows": rows})
                if max_clips is not None and len(clip_result) >= max_clips:
                    return clip_result
            if map_frame_count != expected_frame_count:
                raise BenchmarkDataError(
                    f"Manifest continuous frame count mismatch for {map_name}: expected {expected_frame_count}, got {map_frame_count}"
                )
        return clip_result

    def _continuous_rows(self, map_name: str, clips: Sequence[object]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for clip in clips:
            if not isinstance(clip, Mapping) or not isinstance(clip.get("frames"), list):
                raise BenchmarkDataError(f"Invalid continuous clip entry in {map_name}")
            clip_id = str(clip.get("clip_id", ""))
            if not clip_id:
                raise BenchmarkDataError(f"Continuous clip in {map_name} has no clip_id")
            clip_rows = [
                self._normalise_row(map_name, frame, clip_id=clip_id, frame_index=index)
                for index, frame in enumerate(clip["frames"])
            ]
            self._validate_frame_order(clip_id, clip_rows)
            rows.extend(clip_rows)
        return rows

    def _normalise_row(
        self,
        expected_map: str,
        row: object,
        *,
        clip_id: str | None = None,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise BenchmarkDataError(f"Split row must be an object, got {type(row).__name__}")
        map_name = str(row.get("map", expected_map))
        if map_name != expected_map:
            raise BenchmarkDataError(f"Row map mismatch: expected {expected_map}, got {map_name}")
        file_frame = str(row.get("file_frame", ""))
        if not file_frame or Path(file_frame).name != file_frame:
            raise BenchmarkDataError(f"Invalid file_frame {file_frame!r}")
        parsed = _FILE_FRAME_RE.fullmatch(Path(file_frame).stem)
        if parsed is None:
            raise BenchmarkDataError(f"Invalid file_frame identity {file_frame!r}")
        try:
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            angle_h, angle_v = float(row["angle_h"]), float(row["angle_v"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkDataError(f"Invalid pose fields for {map_name}/{file_frame}") from exc
        if not all(math.isfinite(value) for value in (x, y, z, angle_h, angle_v)):
            raise BenchmarkDataError(f"Non-finite pose for {map_name}/{file_frame}")

        z_min = self.z_ranges[map_name]["z_min"]
        z_max = self.z_ranges[map_name]["z_max"]
        pose = [x / 1024.0, y / 1024.0, (z - z_min) / (z_max - z_min), angle_v / _TAU, angle_h / _TAU]
        image_rel = self._image_target_template.format(map=map_name, file_frame=file_frame)
        image_path = (self.data_root / image_rel).resolve()
        radar_path = (self.data_root / self._radar_root / self._radar_targets[map_name]).resolve()
        self._check_bundle_path(image_path, f"image for {map_name}/{file_frame}")
        self._check_bundle_path(radar_path, f"radar for {map_name}")

        return {
            "sample_id": f"{map_name}/{Path(file_frame).stem}",
            "map_name": map_name,
            "file_frame": Path(file_frame).stem,
            "image_path": str(image_path),
            "radar_path": str(radar_path),
            "pose": pose,
            "pose_raw": {
                "x": x,
                "y": y,
                "z": z,
                "pitch": angle_v,
                "yaw": angle_h,
                "angle_v_rad": angle_v,
                "angle_h_rad": angle_h,
            },
            "z_calibration": {"z_min": z_min, "z_max": z_max},
            "clip_id": clip_id,
            "frame_index": frame_index,
            "file_num": int(parsed.group("file_num")),
            "frame_id": int(parsed.group("frame_id")),
        }

    def _validate_frame_order(self, clip_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            raise BenchmarkDataError(f"Empty continuous clip: {clip_id}")
        first_file_num = rows[0]["file_num"]
        previous_frame = rows[0]["frame_id"]
        for row in rows[1:]:
            if row["file_num"] != first_file_num or row["frame_id"] - previous_frame > 2:
                raise BenchmarkDataError(
                    f"Clip {clip_id} violates fixed file/frame continuity (max gap 2)"
                )
            if row["frame_id"] <= previous_frame:
                raise BenchmarkDataError(f"Clip {clip_id} frame order is not strictly increasing")
            previous_frame = row["frame_id"]

    def _expected_count(self, map_name: str, count_key: str) -> int:
        counts = self._counts.get(map_name, {})
        if count_key == "continuous":
            return int(counts.get("continuous_frames", -1))
        if count_key not in counts:
            raise BenchmarkDataError(f"Manifest count is missing {map_name}/{count_key}")
        return int(counts[count_key])

    def _selected_maps(self, maps: Sequence[str] | None) -> tuple[str, ...]:
        if maps is None:
            return self.maps
        if isinstance(maps, (str, bytes)):
            raise ValueError("maps must be a sequence of names, not a string")
        selected = tuple(maps)
        unknown = [map_name for map_name in selected if map_name not in self.maps]
        if unknown:
            raise ValueError(f"Maps are outside Seen-10: {unknown}")
        if len(set(selected)) != len(selected):
            raise ValueError("maps contains duplicates")
        return tuple(map_name for map_name in self.maps if map_name in selected)

    def _check_bundle_path(self, path: Path, description: str) -> None:
        if self.data_root not in path.parents:
            raise BenchmarkDataError(f"{description} escapes data root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required Benchmark v2 file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkDataError(f"Cannot read JSON file {path}: {exc}") from exc


__all__ = ["BenchmarkData", "BenchmarkDataError", "SEEN_MAPS"]
