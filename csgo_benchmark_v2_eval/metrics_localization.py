"""Benchmark v2 localization metric over standardized prediction JSONL."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


LOCALIZATION_METRICS = ("XY_Dist", "Z_Dist", "Pitch_Dist", "Yaw_Dist")


def load_predictions(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL rows with Benchmark v2 normalized predicted pose fields."""

    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Localization predictions not found: {prediction_path}")
    rows: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at {prediction_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {prediction_path}:{line_no} must be an object")
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def validate_localization_coverage(
    predictions: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    expected_maps: Sequence[str],
) -> dict[str, int]:
    """Require one and only one prediction for every selected manifest row."""

    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_counts = {name: 0 for name in expected_maps}
    for row in expected_rows:
        key = (str(row["map_name"]), str(row["file_frame"]))
        if key in expected:
            raise ValueError(f"Duplicate expected localization sample: {key}")
        if key[0] not in expected_counts:
            raise ValueError(f"Unexpected map in selected localization rows: {key[0]}")
        expected[key] = row
        expected_counts[key[0]] += 1

    actual: dict[tuple[str, str], Mapping[str, Any]] = {}
    actual_counts = {name: 0 for name in expected_maps}
    for index, pred in enumerate(predictions):
        map_name = pred.get("map_name", pred.get("map"))
        sample_id = pred.get("sample_id")
        if not isinstance(map_name, str) or map_name not in actual_counts:
            raise ValueError(f"Prediction row {index} has an invalid map_name: {map_name!r}")
        file_frame = _file_frame_from_sample_id(sample_id, map_name)
        key = (map_name, file_frame)
        if key in actual:
            raise ValueError(f"Duplicate localization prediction: {key}")
        if key not in expected:
            raise ValueError(f"Localization prediction is outside the selected split: {key}")
        for field in ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw"):
            try:
                value = float(pred[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Prediction {key} needs numeric {field}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Prediction {key} has non-finite {field}: {value}")
        actual[key] = pred
        actual_counts[map_name] += 1

    missing = [key for key in expected if key not in actual]
    extra = [key for key in actual if key not in expected]
    if missing or extra or actual_counts != expected_counts:
        raise ValueError(
            "Localization coverage must be exact; "
            f"expected_counts={expected_counts}, actual_counts={actual_counts}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return actual_counts


def localization_metrics(
    predictions: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    map_name: str,
    *,
    pose_space: str = "normalized",
) -> dict[str, float]:
    """Compute the four v2 physical-space localization metrics for one map.

    Prediction fields use Benchmark v2 normalized values by default: X/Y are
    divided by 1024, Z uses the row's frozen published calibration, and both
    angles are divided by 360 degrees. Split rows store angles in radians; the
    metric converts them to degrees. Pitch is linear; yaw uses modulo shortest
    arc. ``pose_space='physical'`` is an explicit interoperability option.
    """

    if pose_space not in ("normalized", "physical"):
        raise ValueError("pose_space must be 'normalized' or 'physical'")

    pred_by_frame: dict[str, Mapping[str, Any]] = {}
    for pred in predictions:
        pred_map = pred.get("map_name", pred.get("map"))
        if pred_map == map_name:
            pred_by_frame[_file_frame_from_sample_id(pred.get("sample_id"), map_name)] = pred

    rows = [row for row in expected_rows if row["map_name"] == map_name]
    if not rows:
        raise ValueError(f"No expected localization rows for map={map_name}")
    gt_values = []
    pred_values = []
    for row in rows:
        file_frame = str(row["file_frame"])
        pred = pred_by_frame[file_frame]
        gt_raw = row["pose_raw"]
        gt_values.append([
            float(gt_raw["x"]),
            float(gt_raw["y"]),
            float(gt_raw["z"]),
            math.degrees(float(gt_raw["pitch"])),
            math.degrees(float(gt_raw["yaw"])),
        ])
        pred_x = float(pred["pred_x"])
        pred_y = float(pred["pred_y"])
        pred_z = float(pred["pred_z"])
        pred_pitch = float(pred["pred_pitch"])
        pred_yaw = float(pred["pred_yaw"])
        if pose_space == "normalized":
            z_min = float(row["z_calibration"]["z_min"])
            z_max = float(row["z_calibration"]["z_max"])
            pred_x *= 1024.0
            pred_y *= 1024.0
            pred_z = pred_z * (z_max - z_min) + z_min
            pred_pitch *= 360.0
            pred_yaw *= 360.0
        pred_values.append([pred_x, pred_y, pred_z, pred_pitch, pred_yaw])

    gt = torch.tensor(gt_values, dtype=torch.float32)
    pred = torch.tensor(pred_values, dtype=torch.float32)
    absolute = torch.abs(pred - gt)
    absolute[:, 4] = yaw_shortest_distance(pred[:, 4], gt[:, 4], period=360.0)
    xy_dist = torch.sqrt(absolute[:, 0].square() + absolute[:, 1].square()).mean().item()
    return {
        "XY_Dist": float(xy_dist),
        "Z_Dist": float(absolute[:, 2].mean().item()),
        "Pitch_Dist": float(absolute[:, 3].mean().item()),
        "Yaw_Dist": float(absolute[:, 4].mean().item()),
    }


def yaw_shortest_distance(pred: Any, gt: Any, *, period: float = 360.0):
    """Shortest circular yaw distance, correct for any number of wraps.

    Tensor inputs return a tensor; scalar inputs return a float. This is the
    benchmark formula ``abs(((pred - gt + period/2) % period) - period/2)``.
    """

    if period <= 0:
        raise ValueError("period must be positive")
    if isinstance(pred, torch.Tensor) or isinstance(gt, torch.Tensor):
        pred_tensor = torch.as_tensor(pred)
        gt_tensor = torch.as_tensor(gt, dtype=pred_tensor.dtype, device=pred_tensor.device)
        return torch.abs(torch.remainder(pred_tensor - gt_tensor + period / 2.0, period) - period / 2.0)
    return abs(((float(pred) - float(gt) + period / 2.0) % period) - period / 2.0)


def _file_frame_from_sample_id(sample_id: object, map_name: str) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Localization row requires a non-empty sample_id")
    parts = sample_id.replace("\\", "/").split("/")
    if len(parts) > 1 and parts[-2] != map_name:
        raise ValueError(
            f"sample_id map does not match map_name: sample_id={sample_id!r}, map_name={map_name!r}"
        )
    frame = Path(parts[-1]).stem
    if not frame:
        raise ValueError(f"Invalid localization sample_id: {sample_id!r}")
    return frame


__all__ = [
    "LOCALIZATION_METRICS",
    "load_predictions",
    "validate_localization_coverage",
    "localization_metrics",
    "yaw_shortest_distance",
]
