#!/usr/bin/env python3
"""Independent, manifest-driven CSGO Benchmark v2 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from protocol import BenchmarkData, SEEN_MAPS  # noqa: E402


TASK_SPLITS = {
    "localization": "seen_discrete_test",
    "discrete": "seen_discrete_test",
    "continuous": "seen_continuous",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_config(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else EVAL_DIR / "benchmark_v2.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("benchmark_id") != "csgo_benchmark_v2":
        raise ValueError(f"Invalid evaluator config: {config_path}")
    maps = tuple(config.get("maps", ()))
    if maps != SEEN_MAPS:
        raise ValueError(f"Evaluator map order must be the fixed Seen-10 order: {maps!r}")
    continuous = config.get("continuous", {})
    required_fixed = {
        "clip_length": 16,
        "clip_stride": 16,
        "fvd_size": 224,
        "frame_diff_threshold": 2,
        "min_track_len": 4,
        "expected_clips_per_map": 20,
        "expected_frames_per_clip": 64,
    }
    for key, expected in required_fixed.items():
        if continuous.get(key) != expected:
            raise ValueError(f"{config_path}: continuous.{key} must stay fixed at {expected}")
    return config


def _prediction_jsonl_path(pred_root: str | Path) -> Path:
    path = Path(pred_root).expanduser().resolve()
    if path.is_file():
        return path
    candidate = path / "predictions.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Expected localization JSONL at {candidate}")


def _generation_root(pred_root: str | Path, task: str) -> Path:
    path = Path(pred_root).expanduser().resolve()
    if path.name == "gen_imgs" and path.is_dir():
        return path
    candidates = (path / "gen_imgs", path / task / "gen_imgs")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Expected generation images in <pred-root>/gen_imgs/<map> or <pred-root>/<task>/gen_imgs/<map>: {path}"
    )


def _expected_image_path(pred_root: Path, row: Mapping[str, Any]) -> Path:
    return pred_root / str(row["map_name"]) / f"{row['file_frame']}.jpg"


def _validate_image_coverage(
    pred_root: Path,
    rows: Sequence[Mapping[str, Any]],
    maps: Sequence[str],
) -> dict[str, dict[str, int]]:
    expected_by_map: dict[str, set[str]] = {map_name: set() for map_name in maps}
    for row in rows:
        expected_by_map[row["map_name"]].add(f"{row['file_frame']}.jpg")
    if any(not expected_by_map[name] for name in maps):
        raise ValueError("Selected generation split has no rows for at least one map")

    actual_by_map: dict[str, set[str]] = {map_name: set() for map_name in maps}
    extras: list[str] = []
    if not pred_root.is_dir():
        raise FileNotFoundError(f"Prediction root is not a directory: {pred_root}")
    expected_map_set = set(maps)
    for path in pred_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(pred_root)
        if len(relative.parts) != 2 or relative.parts[0] not in expected_map_set:
            extras.append(relative.as_posix())
            continue
        map_name, filename = relative.parts
        actual_by_map[map_name].add(filename)

    missing_by_map = {
        map_name: sorted(expected_by_map[map_name] - actual_by_map[map_name])
        for map_name in maps
    }
    extra_by_map = {
        map_name: sorted(actual_by_map[map_name] - expected_by_map[map_name])
        for map_name in maps
    }
    missing_total = sum(len(names) for names in missing_by_map.values())
    extra_total = sum(len(names) for names in extra_by_map.values()) + len(extras)
    if missing_total or extra_total:
        missing_examples = [
            f"{map_name}/{name}"
            for map_name in maps
            for name in missing_by_map[map_name][:3]
        ][:10]
        extra_examples = [
            f"{map_name}/{name}"
            for map_name in maps
            for name in extra_by_map[map_name][:3]
        ][:10] + extras[:10]
        raise ValueError(
            "Official generation evaluation requires exact complete coverage; "
            f"missing={missing_total} {missing_examples}, extra={extra_total} {extra_examples}"
        )

    return {
        map_name: {
            "expected_count": len(expected_by_map[map_name]),
            "prediction_count": len(actual_by_map[map_name]),
            "common_count": len(expected_by_map[map_name]),
            "coverage_gt": 1,
            "coverage_pred": 1,
        }
        for map_name in maps
    }


def _verify_smoke_images(pred_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Smoke selection is empty")
    missing = [
        str(_expected_image_path(pred_root, row))
        for row in rows
        if not _expected_image_path(pred_root, row).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Smoke prediction image(s) missing: {missing[:10]}")


def _prediction_paths(pred_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    return [_expected_image_path(pred_root, row) for row in rows]


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {map_name: [] for map_name in SEEN_MAPS}
    for row in rows:
        grouped[row["map_name"]].append(row)
    return grouped


def _group_tracks(tracks: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {map_name: [] for map_name in SEEN_MAPS}
    for track in tracks:
        grouped[track["map_name"]].append(track)
    return grouped


def _device_name(requested: str | None) -> str:
    if requested:
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_fvd_cache(config: Mapping[str, Any], override: str | None) -> str:
    configured = override or config["continuous"].get("fvd_cache_dir")
    if configured:
        cache_dir = Path(configured).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = (EVAL_DIR / cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = Path("loaded_models").resolve()
    os.environ["UNILIP_FVD_CACHE_DIR"] = str(cache_dir)
    return str(cache_dir)


def _run_localization(
    data: BenchmarkData,
    pred_root: str | Path,
    *,
    config: Mapping[str, Any],
    pose_space: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, Any]]:
    from metrics_localization import (
        load_predictions,
        localization_metrics,
        validate_localization_coverage,
    )

    split = TASK_SPLITS["localization"]
    expected_rows = data.rows(split)
    prediction_file = _prediction_jsonl_path(pred_root)
    predictions = load_predictions(prediction_file)
    counts = validate_localization_coverage(predictions, expected_rows, data.maps)
    metrics_by_map: dict[str, dict[str, float]] = {}
    coverage_by_map: dict[str, dict[str, int]] = {}
    for map_name in data.maps:
        map_count = counts[map_name]
        metrics_by_map[map_name] = localization_metrics(
            predictions,
            expected_rows,
            map_name,
            pose_space=pose_space,
        )
        coverage_by_map[map_name] = {
            "expected_count": map_count,
            "prediction_count": map_count,
            "common_count": map_count,
            "coverage_gt": 1,
            "coverage_pred": 1,
        }
    return metrics_by_map, coverage_by_map, {
        "split": split,
        "predictions_jsonl": str(prediction_file),
        "prediction_pose_space": pose_space,
    }


def _run_generation(
    task: str,
    data: BenchmarkData,
    pred_root_arg: str | Path,
    *,
    config: Mapping[str, Any],
    device: str,
    fvd_cache_dir: str | None,
    smoke: bool = False,
    smoke_limit: int = 1,
    smoke_max_clips: int = 1,
    smoke_frame_only: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, Any]]:
    from metrics_images import compute_boundary_metrics, compute_fid, compute_lpips, compute_psnr_ssim

    image_cfg = config["images"]
    cont_cfg = config["continuous"]
    split = TASK_SPLITS[task]
    tracks: list[dict[str, Any]] = []
    if task == "continuous":
        if smoke:
            tracks = data.clips(max_clips=smoke_max_clips)
            if smoke_frame_only:
                rows = tracks[0]["rows"][:1]
                tracks = []
            else:
                rows = [row for track in tracks for row in track["rows"]]
        else:
            tracks = data.clips()
            rows = [row for track in tracks for row in track["rows"]]
    else:
        rows = data.rows(split, max_samples=smoke_limit if smoke else None)
    pred_root = _generation_root(pred_root_arg, task)
    if smoke:
        _verify_smoke_images(pred_root, rows)
        coverage_by_map = {
            map_name: {
                "expected_count": len([row for row in rows if row["map_name"] == map_name]),
                "prediction_count": len([row for row in rows if row["map_name"] == map_name]),
                "common_count": len([row for row in rows if row["map_name"] == map_name]),
                "coverage_gt": None,
                "coverage_pred": None,
            }
            for map_name in data.maps
            if any(row["map_name"] == map_name for row in rows)
        }
    else:
        coverage_by_map = _validate_image_coverage(pred_root, rows, data.maps)

    rows_by_map = _group_rows(rows)
    tracks_by_map = _group_tracks(tracks)
    metrics_by_map: dict[str, dict[str, float]] = {}
    details: dict[str, Any] = {
        "split": split,
        "prediction_root": str(pred_root),
        "paired_size": int(image_cfg["paired_size"]),
        "dataloader_workers": int(image_cfg["dataloader_workers"]),
    }
    if task == "continuous" and smoke_frame_only:
        details["continuous_frame_only"] = True
        details["temporal_metrics"] = "skipped: only one frame was selected"
        details["distribution_metrics"] = "skipped in frame-only smoke mode"
        for map_name in data.maps:
            map_rows = rows_by_map[map_name]
            if not map_rows:
                continue
            metrics_by_map[map_name] = _quality_metrics(
                [row["image_path"] for row in map_rows],
                _prediction_paths(pred_root, map_rows),
                size=int(image_cfg["paired_size"]),
                batch_size=int(image_cfg["continuous_batch_size"]),
                device=device,
                num_workers=int(image_cfg["dataloader_workers"]),
                include_boundary=False,
                include_fid=False,
                config=image_cfg,
            )
    elif task == "continuous":
        from metrics_continuous import compute_fvd_for_tracks, compute_temporal_metrics, validate_tracks

        cont_metrics_config = {
            key: int(cont_cfg[key])
            for key in (
                "clip_length",
                "clip_stride",
                "fvd_size",
                "frame_diff_threshold",
                "min_track_len",
            )
        }
        details["continuous_protocol"] = cont_metrics_config
        fvd_counts: dict[str, int] = {}
        for map_name in data.maps:
            map_rows = rows_by_map[map_name]
            map_tracks = tracks_by_map[map_name]
            if not map_rows:
                if smoke:
                    continue
                raise ValueError(f"No continuous samples selected for {map_name}")
            validate_tracks(map_tracks)
            gt_paths = [row["image_path"] for row in map_rows]
            pred_paths = _prediction_paths(pred_root, map_rows)
            quality = _quality_metrics(
                gt_paths,
                pred_paths,
                size=int(image_cfg["paired_size"]),
                batch_size=int(image_cfg["continuous_batch_size"]),
                device=device,
                num_workers=int(image_cfg["dataloader_workers"]),
                include_boundary=False,
                include_fid=False,
                config=image_cfg,
            )
            temporal = compute_temporal_metrics(
                map_tracks,
                pred_root,
                size=int(image_cfg["paired_size"]),
            )
            metrics = {**quality, **temporal}
            if smoke:
                fvd_counts[map_name] = 0
            else:
                _set_fvd_cache(config, fvd_cache_dir)
                fvd, fvd_count = compute_fvd_for_tracks(
                    map_tracks,
                    pred_root,
                    clip_length=int(cont_cfg["clip_length"]),
                    clip_stride=int(cont_cfg["clip_stride"]),
                    size=int(cont_cfg["fvd_size"]),
                    batch_size=int(image_cfg["continuous_batch_size"]),
                    device=device,
                    num_workers=int(image_cfg["dataloader_workers"]),
                )
                expected_fvd_count = int(cont_cfg["expected_clips_per_map"]) * (
                    int(cont_cfg["expected_frames_per_clip"]) // int(cont_cfg["clip_length"])
                )
                if fvd_count != expected_fvd_count:
                    raise ValueError(
                        f"FVD window coverage mismatch for {map_name}: expected {expected_fvd_count}, got {fvd_count}"
                    )
                metrics["FVD"] = fvd
                fvd_counts[map_name] = fvd_count
            metrics_by_map[map_name] = metrics
        details["fvd_window_count_by_map"] = fvd_counts
    else:
        for map_name in data.maps:
            map_rows = rows_by_map[map_name]
            if not map_rows:
                if smoke:
                    continue
                raise ValueError(f"No discrete samples selected for {map_name}")
            gt_paths = [row["image_path"] for row in map_rows]
            pred_paths = _prediction_paths(pred_root, map_rows)
            metrics_by_map[map_name] = _quality_metrics(
                gt_paths,
                pred_paths,
                size=int(image_cfg["paired_size"]),
                batch_size=int(image_cfg["discrete_batch_size"]),
                device=device,
                num_workers=int(image_cfg["dataloader_workers"]),
                include_boundary=True,
                include_fid=not smoke,
                config=image_cfg,
            )
        if smoke:
            details["distribution_metrics"] = "skipped in smoke mode"
    if smoke:
        details["smoke_only"] = True
        details["official_output_written"] = False
    return metrics_by_map, coverage_by_map, details


def _quality_metrics(
    gt_paths: Sequence[str | Path],
    pred_paths: Sequence[str | Path],
    *,
    size: int,
    batch_size: int,
    device: str,
    num_workers: int,
    include_boundary: bool,
    include_fid: bool,
    config: Mapping[str, Any],
) -> dict[str, float]:
    from metrics_images import compute_boundary_metrics, compute_fid, compute_lpips, compute_psnr_ssim

    psnr, ssim = compute_psnr_ssim(
        gt_paths,
        pred_paths,
        size=size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    lpips = compute_lpips(
        gt_paths,
        pred_paths,
        size=size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    metrics = {"PSNR": float(psnr), "SSIM": float(ssim), "LPIPS": float(lpips)}
    if include_boundary:
        boundary = compute_boundary_metrics(
            gt_paths,
            pred_paths,
            size=size,
            batch_size=batch_size,
            device=device,
            edge_quantile=float(config["boundary_edge_quantile"]),
            edge_tolerance=int(config["boundary_tolerance_pixels"]),
            num_workers=num_workers,
        )
        metrics["Boundary_F1"] = boundary["Boundary_F1"]
    if include_fid:
        metrics["FID"] = compute_fid(
            gt_paths,
            pred_paths,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
    return metrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-linking a completed sibling temp file commits atomically while
        # preserving no-clobber semantics if another run created the target.
        os.link(temp_path, output_path)
        temp_path.unlink()
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _finite_metrics(metrics_by_map: Mapping[str, Mapping[str, float]]) -> None:
    for map_name, metrics in metrics_by_map.items():
        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite metric {name} for {map_name}: {value}")


def _equal_map_macro(metrics_by_map: Mapping[str, Mapping[str, float]], maps: Sequence[str]) -> dict[str, float]:
    if list(metrics_by_map) != list(maps):
        raise ValueError(f"Cannot aggregate incomplete maps: expected {list(maps)}, got {list(metrics_by_map)}")
    metric_order = list(metrics_by_map[maps[0]])
    for map_name in maps[1:]:
        if list(metrics_by_map[map_name]) != metric_order:
            raise ValueError(f"Metric set/order differs for map={map_name}")
    return {
        name: sum(float(metrics_by_map[map_name][name]) for map_name in maps) / len(maps)
        for name in metric_order
    }


def _write_official_results(
    *,
    task: str,
    data: BenchmarkData,
    output: str | Path,
    metrics_by_map: Mapping[str, Mapping[str, float]],
    coverage_by_map: Mapping[str, Mapping[str, int]],
    details: Mapping[str, Any],
    config_path: Path,
) -> Path:
    if list(metrics_by_map) != list(data.maps) or list(coverage_by_map) != list(data.maps):
        raise ValueError("Refusing to write formal results without every Seen-10 map")
    _finite_metrics(metrics_by_map)
    macro = _equal_map_macro(metrics_by_map, data.maps)
    output_root = Path(output).expanduser().resolve()
    _assert_official_output_available(output_root)
    manifest_hash = _sha256_file(data.manifest_path)
    report_hash = _sha256_file(data.report_path)
    per_map_payloads = {}
    for map_name in data.maps:
        per_map_payloads[map_name] = {
            "schema_version": 1,
            "benchmark_id": "csgo_benchmark_v2",
            "task": task,
            "split": TASK_SPLITS[task],
            "map_name": map_name,
            "coverage_complete": True,
            "coverage": dict(coverage_by_map[map_name]),
            "metrics_ordered": dict(metrics_by_map[map_name]),
            "benchmark_manifest": str(data.manifest_path),
            "benchmark_manifest_sha256": manifest_hash,
            "minimal_dataset_report": str(data.report_path),
            "minimal_dataset_report_sha256": report_hash,
            "evaluator_config": str(config_path),
        }
        per_map_payloads[map_name]["details"] = dict(details)

    summary = {
        "schema_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "task": task,
        "split": TASK_SPLITS[task],
        "maps": list(data.maps),
        "coverage_complete": True,
        "coverage_by_map": {name: dict(coverage_by_map[name]) for name in data.maps},
        "per_map": {name: dict(metrics_by_map[name]) for name in data.maps},
        "metrics_macro_map": macro,
        "benchmark_manifest": str(data.manifest_path),
        "benchmark_manifest_sha256": manifest_hash,
        "minimal_dataset_report": str(data.report_path),
        "minimal_dataset_report_sha256": report_hash,
        "evaluator_config": str(config_path),
        "details": dict(details),
    }

    # Every map and metric has already completed before the first official file
    # is written. The equal-map summary is committed last as the formal marker.
    for map_name in data.maps:
        _atomic_write_json(per_map_payloads[map_name], output_root / "per_map" / f"{map_name}.json")
    summary_path = output_root / "summary_equal_map.json"
    _atomic_write_json(summary, summary_path)
    return summary_path


def _assert_official_output_available(output_root: Path) -> None:
    """Reject existing result directories before computation and before commit."""
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"Official output path exists and is not a directory: {output_root}")
        try:
            next(output_root.iterdir())
        except StopIteration:
            return
        raise ValueError(
            f"Official output directory is not empty: {output_root}. Choose a new output directory; existing results are never overwritten."
        )


def _add_formal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Output directory for per-map JSON and summary_equal_map.json")
    parser.add_argument("--config", default=str(EVAL_DIR / "benchmark_v2.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--pose-space", choices=("normalized", "physical"), default=None)
    parser.add_argument("--fvd-cache-dir", default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict manifest-driven CSGO Benchmark v2 evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for task in TASK_SPLITS:
        task_parser = subparsers.add_parser(task)
        _add_formal_args(task_parser)
    smoke_parser = subparsers.add_parser("smoke", help="read a strict prefix/subset; never writes formal result files")
    smoke_parser.add_argument("task", choices=tuple(TASK_SPLITS))
    smoke_parser.add_argument("--pred-root", required=True)
    smoke_parser.add_argument("--data-root", required=True)
    smoke_parser.add_argument("--config", default=str(EVAL_DIR / "benchmark_v2.yaml"))
    smoke_parser.add_argument("--device", default=None)
    smoke_parser.add_argument("--pose-space", choices=("normalized", "physical"), default=None)
    smoke_parser.add_argument("--limit", type=int, default=1, help="Global sample prefix for localization/discrete smoke")
    smoke_parser.add_argument("--max-clips", type=int, default=1, help="Whole manifest clips for continuous smoke")
    smoke_parser.add_argument(
        "--frame-only",
        action="store_true",
        help="For continuous smoke, read only the first frame of the first selected clip; paired metrics only",
    )
    smoke_parser.add_argument("--fvd-cache-dir", default=None, help="Not used in smoke mode")
    return parser


def _run_smoke(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    if args.limit <= 0 or args.max_clips <= 0:
        raise ValueError("Smoke limits must be positive")
    if args.frame_only and args.task != "continuous":
        raise ValueError("--frame-only is valid only for continuous smoke")
    data = BenchmarkData(args.data_root)
    device = _device_name(args.device)
    if args.task == "localization":
        from metrics_localization import load_predictions, localization_metrics

        split = TASK_SPLITS[args.task]
        expected_rows = data.rows(split, max_samples=args.limit)
        predictions = load_predictions(_prediction_jsonl_path(args.pred_root), max_rows=args.limit)
        row_by_key = {(row["map_name"], row["file_frame"]): row for row in expected_rows}
        pred_keys: set[tuple[str, str]] = set()
        selected_predictions = []
        for pred in predictions:
            map_name = pred.get("map_name", pred.get("map"))
            sample_id = pred.get("sample_id")
            if not isinstance(map_name, str):
                raise ValueError("Smoke localization row is missing map_name")
            from metrics_localization import _file_frame_from_sample_id

            key = (map_name, _file_frame_from_sample_id(sample_id, map_name))
            if key not in row_by_key or key in pred_keys:
                raise ValueError(f"Smoke prediction does not match its selected row: {key}")
            pred_keys.add(key)
            selected_predictions.append(pred)
        if len(selected_predictions) != len(expected_rows):
            raise ValueError(f"Smoke requires {len(expected_rows)} selected prediction(s), got {len(selected_predictions)}")
        first_map = expected_rows[0]["map_name"]
        result = localization_metrics(
            selected_predictions,
            expected_rows,
            first_map,
            pose_space=args.pose_space or config["localization"]["prediction_pose_space"],
        )
        report = {"metrics": result, "sample_count": len(expected_rows), "map_name": first_map}
    else:
        metrics_by_map, coverage, details = _run_generation(
            args.task,
            data,
            args.pred_root,
            config=config,
            device=device,
            fvd_cache_dir=args.fvd_cache_dir,
            smoke=True,
            smoke_limit=args.limit,
            smoke_max_clips=args.max_clips,
            smoke_frame_only=args.frame_only,
        )
        report = {"metrics_by_map": metrics_by_map, "coverage_read": coverage, "details": details}
    output = {
        "smoke_only": True,
        "formal": False,
        "official_output_written": False,
        "task": args.task,
        "split": TASK_SPLITS[args.task],
        "metrics": report,
        "config": str(config_path),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_arg = args.config
    config_path = Path(config_arg).expanduser().resolve()
    try:
        config = _read_config(config_path)
        if args.command == "smoke":
            _run_smoke(args, config, config_path)
            return 0

        task = args.command
        _assert_official_output_available(Path(args.output).expanduser().resolve())
        data = BenchmarkData(args.data_root)
        device = _device_name(args.device)
        if task == "localization":
            pose_space = args.pose_space or config["localization"]["prediction_pose_space"]
            metrics, coverage, details = _run_localization(
                data,
                args.pred_root,
                config=config,
                pose_space=pose_space,
            )
        else:
            fvd_cache = args.fvd_cache_dir
            if task == "continuous":
                fvd_cache = _set_fvd_cache(config, fvd_cache)
            metrics, coverage, details = _run_generation(
                task,
                data,
                args.pred_root,
                config=config,
                device=device,
                fvd_cache_dir=fvd_cache,
            )
        summary_path = _write_official_results(
            task=task,
            data=data,
            output=args.output,
            metrics_by_map=metrics,
            coverage_by_map=coverage,
            details=details,
            config_path=config_path,
        )
        print(f"Wrote per-map JSON under {summary_path.parent / 'per_map'}")
        print(f"Wrote equal-map summary to {summary_path}")
        return 0
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
