"""Deterministic qualitative localization visualizations for Seen-10."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .dataset import SEEN_MAPS, Seen10Dataset


SAMPLE_COLORS = (
    (255, 72, 72),
    (56, 220, 95),
    (72, 132, 255),
    (255, 218, 48),
    (255, 75, 220),
    (30, 225, 225),
    (255, 145, 40),
    (178, 105, 255),
    (245, 245, 245),
    (100, 255, 185),
)
POSE_FIELDS = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")


def select_visualization_identities(records, samples_per_map: int, seed: int):
    """Choose a reproducible random subset independently inside every map."""
    if samples_per_map <= 0:
        raise ValueError("visualization samples_per_map must be positive")
    if samples_per_map > len(SAMPLE_COLORS):
        raise ValueError(f"At most {len(SAMPLE_COLORS)} visualization samples are supported")
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["map_name"])].append(
            (str(record["map_name"]), str(record["sample_id"]))
        )
    rng = random.Random(int(seed))
    selected = []
    for map_name in SEEN_MAPS:
        candidates = grouped[map_name]
        count = min(samples_per_map, len(candidates))
        selected.extend(rng.sample(candidates, count))
    return selected


def gather_visualization_rows(local_rows, accelerator):
    """Gather small JSON-compatible visualization rows without padding."""
    if accelerator.num_processes == 1:
        return list(local_rows)
    import torch.distributed as dist

    gathered = [None] * accelerator.num_processes if accelerator.is_main_process else None
    dist.gather_object(list(local_rows), gathered, dst=0)
    if not accelerator.is_main_process:
        return []
    return [row for rank_rows in gathered for row in rank_rows]


def build_inference_visualization_rows(
    dataset: Seen10Dataset,
    prediction_path: str | Path,
    *,
    samples_per_map: int,
    seed: int,
):
    """Join test GT after inference; GT never enters model inputs or prediction JSONL."""
    selected = select_visualization_identities(dataset.records, samples_per_map, seed)
    selected_set = set(selected)
    predictions = {}
    with Path(prediction_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (str(row["map_name"]), str(row["sample_id"]))
            if identity not in selected_set:
                continue
            values = [float(row[field]) for field in POSE_FIELDS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite visualization prediction at line {line_number}: {identity}")
            if identity in predictions:
                raise ValueError(f"Duplicate visualization prediction at line {line_number}: {identity}")
            predictions[identity] = values

    split_rows = {}
    for map_name in SEEN_MAPS:
        split_path = dataset.data_root / "splits" / "seen" / map_name / f"{dataset.split}.json"
        if not split_path.is_file():
            split_path = split_path.with_suffix(".jsonl")
        if split_path.suffix == ".jsonl":
            with split_path.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        else:
            with split_path.open(encoding="utf-8") as stream:
                rows = json.load(stream)
        for row in rows:
            identity = (map_name, str(row.get("sample_id", row["file_frame"])))
            if identity not in selected_set:
                continue
            if identity in split_rows:
                raise ValueError(f"Duplicate visualization ground truth: {identity}")
            low, high = dataset.z_ranges[map_name]
            split_rows[identity] = [
                float(row["x"]) / 1024.0,
                float(row["y"]) / 1024.0,
                (float(row["z"]) - low) / (high - low),
                float(row["angle_v"]) / math.tau,
                float(row["angle_h"]) / math.tau,
            ]

    missing_predictions = [identity for identity in selected if identity not in predictions]
    missing_gt = [identity for identity in selected if identity not in split_rows]
    if missing_predictions or missing_gt:
        raise ValueError(
            "Cannot render localization visualization; "
            f"missing predictions={missing_predictions[:5]}, missing GT={missing_gt[:5]}"
        )
    return [
        {
            "map_name": map_name,
            "sample_id": sample_id,
            "gt_normalized": split_rows[(map_name, sample_id)],
            "pred_normalized": predictions[(map_name, sample_id)],
        }
        for map_name, sample_id in selected
    ]


def _font(size: int, *, bold: bool = False):
    filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def normalized_to_physical(pose, map_name: str, z_ranges):
    if len(pose) != 5:
        raise ValueError(f"Expected normalized 5DoF pose, got {pose!r}")
    low, high = z_ranges[map_name]
    return (
        float(pose[0]) * 1024.0,
        float(pose[1]) * 1024.0,
        float(pose[2]) * (high - low) + low,
        float(pose[3]) * 360.0,
        float(pose[4]) * 360.0,
    )


def map_pixel_from_normalized(x: float, y: float, width: int, height: int, margin: int):
    """Map normalized XY to a visible radar point, clipping out-of-range predictions."""
    px = min(max(float(x) * width, margin), width - margin - 1)
    py = min(max(float(y) * height, margin), height - margin - 1)
    return px, py


def _fit_fpv(path: str | Path, size):
    with Image.open(path) as source:
        return ImageOps.fit(
            source.convert("RGB"),
            size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def _text(draw, xy, value, font, anchor="la"):
    draw.text(xy, value, font=font, fill="white", stroke_width=2, stroke_fill="black", anchor=anchor)


def render_map_visualizations(
    dataset: Seen10Dataset,
    rows,
    output_dir: str | Path,
    *,
    samples_per_map: int,
    seed: int,
    radar_size: int = 1800,
    fpv_width: int = 480,
):
    """Render map-left / ten-FPV-right qualitative panels, one PNG per map."""
    if samples_per_map > len(SAMPLE_COLORS):
        raise ValueError(f"At most {len(SAMPLE_COLORS)} visualization samples are supported")
    if radar_size <= 0 or fpv_width <= 0:
        raise ValueError("visualization dimensions must be positive")
    record_by_identity = {
        (str(record["map_name"]), str(record["sample_id"])): record
        for record in dataset.records
    }
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (str(row["map_name"]), str(row["sample_id"]))
        if identity in seen:
            raise ValueError(f"Duplicate visualization identity: {identity}")
        if identity not in record_by_identity:
            raise ValueError(f"Visualization identity is outside dataset: {identity}")
        seen.add(identity)
        grouped[identity[0]].append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_height = radar_size // samples_per_map
    panel_height = row_height * samples_per_map
    radar_size = panel_height
    body_font = _font(max(11, min(14, row_height // 12)))
    marker_margin = 20
    outputs = []
    manifest = {
        "seed": int(seed),
        "requested_samples_per_map": int(samples_per_map),
        "pose_order": "xyzhw = x, y, z, pitch, yaw; angles shown in degrees",
        "layout": "radar left, FPV column right; GT solid, prediction hollow",
        "fpv_text_alignment": "top center",
        "sample_colors": len(SAMPLE_COLORS),
        "maps": {
            map_name: [str(row["sample_id"]) for row in grouped[map_name]]
            for map_name in SEEN_MAPS
            if grouped[map_name]
        },
    }
    manifest_path = output_dir / "visualization_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            existing_manifest = json.load(stream)
        if existing_manifest != manifest:
            raise ValueError(
                f"Visualization selection/config differs from existing output at {manifest_path}"
            )
        existing_outputs = [output_dir / f"{map_name}.png" for map_name in manifest["maps"]]
        if all(path.is_file() for path in existing_outputs):
            return existing_outputs

    for map_name in SEEN_MAPS:
        map_rows = grouped[map_name]
        expected = min(samples_per_map, dataset.counts[map_name])
        if len(map_rows) != expected:
            raise ValueError(f"Expected {expected} visualization rows for {map_name}, found {len(map_rows)}")
        if not map_rows:
            continue
        radar_path = Path(record_by_identity[(map_name, str(map_rows[0]["sample_id"]))]["radar_path"])
        with Image.open(radar_path) as source:
            radar = source.convert("RGB").resize((radar_size, radar_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (radar_size + fpv_width, panel_height), "black")
        canvas.paste(radar, (0, 0))
        radar_draw = ImageDraw.Draw(canvas)

        for index, row in enumerate(map_rows):
            color = SAMPLE_COLORS[index]
            record = record_by_identity[(map_name, str(row["sample_id"]))]
            gt_norm = [float(value) for value in row["gt_normalized"]]
            pred_norm = [float(value) for value in row["pred_normalized"]]
            if not all(math.isfinite(value) for value in gt_norm + pred_norm):
                raise ValueError(f"Nonfinite pose in visualization row {(map_name, row['sample_id'])}")
            gt_xy = map_pixel_from_normalized(gt_norm[0], gt_norm[1], radar_size, radar_size, marker_margin)
            pred_xy = map_pixel_from_normalized(pred_norm[0], pred_norm[1], radar_size, radar_size, marker_margin)
            radar_draw.line((gt_xy, pred_xy), fill=(245, 245, 245), width=3)
            gt_radius = 11
            radar_draw.ellipse(
                (gt_xy[0] - gt_radius, gt_xy[1] - gt_radius, gt_xy[0] + gt_radius, gt_xy[1] + gt_radius),
                fill=color,
                outline="black",
                width=3,
            )
            pred_radius = 17
            radar_draw.ellipse(
                (pred_xy[0] - pred_radius, pred_xy[1] - pred_radius, pred_xy[0] + pred_radius, pred_xy[1] + pred_radius),
                outline="black",
                width=8,
            )
            radar_draw.ellipse(
                (pred_xy[0] - pred_radius, pred_xy[1] - pred_radius, pred_xy[0] + pred_radius, pred_xy[1] + pred_radius),
                outline=color,
                width=5,
            )

            fpv = _fit_fpv(record["image_path"], (fpv_width, row_height))
            fpv_draw = ImageDraw.Draw(fpv, "RGBA")
            text_height = max(48, row_height // 3)
            fpv_draw.rectangle((0, 0, fpv_width, text_height), fill=(0, 0, 0, 170))
            tag_radius = max(7, min(11, row_height // 12))
            fpv_draw.ellipse(
                (7, 7, 7 + 2 * tag_radius, 7 + 2 * tag_radius),
                fill=color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )
            gt = normalized_to_physical(gt_norm, map_name, dataset.z_ranges)
            pred = normalized_to_physical(pred_norm, map_name, dataset.z_ranges)
            gt_label = "gt_xyzhw   [" + ",".join(f"{value:.1f}" for value in gt) + "]"
            pred_label = "pred_xyzhw [" + ",".join(f"{value:.1f}" for value in pred) + "]"
            # Keep both pose labels centered in the FPV tile.  The color tag
            # stays at the left edge and does not shift the text alignment.
            text_center_x = fpv_width / 2
            _text(fpv_draw, (text_center_x, 5), gt_label, body_font, anchor="mt")
            _text(fpv_draw, (text_center_x, 25), pred_label, body_font, anchor="mt")
            canvas.paste(fpv, (radar_size, index * row_height))
        output_path = output_dir / f"{map_name}.png"
        if not output_path.exists():
            temporary = output_dir / f".{map_name}.png.tmp"
            canvas.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, output_path)
        outputs.append(output_path)

    if not manifest_path.exists():
        temporary = output_dir / ".visualization_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    return outputs
