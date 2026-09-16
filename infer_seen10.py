"""Run native XVLA action generation for missing Seen-10 test records."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler

from csgo_seen10.dataset import COORDINATE_DESCRIPTION, Seen10Collator, Seen10Dataset
from csgo_seen10.model import load_seen10_model, load_seen10_processor
from csgo_seen10.visualization import build_inference_visualization_rows, render_map_visualizations
from train import set_seed


LOGGER = logging.getLogger("infer_seen10")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/csgo_seen10.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use the isolated, nonformal smoke output root")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", help="Exact output directory for this seed")
    parser.add_argument("--checkpoint", help="Selected native XVLA checkpoint; defaults to OUTPUT_ROOT/checkpoints/best")
    parser.add_argument("--limit-per-map", type=int, help="Allowed only with --smoke")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--inference-seed", type=int)
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_output_root(args, config: dict) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    train_cfg = config["training"]
    root = train_cfg["smoke_output_root"] if args.smoke else train_cfg["output_root"]
    return (Path(root).expanduser() / f"seed_{args.seed}").resolve()


def _load_existing_predictions(path: Path, expected_keys: set[tuple[str, str]]):
    existing = {}
    if not path.exists():
        return existing
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from error
            required = {"sample_id", "map_name", "pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw"}
            if not required.issubset(row):
                raise ValueError(f"Prediction row in {path}:{line_no} is missing {sorted(required - row.keys())}")
            key = (str(row["map_name"]), str(row["sample_id"]))
            if key not in expected_keys:
                raise ValueError(f"Existing prediction is outside the requested test split: {key}")
            if key in existing:
                raise ValueError(f"Duplicate prediction identity in {path}:{line_no}: {key}")
            values = [float(row[name]) for name in ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite prediction in {path}:{line_no}: {key}")
            existing[key] = row
    return existing


def _expected_metadata(
    *, args, checkpoint: Path, progress: dict, data_root: Path, dataset: Seen10Dataset,
    batch_size: int, steps: int, world_size: int, inference_seed: int,
):
    if not progress.get("checkpoint_id"):
        raise ValueError(f"Checkpoint has no immutable checkpoint_id: {checkpoint}")
    if progress.get("seed") != args.seed:
        raise ValueError(
            f"Checkpoint seed {progress.get('seed')} does not match requested seed {args.seed}"
        )
    if progress.get("smoke") != args.smoke:
        raise ValueError("Checkpoint smoke/formal status does not match the requested inference mode")
    if progress.get("data_root") != str(data_root):
        raise ValueError("Checkpoint and inference use different CSGO data roots")
    if progress.get("coordinates") != COORDINATE_DESCRIPTION:
        raise ValueError("Checkpoint coordinate metadata does not match the CSGO normalized 5DoF contract")
    return {
        "schema": "xvla_csgo_seen10_localization_predictions_v1",
        "seed": args.seed,
        "smoke": args.smoke,
        "checkpoint": str(checkpoint),
        "checkpoint_id": progress["checkpoint_id"],
        "checkpoint_step": int(progress["global_step"]),
        "data_root": str(data_root),
        "split": "seen_discrete_test",
        "limit_per_map": dataset.limit_per_map,
        "expected_samples": len(dataset),
        "batch_size": batch_size,
        "steps": steps,
        "world_size": world_size,
        "inference_seed": inference_seed,
        "coordinates": COORDINATE_DESCRIPTION,
    }


def _write_or_verify_metadata(path: Path, expected: dict, predictions_path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            current = json.load(stream)
        if current != expected:
            raise ValueError(
                f"Existing prediction metadata at {path} belongs to a different seed/checkpoint/split; "
                "choose a fresh output-root"
            )
    elif predictions_path.exists() and predictions_path.stat().st_size > 0:
        raise ValueError(
            f"Existing predictions have no provenance sidecar ({path}); refusing to mix checkpoints"
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)


def _prepare_inference_output(output_root: Path, *, args, config: dict):
    formal_root = Path(config["training"]["output_root"]).expanduser().resolve()
    smoke_root = Path(config["training"]["smoke_output_root"]).expanduser().resolve()
    marker_path = output_root / "NON_FORMAL_SMOKE.json"
    if args.smoke and output_root.is_relative_to(formal_root):
        raise ValueError("--smoke inference cannot write beneath the formal benchmark output root")
    if not args.smoke and output_root.is_relative_to(smoke_root):
        raise ValueError("Formal inference cannot write beneath the smoke output root")
    if args.smoke:
        output_root.mkdir(parents=True, exist_ok=True)
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("non_formal_smoke") is not True or marker.get("seed", args.seed) != args.seed:
                raise ValueError(f"Smoke marker does not match this seed/run: {marker_path}")
        else:
            marker = {
                "non_formal_smoke": True,
                "formal_benchmark_result": False,
                "seed": args.seed,
            }
            marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    elif marker_path.exists():
        raise ValueError(f"Formal inference refuses a smoke output directory: {output_root}")


def _write_prediction(stream, metadata: dict, prediction):
    row = {
        "sample_id": str(metadata["sample_id"]),
        "map_name": str(metadata["map_name"]),
        "pred_x": float(prediction[0]),
        "pred_y": float(prediction[1]),
        "pred_z": float(prediction[2]),
        "pred_pitch": float(prediction[3]),
        "pred_yaw": float(prediction[4]),
    }
    if not all(math.isfinite(row[name]) for name in ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")):
        raise ValueError(f"XVLA produced a nonfinite action for {(row['map_name'], row['sample_id'])}")
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _render_inference_visualizations(
    accelerator: Accelerator,
    dataset: Seen10Dataset,
    prediction_path: Path,
    output_root: Path,
    visualization_cfg: dict,
):
    status = [None]
    if accelerator.is_main_process:
        try:
            samples_per_map = int(visualization_cfg.get("samples_per_map", 10))
            visualization_seed = int(visualization_cfg.get("seed", 20260915))
            rows = build_inference_visualization_rows(
                dataset,
                prediction_path,
                samples_per_map=samples_per_map,
                seed=visualization_seed,
            )
            paths = render_map_visualizations(
                dataset,
                rows,
                output_root / "visualizations" / "localization",
                samples_per_map=samples_per_map,
                seed=visualization_seed,
                radar_size=int(visualization_cfg.get("radar_size", 1800)),
                fpv_width=int(visualization_cfg.get("fpv_width", 480)),
            )
            status[0] = {"ok": True, "paths": [str(path) for path in paths]}
        except Exception as error:
            status[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        raise RuntimeError(f"Inference visualization failed: {status[0]['error']}")
    accelerator.wait_for_everyone()
    return status[0]["paths"]


def main():
    args = _parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.inference_seed is not None and args.inference_seed < 0:
        raise ValueError("inference-seed must be nonnegative")
    if args.limit_per_map is not None and not args.smoke:
        raise ValueError("--limit-per-map is only allowed with --smoke")
    config = _load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    inference_cfg = config["inference"]
    visualization_cfg = config.get("visualization", {})
    visualization_enabled = bool(visualization_cfg.get("enabled", True))
    data_root = Path(args.data_root or os.environ.get("CSGO_DATA_ROOT") or data_cfg["root"]).expanduser().resolve()
    output_root = _resolve_output_root(args, config)
    checkpoint = Path(args.checkpoint or (output_root / "checkpoints" / "best")).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")
    with (checkpoint / "progress.json").open(encoding="utf-8") as stream:
        progress = json.load(stream)

    limit_per_map = args.limit_per_map
    if args.smoke and limit_per_map is None:
        limit_per_map = config["training"].get("smoke_limit_per_map", 8)
    batch_size = args.batch_size or (inference_cfg["smoke_batch_size"] if args.smoke else inference_cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else inference_cfg.get("num_workers", data_cfg.get("num_workers", 0))
    steps = args.steps or inference_cfg.get("steps", model_cfg.get("inference_steps", 10))
    inference_seed = args.inference_seed if args.inference_seed is not None else inference_cfg.get("seed", 42)
    if min(batch_size, steps) <= 0 or num_workers < 0:
        raise ValueError("batch-size and steps must be positive; num-workers must be nonnegative")

    accelerator = Accelerator()
    dataset = Seen10Dataset(data_root, "seen_discrete_test", include_targets=False, limit_per_map=limit_per_map)
    expected_keys = {
        (str(record["map_name"]), str(record["sample_id"]))
        for record in dataset.records
    }
    if len(expected_keys) != len(dataset):
        raise ValueError("Test dataset contains duplicate sample identities")
    key_to_index = {
        (str(record["map_name"]), str(record["sample_id"])): index
        for index, record in enumerate(dataset.records)
    }
    prediction_path = output_root / "localization" / "predictions.jsonl"
    metadata_path = output_root / "localization" / "predictions.meta.json"
    logger = logging.getLogger("infer_seen10")
    logger.setLevel(logging.INFO)
    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    preflight = [None]
    if accelerator.is_main_process:
        try:
            _prepare_inference_output(output_root, args=args, config=config)
            existing = _load_existing_predictions(prediction_path, expected_keys)
            expected_metadata = _expected_metadata(
                args=args,
                checkpoint=checkpoint,
                progress=progress,
                data_root=data_root,
                dataset=dataset,
                batch_size=batch_size,
                steps=steps,
                world_size=accelerator.num_processes,
                inference_seed=inference_seed,
            )
            _write_or_verify_metadata(metadata_path, expected_metadata, prediction_path)
            existing_indices = [key_to_index[identity] for identity in existing]
            preflight[0] = {
                "ok": True,
                "existing_indices": existing_indices,
                "complete": len(existing) == len(dataset),
            }
        except Exception as error:
            preflight[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(preflight, src=0)
    if not preflight[0]["ok"]:
        raise RuntimeError(f"Inference preflight failed: {preflight[0]['error']}")
    existing_indices = set(preflight[0]["existing_indices"])
    existing = {
        (str(dataset.records[index]["map_name"]), str(dataset.records[index]["sample_id"]))
        for index in existing_indices
    }
    if preflight[0]["complete"]:
        if accelerator.is_main_process:
            print(f"Predictions already complete ({len(existing)} samples): {prediction_path}")
        if visualization_enabled:
            visualization_paths = _render_inference_visualizations(
                accelerator,
                dataset,
                prediction_path,
                output_root,
                visualization_cfg,
            )
            if accelerator.is_main_process:
                LOGGER.info("Localization visualizations ready for %d maps", len(visualization_paths))
        accelerator.end_training()
        return

    set_seed(inference_seed + accelerator.process_index)

    processor = load_seen10_processor(str(checkpoint))
    model = load_seen10_model(str(checkpoint)).to(accelerator.device)
    model.eval()
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        collate_fn=Seen10Collator(processor, include_targets=False),
    )
    written = set(existing)
    if accelerator.is_main_process:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        output_stream = prediction_path.open("a", encoding="utf-8", buffering=1)
    else:
        output_stream = None

    try:
        for batch_index, batch in enumerate(dataloader):
            local_indices = torch.tensor(
                [key_to_index[(str(meta["map_name"]), str(meta["sample_id"]))] for meta in batch["metadata"]],
                dtype=torch.long,
                device=accelerator.device,
            )
            gathered_indices = accelerator.gather(local_indices).detach().cpu().tolist()
            if all(int(index) in existing_indices for index in gathered_indices):
                continue
            inputs = batch["inputs"]
            if "action" in inputs:
                raise ValueError("seen_discrete_test must not expose ground-truth actions to XVLA")
            required = {"input_ids", "image_input", "image_mask", "domain_id", "proprio"}
            missing = required - inputs.keys()
            if missing:
                raise KeyError(f"Seen10Collator omitted model inputs: {sorted(missing)}")
            inputs = {
                key: value.to(device=accelerator.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            if inputs["proprio"].shape[-1] != 20 or torch.count_nonzero(inputs["proprio"]).item() != 0:
                raise ValueError("CSGO inference requires an all-zero 20D proprio vector")
            if inputs["image_mask"].shape[1] != 2 or not torch.all(inputs["image_mask"]):
                raise ValueError("Expected exactly two valid views: first-person and radar")
            set_seed(inference_seed + batch_index * accelerator.num_processes + accelerator.process_index)
            with torch.no_grad():
                predictions = model.generate_actions(**inputs, steps=steps).float()
            if tuple(predictions.shape[1:]) != (1, 5):
                raise ValueError(f"Expected generated predictions [B,1,5], got {tuple(predictions.shape)}")

            gathered_predictions = accelerator.gather(predictions).detach().cpu().numpy()[:, 0, :]
            if accelerator.is_main_process:
                for index, values in zip(gathered_indices, gathered_predictions):
                    record = dataset.records[int(index)]
                    identity = (str(record["map_name"]), str(record["sample_id"]))
                    if identity in written:
                        continue
                    _write_prediction(output_stream, record, values)
                    written.add(identity)
                output_stream.flush()
                os.fsync(output_stream.fileno())
    finally:
        if output_stream is not None:
            output_stream.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if written != expected_keys:
            missing = len(expected_keys - written)
            raise RuntimeError(f"Inference ended with {missing} missing test predictions")
        LOGGER.info("Wrote complete Seen-10 predictions (%d samples): %s", len(written), prediction_path)
        print(f"Wrote {len(written)} predictions: {prediction_path}")
    if visualization_enabled:
        visualization_paths = _render_inference_visualizations(
            accelerator,
            dataset,
            prediction_path,
            output_root,
            visualization_cfg,
        )
        if accelerator.is_main_process:
            LOGGER.info("Localization visualizations ready for %d maps", len(visualization_paths))
    accelerator.end_training()


if __name__ == "__main__":
    main()
