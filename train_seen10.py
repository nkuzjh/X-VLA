"""Minimal Accelerate wrapper for horizon-one CSGO Seen-10 localization."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from csgo_seen10.dataset import COORDINATE_DESCRIPTION, Seen10Collator, Seen10Dataset
from csgo_seen10.model import load_seen10_model, load_seen10_processor, processor_size
from csgo_seen10.visualization import (
    gather_visualization_rows,
    render_map_visualizations,
    select_visualization_identities,
)
from train import build_optimizer, get_logger, set_seed, update_group_lrs


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding so each record contributes once."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def _load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/csgo_seen10.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use the isolated, nonformal smoke output root")
    parser.add_argument("--data-root")
    parser.add_argument("--pretrained")
    parser.add_argument("--output-root", help="Exact output directory for this seed")
    parser.add_argument("--iters", type=int, help="Total optimizer steps, including steps completed before resume")
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--limit-per-map", type=int, help="Allowed only with --smoke")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"))
    parser.add_argument("--resume", help="Native checkpoint directory containing progress.json and accelerator_state/")
    return parser.parse_args()


def _resolve_output_root(args, config: dict) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    train_cfg = config["training"]
    root = train_cfg["smoke_output_root"] if args.smoke else train_cfg["output_root"]
    return (Path(root).expanduser() / f"seed_{args.seed}").resolve()


def _check_output_root(output_root: Path, *, smoke: bool, resume: Path | None):
    if resume is not None:
        if not resume.is_dir() or not (resume / "progress.json").is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing progress.json: {resume}")
        if not (resume / "accelerator_state").is_dir():
            raise FileNotFoundError(f"Resume checkpoint is missing accelerator_state/: {resume}")
        if not output_root.is_dir():
            raise FileNotFoundError(f"Resume requires the original output directory: {output_root}")
        if not resume.is_relative_to(output_root):
            raise ValueError("Resume checkpoint must belong to the selected output-root")
        return

    if not output_root.exists():
        return
    entries = list(output_root.iterdir())
    if not entries:
        return
    if smoke and len(entries) == 1 and entries[0].name == "NON_FORMAL_SMOKE.json":
        try:
            marker = json.loads(entries[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid smoke marker in {output_root}") from error
        if marker.get("non_formal_smoke") is True:
            return
    raise FileExistsError(
        f"Refusing to train into nonempty output directory {output_root}; "
        "choose a new output-root or resume from one of its checkpoints"
    )


def _prepare_output_root(accelerator, output_root: Path, *, args, config: dict, resume: Path | None):
    """Only rank zero checks and creates the run directory, then broadcasts status."""
    status = [None]
    if accelerator.is_main_process:
        try:
            formal_root = Path(config["training"]["output_root"]).expanduser().resolve()
            smoke_root = Path(config["training"]["smoke_output_root"]).expanduser().resolve()
            if args.smoke and output_root.is_relative_to(formal_root):
                raise ValueError("--smoke cannot write beneath the formal benchmark output root")
            if not args.smoke and output_root.is_relative_to(smoke_root):
                raise ValueError("Formal training cannot write beneath the smoke output root")
            _check_output_root(output_root, smoke=args.smoke, resume=resume)
            output_root.mkdir(parents=True, exist_ok=True)
            if args.smoke:
                marker_path = output_root / "NON_FORMAL_SMOKE.json"
                if marker_path.exists():
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    if marker.get("non_formal_smoke") is not True or marker.get("seed", args.seed) != args.seed:
                        raise ValueError(f"Invalid smoke marker: {marker_path}")
                else:
                    marker = {
                        "non_formal_smoke": True,
                        "formal_benchmark_result": False,
                        "seed": args.seed,
                    }
                    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            elif (output_root / "NON_FORMAL_SMOKE.json").exists():
                raise ValueError(f"Formal training refuses a smoke output directory: {output_root}")
            status[0] = {"ok": True}
        except Exception as error:
            status[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        raise RuntimeError(f"Output preflight failed: {status[0]['error']}")
    accelerator.wait_for_everyone()


def _move_inputs(batch, device: torch.device, *, training: bool):
    inputs = batch["inputs"]
    required = {"input_ids", "image_input", "image_mask", "domain_id", "proprio"}
    missing = required - inputs.keys()
    if missing:
        raise KeyError(f"Seen10Collator omitted model inputs: {sorted(missing)}")
    if training and "action" not in inputs:
        raise KeyError("Training/validation batches must include supervised action targets")
    if not training and "action" in inputs:
        raise ValueError("Test inference inputs must not contain ground-truth actions")
    moved = {
        key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    if moved["proprio"].shape[-1] != 20 or torch.count_nonzero(moved["proprio"]).item() != 0:
        raise ValueError("CSGO localization requires an all-zero 20D proprio vector")
    if moved["image_mask"].shape[1] != 2 or not torch.all(moved["image_mask"]):
        raise ValueError("Expected exactly two valid views: first-person and radar")
    if training and tuple(moved["action"].shape[1:]) != (1, 5):
        raise ValueError(f"Expected normalized action target [B,1,5], got {tuple(moved['action'].shape)}")
    return moved


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate_validation(
    model,
    dataloader,
    accelerator: Accelerator,
    *,
    seed: int,
    steps: int,
    visualization_identities=(),
):
    """Select checkpoints by free-running normalized pose error.

    The model samples an initial action in generate_actions. Evaluation uses a
    fixed seed and restores each rank's training RNG state afterwards.
    """
    rng_state = _capture_rng_state()
    set_seed(seed + accelerator.process_index)
    model.eval()
    local_stats = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    selected = set(visualization_identities)
    local_visualization_rows = []
    try:
        for batch in dataloader:
            inputs = _move_inputs(batch, accelerator.device, training=True)
            target = inputs.pop("action")
            prediction = model.generate_actions(**inputs, steps=steps)
            if prediction.shape != target.shape:
                raise ValueError(f"Generated action {tuple(prediction.shape)} != target {tuple(target.shape)}")
            delta = prediction.float() - target.float()
            # Benchmark v2 pitch/yaw are normalized over a 2*pi period.
            delta[..., 4] = torch.remainder(delta[..., 4] + 0.5, 1.0) - 0.5
            per_sample_mse = delta.square().mean(dim=(1, 2))
            per_sample_mae = delta.abs().mean(dim=(1, 2))
            local_stats[0] += per_sample_mse.double().sum()
            local_stats[1] += per_sample_mae.double().sum()
            local_stats[2] += target.shape[0]
            for sample_index, metadata in enumerate(batch["metadata"]):
                identity = (str(metadata["map_name"]), str(metadata["sample_id"]))
                if identity in selected:
                    local_visualization_rows.append({
                        "map_name": identity[0],
                        "sample_id": identity[1],
                        "gt_normalized": target[sample_index, 0].detach().float().cpu().tolist(),
                        "pred_normalized": prediction[sample_index, 0].detach().float().cpu().tolist(),
                    })
    finally:
        _restore_rng_state(rng_state)

    totals = accelerator.reduce(local_stats, reduction="sum")
    visualization_rows = gather_visualization_rows(local_visualization_rows, accelerator)
    count = max(1.0, float(totals[2].item()))
    return (
        float(totals[0].item() / count),
        float(totals[1].item() / count),
        int(totals[2].item()),
        visualization_rows,
    )


def _render_validation_visualizations(
    accelerator: Accelerator,
    dataset: Seen10Dataset,
    rows,
    output_dir: Path,
    visualization_cfg: dict,
):
    status = [None]
    if accelerator.is_main_process:
        try:
            paths = render_map_visualizations(
                dataset,
                rows,
                output_dir,
                samples_per_map=int(visualization_cfg["samples_per_map"]),
                seed=int(visualization_cfg["seed"]),
                radar_size=int(visualization_cfg.get("radar_size", 1800)),
                fpv_width=int(visualization_cfg.get("fpv_width", 480)),
            )
            status[0] = {"ok": True, "paths": [str(path) for path in paths]}
        except Exception as error:
            status[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        raise RuntimeError(f"Validation visualization failed: {status[0]['error']}")
    accelerator.wait_for_everyone()
    return status[0]["paths"]


def _checkpoint_progress(
    *,
    global_step: int,
    epoch: int,
    batches_in_epoch: int,
    seed: int,
    smoke: bool,
    data_root: Path,
    train_dataset: Seen10Dataset,
    validation_dataset: Seen10Dataset,
    best_validation_mse: float,
    processor_image_size,
    batch_size: int,
    world_size: int,
):
    return {
        "global_step": global_step,
        "epoch": epoch,
        "batches_in_epoch": batches_in_epoch,
        "seed": seed,
        "smoke": smoke,
        "data_root": str(data_root),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_limit_per_map": train_dataset.limit_per_map,
        "validation_limit_per_map": validation_dataset.limit_per_map,
        "batch_size": batch_size,
        "world_size": world_size,
        "best_validation_normalized_mse": (
            best_validation_mse if math.isfinite(best_validation_mse) else None
        ),
        "coordinates": COORDINATE_DESCRIPTION,
        "processor_image_size": processor_image_size,
    }


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    processor,
    output_root: Path,
    name: str,
    progress: dict,
):
    checkpoints_dir = output_root / "checkpoints"
    final_dir = checkpoints_dir / name
    temp_dir = checkpoints_dir / f".{name}.tmp"
    if accelerator.is_main_process:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(temp_dir, safe_serialization=True)
        processor.save_pretrained(temp_dir)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temp_dir / "accelerator_state"))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        progress["checkpoint_id"] = uuid.uuid4().hex
        with (temp_dir / "progress.json").open("w", encoding="utf-8") as stream:
            json.dump(progress, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        if final_dir.exists():
            suffix = 1
            while (checkpoints_dir / f"{name}_resume_{suffix:02d}").exists():
                suffix += 1
            final_dir = checkpoints_dir / f"{name}_resume_{suffix:02d}"
        os.replace(temp_dir, final_dir)
    accelerator.wait_for_everyone()
    return final_dir


def _set_checkpoint_pointer(
    accelerator: Accelerator,
    output_root: Path,
    checkpoint: Path,
    progress: dict,
    name: str,
):
    checkpoints_dir = output_root / "checkpoints"
    if accelerator.is_main_process:
        link = checkpoints_dir / f".{name}.tmp"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(checkpoint.name, link)
        os.replace(link, checkpoints_dir / name)
        pointer = {
            "checkpoint": checkpoint.name,
            "checkpoint_id": progress["checkpoint_id"],
            "global_step": progress["global_step"],
            "seed": progress["seed"],
        }
        if "validation_normalized_mse" in progress:
            pointer["validation_normalized_mse"] = progress["validation_normalized_mse"]
        pointer_tmp = checkpoints_dir / f".{name}.json.tmp"
        pointer_tmp.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
        os.replace(pointer_tmp, checkpoints_dir / f"{name}.json")
    accelerator.wait_for_everyone()


def _set_best_checkpoint(accelerator: Accelerator, output_root: Path, checkpoint: Path, progress: dict):
    _set_checkpoint_pointer(accelerator, output_root, checkpoint, progress, "best")


def _set_last_checkpoint(accelerator: Accelerator, output_root: Path, checkpoint: Path, progress: dict):
    _set_checkpoint_pointer(accelerator, output_root, checkpoint, progress, "last")


_STEP_CHECKPOINT_PATTERN = re.compile(r"^step_(\d{8})(?:_resume_(\d+))?$")


def _prune_periodic_checkpoints(accelerator: Accelerator, output_root: Path, max_to_keep: int):
    """Keep recent periodic checkpoints plus any older best/last target."""
    if max_to_keep <= 0:
        raise ValueError("max_periodic_checkpoints must be positive")
    if accelerator.is_main_process:
        checkpoints_dir = output_root / "checkpoints"
        candidates = []
        if checkpoints_dir.is_dir():
            for path in checkpoints_dir.iterdir():
                match = _STEP_CHECKPOINT_PATTERN.fullmatch(path.name)
                if match and path.is_dir() and not path.is_symlink():
                    candidates.append((int(match.group(1)), int(match.group(2) or 0), path))
        candidates.sort(key=lambda item: (item[0], item[1]))
        protected = {path.name for _, _, path in candidates[-max_to_keep:]}
        for pointer_name in ("best", "last"):
            pointer = checkpoints_dir / pointer_name
            if not pointer.is_symlink():
                continue
            target = os.readlink(pointer)
            if Path(target).name == target:
                protected.add(target)
        for _, _, path in candidates:
            if path.name not in protected:
                shutil.rmtree(path)
    accelerator.wait_for_everyone()


def _validate_resume(
    progress: dict,
    *,
    args,
    data_root: Path,
    train_dataset,
    validation_dataset,
    batch_size: int,
    world_size: int,
):
    checks = {
        "seed": args.seed,
        "smoke": args.smoke,
        "data_root": str(data_root),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_limit_per_map": train_dataset.limit_per_map,
        "validation_limit_per_map": validation_dataset.limit_per_map,
        "batch_size": batch_size,
        "world_size": world_size,
    }
    mismatches = [f"{key}: checkpoint={progress.get(key)!r}, requested={value!r}"
                  for key, value in checks.items() if progress.get(key) != value]
    if mismatches:
        raise ValueError("Resume checkpoint does not match this run: " + "; ".join(mismatches))


def main():
    args = _parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.limit_per_map is not None and not args.smoke:
        raise ValueError("--limit-per-map is only allowed with --smoke")
    config = _load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    visualization_cfg = config.get("visualization", {})
    visualization_enabled = bool(visualization_cfg.get("enabled", True))

    data_root = Path(args.data_root or os.environ.get("CSGO_DATA_ROOT") or data_cfg["root"]).expanduser().resolve()
    pretrained = args.pretrained or os.environ.get("XVLA_PRETRAINED") or model_cfg["pretrained"]
    output_root = _resolve_output_root(args, config)
    resume = Path(args.resume).expanduser().resolve() if args.resume else None

    iters = args.iters if args.iters is not None else (train_cfg["smoke_iters"] if args.smoke else train_cfg["iters"])
    eval_interval = args.eval_interval or (train_cfg["smoke_eval_interval"] if args.smoke else train_cfg["eval_interval"])
    limit_per_map = args.limit_per_map
    if args.smoke and limit_per_map is None:
        limit_per_map = train_cfg["smoke_limit_per_map"]
    batch_size = args.batch_size or (train_cfg["smoke_batch_size"] if args.smoke else train_cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", data_cfg.get("num_workers", 0))
    save_interval = args.save_interval or (
        train_cfg["smoke_eval_interval"] if args.smoke else train_cfg["save_interval"]
    )
    max_periodic_checkpoints = int(train_cfg.get("max_periodic_checkpoints", 5))
    log_interval = args.log_interval or train_cfg["log_interval"]
    mixed_precision = args.mixed_precision or train_cfg.get("mixed_precision", "no")
    if min(iters, eval_interval, save_interval, batch_size, max_periodic_checkpoints) <= 0 or num_workers < 0:
        raise ValueError(
            "iters, eval_interval, save_interval, batch_size, and max_periodic_checkpoints "
            "must be positive; num_workers must be nonnegative"
        )
    if save_interval != eval_interval:
        raise ValueError("save_interval must equal eval_interval so every validation has one checkpoint")

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=str(output_root),
        mixed_precision=mixed_precision,
    )
    _prepare_output_root(accelerator, output_root, args=args, config=config, resume=resume)
    accelerator.init_trackers("X-VLA-CSGO-Seen10")
    logger = get_logger("train_seen10", output_dir=output_root, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index)
    logger.info("Starting Seen-10 localization: %s", vars(args))

    train_dataset = Seen10Dataset(data_root, "seen_train", include_targets=True, limit_per_map=limit_per_map)
    validation_dataset = Seen10Dataset(data_root, "seen_validation", include_targets=True, limit_per_map=limit_per_map)
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Seen-10 train and validation datasets must not be empty")
    visualization_identities = []
    if visualization_enabled:
        visualization_identities = select_visualization_identities(
            validation_dataset.records,
            int(visualization_cfg.get("samples_per_map", 10)),
            int(visualization_cfg.get("seed", 20260915)),
        )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    validation_sampler = DistributedEvalSampler(
        validation_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    processor_source = str(resume) if resume is not None else pretrained
    processor = load_seen10_processor(processor_source)
    processor_image_size = processor_size(processor)
    collator_train = Seen10Collator(processor, include_targets=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        collate_fn=collator_train,
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index + 1000),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        collate_fn=Seen10Collator(processor, include_targets=True),
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index + 2000),
    )

    model_source = str(resume) if resume is not None else pretrained
    model = load_seen10_model(model_source)
    if accelerator.is_main_process:
        expected_size = model_cfg.get("processor_image_size_expected")
        if expected_size is not None and processor_image_size != {"height": expected_size, "width": expected_size}:
            logger.warning(
                "Using pretrained processor image size %s (configured expectation: %s); no resize override is applied",
                processor_image_size,
                expected_size,
            )

    optimizer = build_optimizer(
        model=model,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
        lr_coef_soft=train_cfg["learning_coef"],
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    schedule_args = argparse.Namespace(
        learning_rate=train_cfg["learning_rate"],
        learning_coef=train_cfg["learning_coef"],
        freeze_steps=train_cfg["freeze_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        iters=iters,
        min_lr_ratio=train_cfg["min_lr_ratio"],
        use_cosine_decay=train_cfg["use_cosine_decay"],
    )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    start_epoch = 0
    batches_in_epoch = 0
    best_validation_mse = math.inf
    if resume is not None:
        with (resume / "progress.json").open(encoding="utf-8") as stream:
            resume_progress = json.load(stream)
        _validate_resume(
            resume_progress,
            args=args,
            data_root=data_root,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            world_size=accelerator.num_processes,
        )
        accelerator.load_state(str(resume / "accelerator_state"))
        global_step = int(resume_progress["global_step"])
        start_epoch = int(resume_progress["epoch"])
        batches_in_epoch = int(resume_progress["batches_in_epoch"])
        saved_best = resume_progress.get("best_validation_normalized_mse")
        best_validation_mse = math.inf if saved_best is None else float(saved_best)
        existing_best = output_root / "checkpoints" / "best" / "progress.json"
        if existing_best.is_file():
            with existing_best.open(encoding="utf-8") as stream:
                best_progress = json.load(stream)
            if best_progress.get("seed") != args.seed:
                raise ValueError("Existing best checkpoint belongs to a different seed")
            saved_existing_best = best_progress.get("best_validation_normalized_mse")
            if saved_existing_best is not None:
                best_validation_mse = min(best_validation_mse, float(saved_existing_best))
        if global_step > iters:
            raise ValueError(f"Checkpoint step {global_step} exceeds requested total iters {iters}")
        logger.info("Resuming from %s at step %d", resume, global_step)

    if global_step == iters:
        logger.info("Checkpoint already reached requested total step %d; nothing to train", iters)
        accelerator.end_training()
        return

    inference_steps = int(model_cfg.get("inference_steps", 10))
    validation_seed = int(train_cfg.get("validation_seed", 1729))
    started = time.time()
    current_epoch = start_epoch
    skip_batches = batches_in_epoch

    while global_step < iters:
        train_sampler.set_epoch(current_epoch)
        epoch_had_batch = False
        for batch_index, batch in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
            skip_batches = 0
            epoch_had_batch = True
            model.train()
            inputs = _move_inputs(batch, accelerator.device, training=True)
            update_group_lrs(optimizer, global_step, schedule_args)
            loss_dict = model(**inputs)
            loss = sum(loss_dict.values())
            accelerator.backward(loss)
            if train_cfg.get("max_grad_norm"):
                accelerator.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            batches_in_epoch = batch_index + 1
            if global_step % log_interval == 0 or global_step == 1:
                mean_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
                logs = {
                    "loss_total": float(mean_loss.item()),
                    "global_step": global_step,
                }
                if accelerator.is_main_process:
                    logs.update({f"lr_{group['name']}": float(group["lr"]) for group in optimizer.param_groups})
                    logger.info("[%d/%d] loss=%.5f", global_step, iters, logs["loss_total"])
                accelerator.log(logs, step=global_step)

            should_evaluate = global_step % eval_interval == 0 or global_step == iters
            val_mse = val_mae = None
            is_best = False
            if should_evaluate:
                val_mse, val_mae, val_count, visualization_rows = evaluate_validation(
                    accelerator.unwrap_model(model),
                    validation_loader,
                    accelerator,
                    seed=validation_seed,
                    steps=inference_steps,
                    visualization_identities=visualization_identities,
                )
                is_best = val_mse < best_validation_mse
                if is_best:
                    best_validation_mse = val_mse
                if accelerator.is_main_process:
                    logger.info(
                        "validation step=%d samples=%d normalized_mse=%.7f normalized_mae=%.7f%s",
                        global_step,
                        val_count,
                        val_mse,
                        val_mae,
                        " (best)" if is_best else "",
                    )
                accelerator.log(
                    {"validation/normalized_mse": val_mse, "validation/normalized_mae": val_mae},
                    step=global_step,
                )
                if visualization_enabled:
                    visualization_paths = _render_validation_visualizations(
                        accelerator,
                        validation_dataset,
                        visualization_rows,
                        output_root / "visualizations" / "validation" / f"step_{global_step:08d}",
                        visualization_cfg,
                    )
                    if accelerator.is_main_process:
                        logger.info(
                            "Validation visualizations ready for %d maps at step %d",
                            len(visualization_paths),
                            global_step,
                        )

            should_save = should_evaluate
            if should_save:
                progress = _checkpoint_progress(
                    global_step=global_step,
                    epoch=current_epoch,
                    batches_in_epoch=batches_in_epoch,
                    seed=args.seed,
                    smoke=args.smoke,
                    data_root=data_root,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    best_validation_mse=best_validation_mse,
                    processor_image_size=processor_image_size,
                    batch_size=batch_size,
                    world_size=accelerator.num_processes,
                )
                if val_mse is not None:
                    progress["validation_normalized_mse"] = val_mse
                    progress["validation_normalized_mae"] = val_mae
                checkpoint = _save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    processor=processor,
                    output_root=output_root,
                    name=f"step_{global_step:08d}",
                    progress=progress,
                )
                _set_last_checkpoint(accelerator, output_root, checkpoint, progress)
                if is_best:
                    _set_best_checkpoint(accelerator, output_root, checkpoint, progress)
                _prune_periodic_checkpoints(accelerator, output_root, max_periodic_checkpoints)
                if accelerator.is_main_process:
                    logger.info("Saved native checkpoint to %s", checkpoint)

            if global_step >= iters:
                break

        if not epoch_had_batch and skip_batches:
            # The checkpoint was captured at the final batch of this epoch.
            skip_batches = 0
        elif not epoch_had_batch and len(train_loader) == 0:
            raise RuntimeError("Distributed train loader produced no batches")
        else:
            skip_batches = 0
        current_epoch += 1

    if accelerator.is_main_process:
        logger.info("Training finished at step %d in %.1fs", global_step, time.time() - started)
    accelerator.end_training()


if __name__ == "__main__":
    main()
