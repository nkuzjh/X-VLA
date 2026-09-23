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

from csgo_seen10.dataset import (
    COORDINATE_DESCRIPTION,
    SEEN10_ACTION_ADAPTATION,
    SEEN10_EXTERNAL_ACTION_DIM,
    SEEN10_INFERENCE_STEPS,
    SEEN10_NATIVE_ACTION_DIM,
    SEEN10_NUM_ACTIONS,
    SEEN10_NORMALIZATION_DESCRIPTION,
    SEEN10_STATE_DESCRIPTION,
    SEEN_MAPS,
    Seen10Collator,
    Seen10Dataset,
    seen10_data_contract,
)
from csgo_seen10.model import load_seen10_model, load_seen10_processor, processor_size
from csgo_seen10.visualization import build_inference_visualization_rows, render_map_visualizations
from train import set_seed


LOGGER = logging.getLogger("infer_seen10")
_LEGACY_COORDINATE_DESCRIPTION = {
    "pose": "normalized absolute [x, y, z, pitch, yaw]",
    "x_y": "world coordinate / 1024",
    "z": "(world z - published map z_min) / (published map z_max - published map z_min)",
    "pitch_yaw": "radians / (2*pi), with circular period 1",
    "proprio": "all-zero vector with 20 dimensions",
}


def _canonical_action_mode(mode) -> str:
    value = str(mode).lower()
    if value in {"auto", "legacy_reset_dummy"}:
        return "legacy_reset_dummy"
    if value == "official_auto":
        return value
    raise ValueError(f"Unsupported Seen-10 action mode: {mode!r}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/csgo_seen10.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use the isolated, nonformal smoke output root")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", help="Exact output directory for this seed")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Selected native XVLA checkpoint; fair configs default to OUTPUT_ROOT/checkpoints/last, "
            "legacy configs to best"
        ),
    )
    parser.add_argument("--limit-per-map", type=int, help="Allowed only with --smoke")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--inference-seed", type=int)
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    model = config.get("model", {})
    data = config.get("data", {})
    inference = config.get("inference", {})
    mode = model.get("action_mode", model.get("action_contract"))
    if isinstance(mode, dict):
        mode = mode.get("mode")
    canonical_mode = _canonical_action_mode(mode)
    model_contract_mode = model.get("action_contract")
    if isinstance(model_contract_mode, dict):
        model_contract_mode = model_contract_mode.get("mode")
    if model_contract_mode is not None and _canonical_action_mode(model_contract_mode) != canonical_mode:
        raise ValueError("model.action_mode and model.action_contract select different behaviors")
    top_contract = config.get("action_contract")
    if isinstance(top_contract, dict) and _canonical_action_mode(top_contract.get("mode")) != canonical_mode:
        raise ValueError("Top-level action_contract and model.action_mode select different behaviors")
    common_checks = {
        "model.real_action_dim": (model.get("real_action_dim"), SEEN10_EXTERNAL_ACTION_DIM),
        "model.max_action_dim": (model.get("max_action_dim"), SEEN10_NATIVE_ACTION_DIM),
        "model.num_actions": (model.get("num_actions"), SEEN10_NUM_ACTIONS),
        "model.use_proprio": (model.get("use_proprio"), False),
        "model.inference_steps": (model.get("inference_steps"), SEEN10_INFERENCE_STEPS),
        "inference.steps": (inference.get("steps"), SEEN10_INFERENCE_STEPS),
        "data.test_split": (data.get("test_split"), "seen_discrete_test"),
        "task.scope": (config.get("task", {}).get("scope"), "seen10_only"),
        "task.test_split": (config.get("task", {}).get("test_split"), "seen_discrete_test"),
    }
    if canonical_mode == "official_auto":
        normalization = config.get("normalization", {})
        common_checks.update({
            "task.benchmark": (config.get("task", {}).get("benchmark"), "csgo_benchmark_v2"),
            "task.maps": (config.get("task", {}).get("maps"), list(SEEN_MAPS)),
            "model.padding": (model.get("padding"), "pad_before_noise"),
            "model.loss_dimensions": (model.get("loss_dimensions"), SEEN10_EXTERNAL_ACTION_DIM),
            "model.inference_reset": (model.get("inference_reset"), False),
            "inference.reset": (inference.get("reset"), False),
            "inference.reset_dummy": (inference.get("reset_dummy"), False),
            "normalization.epsilon": (normalization.get("epsilon"), None),
            "normalization.clamp": (normalization.get("clamp"), False),
            "normalization.qnorm": (normalization.get("qnorm"), False),
            "normalization.quantile_statistics": (normalization.get("quantile_statistics"), None),
        })
    mismatches = [
        f"{field}={actual!r}, expected={expected!r}"
        for field, (actual, expected) in common_checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Invalid Seen-10 inference config: " + "; ".join(mismatches))
    return config


def _resolve_output_root(args, config: dict) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    train_cfg = config["training"]
    root = train_cfg["smoke_output_root"] if args.smoke else train_cfg["output_root"]
    return (Path(root).expanduser() / f"seed_{args.seed}").resolve()


def _resolve_checkpoint(args, output_root: Path, config: dict | None = None) -> Path:
    """Resolve an explicit checkpoint or the pointer for this config mode.

    The fair/native-width mode defaults to ``last``.  A legacy config that
    declares ``action_mode=auto`` or ``legacy_reset_dummy`` keeps its
    historical ``best`` default; an explicit ``--checkpoint`` always wins.
    Pointer JSON is understood so copied checkpoints do not need to preserve
    symlinks.
    """
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()

    checkpoints_dir = output_root / "checkpoints"
    if config is None:
        # Direct library callers use the current fair default.  The CLI always
        # supplies a config, which lets legacy files retain their best pointer.
        preferred = "last"
    else:
        model_config = config.get("model", {})
        configured_mode = model_config.get("action_mode", model_config.get("action_contract"))
        if isinstance(configured_mode, dict):
            configured_mode = configured_mode.get("mode")
        preferred = "last" if configured_mode == "official_auto" else "best"
    candidate = checkpoints_dir / preferred
    if candidate.is_dir():
        return candidate.resolve()
    pointer_path = checkpoints_dir / f"{preferred}.json"
    if pointer_path.is_file():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            target_name = pointer.get("checkpoint")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid checkpoint pointer {pointer_path}") from error
        if isinstance(target_name, str) and target_name:
            target = checkpoints_dir / target_name
            if target.is_dir():
                return target.resolve()
    raise FileNotFoundError(
        f"No {preferred} checkpoint found under {checkpoints_dir}; pass --checkpoint explicitly "
        "to select another validation-only or immutable step checkpoint"
    )


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _extract_action_contract(source):
    """Extract canonical action-contract fields from current or legacy metadata."""
    if not isinstance(source, dict):
        return {}
    nested = source.get("action_contract")
    if not isinstance(nested, dict):
        nested = source.get("action_space") if isinstance(source.get("action_space"), dict) else None
    candidates = [source]
    if nested is not None:
        candidates.insert(0, nested)
    aliases = {
        "mode": ("mode", "action_mode", "action_space", "name"),
        "real_action_dim": ("real_action_dim", "external_action_dim", "real_dim", "external_dim"),
        "max_action_dim": (
            "max_action_dim",
            "model_action_dim",
            "native_action_dim",
            "max_dim",
            "native_dim",
        ),
        "num_actions": ("num_actions",),
        "use_proprio": ("use_proprio",),
    }
    result = {}
    for canonical, names in aliases.items():
        for candidate in candidates:
            for name in names:
                if name in candidate and not isinstance(candidate[name], dict):
                    value = candidate[name]
                    if canonical == "use_proprio":
                        value = _as_bool(value)
                    result.setdefault(canonical, value)
                    break
            if canonical in result:
                break
    return result


def _validate_checkpoint_action_contract(checkpoint: Path, progress: dict):
    """Validate Seen-10 action metadata while accepting old checkpoints.

    Current checkpoints can declare ``action_contract`` in ``progress.json``
    or ``config.json``.  Older Seen-10 checkpoints did not carry that nested
    object, so their explicit model fields are checked and the fixed contract
    is inferred only when those fields are absent.  A declared mismatch is
    always an error; a foundation checkpoint with its native EE6D contract is
    therefore rejected instead of being silently relabeled as Seen-10.
    """
    sources = [("progress.json", progress)]
    config_path = checkpoint / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid checkpoint model config: {config_path}") from error
        sources.append(("config.json", config))
    action_contract_path = checkpoint / "action_contract.json"
    if action_contract_path.is_file():
        try:
            standalone_contract = json.loads(action_contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid checkpoint action contract: {action_contract_path}") from error
        sources.append(("action_contract.json", standalone_contract))

    observed = {}
    observed_from = {}
    for source_name, source in sources:
        for field, value in _extract_action_contract(source).items():
            if field in observed and observed[field] != value:
                if field == "mode" and _canonical_action_mode(observed[field]) == _canonical_action_mode(value):
                    # ``auto`` is the historical spelling of the explicit
                    # legacy reset-dummy contract.
                    continue
                raise ValueError(
                    f"Checkpoint action contract disagrees between {observed_from[field]} and "
                    f"{source_name}: {field}={observed[field]!r} vs {value!r}"
                )
            observed[field] = value
            observed_from[field] = source_name

    expected = {
        # ``auto``/``legacy_reset_dummy`` are legacy reset-dummy contracts;
        # ``official_auto`` is the native-width Seen-10 contract.  All remain
        # loadable here so old checkpoints remain usable, while dimensions and
        # state are always strict.
        "mode": observed.get("mode", "legacy_reset_dummy"),
        "real_action_dim": SEEN10_EXTERNAL_ACTION_DIM,
        "max_action_dim": SEEN10_NATIVE_ACTION_DIM,
        "num_actions": SEEN10_NUM_ACTIONS,
        "use_proprio": False,
    }
    mismatches = []
    if expected["mode"] not in {"auto", "legacy_reset_dummy", "official_auto"}:
        mismatches.append(
            f"mode: checkpoint={expected['mode']!r}, "
            "expected='auto', 'legacy_reset_dummy', or 'official_auto'"
        )
    mismatches.extend(
        f"{field}: checkpoint={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if field != "mode" and field in observed and observed[field] != value
    )
    if mismatches:
        raise ValueError("Checkpoint action contract mismatch: " + "; ".join(mismatches))

    canonical_mode = _canonical_action_mode(expected["mode"])
    expected_mode_contract = (
        {
            "padding": "pad_before_noise",
            "noise_width": SEEN10_NATIVE_ACTION_DIM,
            "prediction_width": SEEN10_NATIVE_ACTION_DIM,
            "loss": "valid_action_mse",
            "loss_dimensions": SEEN10_EXTERNAL_ACTION_DIM,
            "loss_scale": 100.0,
            "dummy_channels_in_loss": False,
            "inference_reset": False,
            "target_pad_before_noise": True,
            "final_output_dim": SEEN10_EXTERNAL_ACTION_DIM,
            "objective": "x0_clean_action_denoising_regression",
            "inference_steps": SEEN10_INFERENCE_STEPS,
        }
        if canonical_mode == "official_auto"
        else {
            "padding": "legacy_reset_dummy",
            "inference_reset": True,
        }
    )

    expected_normalization = {
        "epsilon": None,
        "clamp": False,
        "qnorm": False,
    }
    expected_state = {
        "use_proprio": False,
        "dim": 20,
        "values": "all zeros",
    }
    declared_contract = {}
    declared_normalization = {}
    declared_state = {}
    for source_name, source in sources:
        declared_steps = []
        if isinstance(source, dict):
            if "inference_steps" in source:
                declared_steps.append(("inference_steps", source["inference_steps"]))
            contract = source.get("action_contract")
            if isinstance(contract, dict) and "inference_steps" in contract:
                declared_steps.append(("action_contract.inference_steps", contract["inference_steps"]))
            inference = source.get("inference")
            if isinstance(inference, dict) and "steps" in inference:
                declared_steps.append(("inference.steps", inference["steps"]))
            contract = source.get("action_contract")
            if not isinstance(contract, dict) and source_name == "action_contract.json":
                contract = source
            if isinstance(contract, dict):
                declared_contract.update(contract)
                mode_contract_mismatches = [
                    f"{key}: checkpoint={contract[key]!r}, expected={value!r}"
                    for key, value in expected_mode_contract.items()
                    if key in contract and contract[key] != value
                ]
                if mode_contract_mismatches:
                    raise ValueError(
                        f"Checkpoint action behavior mismatch in {source_name}: "
                        + "; ".join(mode_contract_mismatches)
                    )
        for field, value in declared_steps:
            if int(value) != SEEN10_INFERENCE_STEPS:
                raise ValueError(
                    f"Checkpoint inference contract mismatch in {source_name}: "
                    f"{field}={value!r}, expected {SEEN10_INFERENCE_STEPS}"
                )
        normalization = source.get("normalization") if isinstance(source, dict) else None
        if isinstance(normalization, dict):
            declared_normalization.update(normalization)
            normalization_mismatches = [
                f"{key}: checkpoint={normalization[key]!r}, expected={value!r}"
                for key, value in expected_normalization.items()
                if key in normalization and normalization[key] != value
            ]
            if normalization_mismatches:
                raise ValueError(
                    f"Checkpoint normalization contract mismatch in {source_name}: "
                    + "; ".join(normalization_mismatches)
                )
        state = source.get("state") if isinstance(source, dict) else None
        if not isinstance(state, dict) and isinstance(source, dict):
            state = source.get("state_contract")
        if isinstance(state, dict):
            declared_state.update(state)
            state_mismatches = [
                f"{key}: checkpoint={state[key]!r}, expected={value!r}"
                for key, value in expected_state.items()
                if key in state and state[key] != value
            ]
            if state_mismatches:
                raise ValueError(
                    f"Checkpoint state contract mismatch in {source_name}: "
                    + "; ".join(state_mismatches)
                )

    if canonical_mode == "official_auto":
        missing_identity = sorted(
            {"mode", "real_action_dim", "max_action_dim", "num_actions", "use_proprio"}
            - observed.keys()
        )
        missing_behavior = sorted(expected_mode_contract.keys() - declared_contract.keys())
        missing_normalization = sorted(expected_normalization.keys() - declared_normalization.keys())
        missing_state = sorted(expected_state.keys() - declared_state.keys())
        if missing_identity or missing_behavior or missing_normalization or missing_state:
            raise ValueError(
                "Official Seen-10 checkpoint metadata is incomplete: "
                f"identity={missing_identity}, behavior={missing_behavior}, "
                f"normalization={missing_normalization}, state={missing_state}"
            )

    legacy_inferred = not observed
    if legacy_inferred:
        LOGGER.warning(
            "Checkpoint %s has no action-contract metadata; inferring the fixed legacy Seen-10 contract",
            checkpoint,
        )
    result = dict(expected)
    result["source"] = "legacy-inferred" if legacy_inferred else ",".join(sorted(set(observed_from.values())))
    return result


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
    action_contract: dict | None = None, data_contract: dict | None = None,
):
    if steps != SEEN10_INFERENCE_STEPS:
        raise ValueError(
            f"Seen-10 native inference requires exactly {SEEN10_INFERENCE_STEPS} denoising steps; got {steps}"
        )
    if progress.get("seed") is not None and progress.get("seed") != args.seed:
        raise ValueError(
            f"Checkpoint seed {progress.get('seed')} does not match requested seed {args.seed}"
        )
    if progress.get("smoke") is not None and progress.get("smoke") != args.smoke:
        raise ValueError("Checkpoint smoke/formal status does not match the requested inference mode")
    if progress.get("data_root") is not None and progress.get("data_root") != str(data_root):
        raise ValueError("Checkpoint and inference use different CSGO data roots")
    if progress.get("coordinates") is not None and progress.get("coordinates") != COORDINATE_DESCRIPTION:
        if progress.get("coordinates") != _LEGACY_COORDINATE_DESCRIPTION:
            raise ValueError("Checkpoint coordinate metadata does not match the CSGO normalized 5DoF contract")
    checkpoint_id = progress.get("checkpoint_id") or f"legacy:{checkpoint.name}"
    checkpoint_step = progress.get("global_step")
    if checkpoint_step is not None:
        checkpoint_step = int(checkpoint_step)
    action_contract = action_contract or {
        "mode": "official_auto",
        "real_action_dim": SEEN10_EXTERNAL_ACTION_DIM,
        "max_action_dim": SEEN10_NATIVE_ACTION_DIM,
        "num_actions": SEEN10_NUM_ACTIONS,
        "use_proprio": False,
    }
    contract_checks = {
        "real_action_dim": SEEN10_EXTERNAL_ACTION_DIM,
        "max_action_dim": SEEN10_NATIVE_ACTION_DIM,
        "num_actions": SEEN10_NUM_ACTIONS,
        "use_proprio": False,
    }
    contract_mismatches = [
        f"{field}: {action_contract.get(field)!r} != {expected!r}"
        for field, expected in contract_checks.items()
        if action_contract.get(field) != expected
    ]
    if action_contract.get("mode") not in {"auto", "legacy_reset_dummy", "official_auto"}:
        contract_mismatches.append(f"mode: {action_contract.get('mode')!r} is unsupported")
    if contract_mismatches:
        raise ValueError("Invalid Seen-10 action contract: " + "; ".join(contract_mismatches))
    canonical_mode = _canonical_action_mode(action_contract["mode"])
    checkpoint_data_contract = progress.get("data_contract")
    if data_contract is not None:
        current_data_id = data_contract.get("contract_sha256")
        if not isinstance(current_data_id, str) or not current_data_id:
            raise ValueError("Current Seen-10 data contract has no contract_sha256")
        if checkpoint_data_contract is None:
            if canonical_mode == "official_auto":
                raise ValueError(
                    "Official Seen-10 checkpoint has no published-data contract; "
                    "refusing unauditable inference"
                )
        elif checkpoint_data_contract.get("contract_sha256") != current_data_id:
            raise ValueError(
                "Checkpoint and inference use different Seen-10 manifest/calibration/split metadata: "
                f"checkpoint={checkpoint_data_contract.get('contract_sha256')!r}, "
                f"current={current_data_id!r}"
            )
    action_adaptation = dict(SEEN10_ACTION_ADAPTATION)
    action_adaptation["mode"] = action_contract["mode"]
    if canonical_mode == "legacy_reset_dummy":
        action_adaptation.update({
            "objective": "x0_clean_action_denoising_regression",
            "dataset_target_width": SEEN10_EXTERNAL_ACTION_DIM,
            "target_padding": "legacy preprocess after 5D noise sampling",
            "training_noise_width": SEEN10_EXTERNAL_ACTION_DIM,
            "training_model_input_width": SEEN10_NATIVE_ACTION_DIM,
            "training_prediction_width": SEEN10_NATIVE_ACTION_DIM,
            "supervised_loss_width": SEEN10_EXTERNAL_ACTION_DIM,
            "loss_scale": 100.0,
            "model_only_channels_directly_supervised": False,
            "inference_state_width": SEEN10_NATIVE_ACTION_DIM,
            "model_only_channels_reset_each_step": True,
            "native_denoising": (
                "20D loop tensors, with model-only channels reset before every model call; "
                "trim to 5D after the final step"
            ),
        })
    # Include short aliases for consumers that use the generic dim names.
    action_adaptation.update({
        "external_dim": SEEN10_EXTERNAL_ACTION_DIM,
        "native_dim": SEEN10_NATIVE_ACTION_DIM,
        "trimmed_output_dim": SEEN10_EXTERNAL_ACTION_DIM,
    })
    state_description = dict(SEEN10_STATE_DESCRIPTION)
    state_description["proprio_dim"] = state_description["dim"]
    normalization = dict(SEEN10_NORMALIZATION_DESCRIPTION)
    normalization.update({
        "target_dim": SEEN10_EXTERNAL_ACTION_DIM,
        "order": "xyzhw = x, y, z, pitch, yaw",
    })
    return {
        "schema": "xvla_csgo_seen10_localization_predictions_v2",
        "seed": args.seed,
        "smoke": args.smoke,
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": checkpoint_step,
        "initialization": progress.get("initialization"),
        "data_root": str(data_root),
        "data_contract": data_contract,
        "split": "seen_discrete_test",
        "limit_per_map": dataset.limit_per_map,
        "expected_samples": len(dataset),
        "batch_size": batch_size,
        "steps": steps,
        "world_size": world_size,
        "inference_seed": inference_seed,
        "coordinates": COORDINATE_DESCRIPTION,
        "processor_image_size": progress.get("processor_image_size"),
        "action_adaptation": action_adaptation,
        "action_contract": {
            "mode": action_contract["mode"],
            "external_action_dim": action_contract["real_action_dim"],
            "native_action_dim": action_contract["max_action_dim"],
            "num_actions": action_contract["num_actions"],
            "use_proprio": action_contract["use_proprio"],
        },
        "state": state_description,
        "normalization": normalization,
        "inference_steps": steps,
        "returned_action_dim": SEEN10_EXTERNAL_ACTION_DIM,
        "inference": {
            "steps": steps,
            "solver": "native_xvla_iterative_clean_action_refinement",
            "objective": "x0_clean_action_denoising_regression",
            "native_action_dim": SEEN10_NATIVE_ACTION_DIM,
            "returned_action_dim": SEEN10_EXTERNAL_ACTION_DIM,
            "native_width_preserved_through_all_steps": canonical_mode == "official_auto",
            "model_only_channels_reset_each_step": canonical_mode == "legacy_reset_dummy",
            "trim_to_external_only_after_final_step": True,
        },
    }


def _write_or_verify_metadata(path: Path, expected: dict, predictions_path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            current = json.load(stream)
        if current != expected:
            # Upgrade a sidecar written by the original Seen-10 adapter after
            # checking every field it did record.  This keeps old completed
            # prediction files usable while refusing to mix checkpoints,
            # seeds, splits, or inference settings.
            legacy_schema = str(current.get("schema", "")).endswith("_v1")
            conflicting = []
            for key, value in current.items():
                if key not in expected or key == "schema" or expected[key] == value:
                    continue
                if (
                    key == "coordinates"
                    and value == _LEGACY_COORDINATE_DESCRIPTION
                    and expected[key] == COORDINATE_DESCRIPTION
                ):
                    continue
                conflicting.append(key)
            if not legacy_schema or conflicting:
                detail = f" conflicting fields={conflicting}" if conflicting else ""
                raise ValueError(
                    f"Existing prediction metadata at {path} belongs to a different seed/checkpoint/split;"
                    f" choose a fresh output-root.{detail}"
                )
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(expected, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
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
    try:
        values = [float(value) for value in prediction]
    except (TypeError, ValueError) as error:
        raise ValueError("Seen-10 prediction must be a finite 5D action") from error
    if len(values) != SEEN10_EXTERNAL_ACTION_DIM:
        raise ValueError(
            f"Seen-10 prediction must have exactly {SEEN10_EXTERNAL_ACTION_DIM} values; got {len(values)}"
        )
    row = {
        "sample_id": str(metadata["sample_id"]),
        "map_name": str(metadata["map_name"]),
        "pred_x": values[0],
        "pred_y": values[1],
        "pred_z": values[2],
        "pred_pitch": values[3],
        "pred_yaw": values[4],
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
    checkpoint = _resolve_checkpoint(args, output_root, config)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")
    with (checkpoint / "progress.json").open(encoding="utf-8") as stream:
        progress = json.load(stream)
    checkpoint_action_contract = _validate_checkpoint_action_contract(checkpoint, progress)
    requested_mode = model_cfg.get("action_mode", model_cfg.get("action_contract", "official_auto"))
    if isinstance(requested_mode, dict):
        requested_mode = requested_mode.get("mode")
    if _canonical_action_mode(requested_mode) != _canonical_action_mode(checkpoint_action_contract["mode"]):
        raise ValueError(
            "Selected checkpoint action mode does not match the requested config: "
            f"checkpoint={checkpoint_action_contract['mode']!r}, config={requested_mode!r}"
        )

    limit_per_map = args.limit_per_map
    if args.smoke and limit_per_map is None:
        limit_per_map = config["training"].get("smoke_limit_per_map", 8)
    batch_size = args.batch_size or (inference_cfg["smoke_batch_size"] if args.smoke else inference_cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else inference_cfg.get("num_workers", data_cfg.get("num_workers", 0))
    steps = args.steps if args.steps is not None else inference_cfg.get(
        "steps", model_cfg.get("inference_steps", SEEN10_INFERENCE_STEPS)
    )
    inference_seed = (
        args.inference_seed if args.inference_seed is not None else inference_cfg.get("seed", 42)
    )
    if min(batch_size, steps) <= 0 or num_workers < 0:
        raise ValueError("batch-size and steps must be positive; num-workers must be nonnegative")

    accelerator = Accelerator()
    data_contract = seen10_data_contract(data_root)
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
                action_contract=checkpoint_action_contract,
                data_contract=data_contract,
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
    actual_processor_size = processor_size(processor)
    expected_processor_size = int(model_cfg.get("processor_image_size_expected", 224))
    required_processor_size = {
        "height": expected_processor_size,
        "width": expected_processor_size,
    }
    if actual_processor_size != required_processor_size:
        raise ValueError(
            "Checkpoint processor does not preserve the configured native image resolution: "
            f"actual={actual_processor_size}, expected={required_processor_size}"
        )
    recorded_processor_size = progress.get("processor_image_size")
    if recorded_processor_size is not None and recorded_processor_size != actual_processor_size:
        raise ValueError(
            "Checkpoint processor differs from its training metadata: "
            f"actual={actual_processor_size}, recorded={recorded_processor_size}"
        )
    model = load_seen10_model(
        str(checkpoint),
        action_contract={"mode": checkpoint_action_contract["mode"]},
    ).to(accelerator.device)
    action_space = getattr(model, "action_space", None)
    if (
        getattr(model, "num_actions", None) != SEEN10_NUM_ACTIONS
        or getattr(action_space, "real_dim", None) != SEEN10_EXTERNAL_ACTION_DIM
        or getattr(action_space, "dim_action", None) != SEEN10_NATIVE_ACTION_DIM
    ):
        raise RuntimeError(
            "Loaded checkpoint/model does not preserve the Seen-10 action contract "
            f"(native {SEEN10_NATIVE_ACTION_DIM}D, returned {SEEN10_EXTERNAL_ACTION_DIM}D, "
            f"horizon {SEEN10_NUM_ACTIONS})"
        )
    if getattr(model, "use_proprio", None):
        raise RuntimeError(
            "Seen-10 inference requires use_proprio=False with only the structural zero proprio20 tensor"
        )
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
            if tuple(predictions.shape[1:]) != (SEEN10_NUM_ACTIONS, SEEN10_EXTERNAL_ACTION_DIM):
                raise ValueError(
                    "Expected final Seen-10 predictions [B,1,5] after native 20D denoising, "
                    f"got {tuple(predictions.shape)}"
                )

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
