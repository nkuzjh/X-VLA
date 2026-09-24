"""Minimal Accelerate wrapper for horizon-one CSGO Seen-10 localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import inspect
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from csgo_seen10.dataset import (
    COORDINATE_DESCRIPTION,
    SEEN_MAPS,
    Seen10Collator,
    Seen10Dataset,
    seen10_data_contract,
)
from csgo_seen10.model import load_seen10_model, load_seen10_processor, processor_size
try:
    from csgo_seen10.model import pad_seen10_action as _core_pad_seen10_action
except ImportError:
    _core_pad_seen10_action = None
from csgo_seen10.visualization import (
    gather_visualization_rows,
    render_map_visualizations,
    select_visualization_identities,
)
from train import build_optimizer, get_logger, set_seed, update_group_lrs


_FAIR_POLICY = "fair_v1"
_LEGACY_POLICY = "legacy_reproducible"
_ACTION_MODES = {"legacy_reset_dummy", "official_auto", "auto"}
_FAIR_LORA_TARGET_REGEX = (
    r"(?:vlm\.language_model\.model\.encoder\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)|"
    r"transformer\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2)))"
)
_FAIR_MODULES_TO_SAVE = (
    "vlm.image_proj_norm",
    "transformer.vlm_proj",
    "transformer.aux_visual_proj",
    "transformer.action_encoder",
    "transformer.action_decoder",
    "transformer.norm",
    "transformer.soft_prompt_hub",
)
_FAIR_EXTRA_TRAINABLE_PARAMETERS = ("vlm.image_projection",)


def _json_copy(value):
    """Copy config-shaped values while rejecting runtime-only objects."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _action_contract(config: dict) -> dict:
    """Return one explicit, serializable action contract for config/metadata.

    ``models.action_hub`` owns the numerical implementation.  The training
    entry point only records and forwards this contract, so a checkpoint can
    be rejected when it is later opened with another action adaptation.
    The fair model helper accepts ``action_contract`` (or its equivalent
    keyword) and must implement ``pad_before_noise`` before ``forward``.
    """
    model_cfg = config.get("model", {})
    configured = config.get("action_contract", {})
    if isinstance(configured, str):
        configured = {"mode": configured}
    contract = dict(configured)
    mode = str(contract.get("mode", model_cfg.get("action_contract", model_cfg.get("action_mode", "auto"))))
    contract.update({
        "mode": mode,
        "external_action_dim": int(contract.get("external_action_dim", model_cfg.get("real_action_dim", 5))),
        "model_action_dim": int(contract.get("model_action_dim", model_cfg.get("max_action_dim", 20))),
        "num_actions": int(contract.get("num_actions", model_cfg.get("num_actions", 1))),
        "use_proprio": bool(contract.get("use_proprio", model_cfg.get("use_proprio", False))),
    })
    contract["real_action_dim"] = contract["external_action_dim"]
    contract["max_action_dim"] = contract["model_action_dim"]
    if mode == "official_auto":
        contract.setdefault("padding", model_cfg.get("padding", "pad_before_noise"))
        contract.setdefault("noise_width", contract["model_action_dim"])
        contract.setdefault("loss", model_cfg.get("loss", "valid_action_mse"))
        contract.setdefault("loss_dimensions", int(model_cfg.get("loss_dimensions", 5)))
        contract.setdefault("dummy_channels_in_loss", bool(model_cfg.get("dummy_channels_in_loss", False)))
        contract.setdefault("inference_reset", bool(model_cfg.get("inference_reset", False)))
        contract.setdefault("inference_steps", int(model_cfg.get("inference_steps", 10)))
    else:
        contract.setdefault("padding", "legacy_reset_dummy")
        contract.setdefault("inference_reset", True)
    return _json_copy(contract)


def _validate_config(config: dict) -> dict:
    """Validate invariants that materially affect the experiment contract."""
    model_cfg = config.get("model")
    train_cfg = config.get("training")
    if not isinstance(model_cfg, dict) or not isinstance(train_cfg, dict):
        raise ValueError("Seen-10 config must contain model and training objects")
    contract = _action_contract(config)
    if contract["mode"] not in _ACTION_MODES:
        raise ValueError(f"Unsupported Seen-10 action contract: {contract['mode']}")
    if contract["external_action_dim"] != 5 or contract["model_action_dim"] != 20:
        raise ValueError("Seen-10 requires external 5D and native 20D action widths")
    if contract["num_actions"] != 1:
        raise ValueError("Seen-10 requires horizon-one actions")

    # The fair config intentionally repeats a few high-risk dimensions and
    # action semantics near their consumers.  Treat those copies as assertions
    # instead of allowing one field to silently override another.
    duplicate_checks = {
        "model.action_mode": (model_cfg.get("action_mode"), contract["mode"]),
        "model.action_contract": (model_cfg.get("action_contract"), contract["mode"]),
        "model.real_action_dim": (model_cfg.get("real_action_dim"), contract["external_action_dim"]),
        "model.max_action_dim": (model_cfg.get("max_action_dim"), contract["model_action_dim"]),
        "model.num_actions": (model_cfg.get("num_actions"), contract["num_actions"]),
        "model.use_proprio": (model_cfg.get("use_proprio"), contract["use_proprio"]),
    }
    duplicate_mismatches = [
        f"{field}={actual!r}, expected={expected!r}"
        for field, (actual, expected) in duplicate_checks.items()
        if actual is not None and actual != expected
    ]
    if duplicate_mismatches:
        raise ValueError("Seen-10 duplicated model/action fields disagree: " + "; ".join(duplicate_mismatches))

    policy = str(train_cfg.get("training_policy", _LEGACY_POLICY))
    if policy == _FAIR_POLICY:
        frozen_vl_connector = train_cfg.get("freeze_vision_language_connector", False)
        if not isinstance(frozen_vl_connector, bool):
            raise ValueError("training.freeze_vision_language_connector must be a boolean")
        expected = {
            "mode": "official_auto",
            "padding": "pad_before_noise",
            "noise_width": 20,
            "prediction_width": 20,
            "loss": "valid_action_mse",
            "loss_dimensions": 5,
            "loss_scale": 100.0,
            "dummy_channels_in_loss": False,
            "inference_reset": False,
            "inference_steps": 10,
            "target_pad_before_noise": True,
            "final_output_dim": 5,
            "objective": "x0_clean_action_denoising_regression",
            "iters": 19500,
            "effective_batch_size": 128,
            "eval_interval": 4000 if frozen_vl_connector else 3900,
            "save_interval": 4000 if frozen_vl_connector else 3900,
            "mixed_precision": "bf16",
        }
        actual = {
            "mode": contract["mode"],
            "padding": contract.get("padding"),
            "noise_width": contract.get("noise_width"),
            "prediction_width": contract.get("prediction_width"),
            "loss": contract.get("loss"),
            "loss_dimensions": contract.get("loss_dimensions"),
            "loss_scale": contract.get("loss_scale"),
            "dummy_channels_in_loss": contract.get("dummy_channels_in_loss"),
            "inference_reset": contract.get("inference_reset"),
            "inference_steps": contract.get("inference_steps"),
            "target_pad_before_noise": contract.get("target_pad_before_noise"),
            "final_output_dim": contract.get("final_output_dim"),
            "objective": contract.get("objective"),
            "iters": int(train_cfg.get("iters", 0)),
            "effective_batch_size": int(train_cfg.get("effective_batch_size", 0)),
            "eval_interval": int(train_cfg.get("eval_interval", 0)),
            "save_interval": int(train_cfg.get("save_interval", 0)),
            "mixed_precision": train_cfg.get("mixed_precision"),
        }
        mismatches = [f"{key}={actual[key]!r}, expected={value!r}" for key, value in expected.items() if actual[key] != value]
        if mismatches:
            raise ValueError("Fair Seen-10 config violates the fixed experiment contract: " + "; ".join(mismatches))
        if bool(train_cfg.get("use_cosine_decay", False)) or int(train_cfg.get("warmup_steps", 0)) != 0:
            raise ValueError("Fair Seen-10 uses constant LR: cosine decay and warmup must be disabled")
        lora = train_cfg.get("lora", {})
        expected_lora = {
            "enabled": True,
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": _FAIR_LORA_TARGET_REGEX,
            "modules_to_save": [
                name for name in _FAIR_MODULES_TO_SAVE
                if not (frozen_vl_connector and name == "vlm.image_proj_norm")
            ],
            "extra_trainable_parameters": (
                [] if frozen_vl_connector else list(_FAIR_EXTRA_TRAINABLE_PARAMETERS)
            ),
        }
        lora_mismatches = [
            f"lora.{key}={lora.get(key)!r}, expected={value!r}"
            for key, value in expected_lora.items()
            if lora.get(key) != value
        ]
        if lora_mismatches:
            raise ValueError(
                "Fair Seen-10 LoRA config violates the fixed experiment contract: "
                + "; ".join(lora_mismatches)
            )
        fair_checks = {
            "model.model_action_dim": (model_cfg.get("model_action_dim"), 20),
            "model.padding": (model_cfg.get("padding"), "pad_before_noise"),
            "model.noise_action_dim": (model_cfg.get("noise_action_dim"), 20),
            "model.loss": (model_cfg.get("loss"), "valid_action_mse"),
            "model.loss_dimensions": (model_cfg.get("loss_dimensions"), 5),
            "model.dummy_channels_in_loss": (model_cfg.get("dummy_channels_in_loss"), False),
            "model.dummy_loss_weight": (model_cfg.get("dummy_loss_weight"), 0.0),
            "model.target_pad_before_noise": (model_cfg.get("target_pad_before_noise"), True),
            "model.loss_action_dim": (model_cfg.get("loss_action_dim"), 5),
            "model.loss_scale": (model_cfg.get("loss_scale"), 100.0),
            "model.inference_reset": (model_cfg.get("inference_reset"), False),
            "model.reset_dummy_for_inference": (model_cfg.get("reset_dummy_for_inference"), False),
            "model.objective": (model_cfg.get("objective"), "x0_clean_action_denoising_regression"),
            "training.learning_rate": (train_cfg.get("learning_rate"), 1e-4),
            "training.lr_schedule": (train_cfg.get("lr_schedule"), "constant"),
            "training.weight_decay": (train_cfg.get("weight_decay"), 0.0),
            "training.betas": (train_cfg.get("betas"), [0.9, 0.95]),
            "training.max_grad_norm": (train_cfg.get("max_grad_norm"), 1.0),
            "training.freeze_steps": (train_cfg.get("freeze_steps"), 1000),
            "training.max_periodic_checkpoints": (train_cfg.get("max_periodic_checkpoints"), 5),
            "training.localization_samples_per_update": (train_cfg.get("localization_samples_per_update"), 128),
            "training.localization_sample_exposures": (train_cfg.get("localization_sample_exposures"), 2_496_000),
            "training.updates_per_epoch": (train_cfg.get("updates_per_epoch"), 390),
            "training.epochs": (train_cfg.get("epochs"), 50),
            "training.dropped_tail_samples_per_epoch": (train_cfg.get("dropped_tail_samples_per_epoch"), 80),
        }
        normalization = config.get("normalization", {})
        state = config.get("state", {})
        augmentation = train_cfg.get("augmentation", {})
        fair_checks.update({
            "normalization.epsilon": (normalization.get("epsilon"), None),
            "normalization.clamp": (normalization.get("clamp"), False),
            "normalization.qnorm": (normalization.get("qnorm"), False),
            "normalization.quantile_statistics": (normalization.get("quantile_statistics"), None),
            "state.robot_state_information": (state.get("robot_state_information"), "absent"),
            "state.proprio_tensor": (state.get("proprio_tensor"), "zeros[20]"),
            "state.proprio_dim": (state.get("proprio_dim"), 20),
            "state.use_real_robot_state": (state.get("use_real_robot_state"), False),
            "state.use_proprio": (state.get("use_proprio"), False),
            "state.dim": (state.get("dim"), 20),
            "state.values": (state.get("values"), "all zeros"),
            "augmentation.enabled": (augmentation.get("enabled"), True),
            "augmentation.training_only": (augmentation.get("training_only"), True),
            "augmentation.name": (augmentation.get("name"), "native_color_jitter"),
            "inference.steps": (config.get("inference", {}).get("steps"), 10),
            "inference.reset": (config.get("inference", {}).get("reset"), False),
            "inference.reset_dummy": (config.get("inference", {}).get("reset_dummy"), False),
        })
        fair_mismatches = [
            f"{field}={actual!r}, expected={expected!r}"
            for field, (actual, expected) in fair_checks.items()
            if actual != expected
        ]
        if fair_mismatches:
            raise ValueError("Fair Seen-10 config fields disagree with the executable contract: " + "; ".join(fair_mismatches))
        task = config.get("task", {})
        data = config.get("data", {})
        split_checks = {
            "task.benchmark": (task.get("benchmark"), "csgo_benchmark_v2"),
            "task.maps": (task.get("maps"), list(SEEN_MAPS)),
            "task.scope": (task.get("scope"), "seen10_only"),
            "task.train_split": (task.get("train_split"), "seen_train"),
            "task.validation_split": (task.get("validation_split"), "seen_validation"),
            "task.test_split": (task.get("test_split"), "seen_discrete_test"),
            "data.train_split": (data.get("train_split"), "seen_train"),
            "data.validation_split": (data.get("validation_split"), "seen_validation"),
            "data.test_split": (data.get("test_split"), "seen_discrete_test"),
        }
        split_mismatches = [
            f"{field}={actual!r}, expected={expected!r}"
            for field, (actual, expected) in split_checks.items()
            if actual != expected
        ]
        if split_mismatches:
            raise ValueError("Fair Seen-10 split contract mismatch: " + "; ".join(split_mismatches))
        expected_trainable_phases = {
            "requires_grad": [
                "llm_lora",
                "action_expert_lora",
                "soft_prompt",
                "action_heads",
                "action_connector",
            ],
            "steps_0_999_lr_1e-4": ["soft_prompt", "action_heads"],
            "steps_0_999_lr_0": [
                "llm_lora",
                "action_expert_lora",
                "action_connector",
            ],
            "steps_1000_plus_lr_1e-4": "all_requires_grad",
        }
        if not frozen_vl_connector:
            expected_trainable_phases["requires_grad"].insert(4, "vision_language_connector")
            expected_trainable_phases["steps_0_999_lr_0"].insert(2, "vision_language_connector")
        if train_cfg.get("trainable_phases") != expected_trainable_phases:
            raise ValueError("Fair Seen-10 trainable_phases does not match the executable LR/parameter policy")
        expected_checkpoint_policy = {
            "primary": "last",
            "late": "last",
            "best": "validation_only",
            "periodic_keep": 5,
        }
        if train_cfg.get("checkpoint_policy") != expected_checkpoint_policy:
            raise ValueError("Fair Seen-10 checkpoint_policy must select last and retain five periodic checkpoints")
        if train_cfg.get("checkpoint_selection") != {"main": "late/last", "best": "validation_only"}:
            raise ValueError("Fair Seen-10 checkpoint_selection must keep test-independent last/best rules")
    elif policy == _LEGACY_POLICY:
        if contract["mode"] != "legacy_reset_dummy":
            raise ValueError("Legacy Seen-10 config must explicitly select legacy_reset_dummy")
    else:
        raise ValueError(f"Unknown Seen-10 training_policy: {policy}")
    return contract


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(source: str | Path, *, model=None, config: dict | None = None) -> dict:
    """Describe the initialization source with content-addressable evidence."""
    configured = str(source)
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    files = []
    if source_path.is_file():
        candidates = [source_path]
    elif source_path.is_dir():
        candidates = sorted(path for path in source_path.rglob("*") if path.is_file() and "accelerator_state" not in path.parts)
    else:
        candidates = []
    for path in candidates:
        relative = str(path.relative_to(source_path)) if source_path.is_dir() else path.name
        try:
            files.append({
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
        except OSError:
            continue
    descriptor = {
        "configured": configured,
        "resolved": str(source_path),
        "exists": source_path.exists(),
        "files": files,
    }
    if model is not None and getattr(model, "config", None) is not None:
        try:
            config_json = json.dumps(model.config.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
            descriptor["loaded_model_config_sha256"] = hashlib.sha256(config_json).hexdigest()
        except Exception:
            descriptor["loaded_model_config_sha256"] = None
    descriptor["source_id"] = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if config is not None:
        descriptor["training_config_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    return descriptor


def _verify_fair_initialization(initialization: dict, model_cfg: dict, policy: str):
    """Require the pinned original public base for the formal fair run."""
    if policy != _FAIR_POLICY:
        return
    expected = model_cfg.get("pretrained_weights_sha256")
    weights = [entry for entry in initialization.get("files", ()) if entry.get("path") == "model.safetensors"]
    if not expected or len(weights) != 1:
        raise ValueError(
            "Fair Seen-10 initialization must be a local original X-VLA base with one model.safetensors"
        )
    if weights[0].get("sha256") != expected:
        raise ValueError(
            "Fair Seen-10 base checkpoint hash mismatch: "
            f"actual={weights[0].get('sha256')}, expected={expected}"
        )
    initialization.update({
        "repo_id": model_cfg.get("pretrained_repo_id"),
        "revision": model_cfg.get("pretrained_revision"),
        "verified_weights_sha256": expected,
    })


def _make_seen10_dataset(data_root, split, *, include_targets, limit_per_map, augmentation_enabled):
    """Instantiate old or new dataset adapters without changing their API."""
    kwargs = {
        "include_targets": include_targets,
        "limit_per_map": limit_per_map,
    }
    parameters = inspect.signature(Seen10Dataset).parameters
    for name in ("augmentation", "augment", "training_augmentation", "enable_augmentation"):
        if name in parameters:
            kwargs[name] = bool(augmentation_enabled)
            break
    dataset = Seen10Dataset(data_root, split, **kwargs)
    # Older adapters derive augmentation solely from split.  Explicitly
    # disable it for the legacy reproducibility policy when no switch exists.
    if hasattr(dataset, "training") and not augmentation_enabled:
        dataset.training = False
    return dataset


def _make_collator(processor, *, include_targets: bool, training: bool):
    kwargs = {"include_targets": include_targets}
    parameters = inspect.signature(Seen10Collator).parameters
    if "training" in parameters:
        kwargs["training"] = training
    elif "is_training" in parameters:
        kwargs["is_training"] = training
    return Seen10Collator(processor, **kwargs)


def _resolve_gradient_accumulation(train_cfg: dict, *, batch_size: int, world_size: int, smoke: bool) -> dict:
    """Compute and verify exact global samples per optimizer update.

    The fair config records the single-process reference value (4*32).  At
    runtime the actual accumulation is derived from world size.  Any
    non-divisible arrangement fails loudly instead of silently changing the
    effective batch.
    """
    legacy_policy = str(train_cfg.get("training_policy", _LEGACY_POLICY)) == _LEGACY_POLICY
    configured = train_cfg.get("gradient_accumulation_steps")
    reference_world_size = int(train_cfg.get("gradient_accumulation_reference_world_size", 1))
    if not smoke and legacy_policy:
        # The historical entry point used one local batch per optimizer
        # update.  Preserve that behavior, including the natural world-size
        # scaling of a distributed run, instead of imposing fair-policy
        # global-batch arithmetic on old experiments.
        accumulation = int(configured if configured is not None else 1)
        if accumulation <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        target = batch_size * world_size * accumulation
        configured_target = train_cfg.get("effective_batch_size")
        if configured_target is not None and world_size == reference_world_size and int(configured_target) != target:
            raise ValueError(
                "legacy effective_batch_size disagrees with batch_size*gradient_accumulation_steps: "
                f"{configured_target} != {target}"
            )
        return {
            "micro_batch_size": batch_size,
            "world_size": world_size,
            "effective_batch_size": target,
            "gradient_accumulation_steps": accumulation,
            "configured_gradient_accumulation_steps": accumulation,
            "reference_world_size": reference_world_size,
        }
    if smoke:
        target = int(train_cfg.get("smoke_effective_batch_size", batch_size * world_size))
    else:
        target = int(train_cfg.get("effective_batch_size", batch_size * world_size))
    if batch_size <= 0 or world_size <= 0 or target <= 0:
        raise ValueError("batch_size, world_size, and effective_batch_size must be positive")
    per_micro_global = batch_size * world_size
    if target % per_micro_global:
        raise ValueError(
            "effective_batch_size must be divisible by batch_size*world_size "
            f"({target} % {per_micro_global} != 0)"
        )
    accumulation = target // per_micro_global
    if not smoke and configured is not None:
        expected_reference = target // (batch_size * reference_world_size)
        if target % (batch_size * reference_world_size) or int(configured) != expected_reference:
            raise ValueError(
                "gradient_accumulation_steps does not describe the configured effective batch: "
                f"configured={configured}, expected={expected_reference}"
            )
    return {
        "micro_batch_size": batch_size,
        "world_size": world_size,
        "effective_batch_size": target,
        "gradient_accumulation_steps": accumulation,
        "configured_gradient_accumulation_steps": None if configured is None else int(configured),
        "reference_world_size": reference_world_size,
    }


def _parameter_role(name: str) -> str:
    lower = name.lower()
    if ".original_module." in lower:
        return "frozen_base"
    if "lora_" in lower or ".lora" in lower:
        if "vlm.language_model.model.encoder.layers" in lower:
            return "llm_lora"
        if "transformer.blocks" in lower:
            return "action_expert_lora"
        return "unexpected_lora"
    if "soft_prompt" in lower or "soft_prompts" in lower:
        return "soft_prompt"
    if (
        "action_encoder" in lower
        or "action_decoder" in lower
        or "transformer.norm" in lower
        or "action_heads" in lower
    ):
        return "action_heads"
    if "transformer.vlm_proj" in lower or "transformer.aux_visual_proj" in lower:
        return "action_connector"
    if "vlm.image_projection" in lower or "vlm.image_proj_norm" in lower:
        return "vision_language_connector"
    if "vlm.vision_tower" in lower or "vlm.image_pos_embed" in lower:
        return "vision_encoder"
    if "vlm.language_model" in lower:
        return "llm_base"
    if "transformer.blocks" in lower:
        return "action_expert_base"
    if "transformer.pos_emb" in lower:
        return "action_position_embedding"
    return "other_base"


_FAIR_TRAINABLE_ROLES = {
    "llm_lora",
    "action_expert_lora",
    "soft_prompt",
    "action_heads",
    "action_connector",
    "vision_language_connector",
}


def _fair_trainable_roles(train_cfg: dict) -> set[str]:
    roles = set(_FAIR_TRAINABLE_ROLES)
    if train_cfg.get("freeze_vision_language_connector", False):
        roles.remove("vision_language_connector")
    return roles


def _configure_trainable_parameters(model, train_cfg: dict, policy: str) -> dict:
    """Set one stable requires-grad mask before DDP is constructed.

    The first 1,000-update freeze is implemented with zero learning rates.
    Changing ``requires_grad`` after DDP construction would leave newly enabled
    parameters without distributed gradient hooks, so the fair trainable set
    stays fixed for the whole run.
    """
    fair_roles = _fair_trainable_roles(train_cfg)
    by_role = {}
    total_elements = 0
    trainable_elements = 0
    trainable_names = []
    for name, parameter in model.named_parameters():
        role = _parameter_role(name)
        if policy == _LEGACY_POLICY:
            enabled = True
        else:
            enabled = role in fair_roles and ".original_module." not in name
        parameter.requires_grad = enabled
        elements = int(parameter.numel())
        total_elements += elements
        entry = by_role.setdefault(role, {"total": 0, "trainable": 0, "tensors": 0, "trainable_tensors": 0})
        entry["total"] += elements
        entry["tensors"] += 1
        if enabled:
            trainable_elements += elements
            entry["trainable"] += elements
            entry["trainable_tensors"] += 1
            trainable_names.append(name)

    if not trainable_names:
        raise ValueError("Seen-10 training policy leaves no trainable parameters")
    if policy == _FAIR_POLICY:
        missing = [
            role for role in sorted(fair_roles)
            if by_role.get(role, {}).get("trainable", 0) == 0
        ]
        unexpected = [
            name for name in trainable_names
            if _parameter_role(name) not in fair_roles
        ]
        vision_trainable = by_role.get("vision_encoder", {}).get("trainable", 0)
        frozen_connector_trainable = (
            by_role.get("vision_language_connector", {}).get("trainable", 0)
            if train_cfg.get("freeze_vision_language_connector", False) else 0
        )
        if missing or unexpected or vision_trainable or frozen_connector_trainable:
            raise RuntimeError(
                "Invalid fair trainable-parameter map: "
                f"missing_roles={missing}, unexpected={unexpected[:8]}, "
                f"vision_trainable={vision_trainable}, "
                f"frozen_connector_trainable={frozen_connector_trainable}"
            )
    return {
        "policy": policy,
        "total_parameters": total_elements,
        "trainable_parameters": trainable_elements,
        "trainable_ratio": trainable_elements / max(1, total_elements),
        "by_role": by_role,
        "trainable_parameter_names": trainable_names,
    }


def _phase_metadata(train_cfg: dict, global_step: int, policy: str) -> dict:
    freeze_steps = int(train_cfg.get("freeze_steps", 0))
    if policy == _LEGACY_POLICY:
        if global_step < freeze_steps:
            active = ["soft_prompts", "action_heads"]
            phase = "legacy_freeze_lr"
        else:
            active = ["vlm", "transformer_core", "soft_prompts", "action_heads"]
            phase = "legacy_native_schedule"
    elif global_step < freeze_steps:
        active = ["soft_prompt", "action_heads"]
        phase = "freeze_lr"
    else:
        active = sorted(_fair_trainable_roles(train_cfg))
        phase = "all_fair_groups"
    return {
        "step": int(global_step),
        "phase": phase,
        "active_lr_roles": active,
        "requires_grad_is_static": True,
    }


def _keep_frozen_vision_eval(model, policy: str, train_cfg: dict | None = None):
    """Keep frozen backbone dropout disabled while preserving trainable dropout.

    ``model.train()`` is called at every batch, which recursively switches the
    frozen Florence language/vision modules and the action-transformer base
    blocks back to training mode.  Their LoRA adapters are trainable and must
    retain their configured dropout.  In the frozen-connector variant the
    vision-language projection modules also stay in eval mode.  Restore every
    PEFT LoRA-dropout module to train after switching frozen modules to eval.
    """
    if policy != _FAIR_POLICY:
        return
    root = getattr(model, "get_base_model", lambda: model)()
    vision_tower = getattr(getattr(root, "vlm", None), "vision_tower", None)
    if vision_tower is None:
        raise RuntimeError("Fair Seen-10 model has no vision tower to freeze")
    vision_tower.eval()
    if train_cfg is not None and train_cfg.get("freeze_vision_language_connector", False):
        vlm = root.vlm
        for name in ("image_projection", "image_proj_norm"):
            connector = getattr(vlm, name, None)
            if isinstance(connector, torch.nn.Module):
                connector.eval()
    language_model = getattr(getattr(root, "vlm", None), "language_model", None)
    if language_model is not None:
        language_model.eval()
    action_blocks = getattr(getattr(root, "transformer", None), "blocks", None)
    if action_blocks is not None:
        action_blocks.eval()

    # PEFT stores the configured adapter dropout in modules whose names contain
    # ``lora_dropout``.  Re-enable only those modules after the frozen parents
    # have been switched to eval mode.
    for name, module in root.named_modules():
        if "lora_dropout" in name.lower():
            module.train()


def _fair_train_loader_seed(seed: int, rank: int, epoch: int) -> int:
    """Return a stable, distinct worker-seed root for one fair epoch/rank."""
    modulus = (1 << 63) - 1
    return (
        int(seed)
        + 1_000_003 * int(epoch)
        + 10_007 * int(rank)
        + 1_000_000_007
    ) % modulus


def _resolve_train_loader_budget(
    num_samples: int,
    *,
    batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    policy: str,
) -> dict:
    """Resolve epoch lengths from the actual sampler/loader drop policies.

    Fair runs use ``drop_last=True`` at both levels, so only complete global
    updates are exposed.  Legacy runs preserve the original DistributedSampler
    and DataLoader ``drop_last=False`` behavior, including the sampler's
    possible padding and a final partial loader batch when accumulation is one.
    """
    if min(num_samples, batch_size, world_size, gradient_accumulation_steps) <= 0:
        raise ValueError("train loader budget inputs must be positive")
    if policy == _FAIR_POLICY:
        samples_per_rank = num_samples // world_size
        loader_micro_batches = samples_per_rank // batch_size
    elif policy == _LEGACY_POLICY:
        samples_per_rank = math.ceil(num_samples / world_size)
        loader_micro_batches = math.ceil(samples_per_rank / batch_size)
    else:
        raise ValueError(f"Unsupported Seen-10 training policy: {policy}")
    updates_per_epoch = loader_micro_batches // gradient_accumulation_steps
    if updates_per_epoch <= 0:
        raise ValueError(
            "The selected train split is smaller than one complete optimizer update: "
            f"loader_micro_batches={loader_micro_batches}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}"
        )
    micro_batches_per_epoch = updates_per_epoch * gradient_accumulation_steps
    consumed_samples = micro_batches_per_epoch * batch_size * world_size
    return {
        "sampler_samples_per_rank": samples_per_rank,
        "loader_micro_batches_per_epoch": loader_micro_batches,
        "updates_per_epoch": updates_per_epoch,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "consumed_samples_per_epoch": consumed_samples,
        "dropped_samples_per_epoch": max(0, num_samples - consumed_samples),
        "padded_samples_per_epoch": max(0, consumed_samples - num_samples),
    }


def _canonical_resume_action_contract(contract: dict | None) -> dict | None:
    """Normalize the pre-adapter ``auto`` alias for legacy checkpoint resume."""
    if contract is None:
        return None
    normalized = _json_copy(contract)
    if normalized.get("mode") in {"auto", "legacy_reset_dummy"}:
        normalized["mode"] = "legacy_reset_dummy"
    return normalized


def _maybe_attach_lora(model, train_cfg: dict):
    """Attach the configured official PEFT adapter exactly once."""
    lora_cfg = train_cfg.get("lora", {})
    if not bool(lora_cfg.get("enabled", False)):
        return model, False
    if hasattr(model, "peft_config"):
        return model, True
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("The fair Seen-10 policy requires the peft package for LoRA") from error
    peft_config = LoraConfig(
        lora_alpha=int(lora_cfg["alpha"]),
        r=int(lora_cfg["r"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=lora_cfg["target_modules"],
        modules_to_save=list(lora_cfg["modules_to_save"]),
    )
    return get_peft_model(model, peft_config), True


def _build_seen10_optimizer(model, train_cfg: dict, policy: str):
    if policy == _LEGACY_POLICY:
        return build_optimizer(
            model=model,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            betas=tuple(train_cfg["betas"]),
            lr_coef_soft=train_cfg["learning_coef"],
        )

    grouped = {role: [] for role in sorted(_fair_trainable_roles(train_cfg))}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = _parameter_role(name)
        if role not in grouped:
            raise RuntimeError(f"Trainable parameter has no fair optimizer role: {name}")
        grouped[role].append(parameter)
    base_lr = float(train_cfg["learning_rate"])
    phase_one = {"soft_prompt", "action_heads"}
    groups = [
        {
            "name": role,
            "params": parameters,
            "lr": base_lr if role in phase_one else 0.0,
            "weight_decay": float(train_cfg["weight_decay"]),
        }
        for role, parameters in grouped.items()
        if parameters
    ]
    return AdamW(groups, betas=tuple(train_cfg["betas"]))


def _update_seen10_lrs(optimizer, global_step: int, train_cfg: dict, policy: str, legacy_schedule_args):
    if policy == _LEGACY_POLICY:
        update_group_lrs(optimizer, global_step, legacy_schedule_args)
        return
    base_lr = float(train_cfg["learning_rate"])
    freeze_steps = int(train_cfg["freeze_steps"])
    phase_one = {"soft_prompt", "action_heads"}
    for group in optimizer.param_groups:
        group["lr"] = base_lr if global_step >= freeze_steps or group["name"] in phase_one else 0.0


def _call_model_loader(source: str, model_cfg: dict):
    """Call the task loader while remaining compatible with the pending helper API.

    The canonical helper contract is ``load_seen10_model(source,
    action_contract=<dict>)``.  Until that optional keyword lands, the
    existing loader remains usable and receives its native config defaults.
    """
    contract = _action_contract({"model": model_cfg})
    parameters = inspect.signature(load_seen10_model).parameters
    kwargs = {}
    if "action_contract" in parameters:
        kwargs["action_contract"] = contract
    elif "action_mode" in parameters:
        kwargs["action_mode"] = contract["mode"]
    elif "contract" in parameters:
        kwargs["contract"] = contract
    model = load_seen10_model(source, **kwargs)
    # Helpers that expose a runtime contract can be checked/configured here;
    # this is deliberately a no-op for the old loader.
    action_space = getattr(model, "action_space", None)
    if action_space is not None:
        for attr, value in (
            ("inference_reset", contract.get("inference_reset")),
            ("reset_dummy_for_inference", contract.get("inference_reset")),
            ("real_dim", contract["external_action_dim"]),
            ("max_dim", contract["model_action_dim"]),
        ):
            if hasattr(action_space, attr):
                setattr(action_space, attr, value)
    return model


def _prepare_action_for_forward(model, inputs: dict, contract: dict):
    """Invoke the core 5D→20D helper immediately before ``model.forward``.

    Fair action spaces must expose one of ``prepare_action_for_model`` or
    ``pad_before_noise`` on the action-space object.  The private
    ``_pad_to_model_dim`` fallback keeps this wrapper compatible with the
    current checkout while the shared helper is being integrated.  The
    helper only pads the supervised target; noise must still be sampled by
    the model after this call, so dummy channels never enter the loss.
    """
    if contract.get("padding") != "pad_before_noise" or "action" not in inputs:
        return inputs
    if _core_pad_seen10_action is not None:
        prepared = _core_pad_seen10_action(inputs["action"])
        if not torch.is_tensor(prepared) or prepared.shape[:-1] != inputs["action"].shape[:-1] or prepared.shape[-1] != contract["model_action_dim"]:
            raise ValueError(
                "The core Seen-10 pad helper must return [B,T,20]; "
                f"got {getattr(prepared, 'shape', None)}"
            )
        inputs["action"] = prepared
        return inputs
    action_space = getattr(model, "action_space", None)
    if action_space is None:
        raise RuntimeError("Fair Seen-10 model does not expose an action_space helper")
    helper = None
    for name in ("prepare_action_for_model", "pad_before_noise", "prepare_training_action", "pad_to_model_dim", "_pad_to_model_dim"):
        candidate = getattr(action_space, name, None)
        if callable(candidate):
            helper = candidate
            break
    if helper is None:
        raise RuntimeError(
            "Fair action contract requires action_space.prepare_action_for_model(action) "
            "or an equivalent pad-before-noise helper"
        )
    try:
        prepared = helper(inputs["action"])
    except TypeError:
        prepared = helper(inputs["action"], mode="train")
    if not torch.is_tensor(prepared) or prepared.shape[:-1] != inputs["action"].shape[:-1] or prepared.shape[-1] != contract["model_action_dim"]:
        raise ValueError(
            "The pad-before-noise helper must return [B,T,20]; "
            f"got {getattr(prepared, 'shape', None)}"
        )
    inputs["action"] = prepared
    return inputs


def _is_peft_model(model) -> bool:
    return hasattr(model, "peft_config") and hasattr(model, "base_model")


def _peft_adapter_name(model) -> str:
    configs = getattr(model, "peft_config", {})
    if isinstance(configs, dict) and configs:
        return next(iter(configs))
    return "default"


def _module_from_path(root, path: str):
    current = root
    for component in path.split("."):
        if not component:
            continue
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


def _merged_inference_state(model) -> dict[str, torch.Tensor]:
    """Materialize a clean XVLA state dict from a PEFT model.

    ``PeftModel.save_pretrained`` intentionally writes adapter-only files.
    Inference loads the native XVLA loader, so we export a merged state under
    the ordinary ``model.safetensors`` name while retaining adapter artifacts
    and the untouched full PEFT state for exact resume.
    """
    if not _is_peft_model(model):
        return {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        }
    state = model.state_dict()
    adapter = _peft_adapter_name(model)
    base_model = getattr(model, "get_base_model", lambda: model)()
    config = getattr(model, "peft_config", {}).get(adapter)
    scaling_default = None
    if config is not None and getattr(config, "r", None):
        scaling_default = float(getattr(config, "lora_alpha", config.r)) / float(config.r)
    merged = {}
    # First pass preserves ordinary and original-module parameters.  The
    # modules_to_save value is applied in a second pass below so it wins over
    # the frozen original value regardless of state-dict ordering.
    saved_wrappers = []
    lora_layers = {}
    for key, value in state.items():
        clean = key.removeprefix("base_model.model.")
        if ".lora_A." in clean or ".lora_B." in clean or ".lora_embedding_" in clean or ".lora_magnitude_vector." in clean:
            continue
        if ".base_layer." in clean:
            prefix, suffix = clean.split(".base_layer.", 1)
            target_key = f"{prefix}.{suffix}"
            merged[target_key] = value.detach().cpu().contiguous()
            # LoRA's B@A update belongs only to the base weight.  Bias-bearing
            # target modules also expose ``base_layer.bias``; registering that
            # key would later try to add a rank-2 update to a rank-1 bias.
            if suffix == "weight":
                lora_layers[prefix] = target_key
            continue
        if ".modules_to_save." in clean:
            saved_wrappers.append((clean, value))
            continue
        if ".original_module." in clean:
            clean = clean.replace(".original_module.", ".")
        merged[clean] = value.detach().cpu().contiguous()

    for clean, value in saved_wrappers:
        prefix, suffix = clean.split(".modules_to_save.", 1)
        adapter_prefix = f"{adapter}."
        if suffix.startswith(adapter_prefix):
            suffix = suffix[len(adapter_prefix):]
        merged[f"{prefix}.{suffix}"] = value.detach().cpu().contiguous()

    # Merge linear LoRA weights into the native base-layer key.  The fair
    # config targets all linear modules, for which B@A is the PEFT update.
    for prefix, target_key in lora_layers.items():
        a_key = f"base_model.model.{prefix}.lora_A.{adapter}.weight"
        b_key = f"base_model.model.{prefix}.lora_B.{adapter}.weight"
        if a_key not in state or b_key not in state:
            continue
        # The exported base tensor is already on CPU.  PEFT's adapter tensors
        # may still live on the training GPU, so move the two low-rank factors
        # to the base device before multiplying and adding the update.
        base = merged[target_key]
        lora_b = state[b_key].detach().to(device=base.device, dtype=torch.float32)
        lora_a = state[a_key].detach().to(device=base.device, dtype=torch.float32)
        update = lora_b @ lora_a
        module = _module_from_path(base_model, prefix)
        scaling = getattr(module, "scaling", {}).get(adapter, scaling_default) if module is not None else scaling_default
        if scaling is None:
            scaling = scaling_default or 1.0
        merged[target_key] = (base.float() + update * float(scaling)).to(dtype=base.dtype).contiguous()
    return merged


def _save_state_dict(path: Path, state: dict[str, torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        try:
            save_file(state, str(path))
        except RuntimeError as error:
            # Tied embeddings can share storage in the native Florence
            # backbone.  Cloning only on this exceptional path preserves the
            # safe format without changing model values.
            if "share memory" not in str(error).lower() and "shared" not in str(error).lower():
                raise
            save_file({key: value.clone().contiguous() for key, value in state.items()}, str(path))
    except ImportError:
        # Transformers still understands this fallback during local recovery;
        # the metadata records which representation was written.
        torch.save(state, path.with_suffix(".pt"))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.is_file():
        try:
            from safetensors.torch import load_file
            return load_file(str(path), device="cpu")
        except ImportError:
            pass
    fallback = path.with_suffix(".pt")
    if fallback.is_file():
        return torch.load(fallback, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"Missing saved model state: {path}")


def _save_full_resume_state(model, output_dir: Path):
    """Save the trainable PEFT/full-module tensors outside adapter format.

    Accelerate already writes the complete model/optimizer state used for an
    exact resume.  This compact sidecar covers every declared trainable tensor,
    including the bare ``vlm.image_projection`` parameter that PEFT adapter
    serialization cannot represent, without duplicating another 3.5 GB base.
    """
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if torch.is_tensor(value) and key in trainable
    }
    missing = sorted(trainable - state.keys())
    if missing:
        raise RuntimeError(f"Trainable parameters missing from checkpoint sidecar: {missing[:8]}")
    _save_state_dict(output_dir / "trainable_state.safetensors", state)
    return {
        "path": "trainable_state.safetensors",
        "format": "safetensors_trainable_sidecar",
        "parameter_count": len(state),
        "tensor_elements": int(sum(value.numel() for value in state.values())),
    }


def _restore_full_resume_state(model, checkpoint_dir: Path, *, required: bool = False):
    path = checkpoint_dir / "trainable_state.safetensors"
    if not path.exists() and not path.with_suffix(".pt").exists():
        # Compatibility with checkpoints produced during implementation.
        path = checkpoint_dir / "model_state.safetensors"
    if not path.exists() and not path.with_suffix(".pt").exists():
        if required:
            raise FileNotFoundError(
                f"Checkpoint declares the current resume format but has no trainable-state sidecar: {checkpoint_dir}"
            )
        return {"restored": False, "missing": True}
    state = _load_state_dict(path)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(trainable - state.keys())
    if missing_trainable:
        raise RuntimeError(
            "Resume sidecar does not cover every current trainable parameter: "
            f"{missing_trainable[:8]}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in complete resume state: {unexpected[:8]}")
    return {
        "restored": True,
        "restored_trainable_parameters": len(trainable),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _save_inference_model(model, output_dir: Path):
    """Write a native XVLA model that ``load_seen10_model`` can open."""
    if not _is_peft_model(model):
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir, safe_serialization=True)
        else:
            config = getattr(model, "config", None)
            if config is not None and hasattr(config, "save_pretrained"):
                config.save_pretrained(output_dir)
            _save_state_dict(output_dir / "model.safetensors", {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
                if torch.is_tensor(value)
            })
        return {"format": "native_pretrained", "merged": False}
    base_model = getattr(model, "get_base_model", lambda: model)()
    config = getattr(base_model, "config", None)
    if config is None:
        raise RuntimeError("PEFT model has no native XVLA config for inference export")
    config.save_pretrained(output_dir)
    _save_state_dict(output_dir / "model.safetensors", _merged_inference_state(model))
    return {"format": "native_pretrained", "merged": True}


def _save_adapter_artifacts(model, output_dir: Path):
    if not _is_peft_model(model):
        return None
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    return {
        "path": "adapter",
        "format": "peft_adapter",
        "adapter_name": _peft_adapter_name(model),
        "base_model_class": type(getattr(model, "get_base_model", lambda: model)()).__name__,
    }


def _optimizer_metadata(optimizer, model, *, requested: dict | None = None) -> dict:
    names_by_id = {}
    parameters_by_name = dict(model.named_parameters())
    for name, parameter in parameters_by_name.items():
        names_by_id.setdefault(id(parameter), []).append(name)
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        names = [name for parameter in group.get("params", ()) for name in names_by_id.get(id(parameter), ())]
        trainable_names = [name for name in names if parameters_by_name[name].requires_grad]
        groups.append({
            "index": index,
            "name": str(group.get("name", f"group_{index}")),
            "actual_lr": float(group.get("lr", 0.0)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "parameter_tensors": len(names),
            "parameter_elements": int(sum(parameter.numel() for parameter in group.get("params", ()))),
            "trainable_parameter_tensors": len(trainable_names),
            "trainable_roles": sorted({_parameter_role(name) for name in trainable_names}),
            "trainable_parameter_names": trainable_names,
        })
    result = {
        "type": type(optimizer).__name__,
        "param_groups": groups,
    }
    if requested is not None:
        result["requested"] = _json_copy(requested)
    return result


def _gradient_metadata(model) -> dict:
    """Audit which declared trainable roles received a finite gradient."""
    by_role = {}
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"Frozen parameter unexpectedly received a gradient: {name}")
            continue
        role = _parameter_role(name)
        entry = by_role.setdefault(role, {"trainable_tensors": 0, "with_gradient": 0, "nonzero_gradient": 0})
        entry["trainable_tensors"] += 1
        if parameter.grad is None:
            missing.append(name)
            continue
        entry["with_gradient"] += 1
        if not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        if torch.count_nonzero(parameter.grad).item():
            entry["nonzero_gradient"] += 1
    if nonfinite:
        raise FloatingPointError(f"Nonfinite gradients detected: {nonfinite[:8]}")
    return {
        "by_role": by_role,
        "missing_gradient_names": missing,
        "all_gradients_finite": True,
    }


def _runtime_metadata(*, config: dict, args, data_root: Path, pretrained: str, batching: dict, policy: str, model=None, optimizer=None, phase=None) -> dict:
    train_cfg = config["training"]
    runtime = {
        "schema": "xvla_seen10_runtime_v2",
        "policy": policy,
        "freeze_vision_language_connector": bool(train_cfg.get("freeze_vision_language_connector", False)),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "command": list(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "data_root": str(data_root),
        "pretrained": str(pretrained),
        "mixed_precision": str(args.mixed_precision or train_cfg.get("mixed_precision", "no")),
        "batching": _json_copy(batching),
        "phase": phase or {},
        "augmentation": _json_copy(train_cfg.get("augmentation", {"enabled": False})),
    }
    if model is not None:
        runtime["model_class"] = type(model).__name__
        runtime["model_config"] = _json_copy(model.config.to_dict()) if getattr(model, "config", None) is not None else None
    if optimizer is not None:
        runtime["optimizer"] = _optimizer_metadata(optimizer, model)
    return runtime


def _write_json_atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _scale_gradients(model, divisor: int):
    if divisor <= 0:
        raise ValueError("gradient divisor must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(divisor))


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
        config = json.load(stream)
    _validate_config(config)
    return config


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
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--effective-batch-size", type=int)
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
    if (
        moved["proprio"].shape[-1] != 20
        or moved["proprio"].requires_grad
        or not torch.isfinite(moved["proprio"]).all()
        or torch.count_nonzero(moved["proprio"]).item() != 0
    ):
        raise ValueError("CSGO localization requires an all-zero 20D proprio vector")
    if moved["image_mask"].shape[1] != 2 or not torch.all(moved["image_mask"]):
        raise ValueError("Expected exactly two valid views: first-person and radar")
    if training and tuple(moved["action"].shape[1:]) != (1, 5):
        raise ValueError(f"Expected normalized action target [B,1,5], got {tuple(moved['action'].shape)}")
    if training and not torch.isfinite(moved["action"]).all():
        raise ValueError("Seen-10 action targets must be finite")
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
    # sum(MSE), sum(MAE), sample count, nonfinite prediction count
    local_stats = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    selected = set(visualization_identities)
    local_visualization_rows = []
    try:
        for batch in dataloader:
            inputs = _move_inputs(batch, accelerator.device, training=True)
            target = inputs.pop("action")
            prediction = model.generate_actions(**inputs, steps=steps)
            if prediction.shape != target.shape:
                raise ValueError(f"Generated action {tuple(prediction.shape)} != target {tuple(target.shape)}")
            prediction_float = prediction.float()
            finite_prediction = torch.isfinite(prediction_float).all(dim=(1, 2))
            local_stats[3] += (~finite_prediction).double().sum()
            safe_prediction = torch.where(
                torch.isfinite(prediction_float),
                prediction_float,
                torch.zeros_like(prediction_float),
            )
            delta = safe_prediction - target.float()
            # Pitch is scored linearly; yaw uses the normalized 2*pi shortest arc.
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
    if totals[3].item() or not torch.isfinite(totals[:3]).all():
        raise FloatingPointError(
            "Validation produced nonfinite predictions or aggregate metrics: "
            f"nonfinite_predictions={int(totals[3].item())}, totals={totals[:3].tolist()}"
        )
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
    action_contract: dict | None = None,
    state_contract: dict | None = None,
    normalization: dict | None = None,
    runtime: dict | None = None,
    initialization: dict | None = None,
    optimizer_metadata: dict | None = None,
    batching: dict | None = None,
    phase: dict | None = None,
    data_contract: dict | None = None,
):
    progress = {
        "schema": "xvla_seen10_checkpoint_progress_v2",
        "global_step": global_step,
        "optimizer_updates": global_step,
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
    if action_contract is not None:
        progress["action_contract"] = _json_copy(action_contract)
    if state_contract is not None:
        progress["state"] = _json_copy(state_contract)
        progress["state_contract"] = _json_copy(state_contract)
    if normalization is not None:
        progress["normalization"] = _json_copy(normalization)
    if runtime is not None:
        progress["runtime"] = _json_copy(runtime)
    if initialization is not None:
        progress["initialization"] = _json_copy(initialization)
    if optimizer_metadata is not None:
        progress["optimizer"] = _json_copy(optimizer_metadata)
    if batching is not None:
        progress["batching"] = _json_copy(batching)
    if phase is not None:
        progress["trainable_phase"] = _json_copy(phase)
    if data_contract is not None:
        progress["data_contract"] = _json_copy(data_contract)
    return progress


def _checkpoint_data_cursor(current_epoch: int, batches_in_epoch: int, micro_batches_per_epoch: int):
    """Persist the next unread batch, canonicalizing an epoch boundary."""
    if current_epoch < 0 or batches_in_epoch < 0 or micro_batches_per_epoch <= 0:
        raise ValueError("Invalid checkpoint data cursor")
    if batches_in_epoch > micro_batches_per_epoch:
        raise ValueError(
            f"batches_in_epoch={batches_in_epoch} exceeds epoch length {micro_batches_per_epoch}"
        )
    if batches_in_epoch == micro_batches_per_epoch:
        return current_epoch + 1, 0
    return current_epoch, batches_in_epoch


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
        inference_metadata = _save_inference_model(unwrapped, temp_dir)
        processor.save_pretrained(temp_dir)
        adapter_metadata = _save_adapter_artifacts(unwrapped, temp_dir)
        resume_metadata = _save_full_resume_state(unwrapped, temp_dir)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temp_dir / "accelerator_state"))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        progress["model_export"] = inference_metadata
        progress["adapter_export"] = adapter_metadata
        progress["resume_state"] = resume_metadata
        progress["checkpoint_id"] = uuid.uuid4().hex
        with (temp_dir / "progress.json").open("w", encoding="utf-8") as stream:
            json.dump(progress, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        if progress.get("action_contract") is not None:
            _write_json_atomic(temp_dir / "action_contract.json", progress["action_contract"])
        if progress.get("runtime") is not None:
            _write_json_atomic(temp_dir / "runtime.json", progress["runtime"])
        if progress.get("initialization") is not None:
            _write_json_atomic(temp_dir / "initialization.json", progress["initialization"])
        if progress.get("optimizer") is not None:
            _write_json_atomic(temp_dir / "optimizer.json", progress["optimizer"])
        if progress.get("data_contract") is not None:
            _write_json_atomic(temp_dir / "data_contract.json", progress["data_contract"])
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
            "schema": "xvla_seen10_checkpoint_pointer_v2",
            "checkpoint": checkpoint.name,
            "checkpoint_id": progress["checkpoint_id"],
            "global_step": progress["global_step"],
            "seed": progress["seed"],
            "selection": "validation_only" if name == "best" else "late/last",
        }
        if "validation_normalized_mse" in progress:
            pointer["validation_normalized_mse"] = progress["validation_normalized_mse"]
        for key in ("action_contract", "initialization", "batching", "data_contract"):
            if key in progress:
                pointer[key] = progress[key]
        pointer_tmp = checkpoints_dir / f".{name}.json.tmp"
        pointer_tmp.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
        os.replace(pointer_tmp, checkpoints_dir / f"{name}.json")
    accelerator.wait_for_everyone()


def _set_best_checkpoint(accelerator: Accelerator, output_root: Path, checkpoint: Path, progress: dict):
    _set_checkpoint_pointer(accelerator, output_root, checkpoint, progress, "best")


def _set_last_checkpoint(
    accelerator: Accelerator, output_root: Path, checkpoint: Path, progress: dict,
    *, late_alias: bool = False,
):
    _set_checkpoint_pointer(accelerator, output_root, checkpoint, progress, "last")
    if late_alias:
        _set_checkpoint_pointer(accelerator, output_root, checkpoint, progress, "late")


_STEP_CHECKPOINT_PATTERN = re.compile(r"^step_(\d{8})(?:_resume_(\d+))?$")


def _prune_periodic_checkpoints(accelerator: Accelerator, output_root: Path, max_to_keep: int):
    """Keep at most ``max_to_keep`` physical periodic checkpoints.

    Best and last are pointer names, not additional checkpoint copies.  Their
    targets are protected first; remaining slots are filled by the newest
    periodic checkpoints.  This caps checkpoint-selection density even after
    resumes create a duplicate-step directory.
    """
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
        available = {path.name for _, _, path in candidates}
        pointer_targets = []
        for pointer_name in ("best", "last"):
            pointer = checkpoints_dir / pointer_name
            if not pointer.is_symlink():
                continue
            target = os.readlink(pointer)
            if Path(target).name == target and target in available and target not in pointer_targets:
                pointer_targets.append(target)
        if len(pointer_targets) > max_to_keep:
            raise RuntimeError("max_periodic_checkpoints is too small to retain best and last targets")
        protected = set(pointer_targets[:max_to_keep])
        for _, _, path in reversed(candidates):
            if len(protected) >= max_to_keep:
                break
            protected.add(path.name)
        for _, _, path in candidates:
            if path.name not in protected:
                shutil.rmtree(path)
    accelerator.wait_for_everyone()


def _validate_resume_connector_freeze(progress: dict, frozen: bool):
    checkpoint_frozen = bool(
        (progress.get("runtime") or {}).get("freeze_vision_language_connector", False)
    )
    if checkpoint_frozen != frozen:
        raise ValueError("Resume checkpoint vision-language connector freeze policy does not match this run")


def _validate_resume(
    progress: dict,
    *,
    args,
    data_root: Path,
    train_dataset,
    validation_dataset,
    batch_size: int,
    world_size: int,
    action_contract: dict | None = None,
    batching: dict | None = None,
    data_contract: dict | None = None,
    freeze_vision_language_connector: bool | None = None,
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
    if freeze_vision_language_connector is not None:
        _validate_resume_connector_freeze(progress, freeze_vision_language_connector)
    checkpoint_contract = _canonical_resume_action_contract(progress.get("action_contract"))
    requested_contract = _canonical_resume_action_contract(action_contract)
    if requested_contract is not None and checkpoint_contract != requested_contract:
        raise ValueError("Resume checkpoint action contract does not match this run")
    if batching is not None:
        checkpoint_batching = progress.get("batching", {})
        for key in ("micro_batch_size", "world_size", "effective_batch_size", "gradient_accumulation_steps"):
            if checkpoint_batching.get(key) != batching.get(key):
                raise ValueError(
                    "Resume checkpoint batching does not match this run: "
                    f"{key}: checkpoint={checkpoint_batching.get(key)!r}, requested={batching.get(key)!r}"
                )
    if data_contract is not None:
        checkpoint_data_contract = progress.get("data_contract")
        checkpoint_data_id = (
            checkpoint_data_contract.get("contract_sha256")
            if isinstance(checkpoint_data_contract, dict)
            else None
        )
        requested_data_id = data_contract.get("contract_sha256")
        if not isinstance(requested_data_id, str) or not requested_data_id:
            raise ValueError("Current Seen-10 data contract has no contract_sha256")
        if checkpoint_data_id != requested_data_id:
            raise ValueError(
                "Resume checkpoint data contract does not match this run: "
                f"checkpoint={checkpoint_data_id!r}, requested={requested_data_id!r}"
            )


def main():
    args = _parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.limit_per_map is not None and not args.smoke:
        raise ValueError("--limit-per-map is only allowed with --smoke")
    config = _load_config(args.config)
    action_contract = _validate_config(config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    policy = str(train_cfg.get("training_policy", _LEGACY_POLICY))
    visualization_cfg = config.get("visualization", {})
    visualization_enabled = bool(visualization_cfg.get("enabled", True))

    data_root = Path(args.data_root or os.environ.get("CSGO_DATA_ROOT") or data_cfg["root"]).expanduser().resolve()
    data_contract = seen10_data_contract(data_root)
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
    if policy == _FAIR_POLICY and not args.smoke:
        resolved_fixed = {
            "iters": iters,
            "eval_interval": eval_interval,
            "save_interval": save_interval,
            "mixed_precision": mixed_precision,
        }
        expected_fixed = {
            "iters": 19500,
            "eval_interval": 4000 if train_cfg.get("freeze_vision_language_connector", False) else 3900,
            "save_interval": 4000 if train_cfg.get("freeze_vision_language_connector", False) else 3900,
            "mixed_precision": "bf16",
        }
        mismatches = [
            f"{key}={resolved_fixed[key]!r}, expected={value!r}"
            for key, value in expected_fixed.items()
            if resolved_fixed[key] != value
        ]
        if mismatches:
            raise ValueError(
                "Formal fair-run CLI overrides violate the fixed experiment budget: "
                + "; ".join(mismatches)
            )

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=str(output_root),
        mixed_precision=mixed_precision,
    )
    _prepare_output_root(accelerator, output_root, args=args, config=config, resume=resume)
    accelerator.init_trackers("X-VLA-CSGO-Seen10")
    logger = get_logger("train_seen10", output_dir=output_root, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index)
    batching_config = dict(train_cfg)
    if args.effective_batch_size is not None:
        batching_config["effective_batch_size"] = args.effective_batch_size
    if args.gradient_accumulation_steps is not None and policy == _LEGACY_POLICY:
        batching_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    batching = _resolve_gradient_accumulation(
        batching_config,
        batch_size=batch_size,
        world_size=accelerator.num_processes,
        smoke=args.smoke,
    )
    if policy == _FAIR_POLICY and not args.smoke:
        if batching["effective_batch_size"] != 128:
            raise ValueError(
                "Formal fair Seen-10 requires effective localization batch 128; "
                f"got {batching['effective_batch_size']}"
            )
        if (
            args.gradient_accumulation_steps is not None
            and args.gradient_accumulation_steps != batching["gradient_accumulation_steps"]
        ):
            raise ValueError(
                "--gradient-accumulation-steps must equal the world-size-derived fair value "
                f"{batching['gradient_accumulation_steps']}"
            )
    logger.info("Starting Seen-10 localization: %s", vars(args))
    logger.info(
        "Optimizer-update contract: micro_batch=%d accumulation=%d world_size=%d effective_batch=%d",
        batching["micro_batch_size"],
        batching["gradient_accumulation_steps"],
        batching["world_size"],
        batching["effective_batch_size"],
    )

    augmentation_enabled = bool(train_cfg.get("augmentation", {}).get("enabled", False))
    train_dataset = _make_seen10_dataset(
        data_root,
        data_cfg.get("train_split", "seen_train"),
        include_targets=True,
        limit_per_map=limit_per_map,
        augmentation_enabled=augmentation_enabled,
    )
    validation_dataset = _make_seen10_dataset(
        data_root,
        data_cfg.get("validation_split", "seen_validation"),
        include_targets=True,
        limit_per_map=limit_per_map,
        augmentation_enabled=False,
    )
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
        # The formal fair policy consumes only complete global updates.  The
        # legacy policy keeps the original main@4aed8865 sampler behavior.
        drop_last=policy == _FAIR_POLICY,
    )
    validation_sampler = DistributedEvalSampler(
        validation_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    resume_progress = None
    if resume is not None:
        with (resume / "progress.json").open(encoding="utf-8") as stream:
            resume_progress = json.load(stream)
        if policy == _FAIR_POLICY:
            _validate_resume_connector_freeze(
                resume_progress, bool(train_cfg.get("freeze_vision_language_connector", False))
            )
    processor_source = str(resume) if resume is not None else pretrained
    processor = load_seen10_processor(processor_source)
    processor_image_size = processor_size(processor)
    collator_train = _make_collator(processor, include_targets=True, training=True)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + accelerator.process_index + 1000)
    train_loader_persistent_workers = policy == _LEGACY_POLICY and num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=train_loader_persistent_workers,
        collate_fn=collator_train,
        generator=train_generator,
        drop_last=policy == _FAIR_POLICY,
    )
    loader_budget = _resolve_train_loader_budget(
        len(train_dataset),
        batch_size=batch_size,
        world_size=accelerator.num_processes,
        gradient_accumulation_steps=batching["gradient_accumulation_steps"],
        policy=policy,
    )
    if len(train_sampler) != loader_budget["sampler_samples_per_rank"]:
        raise RuntimeError(
            "Distributed train sampler length disagrees with the selected drop policy: "
            f"actual={len(train_sampler)}, expected={loader_budget['sampler_samples_per_rank']}"
        )
    if len(train_loader) != loader_budget["loader_micro_batches_per_epoch"]:
        raise RuntimeError(
            "Distributed train loader length disagrees with the selected drop policy: "
            f"actual={len(train_loader)}, expected={loader_budget['loader_micro_batches_per_epoch']}"
        )
    if len(train_loader) < loader_budget["micro_batches_per_epoch"]:
        raise ValueError(
            "Distributed train loader cannot provide the requested complete effective batches: "
            f"loader_batches={len(train_loader)}, required={loader_budget['micro_batches_per_epoch']}"
        )
    batching.update(loader_budget)
    batching.update({
        "total_updates": int(iters),
        "expected_epochs": math.ceil(int(iters) / loader_budget["updates_per_epoch"]),
        "train_sampler_drop_last": policy == _FAIR_POLICY,
        "train_loader_drop_last": policy == _FAIR_POLICY,
        "train_loader_persistent_workers": train_loader_persistent_workers,
        "train_loader_seed_scheme": (
            "fair_epoch_rank_seed_v1" if policy == _FAIR_POLICY else "legacy_initial_seed"
        ),
    })
    updates_per_epoch = batching["updates_per_epoch"]
    micro_batches_per_epoch = batching["micro_batches_per_epoch"]
    if policy == _FAIR_POLICY and not args.smoke:
        observed_budget = {
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "updates_per_epoch": updates_per_epoch,
            "dropped_samples_per_epoch": batching["dropped_samples_per_epoch"],
            "expected_epochs": batching["expected_epochs"],
            "sample_exposures": int(iters) * batching["effective_batch_size"],
        }
        expected_budget = {
            "train_samples": 50_000,
            "validation_samples": 5_000,
            "updates_per_epoch": 390,
            "dropped_samples_per_epoch": 80,
            "expected_epochs": 50,
            "sample_exposures": 2_496_000,
        }
        mismatches = [
            f"{key}={observed_budget[key]!r}, expected={value!r}"
            for key, value in expected_budget.items()
            if observed_budget[key] != value
        ]
        if mismatches:
            raise ValueError("Formal fair Seen-10 data/update budget mismatch: " + "; ".join(mismatches))
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        collate_fn=_make_collator(processor, include_targets=True, training=False),
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index + 2000),
    )

    initialization = None
    adapter_resume_dir = None
    if resume_progress is not None:
        adapter_export = resume_progress.get("adapter_export") or {}
        candidate = resume / str(adapter_export.get("path", "adapter"))
        if candidate.is_dir():
            adapter_resume_dir = candidate
        initialization = deepcopy(resume_progress.get("initialization")) if resume_progress.get("initialization") else None
    model_source = str(resume) if resume is not None and adapter_resume_dir is None else pretrained
    if adapter_resume_dir is not None and initialization:
        model_source = str(initialization.get("base_model_source", initialization.get("resolved", pretrained)))
    model = _call_model_loader(model_source, model_cfg)
    model, peft_enabled = _maybe_attach_lora(model, train_cfg)
    expected_size = model_cfg.get("processor_image_size_expected")
    if expected_size is not None and processor_image_size != {"height": expected_size, "width": expected_size}:
        message = (
            "Pretrained processor image size does not match the configured native resolution: "
            f"actual={processor_image_size}, expected={expected_size}x{expected_size}"
        )
        if policy == _FAIR_POLICY:
            raise ValueError(message)
        if accelerator.is_main_process:
            logger.warning("%s; no resize override is applied", message)

    parameter_audit = _configure_trainable_parameters(model, train_cfg, policy)
    phase = _phase_metadata(train_cfg, 0, policy)
    optimizer = _build_seen10_optimizer(model, train_cfg, policy)
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

    unwrapped_model = accelerator.unwrap_model(model)
    if initialization is None:
        initialization = _source_provenance(pretrained, model=unwrapped_model, config=config)
        _verify_fair_initialization(initialization, model_cfg, policy)
    initialization.update({
        "base_model_source": str(initialization.get("base_model_source", pretrained)),
        "loaded_model_source": str(model_source),
        "parameterization": "peft" if peft_enabled else "native",
        "adapter_resume": str(adapter_resume_dir) if adapter_resume_dir is not None else None,
    })
    runtime = _runtime_metadata(
        config=config,
        args=args,
        data_root=data_root,
        pretrained=pretrained,
        batching=batching,
        policy=policy,
        model=unwrapped_model,
        optimizer=optimizer,
        phase=phase,
    )
    runtime["initialization_source_id"] = initialization.get("source_id")
    state_contract = _json_copy(config.get("state", {
        "use_proprio": False,
        "dim": 20,
        "values": "all zeros",
    }))
    normalization = _json_copy(config.get("normalization", COORDINATE_DESCRIPTION))
    runtime.update({
        "action_contract": _json_copy(action_contract),
        "data_contract": _json_copy(data_contract),
        "state": _json_copy(state_contract),
        "normalization": _json_copy(normalization),
        "initialization": _json_copy(initialization),
        "parameters": _json_copy(parameter_audit),
    })
    optimizer_request = {
        "type": "AdamW",
        "learning_rate": float(train_cfg["learning_rate"]),
        "betas": [float(value) for value in train_cfg["betas"]],
        "weight_decay": float(train_cfg["weight_decay"]),
        "max_grad_norm": float(train_cfg["max_grad_norm"]),
        "constant_lr": not bool(train_cfg.get("use_cosine_decay", False)) and int(train_cfg.get("warmup_steps", 0)) == 0,
    }
    if accelerator.is_main_process:
        _write_json_atomic(output_root / "runtime.json", runtime)
        _write_json_atomic(output_root / "action_contract.json", action_contract)
        _write_json_atomic(output_root / "data_contract.json", data_contract)
        _write_json_atomic(output_root / "initialization.json", initialization)
        _write_json_atomic(output_root / "parameter_audit.json", parameter_audit)

    global_step = 0
    start_epoch = 0
    batches_in_epoch = 0
    best_validation_mse = math.inf
    if resume is not None:
        _validate_resume(
            resume_progress,
            args=args,
            data_root=data_root,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            world_size=accelerator.num_processes,
            action_contract=(action_contract if policy == _FAIR_POLICY or "action_contract" in resume_progress else None),
            batching=(batching if policy == _FAIR_POLICY or "batching" in resume_progress else None),
            data_contract=(
                data_contract
                if policy == _FAIR_POLICY or "data_contract" in resume_progress
                else None
            ),
            freeze_vision_language_connector=(
                bool(train_cfg.get("freeze_vision_language_connector", False))
                if policy == _FAIR_POLICY else None
            ),
        )
        global_step = int(resume_progress["global_step"])
        phase = _phase_metadata(train_cfg, global_step, policy)
        accelerator.load_state(str(resume / "accelerator_state"))
        # Accelerate restores the native optimizer/model state.  The complete
        # PEFT state is also restored explicitly so bare Parameters (for
        # example image_projection) cannot be lost by adapter-only saves.
        current_resume_format = (
            resume_progress.get("schema") == "xvla_seen10_checkpoint_progress_v2"
            or resume_progress.get("resume_state") is not None
        )
        _restore_full_resume_state(
            accelerator.unwrap_model(model),
            resume,
            required=(policy == _FAIR_POLICY or current_resume_format),
        )
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
    if action_contract.get("inference_steps") is not None:
        inference_steps = int(action_contract["inference_steps"])
    validation_seed = int(train_cfg.get("validation_seed", 1729))
    started = time.time()
    current_epoch = start_epoch
    skip_batches = batches_in_epoch
    micro_batches_in_update = 0
    update_loss_sum = None
    gradient_audit_written = (output_root / "gradient_audit.json").is_file()

    while global_step < iters:
        train_sampler.set_epoch(current_epoch)
        if policy == _FAIR_POLICY:
            epoch_seed = _fair_train_loader_seed(
                args.seed,
                accelerator.process_index,
                current_epoch,
            )
            train_generator.manual_seed(epoch_seed)
        epoch_had_batch = False
        for batch_index, batch in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
            if batch_index >= micro_batches_per_epoch:
                break
            skip_batches = 0
            epoch_had_batch = True
            model.train()
            _keep_frozen_vision_eval(accelerator.unwrap_model(model), policy, train_cfg)
            inputs = _move_inputs(batch, accelerator.device, training=True)
            if micro_batches_in_update == 0:
                phase = _phase_metadata(train_cfg, global_step, policy)
                _update_seen10_lrs(optimizer, global_step, train_cfg, policy, schedule_args)
            _prepare_action_for_forward(accelerator.unwrap_model(model), inputs, action_contract)
            loss_dict = model(**inputs)
            loss = sum(loss_dict.values())
            accelerator.backward(loss)
            micro_batches_in_update += 1
            update_loss_sum = loss.detach().float() if update_loss_sum is None else update_loss_sum + loss.detach().float()
            # This is the manual equivalent of Accelerate's
            # ``accelerator.sync_gradients`` boundary.  We intentionally do
            # not delegate accumulation to a hidden plugin so the effective
            # global batch can be audited from the checkpoint metadata.
            sync_gradients = micro_batches_in_update == batching["gradient_accumulation_steps"]
            if not sync_gradients:
                continue

            # Every optimizer update is an exact mean over the configured
            # global batch.  The loop intentionally never performs a partial
            # final update; the tail of each epoch is dropped below.
            _scale_gradients(model, micro_batches_in_update)
            if train_cfg.get("max_grad_norm"):
                accelerator.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            if not gradient_audit_written:
                gradient_audit = _gradient_metadata(accelerator.unwrap_model(model))
                if accelerator.is_main_process:
                    _write_json_atomic(output_root / "gradient_audit.json", gradient_audit)
                gradient_audit_written = True
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            batches_in_epoch = batch_index + 1
            mean_loss = update_loss_sum / float(micro_batches_in_update)
            micro_batches_in_update = 0
            update_loss_sum = None
            if global_step % log_interval == 0 or global_step == 1:
                mean_loss = accelerator.reduce(mean_loss, reduction="mean")
                logs = {
                    "loss_total": float(mean_loss.item()),
                    "global_step": global_step,
                    "optimizer_updates": global_step,
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
                checkpoint_epoch, checkpoint_batches = _checkpoint_data_cursor(
                    current_epoch,
                    batches_in_epoch,
                    micro_batches_per_epoch,
                )
                runtime_snapshot = deepcopy(runtime)
                runtime_snapshot["phase"] = _json_copy(phase)
                runtime_snapshot["optimizer"] = _optimizer_metadata(
                    optimizer,
                    accelerator.unwrap_model(model),
                    requested=optimizer_request,
                )
                progress = _checkpoint_progress(
                    global_step=global_step,
                    epoch=checkpoint_epoch,
                    batches_in_epoch=checkpoint_batches,
                    seed=args.seed,
                    smoke=args.smoke,
                    data_root=data_root,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    best_validation_mse=best_validation_mse,
                    processor_image_size=processor_image_size,
                    batch_size=batch_size,
                    world_size=accelerator.num_processes,
                    action_contract=action_contract,
                    state_contract=state_contract,
                    normalization=normalization,
                    runtime=runtime_snapshot,
                    initialization=initialization,
                    optimizer_metadata=_optimizer_metadata(optimizer, accelerator.unwrap_model(model), requested=optimizer_request),
                    batching=batching,
                    phase=phase,
                    data_contract=data_contract,
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
                if accelerator.is_main_process:
                    _write_json_atomic(output_root / "runtime.json", runtime_snapshot)
                _set_last_checkpoint(
                    accelerator, output_root, checkpoint, progress,
                    late_alias=bool(train_cfg.get("freeze_vision_language_connector", False)),
                )
                if is_best:
                    _set_best_checkpoint(accelerator, output_root, checkpoint, progress)
                _prune_periodic_checkpoints(accelerator, output_root, max_periodic_checkpoints)
                if accelerator.is_main_process:
                    logger.info("Saved native checkpoint to %s", checkpoint)

            if global_step >= iters:
                break

        if micro_batches_in_update:
            raise RuntimeError("An epoch ended with a partial gradient accumulation; exact effective batch was violated")
        if not epoch_had_batch and skip_batches:
            # A resumed checkpoint was captured at the exact end of this
            # epoch.  Advance once before consuming the next epoch.
            skip_batches = 0
            current_epoch += 1
            continue
        if not epoch_had_batch and len(train_loader) == 0:
            raise RuntimeError("Distributed train loader produced no batches")
        # If training stopped mid-epoch, retain current_epoch and
        # batches_in_epoch for a precise resume.  Otherwise move to the next
        # epoch after consuming exactly the complete update groups.
        if global_step < iters or batches_in_epoch >= micro_batches_per_epoch:
            current_epoch += 1
            skip_batches = 0

    if accelerator.is_main_process:
        logger.info("Training finished at step %d in %.1fs", global_step, time.time() - started)
    accelerator.end_training()


if __name__ == "__main__":
    main()
