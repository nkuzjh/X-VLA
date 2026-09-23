"""Thin CSGO localization adapter for the native XVLA model and processor."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.action_hub import pad_action_to_model_dim
from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


SEEN10_ACTION_MODE = "official_auto"
SEEN10_REAL_ACTION_DIM = 5
SEEN10_MODEL_ACTION_DIM = 20
SEEN10_PROPRIO_DIM = 20


SEEN10_MODEL_CONFIG = {
    "real_action_dim": SEEN10_REAL_ACTION_DIM,
    "max_action_dim": SEEN10_MODEL_ACTION_DIM,
    "num_actions": 1,
    "use_proprio": False,
}


def _contract_mode(action_contract=None, action_mode: str | None = None) -> str:
    """Resolve the persisted action-space implementation for this run."""
    if action_mode is None and isinstance(action_contract, dict):
        action_mode = action_contract.get("mode")
    elif action_mode is None and isinstance(action_contract, str):
        action_mode = action_contract
    mode = str(action_mode or SEEN10_ACTION_MODE).lower()
    if mode not in {"official_auto", "legacy_reset_dummy", "auto"}:
        raise ValueError(f"Unsupported Seen-10 action mode: {mode!r}")
    return mode


def configure_seen10(
    config: XVLAConfig,
    *,
    action_contract=None,
    action_mode: str | None = None,
) -> XVLAConfig:
    """Apply Seen-10 task settings without changing the pretrained head shape.

    The auto action space computes loss and returns predictions over the first
    five dimensions while keeping the native 20-dimensional action decoder.
    """
    for name, value in SEEN10_MODEL_CONFIG.items():
        setattr(config, name, value)
    config.action_mode = _contract_mode(action_contract, action_mode)
    config.action_adaptation = config.action_mode
    return config


def pad_seen10_action(action: torch.Tensor | None) -> torch.Tensor | None:
    """Pad an external Seen-10 target from 5D to the native XVLA width.

    Dataset/collator targets intentionally stay ``[..., 5]``.  Call this
    helper at the boundary immediately before invoking ``XVLA.forward``.  A
    target that was already prepared is returned unchanged, so the same API
    can safely be used by adapters and checkpoint-specific training code.
    """
    return pad_action_to_model_dim(
        action,
        real_dim=SEEN10_REAL_ACTION_DIM,
        model_dim=SEEN10_MODEL_ACTION_DIM,
    )


# Keep a verb that reads naturally at call sites while retaining the concise
# public helper above.
prepare_seen10_action = pad_seen10_action


def load_seen10_config(
    model_path: str,
    *,
    local_files_only: bool = False,
    action_contract=None,
    action_mode: str | None = None,
) -> XVLAConfig:
    """Load and configure a persisted Seen-10 XVLA config."""
    config = XVLAConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    return configure_seen10(
        config,
        action_contract=action_contract,
        action_mode=action_mode,
    )


def build_seen10_base_model(
    model_path: str,
    *,
    local_files_only: bool = False,
    config: XVLAConfig | None = None,
    action_contract=None,
    action_mode: str | None = None,
) -> XVLA:
    """Create a native XVLA base model with the persisted Seen-10 contract.

    Keeping this step separate lets PEFT loaders construct the base model from
    a base checkpoint before attaching an adapter checkpoint.  It also keeps
    action-mode selection in the model config rather than in a training or
    inference script.
    """
    if config is None:
        config = load_seen10_config(
            model_path,
            local_files_only=local_files_only,
            action_contract=action_contract,
            action_mode=action_mode,
        )
    else:
        configure_seen10(
            config,
            action_contract=action_contract,
            action_mode=action_mode,
        )
    return XVLA.from_pretrained(
        model_path,
        config=config,
        local_files_only=local_files_only,
    )


def _adapter_base_model_path(adapter_path: str) -> str | None:
    """Read PEFT's base-model provenance when an adapter is local."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        with config_path.open(encoding="utf-8") as stream:
            adapter_config = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read PEFT adapter config: {config_path}") from error
    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        return None
    return base_model


def _resolve_seen10_checkpoint_paths(
    model_path: str | None,
    *,
    base_model_path: str | None,
    adapter_path: str | None,
) -> tuple[str, str | None]:
    """Resolve a base checkpoint and optional adapter checkpoint.

    ``load_seen10_model(adapter_dir)`` is supported for local PEFT outputs
    when ``adapter_config.json`` contains ``base_model_name_or_path``.  The
    explicit ``base_model_path`` argument takes precedence and is recommended
    for remote adapter identifiers or relocated checkpoints.
    """
    resolved_model = model_path
    resolved_adapter = adapter_path

    if resolved_adapter is None and resolved_model is not None:
        candidate = Path(resolved_model)
        if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
            resolved_adapter = resolved_model
            resolved_model = None

    if resolved_adapter is not None:
        if base_model_path is not None:
            resolved_model = base_model_path
        elif resolved_model is None:
            resolved_model = _adapter_base_model_path(resolved_adapter)
        if resolved_model is None:
            raise ValueError(
                "A PEFT adapter checkpoint needs base_model_path or "
                "adapter_config.json.base_model_name_or_path"
            )

    if resolved_model is None:
        raise ValueError("A base XVLA checkpoint path is required")
    return resolved_model, resolved_adapter


def _attach_seen10_adapter(model: XVLA, adapter_path: str, *, is_trainable: bool = False) -> XVLA:
    """Attach a PEFT adapter lazily, keeping the base loader dependency-light."""
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError(
            "Loading a Seen-10 adapter requires the optional `peft` dependency"
        ) from error
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)


def validate_seen10_model(model: XVLA, *, action_mode: str | None = None) -> XVLA:
    """Assert the immutable Seen-10 action/proprio model contract."""
    config = getattr(model, "config", None)
    config_mode = getattr(config, "action_mode", None)
    expected_mode = _contract_mode(action_mode=action_mode or config_mode)
    if str(config_mode).lower() != expected_mode:
        raise RuntimeError(
            "Seen-10 action mode must be persisted as "
            f"config.action_mode={expected_mode!r}, got {config_mode!r}"
        )

    action_space = getattr(model, "action_space", None)
    if (
        getattr(model, "num_actions", None) != 1
        or getattr(action_space, "real_dim", None) != SEEN10_REAL_ACTION_DIM
    ):
        raise RuntimeError("XVLA did not load with the Seen-10 horizon-one 5DoF action space")
    if getattr(action_space, "dim_action", None) != SEEN10_MODEL_ACTION_DIM:
        raise RuntimeError("Seen-10 must preserve the pretrained 20-dimensional action head")

    transformer = getattr(model, "transformer", None)
    action_encoder = getattr(transformer, "action_encoder", None)
    action_decoder = getattr(transformer, "action_decoder", None)
    if getattr(action_encoder, "input_size", None) != 72:
        raise RuntimeError(
            "Seen-10 requires a 72D action encoder input "
            "(20D action + 20D zero proprio + 32D time)"
        )
    if getattr(action_decoder, "output_size", None) != SEEN10_MODEL_ACTION_DIM:
        raise RuntimeError("The native XVLA action decoder shape changed unexpectedly")
    if getattr(model, "use_proprio", None):
        raise RuntimeError(
            "Seen-10 config must declare use_proprio=False: no real robot-state information is used; "
            "the pretrained 72D encoder still receives a structural zero proprio20 tensor"
        )
    return model


def load_seen10_model(
    model_path: str | None = None,
    *,
    local_files_only: bool = False,
    base_model_path: str | None = None,
    adapter_path: str | None = None,
    action_contract=None,
    action_mode: str | None = None,
    adapter_trainable: bool = False,
) -> XVLA:
    """Load a native or PEFT-adapted XVLA with the Seen-10 contract.

    For an adapter checkpoint, construct the base model from ``base_model_path``
    (or PEFT's local ``base_model_name_or_path`` provenance) and then attach the
    adapter.  This avoids treating an adapter-only directory as a full XVLA
    checkpoint.  Any extra bare-parameter state (for example a trainable
    ``image_projection``) must be restored by the caller's adapter save/load
    protocol after this function returns; PEFT itself cannot infer such state
    from a bare ``nn.Parameter``.
    """
    base_path, resolved_adapter = _resolve_seen10_checkpoint_paths(
        model_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    mode = _contract_mode(action_contract, action_mode)
    model = build_seen10_base_model(
        base_path,
        local_files_only=local_files_only,
        action_contract=action_contract,
        action_mode=mode,
    )
    if resolved_adapter is not None:
        model = _attach_seen10_adapter(
            model,
            resolved_adapter,
            is_trainable=adapter_trainable,
        )
    return validate_seen10_model(model, action_mode=mode)


def load_seen10_processor(model_path: str, *, local_files_only: bool = False) -> XVLAProcessor:
    """Load the published XVLA processor unchanged for native image sizing."""
    processor = XVLAProcessor.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    # The task uses first-person and radar views. The third native processor
    # slot is only padding and has no learned view-specific parameter.
    processor.num_views = 2
    return processor


def processor_size(processor: XVLAProcessor):
    """Return the configured processor size in a JSON-friendly form."""
    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, dict):
        return {str(key): int(value) for key, value in size.items() if isinstance(value, (int, float))}
    if isinstance(size, (int, float)):
        return int(size)
    return size
