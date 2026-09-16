"""Thin CSGO localization adapter for the native XVLA model and processor."""

from __future__ import annotations

from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


SEEN10_MODEL_CONFIG = {
    "action_mode": "auto",
    "real_action_dim": 5,
    "max_action_dim": 20,
    "num_actions": 1,
    "use_proprio": False,
}


def configure_seen10(config: XVLAConfig) -> XVLAConfig:
    """Apply Seen-10 task settings without changing the pretrained head shape.

    The auto action space computes loss and returns predictions over the first
    five dimensions while keeping the native 20-dimensional action decoder.
    """
    for name, value in SEEN10_MODEL_CONFIG.items():
        setattr(config, name, value)
    return config


def load_seen10_model(model_path: str, *, local_files_only: bool = False) -> XVLA:
    """Load XVLA weights with the horizon-one, normalized 5DoF task config."""
    config = XVLAConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    configure_seen10(config)
    model = XVLA.from_pretrained(
        model_path,
        config=config,
        local_files_only=local_files_only,
    )
    if model.num_actions != 1 or model.action_space.real_dim != 5:
        raise RuntimeError("XVLA did not load with the Seen-10 horizon-one 5DoF action space")
    if model.action_space.dim_action != 20:
        raise RuntimeError("Seen-10 must preserve the pretrained 20-dimensional action head")
    if model.transformer.action_decoder.output_size != 20:
        raise RuntimeError("The native XVLA action decoder shape changed unexpectedly")
    if model.use_proprio:
        raise RuntimeError("Seen-10 must disable proprioception")
    return model


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
