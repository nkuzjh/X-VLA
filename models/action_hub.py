# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, Tuple, Dict, Type
import torch
import torch.nn as nn

# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


def pad_action_to_model_dim(
    action: torch.Tensor | None,
    real_dim: int = 5,
    model_dim: int = 20,
) -> torch.Tensor | None:
    """Pad an external action target to the model action width.

    The Seen-10 contract keeps dataset targets at ``[..., 5]`` while the
    native XVLA action encoder/decoder operates at ``[..., 20]``.  This
    helper is deliberately explicit so callers can perform that conversion
    immediately before ``XVLA.forward``.  A tensor that is already at the
    model width is returned unchanged, which is required by the denoising
    loop in ``official_auto``.

    Only the two contract widths are accepted.  Silently truncating another
    width would make a malformed batch look valid and could discard action
    channels without an error.
    """
    if action is None:
        return None
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"action must be a torch.Tensor or None, got {type(action)!r}")
    if action.ndim == 0:
        raise ValueError("action must have a final feature dimension")
    if not isinstance(real_dim, int) or not isinstance(model_dim, int):
        raise TypeError("real_dim and model_dim must be integers")
    if real_dim <= 0 or model_dim < real_dim:
        raise ValueError(
            f"Expected 0 < real_dim <= model_dim, got real_dim={real_dim}, model_dim={model_dim}"
        )

    width = action.size(-1)
    if width == model_dim:
        return action
    if width != real_dim:
        raise ValueError(
            f"Expected action width {real_dim} or {model_dim}, got {width}"
        )

    pad_shape = (*action.shape[:-1], model_dim - real_dim)
    return torch.cat((action, action.new_zeros(pad_shape)), dim=-1)


# A descriptive alias for callers that do not need to know the historical
# helper name.  Keep both names stable because the action contract is used by
# training and inference adapters outside this module.
pad_action_for_model = pad_action_to_model_dim


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # Core supervised loss
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


# =============================================================================
# Implementations
# =============================================================================
@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Joint-space layout with joints + gripper only."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(D) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        gripper_loss = self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """No preprocessing applied in AGIBOT variant."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT does not postprocess."""
        return action





@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    """
    Legacy auto action space that adapts to any action dimension.

    - Model outputs max_dim for compatibility with pretrained models
    - Loss is computed only on the first real_dim dimensions
    - Preprocess trims model-width actions to real_dim and pads them back

    This is the behavior shipped on ``main@4aed8865``.  In particular, a
    model-width denoising action has its dummy channels reset to zero at every
    step.  Keep it available for exact compatibility, while Seen-10 uses the
    ``official_auto`` implementation below.

    Args:
        real_dim: The actual action dimension from the dataset/policy feature
        max_dim: The model's output dimension for pretrained VLA compatibility
    """

    JOINTS_SCALE = 100.0

    def __init__(self, real_dim: int = 5, max_dim: int = 20):
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim  # Model-facing dimension
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad real_dim → max_dim (zeros for the dummy channels)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            # If dimension doesn't match either, pad/trim to real_dim first
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.real_dim]

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Trim model output max_dim → real_dim."""
        return x[..., : self.real_dim]

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute loss only on the first real_dim dimensions.

        pred:   [B, T, max_dim] from the model
        target: [B, T, real_dim] or [B, T, max_dim]

        Loss = MSE(pred[:,:,:real_dim], target[:,:,:real_dim])
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        # only compute loss on the real dimensions
        joints_loss = (
            self.mse(
                pred[:, :, : self.real_dim],
                target[:, :, : self.real_dim],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """
        Keep only real dimensions and pad them to the pretrained model width.

        Denoising starts from a model-width [B,T,max_dim] noise tensor, while
        supervised targets arrive as [B,T,real_dim]. Trimming before padding
        makes both paths use zero-valued dummy dimensions and keeps training
        and generation inputs identical.
        """
        action = action[..., : self.real_dim]
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        Trim model output from max_dim to real_dim for real robot control.
        """
        return self._trim_to_real_dim(action)


# Stable compatibility name for callers that want to make the legacy reset
# behavior explicit.  ``auto`` remains registered for old checkpoints and
# command lines; both names intentionally construct the same implementation.
ACTION_REGISTRY["legacy_reset_dummy"] = AutoActionSpace
LegacyResetDummyActionSpace = AutoActionSpace


@register_action("official_auto")
class OfficialAutoActionSpace(BaseActionSpace):
    """Native-width action space for an external lower-dimensional target.

    ``XVLA`` always owns a model-width action head.  For Seen-10, the dataset
    target is ``[..., 5]`` but the action encoder/decoder remain ``[..., 20]``.
    Call :func:`pad_action_to_model_dim` (or ``pad_to_model_dim`` on this
    instance) at the model boundary.  The preprocessor also accepts a raw 5D
    target as a compatibility fallback for existing callers; once an action is
    20D it is returned *unchanged*.  That identity path is what lets the
    iterative generation loop carry information in the final 15 channels.

    The supervised loss only observes the first ``real_dim`` channels, and
    postprocessing exposes those channels back to the task API.
    """

    JOINTS_SCALE = 100.0
    dim_proprio = 20

    def __init__(self, real_dim: int = 5, max_dim: int = 20):
        super().__init__()
        if not isinstance(real_dim, int) or not isinstance(max_dim, int):
            raise TypeError("real_dim and max_dim must be integers")
        if real_dim <= 0 or max_dim < real_dim:
            raise ValueError(
                f"Expected 0 < real_dim <= max_dim, got real_dim={real_dim}, max_dim={max_dim}"
            )
        self.real_dim = real_dim
        self.dim_action = max_dim
        self.mse = nn.MSELoss()

    def pad_to_model_dim(self, action: torch.Tensor | None) -> torch.Tensor | None:
        """Convert an external target to this action space's model width."""
        return pad_action_to_model_dim(
            action,
            real_dim=self.real_dim,
            model_dim=self.dim_action,
        )

    # Keep the private spelling used by the legacy implementation available
    # to adapter code that treats both Auto action spaces uniformly.
    _pad_to_model_dim = pad_to_model_dim

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return MSE on the real target channels, scaled by 100.

        ``target`` may stay at its external width (5D) or already be padded to
        model width (20D).  The trailing model-only channels are deliberately
        excluded from the loss even when they contain nonzero predictions.
        """
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("pred and target must be torch.Tensor instances")
        if pred.ndim == 0 or target.ndim == 0:
            raise ValueError("pred and target must have a final feature dimension")
        if pred.shape[:-1] != target.shape[:-1]:
            raise ValueError(
                f"pred/target leading shapes must match, got {pred.shape} and {target.shape}"
            )
        if pred.size(-1) < self.real_dim or target.size(-1) < self.real_dim:
            raise ValueError(
                f"pred and target must contain at least {self.real_dim} action channels, "
                f"got {pred.size(-1)} and {target.size(-1)}"
            )

        joints_loss = self.mse(
            pred[..., : self.real_dim],
            target[..., : self.real_dim],
        ) * self.JOINTS_SCALE
        return {"joints_loss": joints_loss}

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare an action while preserving already model-width values.

        ``XVLA.generate_actions`` feeds a 20D denoising state here on every
        iteration.  Returning it unchanged is essential: zero-padding it on
        each iteration would erase the learned dummy-channel state.
        """
        if action is None:
            raise ValueError("action must be a tensor")
        if action.size(-1) == self.dim_action:
            return proprio, action
        return proprio, self.pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Expose only the real task channels to callers."""
        if not isinstance(action, torch.Tensor) or action.ndim == 0:
            raise ValueError("action must be a tensor with a final feature dimension")
        if action.size(-1) < self.real_dim:
            raise ValueError(
                f"action must contain at least {self.real_dim} channels, got {action.size(-1)}"
            )
        return action[..., : self.real_dim]



# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "pad_action_to_model_dim",
    "pad_action_for_model",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "AutoActionSpace",
    "LegacyResetDummyActionSpace",
    "OfficialAutoActionSpace",
    "ACTION_REGISTRY",
]
