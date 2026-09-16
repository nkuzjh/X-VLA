"""Continuous track metrics copied from UniLIP's CSGO metric path."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from fvd_metric import compute_fvd as compute_fvd_metric
from metrics_images import PairedImageDataset


CLIP_LENGTH = 16
CLIP_STRIDE = 16
FVD_SIZE = 224
FRAME_DIFF_THRESHOLD = 2
MIN_TRACK_LEN = 4


def validate_tracks(tracks: Sequence[Mapping[str, Any]]) -> None:
    """Check the fixed v2 clip invariants without reordering/regrouping rows."""

    for track in tracks:
        rows = track["rows"]
        if len(rows) < MIN_TRACK_LEN:
            raise ValueError(f"Continuous clip is shorter than min_track_len={MIN_TRACK_LEN}: {track['clip_id']}")
        if len(rows) != 64:
            raise ValueError(f"Continuous v2 clips must contain 64 frames: {track['clip_id']}")
        file_num = rows[0]["file_num"]
        previous = rows[0]["frame_id"]
        for row in rows[1:]:
            if row["file_num"] != file_num or not 0 < row["frame_id"] - previous <= FRAME_DIFF_THRESHOLD:
                raise ValueError(f"Clip violates frame gap threshold {FRAME_DIFF_THRESHOLD}: {track['clip_id']}")
            previous = row["frame_id"]


def load_rgb_uint8(path: str | Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def load_track_pair(
    track: Mapping[str, Any],
    pred_root: str | Path,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    gt_frames = []
    pred_frames = []
    pred_root = Path(pred_root)
    for row in track["rows"]:
        pred_path = pred_root / row["map_name"] / f"{row['file_frame']}.jpg"
        gt_frames.append(load_rgb_uint8(row["image_path"], size))
        pred_frames.append(load_rgb_uint8(pred_path, size))
    return np.stack(gt_frames, axis=0), np.stack(pred_frames, axis=0)


def tensor_from_uint8(frames: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0).to(device)


def compute_temporal_metrics(
    tracks: Sequence[Mapping[str, Any]],
    pred_root: str | Path,
    *,
    size: int,
) -> dict[str, float]:
    """Compute TWE/TDE; average adjacent-frame values per track, then per map."""

    validate_tracks(tracks)
    track_warping: list[float] = []
    track_difference: list[float] = []
    for track in tracks:
        gt_uint8, pred_uint8 = load_track_pair(track, pred_root, size)
        warping_errors = []
        difference_errors = []
        for idx in range(len(track["rows"]) - 1):
            gt_prev, gt_next = gt_uint8[idx], gt_uint8[idx + 1]
            pred_prev, pred_next = pred_uint8[idx], pred_uint8[idx + 1]

            gt_delta = gt_next.astype(np.float32) - gt_prev.astype(np.float32)
            pred_delta = pred_next.astype(np.float32) - pred_prev.astype(np.float32)
            difference_errors.append(float(np.abs(pred_delta - gt_delta).mean()))

            prev_gray = cv2.cvtColor(gt_prev, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(gt_next, cv2.COLOR_RGB2GRAY)
            flow_gt = cv2.calcOpticalFlowFarneback(
                prev_gray,
                next_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            height, width = gt_prev.shape[:2]
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            map_x = (grid_x - flow_gt[..., 0]).astype(np.float32)
            map_y = (grid_y - flow_gt[..., 1]).astype(np.float32)
            pred_prev_warped = cv2.remap(
                pred_prev,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            warping_errors.append(
                float(np.abs(pred_prev_warped.astype(np.float32) - pred_next.astype(np.float32)).mean())
            )
        track_warping.append(float(np.mean(warping_errors)))
        track_difference.append(float(np.mean(difference_errors)))

    return {
        "TWE": float(np.mean(track_warping)),
        "TDE": float(np.mean(track_difference)),
    }


def compute_frame_quality_metrics(
    gt_paths: Sequence[str | Path],
    pred_paths: Sequence[str | Path],
    *,
    size: int,
    batch_size: int,
    device: str,
    num_workers: int = 4,
) -> dict[str, float]:
    """PSNR/SSIM/LPIPS from the same paired-image implementations as discrete."""

    from metrics_images import compute_lpips, compute_psnr_ssim

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
    return {"PSNR": float(psnr), "SSIM": float(ssim), "LPIPS": float(lpips)}


def build_fvd_clips(
    tracks: Sequence[Mapping[str, Any]],
    *,
    clip_length: int = CLIP_LENGTH,
    clip_stride: int = CLIP_STRIDE,
) -> list[list[Mapping[str, Any]]]:
    clips = []
    for track in tracks:
        rows = track["rows"]
        if len(rows) < clip_length:
            continue
        for start_idx in range(0, len(rows) - clip_length + 1, clip_stride):
            clips.append(list(rows[start_idx : start_idx + clip_length]))
    return clips


class VideoClipDataset(Dataset):
    def __init__(self, rows_by_clip: Sequence[Sequence[Mapping[str, Any]]], pred_root: str | Path, *, size: int, predicted: bool):
        self.rows_by_clip = [list(clip) for clip in rows_by_clip]
        self.pred_root = Path(pred_root)
        self.predicted = predicted
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.PILToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows_by_clip)

    def __getitem__(self, index: int) -> torch.Tensor:
        frames = []
        for row in self.rows_by_clip[index]:
            if self.predicted:
                path = self.pred_root / row["map_name"] / f"{row['file_frame']}.jpg"
            else:
                path = Path(row["image_path"])
            image = Image.open(path).convert("RGB")
            frames.append(self.transform(image))
        return torch.stack(frames, dim=1)


def compute_fvd_for_tracks(
    tracks: Sequence[Mapping[str, Any]],
    pred_root: str | Path,
    *,
    clip_length: int = CLIP_LENGTH,
    clip_stride: int = CLIP_STRIDE,
    size: int = FVD_SIZE,
    batch_size: int = 1,
    device: str,
    num_workers: int = 4,
) -> tuple[float, int]:
    validate_tracks(tracks)
    clips = build_fvd_clips(tracks, clip_length=clip_length, clip_stride=clip_stride)
    if not clips:
        raise ValueError("No clips available for FVD")
    fvd_batch_size = max(1, batch_size // 4)
    gt_dataset = VideoClipDataset(clips, pred_root, size=size, predicted=False)
    pred_dataset = VideoClipDataset(clips, pred_root, size=size, predicted=True)
    gt_loader = DataLoader(gt_dataset, batch_size=fvd_batch_size, num_workers=num_workers)
    pred_loader = DataLoader(pred_dataset, batch_size=fvd_batch_size, num_workers=num_workers)
    gt_clips = []
    pred_clips = []
    with torch.no_grad():
        for batch in gt_loader:
            gt_clips.append(batch.cpu())
        for batch in pred_loader:
            pred_clips.append(batch.cpu())
    gt_tensor = torch.cat(gt_clips, dim=0)
    pred_tensor = torch.cat(pred_clips, dim=0)
    score = compute_fvd_metric(
        gt_tensor,
        pred_tensor,
        max_items=len(clips),
        device=device,
        batch_size=fvd_batch_size,
    )
    if isinstance(score, torch.Tensor):
        score = score.item()
    del gt_tensor, pred_tensor
    gc.collect()
    return float(score), len(clips)


__all__ = [
    "CLIP_LENGTH",
    "CLIP_STRIDE",
    "FVD_SIZE",
    "FRAME_DIFF_THRESHOLD",
    "MIN_TRACK_LEN",
    "compute_temporal_metrics",
    "compute_frame_quality_metrics",
    "compute_fvd_for_tracks",
]
