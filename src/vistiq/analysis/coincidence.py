import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import Field, field_validator
from prefect import task
from skimage.measure import regionprops, regionprops_table

from vistiq.core import (
    ChainProcessor,
    ChainProcessorConfig,
    Configurable,
    Configuration,
    StackProcessor,
    StackProcessorConfig,
    labels_to_masks,
)
from vistiq.utils import (
    ArrayIterator,
    ArrayIteratorConfig,
    _normalize_stack_names,
    create_unique_folder,
    resolve_torch_device,
)
from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig
from vistiq.workflow import Workflow

logger = logging.getLogger(__name__)

# If bbox pruning leaves at least this fraction of pairs, use dense mask batch instead.
_LABELS_IOU_DENSE_PAIR_FRACTION = 1.01


def _label_ids_and_boxes(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segmentation label ids and axis-aligned boxes.

    Uses integer ``label`` values from :class:`RegionAnalyzer` (not ``object_id``,
    which is a hex string). Boxes are ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
    """
    labels = np.asarray(labels)

    ra = RegionAnalyzer(
        RegionAnalyzerConfig(
            properties=["label", "bbox"],
            output_type="dataframe",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    table = ra.run(labels)

    if len(table) == 0:
        return np.array([], dtype=np.int64), np.empty(
            (0, 2 * labels.ndim), dtype=np.float32
        )
    if "label" in table.columns:
        label_ids = table["label"].astype(np.int64, copy=False)
    elif table.index.name == "label":
        label_ids = table.index.astype(np.int64, copy=False)
    else:
        raise ValueError(
            "Label DataFrame must have a 'label' column or index named 'label'"
        )
    if labels.ndim == 3:
        boxes = np.column_stack(
            (
                table["bbox-2"],
                table["bbox-1"],
                table["bbox-0"],
                table["bbox-5"],
                table["bbox-4"],
                table["bbox-3"],
            )
        )
    elif labels.ndim == 2:
        # skimage 2D bbox: (min_row, min_col, max_row, max_col) -> x, y, z slab
        n = len(label_ids)
        boxes = np.column_stack(
            (
                table["bbox-1"],
                table["bbox-0"],
                np.zeros(n, dtype=np.float32),
                table["bbox-3"],
                table["bbox-2"],
                np.ones(n, dtype=np.float32),
            )
        )
    else:
        raise ValueError(
            f"Label arrays must be 2D or 3D for region bounding boxes; got {labels.ndim}D"
        )
    return label_ids, boxes.astype(np.float32, copy=False)


def _positive_label_ids(labels: np.ndarray) -> np.ndarray:
    """Sorted positive label ids (``np.unique`` order, dense-batch alignment)."""
    ids = np.unique(np.asarray(labels))
    return ids[ids > 0].astype(np.int64, copy=False)


def _label_ids_for_overlap_matrix(
    labels: np.ndarray, *, used_pruned_path: bool
) -> np.ndarray:
    """Label ids aligned with rows/cols of :func:`labels_iou_batch_3d` output."""
    if used_pruned_path:
        ids, _ = _label_ids_and_boxes(labels)
        return ids
    return _positive_label_ids(labels)


def _mask_iou_labels_pair(
    labels_a: np.ndarray,
    label_a: int,
    labels_b: np.ndarray,
    label_b: int,
    overlap_metric: Literal["iou", "ios", "dice"],
) -> float:
    """Mask IOU/IOS/Dice for one label id in each volume (same spatial shape)."""
    mask_a = labels_a == label_a
    mask_b = labels_b == label_b
    inter = float(np.logical_and(mask_a, mask_b).sum())
    area_a = float(mask_a.sum())
    area_b = float(mask_b.sum())
    metric = overlap_metric.lower()
    if metric == "iou":
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union
    if metric == "ios":
        denom = min(area_a, area_b)
        return 0.0 if denom <= 0 else inter / denom
    if metric == "dice":
        denom = area_a + area_b
        return 0.0 if denom <= 0 else 2 * inter / denom
    raise ValueError(
        f"overlap_metric {overlap_metric!r} is not supported, "
        "only 'iou', 'ios', and 'dice' are supported"
    )


def _labels_iou_batch_3d_dense(
    labels_true: np.ndarray,
    labels_detection: np.ndarray,
    overlap_metric: Literal["iou", "ios", "dice"],
    memory_limit: int,
) -> np.ndarray[np.float32]:
    """Full mask batch path (materializes all instance masks)."""
    masks_true = labels_to_masks(labels_true)
    masks_detection = labels_to_masks(labels_detection)
    if masks_true.ndim == 3:
        masks_true = masks_true[:, np.newaxis, ...]
        masks_detection = masks_detection[:, np.newaxis, ...]
    return mask_iou_batch_3d(
        masks_true, masks_detection, overlap_metric, memory_limit=memory_limit
    )


def _mask_iou_batch_3d_split(
    masks_true: np.typing.NDArray[np.number],
    masks_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"],
) -> np.ndarray[np.float32]:
    """Pairwise overlap for one chunk of true masks vs all detection masks."""
    count_true, count_det = masks_true.shape[0], masks_detection.shape[0]
    if count_true == 0 or count_det == 0:
        return np.empty((count_true, count_det), dtype=np.float32)

    true_flat = np.reshape(masks_true, (count_true, -1))
    det_flat = np.reshape(masks_detection, (count_det, -1))
    area_true = true_flat.sum(axis=1, dtype=np.float64)
    area_det = det_flat.sum(axis=1, dtype=np.float64)
    intersection = true_flat.astype(np.float32) @ det_flat.astype(np.float32).T

    metric = overlap_metric.lower()
    if metric == "iou":
        numer = intersection
        area_norm = area_true[:, None] + area_det[None, :] - intersection
    elif metric == "ios":
        numer = intersection
        area_norm = np.minimum(area_true[:, None], area_det[None, :])
    elif metric == "dice":
        numer = 2 * intersection
        area_norm = area_true[:, None] + area_det[None, :]
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric!r} is not supported, "
            "only 'iou', 'ios', and 'dice' are supported"
        )

    out: np.ndarray[np.float32] = np.zeros_like(intersection, dtype=np.float32)
    np.divide(numer, area_norm, out=out, where=area_norm > 0)
    return out


def mask_iou_batch_3d(
    masks_true: np.typing.NDArray[np.number],
    masks_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"] = "iou",
    memory_limit: int = 1024 * 5,
) -> np.ndarray[np.float32]:
    """Compute pairwise overlap between batches of 3D binary masks.

    Each mask is a boolean (or 0/1) volume. Supports IOU, IOS, and Dice,
    matching :func:`box_iou_batch_3d`.

    Args:
        masks_true: Ground-truth masks, shape ``(N, Z, Y, X)``.
        masks_detection: Detection masks, same spatial shape as ``masks_true``.
        overlap_metric: ``"iou"``, ``"ios"``, or ``"dice"``.
        memory_limit: Chunk size budget in MB for the pairwise reduction.

    Returns:
        Overlap matrix of shape ``(N, M)`` with values in ``[0, 1]``.
    """
    masks_true = np.asarray(masks_true)
    masks_detection = np.asarray(masks_detection)
    if masks_true.ndim != 4 or masks_detection.ndim != 4:
        raise ValueError(
            "masks_true and masks_detection must be 4D arrays "
            f"(N, Z, Y, X); got {masks_true.ndim}D and {masks_detection.ndim}D"
        )
    if masks_true.shape[1:] != masks_detection.shape[1:]:
        raise ValueError(
            "Spatial shapes must match: "
            f"{masks_true.shape[1:]} vs {masks_detection.shape[1:]}"
        )

    count_true, count_det = masks_true.shape[0], masks_detection.shape[0]
    if count_true == 0 or count_det == 0:
        return np.empty((count_true, count_det), dtype=np.float32)

    voxels_per_pair = int(np.prod(masks_true.shape[1:]))
    memory_mb = count_true * voxels_per_pair * count_det / 1024 / 1024
    if memory_mb <= memory_limit:
        return _mask_iou_batch_3d_split(
            masks_true, masks_detection, overlap_metric
        )

    ious: list[np.ndarray[np.float32]] = []
    step = max(
        memory_limit * 1024 * 1024 // (count_det * voxels_per_pair),
        1,
    )
    for start in range(0, count_true, step):
        ious.append(
            _mask_iou_batch_3d_split(
                masks_true[start : start + step],
                masks_detection,
                overlap_metric,
            )
        )
    return np.vstack(ious).astype(np.float32, copy=False)


def _labels_to_masks_torch(labels: Any) -> Any:
    """One binary mask per positive label id, shape ``(N, *spatial)``."""
    ids = torch.unique(labels)
    ids = ids[ids > 0]
    if ids.numel() == 0:
        return labels.new_empty((0,) + labels.shape, dtype=torch.bool)
    return labels == ids.view(-1, *([1] * labels.ndim))


def _mask_iou_batch_3d_split_torch(
    masks_true: Any,
    masks_detection: Any,
    overlap_metric: Literal["iou", "ios", "dice"],
) -> Any:
    """Pairwise overlap on PyTorch tensors (one chunk of true masks vs all detections)."""
    count_true, count_det = masks_true.shape[0], masks_detection.shape[0]
    if count_true == 0 or count_det == 0:
        return torch.empty((count_true, count_det), dtype=torch.float32, device=masks_true.device)

    true_flat = masks_true.reshape(count_true, -1).to(dtype=torch.float32)
    det_flat = masks_detection.reshape(count_det, -1).to(dtype=torch.float32)
    area_true = true_flat.sum(dim=1, dtype=torch.float64)
    area_det = det_flat.sum(dim=1, dtype=torch.float64)
    intersection = true_flat @ det_flat.T

    if overlap_metric == "iou":
        numer = intersection
        area_norm = area_true[:, None] + area_det[None, :] - intersection
    elif overlap_metric == "ios":
        numer = intersection
        area_norm = torch.minimum(area_true[:, None], area_det[None, :])
    elif overlap_metric == "dice":
        numer = 2 * intersection
        area_norm = area_true[:, None] + area_det[None, :]
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric!r} is not supported, "
            "only 'IOU', 'IOS', and 'DICE' are supported"
        )

    return torch.where(
        area_norm > 0,
        numer / area_norm,
        torch.zeros_like(intersection),
    ).to(dtype=torch.float32)


def mask_iou_batch_3d_torch(
    masks_true: np.typing.NDArray[np.number],
    masks_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"] = "iou",
    memory_limit: int = 1024 * 5,
    device: Union[str, int, Any, None] = None,
) -> np.ndarray[np.float32]:
    """PyTorch counterpart of :func:`mask_iou_batch_3d`.

    Accepts NumPy or torch mask batches ``(N, Z, Y, X)``; returns a NumPy overlap
    matrix on the host. Uses :func:`~vistiq.utils.check_device` when ``device`` is
    omitted (CUDA, then MPS, then CPU).
    """
    torch_device = resolve_torch_device(device, preferred_input_type="torch.Tensor")
    masks_true_t = torch.as_tensor(masks_true, device=torch_device)
    masks_detection_t = torch.as_tensor(masks_detection, device=torch_device)
    if masks_true_t.ndim != 4 or masks_detection_t.ndim != 4:
        raise ValueError(
            "masks_true and masks_detection must be 4D arrays "
            f"(N, Z, Y, X); got {masks_true_t.ndim}D and {masks_detection_t.ndim}D"
        )
    if masks_true_t.shape[1:] != masks_detection_t.shape[1:]:
        raise ValueError(
            "Spatial shapes must match: "
            f"{masks_true_t.shape[1:]} vs {masks_detection_t.shape[1:]}"
        )

    count_true, count_det = masks_true_t.shape[0], masks_detection_t.shape[0]
    if count_true == 0 or count_det == 0:
        return np.empty((count_true, count_det), dtype=np.float32)

    voxels_per_pair = int(np.prod(masks_true_t.shape[1:]))
    memory_mb = count_true * voxels_per_pair * count_det / 1024 / 1024
    if memory_mb <= memory_limit:
        out_t = _mask_iou_batch_3d_split_torch(
            masks_true_t, masks_detection_t, overlap_metric
        )
        return out_t.detach().cpu().numpy().astype(np.float32, copy=False)

    ious_t: list[Any] = []
    step = max(
        memory_limit * 1024 * 1024 // (count_det * voxels_per_pair),
        1,
    )
    for start in range(0, count_true, step):
        ious_t.append(
            _mask_iou_batch_3d_split_torch(
                masks_true_t[start : start + step],
                masks_detection_t,
                overlap_metric,
            )
        )
    return torch.vstack(ious_t).detach().cpu().numpy().astype(np.float32, copy=False)


def _mask_iou_labels_pair_torch(
    labels_a: Any,
    label_a: int,
    labels_b: Any,
    label_b: int,
    overlap_metric: Literal["iou", "ios", "dice"],
) -> float:
    """Mask IOU/IOS/Dice for one label id pair on a torch device."""
    mask_a = labels_a == label_a
    mask_b = labels_b == label_b
    inter = float(torch.logical_and(mask_a, mask_b).sum())
    area_a = float(mask_a.sum())
    area_b = float(mask_b.sum())
    metric = overlap_metric.lower()
    if metric == "iou":
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union
    if metric == "ios":
        denom = min(area_a, area_b)
        return 0.0 if denom <= 0 else inter / denom
    if metric == "dice":
        denom = area_a + area_b
        return 0.0 if denom <= 0 else 2 * inter / denom
    raise ValueError(
        f"overlap_metric {overlap_metric!r} is not supported, "
        "only 'iou', 'ios', and 'dice' are supported"
    )


def _labels_iou_batch_3d_dense_torch(
    labels_true_t: Any,
    labels_detection_t: Any,
    overlap_metric: Literal["iou", "ios", "dice"],
    memory_limit: int,
    device: Union[str, int, Any, None],
) -> np.ndarray[np.float32]:
    """Dense mask-batch path on a torch device."""
    masks_true = _labels_to_masks_torch(labels_true_t)
    masks_detection = _labels_to_masks_torch(labels_detection_t)
    if masks_true.ndim == 3:
        masks_true = masks_true.unsqueeze(1)
        masks_detection = masks_detection.unsqueeze(1)
    return mask_iou_batch_3d_torch(
        masks_true,
        masks_detection,
        overlap_metric,
        memory_limit=memory_limit,
        device=labels_true_t.device,
    )


def labels_iou_batch_3d_torch(
    labels_true: np.typing.NDArray[np.number],
    labels_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"] = "iou",
    prune_bboxes: bool = True,
    dense_pair_fraction: float = _LABELS_IOU_DENSE_PAIR_FRACTION,
    memory_limit: int = 1024 * 5,
    device: Union[str, int, Any, None] = None,
) -> np.ndarray[np.float32]:
    """PyTorch counterpart of :func:`labels_iou_batch_3d`.

    Label volumes are transferred to ``device`` (default from :func:`~vistiq.utils.check_device`).
    Bounding boxes use CPU :func:`skimage.measure.regionprops_table`; mask overlap runs on
    the torch device with the same bbox-prune / dense-fallback strategy as the CPU function.

    Args:
        labels_true: Reference label volume (3D ``Z, Y, X`` for pruning path).
        labels_detection: Detection label volume, same shape as ``labels_true``.
        overlap_metric: ``"IOU"`` or ``"IOS"``.
        prune_bboxes: If True, bbox-prune before per-pair mask overlap.
        dense_pair_fraction: Dense mask batch when bbox-overlap fraction is at or
            above this threshold.
        memory_limit: Chunk budget (MB) for :func:`mask_iou_batch_3d_torch`.
        device: ``torch.device``, device string (e.g. ``"cuda:0"``, ``"cpu"``), or
            ``None`` for automatic selection.

    Returns:
        Host NumPy matrix ``(N, M)`` aligned with regionprops label order.
    """
    torch_device = resolve_torch_device(device, preferred_input_type="torch.Tensor")
    labels_true = np.asarray(labels_true)
    labels_detection = np.asarray(labels_detection)
    if labels_true.shape != labels_detection.shape:
        raise ValueError(
            "labels_true and labels_detection must have the same shape: "
            f"{labels_true.shape} vs {labels_detection.shape}"
        )

    labels_true_t = torch.as_tensor(labels_true, dtype=torch.long, device=torch_device)
    labels_detection_t = torch.as_tensor(
        labels_detection, dtype=torch.long, device=torch_device
    )

    if not prune_bboxes or labels_true.ndim != 3:
        return _labels_iou_batch_3d_dense_torch(
            labels_true_t,
            labels_detection_t,
            overlap_metric,
            memory_limit,
            device=torch_device,
        )

    ids_true, boxes_true = _label_ids_and_boxes(labels_true)
    ids_det, boxes_det = _label_ids_and_boxes(labels_detection)
    n_true, n_det = len(ids_true), len(ids_det)
    out = np.zeros((n_true, n_det), dtype=np.float32)
    if n_true == 0 or n_det == 0:
        return out

    bbox_overlap = box_iou_batch_3d(boxes_true, boxes_det, overlap_metric)
    candidate_ij = np.argwhere(bbox_overlap > 0)
    n_pairs = n_true * n_det
    if len(candidate_ij) >= dense_pair_fraction * n_pairs:
        logger.info(
            "labels_iou_batch_3d_torch: %d/%d pairs overlap in bbox space, using dense masks",
            len(candidate_ij),
            n_pairs,
        )
        return _labels_iou_batch_3d_dense_torch(
            labels_true_t,
            labels_detection_t,
            overlap_metric,
            memory_limit,
            device=torch_device,
        )

    for i, j in candidate_ij:
        out[i, j] = _mask_iou_labels_pair_torch(
            labels_true_t,
            int(ids_true[i]),
            labels_detection_t,
            int(ids_det[j]),
            overlap_metric,
        )
    logger.info(
        "labels_iou_batch_3d_torch: mask IOU for %d/%d pairs after bbox prune",
        len(candidate_ij),
        n_pairs,
    )
    return out


def labels_iou_batch_3d(
    labels_true: np.typing.NDArray[np.number],
    labels_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"] = "iou",
    prune_bboxes: bool = True,
    dense_pair_fraction: float = _LABELS_IOU_DENSE_PAIR_FRACTION,
    memory_limit: int = 1024 * 5,
) -> np.ndarray[np.float32]:
    """Compute pairwise mask IOU/IOS between label volumes (one row/column per label id).

    When ``prune_bboxes`` is True (default), uses tight 3D bounding boxes from
    :func:`skimage.measure.regionprops_table` and :func:`box_iou_batch_3d` to skip
    pairs with non-overlapping boxes, then computes mask overlap only for
    surviving pairs. Avoids building full ``(N, Z, Y, X)`` mask stacks when
    layouts are sparse. Falls back to the dense mask-batch path when most pairs
    overlap in bbox space.

    Args:
        labels_true: Reference label volume (3D ``Z, Y, X`` for pruning path).
        labels_detection: Detection label volume, same shape as ``labels_true``.
        overlap_metric: ``"iou"``, ``"ios"``, or ``"dice"``.
        prune_bboxes: If True, bbox-prune before per-pair mask overlap.
        dense_pair_fraction: Use dense mask batch when fraction of bbox-overlapping
            pairs is at or above this threshold (avoids slow per-pair loops).
        memory_limit: Passed to :func:`mask_iou_batch_3d` on dense fallback.

    Returns:
        Matrix of shape ``(N, M)`` aligned with regionprops label order.
    """
    labels_true = np.asarray(labels_true)
    labels_detection = np.asarray(labels_detection)
    if labels_true.shape != labels_detection.shape:
        raise ValueError(
            "labels_true and labels_detection must have the same shape: "
            f"{labels_true.shape} vs {labels_detection.shape}"
        )

    if not prune_bboxes or labels_true.ndim != 3:
        return _labels_iou_batch_3d_dense(
            labels_true, labels_detection, overlap_metric, memory_limit
        )

    ids_true, boxes_true = _label_ids_and_boxes(labels_true)
    ids_det, boxes_det = _label_ids_and_boxes(labels_detection)
    n_true, n_det = len(ids_true), len(ids_det)
    out = np.zeros((n_true, n_det), dtype=np.float32)
    if n_true == 0 or n_det == 0:
        return out

    bbox_overlap = box_iou_batch_3d(boxes_true, boxes_det, overlap_metric)
    candidate_ij = np.argwhere(bbox_overlap > 0)
    n_pairs = n_true * n_det
    if len(candidate_ij) >= dense_pair_fraction * n_pairs:
        logger.info(
            "labels_iou_batch_3d: %d/%d pairs overlap in bbox space, using dense masks",
            len(candidate_ij),
            n_pairs,
        )
        return _labels_iou_batch_3d_dense(
            labels_true, labels_detection, overlap_metric, memory_limit
        )

    for i, j in candidate_ij:
        out[i, j] = _mask_iou_labels_pair(
            labels_true,
            ids_true[i],
            labels_detection,
            ids_det[j],
            overlap_metric,
        )
    logger.info(
        "labels_iou_batch_3d: mask IOU for %d/%d pairs after bbox prune",
        len(candidate_ij),
        n_pairs,
    )
    return out


def box_iou_batch_3d_torch(
    boxes_true: np.typing.NDArray[np.number],
    boxes_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"] = "iou",
    device: Union[str, int, Any, None] = None,
) -> np.ndarray[np.float32]:
    """PyTorch-backed pairwise box overlap; returns a host NumPy matrix."""
    from vistiq.analysis.overlap import (
        BoxOverlapCalculatorConfig,
        OverlapCalculator,
        metrics_calculator_configs,
    )

    calc = OverlapCalculator(
        BoxOverlapCalculatorConfig(
            metrics_calculators=metrics_calculator_configs(
                (overlap_metric.lower(),)
            ),
            preferred_input_type="torch.Tensor",
            output_type="np.ndarray",
        )
    )
    result = calc.run(boxes_true, boxes_detection, device=device)
    return np.asarray(result.metric(), dtype=np.float32)


def box_iou_batch_3d(
    boxes_true: np.typing.NDArray[np.number],
    boxes_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["iou", "ios", "dice"] = "iou",
) -> np.ndarray[np.float32]:
    """Pairwise iou/ios/dice between batches of 3D axis-aligned boxes.

    Boxes are ``(N, 6)`` as ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
    Adapted from
    https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/utils/iou_and_nms.py
    """
    x_min_true, y_min_true, z_min_true, x_max_true, y_max_true, z_max_true = boxes_true.T
    x_min_det, y_min_det, z_min_det, x_max_det, y_max_det, z_max_det = boxes_detection.T
    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]

    if count_true == 0 or count_det == 0:
        return np.empty((count_true, count_det), dtype=np.float32)

    x_min_inter = np.empty((count_true, count_det), dtype=np.float32)
    x_max_inter = np.empty_like(x_min_inter)
    y_min_inter = np.empty_like(x_min_inter)
    y_max_inter = np.empty_like(x_min_inter)
    z_min_inter = np.empty_like(x_min_inter)
    z_max_inter = np.empty_like(x_min_inter)

    np.maximum(x_min_true[:, None], x_min_det[None, :], out=x_min_inter)
    np.minimum(x_max_true[:, None], x_max_det[None, :], out=x_max_inter)
    np.maximum(y_min_true[:, None], y_min_det[None, :], out=y_min_inter)
    np.minimum(y_max_true[:, None], y_max_det[None, :], out=y_max_inter)
    np.maximum(z_min_true[:, None], z_min_det[None, :], out=z_min_inter)
    np.minimum(z_max_true[:, None], z_max_det[None, :], out=z_max_inter)

    np.subtract(x_max_inter, x_min_inter, out=x_max_inter)
    np.subtract(y_max_inter, y_min_inter, out=y_max_inter)
    np.subtract(z_max_inter, z_min_inter, out=z_max_inter)
    np.clip(x_max_inter, 0.0, None, out=x_max_inter)
    np.clip(y_max_inter, 0.0, None, out=y_max_inter)
    np.clip(z_max_inter, 0.0, None, out=z_max_inter)

    area_inter = x_max_inter * y_max_inter * z_max_inter

    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true) * (z_max_true - z_min_true)
    area_det = (x_max_det - x_min_det) * (y_max_det - y_min_det) * (z_max_det - z_min_det)

    metric = overlap_metric.lower()
    if metric == "iou":
        numer = area_inter
        area_norm = area_true[:, None] + area_det[None, :] - area_inter
    elif metric == "ios":
        numer = area_inter
        area_norm = np.minimum(area_true[:, None], area_det[None, :])
    elif metric == "dice":
        numer = 2 * area_inter
        area_norm = area_true[:, None] + area_det[None, :]
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric!r} is not supported, "
            "only 'iou', 'ios', and 'dice' are supported"
        )

    out: np.ndarray[np.float32] = np.zeros_like(area_inter, dtype=np.float32)
    np.divide(numer, area_norm, out=out, where=area_norm > 0)
    return out


class CoincidenceDetectorConfig(StackProcessorConfig):
    """Configuration for coincidence detection workflow.
    
    Attributes:
        output_type: Output type ("list" or "stack").
        output: Output fields ("score" or "above_threshold").
        method: Overlap method to use ("iou" or "dice").
        mode: Overlap mode ("box" or "strict").
        threshold: Threshold for the overlap score (must be between 0.0 and 1.0).
    """
    output_type: Literal["list"] = Field(default="list", description="Output type")
    output: List[Literal["score", "above_threshold"]] = Field(default=["score", "above_threshold"], description="Output fields")
    method: Literal["iou", "dice", "ios"] = Field(default="iou", description="Overlap method")
    mode: Literal["bounding_box", "outline"] = Field(default="outline", description="Overlap mode")
    threshold: float = Field(default=0.5, description="Threshold for the overlap score")
    
    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate that threshold is between 0.0 and 1.0.
        
        Args:
            v: Threshold value to validate.
            
        Returns:
            Validated threshold value.
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {v}")
        return v


class CoincidenceDetector(StackProcessor):
    """Detector that computes the coincidence/overlap between two labeled imagestacks.

    Args:
        config: Configuration for the coincidence detector.
        
    """
    
    def __init__(self, config: CoincidenceDetectorConfig):
        super().__init__(config)

    def _ios(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute the Intersection Over Similarity (IoS) between two binary masks.
        
        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            
        Returns:
            IoS score between 0.0 and 1.0.
        """
        intersection = np.sum(mask1 & mask2)
        denom = min(np.sum(mask1), np.sum(mask2))
        if denom == 0:
            return 0.0
        return float(intersection / denom)

    def _iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute the Intersection Over Union (IoU) between two binary masks. It's equivalent to the Jaccard index.

        Formula:
            IoU = intersection / union
            intersection = sum(mask1 & mask2)
            union = sum(mask1 | mask2)

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            
        Returns:
            IoU score between 0.0 and 1.0.
        """
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _dice(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute the Dice coefficient between two binary masks. It's equivalent to the F1 score.

        Formula:
            Dice = 2 * intersection / (sum(mask1) + sum(mask2))
            intersection = sum(mask1 & mask2)
            sum_masks = sum(mask1) + sum(mask2)

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            
        Returns:
            Dice coefficient between 0.0 and 1.0.
        """
        intersection = np.sum(mask1 & mask2)
        sum_masks = np.sum(mask1) + np.sum(mask2)
        if sum_masks == 0:
            return 0.0
        return float(2 * intersection / sum_masks)
    
    def _bbox_to_mask(self, bbox: Tuple, shape: Tuple) -> np.ndarray:
        """Create a binary mask from a bounding box.
        
        Args:
            bbox: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                  For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X) in Z-Y-X order.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (Z, Y, X) or (height, width) if using 2D projection.
            
        Returns:
            Binary mask with ones in the bounding box region.
        """
        # Handle both 2D (4 values) and 3D (6 values) bounding boxes
        if len(bbox) == 6:
            # 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            min_Z, min_Y, min_X, max_Z, max_Y, max_X = bbox
            # Check if shape is 3D or 2D
            if len(shape) == 3:
                # Full 3D mask: shape is (Z, Y, X)
                mask = np.zeros(shape, dtype=bool)
                mask[min_Z:max_Z, min_Y:max_Y, min_X:max_X] = True
            else:
                # 2D projection - use only Y and X dimensions
                mask = np.zeros(shape, dtype=bool)
                mask[min_Y:max_Y, min_X:max_X] = True
        else:
            # 2D: (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = bbox
            # Create mask with shape of the full image
            mask = np.zeros(shape, dtype=bool)
            # Set the bounding box region to True
            mask[min_row:max_row, min_col:max_col] = True
        return mask
    
    def _ios_box(self, bbox1: Tuple, bbox2: Tuple, shape: Tuple) -> float:
        """Compute IoS between two bounding boxes by creating masks and using _ios.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice).
            bbox2: Bounding box. Same format as bbox1.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (depth, height, width) or (height, width) if using 2D projection.
            
        Returns:
            IoS score between 0.0 and 1.0.
        """
        mask1 = self._bbox_to_mask(bbox1, shape)
        mask2 = self._bbox_to_mask(bbox2, shape)
        return self._ios(mask1, mask2)

    def _iou_box(self, bbox1: Tuple, bbox2: Tuple, shape: Tuple) -> float:
        """Compute IoU between two bounding boxes by creating masks and using _iou.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice).
            bbox2: Bounding box. Same format as bbox1.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (depth, height, width) or (height, width) if using 2D projection.
            
        Returns:
            IoU score between 0.0 and 1.0.
        """
        mask1 = self._bbox_to_mask(bbox1, shape)
        mask2 = self._bbox_to_mask(bbox2, shape)
        return self._iou(mask1, mask2)
    
    def _dice_box(self, bbox1: Tuple, bbox2: Tuple, shape: Tuple) -> float:
        """Compute Dice coefficient between two bounding boxes by creating masks and using _dice.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice).
            bbox2: Bounding box. Same format as bbox1.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (depth, height, width) or (height, width) if using 2D projection.
            
        Returns:
            Dice coefficient between 0.0 and 1.0.
        """
        mask1 = self._bbox_to_mask(bbox1, shape)
        mask2 = self._bbox_to_mask(bbox2, shape)
        return self._dice(mask1, mask2)
    
    def _extract_region(self, labels: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Extract a sub-region from labels based on bounding box.
        
        Args:
            labels: Labeled image array.
            bbox: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                  For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X).
                  Note: regionprops returns bboxes in Z-Y-X order for 3D arrays.
                  For array shape (Z, Y, X), the bbox directly maps to array[Z, Y, X].
            
        Returns:
            Extracted sub-region from labels.
        """
        if len(bbox) == 6:
            # 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            # regionprops bbox format for 3D is in Z-Y-X order
            # For array shape (Z, Y, X), directly map: bbox[Z, Y, X] -> array[Z, Y, X]
            min_z, min_y, min_x, max_z, max_y, max_x = bbox
            if labels.ndim == 3:
                return labels[min_z:max_z, min_y:max_y, min_x:max_x]
            else:
                # 2D array, bbox is (min_row, min_col, max_row, max_col)
                min_y, min_x, max_y, max_x = bbox[:4]
                return labels[min_y:max_y, min_x:max_x]
        else:
            # 2D: (min_row, min_col, max_row, max_col)
            min_y, min_x, max_y, max_x = bbox
            return labels[min_y:max_y, min_x:max_x]
    
    def _bbox_to_relative(self, bbox: Tuple, union_bbox: Tuple) -> Tuple:
        """Convert a bounding box to coordinates relative to a union bounding box.
        
        Args:
            bbox: Original bounding box. For 2D: (min_row, min_col, max_row, max_col).
                  For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X) in Z-Y-X order.
            union_bbox: Union bounding box in the same format.
            
        Returns:
            Relative bounding box with coordinates relative to the union bbox origin.
        """
        if len(bbox) == 6:
            # 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            min_Z, min_Y, min_X, max_Z, max_Y, max_X = bbox
            u_min_Z, u_min_Y, u_min_X, u_max_Z, u_max_Y, u_max_X = union_bbox
            return (
                min_Z - u_min_Z,
                min_Y - u_min_Y,
                min_X - u_min_X,
                max_Z - u_min_Z,
                max_Y - u_min_Y,
                max_X - u_min_X
            )
        else:
            # 2D
            min_row, min_col, max_row, max_col = bbox
            u_min_row, u_min_col, u_max_row, u_max_col = union_bbox
            return (
                min_row - u_min_row,
                min_col - u_min_col,
                max_row - u_min_row,
                max_col - u_min_col
            )
    
    def _bboxes_overlap(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if two bounding boxes overlap.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X) in Z-Y-X order.
            bbox2: Bounding box. Same format as bbox1.
            
        Returns:
            True if bounding boxes overlap, False otherwise.
        """
        if len(bbox1) == 6 and len(bbox2) == 6:
            # 3D bounding boxes: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            min_Z1, min_Y1, min_X1, max_Z1, max_Y1, max_X1 = bbox1
            min_Z2, min_Y2, min_X2, max_Z2, max_Y2, max_X2 = bbox2
            
            # Check overlap in all three dimensions (Z, Y, X)
            overlap_Z = not (max_Z1 <= min_Z2 or max_Z2 <= min_Z1)
            overlap_Y = not (max_Y1 <= min_Y2 or max_Y2 <= min_Y1)
            overlap_X = not (max_X1 <= min_X2 or max_X2 <= min_X1)
            
            return overlap_Z and overlap_Y and overlap_X
        else:
            # 2D bounding boxes: (min_row, min_col, max_row, max_col)
            min_row1, min_col1, max_row1, max_col1 = bbox1
            min_row2, min_col2, max_row2, max_col2 = bbox2
            
            # Check overlap in both dimensions
            overlap_row = not (max_row1 <= min_row2 or max_row2 <= min_row1)
            overlap_col = not (max_col1 <= min_col2 or max_col2 <= min_col1)
            
            return overlap_row and overlap_col
    
    def _bbox_union(self, bboxes: List[Tuple]) -> Optional[Tuple]:
        """Compute the bounding box that encompasses a list of bounding boxes.
        
        Works for n-dimensional bounding boxes. Supports two input formats:
        
        1. Flat format: (min_0, min_1, ..., min_{n-1}, max_0, max_1, ..., max_{n-1})
           - 2D: (min_row, min_col, max_row, max_col) = 4 values
           - 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice) = 6 values
        
        2. Tuple-of-tuples format: ((min_0, min_1, ..., min_{n-1}), (max_0, max_1, ..., max_{n-1}))
           - 2D: ((min_row, min_col), (max_row, max_col))
           - 3D: ((min_row, min_col, min_slice), (max_row, max_col, max_slice))
        
        Args:
            bboxes: List of bounding boxes. All bounding boxes must have the same
                   dimensionality and format. Can be:
                   - List of flat tuples: [(min_0, ..., max_0, ...), ...]
                   - List of tuple-of-tuples: [((min_0, ...), (max_0, ...)), ...]
            
        Returns:
            Union bounding box in the same format as input, or None if list is empty.
            The union bounding box contains all input bounding boxes.
            
        Raises:
            ValueError: If bounding boxes have different dimensionalities or formats.
        """
        if not bboxes:
            return None
        
        try:
            bboxes = np.array(bboxes)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"All bounding boxes must have the same dimensionality and format. "
                f"Error converting to numpy array: {e}"
            )
        
        ndim = bboxes.ndim
        
        # Check if input is in tuple-of-tuples format (ndim == 3) or flat format (ndim == 2)
        if ndim == 3:
            # Tuple-of-tuples format: ((min_coords...), (max_coords...))
            # bboxes shape: (N, 2, M) where N is number of bboxes, M is number of dimensions
            union_mins = np.min(bboxes[:, 0, :], axis=0)
            union_maxs = np.max(bboxes[:, 1, :], axis=0)
            return (tuple(union_mins), tuple(union_maxs))
        else:
            # Flat format: (min_0, min_1, ..., min_{M-1}, max_0, max_1, ..., max_{M-1})
            # bboxes shape: (N, 2*M) where N is number of bboxes, M is number of dimensions
            # Need to split each bbox into min and max parts
            num_dims = bboxes.shape[1] // 2
            if bboxes.shape[1] % 2 != 0:
                raise ValueError(
                    f"Flat format bounding boxes must have an even number of elements. "
                    f"Got shape {bboxes.shape[1]}"
                )
            bbox_array = bboxes.reshape((len(bboxes), 2, num_dims))
            union_mins = np.min(bbox_array[:, 0, :], axis=0)
            union_maxs = np.max(bbox_array[:, 1, :], axis=0)
            return tuple(np.concatenate([union_mins, union_maxs]))

    def _process_slice(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
        stack_names: Tuple[str, str] = ("stack_1", "stack_2"),
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict]:
        """Compute pairwise overlap between regions in two label volumes."""
        labels1 = np.asarray(labels1)
        labels2 = np.asarray(labels2)
        stack_names = _normalize_stack_names(stack_names)
        if labels1.shape != labels2.shape:
            raise ValueError(
                "labels1 and labels2 must have the same shape: "
                f"{labels1.shape} vs {labels2.shape}"
            )

        if self.config.method not in ("iou", "ios", "dice"):
            raise ValueError(f"Unsupported coincidence method: {self.config.method!r}")

        overlap_metric: Literal["iou", "ios", "dice"] = self.config.method

        if self.config.mode == "outline":
            prune = labels1.ndim == 3
            scores = labels_iou_batch_3d_torch(
                labels1,
                labels2,
                overlap_metric=overlap_metric,
                prune_bboxes=prune,
            )
            ids1 = _label_ids_for_overlap_matrix(labels1, used_pruned_path=prune)
            ids2 = _label_ids_for_overlap_matrix(labels2, used_pruned_path=prune)
        elif self.config.mode == "bounding_box":
            ids1, boxes1 = _label_ids_and_boxes(labels1)
            ids2, boxes2 = _label_ids_and_boxes(labels2)
            if len(ids1) == 0 or len(ids2) == 0:
                return []
            scores = box_iou_batch_3d_torch(boxes1, boxes2, overlap_metric=overlap_metric)
        else:
            raise ValueError(f"Unsupported coincidence mode: {self.config.mode!r}")

        if len(ids1) == 0 or len(ids2) == 0:
            return []

        results: List[Dict] = []
        for i, label1 in enumerate(ids1):
            for j, label2 in enumerate(ids2):
                score = float(scores[i, j])
                results.append(
                    {
                        stack_names[0]: int(label1),
                        stack_names[1]: int(label2),
                        "score": score,
                        "above_threshold": score >= self.config.threshold,
                    }
                )
        return results

    def _process_slice_pairwise(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
        stack_names: Tuple[str, str] = ("stack_1", "stack_2"),
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict]:
        """Per-pair overlap for dice, bounding-box mode, and other non-batch paths."""
        # Get unique labels (excluding background 0)
        unique_labels1 = np.unique(labels1)
        unique_labels1 = unique_labels1[unique_labels1 > 0]
        
        unique_labels2 = np.unique(labels2)
        unique_labels2 = unique_labels2[unique_labels2 > 0]
        
        # If no regions in either image, return empty list
        if len(unique_labels1) == 0 or len(unique_labels2) == 0:
            return []
        
        results = []
        
        # Always get bounding boxes for all regions (regardless of mode)
        # This allows us to check overlap before expensive computations
        props1 = regionprops(labels1)
        props2 = regionprops(labels2)
        bboxes1 = {prop.label: prop.bbox for prop in props1}
        bboxes2 = {prop.label: prop.bbox for prop in props2}
        
        # Debug: Log bbox info for the labels we're comparing
        logger.debug(f"labels1.shape: {labels1.shape}, labels2.shape: {labels2.shape}")
        logger.debug(f"Unique labels1: {unique_labels1[:10]}... (showing first 10)")
        logger.debug(f"Unique labels2: {unique_labels2[:10]}... (showing first 10)")
        
        # Debug: Check a sample bbox to understand the format
        if unique_labels1.size > 0 and unique_labels1[0] in bboxes1:
            sample_bbox = bboxes1[unique_labels1[0]]
            logger.debug(f"Sample bbox for label {unique_labels1[0]}: {sample_bbox}, len: {len(sample_bbox)}")
        
        # Compute overlap for each region pair
        for label1 in unique_labels1:
            # Get bounding box for region1
            bbox1 = bboxes1.get(label1)
            if bbox1 is None:
                continue
            
            for label2 in unique_labels2:
                # Get bounding box for region2
                bbox2 = bboxes2.get(label2)
                if bbox2 is None:
                    continue
                
                # Check if bounding boxes overlap first (early exit optimization)
                if not self._bboxes_overlap(bbox1, bbox2):
                    # No overlap, score is 0 - skip expensive computation
                    overlap_score = 0.0
                else:
                    # Bounding boxes overlap, compute union bbox for optimized mask extraction
                    union_bbox = self._bbox_union([bbox1, bbox2])
                    if union_bbox is None:
                        logger.debug(f"No union bbox found for labels {label1} and {label2}, bbox1: {bbox1}, bbox2: {bbox2}")
                        overlap_score = 0.0
                    else:
                        # Extract sub-regions from both labels using union bbox
                        logger.debug(f"Union bbox found for labels {label1} and {label2}, union_bbox: {union_bbox}, labels1.shape: {labels1.shape}, labels2.shape: {labels2.shape}")
                        sub_labels1 = self._extract_region(labels1, union_bbox)
                        sub_labels2 = self._extract_region(labels2, union_bbox)
                        logger.debug(f"Extracted sub_labels1.shape: {sub_labels1.shape}, sub_labels2.shape: {sub_labels2.shape}")
                        logger.debug(f"Unique labels in sub_labels1: {np.unique(sub_labels1)}, looking for label {label1}")
                        logger.debug(f"Unique labels in sub_labels2: {np.unique(sub_labels2)}, looking for label {label2}")
                        
                        # Create masks on the smaller sub-regions
                        if self.config.mode == "outline":
                            # Pixel-level overlap on sub-regions
                            mask1 = (sub_labels1 == label1)
                            mask2 = (sub_labels2 == label2)
                            
                            # Debug: Check if masks are empty
                            logger.debug(f"mask1 sum: {np.sum(mask1)}, mask2 sum: {np.sum(mask2)}")
                            if not np.any(mask1):
                                logger.debug(f"Warning: mask1 for label {label1} is empty after extraction. Union bbox: {union_bbox}, bbox1: {bbox1}")
                            if not np.any(mask2):
                                logger.debug(f"Warning: mask2 for label {label2} is empty after extraction. Union bbox: {union_bbox}, bbox2: {bbox2}")
                            
                            if self.config.method == "iou":
                                overlap_score = self._iou(mask1, mask2)
                            elif self.config.method == "dice":
                                overlap_score = self._dice(mask1, mask2)
                            elif self.config.method == "ios":
                                overlap_score = self._ios(mask1, mask2)
                            else:
                                raise ValueError(f"Invalid method: {self.config.method}")
                        else:  # bounding_box mode
                            # Convert bboxes to relative coordinates within the union bbox
                            rel_bbox1 = self._bbox_to_relative(bbox1, union_bbox)
                            rel_bbox2 = self._bbox_to_relative(bbox2, union_bbox)
                            sub_shape = sub_labels1.shape
                            if self.config.method == "iou":
                                overlap_score = self._iou_box(rel_bbox1, rel_bbox2, sub_shape)
                            elif self.config.method == "dice":
                                overlap_score = self._dice_box(rel_bbox1, rel_bbox2, sub_shape)
                            elif self.config.method == "ios":
                                overlap_score = self._ios_box(rel_bbox1, rel_bbox2, sub_shape)
                            else:
                                raise ValueError(f"Invalid method: {self.config.method}")
                        logger.debug(f"Union bbox found for labels {label1} and {label2}, union_bbox: {union_bbox}, overlap_score: {overlap_score}")
                
                results.append({
                    stack_names[0]: int(label1),
                    stack_names[1]: int(label2),
                    "score": overlap_score,
                    "above_threshold": overlap_score >= self.config.threshold
                })
        
        return results

    def _consolidate_results(self, results: List[Dict], stack_names: Tuple[str, str] = ["stack_1", "stack_2"]) -> Dict[str, pd.DataFrame]:
        """Consolidate the results of the coincidence detector."""
        stack_names = _normalize_stack_names(stack_names)
        if not results:
            return {
                stack_names[0]: pd.DataFrame(columns=["label", "above_threshold", "max_score"]),
                stack_names[1]: pd.DataFrame(columns=["label", "above_threshold", "max_score"])
            }
               
        # Initialize result structure: {stack_name: {label_id: {'scores': [...], 'bools': [...]}}}
        temp_consolidated: Dict[str, Dict[int, Dict[str, List]]] = {
            stack_names[0]: {},
            stack_names[1]: {}
        }
        
        # Group results by stack and label, collecting both scores and booleans
        for result in results:
            label1 = result[stack_names[0]]
            label2 = result[stack_names[1]]
            score = result["score"]
            above_threshold = result["above_threshold"]
            
            # Add to stack 1 -> stack 2 mapping
            if label1 not in temp_consolidated[stack_names[0]]:
                temp_consolidated[stack_names[0]][label1] = {"scores": [], "bools": []}
            temp_consolidated[stack_names[0]][label1]["scores"].append(score)
            temp_consolidated[stack_names[0]][label1]["bools"].append(above_threshold)
            
            # Add to stack 2 -> stack 1 mapping
            if label2 not in temp_consolidated[stack_names[1]]:
                temp_consolidated[stack_names[1]][label2] = {"scores": [], "bools": []}
            temp_consolidated[stack_names[1]][label2]["scores"].append(score)
            temp_consolidated[stack_names[1]][label2]["bools"].append(above_threshold)
        
        # Build separate DataFrames for each stack
        dataframes = {}
        for stack_name, comp_stack_name in zip(stack_names, stack_names[::-1]):
            rows = []
            for label, data in temp_consolidated[stack_name].items():
                rows.append({
                    "label": label,
                    f"{comp_stack_name} +": any(data["bools"]),
                    f"{self.config.method} {comp_stack_name} +": max(data["scores"]) if data["scores"] else 0.0
                })
            dataframes[stack_name] = pd.DataFrame(rows).set_index("label")
        
        return dataframes

    @task(name="CoincidenceDetector.run", tags=["gpu_concurrency_limited"])
    def run(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
        stack_names: Optional[Tuple[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run the coincidence detector on a labeled image."""
        if stack_names is None or len(stack_names) != 2:
            stack_names = ("stack_1", "stack_2")
        else:
            stack_names = _normalize_stack_names(stack_names)

        results, _updated_metadata = super().run(
            labels1, labels2, stack_names, metadata=metadata, **kwargs
        )
        consolidated_dfs = self._consolidate_results(results, stack_names)
        return results, consolidated_dfs
