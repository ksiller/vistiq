"""IoU, IoS, and Dice overlap for boxes, masks, and label volumes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional, Self, Sequence, Union

import numpy as np
import pandas as pd
import torch
from prefect import task
from pydantic import Field, model_validator

from vistiq.matrix.types import FULL, ArrayBackend, MatrixData, matrix_to_numpy
from vistiq.core import Configurable, Configuration, generate_name
from vistiq.matrix.mask import triangle_valid_mask
from vistiq.segment.analysis import bbox_array_from_dataframe, dataframe_to_numpy
from vistiq.utils import (
    SpacingLike,
    convert_array_like,
    voxel_size,
    resolve_torch_device,
    abs_spacing,
)

logger = logging.getLogger(__name__)

_LABELS_IOU_DENSE_PAIR_FRACTION = 1.01


def box_intersection_nd(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Pairwise intersection hyper-volumes for ``(N, 2 * d)`` axis-aligned boxes."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError("boxes must be 2D arrays")
    if boxes_a.shape[1] != boxes_b.shape[1] or boxes_a.shape[1] % 2:
        raise ValueError(
            "boxes must have matching even width (min/max pairs per axis); "
            f"got {boxes_a.shape[1]} and {boxes_b.shape[1]}"
        )
    n_a, n_b = boxes_a.shape[0], boxes_b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float32)

    half = boxes_a.shape[1] // 2
    mins_a = boxes_a[:, :half]
    maxs_a = boxes_a[:, half:]
    mins_b = boxes_b[:, :half]
    maxs_b = boxes_b[:, half:]
    inter_mins = np.maximum(mins_a[:, None, :], mins_b[None, :, :])
    inter_maxs = np.minimum(maxs_a[:, None, :], maxs_b[None, :, :])
    extents = np.clip(inter_maxs - inter_mins, 0.0, None)
    return np.prod(extents, axis=2).astype(np.float32, copy=False)


def _box_volumes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    half = boxes.shape[1] // 2
    extents = np.maximum(boxes[:, half:] - boxes[:, :half], 0.0)
    return np.prod(extents, axis=1)


def box_areas_nd(boxes: np.ndarray, spacing: SpacingLike = None) -> np.ndarray:
    """Per-box volumes for ``(N, 2 * d)`` axis-aligned boxes in array axis order."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] % 2:
        raise ValueError(
            f"boxes must have shape (N, 2 * d) with even width; got {boxes.shape}"
        )
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    sp = abs_spacing(spacing)
    half = boxes.shape[1] // 2
    extents = np.maximum(boxes[:, half:] - boxes[:, :half], 0.0)
    if sp is not None:
        extents = extents * np.asarray(sp, dtype=np.float64)
    return np.prod(extents, axis=1)


def box_intersection_nd_torch(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """Pairwise intersection hyper-volumes for ``(N, 2 * d)`` axis-aligned boxes."""
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError("boxes must be 2D arrays")
    if boxes_a.shape[1] != boxes_b.shape[1] or boxes_a.shape[1] % 2:
        raise ValueError(
            "boxes must have matching even width (min/max pairs per axis); "
            f"got {boxes_a.shape[1]} and {boxes_b.shape[1]}"
        )
    n_a, n_b = boxes_a.shape[0], boxes_b.shape[0]
    if n_a == 0 or n_b == 0:
        return torch.empty((n_a, n_b), dtype=torch.float32, device=boxes_a.device)

    half = boxes_a.shape[1] // 2
    mins_a = boxes_a[:, :half]
    maxs_a = boxes_a[:, half:]
    mins_b = boxes_b[:, :half]
    maxs_b = boxes_b[:, half:]
    inter_mins = torch.maximum(mins_a[:, None, :], mins_b[None, :, :])
    inter_maxs = torch.minimum(maxs_a[:, None, :], maxs_b[None, :, :])
    extents = torch.clamp(inter_maxs - inter_mins, min=0.0)
    return extents.prod(dim=2).to(dtype=torch.float32)


def box_areas_nd_torch(
    boxes: torch.Tensor, spacing: SpacingLike = None
) -> torch.Tensor:
    """Per-box volumes for ``(N, 2 * d)`` axis-aligned boxes in array axis order."""
    if boxes.ndim != 2 or boxes.shape[1] % 2:
        raise ValueError(
            f"boxes must have shape (N, 2 * d) with even width; got {boxes.shape}"
        )
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32, device=boxes.device)
    half = boxes.shape[1] // 2
    extents = torch.clamp(boxes[:, half:] - boxes[:, :half], min=0.0).to(
        torch.float32
    )
    sp = abs_spacing(spacing)
    if sp is not None:
        scale = convert_array_like(
            sp, dtype="torch.Tensor", device=boxes.device
        ).to(dtype=extents.dtype)
        extents = extents * scale.unsqueeze(0)
    return extents.prod(dim=1)


def mask_areas_numpy(
    masks: np.ndarray, spacing: SpacingLike = None
) -> np.ndarray:
    """Per-mask voxel counts scaled by physical voxel volume when *spacing* is set."""
    masks = np.asarray(masks)
    if masks.ndim < 3:
        raise ValueError(f"mask stack must be at least 3D; got ndim={masks.ndim}")
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1, dtype=np.float64)
    areas *= voxel_size(spacing)
    return areas


def mask_intersection_numpy_split(
    masks_a: np.ndarray, masks_b: np.ndarray
) -> np.ndarray:
    """Dense pairwise intersection for mask batches sharing spatial shape."""
    masks_a = np.asarray(masks_a)
    masks_b = np.asarray(masks_b)
    n_a, n_b = masks_a.shape[0], masks_b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float32)
    flat_a = masks_a.reshape(n_a, -1).astype(np.float32, copy=False)
    flat_b = masks_b.reshape(n_b, -1).astype(np.float32, copy=False)
    return flat_a @ flat_b.T


def mask_intersection_numpy(
    masks_a: np.ndarray,
    masks_b: np.ndarray,
    *,
    memory_limit_mb: int = 5120,
    spacing: SpacingLike = None,
) -> np.ndarray:
    """Chunked pairwise mask intersection."""
    masks_a = np.asarray(masks_a)
    masks_b = np.asarray(masks_b)
    if masks_a.shape[1:] != masks_b.shape[1:]:
        raise ValueError(
            "mask spatial shapes must match: "
            f"{masks_a.shape[1:]} vs {masks_b.shape[1:]}"
        )
    n_a, n_b = masks_a.shape[0], masks_b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float32)

    voxels = int(np.prod(masks_a.shape[1:]))
    memory_mb = n_a * voxels * n_b / 1024 / 1024
    if memory_mb <= memory_limit_mb:
        inter = mask_intersection_numpy_split(masks_a, masks_b)
    else:
        chunks: list[np.ndarray] = []
        step = max(memory_limit_mb * 1024 * 1024 // (n_b * voxels), 1)
        for start in range(0, n_a, step):
            chunks.append(
                mask_intersection_numpy_split(
                    masks_a[start : start + step], masks_b
                )
            )
        inter = np.vstack(chunks).astype(np.float32, copy=False)
    return inter * voxel_size(spacing)


def mask_areas_torch(
    masks: torch.Tensor, spacing: SpacingLike = None
) -> torch.Tensor:
    if masks.ndim < 3:
        raise ValueError(f"mask stack must be at least 3D; got ndim={masks.ndim}")
    flat = masks.reshape(masks.shape[0], -1).to(dtype=torch.float32)
    areas = flat.sum(dim=1)
    areas = areas * voxel_size(spacing)
    return areas


def mask_intersection_torch_split(
    masks_a: torch.Tensor, masks_b: torch.Tensor
) -> torch.Tensor:
    n_a, n_b = masks_a.shape[0], masks_b.shape[0]
    if n_a == 0 or n_b == 0:
        return torch.empty((n_a, n_b), dtype=torch.float32, device=masks_a.device)
    flat_a = masks_a.reshape(n_a, -1).to(dtype=torch.float32)
    flat_b = masks_b.reshape(n_b, -1).to(dtype=torch.float32)
    return flat_a @ flat_b.T


def mask_intersection_torch(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    *,
    memory_limit_mb: int = 5120,
    spacing: SpacingLike = None,
) -> torch.Tensor:
    if masks_a.shape[1:] != masks_b.shape[1:]:
        raise ValueError(
            "mask spatial shapes must match: "
            f"{masks_a.shape[1:]} vs {masks_b.shape[1:]}"
        )
    n_a, n_b = masks_a.shape[0], masks_b.shape[0]
    if n_a == 0 or n_b == 0:
        return torch.empty((n_a, n_b), dtype=torch.float32, device=masks_a.device)

    voxels = int(torch.prod(torch.tensor(masks_a.shape[1:], device=masks_a.device)))
    memory_mb = n_a * int(voxels) * n_b / 1024 / 1024
    if memory_mb <= memory_limit_mb:
        inter = mask_intersection_torch_split(masks_a, masks_b)
    else:
        chunks: list[torch.Tensor] = []
        step = max(memory_limit_mb * 1024 * 1024 // (n_b * int(voxels)), 1)
        for start in range(0, n_a, step):
            chunks.append(
                mask_intersection_torch_split(
                    masks_a[start : start + step], masks_b
                )
            )
        inter = torch.vstack(chunks).to(dtype=torch.float32)
    scale = voxel_size(spacing)
    if scale != 1.0:
        inter = inter * scale
    return inter


def _boxes_from_masks_numpy(masks: np.ndarray) -> np.ndarray:
    """Axis-aligned boxes ``(N, 2 * d)`` from a boolean mask stack."""
    spatial_ndim = masks.ndim - 1
    n = masks.shape[0]
    if n == 0:
        return np.empty((0, 2 * spatial_ndim), dtype=np.float32)
    boxes = np.zeros((n, 2 * spatial_ndim), dtype=np.float32)
    for index in range(n):
        coords = np.argwhere(masks[index])
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        boxes[index, :spatial_ndim] = mins
        boxes[index, spatial_ndim:] = maxs
    return boxes


def _positive_label_ids(labels: np.ndarray) -> np.ndarray:
    """Sorted positive label ids in ``np.unique`` order."""
    ids = np.unique(np.asarray(labels))
    return ids[ids > 0].astype(np.int64, copy=False)


def label_areas(
    labels: np.ndarray,
    label_ids: Sequence[int],
    *,
    spacing: SpacingLike = None,
) -> np.ndarray:
    """Per-object areas from a label array and ordered *label_ids*."""
    labels = np.asarray(labels)
    if len(label_ids) == 0:
        return np.empty(0, dtype=np.float64)
    max_id = max(int(label_id) for label_id in label_ids)
    counts = np.bincount(labels.ravel(), minlength=max_id + 1)
    areas = counts[np.asarray(label_ids, dtype=np.intp)].astype(np.float64)
    areas *= voxel_size(spacing)
    return areas


def label_intersection_linear(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    label_ids_a: Sequence[int],
    label_ids_b: Sequence[int],
    *,
    spacing: SpacingLike = None,
) -> np.ndarray:
    """Pairwise voxel intersections via linear pair encoding and histogram."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            "label arrays must have the same shape: "
            f"{labels_a.shape} vs {labels_b.shape}"
        )
    n_a, n_b = len(label_ids_a), len(label_ids_b)
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float32)

    def remap_table(label_ids: Sequence[int]) -> np.ndarray:
        max_id = max(int(label_id) for label_id in label_ids)
        table = np.zeros(max_id + 1, dtype=np.int32)
        for index, label_id in enumerate(label_ids):
            table[int(label_id)] = index + 1
        return table

    remap_a = remap_table(label_ids_a)
    remap_b = remap_table(label_ids_b)
    index_a = remap_a[labels_a.ravel()].reshape(labels_a.shape)
    index_b = remap_b[labels_b.ravel()].reshape(labels_b.shape)
    both = (index_a > 0) & (index_b > 0)
    stride = n_a
    codes = (index_a[both] - 1) + (index_b[both] - 1) * stride
    counts = np.bincount(codes, minlength=n_a * n_b)
    inter = counts.reshape(n_a, n_b, order="F").astype(np.float64)
    inter *= voxel_size(spacing)
    return inter.astype(np.float32, copy=False)


def label_intersection_sparse(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    label_ids_a: Sequence[int],
    label_ids_b: Sequence[int],
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    *,
    spacing: SpacingLike = None,
) -> np.ndarray:
    """Pairwise voxel intersections via bbox-pruned crops on label arrays."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            "label arrays must have the same shape: "
            f"{labels_a.shape} vs {labels_b.shape}"
        )
    n_a, n_b = len(label_ids_a), len(label_ids_b)
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float32)
    if boxes_a.shape[0] != n_a or boxes_b.shape[0] != n_b:
        raise ValueError(
            "boxes must align with label_ids: "
            f"got {boxes_a.shape[0]} and {boxes_b.shape[0]} for "
            f"{n_a} and {n_b} labels"
        )

    bbox_inter = box_intersection_nd(boxes_a, boxes_b)
    candidates = np.argwhere(bbox_inter > 0)
    out = np.zeros((n_a, n_b), dtype=np.float32)
    scale = voxel_size(spacing)
    for i, j in candidates:
        box_a = boxes_a[i]
        box_b = boxes_b[j]
        half = box_a.shape[0] // 2
        union = np.concatenate(
            [
                np.minimum(box_a[:half], box_b[:half]),
                np.maximum(box_a[half:], box_b[half:]),
            ]
        )
        slices = tuple(slice(int(union[d]), int(union[half + d])) for d in range(half))
        crop_a = labels_a[slices]
        crop_b = labels_b[slices]
        overlap = float(
            np.count_nonzero(
                (crop_a == label_ids_a[i]) & (crop_b == label_ids_b[j])
            )
        )
        out[i, j] = overlap * scale
    return out


@dataclass(frozen=True)
class OverlapStrategy:
    """Resolved label intersection mode from resolve_intersection_mode."""

    mode: Literal["linear", "sparse"]
    reason: str


def resolve_intersection_mode(
    *,
    shape: tuple[int, ...],
    n_objects_a: int,
    n_objects_b: int,
    boxes_a: Optional[np.ndarray] = None,
    boxes_b: Optional[np.ndarray] = None,
    mode: Literal["linear", "sparse", "auto"] = "auto",
    total_memory_limit: int = 512,
    pair_memory_limit: int = 512,
    dense_pair_fraction: float = 0.3,
    max_bbox_volume_fraction: float = 0.25,
) -> OverlapStrategy:
    """Resolve label intersection mode from object counts, shape, and optional boxes."""
    if mode == "linear":
        return OverlapStrategy(mode="linear", reason="linear: explicit")
    if mode == "sparse":
        return OverlapStrategy(mode="sparse", reason="sparse: explicit")

    n_spatial = len(shape)
    voxels = int(np.prod(shape)) if shape else 0
    pair_count = n_objects_a * n_objects_b
    mask_stack_mb = max(n_objects_a, n_objects_b) * voxels * 4 / (1024 * 1024)

    if boxes_a is not None and boxes_b is not None and pair_count > 0:
        bbox_inter = box_intersection_nd(
            np.asarray(boxes_a, dtype=np.float32),
            np.asarray(boxes_b, dtype=np.float32),
        )
        overlap_pairs = int(np.count_nonzero(bbox_inter > 0))
        overlap_fraction = overlap_pairs / pair_count
        box_volumes = _box_volumes(np.concatenate([np.asarray(boxes_a), np.asarray(boxes_b)]))
        max_bbox_frac = float(box_volumes.max()) / voxels if voxels > 0 else 0.0
    else:
        overlap_fraction = 1.0
        max_bbox_frac = 1.0

    # Estimated sparse crop working memory (two union buffers sized to largest object).
    pair_sparse_mb = max_bbox_frac * voxels * 2 * 4 / (1024 * 1024)

    prefer_linear = (
        mask_stack_mb > total_memory_limit
        or pair_sparse_mb > pair_memory_limit
        or overlap_fraction > dense_pair_fraction
        or max_bbox_frac > max_bbox_volume_fraction
    )
    prefer_sparse = (
        overlap_fraction < dense_pair_fraction
        and max_bbox_frac < max_bbox_volume_fraction
    )

    if prefer_linear:
        reason_parts = [
            f"mask_stack_mb={mask_stack_mb:.0f}",
            f"pair_sparse_mb={pair_sparse_mb:.0f}",
            f"f={overlap_fraction:.2f}",
        ]
        if max_bbox_frac > 0:
            reason_parts.append(f"max_bbox_frac={max_bbox_frac:.2f}")
        reason_parts.append(f"M={n_objects_a},N={n_objects_b}")
        return OverlapStrategy(
            mode="linear",
            reason="linear: " + ", ".join(reason_parts),
        )

    if prefer_sparse:
        return OverlapStrategy(
            mode="sparse",
            reason=(
                f"sparse: f={overlap_fraction:.2f}, "
                f"max_bbox_frac={max_bbox_frac:.2f}"
            ),
        )

    return OverlapStrategy(
        mode="linear",
        reason=(
            f"linear: mask_stack_mb={mask_stack_mb:.0f}, f={overlap_fraction:.2f} "
            "(small volume fallback)"
        ),
    )


def union_matrix(
    area_a: Union[np.ndarray, torch.Tensor],
    area_b: Union[np.ndarray, torch.Tensor],
    *,
    inter: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Pairwise union from per-instance areas and intersection matrix."""
    if isinstance(area_a, torch.Tensor):
        return area_a[:, None] + area_b[None, :] - inter
    return area_a[:, None] + area_b[None, :] - inter


def apply_triangle_mask(
    matrix: Union[np.ndarray, torch.Tensor],
    triangle: int,
) -> Union[np.ndarray, torch.Tensor]:
    """Set disallowed triangle regions to NaN (no-op when ``triangle`` is ``FULL``)."""
    if isinstance(matrix, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(matrix))
        as_numpy = True
    else:
        tensor = matrix
        as_numpy = False
    mask = triangle_valid_mask(tensor, triangle)
    if mask is None:
        return matrix
    nan = torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    result = torch.where(mask, tensor, nan)
    if as_numpy:
        return result.detach().cpu().numpy()
    return result


def _divide_pairwise(
    numer: Union[np.ndarray, torch.Tensor],
    denom: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(numer, torch.Tensor):
        return torch.where(
            denom > 0,
            numer / denom,
            torch.zeros_like(numer),
        )
    out = np.zeros_like(numer, dtype=np.float32)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


# ---------------------------------------------------------------------------
# Region map utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionSpec:
    """One region entry in a RegionMap.

    The map key is the globally unique ``object_id`` (typically from
    RegionAnalyzer). ``label_id`` and ``bbox`` describe how to build
    geometry for that object on its side of the comparison.

    Attributes:
        label_id: Integer label value in that side's label volume. Required for
            label overlap when using ``region_map``.
        bbox: Axis-aligned box ``(min_0, ..., min_{n-1}, max_0, ..., max_{n-1})``
            in array axis order. Required for box overlap; optional for label
            sparse pruning and mask intersection pruning when ``prune_bboxes=True``.
    """

    label_id: Optional[int] = None
    bbox: Optional[Sequence[float]] = None


RegionMap = Mapping[str, RegionSpec]


@dataclass(frozen=True)
class RegionBuildContext:
    """Parsed RegionMap for one overlap collection."""

    object_ids: tuple[str, ...]
    label_ids: tuple[int, ...]
    boxes: Optional[np.ndarray]


def parse_region_map(
    region_map: RegionMap,
    *,
    require_label_id: bool = False,
    require_bbox: bool = False,
) -> RegionBuildContext:
    """Validate *region_map* and extract ordered ids and optional boxes."""
    object_ids = tuple(region_map.keys())
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("region_map contains duplicate object_id keys")

    label_ids: list[int] = []
    seen_label_ids: set[int] = set()
    boxes: list[Sequence[float]] = []

    for object_id in object_ids:
        spec = region_map[object_id]
        if require_label_id and spec.label_id is None:
            raise ValueError(
                f"region_map[{object_id!r}] is missing label_id"
            )
        if require_bbox and spec.bbox is None:
            raise ValueError(f"region_map[{object_id!r}] is missing bbox")
        if spec.label_id is not None:
            if spec.label_id in seen_label_ids:
                raise ValueError(
                    f"region_map contains duplicate label_id {spec.label_id}"
                )
            seen_label_ids.add(spec.label_id)
            label_ids.append(int(spec.label_id))
        if spec.bbox is not None:
            if len(spec.bbox) < 2 or len(spec.bbox) % 2:
                raise ValueError(
                    f"region_map[{object_id!r}].bbox must have even length >= 2; "
                    f"got {len(spec.bbox)}"
                )
            boxes.append(tuple(float(v) for v in spec.bbox))

    boxes_arr: Optional[np.ndarray] = None
    if boxes:
        if len(boxes) != len(object_ids):
            raise ValueError(
                "region_map must include bbox for every entry or none at all"
            )
        boxes_arr = np.asarray(boxes, dtype=np.float32)

    if require_label_id and len(label_ids) != len(object_ids):
        raise ValueError("region_map must include label_id for every entry")

    return RegionBuildContext(
        object_ids=object_ids,
        label_ids=tuple(label_ids),
        boxes=boxes_arr,
    )


def region_map_from_dataframe(
    df: pd.DataFrame,
    *,
    object_id_col: str = "object_id",
    label_col: str = "label",
    bbox_cols: Optional[Sequence[str]] = None,
    axes: Optional[Sequence[str]] = None,
) -> dict[str, RegionSpec]:
    """Build a RegionMap from a RegionAnalyzer table.

    Expects columns ``object_id``, ``label``, and optional bbox columns from
    :func:`~vistiq.segment.analysis.bbox_array_from_dataframe`. Pass *axes*
    (e.g. ``metadata["axes"]``) to order mapped ``bbox-start-{axis}`` /
    ``bbox-end-{axis}`` columns.

    Row order becomes matrix row/column order when ``annotate=True`` and no
    custom annotations are passed.

    After RegionFilter, pass the table directly — when the index is named
    ``label`` or ``object_id``, it is promoted to a column automatically.

    Args:
        df: Region property table (e.g. ``l_accepted.reset_index()``).
        object_id_col: Column holding unique object identifiers.
        label_col: Column holding integer label ids in the label volume.
        bbox_cols: Explicit bbox column names; overrides *axes* when set.
        axes: Array axis names (e.g. ``["Z", "Y", "X"]`` from image metadata).

    Returns:
        Mapping from ``object_id`` to RegionSpec.
    """
    index_name = df.index.name
    if index_name is not None and index_name in {label_col, object_id_col}:
        df = df.reset_index()

    if object_id_col not in df.columns:
        raise KeyError(
            f"column {object_id_col!r} not found; available: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise KeyError(
            f"column {label_col!r} not found; available: {list(df.columns)}"
        )

    object_ids = dataframe_to_numpy(df, attributes=[object_id_col], reset_index=False)
    label_ids = dataframe_to_numpy(df, attributes=[label_col], reset_index=False)
    bboxes_arr = bbox_array_from_dataframe(
        df, bbox_cols=bbox_cols, axes=axes, reset_index=False
    )

    region_map: dict[str, RegionSpec] = {}
    for i, object_id in enumerate(object_ids):
        object_id = str(object_id)
        if object_id in region_map:
            raise ValueError(f"duplicate object_id {object_id!r} in dataframe")
        bbox: Optional[tuple[float, ...]] = None
        if bboxes_arr is not None:
            bbox = tuple(float(v) for v in bboxes_arr[i])  # type: ignore[assignment]
        region_map[object_id] = RegionSpec(
            label_id=int(label_ids[i]),
            bbox=bbox,
        )
    return region_map


@dataclass(frozen=True)
class LabelBuild:
    """Label volume plus ordered label ids and axis-aligned boxes ``(N, 2 * ndim)``."""

    labels: np.ndarray
    label_ids: tuple[int, ...]
    boxes: np.ndarray


def _discover_label_boxes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig
    from vistiq.utils import ArrayIteratorConfig

    ndim = labels.ndim
    table = RegionAnalyzer(
        RegionAnalyzerConfig(
            properties=["label", "bbox"],
            output_type="dataframe",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    ).run(labels)
    if len(table) == 0:
        return np.array([], dtype=np.int64), np.empty((0, 2 * ndim), dtype=np.float32)
    label_ids = dataframe_to_numpy(table, attributes=["label"])
    boxes = bbox_array_from_dataframe(table)
    if boxes is None:
        raise KeyError(
            "RegionAnalyzer table has no bbox columns; "
            f"available: {list(table.columns)}"
        )
    return label_ids.astype(np.int64, copy=False), boxes.astype(np.float32, copy=False)


def _select_label_boxes(
    labels: np.ndarray,
    label_ids: Sequence[int],
    boxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    width = 2 * labels.ndim
    if boxes is not None:
        boxes_arr = np.asarray(boxes, dtype=np.float32)
        if boxes_arr.shape != (len(label_ids), width):
            raise ValueError(
                f"boxes must have shape ({len(label_ids)}, {width}); got {boxes_arr.shape}"
            )
        return boxes_arr

    discovered_ids, discovered_boxes = _discover_label_boxes(labels)
    id_to_index = {int(label_id): index for index, label_id in enumerate(discovered_ids)}
    out = np.zeros((len(label_ids), width), dtype=np.float32)
    for index, label_id in enumerate(label_ids):
        source = id_to_index.get(int(label_id))
        if source is not None:
            out[index] = discovered_boxes[source]
    return out


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


class MatrixBuilderConfig(Configuration):
    """Shared settings for overlap input builders."""

    preferred_input_type: ArrayBackend = "torch.Tensor"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class MatrixBuilder(Configurable[MatrixBuilderConfig]):
    """Normalize raw overlap inputs into a representation for area/intersection."""

    def __init__(self, config: MatrixBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixBuilderConfig) -> "MatrixBuilder":
        return cls(config)

    def run(
        self,
        data: Any,
        *,
        region_map: Optional[RegionMap] = None,
        device: Optional[torch.device] = None,
    ) -> Any:
        raise NotImplementedError


class BoxBuilderConfig(MatrixBuilderConfig):
    """Configuration for BoxBuilder."""


class BoxBuilder(MatrixBuilder):
    """Validate and convert box arrays ``(N, 2 * d)`` in array axis order."""

    def __init__(self, config: BoxBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: BoxBuilderConfig) -> "BoxBuilder":
        return cls(config)

    def run(
        self,
        data: Any = None,
        *,
        region_map: Optional[RegionMap] = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        dtype = self.config.preferred_input_type
        if region_map is not None:
            box_ctx = parse_region_map(region_map, require_bbox=True)
            assert box_ctx.boxes is not None
            boxes = box_ctx.boxes
        else:
            if data is None:
                raise ValueError(
                    "BoxBuilder requires box array data or region_map"
                )
            boxes = np.asarray(data)
        boxes = convert_array_like(boxes, dtype=dtype, device=device)
        if boxes.ndim != 2 or boxes.shape[1] % 2:
            raise ValueError(
                "boxes must have shape (N, 2 * d) with even width; "
                f"got {getattr(boxes, 'shape', None)}"
            )
        return boxes


class MaskStackBuilderConfig(MatrixBuilderConfig):
    """Configuration for MaskStackBuilder."""


class MaskStackBuilder(MatrixBuilder):
    """Normalize mask stacks to ``(N, *spatial)``."""

    def __init__(self, config: MaskStackBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MaskStackBuilderConfig) -> "MaskStackBuilder":
        return cls(config)

    def run(
        self,
        data: Any,
        *,
        region_map: Optional[RegionMap] = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        dtype = self.config.preferred_input_type
        masks = convert_array_like(np.asarray(data), dtype=dtype, device=device)
        if masks.ndim < 3:
            raise ValueError(
                f"mask stack must be at least 3D (N, *spatial); got ndim={masks.ndim}"
            )
        return masks


class LabelBuilderConfig(MatrixBuilderConfig):
    """Configuration for LabelBuilder."""

    preferred_input_type: ArrayBackend = "np.ndarray"


class LabelBuilder(MatrixBuilder):
    """Prepare a label volume, object ids, and boxes for overlap."""

    def __init__(self, config: LabelBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: LabelBuilderConfig) -> "LabelBuilder":
        return cls(config)

    def run(
        self,
        data: Any,
        *,
        region_map: Optional[RegionMap] = None,
        device: Optional[torch.device] = None,
    ) -> LabelBuild:
        del device
        if isinstance(data, pd.DataFrame):
            raise TypeError(
                "LabelBuilder expects a 2D/3D integer label volume, not a "
                "region property DataFrame. Pass the labeled image array with "
                "region_map=(map_a, map_b), or use BoxOverlapCalculatorConfig."
            )
        if data is None:
            raise ValueError("LabelBuilder requires a label volume")
        labels = np.asarray(data)
        if labels.ndim < 2:
            raise ValueError(
                f"label arrays must have at least 2 dimensions; got ndim={labels.ndim}"
            )
        if labels.dtype == object or not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(
                f"label arrays must have an integer dtype; got {labels.dtype}"
            )

        if region_map is not None:
            ctx = parse_region_map(region_map, require_label_id=True)
            return LabelBuild(
                labels=labels,
                label_ids=ctx.label_ids,
                boxes=_select_label_boxes(labels, ctx.label_ids, ctx.boxes),
            )
        label_ids = tuple(_positive_label_ids(labels))
        return LabelBuild(
            labels=labels,
            label_ids=label_ids,
            boxes=_select_label_boxes(labels, label_ids),
        )


# ---------------------------------------------------------------------------
# Area calculators
# ---------------------------------------------------------------------------


class AreaCalculatorConfig(Configuration):
    """Shared settings for per-instance area calculators."""

    preferred_input_type: ArrayBackend = "torch.Tensor"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class AreaCalculator(Configurable[AreaCalculatorConfig]):
    """Compute per-instance areas/volumes from a built representation."""

    def __init__(self, config: AreaCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: AreaCalculatorConfig) -> "AreaCalculator":
        return cls(config)

    def run(
        self,
        built: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError


class BoxAreaCalculatorConfig(AreaCalculatorConfig):
    """Configuration for BoxAreaCalculator."""


class BoxAreaCalculator(AreaCalculator):
    """Per-box volumes from axis-aligned boxes."""

    def __init__(self, config: BoxAreaCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: BoxAreaCalculatorConfig) -> "BoxAreaCalculator":
        return cls(config)

    def run(
        self,
        built: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        dtype = self.config.preferred_input_type
        boxes = convert_array_like(built, dtype=dtype, device=device)
        if isinstance(boxes, torch.Tensor):
            return box_areas_nd_torch(boxes, spacing)
        return box_areas_nd(np.asarray(boxes), spacing)


class MaskAreaCalculatorConfig(AreaCalculatorConfig):
    """Configuration for MaskAreaCalculator."""


class MaskAreaCalculator(AreaCalculator):
    """Per-mask voxel counts (optionally scaled to physical volume)."""

    def __init__(self, config: MaskAreaCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MaskAreaCalculatorConfig) -> "MaskAreaCalculator":
        return cls(config)

    def run(
        self,
        built: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        dtype = self.config.preferred_input_type
        masks = convert_array_like(built, dtype=dtype, device=device)
        if isinstance(masks, torch.Tensor):
            return mask_areas_torch(masks, spacing)
        return mask_areas_numpy(np.asarray(masks), spacing)


class LabelAreaCalculatorConfig(AreaCalculatorConfig):
    """Configuration for LabelAreaCalculator."""

    preferred_input_type: ArrayBackend = "np.ndarray"


class LabelAreaCalculator(AreaCalculator):
    """Per-object areas from a LabelBuild."""

    def __init__(self, config: LabelAreaCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: LabelAreaCalculatorConfig) -> "LabelAreaCalculator":
        return cls(config)

    def run(
        self,
        built: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if not isinstance(built, LabelBuild):
            raise TypeError(f"LabelAreaCalculator expects LabelBuild; got {type(built)}")
        areas = label_areas(built.labels, built.label_ids, spacing=spacing)
        return convert_array_like(
            areas,
            dtype=self.config.preferred_input_type,
            device=device,
        )


# ---------------------------------------------------------------------------
# Intersection calculators
# ---------------------------------------------------------------------------


class IntersectionCalculatorConfig(Configuration):
    """Shared settings for pairwise intersection calculators."""

    preferred_input_type: ArrayBackend = "torch.Tensor"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class IntersectionCalculator(Configurable[IntersectionCalculatorConfig]):
    """Compute pairwise intersection volumes between built representations."""

    def __init__(self, config: IntersectionCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: IntersectionCalculatorConfig) -> "IntersectionCalculator":
        return cls(config)

    def run(
        self,
        built_a: Any,
        built_b: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError


class BoxIntersectionCalculatorConfig(IntersectionCalculatorConfig):
    """Configuration for BoxIntersectionCalculator."""


class BoxIntersectionCalculator(IntersectionCalculator):
    """Pairwise intersection volumes for axis-aligned boxes."""

    def __init__(self, config: BoxIntersectionCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(
        cls, config: BoxIntersectionCalculatorConfig
    ) -> "BoxIntersectionCalculator":
        return cls(config)

    def run(
        self,
        built_a: Any,
        built_b: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        dtype = self.config.preferred_input_type
        boxes_a = convert_array_like(built_a, dtype=dtype, device=device)
        boxes_b = convert_array_like(built_b, dtype=dtype, device=device)
        if isinstance(boxes_a, torch.Tensor):
            inter = box_intersection_nd_torch(boxes_a, boxes_b)
            if spacing is not None:
                inter = inter * voxel_size(spacing)
            return inter
        inter = box_intersection_nd(np.asarray(boxes_a), np.asarray(boxes_b))
        if spacing is not None:
            inter = inter * voxel_size(spacing)
        return inter


class MaskIntersectionCalculatorConfig(IntersectionCalculatorConfig):
    """Configuration for MaskIntersectionCalculator.

    Attributes:
        memory_limit_mb: Chunk budget for dense mask intersection.
        prune_bboxes: When True, skip mask pairs with non-overlapping boxes.
        dense_pair_fraction: Use dense intersection when bbox-overlap fraction
            is at or above this threshold.
    """

    memory_limit_mb: int = 5120
    prune_bboxes: bool = False
    dense_pair_fraction: float = _LABELS_IOU_DENSE_PAIR_FRACTION


class MaskIntersectionCalculator(IntersectionCalculator):
    """Pairwise mask intersection with optional bbox pruning."""

    def __init__(self, config: MaskIntersectionCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(
        cls, config: MaskIntersectionCalculatorConfig
    ) -> "MaskIntersectionCalculator":
        return cls(config)

    def run(
        self,
        built_a: Any,
        built_b: Any,
        *,
        spacing: SpacingLike = None,
        boxes_a: Optional[np.ndarray] = None,
        boxes_b: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        dtype = self.config.preferred_input_type
        masks_a = convert_array_like(built_a, dtype=dtype, device=device)
        masks_b = convert_array_like(built_b, dtype=dtype, device=device)

        if not self.config.prune_bboxes:
            if isinstance(masks_a, torch.Tensor):
                return mask_intersection_torch(
                    masks_a,
                    masks_b,
                    memory_limit_mb=self.config.memory_limit_mb,
                    spacing=spacing,
                )
            return mask_intersection_numpy(
                np.asarray(masks_a),
                np.asarray(masks_b),
                memory_limit_mb=self.config.memory_limit_mb,
                spacing=spacing,
            )

        if boxes_a is None:
            boxes_a = _boxes_from_masks_numpy(
                masks_a.detach().cpu().numpy()
                if isinstance(masks_a, torch.Tensor)
                else np.asarray(masks_a)
            )
        if boxes_b is None:
            boxes_b = _boxes_from_masks_numpy(
                masks_b.detach().cpu().numpy()
                if isinstance(masks_b, torch.Tensor)
                else np.asarray(masks_b)
            )
        n_a, n_b = boxes_a.shape[0], boxes_b.shape[0]
        if n_a == 0 or n_b == 0:
            empty = np.empty((n_a, n_b), dtype=np.float32)
            return convert_array_like(empty, dtype=dtype, device=device)

        bbox_inter = box_intersection_nd(boxes_a, boxes_b)
        candidates = np.argwhere(bbox_inter > 0)
        n_pairs = n_a * n_b
        if len(candidates) >= self.config.dense_pair_fraction * n_pairs:
            logger.info(
                "MaskIntersectionCalculator: %d/%d pairs overlap in bbox space, using dense masks",
                len(candidates),
                n_pairs,
            )
            if isinstance(masks_a, torch.Tensor):
                return mask_intersection_torch(
                    masks_a,
                    masks_b,
                    memory_limit_mb=self.config.memory_limit_mb,
                    spacing=spacing,
                )
            return mask_intersection_numpy(
                np.asarray(masks_a),
                np.asarray(masks_b),
                memory_limit_mb=self.config.memory_limit_mb,
                spacing=spacing,
            )

        out = np.zeros((n_a, n_b), dtype=np.float32)
        masks_a_np = (
            masks_a.detach().cpu().numpy()
            if isinstance(masks_a, torch.Tensor)
            else np.asarray(masks_a)
        )
        masks_b_np = (
            masks_b.detach().cpu().numpy()
            if isinstance(masks_b, torch.Tensor)
            else np.asarray(masks_b)
        )
        for i, j in candidates:
            inter = float(
                np.logical_and(masks_a_np[i], masks_b_np[j]).sum(dtype=np.float64)
            )
            if spacing is not None:
                inter *= voxel_size(spacing)
            out[i, j] = inter
        return convert_array_like(out, dtype=dtype, device=device)


class LabelIntersectionCalculatorConfig(IntersectionCalculatorConfig):
    """Configuration for LabelIntersectionCalculator.

    Attributes:
        mode: ``"linear"`` (histogram on pair codes), ``"sparse"`` (bbox-pruned
            crops), or ``"auto"`` (default).
        total_memory_limit: Prefer linear when estimated dense mask-stack memory
            (``max(M,N) × voxels × 4`` bytes) exceeds this limit (MB).
        pair_memory_limit: Prefer linear when estimated sparse crop working memory
            (two buffers sized to the largest bbox) exceeds this limit (MB).
        dense_pair_fraction: Bbox-overlap fraction threshold for ``auto`` mode.
        max_bbox_volume_fraction: Largest-box fraction threshold for ``auto`` linear.
    """

    preferred_input_type: ArrayBackend = "np.ndarray"
    mode: Literal["linear", "sparse", "auto"] = "auto"
    total_memory_limit: int = 512
    pair_memory_limit: int = 512
    dense_pair_fraction: float = 0.3
    max_bbox_volume_fraction: float = 0.25


class LabelIntersectionCalculator(IntersectionCalculator):
    """Pairwise label intersections via linear encoding or bbox-sparse crops."""

    def __init__(self, config: LabelIntersectionCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(
        cls, config: LabelIntersectionCalculatorConfig
    ) -> "LabelIntersectionCalculator":
        return cls(config)

    def run(
        self,
        built_a: Any,
        built_b: Any,
        *,
        spacing: SpacingLike = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if not isinstance(built_a, LabelBuild) or not isinstance(built_b, LabelBuild):
            raise TypeError("LabelIntersectionCalculator expects two LabelBuild inputs")
        strategy = resolve_intersection_mode(
            shape=built_a.labels.shape,
            n_objects_a=len(built_a.label_ids),
            n_objects_b=len(built_b.label_ids),
            boxes_a=built_a.boxes,
            boxes_b=built_b.boxes,
            mode=self.config.mode,
            total_memory_limit=self.config.total_memory_limit,
            pair_memory_limit=self.config.pair_memory_limit,
            dense_pair_fraction=self.config.dense_pair_fraction,
            max_bbox_volume_fraction=self.config.max_bbox_volume_fraction,
        )
        logger.info("Label intersection mode: %s", strategy.reason)
        if strategy.mode == "linear":
            inter = label_intersection_linear(
                built_a.labels,
                built_b.labels,
                built_a.label_ids,
                built_b.label_ids,
                spacing=spacing,
            )
        else:
            inter = label_intersection_sparse(
                built_a.labels,
                built_b.labels,
                built_a.label_ids,
                built_b.label_ids,
                built_a.boxes,
                built_b.boxes,
                spacing=spacing,
            )
        return convert_array_like(
            inter,
            dtype=self.config.preferred_input_type,
            device=device,
        )


# ---------------------------------------------------------------------------
# Metrics calculators
# ---------------------------------------------------------------------------


class MetricsCalculatorConfig(Configuration):
    """Shared settings for overlap metric calculators."""

    name: str = "metric"


class MetricsCalculator(Configurable[MetricsCalculatorConfig]):
    """Compute one overlap metric from areas and intersection."""

    def __init__(self, config: MetricsCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MetricsCalculatorConfig) -> "MetricsCalculator":
        return cls(config)

    @property
    def metric_name(self) -> str:
        return self.config.name

    def compute(
        self,
        *,
        inter: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError


class IoUMetricsCalculatorConfig(MetricsCalculatorConfig):
    name: str = "iou"


class IoUMetricsCalculator(MetricsCalculator):
    def __init__(self, config: IoUMetricsCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: IoUMetricsCalculatorConfig) -> "IoUMetricsCalculator":
        return cls(config)

    def compute(
        self,
        *,
        inter: Union[np.ndarray, torch.Tensor],
        union: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        return _divide_pairwise(inter, union)


class IoSMetricsCalculatorConfig(MetricsCalculatorConfig):
    name: str = "ios"


class IoSMetricsCalculator(MetricsCalculator):
    def __init__(self, config: IoSMetricsCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: IoSMetricsCalculatorConfig) -> "IoSMetricsCalculator":
        return cls(config)

    def compute(
        self,
        *,
        inter: Union[np.ndarray, torch.Tensor],
        area_a: Union[np.ndarray, torch.Tensor],
        area_b: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(area_a, torch.Tensor):
            denom = torch.minimum(area_a[:, None], area_b[None, :])
        else:
            denom = np.minimum(area_a[:, None], area_b[None, :])
        return _divide_pairwise(inter, denom)


class DiceMetricsCalculatorConfig(MetricsCalculatorConfig):
    name: str = "dice"


class DiceMetricsCalculator(MetricsCalculator):
    def __init__(self, config: DiceMetricsCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: DiceMetricsCalculatorConfig) -> "DiceMetricsCalculator":
        return cls(config)

    def compute(
        self,
        *,
        inter: Union[np.ndarray, torch.Tensor],
        area_a: Union[np.ndarray, torch.Tensor],
        area_b: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        denom = area_a[:, None] + area_b[None, :]
        return _divide_pairwise(2 * inter, denom)


# ---------------------------------------------------------------------------
# Overlap orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapResult:
    """Metric matrices and optional geometry from OverlapCalculator.

    Always returned by ``run``. ``metrics`` is always populated with
    :class:`~vistiq.matrix.MatrixData` values. ``area_a``,
    ``area_b``, ``intersection``, and ``union`` are set only when
    ``return_components=True`` on the config. ``object_ids_a`` and
    ``object_ids_b`` mirror ``region_map`` key order when a map was provided.
    ``annotations`` holds resolved row/column labels shared by metric matrices.
    """

    metrics: dict[str, MatrixData]
    area_a: Optional[Union[np.ndarray, torch.Tensor]] = None
    area_b: Optional[Union[np.ndarray, torch.Tensor]] = None
    intersection: Optional[Union[np.ndarray, torch.Tensor]] = None
    union: Optional[Union[np.ndarray, torch.Tensor]] = None
    object_ids_a: Optional[tuple[str, ...]] = None
    object_ids_b: Optional[tuple[str, ...]] = None
    annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None

    def metric(
        self, name: Optional[str] = None
    ) -> MatrixData:
        """Metric matrix; *name* defaults to the sole metric when only one."""
        if name is None:
            if len(self.metrics) != 1:
                names = ", ".join(sorted(self.metrics))
                raise ValueError(
                    f"metric name required when multiple metrics are present: {names}"
                )
            return next(iter(self.metrics.values()))
        return self.metrics[name]


_PIPELINE_BACKEND_FIELDS = ("preferred_input_type", "preferred_device")


class OverlapCalculatorConfig(Configuration):
    """Configuration for OverlapCalculator.

    Wire a compatible ``builder``, ``area_calculator``, and
    ``intersection_calculator`` for the target representation, or use a preset
    subclass: BoxOverlapCalculatorConfig, MaskOverlapCalculatorConfig, or
    LabelOverlapCalculatorConfig.

    Array backend (``preferred_input_type``, ``preferred_device``) is configured
    on the pipeline child configs, not here. Preset subclasses default all
    children to ``preferred_input_type="torch.Tensor"``; use a different backend
    by reconstructing the container with matching child configs.

    Attributes:
        return_components: When True, include ``area_a``, ``area_b``,
            ``intersection``, and ``union`` on OverlapResult.
        triangle: Triangle mask applied to each metric (see ``vistiq.matrix.types``).
    """

    builder: MatrixBuilderConfig
    area_calculator: AreaCalculatorConfig
    intersection_calculator: IntersectionCalculatorConfig
    metrics_calculators: list[MetricsCalculatorConfig] = Field(
        default_factory=lambda: [IoUMetricsCalculatorConfig()]
    )

    return_components: bool = False
    triangle: int = FULL

    @model_validator(mode="after")
    def _check_pipeline_backend_consistency(self) -> Self:
        """Require identical backend settings across pipeline child configs."""
        reference = self.builder
        for name in ("area_calculator", "intersection_calculator"):
            child = getattr(self, name)
            for field in _PIPELINE_BACKEND_FIELDS:
                ref_val = getattr(reference, field)
                child_val = getattr(child, field)
                if child_val != ref_val:
                    raise ValueError(
                        f"{name}.{field}={child_val!r} does not match "
                        f"builder.{field}={ref_val!r}; pipeline children must agree"
                    )
        return self


class BoxOverlapCalculatorConfig(OverlapCalculatorConfig):
    """Preset for axis-aligned box batches ``(N, 2 * d)`` or ``region_map`` only.

    Pass ``(N, 2 * d)`` arrays to ``run`` (min/max pairs in array axis order), or
    supply ``region_map`` with ``bbox`` on every RegionSpec and omit raw box arrays.
    """

    builder: BoxBuilderConfig = Field(default_factory=BoxBuilderConfig)
    area_calculator: BoxAreaCalculatorConfig = Field(
        default_factory=BoxAreaCalculatorConfig
    )
    intersection_calculator: BoxIntersectionCalculatorConfig = Field(
        default_factory=BoxIntersectionCalculatorConfig
    )


class MaskOverlapCalculatorConfig(OverlapCalculatorConfig):
    """Preset for mask stacks ``(N, *spatial)``."""

    builder: MaskStackBuilderConfig = Field(default_factory=MaskStackBuilderConfig)
    area_calculator: MaskAreaCalculatorConfig = Field(
        default_factory=MaskAreaCalculatorConfig
    )
    intersection_calculator: MaskIntersectionCalculatorConfig = Field(
        default_factory=MaskIntersectionCalculatorConfig
    )


class LabelOverlapCalculatorConfig(OverlapCalculatorConfig):
    """Preset for 2D/3D integer label arrays as ``a`` and ``b``.

    Uses label_areas and label_intersection_linear or label_intersection_sparse
    via LabelIntersectionCalculator. Intersection defaults to ``mode="auto"`` on
    LabelIntersectionCalculatorConfig.
    """

    builder: LabelBuilderConfig = Field(default_factory=LabelBuilderConfig)
    area_calculator: LabelAreaCalculatorConfig = Field(
        default_factory=LabelAreaCalculatorConfig
    )
    intersection_calculator: LabelIntersectionCalculatorConfig = Field(
        default_factory=LabelIntersectionCalculatorConfig
    )


class OverlapCalculator(Configurable[OverlapCalculatorConfig]):
    """Compute overlap metrics by composing builder, area, intersection, and metrics."""

    def __init__(self, config: OverlapCalculatorConfig):
        super().__init__(config)
        self._builder = Configurable.create_from_config(config.builder)
        self._area = Configurable.create_from_config(config.area_calculator)
        self._intersection = Configurable.create_from_config(
            config.intersection_calculator
        )
        self._metrics = Configurable.create_many_from_configs(
            config.metrics_calculators,
            expected_type=MetricsCalculator,
            error_header="Failed to instantiate overlap metrics calculators",
        )

    @classmethod
    def from_config(cls, config: OverlapCalculatorConfig) -> "OverlapCalculator":
        return cls(config)

    @task(name="OverlapCalculator.matrix", task_run_name=generate_name)
    def matrix(
        self,
        result: OverlapResult,
        metric: Optional[str] = None,
    ) -> MatrixData:
        """Return the metric matrix for matrix filters."""
        return result.metric(metric)

    @task(name="OverlapCalculator.run", task_run_name=generate_name)
    def run(
        self,
        a: Any = None,
        b: Any = None,
        *,
        region_map: Optional[tuple[RegionMap, RegionMap]] = None,
        spacing: SpacingLike = None,
        annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None,
        device: Optional[torch.device] = None,
    ) -> OverlapResult:
        """Compute overlap metric(s) between two collections.

        Args:
            a: First collection (boxes ``(N, 2 * d)``, masks ``(N, *spatial)``, or
                label volume for label overlap).
            b: Second collection, same representation as ``a``.
            region_map: Pair of maps keyed by ``object_id``. Defines which
                regions to compare and their ``label_id`` / ``bbox``. Row and
                column order follow map insertion order (dataframe row order from
                region_map_from_dataframe).
            spacing: Physical pixel size per axis (same order as array axes). Signs
                encode acquisition direction; magnitudes scale areas and
                intersections. Ratios (IoU/IoS/Dice) are unchanged under uniform
                scaling.
            annotations: Optional ``(row_labels, col_labels)`` for DataFrame
                output when ``annotate=True``. Overrides ``object_id`` display
                names; lengths must match region counts. Ignored when
                ``annotate=False``.
            device: Torch device override when using tensor backends.

        Returns:
            OverlapResult with :class:`~vistiq.matrix.MatrixData` metrics.
            Geometry fields are included only when ``return_components=True``.
            Use :class:`~vistiq.matrix.MatrixFormatter` to export DataFrames
            or raw arrays.
        """
        logger.info(f"Running OverlapCalculator with config: {self.config}")
        is_label_builder = isinstance(self.config.builder, LabelBuilderConfig)
        is_box_builder = isinstance(self.config.builder, BoxBuilderConfig)

        region_ctx_a: Optional[RegionBuildContext] = None
        region_ctx_b: Optional[RegionBuildContext] = None
        resolved_annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None
        if region_map is not None:
            region_ctx_a = parse_region_map(
                region_map[0],
                require_label_id=is_label_builder,
                require_bbox=is_box_builder,
            )
            region_ctx_b = parse_region_map(
                region_map[1],
                require_label_id=is_label_builder,
                require_bbox=is_box_builder,
            )
            default = (
                tuple(region_map[0].keys()),
                tuple(region_map[1].keys()),
            )
            if annotations is None:
                resolved_annotations = default
            else:
                rows, cols = annotations
                if len(rows) == 0 and len(cols) == 0:
                    resolved_annotations = default
                else:
                    expected_rows, expected_cols = default
                    if len(rows) != len(expected_rows) or len(cols) != len(
                        expected_cols
                    ):
                        raise ValueError(
                            "annotations must match region_map size; "
                            f"got {len(rows)} x {len(cols)}, expected "
                            f"{len(expected_rows)} x {len(expected_cols)}"
                        )
                    resolved_annotations = (
                        tuple(str(value) for value in rows),
                        tuple(str(value) for value in cols),
                    )
        elif annotations is not None:
            rows, cols = annotations
            resolved_annotations = (
                tuple(str(value) for value in rows),
                tuple(str(value) for value in cols),
            )

        if is_box_builder and region_map is None and (a is None or b is None):
            raise ValueError(
                "BoxOverlapCalculator requires box arrays or region_map"
            )
        if is_label_builder and (a is None or b is None):
            raise ValueError(
                "LabelOverlapCalculator requires label volumes for both inputs"
            )
        if not is_box_builder and not is_label_builder and (a is None or b is None):
            raise ValueError("Both inputs are required")

        device = resolve_torch_device(
            device,
            preferred_input_type=self.config.builder.preferred_input_type,
            preferred_device=self.config.builder.preferred_device,
        )
        map_a = region_map[0] if region_map is not None else None
        map_b = region_map[1] if region_map is not None else None
        built_a = self._builder.run(a, region_map=map_a, device=device)
        built_b = self._builder.run(b, region_map=map_b, device=device)
        area_a = self._area.run(built_a, spacing=spacing, device=device)
        area_b = self._area.run(built_b, spacing=spacing, device=device)
        inter = self._intersection.run(
            built_a,
            built_b,
            spacing=spacing,
            device=device,
        )
        union = union_matrix(area_a, area_b, inter=inter)

        raw_metrics: dict[str, Union[np.ndarray, torch.Tensor]] = {}
        for metric_calc in self._metrics:
            matrix = metric_calc.compute(
                inter=inter,
                union=union,
                area_a=area_a,
                area_b=area_b,
            )
            matrix = apply_triangle_mask(matrix, self.config.triangle)
            raw_metrics[metric_calc.metric_name] = matrix

        if resolved_annotations is not None:
            rows, cols = resolved_annotations
            for name, matrix in list(raw_metrics.items()):
                if isinstance(matrix, torch.Tensor):
                    shape = tuple(matrix.shape)
                else:
                    shape = np.asarray(matrix).shape
                if len(rows) != shape[0] or len(cols) != shape[1]:
                    raise ValueError(
                        "annotations must match matrix shape "
                        f"{shape}; got {len(rows)} x {len(cols)}"
                    )

        metric_data = {
            name: MatrixData(matrix=matrix, annotations=resolved_annotations)
            for name, matrix in raw_metrics.items()
        }

        object_ids_a = region_ctx_a.object_ids if region_ctx_a is not None else None
        object_ids_b = region_ctx_b.object_ids if region_ctx_b is not None else None

        if self.config.return_components:
            return OverlapResult(
                metrics=metric_data,
                area_a=area_a,
                area_b=area_b,
                intersection=inter,
                union=union,
                object_ids_a=object_ids_a,
                object_ids_b=object_ids_b,
                annotations=resolved_annotations,
            )
        return OverlapResult(
            metrics=metric_data,
            object_ids_a=object_ids_a,
            object_ids_b=object_ids_b,
            annotations=resolved_annotations,
        )


# ---------------------------------------------------------------------------
# Metric config helpers
# ---------------------------------------------------------------------------

_METRIC_CONFIG_BY_NAME: dict[str, type[MetricsCalculatorConfig]] = {
    "iou": IoUMetricsCalculatorConfig,
    "ios": IoSMetricsCalculatorConfig,
    "dice": DiceMetricsCalculatorConfig,
}


def metrics_calculator_configs(
    metrics: Sequence[str] = ("iou",),
) -> list[MetricsCalculatorConfig]:
    """Build metric calculator configs from metric names."""
    configs: list[MetricsCalculatorConfig] = []
    for metric in metrics:
        key = metric.lower()
        if key not in _METRIC_CONFIG_BY_NAME:
            supported = ", ".join(sorted(_METRIC_CONFIG_BY_NAME))
            raise ValueError(
                f"unsupported overlap metric {metric!r}; supported: {supported}"
            )
        configs.append(_METRIC_CONFIG_BY_NAME[key]())
    return configs


def _register_overlap_preset_configs() -> None:
    """Map preset overlap configs to OverlapCalculator for deserialization."""
    for preset in (
        BoxOverlapCalculatorConfig,
        MaskOverlapCalculatorConfig,
        LabelOverlapCalculatorConfig,
    ):
        Configurable._registry[preset] = OverlapCalculator


_register_overlap_preset_configs()
