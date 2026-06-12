"""Composable overlap calculators (IoU, IoS, Dice) for boxes, masks, and labels.

The pipeline is configured with preset subclasses of :class:`OverlapCalculatorConfig`
(:class:`BoxOverlapCalculatorConfig`, :class:`MaskOverlapCalculatorConfig`,
:class:`LabelOverlapCalculatorConfig`) that wire a compatible builder, area
calculator, and intersection calculator. Array backend (NumPy vs PyTorch, device)
is set on those child configs; all pipeline children must agree.

For coincidence workflows, pass a :data:`RegionMap` per collection so each region
is keyed by a globally unique ``object_id`` with optional ``label_id`` (label
path) and ``bbox`` (box path or mask pruning). Build maps from
:class:`RegionAnalyzer` tables via :func:`region_map_from_dataframe`. Label
overlap requires the original label volumes—not region property DataFrames—as
``a`` and ``b``.

Runtime options on :meth:`OverlapCalculator.run` include ``spacing`` (physical
voxel size aligned with array axes), ``region_map``, and optional
``annotations`` for DataFrame row/column labels when ``annotate=True``.
"""

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

from vistiq.constant.matrix import FULL
from vistiq.core import Configurable, Configuration, generate_name, labels_to_masks
from vistiq.utils import convert_array_like, resolve_torch_device, triangle_valid_mask

logger = logging.getLogger(__name__)

_LABELS_IOU_DENSE_PAIR_FRACTION = 1.01

SpacingLike = Optional[Union[dict[str, float], tuple[float, ...], Sequence[float]]]

Box6 = tuple[float, float, float, float, float, float]

_DEFAULT_BBOX_COLS = (
    "bbox-start-x",
    "bbox-start-y",
    "bbox-start-z",
    "bbox-end-x",
    "bbox-end-y",
    "bbox-end-z",
)


@dataclass(frozen=True)
class RegionSpec:
    """One region entry in a :data:`RegionMap`.

    The map key is the globally unique ``object_id`` (typically from
    :class:`RegionAnalyzer`). ``label_id`` and ``bbox`` describe how to build
    geometry for that object on its side of the comparison.

    Attributes:
        label_id: Integer label value in that side's label volume. Required for
            :class:`LabelOverlapCalculatorConfig` when using ``region_map``.
        bbox: Axis-aligned box ``(x_min, y_min, z_min, x_max, y_max, z_max)``
            in voxel/index coordinates. Required for
            :class:`BoxOverlapCalculatorConfig`; optional for mask intersection
            pruning when ``prune_bboxes=True``.
    """

    label_id: Optional[int] = None
    bbox: Optional[Box6 | Sequence[float]] = None


RegionMap = Mapping[str, RegionSpec]


@dataclass(frozen=True)
class RegionBuildContext:
    """Parsed :class:`RegionMap` for one overlap collection."""

    object_ids: tuple[str, ...]
    label_ids: tuple[int, ...]
    boxes: Optional[np.ndarray]


def _annotations_empty(
    annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]],
) -> bool:
    if annotations is None:
        return True
    rows, cols = annotations
    return len(rows) == 0 and len(cols) == 0


def annotations_from_region_map(
    region_map_a: RegionMap,
    region_map_b: RegionMap,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(index_object_ids, column_object_ids)`` from map key order."""
    return (tuple(region_map_a.keys()), tuple(region_map_b.keys()))


def _resolve_output_annotations(
    annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]],
    region_map: Optional[tuple[RegionMap, RegionMap]],
    *,
    annotate: bool,
) -> Optional[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Resolve DataFrame labels from *region_map* and optional *annotations*.

    When ``annotate`` is False, annotations are ignored. When True and
    *annotations* is missing, ``object_id`` keys from *region_map* are used.
    Explicit *annotations* override those keys but must match region count.
    """
    if region_map is None or not annotate:
        return None if region_map is not None and not annotate else annotations
    default = annotations_from_region_map(region_map[0], region_map[1])
    if _annotations_empty(annotations):
        return default
    rows, cols = annotations
    expected_rows, expected_cols = default
    if len(rows) != len(expected_rows) or len(cols) != len(expected_cols):
        raise ValueError(
            "annotations must match region_map size; "
            f"got {len(rows)} x {len(cols)}, expected "
            f"{len(expected_rows)} x {len(expected_cols)}"
        )
    return (tuple(str(value) for value in rows), tuple(str(value) for value in cols))


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
            if len(spec.bbox) != 6:
                raise ValueError(
                    f"region_map[{object_id!r}].bbox must have length 6; "
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
    bbox_cols: Sequence[str] = _DEFAULT_BBOX_COLS,
) -> dict[str, RegionSpec]:
    """Build a :data:`RegionMap` from a :class:`RegionAnalyzer` table.

    Expects columns ``object_id``, ``label``, and optionally the six bbox
    columns ``bbox-start-{x,y,z}`` / ``bbox-end-{x,y,z}``. Row order becomes
    matrix row/column order when ``annotate=True`` and no custom annotations
    are passed.

    After :class:`RegionFilter`, call ``reset_index()`` so ``object_id`` is a
    column rather than the index.

    Args:
        df: Region property table (e.g. ``l_accepted.reset_index()``).
        object_id_col: Column holding unique object identifiers.
        label_col: Column holding integer label ids in the label volume.
        bbox_cols: Bbox column names; omitted bboxes are stored as ``None``.

    Returns:
        Mapping from ``object_id`` to :class:`RegionSpec`.
    """
    if object_id_col not in df.columns:
        raise KeyError(
            f"column {object_id_col!r} not found; available: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise KeyError(
            f"column {label_col!r} not found; available: {list(df.columns)}"
        )
    has_bbox = all(col in df.columns for col in bbox_cols)
    region_map: dict[str, RegionSpec] = {}
    for _, row in df.iterrows():
        object_id = str(row[object_id_col])
        if object_id in region_map:
            raise ValueError(f"duplicate object_id {object_id!r} in dataframe")
        bbox: Optional[Box6] = None
        if has_bbox:
            bbox = tuple(float(row[col]) for col in bbox_cols)  # type: ignore[assignment]
        region_map[object_id] = RegionSpec(
            label_id=int(row[label_col]),
            bbox=bbox,
        )
    return region_map


def masks_from_label_volume(
    labels: np.ndarray,
    region_map: RegionMap,
) -> np.ndarray:
    """Build ``(N, *spatial)`` bool stack from a label volume and region map."""
    if labels.ndim not in (2, 3):
        raise ValueError(
            f"label volumes must be 2D or 3D; got ndim={labels.ndim}"
        )
    masks: list[np.ndarray] = []
    for object_id, spec in region_map.items():
        if spec.label_id is None:
            raise ValueError(
                f"region_map[{object_id!r}] is missing label_id"
            )
        masks.append(labels == spec.label_id)
    if not masks:
        return np.empty((0,) + labels.shape, dtype=bool)
    return np.stack(masks, axis=0)


def boxes_from_region_map(region_map: RegionMap) -> np.ndarray:
    """Stack ``(N, 6)`` boxes from map values."""
    ctx = parse_region_map(region_map, require_bbox=True)
    assert ctx.boxes is not None
    return ctx.boxes

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _spacing_tuple(
    spacing: SpacingLike, *, n_dims: int
) -> Optional[tuple[float, ...]]:
    if spacing is None:
        return None
    if isinstance(spacing, dict):
        values = tuple(float(spacing[k]) for k in sorted(spacing.keys()))
    else:
        values = tuple(float(v) for v in spacing)
    if len(values) < n_dims:
        raise ValueError(
            f"spacing length ({len(values)}) must be at least {n_dims}"
        )
    return values[:n_dims]


def _spacing_magnitudes(
    spacing: SpacingLike, *, n_dims: int
) -> Optional[tuple[float, ...]]:
    """Per-axis physical voxel size; sign on *spacing* encodes axis direction only."""
    sp = _spacing_tuple(spacing, n_dims=n_dims)
    if sp is None:
        return None
    return tuple(abs(value) for value in sp)


def _voxel_volume(spacing: SpacingLike, *, n_spatial: int) -> float:
    sp = _spacing_magnitudes(spacing, n_dims=n_spatial)
    if sp is None:
        return 1.0
    volume = 1.0
    for value in sp:
        volume *= value
    return volume


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


def box_areas_numpy(
    boxes: np.ndarray, spacing: SpacingLike = None
) -> np.ndarray:
    """Per-box volumes for ``(N, 6)`` boxes ``(x_min, y_min, z_min, x_max, y_max, z_max)``."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError(f"boxes must have shape (N, 6); got {boxes.shape}")
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    sp = _spacing_magnitudes(spacing, n_dims=3)
    extents = np.maximum(boxes[:, 3:6] - boxes[:, 0:3], 0.0)
    if sp is not None:
        extents = extents * np.asarray(sp, dtype=np.float64)
    return np.prod(extents, axis=1)


def box_intersection_numpy(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Pairwise intersection volumes for 3D axis-aligned boxes."""
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    n_a, n_b = boxes_a.shape[0], boxes_b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float32)

    x_min_a, y_min_a, z_min_a, x_max_a, y_max_a, z_max_a = boxes_a.T
    x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b = boxes_b.T

    x_min = np.maximum(x_min_a[:, None], x_min_b[None, :])
    x_max = np.minimum(x_max_a[:, None], x_max_b[None, :])
    y_min = np.maximum(y_min_a[:, None], y_min_b[None, :])
    y_max = np.minimum(y_max_a[:, None], y_max_b[None, :])
    z_min = np.maximum(z_min_a[:, None], z_min_b[None, :])
    z_max = np.minimum(z_max_a[:, None], z_max_b[None, :])

    dx = np.clip(x_max - x_min, 0.0, None)
    dy = np.clip(y_max - y_min, 0.0, None)
    dz = np.clip(z_max - z_min, 0.0, None)
    return (dx * dy * dz).astype(np.float32, copy=False)


def box_areas_torch(
    boxes: torch.Tensor, spacing: SpacingLike = None
) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError(f"boxes must have shape (N, 6); got {boxes.shape}")
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32, device=boxes.device)
    extents = torch.clamp(boxes[:, 3:6] - boxes[:, 0:3], min=0.0).to(torch.float32)
    sp = _spacing_magnitudes(spacing, n_dims=3)
    if sp is not None:
        scale = convert_array_like(
            sp, dtype="torch.Tensor", device=boxes.device
        ).to(dtype=extents.dtype)
        extents = extents * scale.unsqueeze(0)
    return extents.prod(dim=1)


def box_intersection_torch(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    n_a, n_b = boxes_a.shape[0], boxes_b.shape[0]
    if n_a == 0 or n_b == 0:
        return torch.empty((n_a, n_b), dtype=torch.float32, device=boxes_a.device)

    x_min_a, y_min_a, z_min_a, x_max_a, y_max_a, z_max_a = boxes_a.T
    x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b = boxes_b.T

    x_min = torch.maximum(x_min_a[:, None], x_min_b[None, :])
    x_max = torch.minimum(x_max_a[:, None], x_max_b[None, :])
    y_min = torch.maximum(y_min_a[:, None], y_min_b[None, :])
    y_max = torch.minimum(y_max_a[:, None], y_max_b[None, :])
    z_min = torch.maximum(z_min_a[:, None], z_min_b[None, :])
    z_max = torch.minimum(z_max_a[:, None], z_max_b[None, :])

    inter = torch.clamp(x_max - x_min, min=0.0)
    inter = inter * torch.clamp(y_max - y_min, min=0.0)
    inter = inter * torch.clamp(z_max - z_min, min=0.0)
    return inter.to(dtype=torch.float32)


def _mask_spatial_axes(mask_ndim: int) -> tuple[int, ...]:
    return tuple(range(1, mask_ndim))


def mask_areas_numpy(
    masks: np.ndarray, spacing: SpacingLike = None
) -> np.ndarray:
    """Per-mask voxel counts scaled by physical voxel volume when *spacing* is set."""
    masks = np.asarray(masks)
    if masks.ndim < 3:
        raise ValueError(f"mask stack must be at least 3D; got ndim={masks.ndim}")
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1, dtype=np.float64)
    areas *= _voxel_volume(spacing, n_spatial=masks.ndim - 1)
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
    return inter * _voxel_volume(spacing, n_spatial=masks_a.ndim - 1)


def mask_areas_torch(
    masks: torch.Tensor, spacing: SpacingLike = None
) -> torch.Tensor:
    if masks.ndim < 3:
        raise ValueError(f"mask stack must be at least 3D; got ndim={masks.ndim}")
    flat = masks.reshape(masks.shape[0], -1).to(dtype=torch.float32)
    areas = flat.sum(dim=1)
    areas = areas * _voxel_volume(spacing, n_spatial=masks.ndim - 1)
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
    scale = _voxel_volume(spacing, n_spatial=masks_a.ndim - 1)
    if scale != 1.0:
        inter = inter * scale
    return inter


def _boxes_from_masks_numpy(masks: np.ndarray) -> np.ndarray:
    """Axis-aligned boxes ``(N, 6)`` from a boolean mask stack."""
    n = masks.shape[0]
    if n == 0:
        return np.empty((0, 6), dtype=np.float32)
    spatial_ndim = masks.ndim - 1
    boxes = np.zeros((n, 2 * spatial_ndim), dtype=np.float32)
    for index in range(n):
        coords = np.argwhere(masks[index])
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        boxes[index, :spatial_ndim] = mins
        boxes[index, spatial_ndim:] = maxs
    if spatial_ndim == 2:
        z_pad = np.zeros((n, 1), dtype=np.float32)
        ones = np.ones((n, 1), dtype=np.float32)
        boxes = np.concatenate(
            [boxes[:, 1:2], boxes[:, 0:1], z_pad, boxes[:, 3:4], boxes[:, 2:3], ones],
            axis=1,
        )
    return boxes


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
# Matrix builders
# ---------------------------------------------------------------------------


class MatrixBuilderConfig(Configuration):
    """Shared settings for overlap input builders."""

    preferred_input_type: Literal["numpy", "torch.Tensor"] = "torch.Tensor"
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
    """Configuration for :class:`BoxBuilder`."""


class BoxBuilder(MatrixBuilder):
    """Validate and convert box arrays ``(N, 6)``."""

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
            boxes = boxes_from_region_map(region_map)
        else:
            if data is None:
                raise ValueError(
                    "BoxBuilder requires box array data or region_map"
                )
            boxes = np.asarray(data)
        boxes = convert_array_like(boxes, dtype=dtype, device=device)
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            raise ValueError(f"boxes must have shape (N, 6); got {getattr(boxes, 'shape', None)}")
        return boxes


class MaskStackBuilderConfig(MatrixBuilderConfig):
    """Configuration for :class:`MaskStackBuilder`."""


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


class LabelMaskBuilderConfig(MatrixBuilderConfig):
    """Configuration for :class:`LabelMaskBuilder`.

    Attributes:
        label_order: Used only when ``region_map`` is **not** passed to
            :meth:`OverlapCalculator.run`. ``"regionprops"`` follows
            regionprops discovery order (default preset, pairs with
            ``prune_bboxes=True``); ``"unique"`` uses sorted positive
            ``np.unique`` ids (pairs with dense ``prune_bboxes=False``).
    """

    label_order: Literal["regionprops", "unique"] = "regionprops"


class LabelMaskBuilder(MatrixBuilder):
    """Convert a label volume to a stack of instance masks."""

    def __init__(self, config: LabelMaskBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: LabelMaskBuilderConfig) -> "LabelMaskBuilder":
        return cls(config)

    def run(
        self,
        data: Any,
        *,
        region_map: Optional[RegionMap] = None,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, pd.DataFrame):
            raise TypeError(
                "LabelMaskBuilder expects a 2D/3D integer label volume, not a "
                "region property DataFrame. Pass the labeled image array with "
                "region_map=(map_a, map_b), or use BoxOverlapCalculatorConfig."
            )
        if data is None:
            raise ValueError(
                "LabelMaskBuilder requires a label volume when building masks"
            )
        labels = np.asarray(data)
        if labels.ndim not in (2, 3):
            raise ValueError(
                f"label volumes must be 2D or 3D; got ndim={labels.ndim}"
            )
        if labels.dtype == object or not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(
                f"label volumes must have an integer dtype; got {labels.dtype}. "
                "Region property tables are not valid input for "
                "LabelOverlapCalculatorConfig."
            )
        if region_map is not None:
            parse_region_map(region_map, require_label_id=True)
            masks = masks_from_label_volume(labels, region_map)
        elif self.config.label_order == "unique":
            masks = labels_to_masks(labels)
        else:
            from vistiq.analysis.coincidence import _label_ids_and_boxes

            ids, _ = _label_ids_and_boxes(labels)
            if len(ids) == 0:
                masks = np.empty((0,) + labels.shape, dtype=bool)
            else:
                masks = np.stack([labels == label_id for label_id in ids], axis=0)
        dtype = self.config.preferred_input_type
        return convert_array_like(masks, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Area calculators
# ---------------------------------------------------------------------------


class AreaCalculatorConfig(Configuration):
    """Shared settings for per-instance area calculators."""

    preferred_input_type: Literal["numpy", "torch.Tensor"] = "torch.Tensor"
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
    """Configuration for :class:`BoxAreaCalculator`."""


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
            return box_areas_torch(boxes, spacing)
        return box_areas_numpy(np.asarray(boxes), spacing)


class MaskAreaCalculatorConfig(AreaCalculatorConfig):
    """Configuration for :class:`MaskAreaCalculator`."""


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


# ---------------------------------------------------------------------------
# Intersection calculators
# ---------------------------------------------------------------------------


class IntersectionCalculatorConfig(Configuration):
    """Shared settings for pairwise intersection calculators."""

    preferred_input_type: Literal["numpy", "torch.Tensor"] = "torch.Tensor"
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
    """Configuration for :class:`BoxIntersectionCalculator`."""


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
            inter = box_intersection_torch(boxes_a, boxes_b)
            if spacing is not None:
                inter = inter * _voxel_volume(spacing, n_spatial=3)
            return inter
        inter = box_intersection_numpy(np.asarray(boxes_a), np.asarray(boxes_b))
        if spacing is not None:
            inter = inter * _voxel_volume(spacing, n_spatial=3)
        return inter


class MaskIntersectionCalculatorConfig(IntersectionCalculatorConfig):
    """Configuration for :class:`MaskIntersectionCalculator`.

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

        bbox_inter = box_intersection_numpy(boxes_a, boxes_b)
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
                inter *= _voxel_volume(spacing, n_spatial=masks_a_np.ndim - 1)
            out[i, j] = inter
        return convert_array_like(out, dtype=dtype, device=device)


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
    """Geometry components and metric matrices from :class:`OverlapCalculator`.

    Returned when ``return_components=True`` on the config. ``object_ids_a`` and
    ``object_ids_b`` mirror ``region_map`` key order when a map was provided.
    """

    area_a: Union[np.ndarray, torch.Tensor]
    area_b: Union[np.ndarray, torch.Tensor]
    intersection: Union[np.ndarray, torch.Tensor]
    union: Union[np.ndarray, torch.Tensor]
    metrics: dict[str, Union[np.ndarray, torch.Tensor]]
    object_ids_a: Optional[tuple[str, ...]] = None
    object_ids_b: Optional[tuple[str, ...]] = None


_PIPELINE_BACKEND_FIELDS = ("preferred_input_type", "preferred_device")


class OverlapCalculatorConfig(Configuration):
    """Configuration for :class:`OverlapCalculator`.

    Wire a compatible ``builder``, ``area_calculator``, and
    ``intersection_calculator`` for the target representation, or use a preset
    subclass: :class:`BoxOverlapCalculatorConfig`,
    :class:`MaskOverlapCalculatorConfig`, or :class:`LabelOverlapCalculatorConfig`.

    Array backend (``preferred_input_type``, ``preferred_device``) is configured
    on the pipeline child configs, not here. Preset subclasses default all
    children to ``preferred_input_type="torch.Tensor"``; use a different backend
    by reconstructing the container with matching child configs.

    Attributes:
        return_components: When True, :meth:`OverlapCalculator.run` returns
            :class:`OverlapResult` instead of formatted metric matrix(es).
        triangle: Triangle mask applied to each metric (see ``vistiq.constant.matrix``).
        output_type: ``"dataframe"``, ``"np.ndarray"``, or ``"torch.Tensor"``.
        annotate: When True and ``output_type="dataframe"``, label rows/columns.
            With ``region_map``, defaults to ``object_id`` keys; explicit
            ``annotations`` override display labels (length must match).
    """

    builder: MatrixBuilderConfig
    area_calculator: AreaCalculatorConfig
    intersection_calculator: IntersectionCalculatorConfig
    metrics_calculators: list[MetricsCalculatorConfig] = Field(
        default_factory=lambda: [IoUMetricsCalculatorConfig()]
    )

    return_components: bool = False
    triangle: int = FULL
    output_type: Literal["np.ndarray", "torch.Tensor", "dataframe"] = "np.ndarray"
    annotate: bool = True

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
    """Preset for axis-aligned box batches ``(N, 6)`` or ``region_map`` only.

    Pass ``(N, 6)`` arrays to :meth:`OverlapCalculator.run`, or supply
    ``region_map`` with ``bbox`` on every :class:`RegionSpec` and omit raw
    box arrays.
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
    """Preset for label volumes (label builder → mask area/intersection path).

    Requires 2D/3D integer label arrays as ``a`` and ``b``. With ``region_map``,
    masks are built per ``object_id`` using each spec's ``label_id``; bboxes
    enable pruning when ``prune_bboxes=True`` (default on this preset).
    """

    builder: LabelMaskBuilderConfig = Field(
        default_factory=lambda: LabelMaskBuilderConfig(label_order="regionprops")
    )
    area_calculator: MaskAreaCalculatorConfig = Field(
        default_factory=MaskAreaCalculatorConfig
    )
    intersection_calculator: MaskIntersectionCalculatorConfig = Field(
        default_factory=lambda: MaskIntersectionCalculatorConfig(prune_bboxes=True)
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

    def _format_matrix(
        self,
        matrix: Union[np.ndarray, torch.Tensor],
        annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor, pd.DataFrame]:
        if self.config.output_type == "dataframe":
            if isinstance(matrix, torch.Tensor):
                values = matrix.detach().cpu().numpy()
            else:
                values = matrix
            if self.config.annotate:
                if annotations is None:
                    raise ValueError(
                        "annotations must be provided when annotate is True"
                    )
                if (
                    len(annotations[0]) != values.shape[0]
                    or len(annotations[1]) != values.shape[1]
                ):
                    raise ValueError(
                        "annotations must match matrix shape "
                        f"{values.shape}; got {len(annotations[0])} x {len(annotations[1])}"
                    )
                columns = [str(value) for value in annotations[1]]
                index = [str(value) for value in annotations[0]]
                return pd.DataFrame(values, columns=columns, index=index)
            return pd.DataFrame(values)
        if self.config.output_type == "np.ndarray":
            if isinstance(matrix, torch.Tensor):
                return matrix.detach().cpu().numpy()
            return matrix
        if self.config.output_type == "torch.Tensor":
            if isinstance(matrix, np.ndarray):
                tensor = torch.from_numpy(np.ascontiguousarray(matrix))
                return tensor.to(device) if device is not None else tensor
            return matrix
        return matrix

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
    ) -> Union[
        np.ndarray,
        torch.Tensor,
        pd.DataFrame,
        dict[str, Union[np.ndarray, torch.Tensor, pd.DataFrame]],
        OverlapResult,
    ]:
        """Compute overlap metric(s) between two collections.

        Args:
            a: First collection (boxes ``(N, 6)``, masks ``(N, *spatial)``, or
                label volume for :class:`LabelOverlapCalculatorConfig`).
            b: Second collection, same representation as ``a``.
            region_map: Pair of maps keyed by ``object_id``. Defines which
                regions to compare and their ``label_id`` / ``bbox``. Row and
                column order follow map insertion order (dataframe row order from
                :func:`region_map_from_dataframe`).
            spacing: Physical voxel size ``(dz, dy, dx)`` aligned with array
                axes (same convention as ``metadata["scale"]`` from
                :class:`RegionAnalyzer`). A negative component indicates
                acquisition direction along that axis; magnitudes are used for
                area and intersection scaling. IoU/IoS/Dice ratios are
                unchanged under uniform scaling.
            annotations: Optional ``(row_labels, col_labels)`` for DataFrame
                output when ``annotate=True``. Overrides ``object_id`` display
                names; lengths must match region counts. Ignored when
                ``annotate=False``.
            device: Torch device override when using tensor backends.

        Returns:
            A single matrix, dict of named matrices (multiple metrics),
            :class:`OverlapResult` when ``return_components=True``, formatted
            per ``output_type`` and ``annotate``.
        """
        is_label_builder = isinstance(self.config.builder, LabelMaskBuilderConfig)
        is_box_builder = isinstance(self.config.builder, BoxBuilderConfig)

        region_ctx_a: Optional[RegionBuildContext] = None
        region_ctx_b: Optional[RegionBuildContext] = None
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
            annotations = _resolve_output_annotations(
                annotations,
                region_map,
                annotate=self.config.annotate,
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
        intersection_kwargs: dict[str, Any] = {}
        if isinstance(self._intersection, MaskIntersectionCalculator):
            if region_ctx_a is not None:
                intersection_kwargs["boxes_a"] = region_ctx_a.boxes
            if region_ctx_b is not None:
                intersection_kwargs["boxes_b"] = region_ctx_b.boxes
        inter = self._intersection.run(
            built_a,
            built_b,
            spacing=spacing,
            device=device,
            **intersection_kwargs,
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

        object_ids_a = region_ctx_a.object_ids if region_ctx_a is not None else None
        object_ids_b = region_ctx_b.object_ids if region_ctx_b is not None else None

        if self.config.return_components:
            return OverlapResult(
                area_a=area_a,
                area_b=area_b,
                intersection=inter,
                union=union,
                metrics=raw_metrics,
                object_ids_a=object_ids_a,
                object_ids_b=object_ids_b,
            )

        if len(raw_metrics) == 1:
            (only_matrix,) = raw_metrics.values()
            return self._format_matrix(only_matrix, annotations, device=device)

        return {
            name: self._format_matrix(matrix, annotations, device=device)
            for name, matrix in raw_metrics.items()
        }


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
    """Map preset overlap configs to :class:`OverlapCalculator` for deserialization."""
    for preset in (
        BoxOverlapCalculatorConfig,
        MaskOverlapCalculatorConfig,
        LabelOverlapCalculatorConfig,
    ):
        Configurable._registry[preset] = OverlapCalculator


_register_overlap_preset_configs()
