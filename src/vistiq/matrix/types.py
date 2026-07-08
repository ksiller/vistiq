"""Matrix types, triangle-selection constants, and pure helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

# Bitmask constants for square matrix region selection.
# Row index = i, column index = j. Atomic flags combine with bitwise OR.
DIAGONAL = 1
LOWER_ND = 2
UPPER_ND = 4
LOWER = DIAGONAL | LOWER_ND
UPPER = DIAGONAL | UPPER_ND
OFF_DIAGONAL = LOWER_ND | UPPER_ND
FULL = DIAGONAL | LOWER_ND | UPPER_ND

MatrixArray = Union[np.ndarray, torch.Tensor]
ArrayBackend = Literal["np.ndarray", "torch.Tensor"]
MatrixAnnotations = tuple[tuple[Any, ...], ...]
AnnotationFactory = Callable[[tuple[int, ...]], MatrixAnnotations]
MatrixFormatOutput = Literal["np.ndarray", "torch.Tensor", "dataframe"]
MatrixContainer = Union[np.ndarray, torch.Tensor, pd.DataFrame, pd.Series]


def default_matrix_annotations(shape: tuple[int, ...]) -> MatrixAnnotations:
    """Integer axis labels ``0..n-1`` for each dimension."""
    return tuple(tuple(range(size)) for size in shape)


def _validate_matrix_data(matrix: MatrixArray, annotations: MatrixAnnotations | None) -> None:
    if isinstance(matrix, torch.Tensor):
        ndim = matrix.ndim
        shape = tuple(matrix.shape)
    else:
        arr = np.asarray(matrix)
        ndim = arr.ndim
        shape = arr.shape
    if ndim < 1:
        raise ValueError(f"matrix must have ndim >= 1; got ndim={ndim}")
    if annotations is None:
        return
    if len(annotations) != ndim:
        raise ValueError(
            f"annotations must have one label tuple per axis; "
            f"got {len(annotations)} for ndim={ndim}"
        )
    for axis, labels in enumerate(annotations):
        if len(labels) != shape[axis]:
            raise ValueError(
                f"annotations axis {axis} length {len(labels)} "
                f"does not match matrix shape {shape[axis]}"
            )


@dataclass(frozen=True)
class MatrixData:
    """N-dimensional matrix (``ndim >= 1``) with optional per-axis labels."""

    matrix: MatrixArray
    annotations: MatrixAnnotations | None = None

    def __post_init__(self) -> None:
        _validate_matrix_data(self.matrix, self.annotations)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.matrix, torch.Tensor):
            return tuple(self.matrix.shape)
        return np.asarray(self.matrix).shape

    @property
    def ndim(self) -> int:
        return len(self.shape)


def matrix_to_numpy(value: MatrixArray | MatrixData | pd.DataFrame | ArrayLike) -> np.ndarray:
    if isinstance(value, MatrixData):
        value = value.matrix
    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=float)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def as_matrix_data(
    value: MatrixData | pd.DataFrame | pd.Series | MatrixArray,
    *,
    annotations: MatrixAnnotations | None = None,
) -> MatrixData:
    """Coerce *value* to :class:`MatrixData`.

    For :class:`~pandas.DataFrame` and :class:`~pandas.Series`, labels are taken
    from the pandas index (and columns for DataFrames). The *annotations*
    argument is ignored for pandas inputs.
    """
    if isinstance(value, MatrixData):
        if annotations is not None and value.annotations != annotations:
            return MatrixData(matrix=value.matrix, annotations=annotations)
        return value
    if isinstance(value, pd.DataFrame):
        if value.ndim != 2:
            raise ValueError("as_matrix_data requires a 2-D DataFrame")
        return MatrixData(
            matrix=value.to_numpy(dtype=float),
            annotations=(tuple(value.index), tuple(value.columns)),
        )
    if isinstance(value, pd.Series):
        return MatrixData(
            matrix=value.to_numpy(dtype=float),
            annotations=(tuple(value.index),),
        )
    return MatrixData(matrix=value, annotations=annotations)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize index coordinates to shape ``(N, ndim)``."""
    array = np.asarray(coords, dtype=np.int64)
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array


def annotations_at_coords(
    annotations: MatrixAnnotations,
    coords: np.ndarray,
) -> MatrixAnnotations:
    """Project per-axis labels onto selected matrix coordinates."""
    normalized = normalize_coords(coords)
    if normalized.shape[1] != len(annotations):
        raise ValueError(
            f"coords width {normalized.shape[1]} does not match "
            f"annotation axes {len(annotations)}"
        )
    return tuple(
        tuple(np.asarray(annotations[axis])[normalized[:, axis]])
        for axis in range(len(annotations))
    )


def composite_matrix_annotations(
    projected: MatrixAnnotations,
    *,
    separator: str = "|",
) -> MatrixAnnotations:
    """Merge projected per-axis labels into one annotation axis for 1-D results."""
    if not projected:
        return ((),)
    if len(projected) == 1:
        return (tuple(projected[0]),)
    count = len(projected[0])
    return (
        tuple(
            separator.join(str(projected[axis][index]) for axis in range(len(projected)))
            for index in range(count)
        ),
    )


def resolve_matrix_annotations(
    data: MatrixData,
    *,
    annotate: bool,
    annotation_factory: AnnotationFactory | None = None,
) -> MatrixAnnotations | None:
    if not annotate:
        return None
    if data.annotations is not None:
        return data.annotations
    factory = annotation_factory or default_matrix_annotations
    return factory(data.shape)


def label_index(labels: Sequence[Any]) -> dict[Any, int]:
    return {label: index for index, label in enumerate(labels)}


def annotations_after_aggregate(
    annotations: MatrixAnnotations | None,
    axis: int,
) -> MatrixAnnotations | None:
    if annotations is None:
        return None
    return tuple(label for index, label in enumerate(annotations) if index != axis)


def ordered_union(existing: list[Any], new_items: list[Any]) -> list[Any]:
    seen = set(existing)
    for item in new_items:
        if item not in seen:
            existing.append(item)
            seen.add(item)
    return existing


def square_matrix(data: MatrixData) -> MatrixData:
    """Reindex a 2-D matrix to shared row/column labels (ordered union)."""
    if data.ndim != 2:
        raise ValueError(f"square_matrix requires ndim=2; got ndim={data.ndim}")
    values = matrix_to_numpy(data)
    if data.annotations is None:
        if values.shape[0] != values.shape[1]:
            raise ValueError(
                "square_matrix requires annotations when row and column sizes differ"
            )
        return data
    rows, cols = data.annotations
    nodes = ordered_union(list(rows), list(cols))
    row_map = label_index(rows)
    col_map = label_index(cols)
    node_map = label_index(nodes)
    squared = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
    for left in rows:
        for right in cols:
            if left not in node_map or right not in node_map:
                continue
            squared[node_map[left], node_map[right]] = values[
                row_map[left], col_map[right]
            ]
    return MatrixData(
        matrix=squared,
        annotations=(tuple(nodes), tuple(nodes)),
    )
