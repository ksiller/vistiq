"""Unary matrix operations on analysis results (e.g. distance matrices)."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union, Any

import numpy as np
import torch
import pandas as pd
from numpy.typing import ArrayLike
from prefect import task
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vistiq.constant.matrix import DIAGONAL, FULL, LOWER_ND, UPPER_ND
from vistiq.core import Configurable, Configuration, generate_name
from vistiq.utils import convert_array_like, prepare_matrix_values, resolve_torch_device

logger = logging.getLogger(__name__)


def _ordered_union(existing: list[Any], new_items: list[Any]) -> list[Any]:
    seen = set(existing)
    for item in new_items:
        if item not in seen:
            existing.append(item)
            seen.add(item)
    return existing


def _matrix_values(matrix: Union[pd.DataFrame, ArrayLike]) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy(dtype=float)
    return np.asarray(matrix, dtype=float)


def _square_dataframe(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reindex to shared row/column labels (ordered union of index and columns)."""
    nodes: list[Any] = []
    nodes = _ordered_union(nodes, list(matrix.index))
    nodes = _ordered_union(nodes, list(matrix.columns))
    return matrix.reindex(index=nodes, columns=nodes)


def _triangle_valid_mask_numpy(values: np.ndarray, flags: int) -> Optional[np.ndarray]:
    """Boolean mask of allowed triangle regions; ``None`` if unrestricted or non-square."""
    if flags == FULL:
        return None
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return None

    n = values.shape[0]
    row_idx = np.arange(n)[:, None]
    col_idx = np.arange(n)[None, :]
    valid = np.zeros((n, n), dtype=bool)
    if flags & DIAGONAL:
        valid |= row_idx == col_idx
    if flags & LOWER_ND:
        valid |= row_idx > col_idx
    if flags & UPPER_ND:
        valid |= row_idx < col_idx
    return valid


def _mask_triangle(values: np.ndarray, flags: int, fill_value: float) -> np.ndarray:
    """Zero out (or fill) cells outside the configured triangle region."""
    mask = _triangle_valid_mask_numpy(values, flags)
    if mask is None:
        return values
    result = np.array(values, copy=True)
    result[~mask] = fill_value
    return result


class MatrixCombinerConfig(Configuration):
    """Configuration for :class:`MatrixCombiner`.

    Attributes:
        fill_value: Value for global matrix cells not covered by any input block,
            and for cells masked out by :attr:`triangle`.
        symmetrize: When ``True``, merge ``(i, j)`` and ``(j, i)`` with
            element-wise ``nanmax`` on the squared global matrix before masking.
        triangle: Bitmask of retained regions on the square global matrix; see
            :mod:`vistiq.constant.matrix`. ``FULL`` keeps all cells.
    """

    fill_value: float = float("nan")
    symmetrize: bool = True
    triangle: int = FULL


class MatrixCombiner(Configurable[MatrixCombinerConfig]):
    """Assemble labeled pairwise blocks into one global matrix."""

    def __init__(self, config: MatrixCombinerConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixCombinerConfig) -> "MatrixCombiner":
        return cls(config)

    @task(name="MatrixCombiner.run", task_run_name=generate_name)
    def run(
        self,
        matrices: list[Union[pd.DataFrame, ArrayLike]],
        *,
        object_ids: Optional[list[tuple[list[Any], list[Any]]]] = None,
    ) -> pd.DataFrame:
        """Combine pairwise metric blocks into a single labeled DataFrame.

        Args:
            matrices: Pairwise metric blocks (e.g. IoS between two object lists).
            object_ids: Optional per-block ``(row_labels, col_labels)``. When set,
                ``len(object_ids)`` must equal ``len(matrices)``, and each label
                list length must match the corresponding matrix shape.

        Returns:
            DataFrame with union row index and union column labels; unfilled cells
            use :attr:`~MatrixCombinerConfig.fill_value`.
        """
        if not matrices:
            return pd.DataFrame(dtype=float)

        if object_ids is not None and len(object_ids) != len(matrices):
            raise ValueError(
                f"object_ids length {len(object_ids)} does not match "
                f"matrices length {len(matrices)}"
            )

        blocks: list[pd.DataFrame] = []
        global_rows: list[Any] = []
        global_cols: list[Any] = []

        for index, matrix in enumerate(matrices):
            labels = object_ids[index] if object_ids is not None else None
            if labels is not None:
                rows, cols = labels
                values = _matrix_values(matrix)
                if values.ndim != 2:
                    raise ValueError(f"matrix {index} must be 2-D; got shape {values.shape}")
                if len(rows) != values.shape[0] or len(cols) != values.shape[1]:
                    raise AssertionError(
                        f"object_ids[{index}] lengths ({len(rows)}, {len(cols)}) "
                        f"do not match matrix shape {values.shape}"
                    )
                block = pd.DataFrame(values, index=list(rows), columns=list(cols))
            elif isinstance(matrix, pd.DataFrame):
                block = matrix.astype(float, copy=False)
            else:
                raise ValueError(
                    f"matrix {index} is not a DataFrame; pass object_ids for array inputs"
                )

            global_rows = _ordered_union(global_rows, list(block.index))
            global_cols = _ordered_union(global_cols, list(block.columns))
            blocks.append(block)

        combined = pd.DataFrame(
            self.config.fill_value, index=global_rows, columns=global_cols, dtype=float
        )
        for block in blocks:
            combined.update(block.astype(float, copy=False))

        if self.config.symmetrize or self.config.triangle != FULL:
            combined = _square_dataframe(combined)
            values = combined.to_numpy(dtype=float)
            if self.config.symmetrize:
                values = _symmetrize_max(values)
            values = _mask_triangle(values, self.config.triangle, self.config.fill_value)
            combined = pd.DataFrame(values, index=combined.index, columns=combined.columns)
        return combined

def _symmetrize_max(values: np.ndarray) -> np.ndarray:
    """Merge ``(i, j)`` and ``(j, i)`` with element-wise ``nanmax``."""
    sym = np.array(values, copy=True)
    n = sym.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sym[i, j], sym[j, i]
            if np.isnan(a) and np.isnan(b):
                continue
            merged = np.nanmax([a, b])
            sym[i, j] = sym[j, i] = merged
    return sym


def upper_triangle_adjacency(
    matrix: pd.DataFrame,
    *,
    symmetrize: bool = True,
    include_diagonal: bool = False,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Square adjacency matrix with only the upper triangle (for undirected graphs).

    Reindexes to a shared node order (union of index and columns), optionally
    merges duplicate directed edges with ``nanmax``, then zeroes the lower
    triangle and diagonal (unless *include_diagonal*).

    Suitable for :func:`networkx.from_pandas_adjacency` on an undirected graph.

    Args:
        matrix: Weighted adjacency (e.g. output of :func:`combine_pairwise_matrices`).
        symmetrize: When ``True``, merge ``(i, j)`` and ``(j, i)`` before masking.
        include_diagonal: Keep diagonal entries; default excludes self-edges.
        fill_value: Value for masked cells (use ``0.0`` for NetworkX).

    Returns:
        Square DataFrame with weights only in the upper triangle.
    """
    nodes: list[Any] = []
    nodes = _ordered_union(nodes, list(matrix.index))
    nodes = _ordered_union(nodes, list(matrix.columns))
    square = matrix.reindex(index=nodes, columns=nodes)
    values = square.to_numpy(dtype=float)
    if symmetrize:
        values = _symmetrize_max(values)

    n = values.shape[0]
    k = 0 if include_diagonal else 1
    result = np.full((n, n), fill_value, dtype=float)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=k)
    result[upper_mask] = values[upper_mask]
    return pd.DataFrame(result, index=nodes, columns=nodes)

def group_matrix_indices(matrix, threshold=0.5):
    """Group row/column indices with pairwise overlap > threshold."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    # Fill upper triangle from lower (self-comparison matrices are often lower-tri)
    sym = np.array(m, copy=True)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sym[i, j], sym[j, i]
            if np.isnan(a) and np.isnan(b):
                continue
            sym[i, j] = sym[j, i] = np.nanmax([a, b])
    adj = (sym > threshold) & ~np.eye(n, dtype=bool)
    _, labels = connected_components(csr_matrix(adj.astype(int)), directed=False)
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, []).append(idx)
    return [sorted(g) for g in groups.values()]

class MatrixAggregatorConfig(Configuration):
    """Configuration for :class:`MatrixAggregator`.

    Attributes:
        operation: One of ``"min"``, ``"max"``, ``"mean"``, ``"sum"``,
            ``"median"``, or ``"count"``.
        axis: Axis along which to aggregate (required).
        ignore_nan: When ``True``, NaN entries are excluded from aggregation.
        triangle: Bitmask of selectable regions on square 2-D matrices; see
            :mod:`vistiq.constant.matrix`.
        preferred_input_type: Backend for :meth:`MatrixAggregator.run`.
        preferred_device: Torch device when ``preferred_input_type`` is
            ``"torch.Tensor"``; ``None`` selects automatically.
    """

    operation: Literal["min", "max", "mean", "sum", "median", "count"] = "mean"
    axis: Optional[int] = 0
    ignore_nan: bool = True
    triangle: int = FULL
    preferred_input_type: Literal["numpy", "torch.Tensor"] = "torch.Tensor"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class MatrixAggregator(Configurable[MatrixAggregatorConfig]):
    """Aggregate values in a matrix along a configured axis."""

    def __init__(self, config: MatrixAggregatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixAggregatorConfig) -> "MatrixAggregator":
        return cls(config)

    @task(name="MatrixAggregator.run", task_run_name=generate_name)
    def run(
        self,
        data: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Reduce *data* along :attr:`~MatrixAggregatorConfig.axis`."""
        as_numpy = isinstance(data, np.ndarray)
        if (
            device is None
            and self.config.preferred_device is None
            and isinstance(data, torch.Tensor)
        ):
            device = data.device
        else:
            device = resolve_torch_device(
                device,
                preferred_input_type=self.config.preferred_input_type,
                preferred_device=self.config.preferred_device,
            )
        values = convert_array_like(
            data,
            dtype=self.config.preferred_input_type,
            device=device,
        )
        if not isinstance(values, torch.Tensor):
            values = convert_array_like(values, dtype="torch.Tensor", device=device)
        result = self._aggregate(values)
        if as_numpy or self.config.preferred_input_type == "numpy":
            return result.detach().cpu().numpy()
        return result

    def _resolve_axis(self, values: torch.Tensor) -> int:
        axis = self.config.axis
        if axis is None:
            raise ValueError("MatrixAggregatorConfig.axis must be set")
        if isinstance(axis, tuple):
            raise ValueError(
                f"MatrixAggregator does not support tuple axis {axis!r}"
            )
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError(
                f"axis {self.config.axis!r} is out of bounds for ndim={values.ndim}"
            )
        return axis

    def _prepare_values(
        self, values: torch.Tensor, exclude: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_matrix_values(
            values,
            exclude,
            ignore_nan=self.config.ignore_nan,
            triangle=self.config.triangle,
        )

    def _aggregate(self, values: torch.Tensor) -> torch.Tensor:
        axis = self._resolve_axis(values)
        operation = self.config.operation
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        nan = torch.tensor(float("nan"), dtype=values.dtype, device=values.device)

        if operation == "count":
            _, valid = self._prepare_values(values, zero)
            return valid.sum(dim=axis)

        if operation == "sum":
            prepared, _ = self._prepare_values(values, zero)
            return prepared.sum(dim=axis)

        if operation == "mean":
            prepared, valid = self._prepare_values(values, zero)
            counts = valid.sum(dim=axis)
            sums = prepared.sum(dim=axis)
            return torch.where(counts > 0, sums / counts.to(values.dtype), nan)

        if operation == "min":
            fill = torch.full((), float("inf"), dtype=values.dtype, device=values.device)
            prepared, valid = self._prepare_values(values, fill)
            result = torch.min(prepared, dim=axis).values
            return torch.where(valid.any(dim=axis), result, nan)

        if operation == "max":
            fill = torch.full((), float("-inf"), dtype=values.dtype, device=values.device)
            prepared, valid = self._prepare_values(values, fill)
            result = torch.max(prepared, dim=axis).values
            return torch.where(valid.any(dim=axis), result, nan)

        if operation == "median":
            prepared, valid = self._prepare_values(values, nan)
            result = torch.nanmedian(prepared, dim=axis).values
            return torch.where(valid.any(dim=axis), result, nan)

        raise ValueError(f"Invalid operation: {operation}")
