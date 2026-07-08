"""Matrix formatters, combiners, and aggregators."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from prefect import task
from pydantic import Field
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vistiq.core import Configurable, Configuration, generate_name
from vistiq.matrix.mask import mask_triangle, prepare_matrix_values
from vistiq.matrix.types import (
    FULL,
    ArrayBackend,
    AnnotationFactory,
    MatrixArray,
    MatrixAnnotations,
    MatrixContainer,
    MatrixData,
    MatrixFormatOutput,
    annotations_after_aggregate,
    as_matrix_data,
    default_matrix_annotations,
    label_index,
    matrix_to_numpy,
    ordered_union,
    resolve_matrix_annotations,
    square_matrix,
)
from vistiq.utils import convert_array_like, resolve_torch_device

logger = logging.getLogger(__name__)


class MatrixFormatterConfig(Configuration):
    """Configuration for :class:`MatrixFormatter`.

    Attributes:
        output_type: Export target (``"dataframe"``, ``"np.ndarray"``, or
            ``"torch.Tensor"``).
        annotate: When ``True``, attach axis labels from :class:`MatrixData` or
            :attr:`annotation_factory`.
        annotation_factory: Fallback labels when :class:`MatrixData` has no
            annotations and :attr:`annotate` is ``True``.
        preferred_device: Torch device when :attr:`output_type` is
            ``"torch.Tensor"``; ``None`` selects automatically.
    """

    output_type: MatrixFormatOutput = "dataframe"
    annotate: bool = True
    annotation_factory: AnnotationFactory = default_matrix_annotations
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class MatrixFormatter(Configurable[MatrixFormatterConfig]):
    """Export :class:`MatrixData` to ndarray, tensor, or labeled pandas containers."""

    def __init__(self, config: MatrixFormatterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixFormatterConfig) -> "MatrixFormatter":
        return cls(config)

    @task(name="MatrixFormatter.run", task_run_name=generate_name)
    def run(
        self,
        data: MatrixData,
        *,
        device: Optional[torch.device] = None,
    ) -> MatrixContainer:
        """Format *data* using :attr:`~MatrixFormatterConfig.output_type`."""
        resolved = resolve_matrix_annotations(
            data,
            annotate=self.config.annotate,
            annotation_factory=self.config.annotation_factory,
        )
        output_type = self.config.output_type
        matrix = data.matrix
        if output_type == "np.ndarray":
            if isinstance(matrix, np.ndarray):
                return matrix
            if isinstance(matrix, torch.Tensor):
                return matrix.detach().cpu().numpy()
            return np.asarray(matrix, dtype=float)
        if output_type == "torch.Tensor":
            if device is None:
                device = resolve_torch_device(
                    None,
                    preferred_input_type="torch.Tensor",
                    preferred_device=self.config.preferred_device,
                )
            if isinstance(matrix, torch.Tensor):
                return matrix.to(device) if device is not None else matrix
            array = (
                matrix
                if isinstance(matrix, np.ndarray)
                else np.asarray(matrix, dtype=float)
            )
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            return tensor.to(device) if device is not None else tensor

        if isinstance(matrix, np.ndarray):
            values = matrix
        elif isinstance(matrix, torch.Tensor):
            values = matrix.detach().cpu().numpy()
        else:
            values = np.asarray(matrix, dtype=float)
        if data.ndim == 1:
            index = None
            if resolved is not None:
                index = [str(label) for label in resolved[0]]
            return pd.Series(values, index=index, name="value")
        if data.ndim != 2:
            raise ValueError(
                f"dataframe output requires ndim 1 or 2; got ndim={data.ndim}"
            )
        if resolved is not None:
            columns = [str(label) for label in resolved[1]]
            index = [str(label) for label in resolved[0]]
            return pd.DataFrame(values, columns=columns, index=index)
        return pd.DataFrame(values)


def square_dataframe(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reindex to shared row/column labels (ordered union of index and columns)."""
    formatted = MatrixFormatter(MatrixFormatterConfig()).run(
        square_matrix(as_matrix_data(matrix))
    )
    assert isinstance(formatted, pd.DataFrame)
    return formatted


class MatrixCombinerConfig(Configuration):
    """Configuration for :class:`MatrixCombiner`.

    Attributes:
        fill_value: Value for global matrix cells not covered by any input block,
            and for cells masked out by :attr:`triangle`.
        symmetrize: When ``True``, merge ``(i, j)`` and ``(j, i)`` with
            element-wise ``nanmax`` on the squared global matrix before masking.
        triangle: Bitmask of retained regions on the square global matrix; see
            :mod:`vistiq.matrix.types`. ``FULL`` keeps all cells.
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
        matrices: list[Union[MatrixData, pd.DataFrame, ArrayLike]],
        *,
        object_ids: Optional[list[tuple[list[Any], list[Any]]]] = None,
    ) -> MatrixData:
        """Combine pairwise metric blocks into a single :class:`MatrixData`.

        Args:
            matrices: Pairwise metric blocks (e.g. IoS between two object lists).
            object_ids: Optional per-block ``(row_labels, col_labels)``. When set,
                ``len(object_ids)`` must equal ``len(matrices)``, and each label
                list length must match the corresponding matrix shape.

        Returns:
            Combined 2-D matrix with union row/column annotations; unfilled cells
            use :attr:`~MatrixCombinerConfig.fill_value`.
        """
        if not matrices:
            return MatrixData(matrix=np.empty((0, 0), dtype=float))

        if object_ids is not None and len(object_ids) != len(matrices):
            raise ValueError(
                f"object_ids length {len(object_ids)} does not match "
                f"matrices length {len(matrices)}"
            )

        blocks: list[MatrixData] = []
        global_rows: list[Any] = []
        global_cols: list[Any] = []

        for index, matrix in enumerate(matrices):
            labels = object_ids[index] if object_ids is not None else None
            if labels is not None:
                rows, cols = labels
                values = matrix_to_numpy(matrix)
                if values.ndim != 2:
                    raise ValueError(f"matrix {index} must be 2-D; got shape {values.shape}")
                if len(rows) != values.shape[0] or len(cols) != values.shape[1]:
                    raise AssertionError(
                        f"object_ids[{index}] lengths ({len(rows)}, {len(cols)}) "
                        f"do not match matrix shape {values.shape}"
                    )
                block = MatrixData(
                    matrix=values,
                    annotations=(tuple(rows), tuple(cols)),
                )
            elif isinstance(matrix, MatrixData):
                if matrix.ndim != 2:
                    raise ValueError(f"matrix {index} must be 2-D; got ndim={matrix.ndim}")
                block = matrix
            elif isinstance(matrix, pd.DataFrame):
                block = as_matrix_data(matrix.astype(float, copy=False))
            else:
                raise ValueError(
                    f"matrix {index} is not MatrixData or DataFrame; "
                    "pass object_ids for array inputs"
                )

            assert block.annotations is not None
            global_rows = ordered_union(global_rows, list(block.annotations[0]))
            global_cols = ordered_union(global_cols, list(block.annotations[1]))
            blocks.append(block)

        combined_values = np.full(
            (len(global_rows), len(global_cols)),
            self.config.fill_value,
            dtype=float,
        )
        row_map = label_index(global_rows)
        col_map = label_index(global_cols)
        for block in blocks:
            assert block.annotations is not None
            block_rows, block_cols = block.annotations
            block_row_map = label_index(block_rows)
            block_col_map = label_index(block_cols)
            values = matrix_to_numpy(block)
            for left in block_rows:
                for right in block_cols:
                    combined_values[row_map[left], col_map[right]] = values[
                        block_row_map[left], block_col_map[right]
                    ]

        combined = MatrixData(
            matrix=combined_values,
            annotations=(tuple(global_rows), tuple(global_cols)),
        )
        if self.config.symmetrize or self.config.triangle != FULL:
            combined = square_matrix(combined)
            values = matrix_to_numpy(combined)
            if self.config.symmetrize:
                values = symmetrize_max(values)
            values = mask_triangle(values, self.config.triangle, self.config.fill_value)
            combined = MatrixData(matrix=values, annotations=combined.annotations)
        return combined


def symmetrize_max(values: np.ndarray) -> np.ndarray:
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
    nodes = ordered_union(nodes, list(matrix.index))
    nodes = ordered_union(nodes, list(matrix.columns))
    square = matrix.reindex(index=nodes, columns=nodes)
    values = square.to_numpy(dtype=float)
    if symmetrize:
        values = symmetrize_max(values)

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
            :mod:`vistiq.matrix.types`.
        preferred_input_type: Backend for :meth:`MatrixAggregator.run`.
        preferred_device: Torch device when ``preferred_input_type`` is
            ``"torch.Tensor"``; ``None`` selects automatically.
    """

    operation: Literal["min", "max", "mean", "sum", "median", "count"] = "mean"
    axis: Optional[int] = 0
    ignore_nan: bool = True
    triangle: int = FULL
    preferred_input_type: ArrayBackend = "torch.Tensor"
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
        data: Union[MatrixData, np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
    ) -> MatrixData:
        """Reduce *data* along :attr:`~MatrixAggregatorConfig.axis`."""
        input_data = as_matrix_data(data) if not isinstance(data, MatrixData) else data
        as_numpy = isinstance(input_data.matrix, np.ndarray)
        if (
            device is None
            and self.config.preferred_device is None
            and isinstance(input_data.matrix, torch.Tensor)
        ):
            device = input_data.matrix.device
        else:
            device = resolve_torch_device(
                device,
                preferred_input_type=self.config.preferred_input_type,
                preferred_device=self.config.preferred_device,
            )
        values = convert_array_like(
            input_data.matrix,
            dtype=self.config.preferred_input_type,
            device=device,
        )
        if not isinstance(values, torch.Tensor):
            values = convert_array_like(values, dtype="torch.Tensor", device=device)
        result = self._aggregate(values)
        if as_numpy or self.config.preferred_input_type == "np.ndarray":
            result_array: MatrixArray = result.detach().cpu().numpy()
        else:
            result_array = result
        axis = self._resolve_axis(values)
        return MatrixData(
            matrix=result_array,
            annotations=annotations_after_aggregate(input_data.annotations, axis),
        )

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
