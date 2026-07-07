"""Matrix formatters, combiners, aggregators, and hierarchical transforms."""

from __future__ import annotations

import logging
import uuid
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
    AnnotationFactory,
    HierarchicalResult,
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

ORPHAN_GROUP_UNKNOWN = "unknown"


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
        if as_numpy or self.config.preferred_input_type == "numpy":
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


def _pairwise_value(table: Union[MatrixData, pd.DataFrame], left: Any, right: Any) -> float:
    """Return the symmetric matrix weight between left and right."""
    if isinstance(table, MatrixData):
        if table.ndim != 2 or table.annotations is None:
            raise ValueError("pairwise lookup requires a labeled 2-D MatrixData")
        values = matrix_to_numpy(table)
        rows, cols = table.annotations
        row_map = label_index(rows)
        col_map = label_index(cols)
        candidates: list[float] = []
        if left in row_map and right in col_map:
            candidates.append(float(values[row_map[left], col_map[right]]))
        if right in row_map and left in col_map:
            candidates.append(float(values[row_map[right], col_map[left]]))
        if not candidates:
            return float("nan")
        return float(np.nanmax(candidates))

    values_list: list[float] = []
    if left in table.index and right in table.columns:
        values_list.append(float(table.loc[left, right]))
    if right in table.index and left in table.columns:
        values_list.append(float(table.loc[right, left]))
    if not values_list:
        return float("nan")
    return float(np.nanmax(values_list))


def _normalize_index(table: pd.DataFrame, index_name: str = "object_id") -> pd.DataFrame:
    if table.index.name == index_name:
        return table
    if isinstance(table.index, pd.MultiIndex) and index_name in table.index.names:
        return table.reset_index().set_index(index_name, drop=False)
    if index_name in table.columns:
        return table.set_index(index_name, drop=False)
    raise KeyError(
        f"table must be indexed by {index_name} or include a {index_name} column; "
        f"index={table.index.name!r}, columns={list(table.columns)}"
    )


def _assign_parents(
    weight_matrix: MatrixData,
    region_table: pd.DataFrame,
    *,
    rank_attribute: str,
    threshold: float,
    parent_strategy: Literal["smallest_enclosing", "max_weight"],
) -> tuple[dict[Any, Optional[Any]], Any]:
    if rank_attribute not in region_table.columns:
        raise KeyError(
            f"rank_attribute {rank_attribute!r} not in regions; "
            f"available: {list(region_table.columns)}"
        )

    ranks = region_table[rank_attribute].astype(float)
    ordered = ranks.sort_values(ascending=False).index.tolist()
    primary_root = ordered[0]

    parents: dict[Any, Optional[Any]] = {primary_root: None}
    for child in ordered[1:]:
        child_rank = float(ranks[child])
        candidates = [
            parent for parent in ordered if float(ranks[parent]) > child_rank
        ]
        candidates = [
            parent
            for parent in candidates
            if _pairwise_value(weight_matrix, parent, child) >= threshold
        ]
        if not candidates:
            parents[child] = None
            continue
        if parent_strategy == "max_weight":
            parent = max(
                candidates,
                key=lambda candidate: _pairwise_value(
                    weight_matrix, candidate, child
                ),
            )
        else:
            parent = min(candidates, key=lambda candidate: float(ranks[candidate]))
        parents[child] = parent

    return parents, primary_root


def _orphan_group_key(orphan: Any, region_table: pd.DataFrame, groupby: str) -> str:
    if groupby not in region_table.columns:
        return ORPHAN_GROUP_UNKNOWN
    value = region_table.loc[orphan, groupby]
    if pd.isna(value):
        return ORPHAN_GROUP_UNKNOWN
    return str(value)


def _drop_orphan_subtrees(
    nodes: list[Any],
    parents: dict[Any, Optional[Any]],
    orphans: list[Any],
) -> set[Any]:
    drop = set(orphans)
    while True:
        added = {
            child
            for child, parent in parents.items()
            if child not in drop and parent in drop
        }
        if not added:
            break
        drop |= added
    return {node for node in nodes if node not in drop}


def _append_region_row(regions: pd.DataFrame, node_id: str, attributes: dict[str, Any]) -> pd.DataFrame:
    attrs = dict(attributes)
    attrs["object_id"] = node_id
    row = pd.DataFrame([attrs]).set_index("object_id", drop=False)
    return pd.concat([regions, row])


def _finalize_region_table(region_table: pd.DataFrame) -> pd.DataFrame:
    region_table = region_table.copy()
    if "object_id" in region_table.columns:
        region_table["object_id"] = region_table.index
    if "synthetic" in region_table.columns:
        region_table["synthetic"] = (
            region_table["synthetic"].fillna(False).astype(bool)
        )
    return region_table


def _apply_orphan_grouping(
    parents: dict[Any, Optional[Any]],
    regions: pd.DataFrame,
    *,
    orphans: list[Any],
    primary_root: Any,
    orphan_groupby: Optional[str],
    orphan_attach: Literal["separate_root", "unify"],
    orphan_node: dict[str, Any],
    all_node: dict[str, Any],
) -> tuple[dict[Any, Optional[Any]], pd.DataFrame]:
    if not orphans:
        return parents, regions

    orphan_id = uuid.uuid4().hex
    regions = _append_region_row(regions, orphan_id, orphan_node)
    parents[orphan_id] = None

    if orphan_groupby is None:
        for orphan in orphans:
            parents[orphan] = orphan_id
    else:
        groups: dict[str, list[Any]] = {}
        for orphan in orphans:
            key = _orphan_group_key(orphan, regions, orphan_groupby)
            groups.setdefault(key, []).append(orphan)

        for group_key, group_orphans in groups.items():
            group_id = uuid.uuid4().hex
            regions = _append_region_row(
                regions,
                group_id,
                {
                    "name": f"orphans:{group_key}",
                    "synthetic": True,
                    "orphan_group": group_key,
                },
            )
            parents[group_id] = orphan_id
            for orphan in group_orphans:
                parents[orphan] = group_id

    if orphan_attach == "unify":
        all_id = uuid.uuid4().hex
        regions = _append_region_row(regions, all_id, all_node)
        parents[all_id] = None
        parents[orphan_id] = all_id
        parents[primary_root] = all_id

    return parents, regions


def _parents_to_matrix(
    parents: dict[Any, Optional[Any]],
    weight_matrix: MatrixData,
    nodes: list[Any],
    *,
    synthetic_weight: float,
) -> MatrixData:
    matrix = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
    node_map = label_index(nodes)
    for child, parent in parents.items():
        if parent is None or child not in node_map or parent not in node_map:
            continue
        weight = _pairwise_value(weight_matrix, parent, child)
        if np.isnan(weight):
            weight = synthetic_weight
        matrix[node_map[parent], node_map[child]] = weight
    return MatrixData(matrix=matrix, annotations=(tuple(nodes), tuple(nodes)))


class MatrixTransformerConfig(Configuration):
    """Base configuration for matrix transformers."""


class MatrixTransformer(Configurable[MatrixTransformerConfig]):
    """Transform a labeled matrix without recomputing overlap metrics."""

    def __init__(self, config: MatrixTransformerConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixTransformerConfig) -> "MatrixTransformer":
        return cls(config)

    @task(name="MatrixTransformer.run", task_run_name=generate_name)
    def run(
        self,
        matrix: Union[MatrixData, pd.DataFrame],
        regions: pd.DataFrame,
    ) -> HierarchicalResult:
        raise NotImplementedError


class HierarchicalMatrixConfig(MatrixTransformerConfig):
    """Configuration for :class:`HierarchicalMatrix`.

    Shapes a filtered overlap matrix into a sparse parent→child adjacency matrix
    and optionally extends ``regions`` with synthetic orphan-group nodes.
    """

    rank_attribute: str = "volume"
    threshold: float = 0.5
    parent_strategy: Literal["smallest_enclosing", "max_weight"] = "smallest_enclosing"
    orphan_strategy: Literal["drop", "as_roots", "group"] = "as_roots"
    orphan_groupby: Optional[str] = None
    orphan_attach: Literal["separate_root", "unify"] = "separate_root"
    synthetic_weight: float = Field(
        default=1.0,
        description=(
            "Fallback matrix weight when a hierarchical edge has no IoS value "
            "(orphan/synthetic links). GraphBuilder marks those edges synthetic=True "
            "when either endpoint is a synthetic node."
        ),
    )
    orphan_node: dict[str, Any] = Field(
        default_factory=lambda: {"name": "Orphans", "synthetic": True}
    )
    all_node: dict[str, Any] = Field(
        default_factory=lambda: {"name": "all", "synthetic": True}
    )


class HierarchicalMatrix(MatrixTransformer):
    """Convert a matrix into a hierarchical adjacency matrix (directed edges only)."""

    def __init__(self, config: HierarchicalMatrixConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: HierarchicalMatrixConfig) -> "HierarchicalMatrix":
        return cls(config)

    @task(name="HierarchicalMatrix.run", task_run_name=generate_name)
    def run(
        self,
        matrix: Union[MatrixData, pd.DataFrame],
        regions: pd.DataFrame,
    ) -> HierarchicalResult:
        weight_matrix = square_matrix(as_matrix_data(matrix))
        assert weight_matrix.annotations is not None
        nodes = list(weight_matrix.annotations[0])
        node_table = _normalize_index(regions).reindex(nodes)
        missing = node_table.index[node_table.isna().all(axis=1)]
        if len(missing) > 0:
            missing_ids = ", ".join(str(value) for value in missing[:5])
            raise KeyError(
                f"regions missing metrics for {len(missing)} object_id(s), "
                f"including: {missing_ids}"
            )

        parents, primary_root = _assign_parents(
            weight_matrix,
            node_table,
            rank_attribute=self.config.rank_attribute,
            threshold=self.config.threshold,
            parent_strategy=self.config.parent_strategy,
        )
        orphans = [
            node
            for node in nodes
            if parents.get(node) is None and node != primary_root
        ]

        if self.config.orphan_strategy == "drop":
            keep_nodes = sorted(_drop_orphan_subtrees(nodes, parents, orphans))
            region_table = node_table.loc[keep_nodes]
            parents = {
                child: parent
                for child, parent in parents.items()
                if child in keep_nodes and (parent is None or parent in keep_nodes)
            }
        else:
            keep_nodes = list(nodes)
            region_table = node_table
            if self.config.orphan_strategy == "group":
                parents, region_table = _apply_orphan_grouping(
                    parents,
                    node_table,
                    orphans=orphans,
                    primary_root=primary_root,
                    orphan_groupby=self.config.orphan_groupby,
                    orphan_attach=self.config.orphan_attach,
                    orphan_node=self.config.orphan_node,
                    all_node=self.config.all_node,
                )
                keep_nodes = list(region_table.index)

        hierarchical = _parents_to_matrix(
            parents,
            weight_matrix,
            keep_nodes,
            synthetic_weight=self.config.synthetic_weight,
        )
        region_table = _finalize_region_table(region_table)
        return HierarchicalResult(matrix=hierarchical, regions=region_table)
