"""Infer containment hierarchies from pairwise weight matrices."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from prefect import task
from pydantic import Field

from vistiq.core import Configurable, Configuration, generate_name
from vistiq.graph.graph import GraphBuilder, GraphBuilderConfig, GraphLike, _normalize_index
from vistiq.matrix.types import MatrixData, as_matrix_data, label_index, matrix_to_numpy, square_matrix

ORPHAN_GROUP_UNKNOWN = "unknown"


@dataclass(frozen=True)
class HierarchyResult:
    """Containment graph and per-node metadata from hierarchy inference.

    ``graph`` is the materialized parent→child DAG. ``nodes`` is a table of node
    attributes (indexed by ``object_id``), including any synthetic nodes added
    during orphan handling. ``matrix`` holds the sparse directed adjacency used
    to build the graph.
    """

    graph: GraphLike
    nodes: pd.DataFrame
    matrix: MatrixData


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


def _assign_parents(
    weight_matrix: MatrixData,
    node_table: pd.DataFrame,
    *,
    rank_attribute: str,
    threshold: float,
    parent_strategy: Literal["smallest_enclosing", "max_weight"],
) -> tuple[dict[Any, Optional[Any]], Any]:
    if rank_attribute not in node_table.columns:
        raise KeyError(
            f"rank_attribute {rank_attribute!r} not in node table; "
            f"available: {list(node_table.columns)}"
        )

    ranks = node_table[rank_attribute].astype(float)
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


def _orphan_group_key(orphan: Any, node_table: pd.DataFrame, groupby: str) -> str:
    if groupby not in node_table.columns:
        return ORPHAN_GROUP_UNKNOWN
    value = node_table.loc[orphan, groupby]
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


def _append_node_row(
    node_table: pd.DataFrame, node_id: str, attributes: dict[str, Any]
) -> pd.DataFrame:
    attrs = dict(attributes)
    attrs["object_id"] = node_id
    row = pd.DataFrame([attrs]).set_index("object_id", drop=False)
    return pd.concat([node_table, row])


def _finalize_node_table(node_table: pd.DataFrame) -> pd.DataFrame:
    node_table = node_table.copy()
    if "object_id" in node_table.columns:
        node_table["object_id"] = node_table.index
    if "synthetic" in node_table.columns:
        node_table["synthetic"] = (
            node_table["synthetic"].fillna(False).astype(bool)
        )
    return node_table


def _apply_orphan_grouping(
    parents: dict[Any, Optional[Any]],
    node_table: pd.DataFrame,
    *,
    orphans: list[Any],
    primary_root: Any,
    orphan_groupby: Optional[str],
    orphan_attach: Literal["separate_root", "unify"],
    orphan_node: dict[str, Any],
    all_node: dict[str, Any],
) -> tuple[dict[Any, Optional[Any]], pd.DataFrame]:
    if not orphans:
        return parents, node_table

    orphan_id = uuid.uuid4().hex
    node_table = _append_node_row(node_table, orphan_id, orphan_node)
    parents[orphan_id] = None

    if orphan_groupby is None:
        for orphan in orphans:
            parents[orphan] = orphan_id
    else:
        groups: dict[str, list[Any]] = {}
        for orphan in orphans:
            key = _orphan_group_key(orphan, node_table, orphan_groupby)
            groups.setdefault(key, []).append(orphan)

        for group_key, group_orphans in groups.items():
            group_id = uuid.uuid4().hex
            node_table = _append_node_row(
                node_table,
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
        node_table = _append_node_row(node_table, all_id, all_node)
        parents[all_id] = None
        parents[orphan_id] = all_id
        parents[primary_root] = all_id

    return parents, node_table


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


class HierarchyBuilderConfig(Configuration):
    """Configuration for :class:`HierarchyBuilder`.

    Shapes a labeled pairwise weight matrix (e.g. IoS, IoU, distance) into a
    sparse parent→child containment graph and optionally extends the node table
    with synthetic orphan-group nodes. Edge weights are taken from the input
    matrix; parent assignment uses :attr:`threshold` against those pairwise
    values.

    :attr:`rank_attribute` names a column in the node metadata table used to
    rank candidates for parent selection (larger values are treated as enclosing
    or higher-priority parents).
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
            "Fallback edge weight when the input matrix has no value for a "
            "parent→child pair (orphan/synthetic links). GraphBuilder marks those "
            "edges synthetic=True when either endpoint is a synthetic node."
        ),
    )
    orphan_node: dict[str, Any] = Field(
        default_factory=lambda: {"name": "Orphans", "synthetic": True}
    )
    all_node: dict[str, Any] = Field(
        default_factory=lambda: {"name": "all", "synthetic": True}
    )
    graph_builder: GraphBuilderConfig = Field(default_factory=GraphBuilderConfig)


class HierarchyBuilder(Configurable[HierarchyBuilderConfig]):
    """Infer a containment hierarchy from pairwise weights and node metadata.

    Each retained parent→child link is weighted by the corresponding entry in
    the input matrix (symmetric lookup); directed edges only. The node table
    supplies per-node attributes (e.g. rank for parent selection) and may
    gain synthetic rows when orphan grouping is enabled.
    """

    def __init__(self, config: HierarchyBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: HierarchyBuilderConfig) -> "HierarchyBuilder":
        return cls(config)

    @task(name="HierarchyBuilder.run", task_run_name=generate_name)
    def run(
        self,
        matrix: Union[MatrixData, pd.DataFrame],
        nodes: pd.DataFrame,
    ) -> HierarchyResult:
        weight_matrix = square_matrix(as_matrix_data(matrix))
        assert weight_matrix.annotations is not None
        node_ids = list(weight_matrix.annotations[0])
        node_table = _normalize_index(nodes).reindex(node_ids)
        missing = node_table.index[node_table.isna().all(axis=1)]
        if len(missing) > 0:
            missing_ids = ", ".join(str(value) for value in missing[:5])
            raise KeyError(
                f"node table missing attributes for {len(missing)} object_id(s), "
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
            for node in node_ids
            if parents.get(node) is None and node != primary_root
        ]

        if self.config.orphan_strategy == "drop":
            keep_nodes = sorted(_drop_orphan_subtrees(node_ids, parents, orphans))
            node_table = node_table.loc[keep_nodes]
            parents = {
                child: parent
                for child, parent in parents.items()
                if child in keep_nodes and (parent is None or parent in keep_nodes)
            }
        else:
            keep_nodes = list(node_ids)
            if self.config.orphan_strategy == "group":
                parents, node_table = _apply_orphan_grouping(
                    parents,
                    node_table,
                    orphans=orphans,
                    primary_root=primary_root,
                    orphan_groupby=self.config.orphan_groupby,
                    orphan_attach=self.config.orphan_attach,
                    orphan_node=self.config.orphan_node,
                    all_node=self.config.all_node,
                )
                keep_nodes = list(node_table.index)

        adjacency = _parents_to_matrix(
            parents,
            weight_matrix,
            keep_nodes,
            synthetic_weight=self.config.synthetic_weight,
        )
        node_table = _finalize_node_table(node_table)
        graph = GraphBuilder(self.config.graph_builder).run(adjacency, node_table)
        return HierarchyResult(graph=graph, nodes=node_table, matrix=adjacency)
