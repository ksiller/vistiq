"""Spatial neighbor analysis from distance matrices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prefect import task
from pydantic import Field

from vistiq.matrix.ops import MatrixFormatter, MatrixFormatterConfig
from vistiq.analysis.distance import DistanceCalculator, DistanceCalculatorConfig
from vistiq.matrix.types import OFF_DIAGONAL
from vistiq.core import Configurable, Configuration, generate_name
from vistiq.graph.graph import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphFormatter,
    GraphFormatterConfig,
    GraphQuery,
    resolve_subtree_origin,
)
from vistiq.segment import TopKFilter, TopKFilterConfig, ValueFilter, ValueFilterConfig, dataframe_to_numpy
from vistiq.segment.select import MatrixFilter

logger = logging.getLogger(__name__)

_UNSET = object()


def distance_matrix_from_regions(
    regions: pd.DataFrame,
    calculator: DistanceCalculatorConfig,
    *,
    centroid: str = "centroid",
    axes: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Pairwise distances between rows in a region property table."""
    points = dataframe_to_numpy(
        regions,
        attributes=[centroid],
        strict=False,
        axes=axes,
        reset_index=False,
    )
    if points is None:
        raise KeyError(
            f"centroid coordinates missing; available: {list(regions.columns)}"
        )
    idx = tuple(regions.index)
    result = DistanceCalculator(calculator).run(
        points,
        points,
        point_annotations=(idx, idx),
    )
    return MatrixFormatter(MatrixFormatterConfig()).run(result)


def _regions_from_containment(
    containment_graph: Any,
    formatter: GraphFormatterConfig,
    *,
    node: Optional[Any] = None,
) -> pd.DataFrame:
    """Format region attributes from a containment DAG, optionally scoped to a subtree."""
    graph = containment_graph
    if node is not None:
        graph = GraphQuery._origin_subgraph(containment_graph, node)
    return GraphFormatter(formatter).run(graph)


def _resolve_distance_matrix(
    regions: pd.DataFrame,
    calculator: DistanceCalculatorConfig,
    distance_matrix: Optional[Union[np.ndarray, pd.DataFrame]],
    *,
    centroid: str,
    axes: Optional[Tuple[str, ...]],
) -> pd.DataFrame:
    idx = tuple(regions.index)
    if distance_matrix is not None:
        if isinstance(distance_matrix, pd.DataFrame):
            matrix = distance_matrix
        else:
            expected = (len(idx), len(idx))
            if distance_matrix.shape != expected:
                raise ValueError(
                    f"distance_matrix shape {distance_matrix.shape} does not match "
                    f"region count {expected}"
                )
            matrix = pd.DataFrame(distance_matrix, index=idx, columns=idx)
        missing = matrix.index.difference(idx)
        if len(missing):
            raise ValueError(
                f"distance_matrix index mismatch; missing regions: {list(missing[:5])}"
            )
        return matrix.reindex(index=idx, columns=idx)
    return distance_matrix_from_regions(
        regions,
        calculator,
        centroid=centroid,
        axes=axes,
    )


def _apply_grouping_mask(
    matrix: pd.DataFrame,
    regions: pd.DataFrame,
    *,
    grouping_attribute: str,
    mode: Literal["homotypic", "heterotypic", "global"],
) -> pd.DataFrame:
    if mode == "global":
        return matrix

    if grouping_attribute not in regions.columns:
        raise KeyError(
            f"grouping_attribute {grouping_attribute!r} not in regions; "
            f"available: {list(regions.columns)}"
        )

    row_groups = regions.reindex(matrix.index)[grouping_attribute].to_numpy()
    col_groups = regions.reindex(matrix.columns)[grouping_attribute].to_numpy()
    same_group = row_groups[:, None] == col_groups[None, :]
    values = matrix.to_numpy(dtype=float, copy=True)
    if mode == "homotypic":
        values[~same_group] = np.nan
    else:
        values[same_group] = np.nan
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def _filtered_neighbor_graph(
    distance_matrix: pd.DataFrame,
    regions: pd.DataFrame,
    matrix_filter: MatrixFilter,
    graph_builder: GraphBuilderConfig,
    *,
    grouping_attribute: str,
    mode: Literal["homotypic", "heterotypic", "global"],
) -> tuple[pd.DataFrame, Any]:
    masked = _apply_grouping_mask(
        distance_matrix,
        regions,
        grouping_attribute=grouping_attribute,
        mode=mode,
    )
    filtered = matrix_filter.run(masked.to_numpy(dtype=np.float64))
    neighbor_matrix = pd.DataFrame(
        filtered.detach().cpu().numpy(),
        index=masked.index,
        columns=masked.columns,
    )
    graph = GraphBuilder(graph_builder).run(neighbor_matrix, regions)
    return neighbor_matrix, graph


@dataclass
class SpatialGraphResult:
    """Distance matrix and filtered neighbor graph from spatial analysis."""

    distance_matrix: pd.DataFrame
    matrix: pd.DataFrame
    graph: Any


class SpatialScopeConfig(Configuration):
    """Runtime subtree selection for spatial neighbor analysis.

    *match* describes containment-graph node attributes (all must match).
    *exclude* removes candidates with the same all-keys-must-match rule; exclude
    wins when a node matches both. Multiple matches run kNN/RNN once per subtree
    root in :class:`AnalysisFlow`. Set ``auto_root=True`` when the DAG has a
    single root. Leave *match* unset and ``auto_root=False`` to analyze all
    exported nodes.
    """

    match: Optional[dict[str, Any]] = None
    exclude: Optional[dict[str, Any]] = None
    auto_root: bool = False


def _resolve_analysis_origin(
    containment_graph: Any,
    node: Any,
    scope: SpatialScopeConfig,
) -> Optional[Any]:
    """Resolve subtree root for one spatial analysis run.

    When *node* is supplied (including ``None`` for the full graph), it is used
    as-is. Otherwise :class:`SpatialScopeConfig` on the analysis config applies
    (standalone ``run(containment_graph)`` calls).
    """
    if node is not _UNSET:
        if node is not None and node not in containment_graph:
            raise KeyError(f"node {node!r} not found in graph")
        return node
    return resolve_subtree_origin(
        containment_graph,
        match=scope.match,
        exclude=scope.exclude,
        auto_root=scope.auto_root,
    )


class SpatialNeighborConfig(Configuration):
    """Shared settings for spatial neighbor graph analysis.

    Region attributes are read from the containment graph via
    :class:`~vistiq.graph.GraphFormatter`. :class:`SpatialScopeConfig` selects
    the subtree; ``run(node=…)`` overrides scope for one-off calls.
    """

    mode: Literal["homotypic", "heterotypic", "global"] = "homotypic"
    grouping_attribute: str = "channel"
    scope: SpatialScopeConfig = Field(default_factory=SpatialScopeConfig)
    """Subtree selection for standalone ``run(containment_graph)`` calls.

    :class:`~vistiq.analysis.workflow.AnalysisFlow` ignores this and uses
    :attr:`~vistiq.analysis.workflow.AnalysisFlowConfig.spatial_scope` instead,
    passing resolved roots via ``run(..., node=…)``.
    """
    centroid: str = "centroid"
    axes: Optional[Tuple[str, ...]] = None
    graph_formatter: GraphFormatterConfig = Field(
        default_factory=lambda: GraphFormatterConfig(exclude_synthetic=True),
    )
    distance_calculator: DistanceCalculatorConfig = Field(
        default_factory=DistanceCalculatorConfig
    )


class KnnAnalysisConfig(SpatialNeighborConfig):
    """k-nearest-neighbor filter and graph materialization settings."""

    k: int = Field(default=5, ge=1)
    knn_filter: TopKFilterConfig = Field(
        default_factory=lambda: TopKFilterConfig(
            k=5,
            axis=1,
            largest=False,
            triangle=OFF_DIAGONAL,
            output="masked_values",
            ignore_nan=True,
        )
    )
    graph_builder: GraphBuilderConfig = Field(
        default_factory=lambda: GraphBuilderConfig(edge_attribute="distance")
    )


class RnnAnalysisConfig(SpatialNeighborConfig):
    """Radius nearest-neighbor filter and graph materialization settings."""

    radius: float = Field(default=50.0, gt=0)
    rnn_filter: ValueFilterConfig = Field(
        default_factory=lambda: ValueFilterConfig(
            operator="<=",
            triangle=OFF_DIAGONAL,
            output="masked_values",
            ignore_nan=True,
        )
    )
    graph_builder: GraphBuilderConfig = Field(
        default_factory=lambda: GraphBuilderConfig(edge_attribute="distance")
    )


class KnnAnalysis(Configurable[KnnAnalysisConfig]):
    """Apply TopK filtering to a distance matrix and build a directed kNN graph."""

    def __init__(self, config: KnnAnalysisConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: KnnAnalysisConfig) -> "KnnAnalysis":
        return cls(config)

    @task(name="KnnAnalysis.run", task_run_name=generate_name)
    def run(
        self,
        containment_graph: Any,
        *,
        node: Any = _UNSET,
        distance_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        axes: Optional[Tuple[str, ...]] = None,
    ) -> SpatialGraphResult:
        cfg = self.config
        origin = _resolve_analysis_origin(containment_graph, node, cfg.scope)
        regions = _regions_from_containment(
            containment_graph, cfg.graph_formatter, node=origin
        )
        if regions.empty:
            raise ValueError(
                "containment graph has no analyzable nodes in the selected subtree"
            )
        dist = _resolve_distance_matrix(
            regions,
            cfg.distance_calculator,
            distance_matrix,
            centroid=cfg.centroid,
            axes=axes if axes is not None else cfg.axes,
        )
        matrix, graph = _filtered_neighbor_graph(
            dist,
            regions,
            TopKFilter(cfg.knn_filter.model_copy(update={"k": cfg.k})),
            cfg.graph_builder,
            grouping_attribute=cfg.grouping_attribute,
            mode=cfg.mode,
        )
        return SpatialGraphResult(distance_matrix=dist, matrix=matrix, graph=graph)


class RnnAnalysis(Configurable[RnnAnalysisConfig]):
    """Apply radius filtering to a distance matrix and build a directed RNN graph."""

    def __init__(self, config: RnnAnalysisConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: RnnAnalysisConfig) -> "RnnAnalysis":
        return cls(config)

    @task(name="RnnAnalysis.run", task_run_name=generate_name)
    def run(
        self,
        containment_graph: Any,
        *,
        node: Any = _UNSET,
        distance_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        axes: Optional[Tuple[str, ...]] = None,
    ) -> SpatialGraphResult:
        cfg = self.config
        origin = _resolve_analysis_origin(containment_graph, node, cfg.scope)
        regions = _regions_from_containment(
            containment_graph, cfg.graph_formatter, node=origin
        )
        if regions.empty:
            raise ValueError(
                "containment graph has no analyzable nodes in the selected subtree"
            )
        dist = _resolve_distance_matrix(
            regions,
            cfg.distance_calculator,
            distance_matrix,
            centroid=cfg.centroid,
            axes=axes if axes is not None else cfg.axes,
        )
        matrix, graph = _filtered_neighbor_graph(
            dist,
            regions,
            ValueFilter(
                cfg.rnn_filter.model_copy(update={"ref_value": cfg.radius})
            ),
            cfg.graph_builder,
            grouping_attribute=cfg.grouping_attribute,
            mode=cfg.mode,
        )
        return SpatialGraphResult(distance_matrix=dist, matrix=matrix, graph=graph)
