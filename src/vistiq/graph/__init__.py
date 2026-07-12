"""Directed graphs, containment hierarchies, and graph queries.

``vistiq.graph`` owns topology and structure:

- :class:`~vistiq.graph.graph.GraphBuilder` — adjacency / edge tables → DAG
- :class:`~vistiq.graph.hierarchy.HierarchyBuilder` — overlap matrix → containment DAG
- :class:`~vistiq.graph.graph.GraphQuery` — attribute summaries on a graph
- :class:`~vistiq.graph.graph.GraphFormatter` / :class:`~vistiq.graph.graph.GraphQueryFormatter` — in-memory projection to tables

Labeled numeric matrices are represented by :class:`~vistiq.matrix.types.MatrixData`
in :mod:`vistiq.matrix`. Domain-specific matrix producers (overlap metrics) remain
in :mod:`vistiq.analysis.overlap`.
"""

from .hierarchy import (
    HierarchyBuilder,
    HierarchyBuilderConfig,
    HierarchyResult,
)
from .graph import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphFormatter,
    GraphFormatterConfig,
    GraphFilter,
    GraphFilterConfig,
    GraphLike,
    GraphQuery,
    GraphQueryConfig,
    GraphQueryFormatter,
    GraphQueryFormatterConfig,
    GraphQueryResult,
    GRAPH_NODE_INDEX,
    NXGraph,
    edges_to_matrix,
    graph_to_dataframe,
    resolve_subtree_origin,
    resolve_subtree_origins,
    subtree_origin_key,
)
from .io import (
    default_spatial_result_key,
    load_analysis,
    save_analysis,
)
from .napari import (
    add_hierarchical_napari_layers,
    edges_to_layer,
    nodes_to_layer,
)

__all__ = [
    "GraphBuilder",
    "GraphBuilderConfig",
    "GraphFormatter",
    "GraphFormatterConfig",
    "GraphFilter",
    "GraphFilterConfig",
    "HierarchyBuilder",
    "HierarchyBuilderConfig",
    "HierarchyResult",
    "GraphLike",
    "GraphQuery",
    "GraphQueryConfig",
    "GraphQueryFormatter",
    "GraphQueryFormatterConfig",
    "GraphQueryResult",
    "GRAPH_NODE_INDEX",
    "NXGraph",
    "add_hierarchical_napari_layers",
    "default_spatial_result_key",
    "edges_to_layer",
    "edges_to_matrix",
    "graph_to_dataframe",
    "load_analysis",
    "nodes_to_layer",
    "resolve_subtree_origin",
    "resolve_subtree_origins",
    "save_analysis",
    "subtree_origin_key",
]
