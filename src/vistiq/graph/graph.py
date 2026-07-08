"""Build graphs from labeled matrices and node metrics.

GraphBuilder materializes each non-NaN matrix cell as a directed edge and
attaches node properties. Hierarchy inference belongs in
:class:`~vistiq.graph.HierarchyBuilder`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Any, ClassVar, List, Literal, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd
from prefect import task
from pydantic import Field, field_validator

from vistiq.core import Configurable, Configuration, generate_name

from vistiq.matrix.types import default_matrix_annotations

logger = logging.getLogger(__name__)


class GraphLike(ABC):
    """Backend-agnostic graph interface.

    Mirrors the networkx graph API so that :class:`NXGraph` is a thin
    forwarder and alternative backends can implement the same contract.
    Methods that networkx exposes as free functions (``has_path``,
    ``descendants``, ``diameter``, ...) are promoted to methods here so that
    consumers never reference the ``networkx`` namespace directly.

    The one deviation from networkx is node/edge attribute access:
    ``graph.nodes[node_id]`` becomes :meth:`node_attrs`, while the callable
    forms ``nodes(data=...)`` / ``edges(data=...)`` keep their networkx shape.
    """

    # --- construction / mutation ---
    @classmethod
    @abstractmethod
    def empty(cls, *, directed: bool = True) -> "GraphLike":
        """Return a new empty graph of the same backend."""

    @abstractmethod
    def add_node(self, node_id: Any, **attributes: Any) -> None: ...

    @abstractmethod
    def add_edge(self, source: Any, target: Any, **attributes: Any) -> None: ...

    # --- structure ---
    @abstractmethod
    def nodes(self, *, data: bool = False) -> Iterable[Any]:
        """Node ids, or ``(node_id, attrs)`` pairs when *data* is true."""

    @abstractmethod
    def node_attrs(self, node_id: Any) -> dict[str, Any]: ...

    @abstractmethod
    def has_node(self, node_id: Any) -> bool: ...

    @abstractmethod
    def edges(self, *, data: bool = False) -> Iterable[Any]:
        """``(source, target)`` pairs, or ``(source, target, attrs)`` triples."""

    @abstractmethod
    def edge_attrs(self, source: Any, target: Any) -> dict[str, Any]: ...

    @abstractmethod
    def has_edge(self, source: Any, target: Any) -> bool: ...

    @abstractmethod
    def out_edges(self, node_id: Any, *, data: bool = False) -> Iterable[Any]: ...

    @abstractmethod
    def number_of_nodes(self) -> int: ...

    @abstractmethod
    def number_of_edges(self) -> int: ...

    @abstractmethod
    def is_directed(self) -> bool: ...

    # --- traversal / degree ---
    @abstractmethod
    def successors(self, node_id: Any) -> Iterable[Any]: ...

    @abstractmethod
    def predecessors(self, node_id: Any) -> Iterable[Any]: ...

    @abstractmethod
    def in_degree(self, node_id: Any) -> int: ...

    @abstractmethod
    def out_degree(self, node_id: Any) -> int: ...

    @abstractmethod
    def degrees(self) -> Iterable[tuple[Any, int]]:
        """Iterate ``(node_id, degree)`` pairs."""

    # --- derived graphs ---
    @abstractmethod
    def subgraph(self, nodes: Iterable[Any]) -> "GraphLike": ...

    @abstractmethod
    def to_undirected(self) -> "GraphLike": ...

    # --- algorithms (networkx free functions promoted to methods) ---
    @abstractmethod
    def descendants(self, node_id: Any) -> set[Any]: ...

    @abstractmethod
    def ancestors(self, node_id: Any) -> set[Any]: ...

    @abstractmethod
    def topological_sort(self) -> Iterable[Any]: ...

    @abstractmethod
    def has_path(self, source: Any, target: Any) -> bool: ...

    @abstractmethod
    def shortest_path(self, source: Any, target: Any) -> list[Any]: ...

    @abstractmethod
    def single_source_shortest_path_length(self, source: Any) -> dict[Any, int]: ...

    @abstractmethod
    def is_dag(self) -> bool: ...

    @abstractmethod
    def dag_longest_path(self) -> list[Any]: ...

    @abstractmethod
    def all_pairs_shortest_path_length(self) -> dict[Any, dict[Any, int]]: ...

    @abstractmethod
    def connected_components(self) -> Iterable[set[Any]]: ...

    @abstractmethod
    def diameter(self) -> int: ...

    @abstractmethod
    def average_shortest_path_length(self) -> float: ...

    @abstractmethod
    def density(self) -> float: ...

    @abstractmethod
    def global_efficiency(self) -> float: ...

    # --- backend access ---
    @property
    @abstractmethod
    def raw(self) -> Any:
        """The underlying backend graph object."""

    def __contains__(self, node_id: Any) -> bool:
        return self.has_node(node_id)


class NXGraph(GraphLike):
    """networkx-backed :class:`GraphLike` implementation.

    Wraps a ``networkx`` ``DiGraph``/``Graph`` and forwards every operation to
    the corresponding method or ``networkx`` free function.
    """

    def __init__(self, graph: Any):
        self._g = graph

    @classmethod
    def empty(cls, *, directed: bool = True) -> "NXGraph":
        return cls(nx.DiGraph() if directed else nx.Graph())

    def add_node(self, node_id: Any, **attributes: Any) -> None:
        self._g.add_node(node_id, **attributes)

    def add_edge(self, source: Any, target: Any, **attributes: Any) -> None:
        self._g.add_edge(source, target, **attributes)

    def nodes(self, *, data: bool = False) -> Iterable[Any]:
        return self._g.nodes(data=data)

    def node_attrs(self, node_id: Any) -> dict[str, Any]:
        return self._g.nodes[node_id]

    def has_node(self, node_id: Any) -> bool:
        return node_id in self._g

    def edges(self, *, data: bool = False) -> Iterable[Any]:
        return self._g.edges(data=data)

    def edge_attrs(self, source: Any, target: Any) -> dict[str, Any]:
        return self._g.edges[source, target]

    def has_edge(self, source: Any, target: Any) -> bool:
        return self._g.has_edge(source, target)

    def out_edges(self, node_id: Any, *, data: bool = False) -> Iterable[Any]:
        return self._g.out_edges(node_id, data=data)

    def number_of_nodes(self) -> int:
        return self._g.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._g.number_of_edges()

    def is_directed(self) -> bool:
        return self._g.is_directed()

    def successors(self, node_id: Any) -> Iterable[Any]:
        return self._g.successors(node_id)

    def predecessors(self, node_id: Any) -> Iterable[Any]:
        return self._g.predecessors(node_id)

    def in_degree(self, node_id: Any) -> int:
        return self._g.in_degree(node_id)

    def out_degree(self, node_id: Any) -> int:
        return self._g.out_degree(node_id)

    def degrees(self) -> Iterable[tuple[Any, int]]:
        return self._g.degree()

    def subgraph(self, nodes: Iterable[Any]) -> "NXGraph":
        return NXGraph(self._g.subgraph(nodes).copy())

    def to_undirected(self) -> "NXGraph":
        return NXGraph(self._g.to_undirected())

    def descendants(self, node_id: Any) -> set[Any]:
        return nx.descendants(self._g, node_id)

    def ancestors(self, node_id: Any) -> set[Any]:
        return nx.ancestors(self._g, node_id)

    def topological_sort(self) -> Iterable[Any]:
        return nx.topological_sort(self._g)

    def has_path(self, source: Any, target: Any) -> bool:
        return nx.has_path(self._g, source, target)

    def shortest_path(self, source: Any, target: Any) -> list[Any]:
        return nx.shortest_path(self._g, source, target)

    def single_source_shortest_path_length(self, source: Any) -> dict[Any, int]:
        return dict(nx.single_source_shortest_path_length(self._g, source))

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def dag_longest_path(self) -> list[Any]:
        return nx.dag_longest_path(self._g)

    def all_pairs_shortest_path_length(self) -> dict[Any, dict[Any, int]]:
        return dict(nx.all_pairs_shortest_path_length(self._g))

    def connected_components(self) -> Iterable[set[Any]]:
        return nx.connected_components(self._g)

    def diameter(self) -> int:
        return int(nx.diameter(self._g))

    def average_shortest_path_length(self) -> float:
        return float(nx.average_shortest_path_length(self._g))

    def density(self) -> float:
        return float(nx.density(self._g))

    def global_efficiency(self) -> float:
        return float(nx.global_efficiency(self._g))

    @property
    def raw(self) -> Any:
        return self._g


def spatial_neighbor_column(
    analysis: str,
    metric: str,
    group: Any,
    *,
    k: Optional[int] = None,
    radius: Optional[float] = None,
) -> str:
    """Build a neighbor-summary column name.

    Returns ``{analysis}_{metric}_{group}_(k=…)`` or
    ``{analysis}_{metric}_{group}_(radius=…)``. Spaces in *group* are
    replaced with underscores (for example ``"lineage Region"`` becomes
    ``lineage_Region``).
    """
    if analysis == "knn":
        if k is None:
            raise ValueError("k is required for knn neighbor column names")
        param = f"k={k}"
    elif analysis == "rnn":
        if radius is None:
            raise ValueError("radius is required for rnn neighbor column names")
        radius_value = int(radius) if float(radius).is_integer() else radius
        param = f"radius={radius_value}"
    else:
        raise ValueError(f"unsupported neighbor analysis {analysis!r}")
    group_part = str(group).replace(" ", "_")
    return f"{analysis}_{metric}_{group_part}_({param})"


def _normalize_index(regions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of regions indexed by object_id.

    Accepts object_id as the index name, a column, or a MultiIndex level.
    """
    if regions.index.name == "object_id":
        return regions
    if isinstance(regions.index, pd.MultiIndex) and "object_id" in regions.index.names:
        return regions.reset_index().set_index("object_id", drop=False)
    if "object_id" in regions.columns:
        return regions.set_index("object_id", drop=False)
    raise KeyError(
        "regions must be indexed by object_id or include an object_id column; "
        f"index={regions.index.name!r}, columns={list(regions.columns)}"
    )


def graph_to_dataframe(graph: GraphLike, *, index: str = "object_id") -> pd.DataFrame:
    """Build a region property table from graph node attributes.

    Args:
        graph: :class:`GraphLike` instance with node attributes.
        index: Index name and column to use for object identifiers.

    Returns:
        DataFrame indexed by ``index`` with one row per node.
    """
    nodes = list(graph.nodes(data=True))
    if not nodes:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [dict(attrs) for _, attrs in nodes],
        index=pd.Index([node_id for node_id, _ in nodes], name=index),
    )
    if index in frame.columns:
        frame = frame.drop(columns=[index])
    return frame


def _attrs_match(attrs: dict[str, Any], criteria: dict[str, Any]) -> bool:
    return all(attrs.get(key) == value for key, value in criteria.items())


def edges_to_matrix(
    records: Sequence[dict[str, Any]],
    *,
    weight_attribute: str = "ios",
    endpoints: tuple[str, str] = ("source", "target"),
) -> pd.DataFrame:
    """Convert edge records to a labeled square adjacency matrix.

    Each record must provide the *endpoints* keys (``source``/``target`` by
    default, or ``parent``/``child`` for containment ``edges``). The
    *weight_attribute* value becomes the cell weight; records lacking it
    default to ``1.0`` (for example synthetic predecessor "jump" edges). The
    result is suitable for :meth:`GraphBuilder.run`.

    Args:
        records: Edge records, e.g. from a :class:`GraphQuery` result.
        weight_attribute: Record key read for cell weights.
        endpoints: ``(source_key, target_key)`` naming the endpoint columns.

    Returns:
        Square DataFrame indexed by the involved node ids; ``NaN`` off-edge.
    """
    if not records:
        return pd.DataFrame()
    source_key, target_key = endpoints
    nodes = sorted(
        {
            node_id
            for record in records
            for node_id in (record[source_key], record[target_key])
        },
        key=str,
    )
    matrix = pd.DataFrame(np.nan, index=nodes, columns=nodes)
    matrix.index.name = "object_id"
    for record in records:
        weight = record.get(weight_attribute)
        if weight is None or pd.isna(weight):
            weight = 1.0
        else:
            weight = float(weight)
        matrix.loc[record[source_key], record[target_key]] = weight
    return matrix


def subtree_origin_key(origin: Any) -> str:
    """String key for a spatial subtree root (``None`` → ``"all"``)."""
    return "all" if origin is None else str(origin)


def resolve_subtree_origins(
    graph: GraphLike,
    *,
    node: Optional[Any] = None,
    match: Optional[dict[str, Any]] = None,
    exclude: Optional[dict[str, Any]] = None,
    auto_root: bool = False,
) -> list[Optional[Any]]:
    """Resolve spatial-analysis subtree root(s) at runtime.

    Returns ``[None]`` when no scoping is requested (analyze the full graph).
    *exclude* removes candidates that match; it wins over *match* when both apply.
    """
    if node is not None:
        if node not in graph:
            raise KeyError(f"node {node!r} not found in graph")
        attrs = graph.node_attrs(node)
        if exclude and _attrs_match(attrs, exclude):
            raise KeyError(f"node {node!r} matches exclude criteria {exclude!r}")
        return [node]
    if match:
        origins = sorted(
            (
                node_id
                for node_id, attrs in graph.nodes(data=True)
                if _attrs_match(attrs, match)
                and not (exclude and _attrs_match(attrs, exclude))
            ),
            key=str,
        )
        if not origins:
            raise KeyError(f"no node matching {match!r} after exclude {exclude!r}")
        return origins
    if auto_root:
        origin = GraphQuery._resolve_origin(graph, None)
        attrs = graph.node_attrs(origin)
        if exclude and _attrs_match(attrs, exclude):
            raise KeyError(
                f"auto_root node {origin!r} matches exclude criteria {exclude!r}"
            )
        return [origin]
    return [None]


def resolve_subtree_origin(
    graph: GraphLike,
    *,
    node: Optional[Any] = None,
    match: Optional[dict[str, Any]] = None,
    exclude: Optional[dict[str, Any]] = None,
    auto_root: bool = False,
) -> Optional[Any]:
    """Resolve a single subtree root; raises when multiple nodes match *match*."""
    origins = resolve_subtree_origins(
        graph,
        node=node,
        match=match,
        exclude=exclude,
        auto_root=auto_root,
    )
    if len(origins) > 1:
        raise ValueError(
            f"spatial scope matched {len(origins)} subtree roots: {origins[:5]}; "
            "use AnalysisFlow spatial_scope mapping or pass node="
        )
    return origins[0]


class GraphExporterConfig(Configuration):
    """Configuration for :class:`GraphExporter`.

    Attributes:
        index: Index name for exported object identifiers.
        columns: Optional subset of node attributes to include. When omitted,
            all node attributes are exported.
        dropna_cols: When ``True``, drop columns that are entirely NA.
        dropna_rows: When ``True``, drop rows with any NA value.
        exclude_synthetic: When ``True``, omit nodes whose ``synthetic``
            attribute is truthy (structural DAG nodes from
            :class:`GraphBuilder`).
    """

    index: str = "object_id"
    columns: Optional[List[str]] = None
    dropna_cols: bool = False
    dropna_rows: bool = False
    exclude_synthetic: bool = False


class GraphExporter(Configurable[GraphExporterConfig]):
    """Export graph node attributes to a region property table."""

    def __init__(self, config: GraphExporterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: GraphExporterConfig) -> "GraphExporter":
        return cls(config)

    @task(name="GraphExporter.run", task_run_name=generate_name)
    def run(self, graph: GraphLike) -> pd.DataFrame:
        """Build a DataFrame from graph node attributes."""
        frame = graph_to_dataframe(graph, index=self.config.index)
        if frame.empty:
            return frame
        if self.config.columns is not None:
            frame = frame[list(self.config.columns)]
        if self.config.exclude_synthetic:
            if "synthetic" in frame.columns:
                frame = frame[~frame["synthetic"].fillna(False).astype(bool)]
            else:
                synthetic_ids = [
                    node_id
                    for node_id, attrs in graph.nodes(data=True)
                    if attrs.get("synthetic")
                ]
                frame = frame.drop(index=synthetic_ids, errors="ignore")
        if self.config.dropna_cols:
            frame = frame.dropna(axis=1, how="all")
        if self.config.dropna_rows:
            frame = frame.dropna(axis=0, how="any")
        return frame


def _regions_for_nodes(regions: pd.DataFrame, nodes: list[Any]) -> pd.DataFrame:
    """Align region metrics to the given node order.

    Args:
        regions: Region property table for all objects.
        nodes: object_id values in matrix row/column order.

    Returns:
        Region table reindexed to nodes.

    Raises:
        KeyError: If any node is missing from regions.
    """
    region_table = _normalize_index(regions)
    aligned = region_table.reindex(nodes)
    missing = aligned.index[aligned.isna().all(axis=1)]
    if len(missing) > 0:
        missing_ids = ", ".join(str(value) for value in missing[:5])
        raise KeyError(
            f"regions missing metrics for {len(missing)} object_id(s), "
            f"including: {missing_ids}"
        )
    return aligned


class GraphBuilderConfig(Configuration):
    """Configuration for GraphBuilder.

    Attributes:
        edge_attribute: Edge attribute key used when storing matrix cell values.
        synthetic_attribute: Edge attribute key set to ``True`` when either
            endpoint is a synthetic node (for example orphan-group roots).
        graph_type: Whether to build a directed or undirected graph.
        annotation_factory: Applied when the input :class:`~vistiq.matrix.MatrixData`
            has no annotations.
    """

    edge_attribute: str = "ios"
    synthetic_attribute: str = "synthetic"
    graph_type: Literal["directed", "undirected"] = "directed"
    annotation_factory: Any = default_matrix_annotations


class GraphBuilder(Configurable[GraphBuilderConfig]):
    """Materialize a labeled matrix into a graph with node attributes.

    Graph construction goes through the configured :class:`GraphLike` backend.
    """
    graph_backend: type[GraphLike] = NXGraph

    def __init__(self, config: GraphBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: GraphBuilderConfig) -> "GraphBuilder":
        return cls(config)

    @task(name="GraphBuilder.run", task_run_name=generate_name)
    def run(
        self,
        matrix: Any,
        nodes: pd.DataFrame,
        annotations: Any = None,
    ) -> GraphLike:
        """Build a graph from matrix weights and node metrics.

        Each non-NaN matrix cell becomes an edge from the row node to the
        column node. Node attributes come from ``nodes``.

        Args:
            matrix: :class:`~vistiq.matrix.MatrixData` or DataFrame
                (e.g. from :class:`HierarchyBuilder` or :func:`edges_to_matrix`).
            nodes: Node metrics for all objects, indexed by object_id.
            annotations: Reserved for future use (ignored).

        Returns:
            Backend-specific graph with node metrics on nodes.
        """
        from vistiq.matrix.types import (
            MatrixData,
            as_matrix_data,
            default_matrix_annotations,
            matrix_to_numpy,
            square_matrix,
        )

        data = as_matrix_data(matrix)
        if data.ndim != 2:
            raise ValueError(f"GraphBuilder requires a 2-D matrix; got ndim={data.ndim}")
        if data.annotations is None:
            factory = self.config.annotation_factory or default_matrix_annotations
            data = MatrixData(matrix=data.matrix, annotations=factory(data.shape))
        data = square_matrix(data)
        assert data.annotations is not None
        values = matrix_to_numpy(data)
        node_ids = list(data.annotations[0])
        logger.info(f"Building graph with config: {self.config}")
        logger.info(f"Matrix shape: {values.shape}")
        logger.info(f"Nodes shape: {nodes.shape}")
        del annotations  # reserved
        return self._build(node_table=nodes, node_ids=node_ids, values=values)

    def _new_graph(self) -> GraphLike:
        if self.config.graph_type not in {"directed", "undirected"}:
            raise ValueError(f"Invalid graph type: {self.config.graph_type}")
        return self.graph_backend.empty(directed=self.config.graph_type == "directed")

    def _build(
        self,
        node_table: pd.DataFrame,
        *,
        node_ids: list[Any],
        values: np.ndarray,
    ) -> GraphLike:
        aligned = _regions_for_nodes(node_table, node_ids)

        graph = self._new_graph()
        edge_attr = self.config.edge_attribute
        synthetic_attr = self.config.synthetic_attribute
        synthetic_nodes = {
            node
            for node in node_ids
            if bool(aligned.loc[node].get("synthetic", False))
        }

        for node in node_ids:
            raw = aligned.loc[node].to_dict()
            attrs = {key: value for key, value in raw.items() if pd.notna(value)}
            attrs["object_id"] = node
            graph.add_node(node, **attrs)

        node_map = {node: index for index, node in enumerate(node_ids)}
        for parent in node_ids:
            for child in node_ids:
                if parent == child:
                    continue
                weight = values[node_map[parent], node_map[child]]
                if pd.isna(weight):
                    continue
                edge_attrs: dict[str, Any] = {edge_attr: float(weight)}
                if parent in synthetic_nodes or child in synthetic_nodes:
                    edge_attrs[synthetic_attr] = True
                graph.add_edge(parent, child, **edge_attrs)

        return graph


_ENDPOINT_KEYS = frozenset({"source", "target"})
NodeMatch = Union[list[Any], dict[str, Any]]


def _is_endpoint_dict(node_match: dict[str, Any]) -> bool:
    return bool(_ENDPOINT_KEYS & node_match.keys())


def _node_in_endpoint(graph: GraphLike, node_id: Any, sel: Any) -> bool:
    if sel is None:
        return True
    if isinstance(sel, list):
        return node_id in sel
    if isinstance(sel, dict):
        return _attrs_match(graph.node_attrs(node_id), sel)
    raise TypeError(f"endpoint selector must be a list or dict; got {type(sel).__name__}")


def _resolve_endpoint_nodes(graph: GraphLike, sel: Any) -> set[Any]:
    if isinstance(sel, list):
        return {node_id for node_id in sel if node_id in graph}
    if isinstance(sel, dict):
        return {
            node_id
            for node_id, attrs in graph.nodes(data=True)
            if _attrs_match(attrs, sel)
        }
    raise TypeError(f"endpoint selector must be a list or dict; got {type(sel).__name__}")


def _parse_node_match(
    node_match: NodeMatch | None,
) -> tuple[Any | None, Any | None, set[Any] | None]:
    if node_match is None:
        return None, None, None
    if isinstance(node_match, list):
        return None, None, set(node_match)
    if isinstance(node_match, dict):
        if _is_endpoint_dict(node_match):
            return node_match.get("source"), node_match.get("target"), None
        return None, None, None
    raise TypeError(f"node_match must be a list or dict; got {type(node_match).__name__}")


def _path_start_nodes(graph: GraphLike, target_sel: Any) -> list[Any]:
    if target_sel is None:
        return sorted(graph.nodes(), key=str)
    return sorted(_resolve_endpoint_nodes(graph, target_sel), key=str)


def _find_upstream_endpoint(graph: GraphLike, start: Any, source_sel: Any) -> Any | None:
    if source_sel is None:
        return None
    queue = deque(graph.predecessors(start))
    seen: set[Any] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if _node_in_endpoint(graph, current, source_sel):
            return current
        queue.extend(graph.predecessors(current))
    return None


class GraphFilterConfig(Configuration):
    """Select nodes or edges from a graph.

    Attributes:
        mode: Selection mode — ``nodes``, existing ``edges``, synthetic
            ``direct_path`` jumps, or ``full_path`` edge chains.
        node_match: Node or endpoint selector. A **list** selects nodes by id;
            for edge modes, edges incident to any listed node are kept. A
            **dict** without ``source``/``target`` keys matches nodes by
            attribute. A **dict** with ``source`` and/or ``target`` keys
            selects edge/path endpoints; each side is a node-id list or an
            attribute dict.
    """

    mode: Literal["nodes", "edges", "direct_path", "full_path"] = "edges"
    node_match: Optional[NodeMatch] = None


class GraphFilter(Configurable[GraphFilterConfig]):
    """Filter a graph by node or edge/path selection."""

    def __init__(self, config: GraphFilterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: GraphFilterConfig) -> "GraphFilter":
        return cls(config)

    @task(name="GraphFilter.run", task_run_name=generate_name)
    def run(self, graph: GraphLike) -> list[Any]:
        """Return selected node ids or edge records.

        Edge records are ``{"source", "target", ...}`` dicts. ``direct_path``
        may emit edges not present in the input graph.
        """
        mode = self.config.mode
        node_match = self.config.node_match
        if mode == "nodes":
            return self._filter_nodes(graph, node_match)
        if mode == "edges":
            return self._filter_edges(graph, node_match)
        if mode == "direct_path":
            return self._filter_direct_paths(graph, node_match)
        if mode == "full_path":
            return self._filter_full_paths(graph, node_match)
        raise ValueError(f"unsupported filter mode {mode!r}")

    def _filter_nodes(
        self, graph: GraphLike, node_match: NodeMatch | None
    ) -> list[Any]:
        if node_match is None:
            return sorted(graph.nodes(), key=str)
        if isinstance(node_match, list):
            return sorted(
                (node_id for node_id in node_match if node_id in graph),
                key=str,
            )
        if isinstance(node_match, dict) and not _is_endpoint_dict(node_match):
            return sorted(_resolve_endpoint_nodes(graph, node_match), key=str)
        raise ValueError(
            "nodes mode expects node_match as a node-id list or attribute dict"
        )

    def _filter_edges(
        self, graph: GraphLike, node_match: NodeMatch | None
    ) -> list[dict[str, Any]]:
        source_sel, target_sel, incident = _parse_node_match(node_match)
        edges: list[dict[str, Any]] = []
        if incident is not None:
            for source, target, edge_data in graph.edges(data=True):
                if source in incident or target in incident:
                    edges.append({"source": source, "target": target, **edge_data})
            return edges

        for source, target, edge_data in graph.edges(data=True):
            if not _node_in_endpoint(graph, source, source_sel):
                continue
            if not _node_in_endpoint(graph, target, target_sel):
                continue
            edges.append({"source": source, "target": target, **edge_data})
        return edges

    def _filter_direct_paths(
        self, graph: GraphLike, node_match: NodeMatch | None
    ) -> list[dict[str, Any]]:
        source_sel, target_sel, incident = _parse_node_match(node_match)
        if incident is not None:
            raise ValueError("direct_path mode requires endpoint node_match dict")
        edges: list[dict[str, Any]] = []
        for target in _path_start_nodes(graph, target_sel):
            source = _find_upstream_endpoint(graph, target, source_sel)
            if source is None:
                continue
            if not graph.has_path(source, target):
                continue
            edges.append({"source": source, "target": target})
        return edges

    def _filter_full_paths(
        self, graph: GraphLike, node_match: NodeMatch | None
    ) -> list[dict[str, Any]]:
        source_sel, target_sel, incident = _parse_node_match(node_match)
        if incident is not None:
            raise ValueError("full_path mode requires endpoint node_match dict")
        edges: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        for target in _path_start_nodes(graph, target_sel):
            source = _find_upstream_endpoint(graph, target, source_sel)
            if source is None or not graph.has_path(source, target):
                continue
            path = graph.shortest_path(source, target)
            for index in range(len(path) - 1):
                u, v = path[index], path[index + 1]
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                edges.append({"source": u, "target": v, **dict(graph.edge_attrs(u, v))})
        return edges


class GraphQuery(Configurable["GraphQueryConfig"]):
    """Query a directed graph such as a containment DAG.

    Query keys are registered in _ATTRIBUTE_METHODS and computed by
    matching _summary_* methods on concrete subclasses. Use
    allowed_attributes() to see every key that may be requested in config;
    default_attributes lists keys computed when config.attributes is left
    empty. origin_attributes lists depth, descendant, and origin-scoped
    subgraph network metrics that require run(node=...) when the graph
    has more than one root.
    """

    default_attributes: ClassVar[tuple[str, ...]] = (
        "n_nodes",
        "n_edges",
        "n_roots",
        "roots",
        "n_leaves",
        "leaves",
        "parent_of",
        "children_of",
        "edges",
    )

    origin_attributes: ClassVar[tuple[str, ...]] = (
        "origin",
        "max_depth",
        "mean_depth",
        "depths",
        "subgraph_nodes",
        "subgraph_n_nodes",
        "subgraph_n_edges",
        "subgraph_longest_path",
        "subgraph_longest_path_length",
        "subgraph_diameter",
        "subgraph_average_shortest_path",
        "subgraph_density",
        "subgraph_average_degree",
        "subgraph_global_efficiency",
    )

    _ATTRIBUTE_METHODS: ClassVar[dict[str, str]] = {
        "n_nodes": "_summary_n_nodes",
        "n_edges": "_summary_n_edges",
        "n_roots": "_summary_n_roots",
        "n_leaves": "_summary_n_leaves",
        "origin": "_summary_origin",
        "roots": "_summary_roots",
        "leaves": "_summary_leaves",
        "max_depth": "_summary_max_depth",
        "mean_depth": "_summary_mean_depth",
        "depths": "_summary_depths",
        "subgraph_nodes": "_summary_subgraph_nodes",
        "subgraph_n_nodes": "_summary_subgraph_n_nodes",
        "subgraph_n_edges": "_summary_subgraph_n_edges",
        "subgraph_longest_path": "_summary_subgraph_longest_path",
        "subgraph_longest_path_length": "_summary_subgraph_longest_path_length",
        "subgraph_diameter": "_summary_subgraph_diameter",
        "subgraph_average_shortest_path": "_summary_subgraph_average_shortest_path",
        "subgraph_density": "_summary_subgraph_density",
        "subgraph_average_degree": "_summary_subgraph_average_degree",
        "subgraph_global_efficiency": "_summary_subgraph_global_efficiency",
        "parent_of": "_summary_parent_of",
        "children_of": "_summary_children_of",
        "node_attributes": "_summary_node_attributes",
        "edges": "_summary_edges",
        "filtered_edges": "_summary_filtered_edges",
        "first_matching_predecessor": "_summary_first_matching_predecessor",
        "matching_predecessor_edges": "_summary_matching_predecessor_edges",
        "node_labels": "_summary_node_labels",
        "nodes_by_attribute": "_summary_nodes_by_attribute",
        "descendant_counts": "_descendant_counts",
        "ancestor_lineage": "_ancestor_lineage",
        "neighbor_summary": "_neighbor_summary",
    }

    @classmethod
    def allowed_attributes(cls) -> List[str]:
        """Return summary keys that may be listed in GraphQueryConfig.attributes."""
        return list(cls._attribute_methods().keys())

    @classmethod
    def _attribute_methods(cls) -> dict[str, str]:
        return {**cls._ATTRIBUTE_METHODS, **cls._register_attribute_methods()}

    @classmethod
    def _register_attribute_methods(cls) -> dict[str, str]:
        """Return extra attribute-to-method entries for subclasses."""
        return {}

    @classmethod
    def from_config(cls, config: "GraphQueryConfig") -> "GraphQuery":
        return cls(config)

    @task(name="GraphQuery.format", task_run_name=generate_name)
    def format(
        self,
        output: list[dict[str, Any]] | dict[str, Any],
        attribute: Optional[str] = None,
    ) -> Any:
        """Format query rows or one attribute from a :meth:`run` result.

        Args:
            output: Tabular query rows, or the dictionary returned by
                :meth:`run` when ``attribute`` is set.
            attribute: Name of the key to format from a :meth:`run` result
                (for example ``"descendant_counts"``).
        """
        if attribute is not None:
            if not isinstance(output, dict):
                raise TypeError(
                    "attribute requires output from GraphQuery.run (a dict)"
                )
            rows = output.get(attribute)
            if rows is None:
                if self.config.output_type == "dataframe":
                    return pd.DataFrame()
                return []
            output = rows

        if self.config.output_type == "dataframe":
            rows = output if isinstance(output, list) else [output]
            if not rows:
                return pd.DataFrame()
            return self._to_dataframe(rows)
        if isinstance(output, dict):
            return output
        return list(output)

    def _to_dataframe(self, output: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(output).set_index(self.config.output_index)

    @task(name="GraphQuery.run", task_run_name=generate_name)
    def run(
        self,
        graph: GraphLike,
        *,
        node: Any = None,
        filter_value: Any = None,
        source_nodes: Optional[Sequence[Any]] = None,
        seed_nodes: Optional[Sequence[Any]] = None,
    ) -> dict[str, Any]:
        """Query a graph and return a plain dictionary of results."""
        updates: dict[str, Any] = {}
        if filter_value is not None:
            updates["filter_value"] = filter_value
        if source_nodes is not None:
            updates["source_nodes"] = list(source_nodes)
        if seed_nodes is not None:
            updates["seed_nodes"] = list(seed_nodes)
        query = self
        if updates:
            query = self.__class__(self.config.model_copy(update=updates))
        logger.info(f"Summarizing graph with config: {query.config}")
        logger.info(f"Node: {node}")
        attributes = query.config.attributes
        return query._summarize(graph, node=node, attributes=attributes)

    def _summarize(
        self,
        graph: GraphLike,
        *,
        node: Any,
        attributes: list[str],
    ) -> dict[str, Any]:
        methods = self._attribute_methods()
        return {
            name: getattr(self, methods[name])(graph, node)
            for name in attributes
        }

    @staticmethod
    def _roots(graph: GraphLike) -> list[Any]:
        return sorted((n for n in graph.nodes() if graph.in_degree(n) == 0), key=str)

    @staticmethod
    def _leaves(graph: GraphLike) -> list[Any]:
        return sorted((n for n in graph.nodes() if graph.out_degree(n) == 0), key=str)

    @staticmethod
    def _resolve_origin(graph: GraphLike, node: Any) -> Any:
        if node is not None:
            if node not in graph:
                raise KeyError(f"node {node!r} not found in graph")
            return node
        roots = GraphQuery._roots(graph)
        if len(roots) == 1:
            return roots[0]
        nodes = list(graph.nodes())
        if len(roots) == 0:
            return nodes[0] if nodes else None
        raise ValueError(
            f"graph has {len(roots)} root nodes; pass node= to choose analysis origin"
        )

    @staticmethod
    def _depths(graph: GraphLike, node: Any) -> dict[Any, int]:
        origin = GraphQuery._resolve_origin(graph, node)
        if origin is None:
            return {}
        return graph.single_source_shortest_path_length(origin)

    @staticmethod
    def _origin_subgraph(graph: GraphLike, node: Any) -> GraphLike:
        origin = GraphQuery._resolve_origin(graph, node)
        if origin is None:
            return graph.subgraph([])
        scope = {origin, *graph.descendants(origin)}
        return graph.subgraph(scope)

    @staticmethod
    def _largest_connected_undirected(graph: GraphLike) -> GraphLike:
        undirected = graph.to_undirected()
        if undirected.number_of_nodes() == 0:
            return undirected
        largest = max(undirected.connected_components(), key=len)
        return undirected.subgraph(largest)

    @staticmethod
    def _subgraph_longest_path(subgraph: GraphLike) -> list[Any]:
        if subgraph.number_of_nodes() == 0:
            return []
        if subgraph.is_directed() and subgraph.is_dag():
            return list(subgraph.dag_longest_path())
        undirected = GraphQuery._largest_connected_undirected(subgraph)
        if undirected.number_of_nodes() <= 1:
            return list(undirected.nodes())
        lengths = undirected.all_pairs_shortest_path_length()
        best_path: list[Any] = []
        best_length = -1
        nodes = list(undirected.nodes())
        for i, source in enumerate(nodes):
            for target in nodes[i + 1 :]:
                length = lengths[source][target]
                if length > best_length:
                    best_length = length
                    best_path = undirected.shortest_path(source, target)
        return best_path

    def _summary_n_nodes(self, graph: GraphLike, node: Any) -> int:
        return graph.number_of_nodes()

    def _summary_n_edges(self, graph: GraphLike, node: Any) -> int:
        return graph.number_of_edges()

    def _summary_n_roots(self, graph: GraphLike, node: Any) -> int:
        return len(self._roots(graph))

    def _summary_n_leaves(self, graph: GraphLike, node: Any) -> int:
        return len(self._leaves(graph))

    def _summary_origin(self, graph: GraphLike, node: Any) -> Any:
        return self._resolve_origin(graph, node)

    def _summary_roots(self, graph: GraphLike, node: Any) -> list[Any]:
        return self._roots(graph)

    def _summary_leaves(self, graph: GraphLike, node: Any) -> list[Any]:
        return self._leaves(graph)

    def _summary_max_depth(self, graph: GraphLike, node: Any) -> int:
        values = list(self._depths(graph, node).values())
        return max(values) if values else 0

    def _summary_mean_depth(self, graph: GraphLike, node: Any) -> float:
        values = list(self._depths(graph, node).values())
        return float(np.mean(values)) if values else 0.0

    def _summary_depths(self, graph: GraphLike, node: Any) -> dict[Any, int]:
        return self._depths(graph, node)

    def _summary_subgraph_nodes(self, graph: GraphLike, node: Any) -> list[Any]:
        return sorted(self._origin_subgraph(graph, node).nodes(), key=str)

    def _summary_subgraph_n_nodes(self, graph: GraphLike, node: Any) -> int:
        return self._origin_subgraph(graph, node).number_of_nodes()

    def _summary_subgraph_n_edges(self, graph: GraphLike, node: Any) -> int:
        return self._origin_subgraph(graph, node).number_of_edges()

    def _summary_subgraph_longest_path(self, graph: GraphLike, node: Any) -> list[Any]:
        return self._subgraph_longest_path(self._origin_subgraph(graph, node))

    def _summary_subgraph_longest_path_length(self, graph: GraphLike, node: Any) -> int:
        path = self._summary_subgraph_longest_path(graph, node)
        return max(len(path) - 1, 0)

    def _summary_subgraph_diameter(self, graph: GraphLike, node: Any) -> int:
        subgraph = self._largest_connected_undirected(self._origin_subgraph(graph, node))
        if subgraph.number_of_nodes() <= 1:
            return 0
        return subgraph.diameter()

    def _summary_subgraph_average_shortest_path(self, graph: GraphLike, node: Any) -> float:
        subgraph = self._largest_connected_undirected(self._origin_subgraph(graph, node))
        if subgraph.number_of_nodes() <= 1:
            return 0.0
        return subgraph.average_shortest_path_length()

    def _summary_subgraph_density(self, graph: GraphLike, node: Any) -> float:
        return self._origin_subgraph(graph, node).density()

    def _summary_subgraph_average_degree(self, graph: GraphLike, node: Any) -> float:
        subgraph = self._origin_subgraph(graph, node)
        n = subgraph.number_of_nodes()
        if n == 0:
            return 0.0
        degree_sum = sum(degree for _, degree in subgraph.degrees())
        return float(degree_sum / n)

    def _summary_subgraph_global_efficiency(self, graph: GraphLike, node: Any) -> float:
        subgraph = self._origin_subgraph(graph, node).to_undirected()
        if subgraph.number_of_nodes() <= 1:
            return 0.0
        return subgraph.global_efficiency()

    def _summary_parent_of(self, graph: GraphLike, node: Any) -> dict[Any, Any | None]:
        parent_of = {node_id: None for node_id in graph.nodes()}
        for parent, child in graph.edges():
            parent_of[child] = parent
        return parent_of

    def _summary_children_of(self, graph: GraphLike, node: Any) -> dict[Any, list[Any]]:
        return {node_id: list(graph.successors(node_id)) for node_id in graph.nodes()}

    def _summary_node_attributes(
        self, graph: GraphLike, node: Any
    ) -> dict[Any, dict[str, Any]]:
        return {node_id: dict(graph.node_attrs(node_id)) for node_id in graph.nodes()}

    def _summary_edges(self, graph: GraphLike, node: Any) -> list[dict[str, Any]]:
        return [
            {"parent": parent, "child": child, **edge_data}
            for parent, child, edge_data in graph.edges(data=True)
        ]

    def _summary_filtered_edges(self, graph: GraphLike, node: Any) -> list[dict[str, Any]]:
        del node
        cfg = self.config
        node_match: dict[str, Any] = {}
        if cfg.source_filter is not None and cfg.source_nodes is not None:
            node_match["source"] = [
                node_id
                for node_id in cfg.source_nodes
                if node_id in graph
                and _attrs_match(graph.node_attrs(node_id), cfg.source_filter)
            ]
        elif cfg.source_filter is not None:
            node_match["source"] = cfg.source_filter
        elif cfg.source_nodes is not None:
            node_match["source"] = list(cfg.source_nodes)
        if cfg.target_filter is not None:
            node_match["target"] = cfg.target_filter
        return GraphFilter(
            GraphFilterConfig(mode="edges", node_match=node_match or None)
        ).run(graph)

    def _summary_first_matching_predecessor(
        self, graph: GraphLike, node: Any
    ) -> dict[Any, Any | None]:
        del node
        match = self.config.predecessor_match
        if not match:
            raise ValueError(
                "predecessor_match is required for first_matching_predecessor"
            )
        target_sel = (
            list(self.config.seed_nodes)
            if self.config.seed_nodes is not None
            else None
        )
        edges = GraphFilter(
            GraphFilterConfig(
                mode="direct_path",
                node_match={"source": match, "target": target_sel},
            )
        ).run(graph)
        predecessors = {seed: None for seed in _path_start_nodes(graph, target_sel)}
        for edge in edges:
            predecessors[edge["target"]] = edge["source"]
        return predecessors

    def _summary_matching_predecessor_edges(
        self, graph: GraphLike, node: Any
    ) -> list[dict[str, Any]]:
        del node
        match = self.config.predecessor_match
        if not match:
            raise ValueError(
                "predecessor_match is required for matching_predecessor_edges"
            )
        target_sel = (
            list(self.config.seed_nodes)
            if self.config.seed_nodes is not None
            else None
        )
        return GraphFilter(
            GraphFilterConfig(
                mode="direct_path",
                node_match={"source": match, "target": target_sel},
            )
        ).run(graph)

    def _summary_node_labels(self, graph: GraphLike, node: Any) -> dict[Any, str]:
        label_attribute = self.config.label_attribute
        if not label_attribute:
            return {}
        return {
            node_id: str(value)
            for node_id in graph.nodes()
            if (value := graph.node_attrs(node_id).get(label_attribute)) is not None
        }

    def _summary_nodes_by_attribute(self, graph: GraphLike, node: Any) -> dict[str, int]:
        return self._nodes_by_attribute(graph, self.config.group_attribute)

    def _nodes_by_attribute(self, graph: GraphLike, attribute: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node_id in graph.nodes():
            value = graph.node_attrs(node_id).get(attribute)
            if value is not None:
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _seed_nodes(
        self,
        graph: GraphLike,
        node: Any,
        attr_key: Optional[str],
        attr_value: Any,
    ) -> list[Any]:
        if node is not None:
            if node not in graph:
                raise KeyError(f"node {node!r} not found in graph")
            scope = {node, *graph.descendants(node)}
            nodes_to_search = [
                (node_id, dict(graph.node_attrs(node_id))) for node_id in scope
            ]
        else:
            nodes_to_search = list(graph.nodes(data=True))

        if attr_key is not None and attr_value is not None:
            return [
                node_id
                for node_id, attrs in nodes_to_search
                if attrs.get(attr_key) == attr_value
            ]
        return [node_id for node_id, _attrs in nodes_to_search]

    def _descendant_counts(self, graph: GraphLike, node: Any) -> list[dict[str, Any]]:
        classify_attribute = self.config.filter_attribute
        include = list(self.config.include_attributes)
        nodes = self._seed_nodes(
            graph, node, classify_attribute, self.config.filter_value
        )

        data = []
        for node_id in nodes:
            base_keys = list(set(["object_id", self.config.output_index, *include]))
            attrs = graph.node_attrs(node_id)
            row = {key: attrs.get(key) for key in base_keys}
            for desc in graph.descendants(node_id):
                bucket = graph.node_attrs(desc).get(classify_attribute, None)
                key = f"count {bucket}"
                row[key] = row.get(key, 0) + 1
            data.append(row)
        return data

    def _ancestor_lineage(self, graph: GraphLike, node: Any) -> list[dict[str, Any]]:
        classify_attribute = self.config.filter_attribute
        include = list(self.config.include_attributes)
        value_attribute = self.config.lineage_value_attribute
        nodes = self._seed_nodes(
            graph, node, classify_attribute, self.config.filter_value
        )

        data = []
        for node_id in nodes:
            base_keys = list(set(["object_id", self.config.output_index, *include]))
            attrs = graph.node_attrs(node_id)
            row = {key: attrs.get(key) for key in base_keys}
            ancestors = graph.ancestors(node_id)
            for ancestor in graph.topological_sort():
                if ancestor not in ancestors:
                    continue
                ancestor_attrs = graph.node_attrs(ancestor)
                group = ancestor_attrs.get(classify_attribute)
                value = ancestor_attrs.get(value_attribute)
                row[f"lineage {group}"] = int(value) if value is not None else None
            data.append(row)
        return data

    def _neighbor_summary(self, graph: GraphLike, node: Any) -> list[dict[str, Any]]:
        del node
        weight_key = self.config.weight_attribute
        group_attr = self.config.group_attribute
        analysis = self.config.neighbor_analysis
        neighbor_k = self.config.neighbor_k
        neighbor_radius = self.config.neighbor_radius
        include = list(self.config.include_attributes)
        data: list[dict[str, Any]] = []
        for node_id in graph.nodes():
            row: dict[str, Any] = {self.config.output_index: node_id}
            node_attrs = graph.node_attrs(node_id)
            if analysis is None:
                for key in set(include):
                    row[key] = node_attrs.get(key)

            by_group: dict[Any, list[tuple[float, Any]]] = {}
            for _, child, edge_data in graph.out_edges(node_id, data=True):
                weight = edge_data.get(weight_key)
                if weight is None or pd.isna(weight):
                    continue
                weight = float(weight)
                group_val = graph.node_attrs(child).get(group_attr, "__none__")
                by_group.setdefault(group_val, []).append((weight, child))

            if analysis is not None and (
                (analysis == "knn" and neighbor_k is not None)
                or (analysis == "rnn" and neighbor_radius is not None)
            ):
                for group_val in sorted(by_group, key=str):
                    entries = by_group[group_val]
                    if not entries:
                        continue
                    weights = [entry[0] for entry in entries]
                    nearest_distance, nearest_id = min(entries, key=lambda item: item[0])
                    column_kwargs = {"k": neighbor_k, "radius": neighbor_radius}
                    row[
                        spatial_neighbor_column(
                            analysis, "count", group_val, **column_kwargs
                        )
                    ] = len(weights)
                    row[
                        spatial_neighbor_column(
                            analysis, "mean_distance", group_val, **column_kwargs
                        )
                    ] = float(sum(weights) / len(weights))
                    row[
                        spatial_neighbor_column(
                            analysis,
                            "nearest_neighbor_distance",
                            group_val,
                            **column_kwargs,
                        )
                    ] = nearest_distance
                    row[
                        spatial_neighbor_column(
                            analysis,
                            "nearest_neighbor_id",
                            group_val,
                            **column_kwargs,
                        )
                    ] = nearest_id
                data.append(row)
                continue

            weights: list[float] = []
            nearest_id: Any = None
            nearest_distance: Optional[float] = None
            for grouped in by_group.values():
                for weight, child in grouped:
                    weights.append(weight)
                    if nearest_distance is None or weight < nearest_distance:
                        nearest_distance = weight
                        nearest_id = child

            row["knn_count"] = len(weights)
            if weights:
                row["nearest_neighbor_distance"] = nearest_distance
                row["knn_mean_distance"] = float(sum(weights) / len(weights))
                row["nearest_neighbor_id"] = nearest_id
            data.append(row)
        return data


class GraphQueryConfig(Configuration):
    """Configuration for GraphQuery.

    Attributes:
        label_attribute: Node attribute read when node_labels is requested.
            Set to None to skip label collection.
        group_attribute: Node attribute used when nodes_by_attribute is requested.
        filter_attribute: Node attribute used to select seed nodes and to
            classify descendants or ancestors in descendant_counts and
            ancestor_lineage. Pair with filter_value for seed selection.
        filter_value: Node attribute value used with filter_attribute.
        include_attributes: Extra node attributes to include in each row of
            descendant_counts and ancestor_lineage.
        lineage_value_attribute: Node attribute stored in each lineage column
            of ancestor_lineage (for example label).
        weight_attribute: Edge attribute read for neighbor_summary distances.
        source_filter: When ``filtered_edges`` is requested, every key in this
            dict must match on the edge source node.
        target_filter: When ``filtered_edges`` is requested, every key in this
            dict must match on the edge target node.
        source_nodes: When ``filtered_edges`` is requested, restrict to edges
            whose source is in this list.
        predecessor_match: When ``first_matching_predecessor`` or
            ``matching_predecessor_edges`` is requested, every key in this dict
            must match the first upstream predecessor found by BFS.
        seed_nodes: Seed node ids for predecessor walks. Defaults to all
            graph nodes when omitted.
        attributes: Query keys to compute. Each name must appear in
            GraphQuery.allowed_attributes(). When omitted, defaults to
            GraphQuery.default_attributes.
    """

    label_attribute: Optional[str] = "object_name"
    group_attribute: str = "channel"
    filter_attribute: Optional[str] = None
    filter_value: Any = None
    include_attributes: List[str] = Field(default_factory=list)
    lineage_value_attribute: str = "label"
    weight_attribute: str = "ios"
    neighbor_analysis: Optional[Literal["knn", "rnn"]] = None
    neighbor_k: Optional[int] = None
    neighbor_radius: Optional[float] = None
    source_filter: Optional[dict[str, Any]] = None
    target_filter: Optional[dict[str, Any]] = None
    source_nodes: Optional[List[Any]] = None
    predecessor_match: Optional[dict[str, Any]] = None
    seed_nodes: Optional[List[Any]] = None
    attributes: List[str] = Field(
        default_factory=lambda: list(GraphQuery.default_attributes)
    )
    output_type: Literal["dataframe", "list"] = "list"
    output_index: str = "object_id"

    @field_validator("attributes")
    @classmethod
    def validate_attributes(cls, value: List[str]) -> List[str]:
        allowed = set(GraphQuery.allowed_attributes())
        invalid = [name for name in value if name not in allowed]
        if invalid:
            raise ValueError(
                f"One or more invalid attributes: {invalid}. "
                f"Use names from {GraphQuery.allowed_attributes()}."
            )
        return list(dict.fromkeys(value))
