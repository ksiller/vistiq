"""Build graphs from labeled matrices and region metrics.

GraphBuilder materializes each non-NaN matrix cell as a directed edge and
attaches region properties to nodes. Hierarchy shaping belongs upstream in
:class:`~vistiq.analysis.matrix.HierarchicalMatrix`.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, List, Literal, Optional

import networkx as nx
import numpy as np
import pandas as pd
from prefect import task
from pydantic import Field, field_validator

from vistiq.analysis.matrix import _square_dataframe
from vistiq.core import Configurable, Configuration, generate_name

logger = logging.getLogger(__name__)


def _sanitize_neighbor_column_part(value: Any) -> str:
    return str(value).replace(" ", "_")


def spatial_neighbor_column(
    analysis: str,
    metric: str,
    group: Any,
    *,
    k: Optional[int] = None,
    radius: Optional[float] = None,
) -> str:
    """Build ``{analysis}_{metric}_{group}_(k=…)`` or ``…_(radius=…)``."""
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
    return (
        f"{analysis}_{metric}_{_sanitize_neighbor_column_part(group)}_({param})"
    )


def _normalize_regions_index(regions: pd.DataFrame) -> pd.DataFrame:
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


def graph_to_dataframe(graph: Any, *, index: str = "object_id") -> pd.DataFrame:
    """Build a region property table from graph node attributes.

    Args:
        graph: Backend graph with ``nodes(data=True)`` (e.g. networkx).
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


def subtree_origin_key(origin: Any) -> str:
    """String key for a spatial subtree root (``None`` → ``"all"``)."""
    return "all" if origin is None else str(origin)


def resolve_subtree_origins(
    graph: Any,
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
        attrs = graph.nodes[node]
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
        origin = NXGraphQuery._resolve_origin(graph, None)
        attrs = graph.nodes[origin]
        if exclude and _attrs_match(attrs, exclude):
            raise KeyError(
                f"auto_root node {origin!r} matches exclude criteria {exclude!r}"
            )
        return [origin]
    return [None]


def resolve_subtree_origin(
    graph: Any,
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
    def run(self, graph: Any) -> pd.DataFrame:
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
    region_table = _normalize_regions_index(regions)
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
        weight_attribute: Edge attribute key used when storing matrix weights.
        synthetic_attribute: Edge attribute key set to ``True`` when either
            endpoint is a synthetic node (for example orphan-group roots).
        graph_type: Whether to build a directed or undirected graph.
    """

    weight_attribute: str = "ios"
    synthetic_attribute: str = "synthetic"
    graph_type: Literal["directed", "undirected"] = "directed"


class GraphBuilder(Configurable[GraphBuilderConfig]):
    """Materialize a labeled matrix into a graph with region node attributes.

    Subclasses implement backend hooks (_add_node, _create_edge, _new_*_graph).
    """

    def __init__(self, config: GraphBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: GraphBuilderConfig) -> "GraphBuilder":
        return cls(config)

    @task(name="GraphBuilder.run", task_run_name=generate_name)
    def run(
        self,
        matrix: pd.DataFrame,
        regions: pd.DataFrame,
        annotations: Any = None,
    ) -> Any:
        """Build a graph from matrix weights and region metrics.

        Each non-NaN matrix cell becomes an edge from the row node to the
        column node. Node attributes come from ``regions``.

        Args:
            matrix: Labeled square matrix (e.g. from :class:`HierarchicalMatrix`).
            regions: Region metrics for all objects, indexed by object_id.
            annotations: Reserved for future use (ignored).

        Returns:
            Backend-specific graph with region metrics on nodes.
        """
        logger.info(f"Building graph with config: {self.config}")
        logger.info(f"Matrix shape: {matrix.shape}")
        logger.info(f"Regions shape: {regions.shape}")
        logger.info(f"Annotations: {annotations}")
        del annotations  # reserved
        return self._build(matrix=matrix, regions=regions)

    def _new_directed_graph(self) -> Any:
        raise NotImplementedError("Subclasses must implement _new_directed_graph")

    def _new_undirected_graph(self) -> Any:
        raise NotImplementedError("Subclasses must implement _new_undirected_graph")

    def _new_graph(self) -> Any:
        if self.config.graph_type == "directed":
            return self._new_directed_graph()
        elif self.config.graph_type == "undirected":
            return self._new_undirected_graph()
        else:
            raise ValueError(f"Invalid graph type: {self.config.graph_type}")

    def _add_node(self, graph: Any, node_id: Any, attributes: dict[str, Any]) -> None:
        raise NotImplementedError("Subclasses must implement _add_node")

    def _create_edge(
        self,
        graph: Any,
        node1: Any,
        node2: Any,
        *,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError("Subclasses must implement _create_edge")

    def _build(
        self,
        matrix: pd.DataFrame,
        regions: pd.DataFrame,
    ) -> Any:
        weight_matrix = _square_dataframe(matrix)
        nodes = list(weight_matrix.index)
        region_table = _regions_for_nodes(regions, nodes)

        graph = self._new_graph()
        weight_attr = self.config.weight_attribute
        synthetic_attr = self.config.synthetic_attribute
        synthetic_nodes = {
            node
            for node in nodes
            if bool(region_table.loc[node].get("synthetic", False))
        }

        for node in nodes:
            raw = region_table.loc[node].to_dict()
            attrs = {key: value for key, value in raw.items() if pd.notna(value)}
            attrs["object_id"] = node
            self._add_node(graph, node, attrs)

        for parent in nodes:
            for child in nodes:
                if parent == child:
                    continue
                weight = weight_matrix.loc[parent, child]
                if pd.isna(weight):
                    continue
                edge_attrs: dict[str, Any] = {weight_attr: float(weight)}
                if parent in synthetic_nodes or child in synthetic_nodes:
                    edge_attrs[synthetic_attr] = True
                self._create_edge(
                    graph,
                    parent,
                    child,
                    attributes=edge_attrs,
                )

        return graph


class NXGraphBuilderConfig(GraphBuilderConfig):
    """Configuration for NXGraphBuilder."""


class NXGraphBuilder(GraphBuilder):
    """NetworkX GraphBuilder using DiGraph or Graph per config.graph_type."""

    def __init__(self, config: NXGraphBuilderConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: NXGraphBuilderConfig) -> "NXGraphBuilder":
        return cls(config)

    def _new_directed_graph(self) -> Any:
        import networkx as nx

        return nx.DiGraph()

    def _new_undirected_graph(self) -> Any:
        import networkx as nx

        return nx.Graph()

    def _add_node(self, graph: Any, node_id: Any, attributes: dict[str, Any]) -> None:
        graph.add_node(node_id, **attributes)

    def _create_edge(
        self,
        graph: Any,
        node1: Any,
        node2: Any,
        *,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        if attributes:
            graph.add_edge(node1, node2, **attributes)
        else:
            graph.add_edge(node1, node2)


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
        raise NotImplementedError("Subclasses must implement _to_dataframe")

    @task(name="GraphQuery.run", task_run_name=generate_name)
    def run(
        self,
        graph: Any,
        *,
        node: Any = None,
        filter_value: Any = None,
    ) -> dict[str, Any]:
        """Query a graph and return a plain dictionary of results.

        Args:
            graph: Backend-specific graph (e.g. networkx DiGraph or Graph).
            node: Origin that defines the subgraph for depth and network
                statistics. Required when config.attributes includes any
                origin_attributes and the graph has more than one root.
            filter_value: Optional override for config.filter_value, useful
                when calling :meth:`run` via ``run.map`` per channel.

        Returns:
            Dictionary whose keys are the configured attributes.
        """
        query = self
        if filter_value is not None:
            query = self.__class__(
                self.config.model_copy(update={"filter_value": filter_value})
            )
        logger.info(f"Summarizing graph with config: {query.config}")
        logger.info(f"Node: {node}")
        attributes = query.config.attributes
        return query._summarize(graph, node=node, attributes=attributes)

    def _summarize(
        self,
        graph: Any,
        *,
        node: Any,
        attributes: list[str],
    ) -> dict[str, Any]:
        methods = self._attribute_methods()
        return {
            name: getattr(self, methods[name])(graph, node)
            for name in attributes
        }

    def _summary_n_nodes(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_n_nodes")

    def _summary_n_edges(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_n_edges")

    def _summary_n_roots(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_n_roots")

    def _summary_n_leaves(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_n_leaves")

    def _summary_origin(self, graph: Any, node: Any) -> Any:
        raise NotImplementedError("Subclasses must implement _summary_origin")

    def _summary_roots(self, graph: Any, node: Any) -> list[Any]:
        raise NotImplementedError("Subclasses must implement _summary_roots")

    def _summary_leaves(self, graph: Any, node: Any) -> list[Any]:
        raise NotImplementedError("Subclasses must implement _summary_leaves")

    def _summary_max_depth(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_max_depth")

    def _summary_mean_depth(self, graph: Any, node: Any) -> float:
        raise NotImplementedError("Subclasses must implement _summary_mean_depth")

    def _summary_depths(self, graph: Any, node: Any) -> dict[Any, int]:
        raise NotImplementedError("Subclasses must implement _summary_depths")

    def _summary_subgraph_nodes(self, graph: Any, node: Any) -> list[Any]:
        raise NotImplementedError("Subclasses must implement _summary_subgraph_nodes")

    def _summary_subgraph_n_nodes(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_subgraph_n_nodes")

    def _summary_subgraph_n_edges(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_subgraph_n_edges")

    def _summary_subgraph_longest_path(self, graph: Any, node: Any) -> list[Any]:
        raise NotImplementedError(
            "Subclasses must implement _summary_subgraph_longest_path"
        )

    def _summary_subgraph_longest_path_length(self, graph: Any, node: Any) -> int:
        raise NotImplementedError(
            "Subclasses must implement _summary_subgraph_longest_path_length"
        )

    def _summary_subgraph_diameter(self, graph: Any, node: Any) -> int:
        raise NotImplementedError(
            "Subclasses must implement _summary_subgraph_diameter"
        )

    def _summary_subgraph_average_shortest_path(
        self, graph: Any, node: Any
    ) -> float:
        raise NotImplementedError(
            "Subclasses must implement _summary_subgraph_average_shortest_path"
        )

    def _summary_subgraph_density(self, graph: Any, node: Any) -> float:
        raise NotImplementedError("Subclasses must implement _summary_subgraph_density")

    def _summary_subgraph_average_degree(self, graph: Any, node: Any) -> float:
        raise NotImplementedError(
            "Subclasses must implement _summary_subgraph_average_degree"
        )

    def _summary_subgraph_global_efficiency(self, graph: Any, node: Any) -> float:
        raise NotImplementedError(
            "Subclasses must implement _summary_subgraph_global_efficiency"
        )

    def _summary_parent_of(self, graph: Any, node: Any) -> dict[Any, Any | None]:
        raise NotImplementedError("Subclasses must implement _summary_parent_of")

    def _summary_children_of(self, graph: Any, node: Any) -> dict[Any, list[Any]]:
        raise NotImplementedError("Subclasses must implement _summary_children_of")

    def _summary_node_attributes(self, graph: Any, node: Any) -> dict[Any, dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _summary_node_attributes")

    def _summary_edges(self, graph: Any, node: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _summary_edges")

    def _summary_node_labels(self, graph: Any, node: Any) -> dict[Any, str]:
        raise NotImplementedError("Subclasses must implement _summary_node_labels")

    def _summary_nodes_by_attribute(self, graph: Any, node: Any) -> dict[str, int]:
        return self._nodes_by_attribute(graph, self.config.group_attribute)

    def _nodes_by_attribute(self, graph: Any, attribute: str) -> dict[str, int]:
        raise NotImplementedError("Subclasses must implement _nodes_by_attribute")

    def _descendant_counts(
        self, graph: Any, node: Any
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _descendant_counts")

    def _ancestor_lineage(
        self, graph: Any, node: Any
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _ancestor_lineage")

    def _neighbor_summary(
        self, graph: Any, node: Any
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _neighbor_summary")


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


class NXGraphQueryConfig(GraphQueryConfig):
    """Configuration for NXGraphQuery."""


class NXGraphQuery(GraphQuery):
    """GraphQuery implementation for networkx DiGraph and Graph."""

    @classmethod
    def from_config(cls, config: NXGraphQueryConfig) -> "NXGraphQuery":
        return cls(config)

    @staticmethod
    def _roots(graph: Any) -> list[Any]:
        return sorted((n for n in graph.nodes if graph.in_degree(n) == 0), key=str)

    @staticmethod
    def _leaves(graph: Any) -> list[Any]:
        return sorted((n for n in graph.nodes if graph.out_degree(n) == 0), key=str)

    @staticmethod
    def _resolve_origin(graph: Any, node: Any) -> Any:
        if node is not None:
            if node not in graph:
                raise KeyError(f"node {node!r} not found in graph")
            return node
        roots = NXGraphQuery._roots(graph)
        if len(roots) == 1:
            return roots[0]
        nodes = list(graph.nodes)
        if len(roots) == 0:
            return nodes[0] if nodes else None
        raise ValueError(
            f"graph has {len(roots)} root nodes; pass node= to choose analysis origin"
        )

    @staticmethod
    def _depths(graph: Any, node: Any) -> dict[Any, int]:
        origin = NXGraphQuery._resolve_origin(graph, node)
        if origin is None:
            return {}
        return dict(nx.single_source_shortest_path_length(graph, origin))

    @staticmethod
    def _origin_subgraph(graph: Any, node: Any) -> Any:
        origin = NXGraphQuery._resolve_origin(graph, node)
        if origin is None:
            return graph.subgraph([]).copy()
        scope = {origin, *nx.descendants(graph, origin)}
        return graph.subgraph(scope).copy()

    @staticmethod
    def _largest_connected_undirected(graph: Any) -> Any:
        undirected = graph.to_undirected()
        if undirected.number_of_nodes() == 0:
            return undirected
        largest = max(nx.connected_components(undirected), key=len)
        return undirected.subgraph(largest).copy()

    @staticmethod
    def _subgraph_longest_path(subgraph: Any) -> list[Any]:
        if subgraph.number_of_nodes() == 0:
            return []
        if isinstance(subgraph, nx.DiGraph) and nx.is_directed_acyclic_graph(
            subgraph
        ):
            return list(nx.dag_longest_path(subgraph))
        undirected = NXGraphQuery._largest_connected_undirected(subgraph)
        if undirected.number_of_nodes() <= 1:
            return list(undirected.nodes)
        lengths = dict(nx.all_pairs_shortest_path_length(undirected))
        best_path: list[Any] = []
        best_length = -1
        nodes = list(undirected.nodes)
        for i, source in enumerate(nodes):
            for target in nodes[i + 1 :]:
                length = lengths[source][target]
                if length > best_length:
                    best_length = length
                    best_path = nx.shortest_path(undirected, source, target)
        return best_path

    def _summarize(
        self,
        graph: Any,
        *,
        node: Any,
        attributes: list[str],
    ) -> dict[str, Any]:
        if not isinstance(graph, (nx.DiGraph, nx.Graph)):
            raise TypeError(
                "NXGraphQuery expects a networkx DiGraph or Graph; "
                f"got {type(graph).__name__}"
            )
        return super()._summarize(graph, node=node, attributes=attributes)

    def _summary_n_nodes(self, graph: Any, node: Any) -> int:
        return graph.number_of_nodes()

    def _summary_n_edges(self, graph: Any, node: Any) -> int:
        return graph.number_of_edges()

    def _summary_n_roots(self, graph: Any, node: Any) -> int:
        return len(self._roots(graph))

    def _summary_n_leaves(self, graph: Any, node: Any) -> int:
        return len(self._leaves(graph))

    def _summary_origin(self, graph: Any, node: Any) -> Any:
        return self._resolve_origin(graph, node)

    def _summary_roots(self, graph: Any, node: Any) -> list[Any]:
        return self._roots(graph)

    def _summary_leaves(self, graph: Any, node: Any) -> list[Any]:
        return self._leaves(graph)

    def _summary_max_depth(self, graph: Any, node: Any) -> int:
        values = list(self._depths(graph, node).values())
        return max(values) if values else 0

    def _summary_mean_depth(self, graph: Any, node: Any) -> float:
        values = list(self._depths(graph, node).values())
        return float(np.mean(values)) if values else 0.0

    def _summary_depths(self, graph: Any, node: Any) -> dict[Any, int]:
        return self._depths(graph, node)

    def _summary_subgraph_nodes(self, graph: Any, node: Any) -> list[Any]:
        return sorted(self._origin_subgraph(graph, node).nodes, key=str)

    def _summary_subgraph_n_nodes(self, graph: Any, node: Any) -> int:
        return self._origin_subgraph(graph, node).number_of_nodes()

    def _summary_subgraph_n_edges(self, graph: Any, node: Any) -> int:
        return self._origin_subgraph(graph, node).number_of_edges()

    def _summary_subgraph_longest_path(self, graph: Any, node: Any) -> list[Any]:
        return self._subgraph_longest_path(self._origin_subgraph(graph, node))

    def _summary_subgraph_longest_path_length(self, graph: Any, node: Any) -> int:
        path = self._summary_subgraph_longest_path(graph, node)
        return max(len(path) - 1, 0)

    def _summary_subgraph_diameter(self, graph: Any, node: Any) -> int:
        subgraph = self._largest_connected_undirected(
            self._origin_subgraph(graph, node)
        )
        if subgraph.number_of_nodes() <= 1:
            return 0
        return int(nx.diameter(subgraph))

    def _summary_subgraph_average_shortest_path(
        self, graph: Any, node: Any
    ) -> float:
        subgraph = self._largest_connected_undirected(
            self._origin_subgraph(graph, node)
        )
        if subgraph.number_of_nodes() <= 1:
            return 0.0
        return float(nx.average_shortest_path_length(subgraph))

    def _summary_subgraph_density(self, graph: Any, node: Any) -> float:
        return float(nx.density(self._origin_subgraph(graph, node)))

    def _summary_subgraph_average_degree(self, graph: Any, node: Any) -> float:
        subgraph = self._origin_subgraph(graph, node)
        n = subgraph.number_of_nodes()
        if n == 0:
            return 0.0
        degree_sum = sum(degree for _, degree in subgraph.degree())
        return float(degree_sum / n)

    def _summary_subgraph_global_efficiency(self, graph: Any, node: Any) -> float:
        subgraph = self._origin_subgraph(graph, node).to_undirected()
        if subgraph.number_of_nodes() <= 1:
            return 0.0
        return float(nx.global_efficiency(subgraph))

    def _summary_parent_of(self, graph: Any, node: Any) -> dict[Any, Any | None]:
        parent_of = {node_id: None for node_id in graph.nodes}
        for parent, child in graph.edges():
            parent_of[child] = parent
        return parent_of

    def _summary_children_of(self, graph: Any, node: Any) -> dict[Any, list[Any]]:
        return {node_id: list(graph.successors(node_id)) for node_id in graph.nodes}

    def _summary_node_attributes(
        self, graph: Any, node: Any
    ) -> dict[Any, dict[str, Any]]:
        return {node_id: dict(graph.nodes[node_id]) for node_id in graph.nodes}

    def _summary_edges(self, graph: Any, node: Any) -> list[dict[str, Any]]:
        return [
            {"parent": parent, "child": child, **edge_data}
            for parent, child, edge_data in graph.edges(data=True)
        ]

    def _summary_node_labels(self, graph: Any, node: Any) -> dict[Any, str]:
        label_attribute = self.config.label_attribute
        if not label_attribute:
            return {}
        return {
            node_id: str(value)
            for node_id in graph.nodes
            if (value := graph.nodes[node_id].get(label_attribute)) is not None
        }

    def _nodes_by_attribute(self, graph: Any, attribute: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node_id in graph.nodes:
            value = graph.nodes[node_id].get(attribute)
            if value is not None:
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _seed_nodes(
        self,
        graph: Any,
        node: Any,
        attr_key: Optional[str],
        attr_value: Any,
    ) -> list[Any]:
        if node is not None:
            if node not in graph:
                raise KeyError(f"node {node!r} not found in graph")
            scope = {node, *nx.descendants(graph, node)}
            nodes_to_search = [
                (node_id, dict(graph.nodes[node_id])) for node_id in scope
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

    def _descendant_counts(
        self, graph: Any, node: Any
    ) -> list[dict[str, Any]]:
        classify_attribute = self.config.filter_attribute
        include = list(self.config.include_attributes)
        nodes = self._seed_nodes(
            graph, node, classify_attribute, self.config.filter_value
        )

        data = []
        for node_id in nodes:
            base_keys = list(set(["object_id", self.config.output_index, *include]))
            row = {key: graph.nodes[node_id].get(key) for key in base_keys}
            for desc in nx.descendants(graph, node_id):
                bucket = graph.nodes[desc].get(classify_attribute, None)
                key = f"count {bucket}"
                row[key] = row.get(key, 0) + 1
            data.append(row)
        return data

    def _ancestor_lineage(
        self, graph: Any, node: Any
    ) -> list[dict[str, Any]]:
        classify_attribute = self.config.filter_attribute
        include = list(self.config.include_attributes)
        value_attribute = self.config.lineage_value_attribute
        nodes = self._seed_nodes(
            graph, node, classify_attribute, self.config.filter_value
        )

        data = []
        for node_id in nodes:
            base_keys = list(set(["object_id", self.config.output_index, *include]))
            row = {key: graph.nodes[node_id].get(key) for key in base_keys}
            ancestors = set(nx.ancestors(graph, node_id))
            for ancestor in nx.topological_sort(graph):
                if ancestor not in ancestors:
                    continue
                group = graph.nodes[ancestor].get(classify_attribute)
                value = graph.nodes[ancestor].get(value_attribute)
                row[f"lineage {group}"] = (
                    int(value) if value is not None else None
                )
            data.append(row)
        return data

    def _neighbor_summary(
        self, graph: Any, node: Any
    ) -> list[dict[str, Any]]:
        del node
        weight_key = self.config.weight_attribute
        group_attr = self.config.group_attribute
        analysis = self.config.neighbor_analysis
        neighbor_k = self.config.neighbor_k
        neighbor_radius = self.config.neighbor_radius
        include = list(self.config.include_attributes)
        data: list[dict[str, Any]] = []
        for node_id in graph.nodes:
            row: dict[str, Any] = {self.config.output_index: node_id}
            if analysis is None:
                for key in set(include):
                    row[key] = graph.nodes[node_id].get(key)

            by_group: dict[Any, list[tuple[float, Any]]] = {}
            for _, child, edge_data in graph.out_edges(node_id, data=True):
                weight = edge_data.get(weight_key)
                if weight is None or pd.isna(weight):
                    continue
                weight = float(weight)
                group_val = graph.nodes[child].get(group_attr, "__none__")
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

    def _to_dataframe(self, output: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(output).set_index(self.config.output_index)