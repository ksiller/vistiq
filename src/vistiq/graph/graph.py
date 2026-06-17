"""Build containment graphs from overlap weight matrices and region metrics.

GraphBuilder turns a labeled overlap matrix plus region properties into a
containment graph (directed or undirected). GraphSummary collects structural
statistics from that graph into a plain dictionary.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, ClassVar, List, Literal, Optional

import networkx as nx
import numpy as np
import pandas as pd
from prefect import task
from pydantic import Field, field_validator

from vistiq.analysis.matrix import _square_dataframe
from vistiq.core import Configurable, Configuration, generate_name

logger = logging.getLogger(__name__)

ORPHAN_GROUP_UNKNOWN = "unknown"


def parents_from_graph(graph, attr_key:str=None, attr_value:Any=None, index="object_id", include=[]):
    """Get the parents of selected nodes in a graph.

    Args:
        graph: Networkx graph.
        attr_key: Attribute key to filter nodes by.
        attr_value: Attribute value to filter nodes by.
        index: Index column name.
        include: List of additional column names to include.

    Returns:
        DataFrame with the parents of the selected nodes.
    """
    if attr_key is not None and attr_value is not None:
        nodes = [n for n, attr in graph.nodes(data=True) if attr.get(attr_key) == attr_value]
    else:
        nodes = [n for n, attr in graph.nodes(data=True)]
    data = []
    for node in nodes:
        base_keys = list(set(["object_id"] + include))
        parents = {k:graph.nodes[node].get(k) for k in base_keys}

        ancestors = list(nx.ancestors(graph, node))
        sorted_ancestors = sorted_ancestors = [n for n in nx.topological_sort(graph) if n in ancestors]
        for ancestor in sorted_ancestors:
            pkey = graph.nodes[ancestor].get(attr_key)
            parents[f"parent {pkey}"] = int(graph.nodes[ancestor].get("label"))
        data.append(parents)
    return pd.DataFrame(data).set_index(index)


def graph_count(graph, attr_key:str=None, attr_value:Any=None, index="object_id", include=[]):
    """Count the number of counts for a given attribute in the descendants of selected nodes in a graph.

    Args:
        graph: Networkx graph.
        attr_key: Attribute key to filter nodes by.
        attr_value: Attribute value to filter nodes by.
        index: Index column name.
        include: List of additional column names to include.

    Returns:
        DataFrame with the count of the given attribute in the descendants of the selected nodes.
    """
    if attr_key is not None and attr_value is not None:
        nodes = [n for n, attr in graph.nodes(data=True) if attr.get(attr_key) == attr_value]
    else:
        nodes = [n for n, attr in graph.nodes(data=True)]
    data = []
    for node in nodes:
        base_keys = list(set(["object_id"] + include))
        count = {k:graph.nodes[node].get(k) for k in base_keys}
        for desc in nx.descendants(graph, node):
            ch = graph.nodes[desc].get(attr_key, None)
            key = f"count {ch}"
            if key in count:
                count[key]=count[key]+1
            else:
                count[key] = 1
        data.append(count)
    return pd.DataFrame(data).set_index(index)


def _pairwise_weight(matrix: pd.DataFrame, left: Any, right: Any) -> float:
    """Return the symmetric matrix weight between left and right."""
    values: list[float] = []
    if left in matrix.index and right in matrix.columns:
        values.append(float(matrix.loc[left, right]))
    if right in matrix.index and left in matrix.columns:
        values.append(float(matrix.loc[right, left]))
    if not values:
        return float("nan")
    return float(np.nanmax(values))


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
        rank_attribute: Region column used to rank objects when assigning
            parents (for example volume or area).
        threshold: Minimum matrix weight required to link a child to a parent.
        parent_strategy: How to choose among eligible parents.
            smallest_enclosing selects the smallest enclosing object;
            max_weight selects the highest matrix weight.
        weight_attribute: Edge attribute key used when storing the matrix
            weight on containment edges (for example ios).
        graph_type: Whether to build a directed or undirected graph.
        orphan_strategy: How to handle objects with no assigned parent.
            drop removes orphans and any objects assigned beneath them.
        orphan_groupby: Region column for grouping orphans when
            orphan_strategy is group. None attaches orphans directly to
            orphan_node. Orphans missing this attribute are grouped under
            orphan subgroup "unknown".
        orphan_node: Node attributes for the synthetic orphan root created
            when orphan_strategy is group. object_id is assigned at build
            time; do not set it in config.
        orphan_attach: How grouped orphans connect to the main hierarchy.
            separate_root keeps orphan_node as a second top-level root;
            unify creates an all_node above both the containment primary
            root and orphan_node.
        all_node: Node attributes for the synthetic unify root when
            orphan_attach is unify. object_id is assigned at build time.
    """

    rank_attribute: str = "volume"
    threshold: float = 0.5
    graph_type: Literal["directed", "undirected"] = "directed"
    parent_strategy: Literal["smallest_enclosing", "max_weight"] = "smallest_enclosing"
    weight_attribute: str = "ios"
    orphan_strategy: Literal["drop", "as_roots", "group"] = "as_roots"
    orphan_groupby: Optional[str] = None
    orphan_attach: Literal["separate_root", "unify"] = "separate_root"
    orphan_node: dict[str, Any] = Field(
        default_factory=lambda: {"name": "Orphans", "synthetic": True}
    )
    all_node: dict[str, Any] = Field(
        default_factory=lambda: {"name": "all", "synthetic": True}
    )


class GraphBuilder(Configurable[GraphBuilderConfig]):
    """Build a containment graph from an overlap matrix and region metrics.

    Parent assignment is always hierarchical. The output graph type is
    controlled by config.graph_type (directed or undirected). Subclasses
    implement backend hooks (_add_node, _create_edge, _new_*_graph).
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
        """Build a containment graph from overlap weights and region metrics.

        Each non-root node is assigned a single parent among larger objects with
        matrix weight at or above the configured threshold. The primary root is
        the highest-ranked object by rank_attribute. Orphan handling follows
        config.orphan_strategy.

        Args:
            matrix: Labeled square overlap matrix (e.g. from MatrixCombiner).
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

    def _create_node(
        self,
        graph: Any,
        attributes: dict[str, Any],
        *,
        node_id: Optional[Any] = None,
    ) -> Any:
        if node_id is None:
            attrs = {
                key: value
                for key, value in attributes.items()
                if key != "object_id"
            }
            node_id = uuid.uuid4().hex
            attrs["object_id"] = node_id
        else:
            attrs = dict(attributes)
            if "object_id" not in attrs:
                attrs["object_id"] = node_id
        self._add_node(graph, node_id, attrs)
        return node_id

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

    def _assign_parents(
        self,
        weight_matrix: pd.DataFrame,
        region_table: pd.DataFrame,
    ) -> tuple[dict[Any, Optional[Any]], Any]:
        rank_attribute = self.config.rank_attribute
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
                parent
                for parent in ordered
                if float(ranks[parent]) > child_rank
            ]
            candidates = [
                parent
                for parent in candidates
                if _pairwise_weight(weight_matrix, parent, child) >= self.config.threshold
            ]
            if not candidates:
                parents[child] = None
                continue
            if self.config.parent_strategy == "max_weight":
                parent = max(
                    candidates,
                    key=lambda candidate: _pairwise_weight(
                        weight_matrix, candidate, child
                    ),
                )
            else:
                parent = min(
                    candidates, key=lambda candidate: float(ranks[candidate])
                )
            parents[child] = parent

        return parents, primary_root

    def _orphan_group_key(
        self, orphan: Any, region_table: pd.DataFrame, groupby: str
    ) -> str:
        if groupby not in region_table.columns:
            return ORPHAN_GROUP_UNKNOWN
        value = region_table.loc[orphan, groupby]
        if pd.isna(value):
            return ORPHAN_GROUP_UNKNOWN
        return str(value)

    def _drop_orphan_subtrees(
        self,
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

    def _add_orphans(
        self,
        graph: Any,
        *,
        orphans: list[Any],
        primary_root: Any,
        parents: dict[Any, Optional[Any]],
        region_table: pd.DataFrame,
    ) -> None:
        if not orphans:
            return

        orphan_id = self._create_node(graph, self.config.orphan_node)
        parents[orphan_id] = None

        if self.config.orphan_groupby is None:
            for orphan in orphans:
                parents[orphan] = orphan_id
                self._create_edge(graph, orphan_id, orphan)
        else:
            groups: dict[str, list[Any]] = {}
            for orphan in orphans:
                key = self._orphan_group_key(
                    orphan, region_table, self.config.orphan_groupby
                )
                groups.setdefault(key, []).append(orphan)

            for group_key, group_orphans in groups.items():
                group_id = self._create_node(
                    graph,
                    {
                        "name": f"orphans:{group_key}",
                        "synthetic": True,
                        "orphan_group": group_key,
                    },
                )
                self._create_edge(graph, orphan_id, group_id)
                for orphan in group_orphans:
                    parents[orphan] = group_id
                    self._create_edge(graph, group_id, orphan)

        if self.config.orphan_attach == "unify":
            all_id = self._create_node(graph, self.config.all_node)
            parents[all_id] = None
            parents[orphan_id] = all_id
            parents[primary_root] = all_id
            self._create_edge(graph, all_id, orphan_id)
            self._create_edge(graph, all_id, primary_root)

    def _build(
        self,
        matrix: pd.DataFrame,
        regions: pd.DataFrame,
    ) -> Any:
        weight_matrix = _square_dataframe(matrix)
        nodes = list(weight_matrix.index)
        region_table = _regions_for_nodes(regions, nodes)
        parents, primary_root = self._assign_parents(weight_matrix, region_table)

        orphans = [
            node
            for node in nodes
            if parents.get(node) is None and node != primary_root
        ]

        graph = self._new_graph()

        if self.config.orphan_strategy == "drop":
            keep_nodes = self._drop_orphan_subtrees(nodes, parents, orphans)
        else:
            keep_nodes = set(nodes)
            if self.config.orphan_strategy == "group":
                self._add_orphans(
                    graph,
                    orphans=orphans,
                    primary_root=primary_root,
                    parents=parents,
                    region_table=region_table,
                )

        for node in keep_nodes:
            self._create_node(
                graph, region_table.loc[node].to_dict(), node_id=node
            )

        for child, parent in parents.items():
            if parent is None or child not in keep_nodes or parent not in keep_nodes:
                continue
            weight = _pairwise_weight(weight_matrix, parent, child)
            self._create_edge(
                graph,
                parent,
                child,
                attributes={self.config.weight_attribute: weight},
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


class GraphSummary(Configurable["GraphSummaryConfig"]):
    """Summarize a directed acyclic graph such as a containment DAG.

    Summary keys are registered in _ATTRIBUTE_METHODS and computed by
    matching _summary_* methods on concrete subclasses. Use
    allowed_attributes() to see every key that may be requested in config;
    default_attributes lists keys computed when config.attributes is left
    empty. origin_attributes lists depth and descendant keys that require
    run(node=...) when the graph has more than one root.
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
        "descendants",
        "n_descendants",
        "max_depth",
        "mean_depth",
        "depths",
    )

    _ATTRIBUTE_METHODS: ClassVar[dict[str, str]] = {
        "n_nodes": "_summary_n_nodes",
        "n_edges": "_summary_n_edges",
        "n_roots": "_summary_n_roots",
        "n_leaves": "_summary_n_leaves",
        "origin": "_summary_origin",
        "roots": "_summary_roots",
        "leaves": "_summary_leaves",
        "descendants": "_summary_descendants",
        "n_descendants": "_summary_n_descendants",
        "max_depth": "_summary_max_depth",
        "mean_depth": "_summary_mean_depth",
        "depths": "_summary_depths",
        "parent_of": "_summary_parent_of",
        "children_of": "_summary_children_of",
        "node_attributes": "_summary_node_attributes",
        "edges": "_summary_edges",
        "node_labels": "_summary_node_labels",
        "nodes_by_channel": "_summary_nodes_by_channel",
    }

    @classmethod
    def allowed_attributes(cls) -> List[str]:
        """Return summary keys that may be listed in GraphSummaryConfig.attributes."""
        return list(cls._attribute_methods().keys())

    @classmethod
    def _attribute_methods(cls) -> dict[str, str]:
        return {**cls._ATTRIBUTE_METHODS, **cls._register_attribute_methods()}

    @classmethod
    def _register_attribute_methods(cls) -> dict[str, str]:
        """Return extra attribute-to-method entries for subclasses."""
        return {}

    @classmethod
    def from_config(cls, config: "GraphSummaryConfig") -> "GraphSummary":
        return cls(config)

    @task(name="GraphSummary.run", task_run_name=generate_name)
    def run(self, graph: Any, *, node: Any = None) -> dict[str, Any]:
        """Summarize graph and return a plain dictionary of graph statistics.

        Args:
            graph: Backend-specific graph (e.g. networkx DiGraph or Graph).
            node: Origin for depth and descendant statistics. Required when
                config.attributes includes any origin_attributes and the
                graph has more than one root (in-degree zero). When omitted
                and the graph has exactly one root, that root is used.

        Returns:
            Dictionary whose keys are the configured attributes.
        """
        logger.info(f"Summarizing graph with config: {self.config}")
        logger.info(f"Node: {node}")
        attributes = self.config.attributes
        return self._summarize(graph, node=node, attributes=attributes)

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

    def _summary_descendants(self, graph: Any, node: Any) -> list[Any]:
        raise NotImplementedError("Subclasses must implement _summary_descendants")

    def _summary_n_descendants(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_n_descendants")

    def _summary_max_depth(self, graph: Any, node: Any) -> int:
        raise NotImplementedError("Subclasses must implement _summary_max_depth")

    def _summary_mean_depth(self, graph: Any, node: Any) -> float:
        raise NotImplementedError("Subclasses must implement _summary_mean_depth")

    def _summary_depths(self, graph: Any, node: Any) -> dict[Any, int]:
        raise NotImplementedError("Subclasses must implement _summary_depths")

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

    def _summary_nodes_by_channel(self, graph: Any, node: Any) -> dict[str, int]:
        raise NotImplementedError("Subclasses must implement _summary_nodes_by_channel")


class GraphSummaryConfig(Configuration):
    """Configuration for GraphSummary.

    Attributes:
        label_attribute: Node attribute read when node_labels is requested.
            Set to None to skip label collection.
        attributes: Summary keys to compute. Each name must appear in
            GraphSummary.allowed_attributes(). When omitted, defaults to
            GraphSummary.default_attributes.
    """

    label_attribute: Optional[str] = "object_name"
    attributes: List[str] = Field(
        default_factory=lambda: list(GraphSummary.default_attributes)
    )

    @field_validator("attributes")
    @classmethod
    def validate_attributes(cls, value: List[str]) -> List[str]:
        allowed = set(GraphSummary.allowed_attributes())
        invalid = [name for name in value if name not in allowed]
        if invalid:
            raise ValueError(
                f"One or more invalid attributes: {invalid}. "
                f"Use names from {GraphSummary.allowed_attributes()}."
            )
        return list(dict.fromkeys(value))


class NXGraphSummaryConfig(GraphSummaryConfig):
    """Configuration for NXGraphSummary."""


class NXGraphSummary(GraphSummary):
    """GraphSummary implementation for networkx.DiGraph."""

    @classmethod
    def from_config(cls, config: NXGraphSummaryConfig) -> "NXGraphSummary":
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
        roots = NXGraphSummary._roots(graph)
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
        import networkx as nx

        origin = NXGraphSummary._resolve_origin(graph, node)
        if origin is None:
            return {}
        return dict(nx.single_source_shortest_path_length(graph, origin))

    def _summarize(
        self,
        graph: Any,
        *,
        node: Any,
        attributes: list[str],
    ) -> dict[str, Any]:
        import networkx as nx

        if not isinstance(graph, nx.DiGraph):
            raise TypeError(
                f"NXGraphSummary expects a networkx.DiGraph; got {type(graph).__name__}"
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

    def _summary_descendants(self, graph: Any, node: Any) -> list[Any]:
        return list(self._depths(graph, node).keys())

    def _summary_n_descendants(self, graph: Any, node: Any) -> int:
        return len(self._depths(graph, node))

    def _summary_max_depth(self, graph: Any, node: Any) -> int:
        values = list(self._depths(graph, node).values())
        return max(values) if values else 0

    def _summary_mean_depth(self, graph: Any, node: Any) -> float:
        values = list(self._depths(graph, node).values())
        return float(np.mean(values)) if values else 0.0

    def _summary_depths(self, graph: Any, node: Any) -> dict[Any, int]:
        return self._depths(graph, node)

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

    def _summary_nodes_by_channel(self, graph: Any, node: Any) -> dict[str, int]:
        channels: dict[str, int] = {}
        for node_id in graph.nodes:
            channel = graph.nodes[node_id].get("channel")
            if channel is not None:
                key = str(channel)
                channels[key] = channels.get(key, 0) + 1
        return channels
