"""Tests for vistiq.graph."""

import pandas as pd
import pytest

networkx = pytest.importorskip("networkx")

from vistiq.graph import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphFormatter,
    GraphFormatterConfig,
    GraphFilter,
    GraphFilterConfig,
    GraphQuery,
    GraphQueryConfig,
    HierarchyBuilder,
    HierarchyBuilderConfig,
    NXGraph,
    edges_to_matrix,
    graph_to_dataframe,
)


def _sample_matrix() -> pd.DataFrame:
    # parent p contains children a and b; c is separate
    return pd.DataFrame(
        [
            [0.0, 0.9, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        index=["p", "a", "b", "c"],
        columns=["p", "a", "b", "c"],
        dtype=float,
    )


def _sample_regions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_id": ["p", "a", "b", "c"],
            "volume": [100.0, 10.0, 20.0, 5.0],
            "channel": ["Brain", "Lobe", "Lobe", "EdU"],
        }
    ).set_index("object_id")


def _build_containment_dag(
    matrix: pd.DataFrame,
    regions: pd.DataFrame,
    **hierarchy_kwargs,
):
    hb_cfg = HierarchyBuilderConfig(**hierarchy_kwargs)
    return HierarchyBuilder(hb_cfg).run(matrix, regions).graph


class TestGraphBuilder:
    def test_graph_to_dataframe(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        frame = graph_to_dataframe(dag)
        assert frame.index.name == "object_id"
        assert set(frame.index) == set(dag.nodes())
        assert frame.loc["a", "volume"] == 10.0
        assert frame.loc["a", "channel"] == "Lobe"

    def test_graph_formatter_dropna_rows(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        for node in ["p", "a", "b", "c"]:
            dag.add_node(node, synthetic=False, name=str(node))
        dag.add_node(
            "synthetic",
            name="Orphans",
            synthetic=True,
        )
        frame = GraphFormatter(GraphFormatterConfig(dropna_rows=True)).run(dag)
        assert "synthetic" not in frame.index
        assert set(frame.index) == {"p", "a", "b", "c"}

    def test_graph_formatter_dropna_cols(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        dag.add_node("a", **{**dag.node_attrs("a"), "unused": float("nan")})
        frame = GraphFormatter(GraphFormatterConfig(dropna_cols=True)).run(dag)
        assert "unused" not in frame.columns

    def test_graph_formatter_exclude_synthetic(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        dag.add_node(
            "synthetic",
            name="Orphans",
            synthetic=True,
        )
        frame = GraphFormatter(GraphFormatterConfig(exclude_synthetic=True)).run(dag)
        assert "synthetic" not in frame.index
        assert set(frame.index) == {"p", "a", "b", "c"}

    def test_smallest_enclosing_parent(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        assert isinstance(dag, NXGraph)
        assert set(dag.successors("p")) == {"a", "b"}
        assert "c" not in dag.successors("p")
        assert dag.node_attrs("a")["volume"] == 10.0
        assert dag.edge_attrs("p", "a")["ios"] == pytest.approx(0.9)
        assert dag.edge_attrs("p", "a").get("synthetic") is not True

    def test_nodes_indexed_by_object_id(self):
        regions = _sample_regions()
        result = HierarchyBuilder(HierarchyBuilderConfig()).run(
            _sample_matrix(), regions
        )
        assert result.graph.number_of_nodes() == 4

    def test_nodes_object_id_column(self):
        regions = _sample_regions().reset_index()
        result = HierarchyBuilder(HierarchyBuilderConfig()).run(
            _sample_matrix(), regions
        )
        assert result.graph.number_of_nodes() == 4

    def test_requires_matrix_and_nodes(self):
        with pytest.raises(TypeError):
            GraphBuilder(GraphBuilderConfig()).run(
                matrix=_sample_matrix(),
            )
        with pytest.raises(TypeError):
            GraphBuilder(GraphBuilderConfig()).run(
                nodes=_sample_regions(),
            )


class TestGraphQuery:
    def test_allowed_attributes(self):
        allowed = GraphQuery.allowed_attributes()
        assert "n_nodes" in allowed
        assert "edges" in allowed
        assert "filtered_edges" in allowed
        assert "first_matching_predecessor" in allowed
        assert "matching_predecessor_edges" in allowed
        assert set(GraphQuery.default_attributes).issubset(set(allowed))

    def test_filtered_edges(self):
        dag = networkx.DiGraph()
        dag.add_node("d1", channel="Dpn")
        dag.add_node("d2", channel="Dpn")
        dag.add_node("e1", channel="EdU")
        dag.add_node("e2", channel="EdU")
        dag.add_edge("d1", "e1", distance=1.0)
        dag.add_edge("d1", "e2", distance=2.0)
        dag.add_edge("d2", "e1", distance=3.0)
        dag = NXGraph(dag)

        all_edges = GraphQuery(
            GraphQueryConfig(attributes=["filtered_edges"])
        ).run(dag)["filtered_edges"]
        assert len(all_edges) == 3

        partner_edges = GraphQuery(
            GraphQueryConfig(
                attributes=["filtered_edges"],
                source_nodes=["d1"],
                target_filter={"channel": "EdU"},
            )
        ).run(dag)["filtered_edges"]
        assert len(partner_edges) == 2
        assert {e["target"] for e in partner_edges} == {"e1", "e2"}
        assert all(e["source"] == "d1" for e in partner_edges)

        runtime_edges = GraphQuery(
            GraphQueryConfig(
                attributes=["filtered_edges"],
                target_filter={"channel": "EdU"},
            )
        ).run(dag, source_nodes=["d2"])["filtered_edges"]
        assert len(runtime_edges) == 1
        assert runtime_edges[0]["source"] == "d2"
        assert runtime_edges[0]["target"] == "e1"

    def test_first_matching_predecessor(self):
        dag = networkx.DiGraph()
        dag.add_node("lobe1", channel="Lobe", label=1)
        dag.add_node("lobe2", channel="Lobe", label=2)
        dag.add_node("d1", channel="Dpn")
        dag.add_node("d2", channel="Dpn")
        dag.add_edge("lobe1", "d1")
        dag.add_edge("lobe2", "d2")
        dag = NXGraph(dag)

        predecessors = GraphQuery(
            GraphQueryConfig(
                attributes=["first_matching_predecessor"],
                predecessor_match={"channel": "Lobe"},
                seed_nodes=["d1", "d2"],
            )
        ).run(dag)["first_matching_predecessor"]
        assert predecessors == {"d1": "lobe1", "d2": "lobe2"}

        runtime = GraphQuery(
            GraphQueryConfig(
                attributes=["first_matching_predecessor"],
                predecessor_match={"channel": "Lobe"},
            )
        ).run(dag, seed_nodes=["d1"])["first_matching_predecessor"]
        assert runtime == {"d1": "lobe1"}

        edges = GraphQuery(
            GraphQueryConfig(
                attributes=["matching_predecessor_edges"],
                predecessor_match={"channel": "Lobe"},
                seed_nodes=["d1", "d2"],
            )
        ).run(dag)["matching_predecessor_edges"]
        assert len(edges) == 2
        assert {e["source"] for e in edges} == {"lobe1", "lobe2"}
        assert {e["target"] for e in edges} == {"d1", "d2"}

    def test_summary_dict(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        summary = GraphQuery(
            GraphQueryConfig(
                attributes=[
                    *GraphQuery.default_attributes,
                    *GraphQuery.origin_attributes,
                    "nodes_by_attribute",
                ]
            )
        ).run(dag, node="p")
        assert isinstance(summary, dict)
        assert summary["n_nodes"] == 4
        assert summary["n_edges"] == 2
        assert summary["n_roots"] == 2
        assert set(summary["roots"]) == {"p", "c"}
        assert set(summary["leaves"]) == {"a", "b", "c"}
        assert summary["max_depth"] == 1
        assert summary["depths"]["a"] == 1
        assert summary["parent_of"]["a"] == "p"
        assert set(summary["children_of"]["p"]) == {"a", "b"}
        assert len(summary["edges"]) == 2
        ios_values = sorted(edge["ios"] for edge in summary["edges"])
        assert ios_values == pytest.approx([0.8, 0.9])
        assert summary["nodes_by_attribute"]["Brain"] == 1
        assert summary["nodes_by_attribute"]["Lobe"] == 2
        assert summary["origin"] == "p"

    def test_summary_requires_node_for_multiple_roots(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        with pytest.raises(ValueError, match="root nodes"):
            GraphQuery(
                GraphQueryConfig(attributes=["origin"])
            ).run(dag)

    def test_summary_selective_attributes(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        summary = GraphQuery(
            GraphQueryConfig(attributes=["n_nodes", "n_edges", "roots"])
        ).run(dag)
        assert set(summary.keys()) == {"n_nodes", "n_edges", "roots"}
        assert summary["n_nodes"] == 4
        assert set(summary["roots"]) == {"p", "c"}

    def test_summary_from_custom_node(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        summary = GraphQuery(
            GraphQueryConfig(
                attributes=[
                    *GraphQuery.default_attributes,
                    *GraphQuery.origin_attributes,
                ]
            )
        ).run(dag, node="p")
        assert summary["origin"] == "p"
        assert summary["subgraph_n_nodes"] == 3
        assert set(summary["subgraph_nodes"]) == {"p", "a", "b"}

        orphan_summary = GraphQuery(
            GraphQueryConfig(attributes=[*GraphQuery.origin_attributes])
        ).run(dag, node="c")
        assert orphan_summary["origin"] == "c"
        assert orphan_summary["depths"] == {"c": 0}
        assert orphan_summary["max_depth"] == 0
        assert orphan_summary["subgraph_n_nodes"] == 1

    def test_origin_subgraph_metrics(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        metrics = GraphQuery(
            GraphQueryConfig(
                attributes=[
                    "subgraph_n_nodes",
                    "subgraph_n_edges",
                    "subgraph_longest_path",
                    "subgraph_longest_path_length",
                    "subgraph_diameter",
                    "subgraph_average_shortest_path",
                    "subgraph_density",
                    "subgraph_average_degree",
                    "subgraph_global_efficiency",
                ]
            )
        ).run(dag, node="p")
        assert metrics["subgraph_n_nodes"] == 3
        assert metrics["subgraph_n_edges"] == 2
        assert metrics["subgraph_longest_path"][0] == "p"
        assert metrics["subgraph_longest_path_length"] == 1
        assert metrics["subgraph_diameter"] == 2
        assert metrics["subgraph_average_shortest_path"] == pytest.approx(4 / 3)
        assert metrics["subgraph_density"] == pytest.approx(1 / 3)
        assert metrics["subgraph_average_degree"] == pytest.approx(4 / 3)
        assert 0.0 < metrics["subgraph_global_efficiency"] <= 1.0

    def test_descendant_counts_and_ancestor_lineage(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        labels = {"p": 1, "a": 2, "b": 3, "c": 4}
        for node_id, label in labels.items():
            dag.add_node(node_id, **{**dag.node_attrs(node_id), "label": label})

        scoped = GraphQuery(
            GraphQueryConfig(
                attributes=["descendant_counts"],
                filter_attribute="channel",
                filter_value="Brain",
            )
        ).run(dag, node="p")
        assert scoped["descendant_counts"][0]["count Lobe"] == 2

        lineage = GraphQuery(
            GraphQueryConfig(
                attributes=["ancestor_lineage"],
                filter_attribute="channel",
                filter_value="Lobe",
            )
        ).run(dag)["ancestor_lineage"]
        assert len(lineage) == 2
        assert lineage[0]["lineage Brain"] == 1

        df = GraphQuery(
            GraphQueryConfig(
                attributes=["ancestor_lineage"],
                filter_attribute="channel",
                filter_value="Lobe",
                include_attributes=["volume"],
                output_type="dataframe",
            )
        ).format(lineage)
        assert len(df) == 2
        assert "lineage Brain" in df.columns

    def test_format_empty_rows_returns_empty_dataframe(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        gq = GraphQuery(
            GraphQueryConfig(
                attributes=["descendant_counts"],
                filter_attribute="channel",
                filter_value="Dpn",
                output_type="dataframe",
            )
        )
        result = gq.run(dag)
        assert result["descendant_counts"] == []
        frame = gq.format(result, attribute="descendant_counts")
        assert isinstance(frame, pd.DataFrame)
        assert frame.empty
        assert gq.format([]).empty

    def test_invalid_attribute_raises(self):
        with pytest.raises(ValueError, match="invalid attributes"):
            GraphQueryConfig(attributes=["not_a_real_key"])



class TestGraphFilter:
    def _hierarchy_dag(self):
        dag = networkx.DiGraph()
        dag.add_node("lobe1", channel="Lobe", label=1)
        dag.add_node("lobe2", channel="Lobe", label=2)
        dag.add_node("d1", channel="Dpn")
        dag.add_node("d2", channel="Dpn")
        dag.add_node("e1", channel="EdU")
        dag.add_edge("lobe1", "d1")
        dag.add_edge("lobe2", "d2")
        dag.add_edge("d1", "e1", distance=2.5)
        return NXGraph(dag)

    def test_nodes_by_attribute(self):
        dag = self._hierarchy_dag()
        nodes = GraphFilter(
            GraphFilterConfig(mode="nodes", node_match={"channel": "Dpn"})
        ).run(dag)
        assert set(nodes) == {"d1", "d2"}

    def test_edges_endpoints(self):
        dag = self._hierarchy_dag()
        edges = GraphFilter(
            GraphFilterConfig(
                mode="edges",
                node_match={
                    "source": ["d1"],
                    "target": {"channel": "EdU"},
                },
            )
        ).run(dag)
        assert len(edges) == 1
        assert edges[0]["source"] == "d1"
        assert edges[0]["target"] == "e1"

    def test_edges_incident_list(self):
        dag = self._hierarchy_dag()
        edges = GraphFilter(
            GraphFilterConfig(mode="edges", node_match=["d1"])
        ).run(dag)
        assert len(edges) == 2
        assert {"source": "d1", "target": "e1", "distance": 2.5} in edges
        assert {"source": "lobe1", "target": "d1"} in edges

    def test_direct_path(self):
        dag = self._hierarchy_dag()
        edges = GraphFilter(
            GraphFilterConfig(
                mode="direct_path",
                node_match={
                    "source": {"channel": "Lobe"},
                    "target": ["d1", "d2"],
                },
            )
        ).run(dag)
        assert {e["source"] for e in edges} == {"lobe1", "lobe2"}
        assert {e["target"] for e in edges} == {"d1", "d2"}

    def test_full_path(self):
        dag = self._hierarchy_dag()
        edges = GraphFilter(
            GraphFilterConfig(
                mode="full_path",
                node_match={
                    "source": {"channel": "Lobe"},
                    "target": ["d1"],
                },
            )
        ).run(dag)
        assert len(edges) == 1
        assert edges[0] == {"source": "lobe1", "target": "d1"}


class TestEdgesToMatrix:
    def test_weighted_and_default_weight(self):
        dag = networkx.DiGraph()
        dag.add_node("lobe1", channel="Lobe", label=1)
        dag.add_node("d1", channel="Dpn")
        dag.add_node("e1", channel="EdU")
        dag.add_edge("lobe1", "d1")
        dag.add_edge("d1", "e1", ios=2.5)
        dag = NXGraph(dag)

        partner_edges = GraphQuery(
            GraphQueryConfig(
                attributes=["filtered_edges"],
                target_filter={"channel": "EdU"},
            )
        ).run(dag)["filtered_edges"]
        partner_matrix = edges_to_matrix(partner_edges)
        assert partner_matrix.loc["d1", "e1"] == pytest.approx(2.5)

        predecessor_edges = GraphQuery(
            GraphQueryConfig(
                attributes=["matching_predecessor_edges"],
                predecessor_match={"channel": "Lobe"},
                seed_nodes=["d1"],
            )
        ).run(dag)["matching_predecessor_edges"]
        predecessor_matrix = edges_to_matrix(predecessor_edges)
        assert list(predecessor_matrix.index) == ["d1", "lobe1"]
        assert predecessor_matrix.loc["lobe1", "d1"] == pytest.approx(1.0)

        built = GraphBuilder(GraphBuilderConfig()).run(
            predecessor_matrix,
            graph_to_dataframe(dag).loc[predecessor_matrix.index],
        )
        assert list(built.edges()) == [("lobe1", "d1")]

    def test_empty_records(self):
        assert edges_to_matrix([]).empty

    def test_custom_endpoints_and_weight(self):
        records = [{"parent": "a", "child": "b", "ios": 0.5}]
        matrix = edges_to_matrix(records, endpoints=("parent", "child"))
        assert matrix.loc["a", "b"] == pytest.approx(0.5)


class TestOrphanHandling:
    def _orphan_roots(self, **hierarchical_kwargs) -> set[str]:
        dag = _build_containment_dag(
            _sample_matrix(), _sample_regions(), **hierarchical_kwargs
        )
        return {
            node
            for node in dag.nodes()
            if dag.in_degree(node) == 0 and not dag.node_attrs(node).get("synthetic")
        }

    def test_as_roots_default(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        assert set(dag.nodes()) == {"p", "a", "b", "c"}
        assert self._orphan_roots(threshold=0.5) == {"p", "c"}

    def test_drop_orphans(self):
        dag = _build_containment_dag(
            _sample_matrix(), _sample_regions(), threshold=0.5, orphan_strategy="drop"
        )
        assert set(dag.nodes()) == {"p", "a", "b"}
        assert dag.in_degree("p") == 0
        summary = GraphQuery(GraphQueryConfig(attributes=["n_roots"])).run(
            dag
        )
        assert summary["n_roots"] == 1

    def test_drop_orphans_skips_edges_to_dropped_parent(self):
        # orphan o is parent of c; drop removes the whole orphan subtree
        matrix = pd.DataFrame(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 0.0],
            ],
            index=["r", "o", "c"],
            columns=["r", "o", "c"],
            dtype=float,
        )
        regions = pd.DataFrame(
            {"object_id": ["r", "o", "c"], "volume": [100.0, 50.0, 5.0]}
        ).set_index("object_id")
        dag = _build_containment_dag(
            matrix, regions, threshold=0.5, orphan_strategy="drop"
        )
        assert set(dag.nodes()) == {"r"}
        summary = GraphQuery(GraphQueryConfig(attributes=["n_roots"])).run(
            dag
        )
        assert summary["n_roots"] == 1

    def test_group_orphans_flat(self):
        dag = _build_containment_dag(
            _sample_matrix(),
            _sample_regions(),
            threshold=0.5,
            orphan_strategy="group",
            orphan_attach="separate_root",
        )
        synthetic_roots = [
            node
            for node in dag.nodes()
            if dag.in_degree(node) == 0 and dag.node_attrs(node).get("synthetic")
        ]
        assert len(synthetic_roots) == 1
        orphan_root = synthetic_roots[0]
        assert dag.node_attrs(orphan_root)["name"] == "Orphans"
        assert dag.has_edge(orphan_root, "c")
        assert dag.edge_attrs(orphan_root, "c")["synthetic"] is True
        assert dag.edge_attrs("p", "a").get("synthetic") is not True
        assert self._orphan_roots(
            threshold=0.5,
            orphan_strategy="group",
            orphan_attach="separate_root",
        ) == {"p"}

    def test_group_orphans_by_channel(self):
        dag = _build_containment_dag(
            _sample_matrix(),
            _sample_regions(),
            threshold=0.5,
            orphan_strategy="group",
            orphan_groupby="channel",
        )
        group_nodes = [
            node
            for node in dag.nodes()
            if dag.node_attrs(node).get("orphan_group") == "EdU"
        ]
        assert len(group_nodes) == 1
        group_id = group_nodes[0]
        assert dag.has_edge(group_id, "c")
        assert dag.edge_attrs(group_id, "c")["synthetic"] is True

    def test_unify_attach(self):
        dag = _build_containment_dag(
            _sample_matrix(),
            _sample_regions(),
            threshold=0.5,
            orphan_strategy="group",
            orphan_attach="unify",
        )
        all_roots = [node for node in dag.nodes() if dag.in_degree(node) == 0]
        assert len(all_roots) == 1
        all_root = all_roots[0]
        assert dag.node_attrs(all_root)["name"] == "all"
        assert dag.out_degree(all_root) == 2
        for _parent, _child in dag.out_edges(all_root):
            assert dag.edge_attrs(_parent, _child)["synthetic"] is True
