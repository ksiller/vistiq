"""Tests for vistiq.graph."""

import pandas as pd
import pytest

networkx = pytest.importorskip("networkx")

from vistiq.analysis.matrix import HierarchicalMatrix, HierarchicalMatrixConfig
from vistiq.graph import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphExporter,
    GraphExporterConfig,
    GraphQuery,
    GraphQueryConfig,
    NXGraphBuilder,
    NXGraphBuilderConfig,
    NXGraphQuery,
    NXGraphQueryConfig,
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
    **hierarchical_kwargs,
):
    hm_cfg = HierarchicalMatrixConfig(**hierarchical_kwargs)
    result = HierarchicalMatrix(hm_cfg).run(matrix, regions)
    return NXGraphBuilder(NXGraphBuilderConfig()).run(result.matrix, result.regions)


class TestNXGraphBuilder:
    def test_graph_to_dataframe(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        frame = graph_to_dataframe(dag)
        assert frame.index.name == "object_id"
        assert set(frame.index) == set(dag.nodes)
        assert frame.loc["a", "volume"] == 10.0
        assert frame.loc["a", "channel"] == "Lobe"

    def test_graph_exporter_dropna_rows(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        for node in ["p", "a", "b", "c"]:
            dag.nodes[node]["synthetic"] = False
            dag.nodes[node]["name"] = str(node)
        dag.add_node(
            "synthetic",
            name="Orphans",
            synthetic=True,
        )
        frame = GraphExporter(GraphExporterConfig(dropna_rows=True)).run(dag)
        assert "synthetic" not in frame.index
        assert set(frame.index) == {"p", "a", "b", "c"}

    def test_graph_exporter_dropna_cols(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        dag.nodes["a"]["unused"] = float("nan")
        frame = GraphExporter(GraphExporterConfig(dropna_cols=True)).run(dag)
        assert "unused" not in frame.columns

    def test_graph_exporter_exclude_synthetic(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        dag.add_node(
            "synthetic",
            name="Orphans",
            synthetic=True,
        )
        frame = GraphExporter(GraphExporterConfig(exclude_synthetic=True)).run(dag)
        assert "synthetic" not in frame.index
        assert set(frame.index) == {"p", "a", "b", "c"}

    def test_smallest_enclosing_parent(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        assert isinstance(dag, networkx.DiGraph)
        assert set(dag.successors("p")) == {"a", "b"}
        assert "c" not in dag.successors("p")
        assert dag.nodes["a"]["volume"] == 10.0
        assert dag.edges["p", "a"]["ios"] == pytest.approx(0.9)

    def test_regions_indexed_by_object_id(self):
        regions = _sample_regions()
        result = HierarchicalMatrix(HierarchicalMatrixConfig()).run(
            _sample_matrix(), regions
        )
        dag = NXGraphBuilder(NXGraphBuilderConfig()).run(
            result.matrix, result.regions
        )
        assert dag.number_of_nodes() == 4

    def test_regions_object_id_column(self):
        regions = _sample_regions().reset_index()
        result = HierarchicalMatrix(HierarchicalMatrixConfig()).run(
            _sample_matrix(), regions
        )
        dag = NXGraphBuilder(NXGraphBuilderConfig()).run(
            result.matrix, result.regions
        )
        assert dag.number_of_nodes() == 4

    def test_requires_matrix_and_regions(self):
        with pytest.raises(TypeError):
            NXGraphBuilder(NXGraphBuilderConfig()).run(
                matrix=_sample_matrix(),
            )
        with pytest.raises(TypeError):
            NXGraphBuilder(NXGraphBuilderConfig()).run(
                regions=_sample_regions(),
            )


class TestNXGraphQuery:
    def test_allowed_attributes(self):
        allowed = GraphQuery.allowed_attributes()
        assert "n_nodes" in allowed
        assert "edges" in allowed
        assert set(GraphQuery.default_attributes).issubset(set(allowed))

    def test_summary_dict(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        summary = NXGraphQuery(
            NXGraphQueryConfig(
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
            NXGraphQuery(
                NXGraphQueryConfig(attributes=["origin"])
            ).run(dag)

    def test_summary_selective_attributes(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        summary = NXGraphQuery(
            NXGraphQueryConfig(attributes=["n_nodes", "n_edges", "roots"])
        ).run(dag)
        assert set(summary.keys()) == {"n_nodes", "n_edges", "roots"}
        assert summary["n_nodes"] == 4
        assert set(summary["roots"]) == {"p", "c"}

    def test_summary_from_custom_node(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        summary = NXGraphQuery(
            NXGraphQueryConfig(
                attributes=[
                    *GraphQuery.default_attributes,
                    *GraphQuery.origin_attributes,
                ]
            )
        ).run(dag, node="p")
        assert summary["origin"] == "p"
        assert summary["subgraph_n_nodes"] == 3
        assert set(summary["subgraph_nodes"]) == {"p", "a", "b"}

        orphan_summary = NXGraphQuery(
            NXGraphQueryConfig(attributes=[*GraphQuery.origin_attributes])
        ).run(dag, node="c")
        assert orphan_summary["origin"] == "c"
        assert orphan_summary["depths"] == {"c": 0}
        assert orphan_summary["max_depth"] == 0
        assert orphan_summary["subgraph_n_nodes"] == 1

    def test_origin_subgraph_metrics(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        metrics = NXGraphQuery(
            NXGraphQueryConfig(
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
            dag.nodes[node_id]["label"] = label

        scoped = NXGraphQuery(
            NXGraphQueryConfig(
                attributes=["descendant_counts"],
                filter_attribute="channel",
                filter_value="Brain",
            )
        ).run(dag, node="p")
        assert scoped["descendant_counts"][0]["count Lobe"] == 2

        lineage = NXGraphQuery(
            NXGraphQueryConfig(
                attributes=["ancestor_lineage"],
                filter_attribute="channel",
                filter_value="Lobe",
            )
        ).run(dag)["ancestor_lineage"]
        assert len(lineage) == 2
        assert lineage[0]["lineage Brain"] == 1

        df = NXGraphQuery(
            NXGraphQueryConfig(
                attributes=["ancestor_lineage"],
                filter_attribute="channel",
                filter_value="Lobe",
                include_attributes=["volume"],
                output_type="dataframe",
            )
        ).format(lineage)
        assert len(df) == 2
        assert "lineage Brain" in df.columns

    def test_invalid_attribute_raises(self):
        with pytest.raises(ValueError, match="invalid attributes"):
            NXGraphQueryConfig(attributes=["not_a_real_key"])

    def test_abstract_summary_raises(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        with pytest.raises(NotImplementedError):
            GraphQuery(GraphQueryConfig()).run(dag, node="p")

    def test_abstract_builder_raises(self):
        with pytest.raises(NotImplementedError):
            GraphBuilder(GraphBuilderConfig()).run(
                matrix=_sample_matrix(),
                regions=_sample_regions(),
            )


class TestOrphanHandling:
    def _orphan_roots(self, **hierarchical_kwargs) -> set[str]:
        dag = _build_containment_dag(
            _sample_matrix(), _sample_regions(), **hierarchical_kwargs
        )
        return {
            node
            for node in dag.nodes
            if dag.in_degree(node) == 0 and not dag.nodes[node].get("synthetic")
        }

    def test_as_roots_default(self):
        dag = _build_containment_dag(_sample_matrix(), _sample_regions())
        assert set(dag.nodes) == {"p", "a", "b", "c"}
        assert self._orphan_roots(threshold=0.5) == {"p", "c"}

    def test_drop_orphans(self):
        dag = _build_containment_dag(
            _sample_matrix(), _sample_regions(), threshold=0.5, orphan_strategy="drop"
        )
        assert set(dag.nodes) == {"p", "a", "b"}
        assert dag.in_degree("p") == 0
        summary = NXGraphQuery(NXGraphQueryConfig(attributes=["n_roots"])).run(
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
        assert set(dag.nodes) == {"r"}
        summary = NXGraphQuery(NXGraphQueryConfig(attributes=["n_roots"])).run(
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
            for node in dag.nodes
            if dag.in_degree(node) == 0 and dag.nodes[node].get("synthetic")
        ]
        assert len(synthetic_roots) == 1
        orphan_root = synthetic_roots[0]
        assert dag.nodes[orphan_root]["name"] == "Orphans"
        assert dag.has_edge(orphan_root, "c")
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
            for node in dag.nodes
            if dag.nodes[node].get("orphan_group") == "EdU"
        ]
        assert len(group_nodes) == 1
        group_id = group_nodes[0]
        assert dag.has_edge(group_id, "c")

    def test_unify_attach(self):
        dag = _build_containment_dag(
            _sample_matrix(),
            _sample_regions(),
            threshold=0.5,
            orphan_strategy="group",
            orphan_attach="unify",
        )
        all_roots = [node for node in dag.nodes if dag.in_degree(node) == 0]
        assert len(all_roots) == 1
        all_root = all_roots[0]
        assert dag.nodes[all_root]["name"] == "all"
        assert dag.out_degree(all_root) == 2
