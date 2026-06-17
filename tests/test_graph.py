"""Tests for vistiq.graph."""

import pandas as pd
import pytest

networkx = pytest.importorskip("networkx")

from vistiq.graph import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphSummary,
    GraphSummaryConfig,
    NXGraphBuilder,
    NXGraphBuilderConfig,
    NXGraphSummary,
    NXGraphSummaryConfig,
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


class TestNXGraphBuilder:
    def test_smallest_enclosing_parent(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        assert isinstance(dag, networkx.DiGraph)
        assert set(dag.successors("p")) == {"a", "b"}
        assert "c" not in dag.successors("p")
        assert dag.nodes["a"]["volume"] == 10.0
        assert dag.edges["p", "a"]["ios"] == pytest.approx(0.9)

    def test_regions_indexed_by_object_id(self):
        regions = _sample_regions()
        dag = NXGraphBuilder(NXGraphBuilderConfig()).run(
            matrix=_sample_matrix(), regions=regions
        )
        assert dag.number_of_nodes() == 4

    def test_regions_object_id_column(self):
        regions = _sample_regions().reset_index()
        dag = NXGraphBuilder(NXGraphBuilderConfig()).run(
            matrix=_sample_matrix(), regions=regions
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


class TestNXGraphSummary:
    def test_allowed_attributes(self):
        allowed = GraphSummary.allowed_attributes()
        assert "n_nodes" in allowed
        assert "edges" in allowed
        assert set(GraphSummary.default_attributes).issubset(set(allowed))

    def test_summary_dict(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        summary = NXGraphSummary(
            NXGraphSummaryConfig(
                attributes=[
                    *GraphSummary.default_attributes,
                    *GraphSummary.origin_attributes,
                    "nodes_by_channel",
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
        assert summary["nodes_by_channel"]["Brain"] == 1
        assert summary["nodes_by_channel"]["Lobe"] == 2
        assert summary["origin"] == "p"

    def test_summary_requires_node_for_multiple_roots(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        with pytest.raises(ValueError, match="root nodes"):
            NXGraphSummary(
                NXGraphSummaryConfig(attributes=["origin"])
            ).run(dag)

    def test_summary_selective_attributes(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        summary = NXGraphSummary(
            NXGraphSummaryConfig(attributes=["n_nodes", "n_edges", "roots"])
        ).run(dag)
        assert set(summary.keys()) == {"n_nodes", "n_edges", "roots"}
        assert summary["n_nodes"] == 4
        assert set(summary["roots"]) == {"p", "c"}

    def test_summary_from_custom_node(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        summary = NXGraphSummary(
            NXGraphSummaryConfig(
                attributes=[
                    *GraphSummary.default_attributes,
                    *GraphSummary.origin_attributes,
                ]
            )
        ).run(dag, node="p")
        assert summary["origin"] == "p"
        assert summary["n_descendants"] == 3
        assert set(summary["descendants"]) == {"p", "a", "b"}

        orphan_summary = NXGraphSummary(
            NXGraphSummaryConfig(attributes=[*GraphSummary.origin_attributes])
        ).run(dag, node="c")
        assert orphan_summary["origin"] == "c"
        assert orphan_summary["depths"] == {"c": 0}
        assert orphan_summary["max_depth"] == 0
        assert orphan_summary["n_descendants"] == 1

    def test_invalid_attribute_raises(self):
        with pytest.raises(ValueError, match="invalid attributes"):
            NXGraphSummaryConfig(attributes=["not_a_real_key"])

    def test_abstract_summary_raises(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        with pytest.raises(NotImplementedError):
            GraphSummary(GraphSummaryConfig()).run(dag, node="p")

    def test_abstract_builder_raises(self):
        with pytest.raises(NotImplementedError):
            GraphBuilder(GraphBuilderConfig()).run(
                matrix=_sample_matrix(),
                regions=_sample_regions(),
            )


class TestOrphanHandling:
    def _orphan_roots(self, config: NXGraphBuilderConfig) -> set[str]:
        dag = NXGraphBuilder(config).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        return {
            node
            for node in dag.nodes
            if dag.in_degree(node) == 0 and not dag.nodes[node].get("synthetic")
        }

    def test_as_roots_default(self):
        dag = NXGraphBuilder(NXGraphBuilderConfig(threshold=0.5)).run(
            matrix=_sample_matrix(),
            regions=_sample_regions(),
        )
        assert set(dag.nodes) == {"p", "a", "b", "c"}
        assert self._orphan_roots(NXGraphBuilderConfig(threshold=0.5)) == {"p", "c"}

    def test_drop_orphans(self):
        dag = NXGraphBuilder(
            NXGraphBuilderConfig(threshold=0.5, orphan_strategy="drop")
        ).run(matrix=_sample_matrix(), regions=_sample_regions())
        assert set(dag.nodes) == {"p", "a", "b"}
        assert dag.in_degree("p") == 0
        summary = NXGraphSummary(NXGraphSummaryConfig(attributes=["n_roots"])).run(
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
        dag = NXGraphBuilder(
            NXGraphBuilderConfig(threshold=0.5, orphan_strategy="drop")
        ).run(matrix=matrix, regions=regions)
        assert set(dag.nodes) == {"r"}
        summary = NXGraphSummary(NXGraphSummaryConfig(attributes=["n_roots"])).run(
            dag
        )
        assert summary["n_roots"] == 1

    def test_group_orphans_flat(self):
        config = NXGraphBuilderConfig(
            threshold=0.5,
            orphan_strategy="group",
            orphan_attach="separate_root",
        )
        dag = NXGraphBuilder(config).run(
            matrix=_sample_matrix(), regions=_sample_regions()
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
        assert self._orphan_roots(config) == {"p"}

    def test_group_orphans_by_channel(self):
        config = NXGraphBuilderConfig(
            threshold=0.5,
            orphan_strategy="group",
            orphan_groupby="channel",
        )
        dag = NXGraphBuilder(config).run(
            matrix=_sample_matrix(), regions=_sample_regions()
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
        config = NXGraphBuilderConfig(
            threshold=0.5,
            orphan_strategy="group",
            orphan_attach="unify",
        )
        dag = NXGraphBuilder(config).run(
            matrix=_sample_matrix(), regions=_sample_regions()
        )
        all_roots = [node for node in dag.nodes if dag.in_degree(node) == 0]
        assert len(all_roots) == 1
        all_root = all_roots[0]
        assert dag.nodes[all_root]["name"] == "all"
        assert dag.out_degree(all_root) == 2
