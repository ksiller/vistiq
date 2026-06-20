"""Tests for vistiq.analysis.spatial."""

import pandas as pd
import pytest

networkx = pytest.importorskip("networkx")

from vistiq.analysis.spatial import (
    KnnAnalysis,
    KnnAnalysisConfig,
    RnnAnalysis,
    RnnAnalysisConfig,
    SpatialScopeConfig,
)
from vistiq.graph import (
    NXGraphQuery,
    NXGraphQueryConfig,
    resolve_subtree_origins,
)


def _sample_regions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "centroid-0": [0.0, 0.0, 10.0, 10.0],
            "centroid-1": [0.0, 3.0, 0.0, 3.0],
            "centroid-2": [0.0, 0.0, 0.0, 0.0],
            "channel": ["A", "A", "B", "B"],
            "label": [1, 2, 1, 2],
        }
    ).set_index("object_id")


def _sample_containment_graph() -> networkx.DiGraph:
    graph = networkx.DiGraph()
    for object_id, row in _sample_regions().iterrows():
        attrs = row.to_dict()
        attrs["object_id"] = object_id
        graph.add_node(object_id, **attrs)
    return graph


def _hierarchy_graph() -> networkx.DiGraph:
    graph = networkx.DiGraph()
    for object_id, row in _sample_regions().iterrows():
        attrs = row.to_dict()
        attrs["object_id"] = object_id
        graph.add_node(object_id, **attrs)
    graph.add_node("root", channel="A", synthetic=True, object_id="root")
    graph.add_edges_from([("root", "a"), ("root", "b"), ("root", "c"), ("root", "d")])
    return graph


def _branched_hierarchy_graph() -> networkx.DiGraph:
    graph = networkx.DiGraph()
    left = _sample_regions().loc[["a", "b"]]
    right = _sample_regions().loc[["c", "d"]]
    for part, scope_tag in ((left, "left"), (right, "right")):
        root = f"{scope_tag}_root"
        graph.add_node(
            root,
            synthetic=True,
            scope=scope_tag,
            tissue=scope_tag,
            object_id=root,
        )
        for object_id, row in part.iterrows():
            attrs = row.to_dict()
            attrs["object_id"] = object_id
            graph.add_node(object_id, **attrs)
            graph.add_edge(root, object_id)
    return graph


def _neighbor_summary(graph, *, analysis="knn", k=1, radius=None):
    gq = NXGraphQuery(
        NXGraphQueryConfig(
            attributes=["neighbor_summary"],
            weight_attribute="distance",
            neighbor_analysis=analysis,
            neighbor_k=k if analysis == "knn" else None,
            neighbor_radius=radius if analysis == "rnn" else None,
            output_type="dataframe",
        )
    )
    return gq.format(gq.run(graph), attribute="neighbor_summary")


class TestResolveSubtreeOrigins:
    def test_match_dict_single(self):
        graph = _branched_hierarchy_graph()
        origins = resolve_subtree_origins(graph, match={"scope": "left"})
        assert origins == ["left_root"]

    def test_match_dict_multiple(self):
        graph = _branched_hierarchy_graph()
        origins = resolve_subtree_origins(graph, match={"synthetic": True})
        assert origins == ["left_root", "right_root"]

    def test_exclude_takes_precedence_over_match(self):
        graph = _branched_hierarchy_graph()
        origins = resolve_subtree_origins(
            graph,
            match={"synthetic": True},
            exclude={"scope": "right"},
        )
        assert origins == ["left_root"]


class TestKnnAnalysis:
    def test_homotypic_knn_pipeline(self):
        result = KnnAnalysis(
            KnnAnalysisConfig(k=1, mode="homotypic", grouping_attribute="channel")
        ).run(_sample_containment_graph())

        assert result.distance_matrix.shape == (4, 4)
        assert result.matrix.shape == (4, 4)
        assert isinstance(result.graph, networkx.DiGraph)
        assert result.graph.edges["a", "b"]["distance"] == pytest.approx(3.0)
        assert ("a", "c") not in result.graph.edges

        summary = _neighbor_summary(result.graph, k=1)
        assert summary.loc["a", "knn_nearest_neighbor_distance_A_(k=1)"] == pytest.approx(
            3.0
        )

    def test_heterotypic_knn_pipeline(self):
        result = KnnAnalysis(
            KnnAnalysisConfig(k=1, mode="heterotypic", grouping_attribute="channel")
        ).run(_sample_containment_graph())

        assert ("a", "b") not in result.graph.edges
        assert ("a", "c") in result.graph.edges
        assert result.graph.edges["a", "c"]["distance"] == pytest.approx(10.0)

        summary = _neighbor_summary(result.graph, k=1)
        assert summary.loc["a", "knn_count_B_(k=1)"] == 1
        assert summary.loc["a", "knn_nearest_neighbor_distance_B_(k=1)"] == (
            pytest.approx(10.0)
        )

    def test_excludes_synthetic_nodes(self):
        graph = _hierarchy_graph()
        result = KnnAnalysis(KnnAnalysisConfig(k=1, mode="global")).run(graph)
        assert "root" not in result.distance_matrix.index

    def test_accepts_precomputed_dist_matrix(self):
        baseline = KnnAnalysis(KnnAnalysisConfig(k=1, mode="global")).run(
            _sample_containment_graph()
        )
        result = KnnAnalysis(KnnAnalysisConfig(k=1, mode="global")).run(
            _sample_containment_graph(),
            distance_matrix=baseline.distance_matrix.to_numpy(),
        )
        assert result.distance_matrix.equals(baseline.distance_matrix)
        assert result.graph.edges["a", "b"]["distance"] == pytest.approx(3.0)

    def test_subtree_scope_by_match_dict(self):
        graph = _branched_hierarchy_graph()
        result = KnnAnalysis(
            KnnAnalysisConfig(
                k=1,
                mode="global",
                scope=SpatialScopeConfig(match={"scope": "left"}),
            )
        ).run(graph)
        assert set(result.distance_matrix.index) == {"a", "b"}

    def test_multiple_scope_matches_require_explicit_node(self):
        graph = _branched_hierarchy_graph()
        with pytest.raises(ValueError, match="matched 2 subtree roots"):
            KnnAnalysis(
                KnnAnalysisConfig(
                    scope=SpatialScopeConfig(match={"synthetic": True}),
                )
            ).run(graph)

    def test_explicit_node_none_ignores_config_scope(self):
        graph = _branched_hierarchy_graph()
        result = KnnAnalysis(
            KnnAnalysisConfig(
                k=1,
                mode="global",
                scope=SpatialScopeConfig(match={"synthetic": True}),
            )
        ).run(graph, node=None)
        assert set(result.distance_matrix.index) == {"a", "b", "c", "d"}

    def test_explicit_node_id_ignores_config_scope(self):
        graph = _branched_hierarchy_graph()
        result = KnnAnalysis(
            KnnAnalysisConfig(
                k=1,
                mode="global",
                scope=SpatialScopeConfig(match={"synthetic": True}),
            )
        ).run(graph, node="left_root")
        assert set(result.distance_matrix.index) == {"a", "b"}


class TestRnnAnalysis:
    def test_radius_neighbor_pipeline(self):
        result = RnnAnalysis(
            RnnAnalysisConfig(radius=5.0, mode="global")
        ).run(_sample_containment_graph())

        assert ("a", "b") in result.graph.edges
        assert ("a", "c") not in result.graph.edges
        summary = _neighbor_summary(result.graph, analysis="rnn", radius=5.0)
        assert summary.loc["a", "rnn_count_A_(radius=5)"] == 1

    def test_homotypic_radius_pipeline(self):
        result = RnnAnalysis(
            RnnAnalysisConfig(
                radius=5.0,
                mode="homotypic",
                grouping_attribute="channel",
            )
        ).run(_sample_containment_graph())

        assert ("a", "b") in result.graph.edges
        assert ("a", "c") not in result.graph.edges

    def test_reuses_distance_matrix(self):
        knn_result = KnnAnalysis(KnnAnalysisConfig()).run(_sample_containment_graph())
        dist = knn_result.distance_matrix
        result = RnnAnalysis(RnnAnalysisConfig(radius=5.0, mode="global")).run(
            _sample_containment_graph(),
            distance_matrix=knn_result.distance_matrix,
        )
        assert result.distance_matrix.equals(dist)
