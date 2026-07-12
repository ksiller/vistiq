"""Tests for graph napari layer builders."""

import numpy as np
import pytest

networkx = pytest.importorskip("networkx")
napari = pytest.importorskip("napari")

from vistiq.graph import edges_to_layer, nodes_to_layer


def _sample_graph():
    graph = networkx.DiGraph()
    graph.add_node(
        "a",
        channel="Dpn",
        volume=10,
        **{"centroid-z": 1.0, "centroid-y": 2.0, "centroid-x": 3.0},
    )
    graph.add_node(
        "b",
        channel="Scrib",
        volume=20,
        **{"centroid-z": 4.0, "centroid-y": 5.0, "centroid-x": 6.0},
    )
    graph.add_edge("a", "b", weight=0.75)
    return graph


def test_nodes_to_layer_constant_visuals():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="red",
        node_border_color="white",
        node_opacity=0.8,
        node_size=12,
    )

    assert points.name == "nodes"
    assert points.data.shape == (2, 3)
    np.testing.assert_allclose(points.data[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(points.data[1], [4.0, 5.0, 6.0])
    assert points.opacity == pytest.approx(0.8)
    assert points.size == 12


def test_edges_to_layer_constant_visuals():
    graph = _sample_graph()
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="cyan",
        edge_thickness=2,
        edge_opacity=0.5,
    )

    assert vectors.name == "edges"
    assert vectors.data.shape == (1, 2, 3)
    np.testing.assert_allclose(vectors.data[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vectors.data[0, 1], [3.0, 3.0, 3.0])
    assert vectors.opacity == pytest.approx(0.5)
    assert vectors.edge_width == 2


def test_nodes_to_layer_attribute_visuals():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="channel",
        node_border_color="white",
        node_opacity=1.0,
        node_size="volume",
    )

    assert "channel" in points.properties
    assert list(points.properties["channel"]) == ["Dpn", "Scrib"]
    assert points.face_color == "channel"
    assert points.face_color_mode == "cycle"
    assert points.size == "volume"


def test_edges_to_layer_attribute_visuals():
    graph = _sample_graph()
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="weight",
        edge_thickness="weight",
        edge_opacity=1.0,
    )

    assert "weight" in vectors.properties
    assert vectors.edge_color == "weight"
    assert vectors.edge_width == "weight"


def test_nodes_to_layer_sparse_numeric_size():
    graph = networkx.DiGraph()
    graph.add_node(
        "leaf",
        channel="Dpn",
        volume=10.0,
        **{"centroid-z": 1.0, "centroid-y": 2.0, "centroid-x": 3.0},
    )
    graph.add_node(
        "root",
        channel="Brain",
        synthetic=True,
        **{"centroid-z": 0.0, "centroid-y": 0.0, "centroid-x": 0.0},
    )
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="channel",
        node_border_color="white",
        node_opacity=1.0,
        node_size="volume",
    )
    assert np.issubdtype(points.properties["volume"].dtype, np.floating)
    assert float(np.nanmax(points.properties["volume"])) == 10.0


def test_nodes_to_layer_channel_color_cycle():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="channel",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        node_face_colormap=["red", "blue", "green"],
    )
    assert points.face_color_mode == "cycle"
    assert points.face_color == "channel"
    assert set(points.face_color_cycle.colormap) == {"Dpn", "Scrib"}
    assert not np.allclose(points._face.colors[0], points._face.colors[1])


def test_nodes_to_layer_channel_default_palette():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="channel",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
    )
    assert points.face_color_mode == "cycle"
    assert not np.allclose(points._face.colors[0], points._face.colors[1])


def test_edges_to_layer_continuous_colormap():
    graph = _sample_graph()
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="weight",
        edge_thickness=1,
        edge_opacity=1.0,
        edge_colormap="viridis",
    )
    assert vectors.edge_color_mode == "colormap"
    assert vectors.edge_colormap.name == "viridis"


def test_nodes_to_layer_continuous_colormap():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="volume",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        node_face_colormap="green",
    )
    assert points.face_color_mode == "colormap"
    assert points.face_colormap.name == "green"
    assert points.face_color == "volume"
    assert np.issubdtype(points.properties["volume"].dtype, np.floating)
    assert points.face_contrast_limits == pytest.approx((10.0, 20.0))
    assert not np.allclose(points._face.colors[0], points._face.colors[1])


def test_nodes_to_layer_external_node_properties():
    import pandas as pd

    graph = _sample_graph()
    props = pd.DataFrame({"lineage Lobe": [1, 2]}, index=["a", "b"])
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="lineage Lobe",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        node_face_colormap=["magenta", "green", "red"],
        node_properties=props,
    )
    assert points.face_color_mode == "cycle"
    assert points.face_color == "lineage Lobe"
    assert set(points.face_color_cycle.colormap) == {"1", "2"}
    assert not np.allclose(points._face.colors[0], points._face.colors[1])


def test_nodes_to_layer_missing_property_with_colormap_raises():
    graph = _sample_graph()
    with pytest.raises(ValueError, match="lineage Lobe"):
        nodes_to_layer(
            graph,
            "nodes",
            node_face_color="lineage Lobe",
            node_border_color="white",
            node_opacity=1.0,
            node_size=8,
            node_face_colormap=["magenta", "green"],
        )


def test_layers_not_editable_by_default():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="red",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
    )
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="cyan",
        edge_thickness=1,
        edge_opacity=1.0,
    )
    assert points.editable is False
    assert vectors.editable is False


def test_layers_editable_true():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="red",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        editable=True,
    )
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="cyan",
        edge_thickness=1,
        edge_opacity=1.0,
        editable=True,
    )
    assert points.editable is True
    assert vectors.editable is True


def test_nodes_to_layer_constant_symbol_and_border_width():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="red",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        node_symbol="square",
        node_border_width=0.2,
    )
    assert points.symbol[0].value == "square"
    assert points.symbol[1].value == "square"
    assert float(points.border_width[0]) == pytest.approx(0.2)


def test_nodes_to_layer_channel_symbol_cycle():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="channel",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        node_symbol="channel",
        node_symbol_cycle=["o", "s", "+"],
    )
    assert points.symbol[0].value == "disc"
    assert points.symbol[1].value == "square"
    assert points.symbol[0].value != points.symbol[1].value


def test_layers_blending():
    graph = _sample_graph()
    points = nodes_to_layer(
        graph,
        "nodes",
        node_face_color="red",
        node_border_color="white",
        node_opacity=1.0,
        node_size=8,
        node_blending="additive",
    )
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="cyan",
        edge_thickness=1,
        edge_opacity=1.0,
        edge_blending="minimum",
    )
    assert points.blending == "additive"
    assert vectors.blending == "minimum"


def test_edges_to_layer_vector_style():
    graph = _sample_graph()
    vectors = edges_to_layer(
        graph,
        "edges",
        edge_color="cyan",
        edge_thickness=1,
        edge_opacity=1.0,
        edge_vector_style="arrow",
    )
    assert vectors.vector_style == "arrow"
