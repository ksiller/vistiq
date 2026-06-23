"""Napari layer helpers for graph visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from vistiq.graph.graph import graph_to_dataframe
from vistiq.segment.analysis import dataframe_to_numpy
from vistiq.utils import SpacingLike, abs_spacing

if TYPE_CHECKING:
    from napari.layers import Points, Vectors

VisualSpec = Union[int, float, str]
ColormapLike = Union[Sequence[Union[int, float, str]], str, Any]
SymbolCycleLike = Sequence[str]
ColorKind = Literal["face", "border", "edge"]

_DEFAULT_SYMBOL_CYCLE = ("o", "s", "+", "*", "diamond", "star", "x", "v", "^", ">")


def _node_attr_names(graph: Any) -> set[str]:
    names: set[str] = set()
    for _, attrs in graph.nodes(data=True):
        names.update(attrs)
    return names


def _node_column_names(graph: Any) -> set[str]:
    if not graph.nodes():
        return set()
    return set(graph_to_dataframe(graph).columns)


def _edge_attr_names(graph: Any) -> set[str]:
    names: set[str] = set()
    for _, _, attrs in graph.edges(data=True):
        names.update(attrs)
    return names


def _resolve_visual(
    spec: VisualSpec,
    *,
    attr_names: set[str],
) -> tuple[VisualSpec, Optional[str]]:
    """Return napari kwarg value and optional property column name."""
    if isinstance(spec, str) and spec in attr_names:
        return spec, spec
    return spec, None


def _resolve_numeric_visual(
    spec: VisualSpec,
    *,
    attr_names: set[str],
    param: str,
) -> tuple[VisualSpec, Optional[str]]:
    """Like :func:`_resolve_visual`, but reject non-attribute strings for size/width."""
    if isinstance(spec, str):
        if spec in attr_names:
            return spec, spec
        raise ValueError(
            f"{param} attribute {spec!r} not found on graph; "
            f"available: {sorted(attr_names)}"
        )
    return spec, None


def _attr_specs(*specs: VisualSpec, attr_names: set[str]) -> frozenset[str]:
    return frozenset(spec for spec in specs if isinstance(spec, str) and spec in attr_names)


def _is_color_list(value: ColormapLike) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, str)


def _color_entry(value: Union[int, float, str]) -> Union[str, tuple[float, ...]]:
    if isinstance(value, int):
        import matplotlib.cm as cm

        return cm.tab10(value % 10)
    if isinstance(value, float) and not isinstance(value, bool):
        import matplotlib.cm as cm

        return cm.tab10(int(value) % 10)
    return value


def _normalize_color_cycle(colors: Sequence[Union[int, float, str]]) -> np.ndarray:
    from napari.utils.colormaps.standardize_color import transform_color

    return transform_color([_color_entry(color) for color in colors])


def _default_palette(count: int = 10) -> list[tuple[float, ...]]:
    import matplotlib.cm as cm

    return [cm.tab10(i % 10) for i in range(max(count, 1))]


def _sample_colormap(colormap: ColormapLike, count: int) -> np.ndarray:
    from napari.utils.colormaps import ensure_colormap

    cmap = ensure_colormap(colormap)
    positions = np.linspace(0.0, 1.0, max(count, 1))
    return cmap.map(positions)


def _coerce_categorical(values: np.ndarray) -> np.ndarray:
    out: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out.append("")
        elif isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode())
        else:
            out.append(str(value))
    return np.asarray(out, dtype=str)


def _is_mapped_property(
    color_spec: VisualSpec, properties: dict[str, np.ndarray]
) -> bool:
    return isinstance(color_spec, str) and color_spec in properties


def _property_is_continuous(
    properties: dict[str, np.ndarray], name: str
) -> bool:
    values = np.asarray(properties[name])
    if values.dtype.kind in {"U", "S", "O"}:
        return False
    return np.issubdtype(values.dtype, np.number)


def _categorical_colormap(
    values: np.ndarray,
    colormap: Optional[ColormapLike],
) -> Any:
    from napari.utils.colormaps.categorical_colormap import CategoricalColormap

    normalized = _coerce_categorical(values)
    unique = list(dict.fromkeys(normalized))
    if _is_color_list(colormap):
        palette = _normalize_color_cycle(colormap)
    elif colormap is not None:
        palette = _sample_colormap(colormap, len(unique) or 1)
    else:
        palette = _normalize_color_cycle(_default_palette(max(len(unique), 1)))

    mapping = {
        key: palette[index % len(palette)]
        for index, key in enumerate(unique)
    }
    return CategoricalColormap(colormap=mapping, fallback_color=palette)


def _continuous_colormap_requested(colormap: Optional[ColormapLike]) -> bool:
    return colormap is not None and not _is_color_list(colormap)


def _continuous_colormap_kwargs(
    color_spec: VisualSpec,
    properties: dict[str, np.ndarray],
    colormap: Optional[ColormapLike],
    *,
    prefix: str,
) -> dict[str, Any]:
    if not _continuous_colormap_requested(colormap):
        return {}
    if not _is_mapped_property(color_spec, properties):
        return {}
    if not _property_is_continuous(properties, color_spec):
        return {}
    return {f"{prefix}_colormap": colormap}


def _contrast_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return (vmin, vmax)


def _apply_continuous_color(
    layer: Any,
    color_spec: VisualSpec,
    properties: dict[str, np.ndarray],
    colormap: Optional[ColormapLike],
    *,
    kind: ColorKind,
) -> None:
    if not _continuous_colormap_requested(colormap):
        return
    if not _is_mapped_property(color_spec, properties):
        return
    if not _property_is_continuous(properties, color_spec):
        return

    from napari.utils.colormaps import ensure_colormap

    cmap = ensure_colormap(colormap)
    contrast_limits = _contrast_limits(properties[color_spec])

    if kind == "face":
        layer.face_color = color_spec
        layer.face_colormap = cmap
        layer.face_color_mode = "colormap"
        layer.face_contrast_limits = contrast_limits
    elif kind == "border":
        layer.border_color = color_spec
        layer.border_colormap = cmap
        layer.border_color_mode = "colormap"
        layer.border_contrast_limits = contrast_limits
    elif kind == "edge":
        layer.edge_color = color_spec
        layer.edge_colormap = cmap
        layer.edge_color_mode = "colormap"
        layer.edge_contrast_limits = contrast_limits


def _apply_categorical_color(
    layer: Any,
    color_spec: VisualSpec,
    properties: dict[str, np.ndarray],
    colormap: Optional[ColormapLike],
    *,
    kind: ColorKind,
) -> None:
    if not _is_mapped_property(color_spec, properties):
        return
    if _property_is_continuous(properties, color_spec) and not _is_color_list(colormap):
        return

    values = _coerce_categorical(properties[color_spec])
    properties[color_spec] = values
    cat_cmap = _categorical_colormap(values, colormap)
    layer_props = dict(layer.properties)
    layer_props[color_spec] = values

    if kind == "face":
        layer.properties = layer_props
        layer.face_color = color_spec
        layer.face_color_cycle = cat_cmap
        layer.face_color_mode = "cycle"
    elif kind == "border":
        layer.properties = layer_props
        layer.border_color = color_spec
        layer.border_color_cycle = cat_cmap
        layer.border_color_mode = "cycle"
    elif kind == "edge":
        layer.properties = layer_props
        layer.edge_color = color_spec
        layer.edge_color_cycle = cat_cmap
        layer.edge_color_mode = "cycle"


def _normalize_symbol_cycle(cycle: Optional[SymbolCycleLike]) -> np.ndarray:
    from napari.layers.points._points_utils import coerce_symbols

    symbols = list(cycle) if cycle is not None else list(_DEFAULT_SYMBOL_CYCLE)
    return coerce_symbols(symbols)


def _categorical_symbols(
    values: np.ndarray,
    cycle: Optional[SymbolCycleLike],
) -> np.ndarray:
    normalized = _coerce_categorical(values)
    palette = _normalize_symbol_cycle(cycle)
    unique = list(dict.fromkeys(normalized))
    mapping = {
        key: palette[index % len(palette)]
        for index, key in enumerate(unique)
    }
    return np.array([mapping[value] for value in normalized], dtype=object)


def _apply_categorical_symbol(
    layer: Any,
    symbol_spec: VisualSpec,
    properties: dict[str, np.ndarray],
    symbol_cycle: Optional[SymbolCycleLike],
) -> None:
    if not _is_mapped_property(symbol_spec, properties):
        return
    if _property_is_continuous(properties, symbol_spec):
        return

    values = _coerce_categorical(properties[symbol_spec])
    properties[symbol_spec] = values
    layer_props = dict(layer.properties)
    layer_props[symbol_spec] = values
    layer.properties = layer_props
    layer.symbol = _categorical_symbols(values, symbol_cycle)


def _validate_mapped_symbol(
    symbol_spec: Optional[VisualSpec],
    attr_names: set[str],
    symbol_cycle: Optional[SymbolCycleLike],
    *,
    param: str,
) -> None:
    if symbol_spec is None or not isinstance(symbol_spec, str):
        return
    if symbol_spec in attr_names:
        return
    if symbol_cycle is None:
        return
    raise ValueError(
        f"{param} {symbol_spec!r} not found on graph or node_properties; "
        f"available: {sorted(attr_names)}"
    )


def _align_properties_frame(
    properties: pd.DataFrame,
    nodes: list[Any],
    *,
    index: str = "object_id",
) -> pd.DataFrame:
    """Reindex an external property table to match graph node order."""
    frame = properties
    if isinstance(frame.index, pd.MultiIndex):
        level_names = list(frame.index.names)
        if index in level_names:
            frame = frame.reset_index(level=[n for n in level_names if n != index])
        else:
            frame = frame.reset_index()
    if index in frame.columns and frame.index.name != index:
        frame = frame.set_index(index)
    return frame.reindex(nodes)


def _merge_property_frames(
    graph_frame: pd.DataFrame,
    extra: Optional[pd.DataFrame],
    nodes: list[Any],
) -> pd.DataFrame:
    if extra is None or extra.empty:
        return graph_frame
    aligned = _align_properties_frame(extra, nodes)
    frame = graph_frame.copy()
    for column in aligned.columns:
        if column not in frame.columns:
            frame[column] = aligned[column]
    return frame


def _collect_node_properties(
    graph: Any,
    nodes: list[Any],
    specs: Sequence[VisualSpec],
    *,
    numeric_specs: frozenset[str],
    properties_df: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, np.ndarray], set[str]]:
    frame = _merge_property_frames(
        graph_to_dataframe(graph).reindex(nodes),
        properties_df,
        nodes,
    )
    attr_names = _node_attr_names(graph) | set(frame.columns)

    property_names: set[str] = set()
    for spec in specs:
        _, name = _resolve_visual(spec, attr_names=attr_names)
        if name is not None:
            property_names.add(name)

    if not property_names:
        return {}, attr_names

    properties: dict[str, np.ndarray] = {}
    for name in property_names:
        if name in frame.columns:
            series = frame[name]
            if name in numeric_specs:
                properties[name] = pd.to_numeric(series, errors="coerce").to_numpy(
                    dtype=float
                )
            else:
                properties[name] = _coerce_categorical(series.to_numpy())
        else:
            properties[name] = _property_values(
                lambda node: graph.nodes[node],
                nodes,
                name,
                numeric=name in numeric_specs,
            )
            if name not in numeric_specs:
                properties[name] = _coerce_categorical(properties[name])
    return properties, attr_names


def _property_values(
    getter,
    items: list[Any],
    name: str,
    *,
    numeric: bool,
) -> np.ndarray:
    if numeric:
        values = [getter(item).get(name, np.nan) for item in items]
        return np.asarray(values, dtype=float)
    return np.asarray([getter(item).get(name) for item in items])


def _collect_edge_properties(
    graph: Any,
    edges: list[Any],
    specs: Sequence[VisualSpec],
    *,
    numeric_specs: frozenset[str],
) -> tuple[dict[str, np.ndarray], set[str]]:
    attr_names = _edge_attr_names(graph)
    getter = lambda item: item[2]

    property_names: set[str] = set()
    for spec in specs:
        _, name = _resolve_visual(spec, attr_names=attr_names)
        if name is not None:
            property_names.add(name)

    if not property_names:
        return {}, attr_names

    properties: dict[str, np.ndarray] = {}
    for name in property_names:
        values = _property_values(getter, edges, name, numeric=name in numeric_specs)
        if name in numeric_specs:
            properties[name] = values
        else:
            properties[name] = _coerce_categorical(values)
    return properties, attr_names


def _validate_mapped_color(
    color_spec: VisualSpec,
    attr_names: set[str],
    colormap: Optional[ColormapLike],
    *,
    param: str,
) -> None:
    if not isinstance(color_spec, str):
        return
    if color_spec in attr_names:
        return
    if colormap is None or not _is_color_list(colormap):
        return
    raise ValueError(
        f"{param} {color_spec!r} not found on graph or node_properties; "
        f"available: {sorted(attr_names)}"
    )


def _node_positions(
    graph: Any,
    *,
    scale: SpacingLike = None,
    axes: Optional[Sequence[str]] = None,
) -> tuple[list[Any], np.ndarray]:
    """Node order and centroid coordinates, optionally rescaled."""
    nodes = list(graph.nodes())
    if not nodes:
        return nodes, np.empty((0, 3), dtype=float)

    regions = graph_to_dataframe(graph)
    coords = dataframe_to_numpy(
        regions,
        attributes=["centroid"],
        strict=False,
        axes=axes,
        reset_index=False,
    )
    if coords is None:
        raise KeyError(
            "graph nodes are missing centroid coordinates; "
            f"available columns: {list(regions.columns)}"
        )
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    spacing = abs_spacing(scale)
    if spacing is not None:
        factors = np.ones(coords.shape[1], dtype=float)
        if len(spacing) <= coords.shape[1]:
            factors[-len(spacing) :] = spacing
        else:
            factors = np.asarray(spacing[: coords.shape[1]], dtype=float)
        coords = coords * factors
    return nodes, coords


def _edge_vectors(
    nodes: list[Any],
    positions: np.ndarray,
    graph: Any,
) -> np.ndarray:
    """Build napari vector data as (N, 2, D) start + displacement segments."""
    if positions.size == 0:
        ndim = 3
    else:
        ndim = positions.shape[1]
    index = {node_id: i for i, node_id in enumerate(nodes)}
    segments: list[np.ndarray] = []
    for source, target in graph.edges():
        if source not in index or target not in index:
            continue
        start = positions[index[source]]
        end = positions[index[target]]
        segments.append(np.stack([start, end - start]))
    if not segments:
        return np.empty((0, 2, ndim), dtype=float)
    return np.stack(segments, axis=0)


def graph_to_layers(
    graph: Any,
    node_face_color: VisualSpec,
    node_border_color: VisualSpec,
    node_opacity: float,
    node_size: VisualSpec,
    edge_color: VisualSpec,
    edge_thickness: VisualSpec,
    edge_opacity: float,
    scale: SpacingLike = None,
    *,
    axes: Optional[Sequence[str]] = None,
    node_face_colormap: Optional[ColormapLike] = None,
    node_border_colormap: Optional[ColormapLike] = None,
    edge_colormap: Optional[ColormapLike] = None,
    node_properties: Optional[pd.DataFrame] = None,
    node_symbol: Optional[VisualSpec] = None,
    node_symbol_cycle: Optional[SymbolCycleLike] = None,
    node_border_width: Optional[VisualSpec] = None,
    node_blending: Optional[str] = None,
    edge_blending: Optional[str] = None,
    edge_vector_style: Optional[str] = None,
    editable: Optional[bool] = False,
) -> tuple[Points, Vectors]:
    """Build napari Points and Vectors layers from a graph.

    Node positions come from each node's ``centroid`` attribute (including
    mapped columns such as ``centroid-z``, ``centroid-y``, ``centroid-x``).
    When *scale* is provided, coordinates are multiplied axis-wise by
    ``abs(spacing)``.

    Visual parameters may be literals (``int`` / ``float``) applied uniformly,
    or ``str`` names of node/edge attributes. Unrecognized string values are
    passed through as napari color names.

    When a visual parameter references a property, optional ``*_colormap``
    arguments control coloring:

    - A **list** of ints, floats, or color strings is used as a categorical
      color cycle and rotated across unique property values.
    - A **named or napari colormap** maps continuous numeric properties.
      For categorical properties, the colormap is sampled into a cycle.

    *node_symbol* may be a napari marker name (for example ``"o"`` or
    ``"square"``) applied to all nodes, or a node attribute name. When an
    attribute is used, *node_symbol_cycle* assigns markers from the list to
    each unique property value (defaults to a built-in cycle when omitted).

    *node_border_width* sets the point border width for all nodes or maps
    from a numeric node attribute.

    *node_blending* and *edge_blending* set napari layer blending modes
    (for example ``"translucent"``, ``"additive"``, ``"opaque"``). When
    omitted, napari's default (``"translucent"``) is used.

    *edge_vector_style* sets the edge rendering style: ``"line"``,
    ``"triangle"``, or ``"arrow"``. When omitted, napari's default
    (``"triangle"``) is used.

    *node_properties* may supply columns that are not stored on graph nodes
    (for example ``lineage Lobe`` from hierarchical analysis). The frame is
    aligned on ``object_id`` and may use a MultiIndex whose first level is
    ``object_id``.

    When *editable* is not ``None``, it is applied to both returned layers
    to control whether napari allows moving or editing them in the viewer.
    Defaults to ``False``.

    Returns:
        ``(points_layer, vectors_layer)`` for graph nodes and edges.
    """
    try:
        from napari.layers import Points, Vectors
    except ImportError as exc:
        raise ImportError(
            "graph_to_layers requires napari; install with "
            "`pip install vistiq[napari]`"
        ) from exc

    nodes, positions = _node_positions(graph, scale=scale, axes=axes)
    node_field_names = _node_attr_names(graph) | _node_column_names(graph)
    if node_properties is not None:
        node_field_names |= set(node_properties.columns)
    node_visual_specs: list[VisualSpec] = [
        node_face_color,
        node_border_color,
        node_size,
    ]
    node_numeric_specs = _attr_specs(node_size, attr_names=node_field_names)
    if _continuous_colormap_requested(node_face_colormap):
        node_numeric_specs = node_numeric_specs | _attr_specs(
            node_face_color, attr_names=node_field_names
        )
    if _continuous_colormap_requested(node_border_colormap):
        node_numeric_specs = node_numeric_specs | _attr_specs(
            node_border_color, attr_names=node_field_names
        )
    if node_symbol is not None:
        node_visual_specs.append(node_symbol)
    if node_border_width is not None:
        node_visual_specs.append(node_border_width)
        node_numeric_specs = node_numeric_specs | _attr_specs(
            node_border_width, attr_names=node_field_names
        )
    node_props, node_attr_names = _collect_node_properties(
        graph,
        nodes,
        node_visual_specs,
        numeric_specs=node_numeric_specs,
        properties_df=node_properties,
    )
    _validate_mapped_color(
        node_face_color,
        node_attr_names,
        node_face_colormap,
        param="node_face_color",
    )
    _validate_mapped_color(
        node_border_color,
        node_attr_names,
        node_border_colormap,
        param="node_border_color",
    )
    _validate_mapped_symbol(
        node_symbol,
        node_attr_names,
        node_symbol_cycle,
        param="node_symbol",
    )
    face_color, _ = _resolve_visual(node_face_color, attr_names=node_attr_names)
    border_color, _ = _resolve_visual(node_border_color, attr_names=node_attr_names)
    size, _ = _resolve_numeric_visual(
        node_size, attr_names=node_attr_names, param="node_size"
    )
    symbol_kw: Optional[VisualSpec] = None
    if node_symbol is not None:
        symbol_kw, _ = _resolve_visual(node_symbol, attr_names=node_attr_names)
    border_width_kw: Optional[VisualSpec] = None
    if node_border_width is not None:
        border_width_kw, _ = _resolve_numeric_visual(
            node_border_width,
            attr_names=node_attr_names,
            param="node_border_width",
        )

    edges = list(graph.edges(data=True))
    vectors = _edge_vectors(nodes, positions, graph)
    edge_properties, edge_attr_names = _collect_edge_properties(
        graph,
        edges,
        (edge_color, edge_thickness),
        numeric_specs=_attr_specs(
            edge_thickness,
            edge_color,
            attr_names=_edge_attr_names(graph),
        ),
    )
    edge_color_kw, _ = _resolve_visual(edge_color, attr_names=edge_attr_names)
    edge_width_kw, _ = _resolve_numeric_visual(
        edge_thickness, attr_names=edge_attr_names, param="edge_thickness"
    )

    points_kwargs: dict[str, Any] = {
        "properties": node_props or None,
        "face_color": face_color,
        "border_color": border_color,
        "size": size,
        "opacity": node_opacity,
        **_continuous_colormap_kwargs(
            face_color, node_props, node_face_colormap, prefix="face"
        ),
        **_continuous_colormap_kwargs(
            border_color, node_props, node_border_colormap, prefix="border"
        ),
    }
    if symbol_kw is not None and not (
        isinstance(symbol_kw, str) and symbol_kw in node_props
    ):
        points_kwargs["symbol"] = symbol_kw
    if border_width_kw is not None:
        points_kwargs["border_width"] = border_width_kw
    if node_blending is not None:
        points_kwargs["blending"] = node_blending

    points = Points(positions, **points_kwargs)
    _apply_continuous_color(
        points,
        face_color,
        node_props,
        node_face_colormap,
        kind="face",
    )
    _apply_continuous_color(
        points,
        border_color,
        node_props,
        node_border_colormap,
        kind="border",
    )
    _apply_categorical_color(
        points,
        face_color,
        node_props,
        node_face_colormap,
        kind="face",
    )
    _apply_categorical_color(
        points,
        border_color,
        node_props,
        node_border_colormap,
        kind="border",
    )
    if node_symbol is not None:
        _apply_categorical_symbol(
            points,
            node_symbol,
            node_props,
            node_symbol_cycle,
        )

    vectors_kwargs: dict[str, Any] = {
        "properties": edge_properties or None,
        "edge_color": edge_color_kw,
        "edge_width": edge_width_kw,
        "opacity": edge_opacity,
        **_continuous_colormap_kwargs(
            edge_color_kw, edge_properties, edge_colormap, prefix="edge"
        ),
    }
    if edge_blending is not None:
        vectors_kwargs["blending"] = edge_blending
    if edge_vector_style is not None:
        vectors_kwargs["vector_style"] = edge_vector_style

    vectors_layer = Vectors(vectors, **vectors_kwargs)
    _apply_continuous_color(
        vectors_layer,
        edge_color_kw,
        edge_properties,
        edge_colormap,
        kind="edge",
    )
    _apply_categorical_color(
        vectors_layer,
        edge_color_kw,
        edge_properties,
        edge_colormap,
        kind="edge",
    )
    if editable is not None:
        points.editable = editable
        vectors_layer.editable = editable
    return points, vectors_layer
