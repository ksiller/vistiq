"""Tests for vistiq.analysis.workflow."""

import numpy as np
import pandas as pd
import pytest

from vistiq.analysis.overlap import (
    LabelOverlapCalculatorConfig,
    metrics_calculator_configs,
)
from vistiq.analysis.workflow import AnalysisFlow, AnalysisFlowConfig
from vistiq.graph import GraphBuilderConfig, GraphQueryConfig, HierarchyBuilderConfig
from vistiq.matrix.ops import (
    MatrixAggregatorConfig,
    MatrixCombinerConfig,
)
from vistiq.matrix.types import UPPER
from vistiq.analysis.spatial import KnnAnalysisConfig
from vistiq.segment import ValueFilterConfig
from vistiq.segment.analysis import RegionAnalyzerConfig
from vistiq.utils import ArrayIteratorConfig as UtilsArrayIteratorConfig


@pytest.fixture
def overlapping_label_pair():
    """Two 3D label stacks with one partially overlapping region each."""
    labels_a = np.zeros((4, 8, 8), dtype=np.int32)
    labels_a[0:2, 1:4, 1:4] = 1
    labels_b = np.zeros_like(labels_a)
    labels_b[1:3, 2:5, 2:5] = 1
    metadata = [
        {"channel_names": ["A"], "axes": ["Z", "Y", "X"]},
        {"channel_names": ["B"], "axes": ["Z", "Y", "X"]},
    ]
    return labels_a, labels_b, metadata


@pytest.fixture
def hierarchical_label_pair():
    """Two stacks where the A object encloses the B object (cross-channel hierarchy)."""
    labels_a = np.zeros((4, 8, 8), dtype=np.int32)
    labels_a[0:2, 1:6, 1:6] = 1
    labels_b = np.zeros_like(labels_a)
    labels_b[1:2, 2:5, 2:5] = 1
    metadata = [
        {"channel_names": ["A"], "axes": ["Z", "Y", "X"]},
        {"channel_names": ["B"], "axes": ["Z", "Y", "X"]},
    ]
    return labels_a, labels_b, metadata


@pytest.fixture
def multi_object_label_pair():
    """Stack A: 2 objects, stack B: 3 objects → 2×3 overlap matrix."""
    labels_a = np.zeros((4, 16, 16), dtype=np.int32)
    labels_a[0:2, 1:5, 1:5] = 1
    labels_a[0:2, 8:12, 8:12] = 2

    labels_b = np.zeros_like(labels_a)
    labels_b[1:3, 2:6, 2:6] = 1
    labels_b[1:3, 8:12, 8:12] = 2
    labels_b[1:3, 13:15, 13:15] = 3

    metadata = [
        {"channel_names": ["A"], "axes": ["Z", "Y", "X"]},
        {"channel_names": ["B"], "axes": ["Z", "Y", "X"]},
    ]
    return labels_a, labels_b, metadata


@pytest.fixture
def three_stack_labels():
    """Three stacks with one object each for pairing-mode counts."""
    shape = (4, 8, 8)
    labels = [np.zeros(shape, dtype=np.int32) for _ in range(3)]
    for label in labels:
        label[0:2, 1:4, 1:4] = 1
    metadata = [
        {"channel_names": [name], "axes": ["Z", "Y", "X"]}
        for name in ("A", "B", "C")
    ]
    return labels, metadata


def _region_analyzer_config() -> RegionAnalyzerConfig:
    return RegionAnalyzerConfig(
        properties=["bbox"],
        output_type="dataframe",
        index_on="object_id",
        iterator_config=UtilsArrayIteratorConfig(slice_def=()),
    )


def _overlap_calculator_config(**updates) -> LabelOverlapCalculatorConfig:
    return LabelOverlapCalculatorConfig(
        metrics_calculators=metrics_calculator_configs(("ios",)),
        **updates,
    )


def _flow_config(**updates) -> AnalysisFlowConfig:
    """Build flow config; default to combinations for single-pair tests."""
    defaults: dict = {
        "pairing_mode": "combinations",
        "matrix_combiner": None,
        "hierarchy_builder": None,
        "graph_builder": None,
        "graph_query": None,
        "knn_analysis": None,
        "rnn_analysis": None,
    }
    defaults.update(updates)
    return AnalysisFlowConfig(**defaults)


def _run_flow(config: AnalysisFlowConfig, overlapping_label_pair):
    labels_a, labels_b, metadata = overlapping_label_pair
    return AnalysisFlow(config).run([labels_a, labels_b], metadata=metadata)


def _run_flow_stacks(
    config: AnalysisFlowConfig,
    labels: list[np.ndarray],
    metadata: list[dict],
):
    return AnalysisFlow(config).run(labels, metadata=metadata)


def _overlap_keys(measurements: dict) -> list[str]:
    return sorted(key for key in measurements if key.startswith("overlap:"))


@pytest.mark.parametrize(
    ("pairing_mode", "n_items", "expected"),
    [
        ("combinations", 3, [(0, 1), (0, 2), (1, 2)]),
        ("permutations", 2, [(0, 1), (1, 0)]),
        ("product", 2, [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ],
)
def test_pair_indices(pairing_mode, n_items, expected):
    flow = AnalysisFlow(AnalysisFlowConfig(pairing_mode=pairing_mode))
    assert flow._pair_indices(n_items) == expected


def test_analysis_flow_config_accepts_overlap_calculator():
    cfg = _flow_config(overlap_calculator=_overlap_calculator_config())
    assert cfg.overlap_calculator is not None
    assert cfg.coincidence_detector is None


def test_analysis_flow_pairing_mode_combinations_two_stacks(overlapping_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        pairing_mode="combinations",
    )
    measurements = _run_flow(cfg, overlapping_label_pair)
    assert _overlap_keys(measurements) == ["overlap: A vs B"]


def test_analysis_flow_pairing_mode_permutations_two_stacks(overlapping_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        pairing_mode="permutations",
    )
    measurements = _run_flow(cfg, overlapping_label_pair)
    assert _overlap_keys(measurements) == ["overlap: A vs B", "overlap: B vs A"]


def test_analysis_flow_pairing_mode_product_two_stacks(overlapping_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        pairing_mode="product",
    )
    measurements = _run_flow(cfg, overlapping_label_pair)
    assert _overlap_keys(measurements) == [
        "overlap: A vs A",
        "overlap: A vs B",
        "overlap: B vs A",
        "overlap: B vs B",
    ]


def test_analysis_flow_pairing_mode_combinations_three_stacks(three_stack_labels):
    labels, metadata = three_stack_labels
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        pairing_mode="combinations",
    )
    measurements = _run_flow_stacks(cfg, labels, metadata)
    assert _overlap_keys(measurements) == [
        "overlap: A vs B",
        "overlap: A vs C",
        "overlap: B vs C",
    ]


def test_analysis_flow_overlap_pairwise(overlapping_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    assert "region_analyzer: A" in measurements
    assert "region_analyzer: B" in measurements
    overlap = measurements["overlap: A vs B"]
    assert overlap.shape == (1, 1)
    assert float(overlap.iloc[0, 0]) > 0.0


def test_analysis_flow_overlap_filter_indices_returns_array(overlapping_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(ref_value=0.0, operator=">"),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    assert "overlap_filtered: A vs B" in measurements
    filtered = measurements["overlap_filtered: A vs B"]
    assert isinstance(filtered, np.ndarray)
    assert filtered.ndim == 2
    assert filtered.shape[1] == 2


def test_analysis_flow_overlap_filter_masked_values_returns_labeled_dataframe(
    overlapping_label_pair,
):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    overlap = measurements["overlap: A vs B"]
    filtered = measurements["overlap_filtered: A vs B"]

    assert isinstance(filtered, pd.DataFrame)
    assert filtered.shape == overlap.shape
    assert filtered.index.tolist() == overlap.index.tolist()
    assert filtered.columns.tolist() == overlap.columns.tolist()
    assert float(filtered.iloc[0, 0]) > 0.0
    assert np.isnan(filtered.to_numpy()).sum() == 0


def test_analysis_flow_overlap_filter_mask_returns_labeled_dataframe(
    overlapping_label_pair,
):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="mask",
        ),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    overlap = measurements["overlap: A vs B"]
    filtered = measurements["overlap_filtered: A vs B"]

    assert isinstance(filtered, pd.DataFrame)
    assert filtered.shape == overlap.shape
    assert filtered.index.tolist() == overlap.index.tolist()
    assert filtered.columns.tolist() == overlap.columns.tolist()
    assert bool(filtered.iloc[0, 0])
    assert filtered.to_numpy(dtype=bool).shape == (1, 1)


def test_analysis_flow_overlap_aggregated_returns_labeled_series_axis_0(
    overlapping_label_pair,
):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
        overlap_aggregator=MatrixAggregatorConfig(
            operation="max",
            axis=0,
            preferred_input_type="np.ndarray",
        ),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    overlap = measurements["overlap: A vs B"]
    aggregated = measurements["overlap_aggregated: A vs B"]

    assert isinstance(aggregated, pd.Series)
    assert aggregated.index.tolist() == overlap.columns.tolist()
    assert aggregated.name == "max A vs B"
    assert len(aggregated) == overlap.shape[1]
    assert float(aggregated.iloc[0]) > 0.0


def test_analysis_flow_overlap_aggregated_returns_labeled_series_axis_1(
    overlapping_label_pair,
):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
        overlap_aggregator=MatrixAggregatorConfig(
            operation="max",
            axis=1,
            preferred_input_type="np.ndarray",
        ),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    overlap = measurements["overlap: A vs B"]
    aggregated = measurements["overlap_aggregated: A vs B"]

    assert isinstance(aggregated, pd.Series)
    assert aggregated.index.tolist() == overlap.index.tolist()
    assert aggregated.name == "max A vs B"
    assert len(aggregated) == overlap.shape[0]
    assert float(aggregated.iloc[0]) > 0.0


def test_analysis_flow_aggregated_axis_0_labels_columns_on_rectangular_matrix(
    multi_object_label_pair,
):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
        overlap_aggregator=MatrixAggregatorConfig(
            operation="max",
            axis=0,
            preferred_input_type="np.ndarray",
        ),
    )
    measurements = _run_flow(cfg, multi_object_label_pair)

    overlap = measurements["overlap: A vs B"]
    aggregated = measurements["overlap_aggregated: A vs B"]

    assert overlap.shape == (2, 3)
    assert isinstance(aggregated, pd.Series)
    assert len(aggregated) == 3
    assert aggregated.index.tolist() == overlap.columns.tolist()


def test_analysis_flow_aggregated_axis_1_labels_rows_on_rectangular_matrix(
    multi_object_label_pair,
):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
        overlap_aggregator=MatrixAggregatorConfig(
            operation="max",
            axis=1,
            preferred_input_type="np.ndarray",
        ),
    )
    measurements = _run_flow(cfg, multi_object_label_pair)

    overlap = measurements["overlap: A vs B"]
    aggregated = measurements["overlap_aggregated: A vs B"]

    assert overlap.shape == (2, 3)
    assert isinstance(aggregated, pd.Series)
    assert len(aggregated) == 2
    assert aggregated.index.tolist() == overlap.index.tolist()


def test_analysis_flow_region_analyzer_all_concatenates_stacks(overlapping_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
    )
    measurements = _run_flow(cfg, overlapping_label_pair)

    assert "region_analyzer_all" in measurements
    combined = measurements["region_analyzer_all"]
    per_stack_a = measurements["region_analyzer: A"]
    per_stack_b = measurements["region_analyzer: B"]

    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == len(per_stack_a) + len(per_stack_b)
    assert combined.index.name == "object_id"
    assert len(combined.index.unique()) == len(combined)


def test_analysis_flow_hierarchical_analysis(hierarchical_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
        matrix_combiner=MatrixCombinerConfig(
            fill_value=float("nan"),
            symmetrize=True,
            triangle=UPPER,
        ),
        hierarchy_builder=HierarchyBuilderConfig(
            orphan_strategy="as_roots",
            rank_attribute="volume",
            threshold=0.2,
        ),
        graph_builder=GraphBuilderConfig(),
        graph_query=GraphQueryConfig(
            attributes=["descendant_counts", "ancestor_lineage"],
            filter_attribute="channel",
            include_attributes=["label", "channel"],
            lineage_value_attribute="label",
        ),
    )
    measurements = _run_flow(cfg, hierarchical_label_pair)

    assert "ios_global" in measurements
    assert "containment_graph" in measurements
    assert measurements["ios_global"].shape[0] == measurements["ios_global"].shape[1]
    combined = measurements["region_analyzer_all"]
    assert len(combined) == measurements["ios_global"].shape[0]
    assert list(combined.columns).count("label") == 1
    assert list(combined.columns).count("channel") == 1
    assert combined["channel"].map(type).eq(str).all()
    assert len(np.unique(combined["channel"])) >= 1
    assert any(col.startswith("count ") or col.startswith("lineage ") for col in combined.columns)


def test_analysis_flow_knn_analysis(multi_object_label_pair):
    cfg = _flow_config(
        region_analyzer=_region_analyzer_config(),
        overlap_calculator=_overlap_calculator_config(),
        overlap_filter=ValueFilterConfig(
            ref_value=0.0,
            operator=">",
            output="masked_values",
        ),
        matrix_combiner=MatrixCombinerConfig(
            fill_value=float("nan"),
            symmetrize=True,
            triangle=UPPER,
        ),
        hierarchy_builder=HierarchyBuilderConfig(
            orphan_strategy="as_roots",
            rank_attribute="volume",
            threshold=0.2,
        ),
        graph_builder=GraphBuilderConfig(),
        graph_query=None,
        knn_analysis=KnnAnalysisConfig(
            k=1,
            mode="homotypic",
            grouping_attribute="channel",
        ),
    )
    measurements = _run_flow(cfg, multi_object_label_pair)

    assert "containment_graph" in measurements
    knn = measurements["knn_analysis"]
    assert hasattr(knn, "distance_matrix")
    assert hasattr(knn, "matrix")
    assert hasattr(knn, "graph")
    combined = measurements["region_analyzer_all"]
    assert "knn_nearest_neighbor_distance_A_(k=1)" in combined.columns
    assert "knn_mean_distance_A_(k=1)" in combined.columns
