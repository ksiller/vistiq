"""Tests for coincidence detection."""

import numpy as np
import pytest

from vistiq.matrix.ops import MatrixFormatter, MatrixFormatterConfig
from vistiq.analysis.overlap import (
    DiceMetricsCalculatorConfig,
    IoUMetricsCalculatorConfig,
    LabelAreaCalculatorConfig,
    LabelBuilderConfig,
    LabelIntersectionCalculatorConfig,
    LabelOverlapCalculatorConfig,
    OverlapCalculator,
    metrics_calculator_configs,
)


def _label_overlap(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    intersection_mode: str = "auto",
    preferred_input_type: str = "numpy",
) -> np.ndarray:
    backend = {"preferred_input_type": preferred_input_type}
    calc = OverlapCalculator(
        LabelOverlapCalculatorConfig(
            builder=LabelBuilderConfig(**backend),
            area_calculator=LabelAreaCalculatorConfig(**backend),
            intersection_calculator=LabelIntersectionCalculatorConfig(
                mode=intersection_mode,
                **backend,
            ),
            metrics_calculators=metrics_calculator_configs(("iou",)),
        )
    )
    result = calc.run(labels_a, labels_b)
    return MatrixFormatter(
        MatrixFormatterConfig(output_type="np.ndarray", annotate=False)
    ).run(result.metric())


def test_label_overlap_sparse_matches_linear():
    """Sparse and linear label paths should agree on small 3D volumes."""
    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    labels[4:7, 10:14, 10:14] = 2

    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1
    other[5:8, 11:15, 11:15] = 2

    linear = _label_overlap(labels, other, intersection_mode="linear")
    sparse = _label_overlap(labels, other, intersection_mode="sparse")
    np.testing.assert_allclose(linear, sparse, rtol=1e-5, atol=1e-5)


def test_label_overlap_sparse_skips_disjoint():
    """Disjoint regions should yield zero overlap."""
    labels = np.zeros((4, 8, 8), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1

    other = np.zeros_like(labels)
    other[2:4, 6:8, 6:8] = 1

    out = _label_overlap(labels, other, intersection_mode="sparse")
    assert out.shape == (1, 1)
    assert out[0, 0] == 0.0


def test_label_overlap_torch_matches_numpy():
    pytest.importorskip("torch")
    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    labels[4:7, 10:14, 10:14] = 2

    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1
    other[5:8, 11:15, 11:15] = 2

    numpy_out = _label_overlap(labels, other, intersection_mode="linear")
    torch_out = _label_overlap(
        labels, other, intersection_mode="linear", preferred_input_type="torch.Tensor"
    )
    np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-5, atol=1e-5)


def test_coincidence_detector_process_slice_outline_iou():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method=IoUMetricsCalculatorConfig(),
            mode="outline",
            threshold=0.2,
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, ("Lobe", "Cell"))
    assert len(results) == 1
    assert results[0]["Lobe"] == 1
    assert results[0]["Cell"] == 1
    assert results[0]["score"] > 0.0
    assert results[0]["above_threshold"] is True


def test_coincidence_detector_accepts_list_stack_names():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method=IoUMetricsCalculatorConfig(),
            mode="outline",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, (["Scrib"], ["EdU"]))
    assert len(results) == 1
    assert results[0]["Scrib"] == 1
    assert results[0]["EdU"] == 1


def test_coincidence_detector_process_slice_bounding_box():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method=IoUMetricsCalculatorConfig(),
            mode="bounding_box",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, ("A", "B"))
    assert len(results) == 1
    assert results[0]["score"] > 0.0


def test_coincidence_detector_process_slice_outline_dice():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method=DiceMetricsCalculatorConfig(),
            mode="outline",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, ("Lobe", "Cell"))
    assert len(results) == 1
    assert results[0]["score"] > 0.0
