"""Tests for vistiq.analysis.overlap."""

import numpy as np
import pandas as pd
import pytest

from vistiq.analysis.coincidence import box_iou_batch_3d, mask_iou_batch_3d
from vistiq.analysis.overlap import (
    BoxAreaCalculatorConfig,
    BoxBuilderConfig,
    BoxIntersectionCalculatorConfig,
    BoxOverlapCalculatorConfig,
    LabelAreaCalculatorConfig,
    LabelBuilderConfig,
    LabelIntersectionCalculatorConfig,
    LabelOverlapCalculatorConfig,
    MaskAreaCalculatorConfig,
    MaskIntersectionCalculatorConfig,
    MaskOverlapCalculatorConfig,
    MaskStackBuilderConfig,
    OverlapCalculator,
    RegionSpec,
    resolve_intersection_mode,
    label_areas,
    label_intersection_linear,
    label_intersection_sparse,
    metrics_calculator_configs,
    region_map_from_dataframe,
    union_matrix,
)
from vistiq.analysis.overlap import IoUMetricsCalculator
from vistiq.constant.matrix import UPPER


def _boxes_pair() -> tuple[np.ndarray, np.ndarray]:
    boxes_a = np.array(
        [[0.0, 0.0, 0.0, 4.0, 4.0, 4.0], [10.0, 10.0, 10.0, 14.0, 14.0, 14.0]],
        dtype=np.float32,
    )
    boxes_b = np.array(
        [[2.0, 2.0, 2.0, 6.0, 6.0, 6.0], [12.0, 12.0, 12.0, 16.0, 16.0, 16.0]],
        dtype=np.float32,
    )
    return boxes_a, boxes_b


def _mask_pair() -> tuple[np.ndarray, np.ndarray]:
    shape = (4, 8, 8)
    masks_a = np.zeros((2,) + shape, dtype=bool)
    masks_a[0, 0:2, 0:2, 0:2] = True
    masks_a[1, 2:4, 4:6, 4:6] = True
    masks_b = np.zeros((2,) + shape, dtype=bool)
    masks_b[0, 1:3, 1:3, 1:3] = True
    masks_b[1, 3:5, 5:7, 5:7] = True
    return masks_a, masks_b


def _label_pair() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    labels[4:7, 10:14, 10:14] = 2

    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1
    other[5:8, 11:15, 11:15] = 2
    return labels, other


def _label_pair_2d() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[10:14, 10:14] = 2

    other = np.zeros_like(labels)
    other[3:7, 3:7] = 1
    other[11:15, 11:15] = 2
    return labels, other


def _boxes_for_labels(
    labels: np.ndarray, label_ids: tuple[int, ...]
) -> np.ndarray:
    width = 2 * labels.ndim
    boxes = np.zeros((len(label_ids), width), dtype=np.float32)
    half = labels.ndim
    for index, label_id in enumerate(label_ids):
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        boxes[index, :half] = mins
        boxes[index, half:] = maxs
    return boxes


def _reference_label_intersection(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    label_ids_a: tuple[int, ...],
    label_ids_b: tuple[int, ...],
) -> np.ndarray:
    out = np.zeros((len(label_ids_a), len(label_ids_b)), dtype=np.float32)
    for i, label_a in enumerate(label_ids_a):
        for j, label_b in enumerate(label_ids_b):
            out[i, j] = float(
                np.count_nonzero((labels_a == label_a) & (labels_b == label_b))
            )
    return out


class TestOverlapCalculatorBoxes:
    def test_box_iou_matches_coincidence(self):
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        calc = OverlapCalculator(BoxOverlapCalculatorConfig())
        result = calc.run(boxes_a, boxes_b)
        np.testing.assert_allclose(calc.format(result), expected, rtol=1e-5, atol=1e-5)

    def test_box_multi_metric(self):
        boxes_a, boxes_b = _boxes_pair()
        calc = OverlapCalculator(
            BoxOverlapCalculatorConfig(
                metrics_calculators=metrics_calculator_configs(
                    ("iou", "ios", "dice")
                )
            )
        )
        result = calc.run(boxes_a, boxes_b)
        formatted = calc.format(result)
        assert set(formatted.keys()) == {"iou", "ios", "dice"}
        for matrix in formatted.values():
            assert matrix.shape == (2, 2)


class TestOverlapCalculatorMasks:
    def test_mask_iou_matches_coincidence(self):
        masks_a, masks_b = _mask_pair()
        expected = mask_iou_batch_3d(masks_a, masks_b, overlap_metric="iou")
        calc = OverlapCalculator(MaskOverlapCalculatorConfig())
        result = calc.run(masks_a, masks_b)
        np.testing.assert_allclose(calc.format(result), expected, rtol=1e-5, atol=1e-5)

    def test_mask_iou_invariant_under_anisotropic_spacing(self):
        masks_a, masks_b = _mask_pair()
        spacing = (2.0, 1.0, 1.0)
        numpy_backend = {"preferred_input_type": "numpy"}
        config = MaskOverlapCalculatorConfig(
            builder=MaskStackBuilderConfig(**numpy_backend),
            area_calculator=MaskAreaCalculatorConfig(**numpy_backend),
            intersection_calculator=MaskIntersectionCalculatorConfig(**numpy_backend),
            return_components=True,
        )
        calc = OverlapCalculator(config)
        without = calc.run(masks_a, masks_b)
        with_spacing = calc.run(masks_a, masks_b, spacing=spacing)
        np.testing.assert_allclose(
            without.metrics["iou"], with_spacing.metrics["iou"], rtol=1e-5, atol=1e-5
        )
        voxel_volume = 2.0
        np.testing.assert_allclose(
            with_spacing.intersection, without.intersection * voxel_volume
        )
        np.testing.assert_allclose(with_spacing.area_a, without.area_a * voxel_volume)
        np.testing.assert_allclose(with_spacing.area_b, without.area_b * voxel_volume)

    def test_mask_metrics_with_signed_spacing(self):
        masks_a, masks_b = _mask_pair()
        positive_spacing = (2.0, 1.0, 1.0)
        signed_spacing = (-2.0, 1.0, -1.0)
        numpy_backend = {"preferred_input_type": "numpy"}
        config = MaskOverlapCalculatorConfig(
            builder=MaskStackBuilderConfig(**numpy_backend),
            area_calculator=MaskAreaCalculatorConfig(**numpy_backend),
            intersection_calculator=MaskIntersectionCalculatorConfig(**numpy_backend),
            metrics_calculators=metrics_calculator_configs(
                ("iou", "ios", "dice")
            ),
            return_components=True,
        )
        calc = OverlapCalculator(config)
        without = calc.run(masks_a, masks_b)
        positive = calc.run(masks_a, masks_b, spacing=positive_spacing)
        signed = calc.run(masks_a, masks_b, spacing=signed_spacing)
        for metric in ("iou", "ios", "dice"):
            np.testing.assert_allclose(
                without.metrics[metric],
                positive.metrics[metric],
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                positive.metrics[metric],
                signed.metrics[metric],
                rtol=1e-5,
                atol=1e-5,
            )
        assert np.all(signed.area_a > 0)
        assert np.all(signed.area_b > 0)
        assert np.all(signed.intersection >= 0)


class TestOverlapCalculatorLabels:
    def _expected_iou(
        self,
        labels: np.ndarray,
        other: np.ndarray,
        label_ids: tuple[int, ...] = (1, 2),
    ) -> np.ndarray:
        inter = label_intersection_linear(labels, other, label_ids, label_ids)
        area_a = label_areas(labels, label_ids)
        area_b = label_areas(other, label_ids)
        union = union_matrix(area_a, area_b, inter=inter)
        return IoUMetricsCalculator.from_config(
            metrics_calculator_configs(("iou",))[0]
        ).compute(inter=inter, union=union)

    def test_labels_iou_linear_path(self):
        labels, other = _label_pair()
        expected = self._expected_iou(labels, other)
        result = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                builder=LabelBuilderConfig(),
                intersection_calculator=LabelIntersectionCalculatorConfig(
                    mode="linear"
                ),
            )
        ).run(labels, other)
        np.testing.assert_allclose(result.metric(), expected, rtol=1e-5, atol=1e-5)

    def test_labels_iou_anisotropic_spacing(self):
        labels, other = _label_pair()
        spacing = (2.0, 1.0, 1.0)
        numpy_backend = {"preferred_input_type": "numpy"}
        config = LabelOverlapCalculatorConfig(
            builder=LabelBuilderConfig(**numpy_backend),
            area_calculator=LabelAreaCalculatorConfig(**numpy_backend),
            intersection_calculator=LabelIntersectionCalculatorConfig(
                mode="linear", **numpy_backend
            ),
            return_components=True,
        )
        calc = OverlapCalculator(config)
        without = calc.run(labels, other)
        with_spacing = calc.run(labels, other, spacing=spacing)
        np.testing.assert_allclose(
            without.metrics["iou"], with_spacing.metrics["iou"], rtol=1e-5, atol=1e-5
        )
        voxel_volume = 2.0
        np.testing.assert_allclose(
            with_spacing.intersection, without.intersection * voxel_volume
        )

    def test_labels_metrics_with_signed_spacing(self):
        labels, other = _label_pair()
        positive_spacing = (2.0, 1.0, 1.0)
        signed_spacing = (-2.0, -1.0, 1.0)
        numpy_backend = {"preferred_input_type": "numpy"}
        config = LabelOverlapCalculatorConfig(
            builder=LabelBuilderConfig(**numpy_backend),
            area_calculator=LabelAreaCalculatorConfig(**numpy_backend),
            intersection_calculator=LabelIntersectionCalculatorConfig(
                mode="linear", **numpy_backend
            ),
            metrics_calculators=metrics_calculator_configs(
                ("iou", "ios", "dice")
            ),
            return_components=True,
        )
        calc = OverlapCalculator(config)
        without = calc.run(labels, other)
        positive = calc.run(labels, other, spacing=positive_spacing)
        signed = calc.run(labels, other, spacing=signed_spacing)
        for metric in ("iou", "ios", "dice"):
            np.testing.assert_allclose(
                without.metrics[metric],
                positive.metrics[metric],
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                positive.metrics[metric],
                signed.metrics[metric],
                rtol=1e-5,
                atol=1e-5,
            )
        assert np.all(signed.area_a > 0)
        assert np.all(signed.area_b > 0)
        assert np.all(signed.intersection >= 0)

    def test_labels_sparse_matches_linear(self):
        labels, other = _label_pair()
        linear = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                builder=LabelBuilderConfig(),
                intersection_calculator=LabelIntersectionCalculatorConfig(
                    mode="linear"
                ),
            )
        ).run(labels, other)
        sparse = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                builder=LabelBuilderConfig(),
                intersection_calculator=LabelIntersectionCalculatorConfig(
                    mode="sparse",
                ),
            )
        ).run(labels, other)
        np.testing.assert_allclose(sparse.metric(), linear.metric(), rtol=1e-5, atol=1e-5)


class TestOverlapCalculatorExtras:
    def test_overlap_result_metrics_only_by_default(self):
        boxes_a, boxes_b = _boxes_pair()
        result = OverlapCalculator(BoxOverlapCalculatorConfig()).run(
            boxes_a, boxes_b
        )
        assert result.metrics["iou"].shape == (2, 2)
        assert result.intersection is None
        assert result.union is None
        assert result.area_a is None
        assert result.area_b is None

    def test_overlap_result_with_components(self):
        boxes_a, boxes_b = _boxes_pair()
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(return_components=True)
        ).run(boxes_a, boxes_b)
        assert result.metrics["iou"].shape == (2, 2)
        assert result.intersection.shape == (2, 2)
        assert result.union.shape == (2, 2)
        assert result.area_a.shape == (2,)
        assert result.area_b.shape == (2,)

    def test_triangle_mask(self):
        boxes = _boxes_pair()[0]
        calc = OverlapCalculator(BoxOverlapCalculatorConfig(triangle=UPPER))
        result = calc.run(boxes, boxes)
        matrix = calc.format(result)
        assert np.isnan(matrix[1, 0])
        assert not np.isnan(matrix[0, 1])

    def test_torch_backend_default(self):
        pytest.importorskip("torch")
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        calc = OverlapCalculator(BoxOverlapCalculatorConfig())
        result = calc.run(boxes_a, boxes_b)
        np.testing.assert_allclose(calc.format(result), expected, rtol=1e-5, atol=1e-5)

    def test_numpy_backend_requires_child_reconfiguration(self):
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        numpy_backend = {"preferred_input_type": "numpy"}
        calc = OverlapCalculator(
            BoxOverlapCalculatorConfig(
                builder=BoxBuilderConfig(**numpy_backend),
                area_calculator=BoxAreaCalculatorConfig(**numpy_backend),
                intersection_calculator=BoxIntersectionCalculatorConfig(
                    **numpy_backend
                ),
            )
        )
        result = calc.run(boxes_a, boxes_b)
        np.testing.assert_allclose(calc.format(result), expected, rtol=1e-5, atol=1e-5)

    def test_backend_mismatch_rejected(self):
        with pytest.raises(ValueError, match="pipeline children must agree"):
            BoxOverlapCalculatorConfig(
                builder=BoxBuilderConfig(preferred_input_type="torch.Tensor"),
                area_calculator=BoxAreaCalculatorConfig(preferred_input_type="numpy"),
            )


class TestRegionMap:
    def _box_region_maps(self) -> tuple[dict[str, RegionSpec], dict[str, RegionSpec]]:
        map_a = {
            "obj-a0": RegionSpec(bbox=(0.0, 0.0, 0.0, 4.0, 4.0, 4.0)),
            "obj-a1": RegionSpec(bbox=(10.0, 10.0, 10.0, 14.0, 14.0, 14.0)),
        }
        map_b = {
            "obj-b0": RegionSpec(bbox=(2.0, 2.0, 2.0, 6.0, 6.0, 6.0)),
            "obj-b1": RegionSpec(bbox=(12.0, 12.0, 12.0, 16.0, 16.0, 16.0)),
        }
        return map_a, map_b

    def test_box_region_map_auto_annotations(self):
        map_a, map_b = self._box_region_maps()
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        calc = OverlapCalculator(
            BoxOverlapCalculatorConfig(
                output_type="dataframe",
                annotate=True,
            )
        )
        result = calc.run(region_map=(map_a, map_b))
        df = calc.format(result)
        assert list(df.index) == list(map_a.keys())
        assert list(df.columns) == list(map_b.keys())
        np.testing.assert_allclose(df.to_numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_label_region_map_auto_annotations(self):
        labels, other = _label_pair()
        map_a = {
            "obj-l1": RegionSpec(label_id=1),
            "obj-l2": RegionSpec(label_id=2),
        }
        map_b = {
            "obj-a1": RegionSpec(label_id=1),
            "obj-a2": RegionSpec(label_id=2),
        }
        numpy_backend = {"preferred_input_type": "numpy"}
        calc = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                builder=LabelBuilderConfig(**numpy_backend),
                area_calculator=LabelAreaCalculatorConfig(**numpy_backend),
                intersection_calculator=LabelIntersectionCalculatorConfig(
                    mode="linear", **numpy_backend
                ),
                output_type="dataframe",
                annotate=True,
            )
        )
        result = calc.run(labels, other, region_map=(map_a, map_b))
        df = calc.format(result)
        assert list(df.index) == ["obj-l1", "obj-l2"]
        assert list(df.columns) == ["obj-a1", "obj-a2"]

    def test_annotations_override_region_map_labels(self):
        map_a, map_b = self._box_region_maps()
        custom = (("row-a", "row-b"), ("col-a", "col-b"))
        calc = OverlapCalculator(
            BoxOverlapCalculatorConfig(output_type="dataframe", annotate=True)
        )
        result = calc.run(region_map=(map_a, map_b), annotations=custom)
        df = calc.format(result)
        assert list(df.index) == ["row-a", "row-b"]
        assert list(df.columns) == ["col-a", "col-b"]

    def test_annotate_false_ignores_annotations_with_region_map(self):
        map_a, map_b = self._box_region_maps()
        calc = OverlapCalculator(
            BoxOverlapCalculatorConfig(output_type="dataframe", annotate=False)
        )
        result = calc.run(
            region_map=(map_a, map_b),
            annotations=(("wrong",), ("obj-b0", "obj-b1")),
        )
        df = calc.format(result)
        assert list(df.index) == [0, 1]
        assert list(df.columns) == [0, 1]

    def test_annotation_length_mismatch_rejected(self):
        map_a, map_b = self._box_region_maps()
        with pytest.raises(ValueError, match="annotations must match region_map size"):
            OverlapCalculator(
                BoxOverlapCalculatorConfig(output_type="dataframe", annotate=True)
            ).run(
                region_map=(map_a, map_b),
                annotations=(("only-one",), ("obj-b0", "obj-b1")),
            )

    def test_region_map_from_dataframe(self):
        df = pd.DataFrame(
            {
                "object_id": ["obj-1", "obj-2"],
                "label": [1, 2],
                "bbox-start-x": [0.0, 10.0],
                "bbox-start-y": [0.0, 10.0],
                "bbox-start-z": [0.0, 10.0],
                "bbox-end-x": [4.0, 14.0],
                "bbox-end-y": [4.0, 14.0],
                "bbox-end-z": [4.0, 14.0],
            }
        )
        region_map = region_map_from_dataframe(df)
        assert tuple(region_map.keys()) == ("obj-1", "obj-2")
        assert region_map["obj-1"].label_id == 1
        assert region_map["obj-1"].bbox == (0.0, 0.0, 0.0, 4.0, 4.0, 4.0)


class TestLabelOverlapPrimitives:
    def test_label_areas_matches_manual_counts_3d(self):
        labels, _ = _label_pair()
        ids = (1, 2)
        areas = label_areas(labels, ids)
        expected = np.array(
            [np.count_nonzero(labels == 1), np.count_nonzero(labels == 2)],
            dtype=np.float64,
        )
        np.testing.assert_allclose(areas, expected)

    def test_label_areas_matches_manual_counts_2d(self):
        labels, _ = _label_pair_2d()
        ids = (1, 2)
        areas = label_areas(labels, ids, spacing=(2.0, 1.0))
        expected = np.array(
            [np.count_nonzero(labels == 1), np.count_nonzero(labels == 2)],
            dtype=np.float64,
        ) * 2.0
        np.testing.assert_allclose(areas, expected)

    def test_label_intersection_linear_matches_reference_3d(self):
        labels, other = _label_pair()
        ids = (1, 2)
        expected = _reference_label_intersection(labels, other, ids, ids)
        linear = label_intersection_linear(labels, other, ids, ids)
        np.testing.assert_allclose(linear, expected, rtol=1e-5, atol=1e-5)

    def test_label_intersection_linear_matches_reference_2d(self):
        labels, other = _label_pair_2d()
        ids = (1, 2)
        expected = _reference_label_intersection(labels, other, ids, ids)
        linear = label_intersection_linear(labels, other, ids, ids)
        np.testing.assert_allclose(linear, expected, rtol=1e-5, atol=1e-5)

    def test_label_intersection_linear_discontinuous_ids(self):
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[4:7, 4:7] = 5
        other = np.zeros_like(labels)
        other[2:5, 2:5] = 2
        other[5:8, 5:8] = 6
        ids_a = (1, 5)
        ids_b = (2, 6)
        expected = _reference_label_intersection(labels, other, ids_a, ids_b)
        linear = label_intersection_linear(labels, other, ids_a, ids_b)
        np.testing.assert_allclose(linear, expected, rtol=1e-5, atol=1e-5)

    def test_label_intersection_sparse_matches_linear_3d(self):
        labels, other = _label_pair()
        ids = (1, 2)
        boxes_a = _boxes_for_labels(labels, ids)
        boxes_b = _boxes_for_labels(other, ids)
        linear = label_intersection_linear(labels, other, ids, ids)
        sparse = label_intersection_sparse(
            labels, other, ids, ids, boxes_a, boxes_b
        )
        np.testing.assert_allclose(sparse, linear, rtol=1e-5, atol=1e-5)

    def test_label_intersection_sparse_matches_linear_2d(self):
        labels, other = _label_pair_2d()
        ids = (1, 2)
        boxes_a = _boxes_for_labels(labels, ids)
        boxes_b = _boxes_for_labels(other, ids)
        linear = label_intersection_linear(labels, other, ids, ids)
        sparse = label_intersection_sparse(
            labels, other, ids, ids, boxes_a, boxes_b
        )
        np.testing.assert_allclose(sparse, linear, rtol=1e-5, atol=1e-5)

    def test_resolve_intersection_mode_auto_prefers_linear_for_large_volume(self):
        strategy = resolve_intersection_mode(
            shape=(204, 512, 512),
            n_objects_a=1,
            n_objects_b=204,
            mode="auto",
        )
        assert strategy.mode == "linear"

    def test_resolve_intersection_mode_auto_prefers_sparse_for_small_separated(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[0:4, 0:4] = 1
        labels[0:4, 28:32] = 2
        other = np.zeros_like(labels)
        other[28:32, 0:4] = 1
        other[28:32, 28:32] = 2
        ids = (1, 2)
        boxes_a = _boxes_for_labels(labels, ids)
        boxes_b = _boxes_for_labels(other, ids)
        strategy = resolve_intersection_mode(
            shape=labels.shape,
            n_objects_a=2,
            n_objects_b=2,
            boxes_a=boxes_a,
            boxes_b=boxes_b,
            mode="auto",
            total_memory_limit=10_000,
        )
        assert strategy.mode == "sparse"

    def test_resolve_intersection_mode_honors_explicit_linear(self):
        strategy = resolve_intersection_mode(
            shape=(8, 16, 16),
            n_objects_a=2,
            n_objects_b=2,
            mode="linear",
        )
        assert strategy.mode == "linear"

    def test_resolve_intersection_mode_honors_explicit_sparse(self):
        strategy = resolve_intersection_mode(
            shape=(204, 512, 512),
            n_objects_a=1,
            n_objects_b=204,
            mode="sparse",
        )
        assert strategy.mode == "sparse"
