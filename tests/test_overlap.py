"""Tests for vistiq.analysis.overlap."""

import numpy as np
import pandas as pd
import pytest

from vistiq.analysis.coincidence import (
    box_iou_batch_3d,
    labels_iou_batch_3d,
    mask_iou_batch_3d,
)
from vistiq.analysis.overlap import (
    BoxAreaCalculatorConfig,
    BoxBuilderConfig,
    BoxIntersectionCalculatorConfig,
    BoxOverlapCalculatorConfig,
    LabelMaskBuilderConfig,
    LabelOverlapCalculatorConfig,
    MaskAreaCalculatorConfig,
    MaskIntersectionCalculatorConfig,
    MaskOverlapCalculatorConfig,
    MaskStackBuilderConfig,
    OverlapCalculator,
    RegionSpec,
    metrics_calculator_configs,
    region_map_from_dataframe,
)
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


class TestOverlapCalculatorBoxes:
    def test_box_iou_matches_coincidence(self):
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        result = OverlapCalculator(BoxOverlapCalculatorConfig()).run(boxes_a, boxes_b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_box_multi_metric(self):
        boxes_a, boxes_b = _boxes_pair()
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(
                metrics_calculators=metrics_calculator_configs(
                    ("iou", "ios", "dice")
                )
            )
        ).run(boxes_a, boxes_b)
        assert set(result.keys()) == {"iou", "ios", "dice"}
        for matrix in result.values():
            assert matrix.shape == (2, 2)


class TestOverlapCalculatorMasks:
    def test_mask_iou_matches_coincidence(self):
        masks_a, masks_b = _mask_pair()
        expected = mask_iou_batch_3d(masks_a, masks_b, overlap_metric="iou")
        result = OverlapCalculator(MaskOverlapCalculatorConfig()).run(masks_a, masks_b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

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


class TestOverlapCalculatorLabels:
    def test_labels_iou_dense_matches_coincidence(self):
        labels, other = _label_pair()
        expected = labels_iou_batch_3d(labels, other, prune_bboxes=False)
        result = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                builder=LabelMaskBuilderConfig(label_order="unique"),
                intersection_calculator=MaskIntersectionCalculatorConfig(
                    prune_bboxes=False
                ),
            )
        ).run(labels, other)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_labels_iou_dense_anisotropic_spacing(self):
        labels, other = _label_pair()
        spacing = (2.0, 1.0, 1.0)
        numpy_backend = {"preferred_input_type": "numpy"}
        config = LabelOverlapCalculatorConfig(
            builder=LabelMaskBuilderConfig(label_order="unique", **numpy_backend),
            area_calculator=MaskAreaCalculatorConfig(**numpy_backend),
            intersection_calculator=MaskIntersectionCalculatorConfig(
                prune_bboxes=False, **numpy_backend
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

    def test_labels_iou_pruned_matches_coincidence(self):
        labels, other = _label_pair()
        expected = labels_iou_batch_3d(
            labels, other, prune_bboxes=True, dense_pair_fraction=1.0
        )
        result = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                intersection_calculator=MaskIntersectionCalculatorConfig(
                    prune_bboxes=True,
                    dense_pair_fraction=1.0,
                ),
            )
        ).run(labels, other)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestOverlapCalculatorExtras:
    def test_return_components(self):
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
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(triangle=UPPER)
        ).run(boxes, boxes)
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    def test_torch_backend_default(self):
        pytest.importorskip("torch")
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        result = OverlapCalculator(BoxOverlapCalculatorConfig()).run(boxes_a, boxes_b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_numpy_backend_requires_child_reconfiguration(self):
        boxes_a, boxes_b = _boxes_pair()
        expected = box_iou_batch_3d(boxes_a, boxes_b, overlap_metric="iou")
        numpy_backend = {"preferred_input_type": "numpy"}
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(
                builder=BoxBuilderConfig(**numpy_backend),
                area_calculator=BoxAreaCalculatorConfig(**numpy_backend),
                intersection_calculator=BoxIntersectionCalculatorConfig(
                    **numpy_backend
                ),
            )
        ).run(boxes_a, boxes_b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

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
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(
                output_type="dataframe",
                annotate=True,
            )
        ).run(region_map=(map_a, map_b))
        assert list(result.index) == list(map_a.keys())
        assert list(result.columns) == list(map_b.keys())
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5, atol=1e-5)

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
        result = OverlapCalculator(
            LabelOverlapCalculatorConfig(
                builder=LabelMaskBuilderConfig(**numpy_backend),
                area_calculator=MaskAreaCalculatorConfig(**numpy_backend),
                intersection_calculator=MaskIntersectionCalculatorConfig(
                    prune_bboxes=False, **numpy_backend
                ),
                output_type="dataframe",
                annotate=True,
            )
        ).run(labels, other, region_map=(map_a, map_b))
        assert list(result.index) == ["obj-l1", "obj-l2"]
        assert list(result.columns) == ["obj-a1", "obj-a2"]

    def test_annotations_override_region_map_labels(self):
        map_a, map_b = self._box_region_maps()
        custom = (("row-a", "row-b"), ("col-a", "col-b"))
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(output_type="dataframe", annotate=True)
        ).run(region_map=(map_a, map_b), annotations=custom)
        assert list(result.index) == ["row-a", "row-b"]
        assert list(result.columns) == ["col-a", "col-b"]

    def test_annotate_false_ignores_annotations_with_region_map(self):
        map_a, map_b = self._box_region_maps()
        result = OverlapCalculator(
            BoxOverlapCalculatorConfig(output_type="dataframe", annotate=False)
        ).run(
            region_map=(map_a, map_b),
            annotations=(("wrong",), ("obj-b0", "obj-b1")),
        )
        assert list(result.index) == [0, 1]
        assert list(result.columns) == [0, 1]

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
