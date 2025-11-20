"""Tests for vistiq.analysis module."""
import numpy as np
import pytest
from vistiq.analysis import (
    CoincidenceDetectorConfig,
    CoincidenceDetector,
)
from vistiq.utils import ArrayIteratorConfig


class TestCoincidenceDetectorConfig:
    """Tests for CoincidenceDetectorConfig class."""

    def test_default_config(self):
        """Test default CoincidenceDetectorConfig."""
        config = CoincidenceDetectorConfig()
        assert config.output_type == "list"
        assert config.method == "iou"
        assert config.mode == "outline"
        assert config.threshold == 0.5

    def test_custom_config(self):
        """Test custom CoincidenceDetectorConfig."""
        config = CoincidenceDetectorConfig(
            method="dice", mode="bounding_box", threshold=0.7
        )
        assert config.method == "dice"
        assert config.mode == "bounding_box"
        assert config.threshold == 0.7

    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid threshold
        config = CoincidenceDetectorConfig(threshold=0.5)
        assert config.threshold == 0.5

        # Invalid threshold - too low
        with pytest.raises(Exception):  # Pydantic validation error
            CoincidenceDetectorConfig(threshold=-0.1)

        # Invalid threshold - too high
        with pytest.raises(Exception):  # Pydantic validation error
            CoincidenceDetectorConfig(threshold=1.5)


class TestCoincidenceDetector:
    """Tests for CoincidenceDetector class."""

    def test_initialization(self):
        """Test CoincidenceDetector initialization."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        assert detector.config == config

    def test_iou_method(self):
        """Test _iou method."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        
        # 2D masks - Perfect overlap
        mask1 = np.array([[1, 1], [1, 1]], dtype=bool)
        mask2 = np.array([[1, 1], [1, 1]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert iou == 1.0

        # 2D masks - No overlap
        mask1 = np.array([[1, 0], [0, 0]], dtype=bool)
        mask2 = np.array([[0, 0], [0, 1]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert iou == 0.0

        # 2D masks - Partial overlap
        mask1 = np.array([[1, 1], [0, 0]], dtype=bool)
        mask2 = np.array([[1, 0], [1, 0]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert 0.0 < iou < 1.0
        
        # 3D masks - Perfect overlap
        mask1 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        mask2 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert iou == 1.0

        # 3D masks - No overlap
        mask1 = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
        mask2 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert iou == 0.0

        # 3D masks - Partial overlap
        mask1 = np.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
        mask2 = np.array([[[1, 0], [1, 0]], [[0, 0], [0, 0]]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert 0.0 < iou < 1.0
        
        # 2D masks - One mask completely encompasses the other (identical masks, score = 1.0)
        mask1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        mask2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        # mask1 completely encompasses mask2 (they are identical)
        assert np.all(mask2 <= mask1)
        iou = detector._iou(mask1, mask2)
        assert iou == 1.0
        
        # 2D masks - Another case: identical masks where one encompasses the other
        mask1 = np.array([[1, 1], [1, 1]], dtype=bool)
        mask2 = np.array([[1, 1], [1, 1]], dtype=bool)
        assert np.all(mask2 <= mask1)
        iou = detector._iou(mask1, mask2)
        assert iou == 1.0
        
        # 3D masks - One mask completely encompasses the other (identical masks)
        mask1 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        mask2 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        iou = detector._iou(mask1, mask2)
        assert iou == 1.0

    def test_dice_method(self):
        """Test _dice method."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        
        # 2D masks - Perfect overlap
        mask1 = np.array([[1, 1], [1, 1]], dtype=bool)
        mask2 = np.array([[1, 1], [1, 1]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert dice == 1.0

        # 2D masks - No overlap
        mask1 = np.array([[1, 0], [0, 0]], dtype=bool)
        mask2 = np.array([[0, 0], [0, 1]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert dice == 0.0

        # 2D masks - Partial overlap
        mask1 = np.array([[1, 1], [0, 0]], dtype=bool)
        mask2 = np.array([[1, 0], [1, 0]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert 0.0 < dice < 1.0
        
        # 3D masks - Perfect overlap
        mask1 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        mask2 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert dice == 1.0

        # 3D masks - No overlap
        mask1 = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
        mask2 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert dice == 0.0

        # 3D masks - Partial overlap
        mask1 = np.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
        mask2 = np.array([[[1, 0], [1, 0]], [[0, 0], [0, 0]]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert 0.0 < dice < 1.0
        
        # 2D masks - One mask completely encompasses the other (identical masks, score = 1.0)
        mask1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        mask2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        # mask1 completely encompasses mask2 (they are identical)
        assert np.all(mask2 <= mask1)
        dice = detector._dice(mask1, mask2)
        assert dice == 1.0
        
        # 2D masks - Another case: identical masks where one encompasses the other
        mask1 = np.array([[1, 1], [1, 1]], dtype=bool)
        mask2 = np.array([[1, 1], [1, 1]], dtype=bool)
        assert np.all(mask2 <= mask1)
        dice = detector._dice(mask1, mask2)
        assert dice == 1.0
        
        # 3D masks - One mask completely encompasses the other (identical masks)
        mask1 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        mask2 = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        dice = detector._dice(mask1, mask2)
        assert dice == 1.0

    def test_bboxes_overlap_2d(self):
        """Test _bboxes_overlap for 2D bounding boxes."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        # Overlapping boxes
        bbox1 = (10, 10, 30, 30)  # (min_row, min_col, max_row, max_col)
        bbox2 = (20, 20, 40, 40)
        assert detector._bboxes_overlap(bbox1, bbox2) is True

        # Non-overlapping boxes
        bbox1 = (10, 10, 20, 20)
        bbox2 = (30, 30, 40, 40)
        assert detector._bboxes_overlap(bbox1, bbox2) is False

    def test_bboxes_overlap_3d(self):
        """Test _bboxes_overlap for 3D bounding boxes."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        # Overlapping boxes
        bbox1 = (10, 10, 5, 30, 30, 15)  # (min_row, min_col, min_slice, max_row, max_col, max_slice)
        bbox2 = (20, 20, 10, 40, 40, 20)
        assert detector._bboxes_overlap(bbox1, bbox2) is True

        # Non-overlapping boxes
        bbox1 = (10, 10, 5, 20, 20, 10)
        bbox2 = (30, 30, 15, 40, 40, 20)
        assert detector._bboxes_overlap(bbox1, bbox2) is False

    def test_bbox_union_2d(self):
        """Test _bbox_union for 2D bounding boxes."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        bboxes = [(10, 10, 20, 20), (15, 15, 25, 25)]
        union = detector._bbox_union(bboxes)
        assert union == (10, 10, 25, 25)

    def test_bbox_union_3d(self):
        """Test _bbox_union for 3D bounding boxes."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        # Test flat format
        bboxes = [(10, 10, 5, 20, 20, 10), (15, 15, 8, 25, 25, 15)]
        union = detector._bbox_union(bboxes)
        assert union == (10, 10, 5, 25, 25, 15)
        
        # Test tuple-of-tuples format
        bboxes_tuple = [((10, 10, 5), (20, 20, 10)), ((15, 15, 8), (25, 25, 15))]
        union_tuple = detector._bbox_union(bboxes_tuple)
        assert union_tuple == ((10, 10, 5), (25, 25, 15))

    def test_bbox_union_empty_list(self):
        """Test _bbox_union with empty list."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        union = detector._bbox_union([])
        assert union is None

    def test_extract_region_2d(self, sample_labels_2d):
        """Test _extract_region for 2D."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        bbox = (10, 10, 30, 30)
        region = detector._extract_region(sample_labels_2d, bbox)
        assert region.shape == (20, 20)

    def test_extract_region_3d(self, sample_labels_3d):
        """Test _extract_region for 3D."""
        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        bbox = (2, 10, 10, 7, 30, 30)
        region = detector._extract_region(sample_labels_3d, bbox)
        assert region.shape == (5, 20, 20)

    def test_process_slice_outline_mode(self, sample_labels_2d):
        """Test _process_slice with outline mode."""
        labels1 = sample_labels_2d.copy()
        labels2 = sample_labels_2d.copy()
        # Shift labels2 slightly
        labels2 = np.roll(labels2, 5, axis=0)
        labels2 = np.roll(labels2, 5, axis=1)

        config = CoincidenceDetectorConfig(mode="outline", method="iou")
        detector = CoincidenceDetector(config)
        results = detector._process_slice(labels1, labels2)
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert "score" in result
            assert "above_threshold" in result

    def test_process_slice_bounding_box_mode(self, sample_labels_2d):
        """Test _process_slice with bounding_box mode."""
        labels1 = sample_labels_2d.copy()
        labels2 = sample_labels_2d.copy()

        config = CoincidenceDetectorConfig(mode="bounding_box", method="iou")
        detector = CoincidenceDetector(config)
        results = detector._process_slice(labels1, labels2)
        assert isinstance(results, list)

    def test_run(self, sample_labels_2d):
        """Test run method."""
        labels1 = sample_labels_2d.copy()
        labels2 = sample_labels_2d.copy()

        config = CoincidenceDetectorConfig()
        detector = CoincidenceDetector(config)
        results, consolidated = detector.run(labels1, labels2)
        assert isinstance(results, list)
        assert isinstance(consolidated, dict)
        assert len(consolidated) == 2  # Two stacks

