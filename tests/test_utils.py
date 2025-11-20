"""Tests for vistiq.utils module."""
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from vistiq.utils import (
    ArrayIteratorConfig,
    ArrayIterator,
    create_unique_folder,
    masks_to_labels,
    labels_to_mask,
)


class TestArrayIteratorConfig:
    """Tests for ArrayIteratorConfig class."""

    def test_default_config(self):
        """Test default ArrayIteratorConfig."""
        config = ArrayIteratorConfig()
        assert config.slice_def == (-2, -1)

    def test_custom_slice_def(self):
        """Test custom slice_def."""
        config = ArrayIteratorConfig(slice_def=(0, 1))
        assert config.slice_def == (0, 1)

    def test_negative_indices(self):
        """Test negative indices in slice_def."""
        config = ArrayIteratorConfig(slice_def=(-3, -2, -1))
        assert config.slice_def == (-3, -2, -1)


class TestArrayIterator:
    """Tests for ArrayIterator class."""

    def test_2d_array_default_slice(self, sample_2d_array):
        """Test iterating over 2D array with default slice_def."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_2d_array, config)
        slices = list(iterator)
        # For 2D array with slice_def=(-2,-1), should iterate over nothing (keep all)
        assert len(slices) == 1
        np.testing.assert_array_equal(slices[0], sample_2d_array)

    def test_3d_array_iterate_first_axis(self, sample_3d_array):
        """Test iterating over first axis of 3D array."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_3d_array, config)
        slices = list(iterator)
        assert len(slices) == sample_3d_array.shape[0]
        for i, slice in enumerate(slices):
            np.testing.assert_array_equal(slice, sample_3d_array[i])

    def test_3d_array_iterate_last_axis(self, sample_3d_array):
        """Test iterating over last axis of 3D array."""
        config = ArrayIteratorConfig(slice_def=(0, 1))
        iterator = ArrayIterator(sample_3d_array, config)
        slices = list(iterator)
        assert len(slices) == sample_3d_array.shape[2]

    def test_4d_array_iteration(self, sample_4d_array):
        """Test iterating over 4D array."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_4d_array, config)
        slices = list(iterator)
        # Should iterate over first 2 axes
        expected_slices = sample_4d_array.shape[0] * sample_4d_array.shape[1]
        assert len(slices) == expected_slices

    def test_invalid_axis_index(self):
        """Test that invalid axis index raises error."""
        arr = np.zeros((10, 20))
        config = ArrayIteratorConfig(slice_def=(5,))  # Invalid for 2D array
        with pytest.raises(ValueError):
            ArrayIterator(arr, config)

    def test_negative_axis_normalization(self, sample_3d_array):
        """Test that negative indices are normalized correctly."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_3d_array, config)
        # Should work the same as (1, 2) for 3D array
        config2 = ArrayIteratorConfig(slice_def=(1, 2))
        iterator2 = ArrayIterator(sample_3d_array, config2)
        slices1 = list(iterator)
        slices2 = list(iterator2)
        assert len(slices1) == len(slices2)
        for s1, s2 in zip(slices1, slices2):
            np.testing.assert_array_equal(s1, s2)

    def test_indices_property(self, sample_3d_array):
        """Test that indices property is set correctly."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_3d_array, config)
        assert len(iterator.indices) == sample_3d_array.shape[0]
        assert all(isinstance(idx, tuple) for idx in iterator.indices)

    def test_len(self, sample_3d_array):
        """Test __len__ method."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_3d_array, config)
        assert len(iterator) == sample_3d_array.shape[0]

    def test_iteration_protocol(self, sample_3d_array):
        """Test that iterator follows iteration protocol."""
        config = ArrayIteratorConfig(slice_def=(-2, -1))
        iterator = ArrayIterator(sample_3d_array, config)
        # Test manual iteration
        first_slice = next(iterator)
        np.testing.assert_array_equal(first_slice, sample_3d_array[0])
        # Test for loop
        iterator2 = ArrayIterator(sample_3d_array, config)
        count = 0
        for _ in iterator2:
            count += 1
        assert count == sample_3d_array.shape[0]


class TestCreateUniqueFolder:
    """Tests for create_unique_folder function."""

    def test_create_basic_folder(self):
        """Test creating a basic unique folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = create_unique_folder(base_path=tmpdir, prefix="test_")
            assert os.path.exists(folder)
            assert os.path.isdir(folder)
            assert folder.startswith(os.path.join(tmpdir, "test_"))

    def test_create_with_suffix(self):
        """Test creating folder with suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = create_unique_folder(
                base_path=tmpdir, prefix="test_", suffix="_suffix"
            )
            assert folder.endswith("_suffix")
            assert os.path.exists(folder)

    def test_exist_ok_false(self):
        """Test exist_ok=False raises error if folder exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = create_unique_folder(base_path=tmpdir, prefix="test_")
            # Try to create same folder again
            with pytest.raises(OSError):
                create_unique_folder(
                    base_path=tmpdir, prefix="test_", exist_ok=False
                )


class TestMasksToLabels:
    """Tests for masks_to_labels function."""

    def test_single_mask(self):
        """Test converting single mask to labels."""
        mask = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)
        masks = [mask]
        labels = masks_to_labels(masks)
        assert labels.shape == mask.shape
        assert np.max(labels) == 1
        assert np.min(labels) == 0

    def test_multiple_masks(self):
        """Test converting multiple masks to labels."""
        mask1 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
        mask2 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool)
        masks = [mask1, mask2]
        labels = masks_to_labels(masks)
        assert labels.shape == mask1.shape
        assert np.max(labels) == 2
        # Check that masks don't overlap in labels
        assert np.sum(labels == 1) == np.sum(mask1)
        assert np.sum(labels == 2) == np.sum(mask2)

    def test_empty_mask_list(self):
        """Test with empty mask list."""
        labels = masks_to_labels([])
        assert labels.size == 0


class TestLabelsToMask:
    """Tests for labels_to_mask function."""

    def test_single_label(self):
        """Test converting labels with single label."""
        labels = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int32)
        masks = labels_to_mask(labels)
        assert len(masks) == 1
        assert masks[0].dtype == bool
        np.testing.assert_array_equal(masks[0], labels == 1)

    def test_multiple_labels(self):
        """Test converting labels with multiple labels."""
        labels = np.array([[0, 1, 2], [0, 1, 2], [0, 0, 0]], dtype=np.int32)
        masks = labels_to_mask(labels)
        assert len(masks) == 2
        np.testing.assert_array_equal(masks[0], labels == 1)
        np.testing.assert_array_equal(masks[1], labels == 2)

    def test_no_labels(self):
        """Test with array containing no labels (all zeros)."""
        labels = np.zeros((10, 10), dtype=np.int32)
        masks = labels_to_mask(labels)
        assert len(masks) == 0

    def test_3d_labels(self):
        """Test converting 3D labels."""
        labels = np.zeros((5, 10, 10), dtype=np.int32)
        labels[1, 2:5, 2:5] = 1
        labels[3, 6:9, 6:9] = 2
        masks = labels_to_mask(labels)
        assert len(masks) == 2
        assert masks[0].shape == labels.shape
        assert masks[1].shape == labels.shape

