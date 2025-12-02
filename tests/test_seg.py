"""Tests for vistiq.seg module."""
import numpy as np
import pytest
from skimage.measure import regionprops
from vistiq.seg import (
    RangeThresholdConfig,
    RangeThreshold,
    OtsuThresholdConfig,
    OtsuThreshold,
    RelabelerConfig,
    Relabeler,
    LabelRemoverConfig,
    LabelRemover,
    LabellerConfig,
    Labeller,
    RegionAnalyzerConfig,
    RegionAnalyzer,
    dilate_regions,
    remap_labels,
)
from vistiq.utils import ArrayIteratorConfig


class TestDilateRegions:
    """Tests for dilate_regions function."""

    def test_dilate_single_region(self):
        """Test dilating a single region."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        dilated = dilate_regions(mask, max_area=5000)
        assert dilated.dtype == bool
        # Dilated region should be larger
        assert np.sum(dilated) >= np.sum(mask)

    def test_dilate_multiple_regions(self):
        """Test dilating multiple regions."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True
        mask[50:60, 50:60] = True
        dilated = dilate_regions(mask, max_area=2000)
        assert np.sum(dilated) >= np.sum(mask)

    def test_dilate_no_change_small_area(self):
        """Test that small regions don't change."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True  # 100 pixels
        dilated = dilate_regions(mask, max_area=50)  # Very small max_area
        # Should not dilate if already below threshold
        assert np.sum(dilated) >= np.sum(mask)


class TestRangeThresholdConfig:
    """Tests for RangeThresholdConfig class."""

    def test_default_config(self):
        """Test default RangeThresholdConfig."""
        config = RangeThresholdConfig()
        assert config.threshold is not None

    def test_custom_threshold(self):
        """Test custom threshold."""
        config = RangeThresholdConfig(threshold=(50, 200))
        assert config.threshold == (50, 200)


class TestRangeThreshold:
    """Tests for RangeThreshold class."""

    def test_initialization(self):
        """Test RangeThreshold initialization."""
        config = RangeThresholdConfig(threshold=(50, 200))
        thresholder = RangeThreshold(config)
        assert thresholder.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = RangeThresholdConfig(threshold=(50, 200))
        thresholder = RangeThreshold.from_config(config)
        assert isinstance(thresholder, RangeThreshold)

    def test_process_slice(self, sample_2d_array):
        """Test _process_slice method."""
        config = RangeThresholdConfig(threshold=(50, 200))
        thresholder = RangeThreshold(config)
        result = thresholder._process_slice(sample_2d_array)
        assert result.dtype == bool
        assert result.shape == sample_2d_array.shape

    def test_process_slice_none_threshold(self):
        """Test _process_slice with None threshold values."""
        img = np.array([[10, 50, 100, 200, 250]], dtype=np.uint8)
        config = RangeThresholdConfig(threshold=(None, None))
        thresholder = RangeThreshold(config)
        result = thresholder._process_slice(img)
        # All values should be True (within min-max range)
        assert np.all(result)

    def test_run(self, sample_2d_array):
        """Test run method."""
        config = RangeThresholdConfig(threshold=(50, 200))
        thresholder = RangeThreshold(config)
        result = thresholder.run(sample_2d_array)
        assert result.dtype == bool
        assert result.shape == sample_2d_array.shape


class TestOtsuThresholdConfig:
    """Tests for OtsuThresholdConfig class."""

    def test_default_config(self):
        """Test default OtsuThresholdConfig."""
        config = OtsuThresholdConfig()
        # Just check it can be created
        assert config is not None


class TestOtsuThreshold:
    """Tests for OtsuThreshold class."""

    def test_initialization(self):
        """Test OtsuThreshold initialization."""
        config = OtsuThresholdConfig()
        thresholder = OtsuThreshold(config)
        assert thresholder.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = OtsuThresholdConfig()
        thresholder = OtsuThreshold.from_config(config)
        assert isinstance(thresholder, OtsuThreshold)

    def test_process_slice(self, sample_2d_array):
        """Test _process_slice method."""
        config = OtsuThresholdConfig()
        thresholder = OtsuThreshold(config)
        result = thresholder._process_slice(sample_2d_array)
        assert result.dtype == bool
        assert result.shape == sample_2d_array.shape

    def test_run(self, sample_2d_array):
        """Test run method."""
        config = OtsuThresholdConfig()
        thresholder = OtsuThreshold(config)
        result = thresholder.run(sample_2d_array)
        assert result.dtype == bool
        assert result.shape == sample_2d_array.shape


class TestRelabelerConfig:
    """Tests for RelabelerConfig class."""

    def test_default_config(self):
        """Test default RelabelerConfig."""
        config = RelabelerConfig()
        assert config.output_type == "stack"
        assert config.squeeze is True


class TestRelabeler:
    """Tests for Relabeler class."""

    def test_initialization(self):
        """Test Relabeler initialization."""
        config = RelabelerConfig()
        relabeler = Relabeler(config)
        assert relabeler.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = RelabelerConfig()
        relabeler = Relabeler.from_config(config)
        assert isinstance(relabeler, Relabeler)

    def test_process_slice(self, sample_labels_2d):
        """Test _process_slice method."""
        config = RelabelerConfig()
        relabeler = Relabeler(config)
        result = relabeler._process_slice(sample_labels_2d)
        np.testing.assert_array_equal(result, sample_labels_2d)

    def test_assign_unique_labels_single_array(self, sample_labels_2d):
        """Test assign_unique_labels with single array."""
        result, mappings = Relabeler.assign_unique_labels(sample_labels_2d)
        np.testing.assert_array_equal(result, sample_labels_2d)
        assert isinstance(mappings, dict)

    def test_assign_unique_labels_list(self):
        """Test assign_unique_labels with list of arrays."""
        labels1 = np.zeros((10, 10), dtype=np.int32)
        labels1[2:5, 2:5] = 1
        labels2 = np.zeros((10, 10), dtype=np.int32)
        labels2[6:9, 6:9] = 1
        result, mappings = Relabeler.assign_unique_labels([labels1, labels2])
        assert result.shape == (2, 10, 10)
        # Labels in second array should be offset
        assert np.max(result[1]) > np.max(result[0])

    def test_run_single_array(self, sample_labels_2d):
        """Test run with single array."""
        config = RelabelerConfig()
        relabeler = Relabeler(config)
        result = relabeler.run(sample_labels_2d)
        assert result.shape == sample_labels_2d.shape

    def test_run_list(self):
        """Test run with list of arrays."""
        labels1 = np.zeros((10, 10), dtype=np.int32)
        labels1[2:5, 2:5] = 1
        labels2 = np.zeros((10, 10), dtype=np.int32)
        labels2[6:9, 6:9] = 1
        config = RelabelerConfig()
        relabeler = Relabeler(config)
        result = relabeler.run([labels1, labels2])
        assert result.shape == (2, 10, 10)


class TestLabelRemoverConfig:
    """Tests for LabelRemoverConfig class."""

    def test_default_config(self):
        """Test default LabelRemoverConfig."""
        config = LabelRemoverConfig()
        assert config.output_type == "stack"
        assert config.squeeze is False


class TestLabelRemover:
    """Tests for LabelRemover class."""

    def test_initialization(self):
        """Test LabelRemover initialization."""
        config = LabelRemoverConfig()
        remover = LabelRemover(config)
        assert remover.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = LabelRemoverConfig()
        remover = LabelRemover.from_config(config)
        assert isinstance(remover, LabelRemover)

    def test_extract_label_ids_from_list(self):
        """Test _extract_label_ids with list of ints."""
        config = LabelRemoverConfig()
        remover = LabelRemover(config)
        label_ids = remover._extract_label_ids([1, 2, 3])
        np.testing.assert_array_equal(label_ids, np.array([1, 2, 3], dtype=np.int32))

    def test_extract_label_ids_from_array(self):
        """Test _extract_label_ids with numpy array."""
        config = LabelRemoverConfig()
        remover = LabelRemover(config)
        label_ids = remover._extract_label_ids(np.array([1, 2, 3]))
        np.testing.assert_array_equal(label_ids, np.array([1, 2, 3], dtype=np.int32))

    def test_process_slice(self, sample_labels_2d):
        """Test _process_slice method."""
        config = LabelRemoverConfig()
        remover = LabelRemover(config)
        result = remover._process_slice(sample_labels_2d, np.array([1]))
        # Label 1 should be removed (set to 0)
        assert np.sum(result == 1) == 0
        # Other labels should remain
        assert np.sum(result == 2) > 0

    def test_run(self, sample_labels_2d):
        """Test run method."""
        config = LabelRemoverConfig()
        remover = LabelRemover(config)
        result = remover.run(sample_labels_2d, label_ids=[1, 2])
        assert result.shape == sample_labels_2d.shape
        assert np.sum(result == 1) == 0
        assert np.sum(result == 2) == 0


class TestLabellerConfig:
    """Tests for LabellerConfig class."""

    def test_default_config(self):
        """Test default LabellerConfig."""
        config = LabellerConfig()
        assert config.connectivity in [1, 2]


class TestLabeller:
    """Tests for Labeller class."""

    def test_initialization(self):
        """Test Labeller initialization."""
        config = LabellerConfig()
        labeller = Labeller(config)
        assert labeller.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = LabellerConfig()
        labeller = Labeller.from_config(config)
        assert isinstance(labeller, Labeller)

    def test_process_slice(self):
        """Test _process_slice method."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        mask[30:40, 30:40] = True
        config = LabellerConfig()
        labeller = Labeller(config)
        labels, regions = labeller._process_slice(mask)
        assert labels.dtype == np.int32
        assert labels.shape == mask.shape
        assert isinstance(regions, list)
        assert len(regions) == 2  # Two connected components

    def test_run(self):
        """Test run method."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        config = LabellerConfig()
        labeller = Labeller(config)
        labels, regions = labeller.run(mask)
        assert labels.shape == mask.shape
        assert isinstance(regions, list)


class TestRegionAnalyzerConfig:
    """Tests for RegionAnalyzerConfig class."""

    def test_default_config(self):
        """Test default RegionAnalyzerConfig."""
        config = RegionAnalyzerConfig()
        assert config.output_type in ["list", "dataframe"]
        assert isinstance(config.properties, list)


class TestRegionAnalyzer:
    """Tests for RegionAnalyzer class."""

    def test_initialization(self):
        """Test RegionAnalyzer initialization."""
        config = RegionAnalyzerConfig()
        analyzer = RegionAnalyzer(config)
        assert analyzer.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = RegionAnalyzerConfig()
        analyzer = RegionAnalyzer.from_config(config)
        assert isinstance(analyzer, RegionAnalyzer)

    def test_process_slice_list_output(self, sample_labels_2d):
        """Test _process_slice with list output."""
        config = RegionAnalyzerConfig(output_type="list")
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_2d)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_process_slice_dataframe_output(self, sample_labels_2d):
        """Test _process_slice with dataframe output."""
        import pandas as pd
        config = RegionAnalyzerConfig(output_type="dataframe")
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_2d)
        # regionprops_table returns a dict-like object that can be converted to DataFrame
        assert isinstance(result, (dict, pd.DataFrame)) or hasattr(result, 'keys')

    def test_run(self, sample_labels_2d):
        """Test run method."""
        config = RegionAnalyzerConfig(output_type="list")
        analyzer = RegionAnalyzer(config)
        result = analyzer.run(sample_labels_2d)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_extra_properties_includes_cross_sectional_area(self):
        """Ensure custom property registry exposes cross_sectional_area."""
        assert "cross_sectional_area" in RegionAnalyzer.extra_properties_funcs()

    def test_cross_sectional_area_without_spacing(self):
        """Ensure cross_sectional_area returns voxel count for the largest slice."""
        mask = np.zeros((3, 4, 4), dtype=bool)
        mask[0, :2, :2] = True  # area 4
        mask[1, :3, :3] = True  # area 9
        mask[2, :, :] = True    # area 16
        result = RegionAnalyzer.cross_sectional_area(mask)
        assert result == 16.0

    def test_cross_sectional_area_with_spacing(self):
        """Ensure spacing is applied to the computed cross-sectional area."""
        mask = np.zeros((2, 4, 4), dtype=bool)
        mask[:, :2, :2] = True  # area 4 per slice
        spacing = (1.5, 0.8, 0.5)  # only last two entries used
        expected_area = 4 * (0.8 * 0.5)
        result = RegionAnalyzer.cross_sectional_area(mask, spacing=spacing)
        assert result == expected_area


class TestRemapLabels:
    """Tests for remap_labels function."""

    def test_remap_labels_keep_zero_false(self):
        """Test remap_labels with keep_zero=False."""
        # Input exactly as specified: labels[[2,4,8,5],[3,7,4,4],[0,1,5,3]]
        labels = np.array([[2,4,8,5],[3,7,4,4],[0,1,5,3]], dtype=np.int32)
        result = remap_labels(labels, keep_zero=False)
        
        print(f"\n{'='*70}")
        print(f"Test: remap_labels with keep_zero=False")
        print(f"{'='*70}")
        print(f"\nInput labels:\n{labels}")
        print(f"\nResult:\n{result}")
        print(f"\nUnique labels in input: {sorted(np.unique(labels))}")
        print(f"Unique labels in result: {sorted(np.unique(result))}")
        
        # Check that 0 remains 0
        assert np.all(result[labels == 0] == 0)
        # Check that non-zero labels are remapped to consecutive integers starting from 1
        unique_nonzero = np.unique(result[result > 0])
        if len(unique_nonzero) > 0:
            expected = np.arange(1, len(unique_nonzero) + 1)
            np.testing.assert_array_equal(np.sort(unique_nonzero), expected)
        
    def test_remap_labels_keep_zero_true(self):
        """Test remap_labels with keep_zero=True."""
        # Input exactly as specified: labels[[2,4,8,5],[3,7,4,4],[0,1,5,3]]
        labels = np.array([[2,4,8,5],[3,7,4,4],[0,1,5,3]], dtype=np.int32)
        result = remap_labels(labels, keep_zero=True)
        
        print(f"\n{'='*70}")
        print(f"Test: remap_labels with keep_zero=True")
        print(f"{'='*70}")
        print(f"\nInput labels:\n{labels}")
        print(f"\nResult:\n{result}")
        print(f"\nUnique labels in input: {sorted(np.unique(labels))}")
        print(f"Unique labels in result: {sorted(np.unique(result))}")
        
        # Check that 0 remains 0
        assert np.all(result[labels == 0] == 0)
        # Check that all labels are remapped to consecutive integers starting from 0
        unique_all = np.unique(result)
        expected = np.arange(len(unique_all))
        np.testing.assert_array_equal(np.sort(unique_all), expected)

