"""Tests for vistiq.segment module."""
import numpy as np
import pytest
from vistiq.utils import ArrayIteratorConfig
from vistiq.segment import (
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
    RangeFilter,
    RangeFilterConfig,
    RegionFilter,
    RegionFilterConfig,
    dilate_regions,
    remap_labels,
)


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
        result, _ = thresholder.run(sample_2d_array)
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
        result, _ = thresholder.run(sample_2d_array)
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
        result, _ = remover.run(sample_labels_2d, region_properties=[1, 2])
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
        assert config.index_on == "label"
        assert isinstance(config.properties, list)
        for name in RegionAnalyzer.mandatory_properties:
            assert name in config.properties

    def test_mandatory_ids_when_properties_omitted(self):
        """Mandatory ids are always present even if properties list is empty."""
        config = RegionAnalyzerConfig(properties=["area"])
        for name in ("label", "object_id", "slice_id", "stack_id"):
            assert name in config.properties

    def test_accepts_mapped_property_names(self):
        """properties may list map_axes column names such as cross_sectional_area-xy."""
        config = RegionAnalyzerConfig(
            properties=[
                "label",
                "object_id",
                "slice_id",
                "stack_id",
                "centroid",
                "cross_sectional_area-xy",
                "cross_sectional_area-xz",
                "cross_sectional_area-yz",
                "aspect_ratio",
            ],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        assert "cross_sectional_area-xy" in config.properties

    def test_iterator_config_slice_def_validation(self):
        """slice_def may be (), None, or keep >= 2 axes; single-axis is invalid."""
        RegionAnalyzerConfig(iterator_config=ArrayIteratorConfig(slice_def=()))
        RegionAnalyzerConfig(iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)))
        RegionAnalyzerConfig(
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1))
        )
        with pytest.raises(ValueError, match="length 1"):
            RegionAnalyzerConfig(iterator_config=ArrayIteratorConfig(slice_def=(0,)))
        with pytest.raises(ValueError, match="length 1"):
            RegionAnalyzerConfig(iterator_config=ArrayIteratorConfig(slice_def=(-1,)))


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

    def test_assign_channel_names_from_metadata(self, sample_labels_3d):
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "bbox"],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        result = RegionAnalyzer(config)._process_slice(
            sample_labels_3d,
            metadata={"channel_names": ["Scrib"]},
        )
        assert isinstance(result, pd.DataFrame)
        assert "channel" in result.columns
        assert result["channel"].tolist() == ["Scrib"] * len(result)
        assert result.index.name == "label"
        assert "label" not in result.columns
        assert "object_name" in result.columns
        assert result["object_name"].tolist() == [
            f"Scrib {int(label)}" for label in result.index
        ]

    def test_assign_object_names_index_on_object_id(self, sample_labels_2d):
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            index_on="object_id",
            properties=["label", "bbox"],
        )
        result = RegionAnalyzer(config)._process_slice(
            sample_labels_2d,
            metadata={"channel_names": ["Dpn"]},
        )
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "object_id"
        assert "label" in result.columns
        assert "object_name" in result.columns
        assert result["object_name"].tolist() == [
            f"Dpn {int(label)}" for label in result["label"]
        ]

    def test_assign_object_names_list_output(self, sample_labels_3d):
        config = RegionAnalyzerConfig(
            output_type="list",
            properties=["label", "bbox"],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        result = RegionAnalyzer(config)._process_slice(
            sample_labels_3d,
            metadata={"channel_names": ["EdU"]},
        )
        assert len(result) > 0
        for region in result:
            assert region.object_name == f"EdU {region.label}"

    def test_assign_channel_names_skips_when_missing(self, sample_labels_3d):
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "bbox"],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        result = RegionAnalyzer(config)._process_slice(
            sample_labels_3d,
            metadata=None,
        )
        assert isinstance(result, pd.DataFrame)
        assert "channel" not in result.columns
        assert "object_name" not in result.columns

    def test_channel_names_string_accepts_scalar(self):
        assert RegionAnalyzer._channel_names_string("Scrib") == "Scrib"
        assert RegionAnalyzer._channel_names_string(["Scrib", "EdU"]) == "Scrib,EdU"
        assert RegionAnalyzer._channel_names_string(None) is None

    def test_area_relabeled_as_volume_for_3d(self, sample_labels_3d):
        """regionprops area on 3D label slices is exposed as volume."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "area"],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        result = RegionAnalyzer(config)._process_slice(sample_labels_3d)
        assert isinstance(result, pd.DataFrame)
        assert "volume" in result.columns
        assert "area" not in result.columns

    def test_area_unchanged_for_2d(self, sample_labels_2d):
        """regionprops area on 2D label slices stays area."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "area"],
        )
        result = RegionAnalyzer(config)._process_slice(sample_labels_2d)
        assert isinstance(result, pd.DataFrame)
        assert "area" in result.columns
        assert "volume" not in result.columns

    def test_list_output_volume_alias_for_3d(self, sample_labels_3d):
        """List output sets volume from regionprops area on 3D slices."""
        config = RegionAnalyzerConfig(
            output_type="list",
            properties=["label", "area"],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        result = RegionAnalyzer(config)._process_slice(sample_labels_3d)
        assert len(result) > 0
        region = result[0]
        assert region.__dict__["volume"] == region.area
        assert RegionAnalyzer.get_region_attribute(region, "volume") == region.area

    def test_area_and_volume_positive_with_negative_spacing(
        self, sample_labels_3d
    ):
        """Negative axis spacing (e.g. inverted Z) still yields positive volume."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "area"],
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
        result = RegionAnalyzer(config)._process_slice(
            sample_labels_3d,
            metadata={"scale": (-1.0, 0.3, 0.3)},
        )
        assert isinstance(result, pd.DataFrame)
        assert (result["volume"] > 0).all()

    def test_process_slice_metadata_none(self, sample_labels_3d):
        """_process_slice and run tolerate metadata=None (list and dataframe)."""
        list_config = RegionAnalyzerConfig(
            output_type="list",
            map_axes=True,
            properties=["label", "cross_sectional_area-xy", "centroid-x"],
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        list_result = RegionAnalyzer(list_config)._process_slice(sample_labels_3d)
        assert len(list_result) > 0
        assert hasattr(list_result[0], "_vistiq_slice_axes")

        df_config = RegionAnalyzerConfig(
            output_type="dataframe",
            map_axes=True,
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        df_result = RegionAnalyzer(df_config)._process_slice(sample_labels_3d)
        assert len(df_result) > 0

        run_result = RegionAnalyzer(list_config).run(sample_labels_3d, metadata=None)
        assert isinstance(run_result, list)
        assert len(run_result) > 0

    def test_process_slice_dataframe_output(self, sample_labels_2d):
        """Test _process_slice with dataframe output."""
        import pandas as pd
        config = RegionAnalyzerConfig(output_type="dataframe")
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_2d)
        # regionprops_table returns a dict-like object that can be converted to DataFrame
        assert isinstance(result, (dict, pd.DataFrame)) or hasattr(result, 'keys')
        if isinstance(result, pd.DataFrame):
            assert result.index.name == "label"
            assert "label" not in result.columns
            assert "object_id" in result.columns
            assert "slice_id" in result.columns
            assert "stack_id" in result.columns
            assert len(result["object_id"].unique()) == len(result)
            assert len(result["slice_id"].unique()) == 1
            assert len(result["stack_id"].unique()) == 1

    def test_process_slice_dataframe_index_on_object_id(self, sample_labels_2d):
        """DataFrame output can be indexed by object_id instead of label."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            index_on="object_id",
        )
        result = RegionAnalyzer(config)._process_slice(sample_labels_2d)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "object_id"
        assert "object_id" not in result.columns
        assert "label" in result.columns
        assert len(result.index.unique()) == len(result)

    def test_process_slice_dataframe_includes_slice_annotations(self, sample_labels_2d):
        """Slice annotation axis columns are replicated per region row."""
        import pandas as pd
        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "slice_annotations"],
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(
            sample_labels_2d,
            slice_annotations={"C": 0, "Z": 1},
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result["c"]) == [0] * len(result)
        assert list(result["z"]) == [1] * len(result)
        assert result["c"].dtype == np.int64
        assert result["z"].dtype == np.int64

    def test_list_output_exposes_mapped_cross_sectional_area(self, sample_labels_3d):
        """List output sets plane-specific attributes for RegionFilter."""
        config = RegionAnalyzerConfig(
            output_type="list",
            properties=[
                "label",
                "object_id",
                "slice_id",
                "stack_id",
                "cross_sectional_area-xy",
            ],
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_3d, metadata={"axes": list("ZYX")})
        assert len(result) > 0
        assert hasattr(result[0], "cross_sectional_area-xy")
        value = RegionAnalyzer.get_region_attribute(result[0], "cross_sectional_area-xy")
        assert isinstance(value, (float, int, np.floating, np.integer))

    def test_get_region_attribute_aspect_ratio_returns_overall_scalar(
        self, sample_labels_3d
    ):
        """Bare aspect_ratio resolves to the overall scalar, not the full tuple."""
        config = RegionAnalyzerConfig(
            output_type="list",
            properties=["label", "aspect_ratio"],
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_3d, metadata={"axes": list("ZYX")})
        assert len(result) > 0
        overall = RegionAnalyzer.get_region_attribute(result[0], "aspect_ratio")
        raw = result[0].aspect_ratio
        assert isinstance(raw, tuple)
        assert overall == raw[-1]
        assert isinstance(overall, (float, int, np.floating, np.integer))

    def test_region_filter_accepts_vector_property_scalars(self, sample_labels_3d):
        """RegionFilter compares scalars for mapped and overall vector properties."""
        config = RegionAnalyzerConfig(
            output_type="list",
            properties=[
                "label",
                "cross_sectional_area-xy",
                "aspect_ratio",
            ],
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        regions = analyzer._process_slice(sample_labels_3d, metadata={"axes": list("ZYX")})
        region_filter = RegionFilter(
            RegionFilterConfig(
                filters=[
                    RangeFilterConfig(
                        attribute="cross_sectional_area-xy",
                        range=(0.0, float("inf")),
                    ),
                    RangeFilterConfig(
                        attribute="aspect_ratio",
                        range=(0.0, 1.0),
                    ),
                ]
            )
        )
        accepted, removed = region_filter.run(regions)
        assert isinstance(accepted, list)
        assert isinstance(removed, list)

    def test_region_filter_bare_cross_sectional_area_on_3d_raises(
        self, sample_labels_3d
    ):
        """Bare cross_sectional_area on 3D+ raises; use a plane-specific name."""
        config = RegionAnalyzerConfig(
            output_type="list",
            properties=["label", "cross_sectional_area"],
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        regions = RegionAnalyzer(config)._process_slice(
            sample_labels_3d, metadata={"axes": list("ZYX")}
        )
        with pytest.raises(AttributeError, match="cross_sectional_area-xy"):
            RegionAnalyzer.get_region_attribute(regions[0], "cross_sectional_area")

    def test_process_slice_list_output_includes_mandatory_ids(
        self, sample_labels_2d
    ):
        """List output includes mandatory ids on each region."""
        config = RegionAnalyzerConfig(output_type="list", properties=["area"])
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_2d)
        assert len(result) > 0
        for region in result:
            assert hasattr(region, "object_id")
            assert hasattr(region, "slice_id")
            assert hasattr(region, "stack_id")
        assert len({r.object_id for r in result}) == len(result)
        assert len({r.slice_id for r in result}) == 1
        assert len({r.stack_id for r in result}) == 1

    def test_run(self, sample_labels_2d):
        """Test run method."""
        config = RegionAnalyzerConfig(output_type="list")
        analyzer = RegionAnalyzer(config)
        result = analyzer.run(sample_labels_2d)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_run_dataframe_slice_annotations_over_iterations(self):
        """Iterated stacks add C/Z columns from metadata axes and iterator indices."""
        import pandas as pd

        labels = np.zeros((2, 2, 40, 40), dtype=np.int32)
        labels[0, 0, 5:15, 5:15] = 1
        labels[1, 1, 20:30, 20:30] = 2
        metadata = {"axes": list("CZYX")}
        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "object_id", "slice_id", "stack_id", "slice_annotations"],
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer.run(labels, metadata=metadata)
        assert isinstance(result, pd.DataFrame)
        assert "c" in result.columns
        assert "z" in result.columns
        assert set(zip(result["c"], result["z"])) == {(0, 0), (1, 1)}
        assert result["c"].dtype == np.int64
        assert result["z"].dtype == np.int64
        assert len(result) == 2
        assert len(result["stack_id"].unique()) == 1
        assert len(result["slice_id"].unique()) == 2

    def test_map_axes_renames_centroid_and_bbox_columns(self, sample_labels_2d):
        """map_axes uses metadata axes and slice_def for column names."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "object_id", "slice_id", "stack_id", "centroid", "bbox"],
            map_axes=True,
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        metadata = {"axes": list("CZYX")}
        result = analyzer._process_slice(sample_labels_2d, metadata=metadata)
        assert isinstance(result, pd.DataFrame)
        assert "centroid-y" in result.columns
        assert "centroid-x" in result.columns
        assert "bbox-start-y" in result.columns
        assert "bbox-end-x" in result.columns
        assert "centroid-0" not in result.columns
        assert "bbox-0" not in result.columns

    def test_map_axes_3d_slice_labels(self, sample_labels_3d):
        """3D slices map bbox/centroid suffixes to Z, Y, X from metadata."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "object_id", "slice_id", "stack_id", "centroid", "bbox"],
            map_axes=True,
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        metadata = {"axes": list("ZYX")}
        result = analyzer._process_slice(sample_labels_3d, metadata=metadata)
        assert isinstance(result, pd.DataFrame)
        assert "centroid-z" in result.columns
        assert "centroid-y" in result.columns
        assert "centroid-x" in result.columns
        assert "bbox-start-z" in result.columns
        assert "bbox-end-x" in result.columns

    def test_map_axes_disabled_keeps_numeric_suffixes(self, sample_labels_2d):
        """With map_axes=False, regionprops numeric column names are unchanged."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "object_id", "slice_id", "stack_id", "centroid", "bbox"],
            map_axes=False,
        )
        analyzer = RegionAnalyzer(config)
        metadata = {"axes": list("CZYX")}
        result = analyzer._process_slice(sample_labels_2d, metadata=metadata)
        assert isinstance(result, pd.DataFrame)
        assert "centroid-0" in result.columns
        assert "bbox-0" in result.columns
        assert "centroid-y" not in result.columns

    def test_run_assigns_one_stack_id_across_slices(self):
        """Each run gets one stack_id shared by all slices."""
        import pandas as pd

        labels = np.zeros((2, 10, 10), dtype=np.int32)
        labels[0, 2:5, 2:5] = 1
        labels[1, 4:7, 4:7] = 2
        config = RegionAnalyzerConfig(
            output_type="dataframe",
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer.run(labels)
        assert isinstance(result, pd.DataFrame)
        assert len(result["stack_id"].unique()) == 1
        assert len(result["slice_id"].unique()) == 2

    def test_extra_properties_includes_cross_sectional_area(self):
        """Ensure custom property registry exposes cross_sectional_area."""
        assert "cross_sectional_area" in RegionAnalyzer.extra_properties_funcs()

    def test_cross_sectional_area_3d_returns_plane_vector(self):
        """3D masks return one value per orthogonal plane (yz, xz, xy)."""
        mask = np.zeros((3, 4, 4), dtype=bool)
        mask[0, :2, :2] = True  # xy area 4
        mask[1, :3, :3] = True  # xy area 9
        mask[2, :, :] = True    # xy area 16
        result = RegionAnalyzer.cross_sectional_area(mask)
        assert result == (9.0, 9.0, 16.0)  # yz, xz, xy planes

    def test_cross_sectional_area_4d_sums_in_plane_maxes_all_ortho(self):
        """4D masks sum within each plane and max over every orthogonal axis."""
        mask = np.zeros((2, 3, 4, 4), dtype=bool)
        mask[0, 0, :2, :2] = True  # xy area 4 at c=0, z=0
        mask[1, 1, :3, :3] = True  # xy area 9 at c=1, z=1
        mask[1, 2, :, :] = True    # xy area 16 at c=1, z=2
        result = RegionAnalyzer.cross_sectional_area(mask)
        planes = RegionAnalyzer.cross_sectional_area_plane_indices(mask.ndim)
        # yx plane (axes 2, 3): max in-plane slice area is 16.
        assert result[planes.index((2, 3))] == 16.0
        # zy plane (axes 1, 2): max in-plane slice area is 7.
        assert result[planes.index((1, 2))] == 7.0

    def test_cross_sectional_area_2d_returns_scalar(self):
        """2D masks return a single scalar cross-sectional area."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        assert RegionAnalyzer.cross_sectional_area(mask) == 4.0

    def test_cross_sectional_area_with_spacing(self):
        """Spacing uses the two in-plane axis spacings for each component."""
        mask = np.zeros((2, 4, 4), dtype=bool)
        mask[:, :2, :2] = True  # xy area 4 per z slice
        spacing = (1.5, 0.8, 0.5)
        result = RegionAnalyzer.cross_sectional_area(mask, spacing=spacing)
        assert result[2] == 4 * (0.8 * 0.5)  # xy plane (y, x) for 3D

    def test_cross_sectional_area_2d_with_spacing(self):
        """2D spacing returns a single scaled scalar."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        spacing = (0.8, 0.5)
        assert RegionAnalyzer.cross_sectional_area(mask, spacing=spacing) == 4 * (0.8 * 0.5)

    def test_aspect_ratio_3d_returns_planes_and_overall(self):
        """3D aspect_ratio returns one value per plane plus an all-axis value."""
        mask = np.zeros((5, 6, 8), dtype=bool)
        mask[1:4, 1:5, 2:7] = True
        result = RegionAnalyzer.aspect_ratio(mask)
        assert len(result) == 4  # yz, xz, xy, overall

    def test_aspect_ratio_2d_returns_scalar(self):
        """2D aspect_ratio returns a single scalar."""
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:5, 2:7] = True
        result = RegionAnalyzer.aspect_ratio(mask)
        assert isinstance(result, float)
        assert 0.0 < result <= 1.0

    def test_map_axes_renames_aspect_ratio_planes_and_overall(self, sample_labels_3d):
        """aspect_ratio plane components and overall column names."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=[
                "label",
                "object_id",
                "slice_id",
                "stack_id",
                "aspect_ratio",
            ],
            map_axes=True,
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_3d, metadata={"axes": list("ZYX")})
        assert isinstance(result, pd.DataFrame)
        assert "aspect_ratio-yz" in result.columns
        assert "aspect_ratio-xz" in result.columns
        assert "aspect_ratio-xy" in result.columns
        assert "aspect_ratio" in result.columns

    def test_map_axes_renames_cross_sectional_area_planes(self, sample_labels_3d):
        """cross_sectional_area vector columns map to plane labels (yz, xz, xy)."""
        import pandas as pd

        config = RegionAnalyzerConfig(
            output_type="dataframe",
            properties=[
                "label",
                "object_id",
                "slice_id",
                "stack_id",
                "cross_sectional_area",
            ],
            map_axes=True,
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        )
        analyzer = RegionAnalyzer(config)
        result = analyzer._process_slice(sample_labels_3d, metadata={"axes": list("ZYX")})
        assert isinstance(result, pd.DataFrame)
        assert "cross_sectional_area-yz" in result.columns
        assert "cross_sectional_area-xz" in result.columns
        assert "cross_sectional_area-xy" in result.columns


class TestRegionFilterConfig:
    """Tests for RegionFilterConfig validation."""

    def test_filter_create_from_config_resolves_range_filter_config(self):
        """Filter.create_from_config maps RangeFilterConfig to RangeFilter."""
        from vistiq.segment.select import Filter

        cfg = RangeFilterConfig(attribute="volume", range=(0.0, float("inf")))
        resolved = Filter.create_from_config(cfg)
        assert isinstance(resolved, RangeFilter)
        assert resolved.config is cfg

    def test_filter_create_from_config_passes_through_filter_instance(self):
        """Filter.create_from_config returns an existing Filter unchanged."""
        from vistiq.segment.select import Filter

        existing = RangeFilter(
            RangeFilterConfig(attribute="volume", range=(1.0, 10.0))
        )
        assert Filter.create_from_config(existing) is existing

    def test_region_filter_run_accepts_bare_range_filter_config(self):
        """RegionFilter.run accepts filters as bare config objects."""
        import pandas as pd

        regions = pd.DataFrame({"label": [1, 2, 3], "volume": [500.0, 2000.0, 3000.0]})
        accepted, removed = RegionFilter(
            RegionFilterConfig(
                filters=[
                    RangeFilterConfig(attribute="volume", range=(1800.0, float("inf")))
                ]
            )
        ).run(regions)
        assert isinstance(accepted, pd.DataFrame)
        assert isinstance(removed, np.ndarray)
        assert list(accepted["label"]) == [2, 3]
        assert list(removed) == [1]

    def test_region_filter_chains_filters_with_and(self):
        """Multiple RangeFilters are AND-combined; a region must pass every filter."""
        import pandas as pd

        regions = pd.DataFrame(
            {
                "label": [1, 2, 3, 4],
                "volume": [500.0, 2000.0, 3000.0, 2500.0],
                "solidity": [0.9, 0.95, 0.5, 0.99],
            }
        )
        accepted, removed = RegionFilter(
            RegionFilterConfig(
                filters=[
                    RangeFilterConfig(attribute="volume", range=(1800.0, float("inf"))),
                    RangeFilterConfig(attribute="solidity", range=(0.9, 1.0)),
                ]
            )
        ).run(regions)
        assert list(accepted["label"]) == [2, 4]
        assert set(removed) == {1, 3}

    def test_region_filter_lookup_with_bare_configs(self):
        """has_filter/get_filter/get_attribute_names accept bare FilterConfig entries."""
        rf = RegionFilter(
            RegionFilterConfig(
                filters=[
                    RangeFilterConfig(attribute="volume", range=(0.0, float("inf"))),
                    RangeFilterConfig(attribute="solidity", range=(0.0, 1.0)),
                ]
            )
        )
        assert rf.has_filter("volume") is True
        assert rf.has_filter("missing") is False
        assert rf.get_attribute_names() == ["volume", "solidity"]
        resolved = rf.get_filter("volume")
        assert resolved.config.attribute == "volume"

    def test_filter_ops_and_with_bare_configs(self):
        """FilterOps combines bare filter configs."""
        from vistiq.segment.select import FilterOps, FilterOpsConfig

        values = np.array([10.0, 50.0, 90.0])
        result = FilterOps(
            FilterOpsConfig(
                filters=[
                    RangeFilterConfig(attribute=None, range=(40.0, 100.0)),
                    RangeFilterConfig(attribute=None, range=(0.0, 60.0)),
                ],
                operation="and",
            )
        ).run(values)
        np.testing.assert_array_equal(result, np.array([50.0]))

    def test_accepts_mapped_cross_sectional_area_columns(self):
        """RangeFilter may target map_axes plane columns."""
        config = RegionFilterConfig(
            filters=[
                RangeFilterConfig(
                    attribute="cross_sectional_area-xy", range=(100.0, float("inf"))
                ),
                RangeFilterConfig(
                    attribute="cross_sectional_area-xz", range=(50.0, float("inf"))
                ),
            ]
        )
        assert len(config.filters) == 2

    def test_accepts_mapped_centroid_and_bbox_columns(self):
        """RangeFilter may target other map_axes column names."""
        RegionFilterConfig(
            filters=[
                RangeFilterConfig(attribute="centroid-y", range=(0.0, 100.0)),
                RangeFilterConfig(attribute="bbox-end-x", range=(0.0, 512.0)),
            ]
        )

    def test_accepts_mapped_aspect_ratio_columns(self):
        """RangeFilter may target plane-specific and overall aspect_ratio columns."""
        RegionFilterConfig(
            filters=[
                RangeFilterConfig(attribute="aspect_ratio-xy", range=(0.5, 1.0)),
                RangeFilterConfig(attribute="aspect_ratio", range=(0.5, 1.0)),
            ]
        )

    def test_rejects_filter_instances_in_filters(self):
        """Filter configurables cannot be stored in RegionFilterConfig.filters."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be a FilterConfig subclass"):
            RegionFilterConfig(
                filters=[
                    RangeFilter(
                        RangeFilterConfig(attribute="volume", range=(0.0, float("inf")))
                    )
                ]
            )

    def test_rejects_unknown_attribute(self):
        """Invalid filter attributes still fail validation."""
        with pytest.raises(ValueError, match="not allowed"):
            RegionFilterConfig(
                filters=[
                    RangeFilterConfig(attribute="not_a_property", range=(0.0, 1.0))
                ]
            )


class TestRemapLabels:
    """Tests for remap_labels function."""

    def test_remap_labels_keep_zero_false(self):
        """Test remap_labels with background label 0 excluded from remapping."""
        # Input exactly as specified: labels[[2,4,8,5],[3,7,4,4],[0,1,5,3]]
        labels = np.array([[2,4,8,5],[3,7,4,4],[0,1,5,3]], dtype=np.int32)
        result, _ = remap_labels(labels, exclude=[0])
        
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
        """Test remap_labels remapping all labels including background."""
        # Input exactly as specified: labels[[2,4,8,5],[3,7,4,4],[0,1,5,3]]
        labels = np.array([[2,4,8,5],[3,7,4,4],[0,1,5,3]], dtype=np.int32)
        result, _ = remap_labels(labels, exclude=[])
        
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


class TestTopKFilter:
    """Tests for TopKFilter index selection."""

    def test_axis_none_global_smallest(self):
        """axis=None selects globally over a flattened array."""
        import torch
        from vistiq.matrix.types import FULL
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
        coords = TopKFilter(
            TopKFilterConfig(k=2, axis=None, largest=False, triangle=FULL)
        ).accept_indices(values)
        assert coords.shape == (2, 2)
        selected = sorted(tuple(int(v) for v in row) for row in coords)
        assert selected == [(0, 1), (1, 1)]

    def test_off_diagonal_rowwise_nearest(self):
        """OFF_DIAGONAL skips self-pairs on square distance matrices."""
        import torch
        from vistiq.matrix.types import OFF_DIAGONAL
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor(
            [
                [0.0, 5.0, 2.0],
                [5.0, 0.0, 4.0],
                [2.0, 4.0, 0.0],
            ]
        )
        coords = TopKFilter(
            TopKFilterConfig(
                k=1, axis=1, largest=False, triangle=OFF_DIAGONAL, ignore_nan=False
            )
        ).accept_indices(values)
        assert coords.shape == (3, 2)
        assert coords[0].tolist() == [0, 2]
        assert coords[1].tolist() == [1, 2]
        assert coords[2].tolist() == [2, 0]

    def test_ignore_nan(self):
        """ignore_nan excludes NaN entries from selection."""
        import torch
        from vistiq.segment.select import TopKFilter, TopKFilterConfig

        values = torch.tensor([float("nan"), 3.0, 1.0, 2.0])
        idx = TopKFilter(
            TopKFilterConfig(k=2, axis=None, largest=False, ignore_nan=True)
        ).accept_indices(values)
        np.testing.assert_array_equal(idx, [2, 3])

    def test_axis_none_off_diagonal(self):
        """axis=None with OFF_DIAGONAL excludes diagonal on square matrices."""
        import torch
        from vistiq.matrix.types import OFF_DIAGONAL
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor(
            [
                [0.0, 4.0],
                [3.0, 0.0],
            ]
        )
        coords = TopKFilter(
            TopKFilterConfig(
                k=1, axis=None, largest=False, triangle=OFF_DIAGONAL, ignore_nan=False
            )
        ).accept_indices(values)
        assert coords.shape == (1, 2)
        assert coords[0].tolist() == [1, 0]

    def test_output_values(self):
        """output='values' returns selected entries as a tensor."""
        import torch
        from vistiq.matrix.types import OFF_DIAGONAL
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor(
            [
                [0.0, 5.0, 2.0],
                [5.0, 0.0, 4.0],
                [2.0, 4.0, 0.0],
            ]
        )
        selected = TopKFilter(
            TopKFilterConfig(
                k=1,
                axis=1,
                largest=False,
                triangle=OFF_DIAGONAL,
                ignore_nan=False,
                output="values",
            )
        ).apply(values)
        assert isinstance(selected, torch.Tensor)
        assert selected.tolist() == [2.0, 4.0, 2.0]

    def test_output_mask(self):
        """output='mask' returns a boolean tensor with True at selected cells."""
        import torch
        from vistiq.segment.select import TopKFilter, TopKFilterConfig

        values = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
        mask = TopKFilter(
            TopKFilterConfig(k=1, axis=1, largest=False, output="mask")
        ).apply(values)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.tolist() == [[False, True], [False, True]]

    def test_output_masked_values(self):
        """output='masked_values' keeps matrix shape and NaN-fills unselected cells."""
        import torch
        from vistiq.segment.select import TopKFilter, TopKFilterConfig

        values = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
        masked = TopKFilter(
            TopKFilterConfig(k=1, axis=1, largest=False, output="masked_values")
        ).apply(values)
        assert isinstance(masked, torch.Tensor)
        assert masked.shape == values.shape
        assert masked[0, 1].item() == 1.0
        assert masked[1, 1].item() == 2.0
        assert torch.isnan(masked[0, 0])
        assert torch.isnan(masked[1, 0])


class TestValueFilter:
    """Tests for ValueFilter matrix thresholding."""

    def test_lte_mask_default_output(self):
        """Default output='mask' returns True where values <= threshold."""
        import torch
        from vistiq.segment import MatrixFilter, ValueFilter, ValueFilterConfig

        values = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        mask = ValueFilter(
            ValueFilterConfig(ref_value=2.5, operator="<=", output="mask")
        ).apply(values)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.tolist() == [[True, False], [False, True]]
        assert isinstance(ValueFilter(ValueFilterConfig(ref_value=1.0)), MatrixFilter)

    def test_off_diagonal(self):
        """OFF_DIAGONAL excludes self-pairs even when they pass the threshold."""
        import torch
        from vistiq.matrix.types import OFF_DIAGONAL
        from vistiq.segment import ValueFilter, ValueFilterConfig

        values = torch.tensor([[0.0, 4.0], [3.0, 0.0]])
        mask = ValueFilter(
            ValueFilterConfig(
                ref_value=1.0,
                operator="<=",
                triangle=OFF_DIAGONAL,
                ignore_nan=False,
                output="mask",
            )
        ).apply(values)
        assert mask.tolist() == [[False, False], [False, False]]

    def test_lower_triangle_only(self):
        """LOWER selects lower triangle including diagonal (i >= j)."""
        import torch
        from vistiq.matrix.types import LOWER
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor([[9.0, 1.0, 2.0], [3.0, 8.0, 4.0], [5.0, 6.0, 7.0]])
        mask = TopKFilter(
            TopKFilterConfig(k=10, axis=None, largest=False, triangle=LOWER, output="mask")
        ).apply(values)
        assert mask.tolist() == [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]

    def test_lower_triangle_rowwise(self):
        """LOWER with axis=1 does not zero the full matrix when one row lacks strict-lower cells."""
        import torch
        from vistiq.matrix.types import LOWER
        from vistiq.segment import TopKFilter, TopKFilterConfig

        dist = torch.tensor([[0.0, 5.0, 2.0], [5.0, 0.0, 4.0], [2.0, 4.0, 0.0]])
        masked = TopKFilter(
            TopKFilterConfig(
                k=1, axis=1, largest=False, triangle=LOWER, output="masked_values"
            )
        ).run(dist)
        assert masked[0, 0].item() == 0.0
        assert masked[1, 1].item() == 0.0
        assert masked[2, 2].item() == 0.0
        assert torch.isnan(masked[0, 1])

    def test_lower_nd_triangle(self):
        """LOWER_ND selects strict lower triangle (i > j) only."""
        import torch
        from vistiq.matrix.types import LOWER_ND
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor([[9.0, 1.0, 2.0], [3.0, 8.0, 4.0], [5.0, 6.0, 7.0]])
        mask = TopKFilter(
            TopKFilterConfig(k=10, axis=None, largest=False, triangle=LOWER_ND, output="mask")
        ).apply(values)
        assert mask.tolist() == [
            [False, False, False],
            [True, False, False],
            [True, True, False],
        ]

    def test_upper_nd_triangle(self):
        """UPPER_ND selects strict upper triangle (i < j) only."""
        import torch
        from vistiq.matrix.types import UPPER_ND
        from vistiq.segment import TopKFilter, TopKFilterConfig

        values = torch.tensor([[9.0, 1.0, 2.0], [3.0, 8.0, 4.0], [5.0, 6.0, 7.0]])
        mask = TopKFilter(
            TopKFilterConfig(k=10, axis=None, largest=False, triangle=UPPER_ND, output="mask")
        ).apply(values)
        assert mask.tolist() == [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]

    def test_masked_values_output(self):
        """output='masked_values' preserves passing entries only."""
        import torch
        from vistiq.segment import ValueFilter, ValueFilterConfig

        values = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        masked = ValueFilter(
            ValueFilterConfig(
                ref_value=2.5, operator="<=", output="masked_values"
            )
        ).apply(values)
        assert masked[0, 0].item() == 1.0
        assert masked[1, 1].item() == 2.0
        assert torch.isnan(masked[0, 1])
        assert torch.isnan(masked[1, 0])

    def test_matrix_data_masked_values_preserves_annotations(self):
        """MatrixData input keeps original annotations for masked_values."""
        import torch
        from vistiq.matrix import MatrixData
        from vistiq.segment import ValueFilter, ValueFilterConfig

        data = MatrixData(
            matrix=torch.tensor([[1.0, 5.0], [3.0, 2.0]]),
            annotations=(("r0", "r1"), ("c0", "c1")),
        )
        result = ValueFilter(
            ValueFilterConfig(ref_value=2.5, operator="<=", output="masked_values")
        ).run(data)
        assert isinstance(result, MatrixData)
        assert result.annotations == data.annotations
        assert result.matrix[0, 0].item() == 1.0
        assert result.matrix[1, 1].item() == 2.0
        assert torch.isnan(result.matrix[0, 1])

    def test_matrix_data_values_uses_composite_annotations(self):
        """MatrixData values output merges row/col labels with a separator."""
        import torch
        from vistiq.matrix import MatrixData
        from vistiq.segment import ValueFilter, ValueFilterConfig

        data = MatrixData(
            matrix=torch.tensor([[1.0, 5.0], [3.0, 2.0]]),
            annotations=(("r0", "r1"), ("c0", "c1")),
        )
        result = ValueFilter(
            ValueFilterConfig(ref_value=2.5, operator="<=", output="values")
        ).run(data)
        assert isinstance(result, MatrixData)
        assert result.matrix.tolist() == [1.0, 2.0]
        assert result.annotations == (("r0|c0", "r1|c1"),)

    def test_matrix_data_indices_returns_raw_coords(self):
        """MatrixData input with indices output still returns ndarray coordinates."""
        import torch
        from vistiq.matrix import MatrixData
        from vistiq.segment import ValueFilter, ValueFilterConfig

        data = MatrixData(
            matrix=torch.tensor([[1.0, 5.0], [3.0, 2.0]]),
            annotations=(("r0", "r1"), ("c0", "c1")),
        )
        result = ValueFilter(
            ValueFilterConfig(ref_value=2.5, operator="<=", output="indices")
        ).run(data)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [[0, 0], [1, 1]]
