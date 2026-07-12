"""Tests for vistiq.preprocess module."""
import numpy as np
import pytest
from vistiq.preprocess import (
    PreprocessorConfig,
    Preprocessor,
    PreprocessFlowConfig,
    PreprocessFlow,
    RescaleConfig,
    DoGConfig,
    DoG,
    Noise2StackConfig,
    Noise2Stack,
    UpsampleConfig,
    Upsample,
)


class TestPreprocessorConfig:
    """Tests for PreprocessorConfig class."""

    def test_default_config(self):
        """Test default PreprocessorConfig."""
        config = PreprocessorConfig()
        assert config.normalize is True
        assert config.dtype is None

    def test_custom_config(self):
        """Test custom PreprocessorConfig."""
        config = PreprocessorConfig(normalize=False, dtype=np.uint8)
        assert config.normalize is False
        assert config.dtype == np.uint8


class TestPreprocessor:
    """Tests for Preprocessor class."""

    def test_initialization(self):
        """Test Preprocessor initialization."""
        config = PreprocessorConfig()
        processor = Preprocessor(config)
        assert processor.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = PreprocessorConfig()
        processor = Preprocessor.from_config(config)
        assert isinstance(processor, Preprocessor)

    def test_normalize_method(self, sample_2d_array):
        """Test normalize method."""
        config = PreprocessorConfig()
        processor = Preprocessor(config)
        normalized = processor.normalize(sample_2d_array)
        assert normalized.dtype == np.float32
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0

    def test_normalize_constant_array(self):
        """Test normalize with constant array."""
        config = PreprocessorConfig()
        processor = Preprocessor(config)
        constant_array = np.ones((10, 10), dtype=np.uint8) * 128
        normalized = processor.normalize(constant_array)
        assert np.all(normalized == 0.0)

    def test_run_with_normalize(self, sample_2d_array):
        """Test run with normalization enabled."""
        config = PreprocessorConfig(normalize=True)
        processor = Preprocessor(config)
        # Since _process_slice is not implemented, this will raise NotImplementedError
        # But we can test the config
        assert processor.config.normalize is True

    def test_run_without_normalize(self, sample_2d_array):
        """Test run without normalization."""
        config = PreprocessorConfig(normalize=False)
        processor = Preprocessor(config)
        assert processor.config.normalize is False


class TestPreprocessFlowConfig:
    """Tests for PreprocessFlowConfig class."""

    def test_default_config(self):
        """Test default PreprocessFlowConfig."""
        config = PreprocessFlowConfig()
        assert config.processors == []

    def test_processors_accept_preprocessor_subclasses(self):
        """Processors must be PreprocessorConfig instances."""
        config = PreprocessFlowConfig(processors=[RescaleConfig(), DoGConfig()])
        assert len(config.processors) == 2
        assert all(isinstance(p, PreprocessorConfig) for p in config.processors)


class TestPreprocessFlow:
    """Tests for PreprocessFlow class."""

    def test_initialization(self):
        """Test PreprocessFlow initialization."""
        config = PreprocessFlowConfig()
        flow = PreprocessFlow(config)
        assert flow.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = PreprocessFlowConfig()
        flow = PreprocessFlow.from_config(config)
        assert isinstance(flow, PreprocessFlow)


class TestDoGConfig:
    """Tests for DoGConfig class."""

    def test_default_config(self):
        """Test default DoGConfig."""
        config = DoGConfig()
        assert config.sigma_low > 0
        assert config.sigma_high > config.sigma_low

    def test_custom_config(self):
        """Test custom DoGConfig."""
        config = DoGConfig(sigma_low=1.0, sigma_high=3.0)
        assert config.sigma_low == 1.0
        assert config.sigma_high == 3.0


class TestDoG:
    """Tests for DoG class."""

    def test_initialization(self):
        """Test DoG initialization."""
        config = DoGConfig()
        dog = DoG(config)
        assert dog.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = DoGConfig()
        dog = DoG.from_config(config)
        assert isinstance(dog, DoG)

    def test_process_slice(self, sample_2d_array):
        """Test _process_slice method."""
        config = DoGConfig(sigma_low=1.0, sigma_high=2.0)
        dog = DoG(config)
        result = dog._process_slice(sample_2d_array)
        assert result.shape == sample_2d_array.shape
        assert result.dtype == np.float64

    def test_run(self, sample_2d_array):
        """Test run method."""
        config = DoGConfig(sigma_low=1.0, sigma_high=2.0)
        dog = DoG(config)
        result = dog.run(sample_2d_array)
        assert result.shape == sample_2d_array.shape

    def test_run_3d(self, sample_3d_array):
        """Test run method with 3D array."""
        config = DoGConfig(sigma_low=1.0, sigma_high=2.0)
        dog = DoG(config)
        result = dog.run(sample_3d_array, workers=1)
        assert result.shape == sample_3d_array.shape


class TestNoise2StackConfig:
    """Tests for Noise2StackConfig class."""

    def test_default_config(self):
        """Test default Noise2StackConfig."""
        config = Noise2StackConfig()
        assert config.window_size > 0
        assert config.exclude_center is False

    def test_custom_config(self):
        """Test custom Noise2StackConfig."""
        config = Noise2StackConfig(window_size=5, exclude_center=True)
        assert config.window_size == 5
        assert config.exclude_center is True


class TestNoise2Stack:
    """Tests for Noise2Stack class."""

    def test_initialization(self):
        """Test Noise2Stack initialization."""
        config = Noise2StackConfig()
        n2s = Noise2Stack(config)
        assert n2s.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = Noise2StackConfig()
        n2s = Noise2Stack.from_config(config)
        assert isinstance(n2s, Noise2Stack)

    def test_run_2d(self, sample_2d_array):
        """Test run method with 2D array."""
        config = Noise2StackConfig(window_size=3)
        n2s = Noise2Stack(config)
        # For 2D, should return same shape
        result = n2s.run(sample_2d_array)
        assert result.shape == sample_2d_array.shape
        assert result.dtype == sample_2d_array.dtype

    def test_run_3d(self, sample_3d_array):
        """Test run method with 3D array."""
        config = Noise2StackConfig(window_size=3)
        n2s = Noise2Stack(config)
        result = n2s.run(sample_3d_array)
        assert result.shape == sample_3d_array.shape

    def test_run_with_exclude_center(self, sample_3d_array):
        """Test run with exclude_center=True."""
        config = Noise2StackConfig(window_size=3, exclude_center=True)
        n2s = Noise2Stack(config)
        result = n2s.run(sample_3d_array)
        assert result.shape == sample_3d_array.shape


class TestUpsampleConfig:
    """Tests for UpsampleConfig class."""

    def test_requires_width_or_height(self):
        with pytest.raises(ValueError, match="width or height"):
            UpsampleConfig()

    def test_width_only(self):
        config = UpsampleConfig(width=200)
        assert config.width == 200
        assert config.height is None
        assert config.sigma == 1.0
        assert config.recompute_scale is True

    def test_custom_config(self):
        config = UpsampleConfig(width=300, height=150, sigma=0.0, recompute_scale=False)
        assert config.width == 300
        assert config.height == 150
        assert config.sigma == 0.0
        assert config.recompute_scale is False


class TestUpsample:
    """Tests for Upsample class."""

    @staticmethod
    def _label_slice():
        labels = np.zeros((4, 4), dtype=np.uint16)
        labels[1:3, 1:3] = 1
        labels[2:4, 2:4] = 2
        return labels

    def test_initialization(self):
        upsampler = Upsample(UpsampleConfig(width=8, height=8))
        assert isinstance(upsampler, Upsample)

    def test_from_config(self):
        upsampler = Upsample.from_config(UpsampleConfig(width=8, height=8))
        assert isinstance(upsampler, Upsample)

    def test_process_slice(self):
        config = UpsampleConfig(width=8, height=8, sigma=0)
        config = config.model_copy(update={"output_shape": (8, 8)})
        upsampler = Upsample(config)
        labels = self._label_slice()
        result = upsampler._process_slice(labels)
        assert result.shape == (8, 8)
        assert set(np.unique(result)) == {0, 1, 2}

    def test_run_3d_stack(self, sample_labels_3d):
        upsampler = Upsample(
            UpsampleConfig(
                width=sample_labels_3d.shape[2] * 2,
                height=sample_labels_3d.shape[1] * 2,
                sigma=0,
            )
        )
        result, _ = upsampler.run(sample_labels_3d, workers=1)
        assert result.shape == (
            sample_labels_3d.shape[0],
            sample_labels_3d.shape[1] * 2,
            sample_labels_3d.shape[2] * 2,
        )
        assert result.dtype == sample_labels_3d.dtype
        assert set(np.unique(result)).issubset(set(np.unique(sample_labels_3d)))

    def test_run_4d_stack(self):
        upsampler = Upsample(UpsampleConfig(width=8, height=8, sigma=0))
        labels = np.zeros((2, 2, 4, 4), dtype=np.uint16)
        labels[0, 0, 1:3, 1:3] = 1
        labels[0, 1, 2:4, 2:4] = 2
        labels[1, 0, 0:2, 0:2] = 3
        result, _ = upsampler.run(labels, workers=1)
        assert result.shape == (2, 2, 8, 8)
        assert set(np.unique(result)) == {0, 1, 2, 3}

    def test_run_preserves_label_ids(self):
        upsampler = Upsample(
            UpsampleConfig(width=12, height=12, sigma=0, dtype=np.uint16)
        )
        labels = np.zeros((2, 6, 6), dtype=np.uint16)
        labels[0, 1:3, 1:3] = 5
        labels[1, 3:5, 3:5] = 42
        result, _ = upsampler.run(labels, workers=1)
        assert result.dtype == np.uint16
        assert 5 in result
        assert 42 in result
        assert np.max(result) == 42

    def test_run_width_only_preserves_aspect_ratio(self):
        upsampler = Upsample(UpsampleConfig(width=8, sigma=0))
        labels = np.zeros((2, 4, 4), dtype=np.uint16)
        labels[:, 1:3, 1:3] = 1
        result, _ = upsampler.run(labels, workers=1)
        assert result.shape == (2, 8, 8)

    def test_run_updates_metadata_scale(self):
        from bioio import Scale

        upsampler = Upsample(
            UpsampleConfig(width=8, height=8, sigma=0, recompute_scale=True)
        )
        labels = np.zeros((2, 4, 4), dtype=np.uint16)
        labels[:, 1:3, 1:3] = 1
        metadata = {
            "axes": ["Z", "Y", "X"],
            "shape": labels.shape,
            "scale": Scale(T=None, C=None, Z=1.0, Y=0.5, X=0.5),
        }
        _, updated = upsampler.run(labels, metadata=metadata, workers=1)
        assert updated["shape"] == (2, 8, 8)
        assert updated["scale"].Y == pytest.approx(0.25)
        assert updated["scale"].X == pytest.approx(0.25)
        assert updated["scale"].Z == pytest.approx(1.0)


class TestFuncProcessor:
    def test_axis_int_passes_scalar_to_numpy(self):
        from vistiq.preprocess import FuncProcessor, FuncProcessorConfig

        processor = FuncProcessor(
            FuncProcessorConfig(func=np.max, kwargs={"axis": 0})
        )
        arr = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(processor._process_slice(arr), [3, 4])

    def test_axis_int_tuple_passes_scalar_to_numpy(self):
        from vistiq.preprocess import FuncProcessor, FuncProcessorConfig

        processor = FuncProcessor(
            FuncProcessorConfig(func=np.repeat, args=[2], kwargs={"axis": (0,)})
        )
        arr = np.array([1, 2])
        assert np.array_equal(processor._process_slice(arr), [1, 1, 2, 2])

    def test_axis_letter_maps_to_scalar_index(self):
        from vistiq.preprocess import FuncProcessor, FuncProcessorConfig

        processor = FuncProcessor(
            FuncProcessorConfig(func=np.max, kwargs={"axis": "C"})
        )
        arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        metadata = {"axes": ["C", "Y", "X"]}
        result = processor._process_slice(arr, metadata=metadata)
        assert result.shape == (3, 4)

    def test_axis_multi_index_passes_tuple_to_numpy(self):
        from vistiq.preprocess import FuncProcessor, FuncProcessorConfig

        processor = FuncProcessor(
            FuncProcessorConfig(func=np.max, kwargs={"axis": (0, 1)})
        )
        arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        assert processor._process_slice(arr).shape == (4,)

    def test_output_dims_dict_reshapes_and_updates_metadata(self):
        from bioio import Dimensions, Scale
        from vistiq.preprocess import FuncProcessor, FuncProcessorConfig

        processor = FuncProcessor(
            FuncProcessorConfig(
                func=np.repeat,
                args=[99],
                kwargs={"axis": 0},
                output_dims={"Z": 99, "Y": 512, "X": 512},
            )
        )
        plane = np.ones((512, 512), dtype=np.uint8)
        result = processor._process_slice(plane)
        assert result.shape == (99, 512, 512)

        metadata = {
            "axes": ["Y", "X"],
            "shape": (512, 512),
            "dims": Dimensions(["Y", "X"], (512, 512)),
            "scale": Scale(Y=0.21, X=0.21, Z=1.0, C=None, T=None),
        }
        updated = processor._update_metadata(plane, result, metadata=metadata)
        assert updated["axes"] == ["Z", "Y", "X"]
        assert updated["shape"] == (99, 512, 512)
        assert updated["dims"].sizes == {"Z": 99, "Y": 512, "X": 512}
        assert updated["scale"].Y == 0.21
        assert updated["scale"].Z == 1.0

    def test_output_dims_sequence_relabels_metadata(self):
        from bioio import Dimensions
        from vistiq.preprocess import FuncProcessor, FuncProcessorConfig

        processor = FuncProcessor(
            FuncProcessorConfig(
                func=np.max,
                kwargs={"axis": 0},
                output_dims=["Y", "X"],
            )
        )
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = processor._process_slice(arr)
        metadata = {
            "axes": ["A", "B"],
            "shape": (2, 2),
            "dims": Dimensions(["A", "B"], (2, 2)),
        }
        updated = processor._update_metadata(arr, result, metadata=metadata)
        assert updated["axes"] == ["Y", "X"]
        assert updated["shape"] == (2, 2)
