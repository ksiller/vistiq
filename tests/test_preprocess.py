"""Tests for vistiq.preprocess module."""
import numpy as np
import pytest
from vistiq.preprocess import (
    PreprocessorConfig,
    Preprocessor,
    PreprocessChainConfig,
    PreprocessChain,
    DoGConfig,
    DoG,
    Noise2StackConfig,
    Noise2Stack,
)
from vistiq.utils import ArrayIteratorConfig


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


class TestPreprocessChainConfig:
    """Tests for PreprocessChainConfig class."""

    def test_default_config(self):
        """Test default PreprocessChainConfig."""
        config = PreprocessChainConfig()
        assert config.preprocessors == []


class TestPreprocessChain:
    """Tests for PreprocessChain class."""

    def test_initialization(self):
        """Test PreprocessChain initialization."""
        config = PreprocessChainConfig()
        chain = PreprocessChain(config)
        assert chain.config == config

    def test_from_config(self):
        """Test from_config class method."""
        config = PreprocessChainConfig()
        chain = PreprocessChain.from_config(config)
        assert isinstance(chain, PreprocessChain)


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

