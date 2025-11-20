"""Tests for vistiq.core module."""
import numpy as np
import pytest
from vistiq.core import (
    Configuration,
    Configurable,
    StackProcessorConfig,
    StackProcessor,
    ChainProcessorConfig,
    ChainProcessor,
)
from vistiq.utils import ArrayIteratorConfig


class TestConfiguration:
    """Tests for Configuration class."""

    def test_configuration_creation(self):
        """Test creating a basic Configuration."""
        config = Configuration()
        assert config.classname is None

    def test_configuration_with_classname(self):
        """Test Configuration with classname."""
        config = Configuration(classname="TestClass")
        assert config.classname == "TestClass"

    def test_configuration_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # Pydantic validation error
            Configuration(extra_field="not_allowed")


class TestStackProcessorConfig:
    """Tests for StackProcessorConfig class."""

    def test_default_config(self):
        """Test default StackProcessorConfig."""
        config = StackProcessorConfig()
        assert config.iterator_config is not None
        assert config.batch_size == 1
        assert config.output_type == "stack"
        assert config.squeeze is True

    def test_custom_config(self):
        """Test custom StackProcessorConfig."""
        iterator_config = ArrayIteratorConfig(slice_def=(0, 1))
        config = StackProcessorConfig(
            iterator_config=iterator_config,
            batch_size=5,
            output_type="list",
            squeeze=False,
        )
        assert config.batch_size == 5
        assert config.output_type == "list"
        assert config.squeeze is False


class ConcreteProcessor(StackProcessor):
    """Concrete implementation of StackProcessor for testing."""

    def _process_slice(self, slice, *args, metadata=None, **kwargs):
        """Process slice by doubling values."""
        return slice * 2


class TestStackProcessor:
    """Tests for StackProcessor class."""

    def test_initialization(self, stack_processor_config):
        """Test StackProcessor initialization."""
        processor = ConcreteProcessor(stack_processor_config)
        assert processor.config == stack_processor_config

    def test_from_config(self, stack_processor_config):
        """Test from_config class method."""
        processor = ConcreteProcessor.from_config(stack_processor_config)
        assert isinstance(processor, ConcreteProcessor)
        assert processor.config == stack_processor_config

    def test_process_single_slice(self, stack_processor_config, sample_2d_array):
        """Test processing a single slice."""
        processor = ConcreteProcessor(stack_processor_config)
        result = processor.run(sample_2d_array)
        np.testing.assert_array_equal(result, sample_2d_array * 2)

    def test_process_multiple_slices(self, sample_3d_array):
        """Test processing multiple slices."""
        config = StackProcessorConfig(
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1))
        )
        processor = ConcreteProcessor(config)
        result = processor.run(sample_3d_array, workers=1)
        assert result.shape == sample_3d_array.shape
        np.testing.assert_array_equal(result, sample_3d_array * 2)

    def test_process_with_metadata(self, stack_processor_config, sample_2d_array, mock_metadata):
        """Test processing with metadata."""
        processor = ConcreteProcessor(stack_processor_config)
        result = processor.run(sample_2d_array, metadata=mock_metadata)
        np.testing.assert_array_equal(result, sample_2d_array * 2)

    def test_process_with_kwargs(self, stack_processor_config, sample_2d_array):
        """Test processing with kwargs."""
        processor = ConcreteProcessor(stack_processor_config)
        result = processor.run(sample_2d_array, extra_param=42)
        np.testing.assert_array_equal(result, sample_2d_array * 2)

    def test_output_type_list(self, sample_3d_array):
        """Test output_type='list'."""
        config = StackProcessorConfig(
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
            output_type="list",
        )
        processor = ConcreteProcessor(config)
        result = processor.run(sample_3d_array, workers=1)
        assert isinstance(result, list)
        assert len(result) == sample_3d_array.shape[0]

    def test_squeeze_output(self, sample_3d_array):
        """Test squeeze output option."""
        config = StackProcessorConfig(
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
            squeeze=True,
        )
        processor = ConcreteProcessor(config)
        result = processor.run(sample_3d_array, workers=1)
        # When squeeze=True and single slice, should be squeezed
        if sample_3d_array.shape[0] == 1:
            assert result.ndim < sample_3d_array.ndim


class TestChainProcessorConfig:
    """Tests for ChainProcessorConfig class."""

    def test_default_config(self):
        """Test default ChainProcessorConfig."""
        config = ChainProcessorConfig()
        assert config.processors == []

    def test_config_with_processors(self, stack_processor_config):
        """Test ChainProcessorConfig with processors."""
        processor1 = ConcreteProcessor(stack_processor_config)
        processor2 = ConcreteProcessor(stack_processor_config)
        config = ChainProcessorConfig(processors=[processor1, processor2])
        assert len(config.processors) == 2


class TestChainProcessor:
    """Tests for ChainProcessor class."""

    def test_initialization(self, stack_processor_config):
        """Test ChainProcessor initialization."""
        processor1 = ConcreteProcessor(stack_processor_config)
        processor2 = ConcreteProcessor(stack_processor_config)
        config = ChainProcessorConfig(processors=[processor1, processor2])
        chain = ChainProcessor(config)
        assert len(chain.config.processors) == 2

    def test_from_config(self, stack_processor_config):
        """Test from_config class method."""
        processor1 = ConcreteProcessor(stack_processor_config)
        processor2 = ConcreteProcessor(stack_processor_config)
        config = ChainProcessorConfig(processors=[processor1, processor2])
        chain = ChainProcessor.from_config(config)
        assert isinstance(chain, ChainProcessor)

    def test_run_chain(self, stack_processor_config, sample_2d_array):
        """Test running a chain of processors."""
        processor1 = ConcreteProcessor(stack_processor_config)
        processor2 = ConcreteProcessor(stack_processor_config)
        config = ChainProcessorConfig(processors=[processor1, processor2])
        chain = ChainProcessor(config)
        result = chain.run(sample_2d_array)
        # Each processor doubles, so result should be 4x original
        np.testing.assert_array_equal(result, sample_2d_array * 4)

    def test_run_chain_with_metadata(self, stack_processor_config, sample_2d_array, mock_metadata):
        """Test running chain with metadata."""
        processor1 = ConcreteProcessor(stack_processor_config)
        processor2 = ConcreteProcessor(stack_processor_config)
        config = ChainProcessorConfig(processors=[processor1, processor2])
        chain = ChainProcessor(config)
        result = chain.run(sample_2d_array, metadata=mock_metadata)
        np.testing.assert_array_equal(result, sample_2d_array * 4)

