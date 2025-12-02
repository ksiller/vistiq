"""Tests for vistiq.app module."""
import pytest
import logging
from vistiq.app import (
    configure_logger,
    parse_frames,
    AppConfig,
    SegmentConfig,
    AnalyzeConfig,
    PreprocessorConfig,
    CoincidenceConfig,
    FullConfig,
)


class TestConfigureLogger:
    """Tests for configure_logger function."""

    def test_configure_logger_default(self):
        """Test configure_logger with default parameters."""
        logger = configure_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.level <= logging.INFO

    def test_configure_logger_debug(self):
        """Test configure_logger with DEBUG level."""
        logger = configure_logger(level="DEBUG")
        assert logger.level <= logging.DEBUG

    def test_configure_logger_force(self):
        """Test configure_logger with force=True."""
        logger = configure_logger(force=True)
        assert isinstance(logger, logging.Logger)

    def test_configure_logger_warning(self):
        """Test configure_logger with WARNING level."""
        logger = configure_logger(level="WARNING")
        assert logger.level <= logging.WARNING


class TestParseFrames:
    """Tests for parse_frames function."""

    def test_parse_frames_none(self):
        """Test parse_frames with None."""
        start, end = parse_frames(None)
        assert start is None
        assert end is None

    def test_parse_frames_empty_string(self):
        """Test parse_frames with empty string."""
        start, end = parse_frames("")
        assert start is None
        assert end is None

    def test_parse_frames_single_frame(self):
        """Test parse_frames with single frame."""
        start, end = parse_frames("10")
        assert start == 9  # 1-based to 0-based
        assert end == 9

    def test_parse_frames_range(self):
        """Test parse_frames with range."""
        start, end = parse_frames("2-40")
        assert start == 1  # 1-based to 0-based
        assert end == 39

    def test_parse_frames_invalid_range(self):
        """Test parse_frames with invalid range."""
        with pytest.raises(ValueError):
            parse_frames("a-b")

    def test_parse_frames_invalid_single(self):
        """Test parse_frames with invalid single frame."""
        with pytest.raises(ValueError):
            parse_frames("abc")

    def test_parse_frames_negative(self):
        """Test parse_frames with negative frame."""
        with pytest.raises(ValueError):
            parse_frames("-1")

    def test_parse_frames_reversed_range(self):
        """Test parse_frames with reversed range."""
        with pytest.raises(ValueError):
            parse_frames("40-2")

    def test_parse_frames_zero(self):
        """Test parse_frames with zero."""
        with pytest.raises(ValueError):
            parse_frames("0")


class TestAppConfig:
    """Tests for AppConfig class."""

    def test_default_config(self):
        """Test default AppConfig."""
        config = AppConfig()
        assert config.input_path is None
        assert config.output_path is None

    def test_config_with_paths(self):
        """Test AppConfig with paths."""
        config = AppConfig(input_path="/path/to/input", output_path="/path/to/output")
        assert config.input_path == "/path/to/input"
        assert config.output_path == "/path/to/output"


class TestSegmentConfig:
    """Tests for SegmentConfig class."""

    def test_default_config(self):
        """Test default SegmentConfig."""
        config = SegmentConfig()
        assert config.input_path is None
        assert config.output_path is None

    def test_config_creation(self):
        """Test creating SegmentConfig."""
        config = SegmentConfig(input_path="/input", output_path="/output")
        assert config.input_path == "/input"
        assert config.output_path == "/output"


class TestAnalyzeConfig:
    """Tests for AnalyzeConfig class."""

    def test_default_config(self):
        """Test default AnalyzeConfig."""
        config = AnalyzeConfig()
        assert config.input_path is None
        assert config.output_path is None


class TestPreprocessorConfig:
    """Tests for PreprocessorConfig class."""

    def test_default_config(self):
        """Test default PreprocessorConfig."""
        config = PreprocessorConfig()
        assert config.input_path is None
        assert config.output_path is None


class TestCoincidenceConfig:
    """Tests for CoincidenceConfig class."""

    def test_default_config(self):
        """Test default CoincidenceConfig."""
        config = CoincidenceConfig()
        assert config.input_path is None
        assert config.output_path is None


class TestFullConfig:
    """Tests for FullConfig class."""

    def test_default_config(self):
        """Test default FullConfig."""
        config = FullConfig()
        assert config.input_path is None
        assert config.output_path is None

