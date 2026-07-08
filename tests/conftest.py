"""Pytest configuration and shared fixtures."""
import logging
import os

# Prefect tasks run in unit tests; disable Cloud log shipping and avoid
# background threads logging to stdout after pytest teardown (wandb/napari
# console capture can close streams early).
os.environ.setdefault("PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW", "ignore")
os.environ.setdefault("PREFECT_LOGGING_TO_API_ENABLED", "false")
logging.getLogger("httpx").setLevel(logging.WARNING)

import numpy as np
import pytest
from prefect.testing.utilities import prefect_test_harness

from vistiq.core import Configuration, StackProcessorConfig
from vistiq.utils import ArrayIteratorConfig


@pytest.fixture(scope="session", autouse=True)
def _prefect_test_harness():
    with prefect_test_harness():
        yield


@pytest.fixture
def sample_2d_array():
    """Create a sample 2D numpy array for testing."""
    return np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def sample_3d_array():
    """Create a sample 3D numpy array for testing."""
    return np.random.randint(0, 255, size=(10, 100, 100), dtype=np.uint8)


@pytest.fixture
def sample_4d_array():
    """Create a sample 4D numpy array for testing."""
    return np.random.randint(0, 255, size=(5, 10, 100, 100), dtype=np.uint8)


@pytest.fixture
def sample_labels_2d():
    """Create a sample 2D labeled array for testing."""
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[40:60, 40:60] = 2
    labels[70:90, 70:90] = 3
    return labels


@pytest.fixture
def sample_labels_3d():
    """Create a sample 3D labeled array for testing."""
    labels = np.zeros((10, 100, 100), dtype=np.int32)
    labels[2:4, 10:30, 10:30] = 1
    labels[5:7, 40:60, 40:60] = 2
    return labels


@pytest.fixture
def basic_config():
    """Create a basic Configuration instance."""
    return Configuration()


@pytest.fixture
def stack_processor_config():
    """Create a StackProcessorConfig instance."""
    return StackProcessorConfig(
        iterator_config=ArrayIteratorConfig(slice_def=(-2, -1))
    )


@pytest.fixture
def mock_metadata():
    """Create mock metadata dictionary."""
    return {"test_key": "test_value", "pixel_size": 0.1}

