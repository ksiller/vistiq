"""Tests for vistiq.workflow module."""
import numpy as np
import pytest
from vistiq.workflow import (
    WorkflowStepConfig,
    WorkflowConfig,
    BaseClass,
    WorkflowStep,
    Workflow,
)


class TestWorkflowStepConfig:
    """Tests for WorkflowStepConfig class."""

    def test_config_creation(self):
        """Test creating WorkflowStepConfig."""
        config = WorkflowStepConfig()
        assert config is not None


class TestWorkflowConfig:
    """Tests for WorkflowConfig class."""

    def test_config_creation(self):
        """Test creating WorkflowConfig."""
        step_configs = [WorkflowStepConfig()]
        config = WorkflowConfig(step_configs=step_configs)
        assert len(config.step_configs) == 1


class TestBaseClass:
    """Tests for BaseClass."""

    def test_initialization(self):
        """Test BaseClass initialization."""
        config = WorkflowStepConfig()
        base = BaseClass(config)
        assert base.config == config

    def test_name(self):
        """Test name method."""
        config = WorkflowStepConfig()
        base = BaseClass(config)
        assert base.name() == "BaseClass"

    def test_get_config(self):
        """Test get_config method."""
        config = WorkflowStepConfig()
        base = BaseClass(config)
        assert base.get_config() == config

    def test_set_config(self):
        """Test set_config method."""
        config1 = WorkflowStepConfig()
        config2 = WorkflowStepConfig()
        base = BaseClass(config1)
        base.set_config(config2)
        assert base.config == config2

    def test_str_repr(self):
        """Test __str__ and __repr__ methods."""
        config = WorkflowStepConfig()
        base = BaseClass(config)
        str_repr = str(base)
        assert "BaseClass" in str_repr
        repr_str = repr(base)
        assert "BaseClass" in repr_str

    def test_run_not_implemented(self, sample_2d_array):
        """Test that run raises NotImplementedError."""
        config = WorkflowStepConfig()
        base = BaseClass(config)
        with pytest.raises(NotImplementedError):
            base.run(sample_2d_array)


class TestWorkflowStep:
    """Tests for WorkflowStep class."""

    def test_initialization(self):
        """Test WorkflowStep initialization."""
        config = WorkflowStepConfig()
        step = WorkflowStep(config)
        assert step.config == config

    def test_run_not_implemented(self, sample_2d_array):
        """Test that run raises NotImplementedError."""
        config = WorkflowStepConfig()
        step = WorkflowStep(config)
        with pytest.raises(NotImplementedError):
            step.run(sample_2d_array)


class TestWorkflow:
    """Tests for Workflow class."""

    def test_initialization(self):
        """Test Workflow initialization."""
        step_configs = [WorkflowStepConfig()]
        config = WorkflowConfig(step_configs=step_configs)
        workflow = Workflow(config)
        assert workflow.config == config

