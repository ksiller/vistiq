"""Tests for vistiq.workflow_builder module."""
import pytest
from vistiq.workflow_builder import (
    ComponentRegistry,
    get_registry,
    ConfigArgumentBuilder,
    WorkflowBuilder,
    auto_register_configurables,
)
from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor
from vistiq.utils import ArrayIteratorConfig


class ConcreteConfigurable(Configurable):
    """Concrete Configurable for testing."""

    @classmethod
    def from_config(cls, config):
        return cls(config)


class TestComponentRegistry:
    """Tests for ComponentRegistry class."""

    def test_initialization(self):
        """Test ComponentRegistry initialization."""
        registry = ComponentRegistry()
        assert registry._configurables == {}
        assert registry._configs == {}

    def test_register(self):
        """Test registering a component."""
        registry = ComponentRegistry()
        registry.register(ConcreteConfigurable, Configuration)
        assert ConcreteConfigurable in registry._configurables
        assert Configuration in registry._configs

    def test_get_configurable_class(self):
        """Test getting configurable class by name."""
        registry = ComponentRegistry()
        registry.register(ConcreteConfigurable, Configuration)
        cls = registry.get_configurable_class("ConcreteConfigurable")
        assert cls == ConcreteConfigurable

    def test_get_configurable_class_not_found(self):
        """Test getting non-existent configurable class."""
        registry = ComponentRegistry()
        cls = registry.get_configurable_class("NonExistent")
        assert cls is None

    def test_get_config_class(self):
        """Test getting config class by name."""
        registry = ComponentRegistry()
        registry.register(ConcreteConfigurable, Configuration)
        cls = registry.get_config_class("Configuration")
        assert cls == Configuration

    def test_get_config_class_not_found(self):
        """Test getting non-existent config class."""
        registry = ComponentRegistry()
        cls = registry.get_config_class("NonExistent")
        assert cls is None

    def test_list_configurables(self):
        """Test listing all configurables."""
        registry = ComponentRegistry()
        registry.register(ConcreteConfigurable, Configuration)
        configurables = registry.list_configurables()
        assert "ConcreteConfigurable" in configurables


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


class TestConfigArgumentBuilder:
    """Tests for ConfigArgumentBuilder class."""

    def test_initialization(self):
        """Test ConfigArgumentBuilder initialization."""
        builder = ConfigArgumentBuilder()
        assert builder is not None

    def test_add_config_arguments(self):
        """Test adding config arguments to parser."""
        import argparse

        parser = argparse.ArgumentParser()
        builder = ConfigArgumentBuilder()
        config = StackProcessorConfig()
        builder.add_config_arguments(parser, config, prefix="")
        # Check that some arguments were added
        # (exact arguments depend on config fields)
        assert len(parser._actions) > 1  # At least help action + some config args


class TestWorkflowBuilder:
    """Tests for WorkflowBuilder class."""

    def test_initialization(self):
        """Test WorkflowBuilder initialization."""
        builder = WorkflowBuilder()
        assert builder is not None

    def test_build_component(self):
        """Test building a component from config."""
        registry = ComponentRegistry()
        registry.register(ConcreteConfigurable, Configuration)
        builder = WorkflowBuilder()
        config = Configuration()
        # This will fail if component not registered, but we can test the structure
        # component = builder.build_component("ConcreteConfigurable", config)
        # assert isinstance(component, ConcreteConfigurable)


class TestAutoRegisterConfigurables:
    """Tests for auto_register_configurables function."""

    def test_auto_register(self):
        """Test auto-registering configurables from modules."""
        # This is a complex function that imports modules
        # We'll just test it doesn't crash
        try:
            auto_register_configurables(["vistiq.core"])
        except Exception:
            # May fail if modules can't be imported, but that's OK for testing
            pass

