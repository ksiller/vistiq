"""Workflow builder for modular CLI workflows with dependency injection.

This module provides functionality to build workflows from Configurable components
specified via command-line arguments.
"""

import argparse
import inspect
import logging
from typing import Dict, Type, Any, Optional, List, Tuple, get_type_hints, get_origin, get_args
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from vistiq.core import Configuration, Configurable

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for Configurable classes and their Configuration classes."""
    
    def __init__(self):
        self._configurable_classes: Dict[str, Type[Configurable]] = {}
        self._config_classes: Dict[str, Type[Configuration]] = {}
        self._config_to_configurable: Dict[Type[Configuration], Type[Configurable]] = {}
    
    def register(self, configurable_class: Type[Configurable], config_class: Type[Configuration]):
        """Register a Configurable class with its Configuration class.
        
        Args:
            configurable_class: The Configurable class to register.
            config_class: The corresponding Configuration class.
        """
        configurable_name = configurable_class.__name__
        config_name = config_class.__name__
        
        self._configurable_classes[configurable_name] = configurable_class
        self._config_classes[config_name] = config_class
        self._config_to_configurable[config_class] = configurable_class
    
    def get_configurable(self, config_class: Type[Configuration]) -> Optional[Type[Configurable]]:
        """Get the Configurable class for a given Configuration class.
        
        Args:
            config_class: The Configuration class.
            
        Returns:
            The corresponding Configurable class, or None if not found.
        """
        return self._config_to_configurable.get(config_class)
    
    def get_config_class(self, name: str) -> Optional[Type[Configuration]]:
        """Get a Configuration class by name.
        
        Args:
            name: Name of the Configuration class.
            
        Returns:
            The Configuration class, or None if not found.
        """
        return self._config_classes.get(name)
    
    def get_configurable_class(self, name: str) -> Optional[Type[Configurable]]:
        """Get a Configurable class by name.
        
        Args:
            name: Name of the Configurable class.
            
        Returns:
            The Configurable class, or None if not found.
        """
        return self._configurable_classes.get(name)
    
    def list_configs(self) -> List[str]:
        """List all registered Configuration class names.
        
        Returns:
            List of Configuration class names.
        """
        return list(self._config_classes.keys())
    
    def list_configurables(self) -> List[str]:
        """List all registered Configurable class names.
        
        Returns:
            List of Configurable class names.
        """
        return list(self._configurable_classes.keys())


# Global registry instance
_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry.
    
    Returns:
        The global ComponentRegistry instance.
    """
    return _registry


class ConfigArgumentBuilder:
    """Builder for creating argparse arguments from Pydantic Configuration models."""
    
    @staticmethod
    def _get_field_type(field_info: FieldInfo, field_name: str) -> Tuple[Type, bool]:
        """Get the Python type for a Pydantic field.
        
        Args:
            field_info: Pydantic FieldInfo object.
            field_name: Name of the field.
            
        Returns:
            Tuple of (type, is_optional).
        """
        annotation = field_info.annotation
        if annotation is None:
            return str, False
        
        origin = get_origin(annotation)
        if origin is None:
            return annotation, False
        
        # Handle Optional, Union types
        if origin is type(None) or (hasattr(origin, '__origin__') and origin.__origin__ is type(None)):
            return str, True
        
        args = get_args(annotation)
        if args:
            # Check if None is in the union
            if type(None) in args:
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    return non_none_args[0], True
            return args[0], False
        
        return annotation, False
    
    @staticmethod
    def _get_default_value(field_info: FieldInfo) -> Any:
        """Get the default value for a field.
        
        Args:
            field_info: Pydantic FieldInfo object.
            
        Returns:
            Default value, or None if no default.
        """
        if hasattr(field_info, 'default'):
            default = field_info.default
            if default is not ...:
                return default
        return None
    
    @classmethod
    def add_config_arguments(
        cls,
        parser: argparse.ArgumentParser,
        config_class: Type[Configuration],
        prefix: str = "",
        required: bool = False
    ) -> argparse.ArgumentParser:
        """Add arguments to a parser based on a Configuration class.
        
        Args:
            parser: ArgumentParser to add arguments to.
            config_class: Configuration class to generate arguments from.
            prefix: Prefix to add to argument names (e.g., "preprocess.").
            required: Whether arguments are required.
            
        Returns:
            The modified ArgumentParser.
        """
        if not issubclass(config_class, BaseModel):
            return parser
        
        model_fields = config_class.model_fields
        
        # Get base Configuration class fields to skip
        from vistiq.core import Configuration as BaseConfiguration
        base_fields = set(BaseConfiguration.model_fields.keys()) if hasattr(BaseConfiguration, 'model_fields') else set()
        
        # Track which arguments have been added to avoid conflicts
        added_args = set()
        for action in parser._actions:
            if action.option_strings:
                for opt in action.option_strings:
                    if opt.startswith('--'):
                        added_args.add(opt[2:])  # Remove '--' prefix
        
        for field_name, field_info in model_fields.items():
            # Skip base Configuration fields (like classname) to avoid conflicts
            if field_name in base_fields:
                continue
                
            arg_name = f"{prefix}{field_name}" if prefix else field_name
            arg_name = arg_name.replace("_", "-")
            
            # Skip if argument already exists
            if arg_name in added_args:
                logger.debug(f"Skipping duplicate argument: --{arg_name}")
                continue
            
            # Get field type and default
            field_type, is_optional = cls._get_field_type(field_info, field_name)
            default_value = cls._get_default_value(field_info)
            
            # Get description from Field
            description = None
            if hasattr(field_info, 'description'):
                description = field_info.description
            elif isinstance(field_info, FieldInfo) and field_info.description:
                description = field_info.description
            
            # Determine argument type and action
            if field_type == bool:
                # Boolean fields use store_true/store_false
                action = "store_true" if default_value is False else "store_false"
                parser.add_argument(
                    f"--{arg_name}",
                    action=action,
                    help=description or f"{field_name}",
                    required=False
                )
                added_args.add(arg_name)
            elif field_type == int:
                parser.add_argument(
                    f"--{arg_name}",
                    type=int,
                    default=default_value,
                    help=description or f"{field_name}",
                    required=required and default_value is None
                )
                added_args.add(arg_name)
            elif field_type == float:
                parser.add_argument(
                    f"--{arg_name}",
                    type=float,
                    default=default_value,
                    help=description or f"{field_name}",
                    required=required and default_value is None
                )
                added_args.add(arg_name)
            elif field_type == str or (hasattr(field_type, '__origin__') and 
                                       get_origin(field_type) is type):
                # String or Literal type
                if hasattr(field_type, '__args__'):
                    # Literal type - use choices
                    choices = field_type.__args__
                    parser.add_argument(
                        f"--{arg_name}",
                        type=str,
                        choices=choices,
                        default=default_value,
                        help=description or f"{field_name} (choices: {', '.join(map(str, choices))})",
                        required=required and default_value is None
                    )
                    added_args.add(arg_name)
                else:
                    parser.add_argument(
                        f"--{arg_name}",
                        type=str,
                        default=default_value,
                        help=description or f"{field_name}",
                        required=required and default_value is None
                    )
                    added_args.add(arg_name)
            elif is_optional:
                # Optional type - treat as string for now
                parser.add_argument(
                    f"--{arg_name}",
                    type=str,
                    default=default_value,
                    help=description or f"{field_name}",
                    required=False
                )
                added_args.add(arg_name)
            else:
                # Complex type - use string and parse later
                parser.add_argument(
                    f"--{arg_name}",
                    type=str,
                    default=default_value,
                    help=description or f"{field_name}",
                    required=required and default_value is None
                )
                added_args.add(arg_name)
        
        return parser
    
    @classmethod
    def build_config_from_args(
        cls,
        args: argparse.Namespace,
        config_class: Type[Configuration],
        prefix: str = ""
    ) -> Configuration:
        """Build a Configuration instance from parsed arguments.
        
        Args:
            args: Parsed arguments namespace.
            config_class: Configuration class to instantiate.
            prefix: Prefix used in argument names (e.g., "preprocess.").
            
        Returns:
            Configuration instance.
        """
        config_dict = {}
        model_fields = config_class.model_fields
        
        for field_name, field_info in model_fields.items():
            arg_name = f"{prefix}{field_name}" if prefix else field_name
            arg_name = arg_name.replace("_", "-")
            
            # Get value from args
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
            elif hasattr(args, field_name):
                value = getattr(args, field_name)
            else:
                # Use default
                default_value = cls._get_default_value(field_info)
                value = default_value
            
            # Handle None values for optional fields
            if value is None:
                field_type, is_optional = cls._get_field_type(field_info, field_name)
                if is_optional:
                    config_dict[field_name] = None
                    continue
            
            # Handle nested Configuration objects
            field_type, _ = cls._get_field_type(field_info, field_name)
            if inspect.isclass(field_type) and issubclass(field_type, Configuration):
                # Recursively build nested config
                nested_prefix = f"{prefix}{field_name}." if prefix else f"{field_name}."
                nested_config = cls.build_config_from_args(args, field_type, nested_prefix)
                config_dict[field_name] = nested_config
            else:
                config_dict[field_name] = value
        
        return config_class(**config_dict)


class WorkflowBuilder:
    """Builder for creating workflows from CLI-specified components."""
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """Initialize the workflow builder.
        
        Args:
            registry: Component registry to use. If None, uses global registry.
        """
        self.registry = registry or get_registry()
    
    def build_component(
        self,
        component_name: str,
        args: argparse.Namespace,
        prefix: Optional[str] = None
    ) -> Configurable:
        """Build a Configurable component from CLI arguments.
        
        Args:
            component_name: Name of the Configurable class.
            args: Parsed CLI arguments.
            prefix: Optional prefix for argument names (e.g., "preprocess.").
            
        Returns:
            Configured Configurable instance.
            
        Raises:
            ValueError: If component is not registered or cannot be built.
        """
        # Get config class name (typically ComponentNameConfig)
        config_class_name = f"{component_name}Config"
        config_class = self.registry.get_config_class(config_class_name)
        
        if config_class is None:
            raise ValueError(f"Configuration class '{config_class_name}' not found in registry")
        
        # Build config from args
        config_prefix = f"{prefix}." if prefix else ""
        config = ConfigArgumentBuilder.build_config_from_args(
            args, config_class, config_prefix
        )
        
        # Get configurable class
        configurable_class = self.registry.get_configurable(config_class)
        
        if configurable_class is None:
            raise ValueError(f"Configurable class '{component_name}' not found in registry")
        
        # Instantiate configurable
        return configurable_class(config)


def auto_register_configurables(modules: List[str]):
    """Automatically register Configurable classes from specified modules.
    
    This function scans modules for classes that:
    1. Inherit from Configurable
    2. Have a corresponding *Config class that inherits from Configuration
    
    Args:
        modules: List of module names to scan (e.g., ["vistiq.preprocess", "vistiq.seg"]).
    """
    import importlib
    
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            
            # Find all Configurable classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Configurable) and 
                    obj is not Configurable):
                    
                    # Look for corresponding Config class
                    config_name = f"{name}Config"
                    if hasattr(module, config_name):
                        config_class = getattr(module, config_name)
                        if inspect.isclass(config_class) and issubclass(config_class, Configuration):
                            _registry.register(obj, config_class)
                            logger.info(f"Registered {name} with {config_name}")
        except ImportError as e:
            logger.warning(f"Could not import module {module_name}: {e}")

