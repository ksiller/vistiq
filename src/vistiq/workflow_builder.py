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
        """Get a Configuration class by name (case-insensitive).
        
        Args:
            name: Name of the Configuration class.
            
        Returns:
            The Configuration class, or None if not found.
        """
        # Try exact match first
        if name in self._config_classes:
            return self._config_classes[name]
        # Try case-insensitive match
        name_lower = name.lower()
        for registered_name, config_class in self._config_classes.items():
            if registered_name.lower() == name_lower:
                return config_class
        return None
    
    def get_configurable_class(self, name: str) -> Optional[Type[Configurable]]:
        """Get a Configurable class by name (case-insensitive).
        
        Args:
            name: Name of the Configurable class.
            
        Returns:
            The Configurable class, or None if not found.
        """
        # Try exact match first
        if name in self._configurable_classes:
            return self._configurable_classes[name]
        # Try case-insensitive match
        name_lower = name.lower()
        for registered_name, configurable_class in self._configurable_classes.items():
            if registered_name.lower() == name_lower:
                return configurable_class
        return None
    
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
    
    def list_configurables_by_base_class(self, base_class: Type[Configurable]) -> List[str]:
        """List registered Configurable class names that inherit from a specific base class.
        
        Args:
            base_class: Base class to filter by.
            
        Returns:
            List of Configurable class names that inherit from the base class.
        """
        result = []
        for name, configurable_class in self._configurable_classes.items():
            if (inspect.isclass(configurable_class) and 
                issubclass(configurable_class, base_class) and 
                configurable_class is not base_class):
                result.append(name)
        return sorted(result)


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
            Default value, ... if no default (Pydantic sentinel), or None if explicitly None.
        """
        if hasattr(field_info, 'default'):
            default = field_info.default
            return default  # Return ... if no default, None if explicitly None, or the actual default
        return ...  # No default attribute means no default
    
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
            required: Whether arguments are required (only used if field has no default).
                     Individual fields determine their own required status based on defaults.
            
        Returns:
            The modified ArgumentParser.
        """
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
            
            # Check if field is marked for CLI exposure
            # Fields are included by default, unless explicitly excluded
            include_in_cli = True
            
            # Check class-level CLI config from @cli_config decorator
            if hasattr(config_class, '__cli_config__'):
                cli_config = config_class.__cli_config__
                if cli_config.get('include_only') is not None:
                    include_in_cli = field_name in cli_config['include_only']
                    if not include_in_cli:
                        logger.debug(f"Skipping field '{field_name}' - not in include_only list")
                elif cli_config.get('exclude') is not None:
                    include_in_cli = field_name not in cli_config['exclude']
                    if not include_in_cli:
                        logger.debug(f"Skipping field '{field_name}' - in exclude list")
            
            # Check field-level CLI marker in json_schema_extra
            if include_in_cli and hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                cli_flag = field_info.json_schema_extra.get('cli')
                if cli_flag is False:
                    include_in_cli = False
                    logger.debug(f"Skipping field '{field_name}' - marked cli=False")
            # Also check FieldInfo.extra (Pydantic v2)
            elif include_in_cli and hasattr(field_info, 'extra') and isinstance(field_info.extra, dict):
                cli_flag = field_info.extra.get('cli')
                if cli_flag is False:
                    include_in_cli = False
                    logger.debug(f"Skipping field '{field_name}' - marked cli=False")
            
            if not include_in_cli:
                continue
            
            # Get field type and default
            field_type, is_optional = cls._get_field_type(field_info, field_name)
            default_value = cls._get_default_value(field_info)
            
            # Handle nested Configuration fields - flatten them without prefix
            # Special handling for fields like input_config that should be flattened
            if inspect.isclass(field_type) and issubclass(field_type, Configuration):
                # For nested configs, flatten their fields into the parent parser without prefix
                # This makes fields like input_config.input_path available as --input-path directly
                # Don't pass required parameter - let each nested field determine its own required status
                # based on its own default value (handled inside add_config_arguments)
                cls.add_config_arguments(parser, field_type, prefix="")
                continue  # Skip adding the nested config itself as an argument
            
            arg_name = f"{prefix}{field_name}" if prefix else field_name
            arg_name = arg_name.replace("_", "-")
            
            # Skip if argument already exists
            if arg_name in added_args:
                logger.debug(f"Skipping duplicate argument: --{arg_name}")
                continue
            
            # Get description from Field
            description = None
            if hasattr(field_info, 'description'):
                description = field_info.description
            elif isinstance(field_info, FieldInfo) and field_info.description:
                description = field_info.description
            
            # Determine short flag for common arguments
            short_flag = None
            if field_name == "input_path" or arg_name == "input-path":
                short_flag = "-i"
            elif field_name == "output_path" or arg_name == "output-path":
                short_flag = "-o"
            elif field_name == "components" or arg_name == "component":
                short_flag = "-c"
            elif field_name == "substack":
                short_flag = "-f"
            elif field_name == "grayscale":
                short_flag = "-g"
            
            # Build option strings list
            option_strings = [f"--{arg_name}"]
            if short_flag and short_flag not in [opt for action in parser._actions for opt in (action.option_strings or [])]:
                option_strings.insert(0, short_flag)
            
            # Determine argument type and action
            if field_type == bool:
                # Boolean fields use store_true/store_false
                action = "store_true" if default_value is False else "store_false"
                parser.add_argument(
                    *option_strings,
                    action=action,
                    help=description or f"{field_name}",
                    required=False
                )
                added_args.add(arg_name)
            elif field_type == int:
                parser.add_argument(
                    *option_strings,
                    type=int,
                    default=default_value,
                    help=description or f"{field_name}",
                    required=required and default_value is None
                )
                added_args.add(arg_name)
            elif field_type == float:
                parser.add_argument(
                    *option_strings,
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
                        *option_strings,
                        type=str,
                        choices=choices,
                        default=default_value,
                        help=description or f"{field_name} (choices: {', '.join(map(str, choices))})",
                        required=required and default_value is None
                    )
                    added_args.add(arg_name)
                else:
                    parser.add_argument(
                        *option_strings,
                        type=str,
                        default=default_value,
                        help=description or f"{field_name}",
                        required=required and default_value is None
                    )
                    added_args.add(arg_name)
            elif is_optional:
                # Optional type - treat as string for now
                parser.add_argument(
                    *option_strings,
                    type=str,
                    default=default_value,
                    help=description or f"{field_name}",
                    required=False
                )
                added_args.add(arg_name)
            else:
                # Complex type - use string and parse later
                parser.add_argument(
                    *option_strings,
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
            # Special handling for 'components' field - skip it here, will be handled separately
            if field_name == "components":
                continue
            
            # Get field type to check if it's a nested Configuration
            field_type, _ = cls._get_field_type(field_info, field_name)
            
            # Handle nested Configuration objects - reconstruct from flattened args
            if inspect.isclass(field_type) and issubclass(field_type, Configuration):
                # For nested configs, reconstruct from flattened arguments (no prefix)
                # Fields from input_config are flattened directly, so we use empty prefix
                nested_config = cls.build_config_from_args(args, field_type, prefix="")
                config_dict[field_name] = nested_config
                continue
            
            # For regular fields, try both the prefixed and non-prefixed argument names
            arg_name = f"{prefix}{field_name}" if prefix else field_name
            arg_name_hyphen = arg_name.replace("_", "-").replace(".", "-")
            arg_name_underscore = arg_name.replace("-", "_").replace(".", "_")
            
            # Get value from args
            # Argparse stores arguments with underscores, so try underscore version first
            if hasattr(args, arg_name_underscore):
                value = getattr(args, arg_name_underscore)
            elif hasattr(args, arg_name_hyphen):
                value = getattr(args, arg_name_hyphen)
            elif hasattr(args, arg_name):
                value = getattr(args, arg_name)
            elif hasattr(args, field_name):
                value = getattr(args, field_name)
            else:
                # Use default if available
                default_value = cls._get_default_value(field_info)
                # If default is ... (Pydantic's sentinel for "no default"), skip this field
                # Pydantic will use parent class defaults or raise validation error if required
                if default_value is ...:
                    continue
                value = default_value
            
            # Handle None values for optional fields
            if value is None:
                _, is_optional = cls._get_field_type(field_info, field_name)
                if is_optional:
                    config_dict[field_name] = None
                    continue
            
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
            component_name: Name of the Configurable class (case-insensitive).
            args: Parsed CLI arguments.
            prefix: Optional prefix for argument names (e.g., "preprocess.").
            
        Returns:
            Configured Configurable instance.
            
        Raises:
            ValueError: If component is not registered or cannot be built.
        """
        # First, try to find the actual registered component name (case-insensitive)
        configurable_class = self.registry.get_configurable_class(component_name)
        
        if configurable_class is None:
            raise ValueError(f"Configurable class '{component_name}' not found in registry")
        
        # Get the actual registered name for building config class name
        actual_component_name = configurable_class.__name__
        config_class_name = f"{actual_component_name}Config"
        config_class = self.registry.get_config_class(config_class_name)
        
        if config_class is None:
            raise ValueError(f"Configuration class '{config_class_name}' not found in registry")
        
        # Build config from args
        config_prefix = f"{prefix}." if prefix else ""
        config = ConfigArgumentBuilder.build_config_from_args(
            args, config_class, config_prefix
        )
        
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


def auto_register_configurables_by_base_class(base_class: Type[Configurable], modules: Optional[List[str]] = None):
    """Automatically register Configurable classes that inherit from a specific base class.
    
    This function scans modules for classes that:
    1. Inherit from the specified base class (but are not the base class itself)
    2. Have a corresponding *Config class that inherits from Configuration
    
    Args:
        base_class: Base class to filter by (e.g., Preprocessor, Segmenter, Trainer).
        modules: Optional list of module names to scan. If None, will attempt to find
                modules containing the base class.
    """
    import importlib
    
    # If no modules specified, try to find the module containing the base class
    if modules is None:
        base_module = inspect.getmodule(base_class)
        if base_module is not None:
            modules = [base_module.__name__]
        else:
            logger.warning(f"Could not determine module for base class {base_class.__name__}")
            return
    
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            
            # Find all classes that inherit from the base class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (inspect.isclass(obj) and 
                    issubclass(obj, base_class) and 
                    obj is not base_class and
                    issubclass(obj, Configurable)):
                    
                    # Look for corresponding Config class
                    config_name = f"{name}Config"
                    if hasattr(module, config_name):
                        config_class = getattr(module, config_name)
                        if inspect.isclass(config_class) and issubclass(config_class, Configuration):
                            _registry.register(obj, config_class)
                            logger.info(f"Registered {name} (inherits from {base_class.__name__}) with {config_name}")
        except ImportError as e:
            logger.warning(f"Could not import module {module_name}: {e}")

