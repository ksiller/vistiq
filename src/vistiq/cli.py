"""CLI configuration models for vistiq command-line interface."""

import sys
import logging
import json
import numpy as np
import os
from pathlib import Path
from typing import Optional, Literal, List, Union, Any
from pydantic import BaseModel, Field, field_validator
import typer
from typer import Option, Argument, Context
from prefect import flow
from prefect.artifacts import create_progress_artifact, update_progress_artifact
try:
    from bioio_ome_tiff.writers import OmeTiffWriter
    OME_TIFF_AVAILABLE = True
except ImportError:
    OME_TIFF_AVAILABLE = False
    OmeTiffWriter = None

from .core import StackProcessorConfig, cli_config, Configuration, Configurable 
from .io import DataLoaderConfig, ImageLoaderConfig, ImageLoader,ImageWriter, FileListConfig, FileList, DataWriterConfig, ImageWriterConfig
from .preprocess import PreprocessorConfig
from .seg import ThresholderConfig, SegmenterConfig, LabellerConfig
from .train import TrainerConfig, DatasetCreatorConfig, DatasetCreator, MicroSAMTrainerConfig, MicroSAMTrainer
from .workflow_builder import ConfigArgumentBuilder, WorkflowBuilder, get_registry, auto_register_configurables
from .utils import load_image, get_scenes
# from .app import configure_logger

logger = logging.getLogger(__name__)


def _register_all_configurables() -> None:
    """Register all Configurable classes from all modules in a single place.
    
    This function should be called once at module import time to ensure all
    Configurable classes and their Configuration classes are available in the registry.
    """
    auto_register_configurables(Configurable, [
        "vistiq.io",          # FileList, DataLoader, ImageLoader, DataWriter, ImageWriter
        "vistiq.preprocess",  # Preprocessor classes
        "vistiq.seg",         # Segmenter classes
        "vistiq.analysis",    # Analysis classes
        "vistiq.train",       # Trainer classes
        "vistiq.core",        # Core classes
    ])


# Register all configurables once at module import time
_register_all_configurables()


def parse_substack(substack: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """Parse substack string into start/end frame indices (legacy format).

    Args:
        substack: Substack specification string like '10' or '2-40' (1-based, inclusive).

    Returns:
        Tuple of (start_frame, end_frame) as zero-based indices, or (None, None) if not specified.

    Raises:
        ValueError: If substack string is invalid.
    """
    if not substack:
        return None, None

    fs = str(substack).strip()
    if "-" in fs:
        a, b = fs.split("-", 1)
        if not a.isdigit() or not b.isdigit():
            raise ValueError(
                "--substack must be positive integers like '10' or '2-40'"
            )
        a_i = int(a)
        b_i = int(b)
        if a_i < 1 or b_i < 1:
            raise ValueError("--substack indices are 1-based and must be >= 1")
        if b_i < a_i:
            raise ValueError("--substack end must be >= start")
        # Convert to zero-based inclusive indices
        return a_i - 1, b_i - 1
    else:
        if not fs.isdigit():
            raise ValueError("--substack must be a positive integer or a range 'A-B'")
        n = int(fs)
        if n < 1:
            raise ValueError("--substack index is 1-based and must be >= 1")
        start_frame = n - 1
        return start_frame, start_frame


class CLIAppConfig(BaseModel):
    """Base configuration for all vistiq commands.
    
    Provides common configuration options shared across all subcommands,
    including logging, device selection, and process management.
    
    Attributes:
        loglevel: Logging level for the application (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        device: Device to use for processing (cuda, mps, cpu, or auto).
        processes: Number of processes to use for processing. Defaults to 1.
    """

    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    device: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto", description="Device to use for processing"
    )
    processes: Optional[int] = Field(
        default=1, description="Number of processes to use for processing"
    )

    class Config:
        """Pydantic configuration.
        
        Configures Pydantic model behavior:
        - extra: "forbid" prevents extra fields from being accepted
        - validate_assignment: Validates fields when assigned
        - arbitrary_types_allowed: Allows arbitrary types (e.g., numpy arrays)
        """

        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
        arbitrary_types_allowed=True

class CLISubcommandConfig(CLIAppConfig):
    """Base configuration for all vistiq subcommands.
    
    Extends CLIAppConfig with subcommand-specific options including input/output
    configuration and processing step chains.
    
    Attributes:
        input: Configuration for collecting input file(s) or directories (FileListConfig).
        loader: Configuration for data loading (DataLoaderConfig). Defaults to DataLoaderConfig.
        output: Configuration for output data writing (DataWriterConfig). Defaults to None.
        step: List of processing step configurations to run in sequence.
    """
    input: Optional[FileListConfig] = Field(
        default=None, description="Configuration for collecting input file(s) or directories"
    )
    loader: Optional[DataLoaderConfig] = Field(
        default_factory=DataLoaderConfig, description="Configuration to specify the data loader to use"
    )
    output: Optional[DataWriterConfig] = Field(
        default=None, description="Configuration for output data writing"
    )
    #substack: Optional[str] = Field(
    #    default=None, description="Substack to process. Legacy: '10' or '2-40' (first axis). New: 'T:4-10,Z:2-20' (multiple dimensions)"
    #)
    step: Optional[list[Configuration]] = Field(
        default=None, description="Configs for chain of processing components to run"
    )

    @field_validator("step")
    @classmethod
    def validate_step(cls, v: Optional[list[Configuration]]) -> Optional[list[Configuration]]:
        """Validate that processing steps are valid.
        
        Args:
            v: Step configurations to validate.
            
        Returns:
            Validated list of step configurations, or None.
        """
        if v is None:
            return None
        # Validate that each item is a Configuration instance
        for item in v:
            if not isinstance(item, Configuration):
                raise ValueError(f"Each item in 'step' must be a Configuration instance, got {type(item)}")
        return v


class CLIPreprocessorConfig(CLISubcommandConfig):
    """Configuration for the preprocess subcommand.
    
    Defines parameters for preprocessing images including denoising, resizing, and other operations.
    Extends CLISubcommandConfig with preprocessing-specific configuration.
    
    Attributes:
        input: Configuration for collecting input file(s) or directories (FileListConfig).
        loader: Configuration for image loading (ImageLoaderConfig). Defaults to ImageLoaderConfig.
        step: List of preprocessing component configurations to run in sequence (PreprocessorConfig).
        output: Configuration for output image writing (ImageWriterConfig). Defaults to ImageWriterConfig.
    """
    loader: Optional[ImageLoaderConfig] = Field(
        default_factory=ImageLoaderConfig, description="Configuration for input data loading"
    )
    step: Optional[list[PreprocessorConfig]] = Field(
        default=None, description="Configs for chain of preprocessing components to run"
    )
    output: Optional[ImageWriterConfig] = Field(
        default_factory=ImageWriterConfig, description="Configuration for output data writing"
    )

    @field_validator("step")
    @classmethod
    def validate_step(cls, v: Optional[list[PreprocessorConfig]]) -> Optional[list[PreprocessorConfig]]:
        """Validate that steps are valid.
        
        Args:
            v: Steps to validate.
            
        Returns:
            Validated list of preprocessing step configurations, or None.
        """
        if v is None:
            return None
        # Validate that each item is a PreprocessorConfig instance
        for item in v:
            if not isinstance(item, PreprocessorConfig):
                raise ValueError(f"Each item in 'step' must be a PreprocessorConfig instance, got {type(item)}")
        return v

class CLISegmenterConfig(CLISubcommandConfig):
    """Configuration for the segment subcommand.
    
    Defines parameters for segmenting images including thresholding, segmentation, labelling, and other operations.
    Extends CLISubcommandConfig with segmenting-specific input configuration.
    
    Attributes:
        input: Configuration for collecting input file(s) or directories.
        step: List of thresholding/segmenting/labelling component configurations to run in sequence.
        output: Configuration for output data writing.
    """
    loader: Optional[ImageLoaderConfig] = Field(
        default_factory=ImageLoaderConfig, description="Configuration for input data loading"
    )
    step: Optional[list[StackProcessorConfig]] = Field(
        default=None, description="Configs for chain of thresholding/segmenting/labelling components to run"
    )
    output: Optional[ImageWriterConfig] = Field(
        default_factory=ImageWriterConfig, description="Configuration for output data writing"
    )

    @field_validator("step")
    @classmethod
    def validate_step(cls, v: Optional[List[StackProcessorConfig]]) -> Optional[List[StackProcessorConfig]]:
        """Validate that steps are valid.
        
        Args:
            v: Steps to validate.
            
        Returns:
            Validated list of segmenting step configurations, or None.
        """
        if v is None:
            return None
        # Validate that each item is one of the allowed config types
        allowed_types = (ThresholderConfig, SegmenterConfig, LabellerConfig)
        for item in v:
            if not isinstance(item, allowed_types):
                raise ValueError(f"Each item in 'step' must be a ThresholderConfig, SegmenterConfig, or LabellerConfig instance, got {type(item)}")
        return v

class CLITrainerConfig(CLISubcommandConfig):
    """Configuration for the train subcommand.
    
    Defines parameters for training a model including dataset creation, data loading, and training.
    Extends CLISubcommandConfig with training-specific configuration.
    
    Attributes:
        input: Configuration for collecting input image file(s) or directories (FileListConfig).
        labels: Configuration for collecting labelled image file(s) or directories (FileListConfig).
        loader: Configuration for image loading (ImageLoaderConfig). Defaults to ImageLoaderConfig.
        dataset: Configuration for dataset creation (DatasetCreatorConfig). Defaults to DatasetCreatorConfig.
        step: List of training component configurations to run in sequence (TrainerConfig).
        output: Configuration for output data writing (ImageWriterConfig). Defaults to ImageWriterConfig.
    """
    input: Optional[FileListConfig] = Field(
        default=None, description="Configuration for input images"
    )
    labels: Optional[FileListConfig] = Field(
        default=None, description="Configuration for labelled images"
    )
    loader: Optional[ImageLoaderConfig] = Field(
        default_factory=ImageLoaderConfig, description="Configuration for input data loading"
    )
    dataset: Optional[DatasetCreatorConfig] = Field(
        default_factory=DatasetCreatorConfig, description="Configuration to specify the dataset creator to split the data into training and validation sets"
    )
    step: Optional[list[Configuration]] = Field(
        default=None, description="Configs for chain of training components to run"
    )
    output: Optional[ImageWriterConfig] = Field(
        default_factory=ImageWriterConfig, description="Configuration for output data writing"
    )

    @field_validator("step")
    @classmethod
    def validate_step(cls, v: Optional[List[TrainerConfig]]) -> Optional[List[TrainerConfig]]:
        """Validate that steps are valid.
        
        Args:
            v: Steps to validate.
            
        Returns:
            Validated list of segmenting step configurations, or None.
        """
        if v is None:
            return None
        # Validate that each item is one of the allowed config types
        allowed_types = (TrainerConfig)
        for item in v:
            if not isinstance(item, allowed_types):
                raise ValueError(f"Each item in 'step' must be a ThresholderConfig, SegmenterConfig, or LabellerConfig instance, got {type(item)}")
        return v

# Rebuild models to resolve forward references and Literal types
# This is needed when using Literal types or forward references in nested configs
CLISubcommandConfig.model_rebuild()
CLIPreprocessorConfig.model_rebuild()


# Create Typer app
# Note: We use add_completion=False to avoid completion-related help text issues
# We also need to handle component arguments specially to avoid ambiguity
app = typer.Typer(
    name="vistiq",
    help="Turn complex imaging data into quantitative insight with modular, multi-step hierarchical segmentation and spatio-temporal analysis.",
    no_args_is_help=True,
    add_completion=False,  # Disable shell completion to avoid conflicts with dynamic component arguments
    chain=True,  # Allow chaining commands (though we don't use it, it helps with argument parsing)
)


def _infer_dim_order(ndim: int) -> str:
    """Infer dimension order string from number of dimensions.
    
    Args:
        ndim: Number of dimensions in the array.
        
    Returns:
        Dimension order string (e.g., "YX", "ZYX", "CZYX", "TCZYX").
    """
    if ndim == 2:
        return "YX"
    elif ndim == 3:
        return "ZYX"
    elif ndim == 4:
        return "CZYX"
    elif ndim == 5:
        return "TCZYX"
    else:
        # For other dimensions, use generic labels
        return "".join([chr(ord('A') + i) for i in range(ndim - 2)]) + "YX"


def substack_to_slices(substack: Optional[str]) -> Optional[dict[str, slice]]:
    """Convert substack string to a dictionary of dimension slices for load_image.
    
    Supports two formats:
    1. Legacy format: '10' or '2-40' (applied to first axis of the image)
    2. New format: 'T:4-10;Z:2-20' (multiple dimensions with explicit names)
    
    For legacy format, use None as the key to indicate "first dimension".
    load_image will detect the first dimension from image metadata and apply the slice.
    
    Args:
        substack: Substack specification string.
                 - Legacy: '10' or '2-40' (1-based, inclusive, applied to first axis)
                 - New: 'T:4-10;Z:2-20' (dimension:range pairs, semicolon-separated)
        
    Returns:
        Dictionary with dimension slices, or None if substack is not specified.
    """
    if not substack:
        return None
    
    substack = str(substack).strip()
    
    # Check if it's the new format (contains ':' and possibly multiple dimensions)
    if ":" in substack:
        # New format: T:4-10;Z:2-20
        def parse_dimension_range(range_str: str) -> tuple[int, int]:
            """Parse a single dimension range like '4-10' or '5' into start/end indices."""
            range_str = range_str.strip()
            if "-" in range_str:
                parts = range_str.split("-", 1)
                if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                    raise ValueError(f"Invalid range format: '{range_str}'. Expected 'A-B' or 'A'")
                start = int(parts[0])
                end = int(parts[1])
                if start < 1 or end < 1:
                    raise ValueError("Range indices are 1-based and must be >= 1")
                if end < start:
                    raise ValueError(f"Range end ({end}) must be >= start ({start})")
                return start - 1, end - 1
            else:
                if not range_str.isdigit():
                    raise ValueError(f"Invalid range format: '{range_str}'. Expected 'A-B' or 'A'")
                val = int(range_str)
                if val < 1:
                    raise ValueError("Range index is 1-based and must be >= 1")
                return val - 1, val - 1
        
        slices_dict = {}
        parts = [p.strip() for p in substack.split(";")]
        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid substack format: '{part}'. Expected 'DIM:RANGE'")
            dim_name, range_str = part.split(":", 1)
            dim_name = dim_name.strip().upper()
            if not dim_name:
                raise ValueError(f"Invalid dimension name in: '{part}'")
            
            start, end = parse_dimension_range(range_str)
            slices_dict[dim_name] = slice(start, end + 1)
        
        return slices_dict if slices_dict else None
    else:
        # Legacy format: '10' or '2-40' (applied to first axis)
        start, end = parse_substack(substack)
        if start is None or end is None:
            return None
        return {None: slice(start, end + 1)}


def build_component_chain(
    component_configs: Optional[list[Configuration]] = None,
) -> tuple[list[str], list[Configurable]]:
    """Build a chain of processing steps from their configuration objects.
    
    Takes a list of step configuration objects, looks up their corresponding
    Configurable classes from the registry, and instantiates them to create a
    processing chain.
    
    Args:
        component_configs: Optional list of step configuration objects.
                          If None or empty, returns empty lists.
        
    Returns:
        Tuple of (component_names, built_components) where:
        - component_names: List of step class names as strings
        - built_components: List of instantiated Configurable step objects
        
    Raises:
        Exception: If a step cannot be built from its configuration.
    """
    # Build workflow builder
    builder = WorkflowBuilder()
    registry = builder.registry
    
    component_names = []
    built_components = []
    if component_configs:
        for i, component_config in enumerate(component_configs):
            # Get the configurable class from the config class
            config_class = type(component_config)
            configurable_class = registry.get_configurable(config_class)
            
            if configurable_class is None:
                # Try to get by class name as fallback
                config_name = config_class.__name__
                component_name = config_name.replace('Config', '')
                configurable_class = registry.get_configurable_class(component_name)
            
            if configurable_class:
                try:
                    component = configurable_class(component_config)
                    built_components.append(component)
                    component_names.append(configurable_class.__name__)
                    logger.info(f"Built component {i+1}/{len(component_configs)}: {configurable_class.__name__}")
                except Exception as e:
                    logger.error(f"Failed to build component from config: {e}")
                    raise
            else:
                logger.warning(f"Could not find configurable class for config type {config_class.__name__}")
    
    return component_names, built_components



@app.callback(invoke_without_command=True)
def common_callback(
    ctx: Context,
    loglevel: str = Option("INFO", help="Logging level"),
    device: Literal["cuda", "mps", "cpu", "auto"] = Option("auto", help="Device to use for processing"),
    processes: Optional[int] = Option(1, help="Number of processes to use"),
) -> None:
    """Common options for all vistiq commands.
    
    This callback is executed before any subcommand and stores common configuration
    options (loglevel, device, processes) in the Typer context for subcommands to access.
    
    Args:
        ctx: Typer context object for sharing data between callbacks and commands.
        loglevel: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        device: Device to use for processing (cuda, mps, cpu, or auto).
        processes: Number of processes to use for processing.
    """
    # Configure logging as early as possible
    #configure_logger(loglevel.upper())
    
    # Store common config in context for subcommands to access
    # Use ctx.obj (recommended) and ctx.params (for compatibility)
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["loglevel"] = loglevel.upper()
    ctx.obj["device"] = device
    ctx.obj["processes"] = processes
    
    # Also store in ctx.params for backward compatibility
    if not hasattr(ctx, 'params') or ctx.params is None:
        ctx.params = {}
    ctx.params["loglevel"] = loglevel.upper()
    ctx.params["device"] = device
    ctx.params["processes"] = processes
    
    logger.debug(f"common_callback: Set context - loglevel={loglevel.upper()}, device={device}, processes={processes}")
    
    # If no command was provided, just return (don't execute anything)
    if ctx.invoked_subcommand is None:
        return
    

def cli_to_config(value: str, default_value_field:str="classname", alt_classname:str=None) -> Configuration:
    """Convert a CLI string (component name or JSON) to a Configuration.
    
    This function can handle two formats:
    1. Component name string (e.g., "DoG", "Resize") - creates a default config
    2. JSON string (e.g., '{"classname": "DoG", "sigma_low": 1.0}') - uses Pydantic's model_validate_json
    
    The actual configuration values can also be provided via component-specific 
    arguments (e.g., --component0-sigma-low).
    
    Args:
        value: Component name string or JSON string with classname and config values.
        
    Returns:
        Configuration instance for the specified component.
        
    Raises:
        ValueError: If the component name is not found in the registry or JSON is invalid.
    """
    logger.debug(f"cli_to_config received: {value!r}")
    
    if not value or not isinstance(value, str):
        raise ValueError(f"Component name must be a non-empty string, got: {value!r}")
    
    registry = get_registry()
    
    # Try to parse as JSON first
    config_dict = {}
    try:
        # Preprocess: fix invalid escape sequences like \ (backslash space) which are common
        # when users escape spaces in shell commands. In JSON, spaces don't need escaping.
        # Replace \ followed by space or other non-escape chars with just the char itself
        # This handles cases where shell escaping gets into JSON strings
        # Valid JSON escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
        import re
        fixed_value = re.sub(r'\\([^"\\/bfnrtu])', r'\1', value)
        
        parsed = json.loads(fixed_value)
        if isinstance(parsed, dict):
            config_dict = parsed.copy()  # Use a copy to avoid modifying the original
            logger.debug(f"Parsed JSON dict: {config_dict}")
        else:
            # JSON parsed but not a dict, treat as plain string
            config_dict = {default_value_field: value}
            logger.debug(f"JSON parsed but not a dict, treating as plain string")
    except (json.JSONDecodeError, ValueError) as e:
        # Not valid JSON, treat as plain component name string
        config_dict = {default_value_field: value}
        logger.debug(f"Not valid JSON (error: {e}), treating as plain string")
    
    # Set classname if alt_classname is provided (overwrites any existing classname in the dict)
    if alt_classname is not None:
        config_dict["classname"] = alt_classname
        logger.debug(f"Set classname to {alt_classname}")
    
    config_class_name = config_dict.get("classname")
    logger.debug(f"Final config_dict={config_dict}, config_class_name={config_class_name}")
    
    if config_class_name is None:
        raise ValueError("Config dict must contain 'classname' field")
    
    # Get the Config class from the registry
    config_class = registry.get_config_class(config_class_name)
    
    # If not found in registry, try to get it from configurable class name
    if config_class is None:
        # Try treating config_class_name as a configurable class name (e.g., "DoG" -> "DoGConfig")
        configurable_class = registry.get_configurable_class(config_class_name)
        if configurable_class is not None:
            configurable_name = configurable_class.__name__
            config_class_name = f"{configurable_name}Config"
            config_class = registry.get_config_class(config_class_name)
    
    if config_class is None:
        raise ValueError(f"Config class '{config_class_name}' not found in registry. Available configs: {registry.list_configs()}")
    
    try:
        # Use Pydantic's model_validate to create the config instance from dict
        logger.debug(f"Using config_class={config_class}, type={type(config_class)}")
        config = config_class.model_validate(config_dict)
        logger.debug(f"Created {config_class_name} instance using model_validate")
        return config
    except Exception as e:
        logger.error(f"Failed to create {config_class_name} using model_validate: {e}")
        raise ValueError(f"Failed to create configuration for '{config_class_name}': {e}") from e

def cli_to_filelist_config(value: str) -> FileListConfig:
    """Convert a CLI string (path or JSON config) to a FileListConfig.
    
    This function can handle:
    1. Simple path string (e.g., "path/to/file" or "~/test/")
    2. JSON configuration string (e.g., '{"paths": "path/to/dir", "include": "*.tif"}')
    
    Args:
        value: Path string or JSON configuration. Paths with spaces must be quoted in the shell.
        
    Returns:
        FileListConfig instance with the provided paths.
        
    Raises:
        ValueError: If the value cannot be parsed as a valid path or JSON config.
    """
    return cli_to_config(value, default_value_field="paths", alt_classname="FileListConfig")


def cli_to_imageloader_config(value: str) -> ImageLoaderConfig:
    """Convert a CLI string (component name or JSON config) to an ImageLoaderConfig.
    
    This function can handle:
    1. Component name string (e.g., "ImageLoader") - creates a default config
    2. JSON configuration string (e.g., '{"classname": "ImageLoader", "path": "path/to/file"}')
    
    Args:
        value: Component name string or JSON configuration. Paths with spaces should be quoted.
        
    Returns:
        ImageLoaderConfig instance.
        
    Raises:
        ValueError: If the value cannot be parsed as a valid config.
    """
    #substack_str = value.get("substack", None)
    # substack_slices = substack_to_slices(substack_str if substack_str else None)
    return cli_to_config(value, default_value_field="classname", alt_classname="ImageLoaderConfig")



def cli_to_component_config(value: str) -> Configuration:
    """Convert a CLI string (component name or JSON) to a Configuration.
    
    This function can handle two formats:
    1. Component name string (e.g., "DoG", "Resize", "OtsuThreshold") - creates a default config
    2. JSON string (e.g., '{"classname": "DoG", "sigma_low": 1.0}') - uses Pydantic's model_validate_json
    
    Args:
        value: Component name string or JSON string with classname and config values.
        
    Returns:
        Configuration instance for the specified component.
        
    Raises:
        ValueError: If the component name is not found in the registry or JSON is invalid.
    """
    return cli_to_config(value)


def cli_to_imagewriter_config(value: str) -> ImageWriterConfig:
    """Convert a CLI string (output path or JSON config) to an ImageWriterConfig.
    
    This function can handle:
    1. Simple path string (e.g., "path/to/output" or "~/results/")
    2. JSON configuration string (e.g., '{"path": "path/to/output", "format": "tif", "overwrite": false}')
    
    Args:
        value: Output path string or JSON configuration. Paths with spaces must be quoted in the shell.
        
    Returns:
        ImageWriterConfig instance with the provided path.
        
    Raises:
        ValueError: If the value cannot be parsed as a valid path or JSON config.
    """
    return cli_to_config(value, default_value_field="path", alt_classname="ImageWriterConfig")


def cli_command_config(
    ctx: Context,
    input: Optional[FileListConfig] = None,
    labels: Optional[FileListConfig] = None,
    dataset: Optional[DatasetCreatorConfig] = None,
    loader: Optional[DataLoaderConfig] = None,
    step: Optional[List[Configuration]] = None,
    output: Optional[DataWriterConfig] = None,
) -> dict:
    """Build configuration dictionary from CLI arguments.
    
    Helper function that extracts common options from Typer context and combines them
    with command-specific arguments to create a configuration dictionary for Pydantic models.
    
    Args:
        ctx: Typer context containing global options (loglevel, device, processes).
        input: Optional input file list configuration.
        labels: Optional labels file list configuration (for train command).
        dataset: Optional dataset creator configuration (for train command).
        loader: Optional data loader configuration.
        step: Optional list of processing step configurations.
        output: Optional output writer configuration.
        
    Returns:
        Dictionary containing configuration values ready for Pydantic model instantiation.
    """
    logger.info(f"CLI input: {input}")
    logger.info(f"CLI label: {labels}")
    logger.info(f"CLI dataset: {dataset}")
    logger.info(f"CLI loader: {loader}")
    logger.info(f"CLI component: {step}")
    logger.info(f"CLI output: {output}")

    # Get common options from context (set by common_callback)
    # Try ctx.obj first (recommended), then ctx.params (for compatibility)
    if ctx.obj is not None and isinstance(ctx.obj, dict):
        loglevel = ctx.obj.get("loglevel", "INFO")
        device = ctx.obj.get("device", "auto")
        processes = ctx.obj.get("processes", 1)
        logger.debug(f"Retrieved from ctx.obj: loglevel={loglevel}, device={device}, processes={processes}")
    elif hasattr(ctx, 'params') and ctx.params is not None:
        loglevel = ctx.params.get("loglevel", "INFO")
        device = ctx.params.get("device", "auto")
        processes = ctx.params.get("processes", 1)
        logger.debug(f"Retrieved from ctx.params: loglevel={loglevel}, device={device}, processes={processes}")
    else:
        logger.info("Context not available, using defaults")
        loglevel = "INFO"
        device = "auto"
        processes = 1
    
    # Validate device value matches Literal type
    valid_devices = ("cuda", "mps", "cpu", "auto")
    if device not in valid_devices:
        logger.warning(f"Invalid device value '{device}', defaulting to 'auto'")
        device = "auto"

    # Only pass loader/output if they're not None, otherwise use defaults from default_factory
    config_kwargs = {
        "input": input,
        "step": step,
        "loglevel": loglevel,
        "device": device,
        "processes": processes,
    }
    if labels is not None:
        config_kwargs["labels"] = labels
    if dataset is not None:
        config_kwargs["dataset"] = dataset
    if loader is not None:
        config_kwargs["loader"] = loader
    if output is not None:
        config_kwargs["output"] = output
    return config_kwargs


@app.command("preprocess")
def preprocess_cmd(
    ctx: Context,
    input: Optional[FileListConfig] = Option(None, "--input", "-i", help="Input file or directory configuration", parser=cli_to_filelist_config),
    loader: Optional[ImageLoaderConfig] = Option(None, "--loader", help="Configuration to specify the data loader to use", parser=cli_to_imageloader_config),
    step: List[PreprocessorConfig] = Option(None, "--step", "-s", help="Processing step/component to include (can be specified multiple times). Use --step NAME to add a step.", parser=cli_to_component_config),
    output: Optional[ImageWriterConfig] = Option(None, "--output", "-o", help="Output file or directory configuration", parser=cli_to_imagewriter_config),
) -> None:
    """Preprocess images with a chain of preprocessing steps.

    Processes images through a sequence of preprocessing steps (e.g., denoising, resizing).
    Each step can be specified multiple times using --step/-s and configured with step-specific 
    arguments using the --step{i}-* prefix pattern (e.g., --step0-sigma-low, --step1-width).
    
    Input and output can be specified as:
    - Simple path: --input path/to/file or --input '{"paths": "path/to/file"}'
    - JSON config: --input '{"paths": "path/to/dir", "include": "*.tif", "exclude": ["*.tmp"]}'
    - Simple path: --output path/to/output or --output '{"path": "path/to/output"}'
    
    Examples:
        # Simple usage with default DoG configuration
        vistiq preprocess -i input.tif -o output -s DoG
        
        # Multiple steps with JSON configuration
        vistiq preprocess -i input.tif -o output -s '{"classname":"Resize", "width":256, "height":256}' -s '{"classname": "DoG", "sigma_low": 10, "sigma_high": 20}'
        
        # Step name with default config
        vistiq preprocess -i input.tif -o output -s Resize -s DoG
    
    """
    # Create config from CLI arguments (labels and dataset are not used in preprocess)
    config_kwargs = cli_command_config(ctx, input=input, labels=None, dataset=None, loader=loader, step=step, output=output)
    config = CLIPreprocessorConfig(**config_kwargs)
    # Call run_preprocess with the complete CLIPreprocessorConfig
    run_preprocess(config)


@app.command("segment")
def segment_cmd(
    ctx: Context,
    input: Optional[FileListConfig] = Option(None, "--input", "-i", help="Input file or directory configuration", parser=cli_to_filelist_config),
    loader: Optional[ImageLoaderConfig] = Option(None, "--loader", help="Configuration to specify the data loader to use", parser=cli_to_imageloader_config),
    step: List[StackProcessorConfig] = Option(None, "--step", "-s", help="Processing step/component to include (can be specified multiple times). Use --step NAME to add a step.", parser=cli_to_component_config),
    output: Optional[ImageWriterConfig] = Option(None, "--output", "-o", help="Output file or directory configuration", parser=cli_to_imagewriter_config),
) -> None:
    """Segment and label images with a chain of processing steps.

    Processes images through a sequence of thresholding, segmentation, and labelling steps.
    Each step can be specified multiple times using --step/-s and configured with step-specific 
    arguments using the --step{i}-* prefix pattern (e.g., --step0-threshold, --step1-method).
    
    Input and output can be specified as:
    - Simple path: --input path/to/file or --input '{"paths": "path/to/file"}'
    - JSON config: --input '{"paths": "path/to/dir", "include": "*.tif", "exclude": ["*.tmp"]}'
    - Simple path: --output path/to/output or --output '{"path": "path/to/output"}'
    
    Examples:
        # Simple usage with default Otsu thresholding
        vistiq segment -i input.tif -o output -s OtsuThreshold
        
        # Multiple steps with JSON configuration
        vistiq segment -i '{"paths": "~/test/", "include": "*.tif"}' -o ~/output -s OtsuThreshold -s Watershed
        
        # Step with custom configuration via JSON
        vistiq segment -i input.tif -o output -s '{"classname": "OtsuThreshold", "threshold": 0.5}'
    
    """
    # Create config from CLI arguments (labels and dataset are not used in segment, loader is optional)
    config_kwargs = cli_command_config(ctx, input=input, labels=None, dataset=None, loader=loader, step=step, output=output)
    config = CLISegmenterConfig(**config_kwargs)
    # Call run_segment with the complete CLISegmenterConfig
    run_segment(config)

@app.command("train")
def train_cmd(
    ctx: Context,
    input: Optional[FileListConfig] = Option(None, "--input", "-i", help="Input file or directory configuration", parser=cli_to_filelist_config),
    labels: Optional[FileListConfig] = Option(None, "--labels", "-l", help="Input file or directory configuration", parser=cli_to_filelist_config),
    dataset: Optional[DatasetCreatorConfig] = Option(None, "--dataset-creator", "-d", help="Configuration to specify the dataset creator to use", parser=cli_to_component_config),
    loader: Optional[ImageLoaderConfig] = Option(None, "--loader", help="Configuration to specify the data loader to use", parser=cli_to_imageloader_config),
    step: List[TrainerConfig] = Option(None, "--step", "-s", help="Processing step/component to include (can be specified multiple times). Use --step NAME to add a step.", parser=cli_to_component_config),
    output: Optional[ImageWriterConfig] = Option(None, "--output", "-o", help="Output file or directory configuration", parser=cli_to_imagewriter_config),
) -> None:
    """Train a model with a chain of processing steps.

    Trains models (e.g., MicroSAM) using paired image and label datasets. The command handles
    dataset creation, splitting into training/validation sets, and model training.
    Each step can be specified multiple times using --step/-s and configured with step-specific 
    arguments using the --step{i}-* prefix pattern.
    
    Input and output can be specified as:
    - Simple path: --input path/to/file or --input '{"paths": "path/to/file"}'
    - JSON config: --input '{"paths": "path/to/dir", "include": "*.tif", "exclude": ["*.tmp"]}'
    - Same for --labels (required for training)
    - Simple path: --output path/to/output or --output '{"path": "path/to/output"}'
    
    Examples:
        # Train MicroSAM with image and label pairs
        vistiq train --input '{"paths":"~/test/", "include":"*Preprocessed_Red.tif", "exclude": "*training*"}' --labels '{"paths":"~/test/", "include":"*Labelled_Red.tif", "exclude": "*training*"}' --output ~/trained --step '{"classname": "MicroSAMTrainer"}'
        
        # With custom dataset creator configuration
        vistiq train -i '{"paths": "~/images/"}' -l '{"paths": "~/labels/"}' -o ~/trained -d '{"classname": "DatasetCreator", "random_samples": 10}' -s MicroSAMTrainer
    
    """
    # Create config from CLI arguments
    config_kwargs = cli_command_config(ctx, input=input, labels=labels, dataset=dataset, loader=loader, step=step, output=output)
    config = CLITrainerConfig(**config_kwargs)

    # properly connect components to dataset creator
    patterns = (config.input.include[0] if config.input.include is not None else None, config.labels.include[0] if config.labels.include is not None else None) 
    dc = config.dataset.copy(update={
        "patterns": patterns, 
        "out_path": config.output.path,
        "random_samples": 5,
        "remove_empty_labels": True})
    config = config.copy(update={"dataset": dc})

    # Call run_training with the complete CLITrainerConfig
    run_training(config)


@flow(name="vistiq.preprocess")
def run_preprocess(config: CLIPreprocessorConfig) -> None:
    """Run the preprocess command.
    
    Processes images through a chain of preprocessing steps, handling multiple
    scenes and channels. Saves preprocessed images as OME-TIFF files with metadata.

    Args:
        config: Preprocess configuration containing input/output configuration,
                step configurations, and processing parameters. The config includes
                loglevel, device, and processes from CLIAppConfig (inherited from
                CLIAppConfig via CLISubcommandConfig).
                
    The function:
    1. Builds processing step chain from configuration
    2. Loads images from input path (supports multiple scenes)
    3. Processes each channel through the step chain
    4. Saves preprocessed images to output directory with metadata
    """
    logger.info(f"Running preprocess command with config: {config}")
    
    # Build FileList from config and get files
    file_list_result = FileList(config.input).run()
    # Prefect tasks return the actual value, but ensure we have a list
    if isinstance(file_list_result, list):
        file_list = file_list_result
    else:
        # If Prefect wrapped it somehow, try to unwrap
        file_list = list(file_list_result) if hasattr(file_list_result, '__iter__') else [file_list_result]
    
    if not file_list:
        raise ValueError(f"No files found for input path: {config.input.paths}")
    
    first_file = file_list[0]
    logger.info(f"Found {len(file_list)} files, using the first one: {first_file}")

    # Build component chain from config
    component_names, built_components = build_component_chain(config.step)
    
    # Get absolute path of input and strip extension for output directory
    # first_file should already be a Path object from the validator
    if isinstance(first_file, Path):
        input_path_obj = first_file.resolve()
        input_path_str = str(first_file)
    else:
        # Fallback: convert to Path if somehow it's not
        input_path_obj = Path(first_file).resolve()
        input_path_str = str(first_file)
    
    # Determine output directory (defaults to current directory if not specified)
    if config.output and config.output.path:
        # config.output.path is already a Path object from the validator, but ~/ hasn't been expanded yet
        output_base = config.output.path.expanduser().resolve()
    else:
        output_base = Path.cwd().resolve()
    
    scenes = get_scenes(input_path_str)
    progress_id = create_progress_artifact(
        progress=0.0,
        description="Indicates the progress of processing image scenes.",
    )
    for f_idx, first_file in enumerate(file_list):
        for idx, sc in enumerate(scenes):
            logger.info(f"Processing scene: {sc}")
            # Load image with substack slicing if specified
            
            loader_config = config.loader.copy(update={"scene_index": idx})
            img, metadata = ImageLoader(loader_config).run(path=first_file)

            #img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.loader.rename_channel if config.loader else None)
            channel_names = metadata["channel_names"]
            
            ishape = str(img.shape).replace(" ", "")
            component_names_str = "-".join(component_names) if component_names else "none"
            output_dir = output_base / input_path_obj.stem / f"{sc}-{ishape}-{component_names_str}"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")

            logger.info(f"Image shape: {img.shape}, channel names: {channel_names}, metadata: {metadata}")

            result = img
            result_metadata = metadata
            # Use config.processes if available, otherwise default to 1 (single process)
            # Ensure workers is a positive integer, not -1 (which means "use all cores")
            workers = config.processes if config.processes is not None and config.processes > 0 else 1
            logger.info(f"Using workers={workers} (from config.processes={config.processes}) for component processing")
            for i, component in enumerate(built_components):
                logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()} with workers={workers}")
                result, result_metadata = component.run(result, workers=workers, metadata=result_metadata)
            
            result_metadata.update({"dim_order":_infer_dim_order(result.ndim)})
            logger.info(f"Result shape: {result.shape}, metadata: {result_metadata}")
            logger.info(f"Channel axis: {result_metadata.get('channel_axis', None)}")
            logger.info(f"result metadata: {result_metadata}")
            imgwriter = ImageWriter(config.output)
            #preprocessed = np.stack(result, axis=0)
            output_path = output_dir / f"Preprocessed.tif"
            #OmeTiffWriter.save(preprocessed, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=channel_names, dim_order=_infer_dim_order(preprocessed.ndim))
            imgwriter.run(result, output_path, metadata=result_metadata)
        update_progress_artifact(
            artifact_id=progress_id,
            progress=float(f_idx + 1) / len(file_list),
            description=f"Processed file {first_file.name}: number {f_idx + 1} of {len(file_list)}",
        )

@flow(name="vistiq.segment")
def run_segment(config: CLISegmenterConfig) -> None:
    """Run the segment command.
    
    Processes images through a chain of segmentation steps (thresholding, segmentation, labelling),
    handling multiple scenes and channels. Saves segmented images as OME-TIFF files with metadata.

    Args:
        config: Segment configuration containing input/output configuration,
                step configurations, and processing parameters. The config includes
                loglevel, device, and processes from CLIAppConfig (inherited from
                CLIAppConfig via CLISubcommandConfig).
                
    The function:
    1. Builds processing step chain from configuration
    2. Loads images from input path (supports multiple scenes)
    3. Processes each channel through the step chain
    4. Saves segmented images to output directory with metadata
    """
    logger.info(f"Running preprocess command with config: {config}")
    
    # Build FileList from config and get files
    file_list_result = FileList(config.input).run()
    # Prefect tasks return the actual value, but ensure we have a list
    if isinstance(file_list_result, list):
        file_list = file_list_result
    else:
        # If Prefect wrapped it somehow, try to unwrap
        file_list = list(file_list_result) if hasattr(file_list_result, '__iter__') else [file_list_result]
    
    if not file_list:
        raise ValueError(f"No files found for input path: {config.input.paths}")
    
    first_file = file_list[0]
    logger.info(f"Found {len(file_list)} files, using the first one: {first_file}")

    # Build component chain from config
    component_names, built_components = build_component_chain(config.step)
    
    # Get absolute path of input and strip extension for output directory
    # first_file should already be a Path object from the validator
    if isinstance(first_file, Path):
        input_path_obj = first_file.resolve()
        input_path_str = str(first_file)
    else:
        # Fallback: convert to Path if somehow it's not
        input_path_obj = Path(first_file).resolve()
        input_path_str = str(first_file)
    
    # Determine output directory (defaults to current directory if not specified)
    if config.output and config.output.path:
        # config.output.path is already a Path object from the validator, but ~/ hasn't been expanded yet
        output_base = config.output.path.expanduser().resolve()
    else:
        output_base = Path.cwd().resolve()
    
    scenes = get_scenes(input_path_str)
    progress_id = create_progress_artifact(
        progress=0.0,
        description="Indicates the progress of processing image scenes.",
    )
    for f_idx, first_file in enumerate(file_list):
        for idx, sc in enumerate(scenes):
            logger.info(f"Processing scene: {sc}")
            # Load image with substack slicing if specified
            
            loader_config = config.loader.copy(update={"scene_index": idx})
            img, metadata = ImageLoader(loader_config).run(path=first_file)

            #img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.loader.rename_channel if config.loader else None)
            channel_names = metadata["channel_names"]
            
            ishape = str(img.shape).replace(" ", "")
            component_names_str = "-".join(component_names) if component_names else "none"
            output_dir = output_base / input_path_obj.stem / f"{sc}-{ishape}-{component_names_str}"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")

            logger.info(f"Image shape: {img.shape}, channel names: {channel_names}, metadata: {metadata}")

            result = img
            result_metadata = metadata
            # Use config.processes if available, otherwise default to 1 (single process)
            # Ensure workers is a positive integer, not -1 (which means "use all cores")
            workers = config.processes if config.processes is not None and config.processes > 0 else 1
            logger.info(f"Using workers={workers} (from config.processes={config.processes}) for component processing")
            for i, component in enumerate(built_components):
                logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()} with workers={workers}")
                result, result_metadata = component.run(result, workers=workers, metadata=result_metadata)
            
            result_metadata.update({"dim_order":_infer_dim_order(result.ndim)})
            logger.info(f"Result shape: {result.shape}, metadata: {result_metadata}")
            logger.info(f"Channel axis: {result_metadata.get('channel_axis', None)}")
            logger.info(f"result metadata: {result_metadata}")
            imgwriter = ImageWriter(config.output)
            #preprocessed = np.stack(result, axis=0)
            output_path = output_dir / f"Preprocessed.tif"
            #OmeTiffWriter.save(preprocessed, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=channel_names, dim_order=_infer_dim_order(preprocessed.ndim))
            imgwriter.run(result, output_path, metadata=result_metadata)
        update_progress_artifact(
            artifact_id=progress_id,
            progress=float(f_idx + 1) / len(file_list),
            description=f"Processed file {first_file.name}: number {f_idx + 1} of {len(file_list)}",
        )

@flow(name="vistiq.train")
def run_training(config: CLITrainerConfig) -> None:
    """Run the train command.
    
    Trains a model using the input images and labelled images.
    """
    logger.info(f"Running train command with config: {config}")

    # Build FileList from config and get image files
    input_list_result = FileList(config.input).run()
    # Prefect tasks return the actual value, but ensure we have a list
    if isinstance(input_list_result, list):
        input_list = input_list_result
    else:
        # If Prefect wrapped it somehow, try to unwrap
        input_list = list(input_list_result) if hasattr(input_list_result, '__iter__') else [input_list_result]
    if not input_list:
        logger.warning(f"No files found for input path: {config.input.paths}")
        return

    # Build FileList from config and get labelled image files
    labels_list_result = FileList(config.labels).run()
    if isinstance(labels_list_result, list):
        labels_list = labels_list_result
    else:
        labels_list = list(labels_list_result) if hasattr(labels_list_result, '__iter__') else [labels_list_result]
    if not labels_list:
        raise ValueError(f"No files found for labels path: {config.labels.paths}")

    logger.info(input_list)
    logger.info(labels_list)
    dataset_creator = DatasetCreator(config.dataset)
    img_label_pairs = dataset_creator.run(input_list, labels_list)

    if len(img_label_pairs) == 0:
        logger.warning("No matching pairs found, exiting")
        return

    # Reformat pairs into separate lists for MicroSAMTrainer.run
    image_paths: List[str] = [str(img_path) for img_path, _ in img_label_pairs]
    label_paths: List[str] = [str(lbl_path) for _, lbl_path in img_label_pairs]

    # Create trainer and run
    # Create MicroSAMTrainerConfig from TrainConfig
    # MicroSAMTrainerConfig inherits from TrainerConfig, so we pass all fields
    # Note: device is already resolved in main() function (auto -> actual device)
    microsam_trainer_config = None
    for c in config.step:
        if isinstance(c, MicroSAMTrainerConfig):
            microsam_trainer_config = c.copy(update={"device": config.device})
            break
    if microsam_trainer_config is None:
        raise ValueError("No MicroSAMTrainerConfig found in 'step'")
    trainer = MicroSAMTrainer(microsam_trainer_config)
    trainer.run(image_paths, label_paths)

def main() -> None:
    """Entry point for the vistiq CLI using Typer.
    
    This function is called when vistiq is invoked from the command line.
    It delegates to the Typer app which handles argument parsing and command routing.
    """
    app()


if __name__ == "__main__":
    main()

