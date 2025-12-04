"""CLI configuration models for vistiq command-line interface."""

import sys
import logging
import json
import numpy as np
import os
from pathlib import Path
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
import typer
from typer import Option, Argument, Context

try:
    from bioio_ome_tiff.writers import OmeTiffWriter
    OME_TIFF_AVAILABLE = True
except ImportError:
    OME_TIFF_AVAILABLE = False
    OmeTiffWriter = None

from .core import cli_config, Configuration, Configurable 
from .io import DataLoaderConfig, ImageLoaderConfig, ImageLoader,ImageWriter, FileListConfig, FileList, DataWriterConfig, ImageWriterConfig
from .preprocess import PreprocessorConfig, Preprocessor
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
        input_config: Configuration for input data loading.
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
        return v


class CLIPreprocessorConfig(CLISubcommandConfig):
    """Configuration for the preprocess subcommand.
    
    Defines parameters for preprocessing images including denoising, resizing, and other operations.
    Extends CLISubcommandConfig with preprocessing-specific input configuration.
    
    Attributes:
        input_config: Configuration for input image loading (ImageLoaderConfig).
        component: List of preprocessing component configurations to run in sequence.
                  Note: This field is specific to CLIPreprocessorConfig, while the parent
                  class CLISubcommandConfig uses 'step' for general processing steps.
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
        """
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



@app.callback()
def common_callback(
    ctx: Context,
    loglevel: str = Option("INFO", help="Logging level"),
    device: str = Option("auto", help="Device to use for processing"),
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
    ctx.params["loglevel"] = loglevel.upper()
    ctx.params["device"] = device
    ctx.params["processes"] = processes
    

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
        parsed = json.loads(value)
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
    """Convert a CLI string (path or comma-separated paths) to a FileListConfig.
    
    Args:
        value: Path string or comma-separated paths. Paths with spaces must be quoted in the shell.
        
    Returns:
        FileListConfig instance with the provided paths.
        
    Raises:
        ValueError: If the value cannot be parsed as a valid path.
    """
    return cli_to_config(value, default_value_field="paths", alt_classname="FileListConfig")


def cli_to_imageloader_config(value: str) -> ImageLoaderConfig:
    """Convert a CLI string (path) to an ImageLoaderConfig.
    
    Args:
        value: Path string for the image loader. Paths with spaces should be quoted.
        
    Returns:
        ImageLoaderConfig instance with the provided path.
        
    Raises:
        ValueError: If the value cannot be parsed as a valid path.
    """
    #substack_str = value.get("substack", None)
    # substack_slices = substack_to_slices(substack_str if substack_str else None)
    return cli_to_config(value, default_value_field="classname", alt_classname="ImageLoaderConfig")



def cli_to_component_config(value: str) -> Configuration:
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
    return cli_to_config(value)



def cli_to_imagewriter_config(value: str) -> ImageWriterConfig:
    """Convert a CLI string (output path) to an ImageWriterConfig.
    
    Args:
        value: Output path string. Paths with spaces must be quoted in the shell.
        
    Returns:
        ImageWriterConfig instance with the provided path.
        
    Raises:
        ValueError: If the value cannot be parsed as a valid path.
    """
    return cli_to_config(value, default_value_field="path", alt_classname="ImageWriterConfig")

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
    Each step can be configured with step-specific arguments using the --component{i}-* prefix.
    
    Input and output can be specified either directly or using nested config options:
    - Direct: --input path/to/file (uses parser to create FileListConfig)
    - Nested: --input-paths path/to/file (uses Typer's default prefix strategy)
    - Direct: --output path/to/output (uses parser to create ImageWriterConfig)
    - Nested: --output-path path/to/output (uses Typer's default prefix strategy)
    
    Examples:
        vistiq preprocess --input input.tif --output output --step DoG --component0-sigma-low 1.0
        vistiq preprocess --input input.tif --step Resize --component0-width 256 --component0-height 256 --step DoG --component1-sigma-low 1.0
    """
    logger.info(f"CLI input: {input}")
    logger.info(f"CLI loader: {loader}")
    logger.info(f"CLI component: {step}")
    logger.info(f"CLI output: {output}")

    # Only pass loader/output if they're not None, otherwise use defaults from default_factory
    config_kwargs = {"input": input, "step": step}
    if loader is not None:
        config_kwargs["loader"] = loader
    if output is not None:
        config_kwargs["output"] = output
    config = CLIPreprocessorConfig(**config_kwargs)

    # Call run_preprocess with the complete CLIPreprocessorConfig
    run_preprocess(config, ctx)


def run_preprocess(config: CLIPreprocessorConfig, ctx: Context) -> None:
    """Run the preprocess command.
    
    Processes images through a chain of preprocessing steps, handling multiple
    scenes and channels. Saves preprocessed images as OME-TIFF files with metadata.

    Args:
        config: Preprocess configuration containing input/output configuration,
                step configurations, and processing parameters.
                
    The function:
    1. Builds processing step chain from configuration
    2. Loads images from input path (supports multiple scenes)
    3. Processes each channel through the step chain
    4. Saves preprocessed images to output directory with metadata
    """
    logger.info(f"Running preprocess command with config: {config}")
    
    # Build FileList from config and get files
    file_list = FileList(config.input).run()
    if not file_list:
        raise ValueError("No files found for input path: {config.input.paths}")
    first_file = file_list[0]
    logger.info(f"Found {len(file_list)} files, using the first one: {first_file}")

    # Build component chain from config
    component_names, built_components = build_component_chain(config.step)
    
    # Get absolute path of input and strip extension for output directory
    input_path = first_file
    input_path_obj = Path(input_path).resolve()
    input_path_str = str(input_path)
    
    # Determine output directory (defaults to current directory if not specified)
    if config.output and config.output.path:
        output_base = Path(config.output.path).resolve()
    else:
        output_base = Path.cwd().resolve()
    
    scenes = get_scenes(input_path_str)
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
        for i, component in enumerate(built_components):
            logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()}")
            result, result_metadata = component.run(result, workers=config.processes, metadata=result_metadata)
        
        result_metadata.update({"dim_order":_infer_dim_order(result.ndim)})
        logger.info(f"Result shape: {result.shape}, metadata: {result_metadata}")
        logger.info(f"Channel axis: {result_metadata.get('channel_axis', None)}")
        logger.info(f"result metadata: {result_metadata}")
        imgwriter = ImageWriter(config.output)
        #preprocessed = np.stack(result, axis=0)
        output_path = output_dir / f"Preprocessed.tif"
        #OmeTiffWriter.save(preprocessed, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=channel_names, dim_order=_infer_dim_order(preprocessed.ndim))
        imgwriter.run(result, output_path, metadata=result_metadata)

def main() -> None:
    """Entry point for the vistiq CLI using Typer.
    
    This function is called when vistiq is invoked from the command line.
    It delegates to the Typer app which handles argument parsing and command routing.
    """
    app()


if __name__ == "__main__":
    main()

