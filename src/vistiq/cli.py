"""CLI configuration models for vistiq command-line interface."""

import sys
import argparse
import logging
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
from .io import DataLoaderConfig, ImageLoaderConfig, FileListConfig, FileList
from .preprocess import PreprocessorConfig, Preprocessor
from .workflow_builder import ConfigArgumentBuilder, WorkflowBuilder, get_registry, auto_register_configurables_by_base_class
from .utils import load_image, get_scenes
# from .app import configure_logger

logger = logging.getLogger(__name__)


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
    configuration and component chains.
    
    Attributes:
        input_config: Configuration for input data loading.
        output_path: Path to output file or directory. Defaults to current directory.
        component: List of processing component configurations to run in sequence.
    """
    input_config: DataLoaderConfig = Field(
        description="Input file or directory path"
    )
    output_path: Optional[Path] = Field(
        default=None, description="Output file or directory path"
    )
    #substack: Optional[str] = Field(
    #    default=None, description="Substack to process. Legacy: '10' or '2-40' (first axis). New: 'T:4-10,Z:2-20' (multiple dimensions)"
    #)
    component: list[PreprocessorConfig] = Field(
        default=None, description="Configs for chain of processing components to run"
    )

    @field_validator("component")
    @classmethod
    def validate_component(cls, v: Optional[list[PreprocessorConfig]]) -> Optional[list[PreprocessorConfig]]:
        """Validate that components are valid.
        
        Args:
            v: Components to validate.
        """
        return v


class CLIPreprocessorConfig(CLISubcommandConfig):
    """Configuration for the preprocess subcommand.
    
    Defines parameters for preprocessing images including denoising, resizing, and other operations.
    Extends CLISubcommandConfig with preprocessing-specific input configuration.
    
    Attributes:
        input_config: Configuration for input image loading (ImageLoaderConfig).
        component: List of preprocessing component configurations to run in sequence.
    """
    input_config: ImageLoaderConfig = Field(
        default=None, description="Configuration for input data loading"
    )
    component: list[PreprocessorConfig] = Field(
        default=None, description="Configs for chain of preprocessing components to run"
    )

    @field_validator("component")
    @classmethod
    def validate_components(cls, v: Optional[list[PreprocessorConfig]]) -> Optional[list[PreprocessorConfig]]:
        """Validate that components are valid.
        
        Args:
            v: Components to validate.
        """
        return v


# Create Typer app
# Note: We use add_completion=False to avoid completion-related help text issues
# We also need to handle component arguments specially to avoid ambiguity
app = typer.Typer(
    name="vistiq",
    help="Turn complex imaging data into quantitative insight with modular, multi-step analysis.",
    no_args_is_help=True,
    add_completion=False,  # Disable shell completion to avoid conflicts with dynamic component arguments
    chain=True,  # Allow chaining commands (though we don't use it, it helps with argument parsing)
)


def _build_component_configs(component: List[str]) -> list[Configuration]:
    """Build component configuration objects from component names and CLI arguments.
    
    Parses component-specific arguments from sys.argv with prefixes (e.g., --component0-sigma-low)
    and builds corresponding Configuration objects for each component.
    
    Args:
        component: List of component names to build configurations for.
        
    Returns:
        List of Configuration objects, one for each component specified.
    """
    registry = get_registry()
    # Use allow_abbrev=False to prevent argparse from matching --component to --component0-*
    component_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for i, component_name in enumerate(component):
        configurable_class = registry.get_configurable_class(component_name)
        if configurable_class:
            actual_name = configurable_class.__name__
            config_class_name = f"{actual_name}Config"
            config_class = registry.get_config_class(config_class_name)
            if config_class:
                # Always use prefix format for consistency, even with single component
                prefix = f"component{i}-"
                ConfigArgumentBuilder.add_config_arguments(
                    component_parser,
                    config_class,
                    prefix=prefix
                )
    # Parse component arguments
    # Use original argv (before Typer filtering) to get component-specific arguments
    # This avoids Typer's ambiguity detection (we use --step instead of --component for the main option)
    import vistiq.cli as cli_module
    original_argv = getattr(cli_module, '_ORIGINAL_ARGV', sys.argv)
    cmd_idx = original_argv.index("preprocess") if "preprocess" in original_argv else 1
    # Only parse arguments that look like component arguments (--component0-*, --component1-*, etc.)
    # or are values for component arguments
    component_argv = []
    i = cmd_idx + 1
    while i < len(original_argv):
        arg = original_argv[i]
        # Include component-specific arguments (--component0-*, --component1-*, etc.)
        # but exclude --step itself (which Typer handles)
        if arg.startswith("--component") and len(arg) > 11 and arg[11].isdigit():
            # This is a component-specific arg like --component0-sigma-low
            component_argv.append(arg)
            i += 1
            # Include the next argument if it doesn't start with - (it's a value)
            if i < len(original_argv) and not original_argv[i].startswith("-"):
                component_argv.append(original_argv[i])
                i += 1
        elif arg.startswith("--component") and ("." in arg or (len(arg) > 12 and arg[12] in ["-", "."])):
            # Handle --component0.xxx or --component0-xxx patterns
            component_argv.append(arg)
            i += 1
            if i < len(original_argv) and not original_argv[i].startswith("-"):
                component_argv.append(original_argv[i])
                i += 1
        else:
            i += 1
    
    component_args, _ = component_parser.parse_known_args(component_argv)
    
    # Build component configs
    component_configs = []
    for i, component_name in enumerate(component):
        configurable_class = registry.get_configurable_class(component_name)
        if configurable_class:
            actual_name = configurable_class.__name__
            config_class_name = f"{actual_name}Config"
            config_class = registry.get_config_class(config_class_name)
            if config_class:
                # Always use prefix format for consistency, even with single component
                prefix = f"component{i}-"
                component_config = ConfigArgumentBuilder.build_config_from_args(
                    component_args, config_class, prefix=prefix
                )
                component_configs.append(component_config)

    return component_configs


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
    """Build a chain of components from their configuration objects.
    
    Takes a list of component configuration objects, looks up their corresponding
    Configurable classes from the registry, and instantiates them to create a
    processing chain.
    
    Args:
        component_configs: Optional list of component configuration objects.
                          If None or empty, returns empty lists.
        
    Returns:
        Tuple of (component_names, built_components) where:
        - component_names: List of component class names as strings
        - built_components: List of instantiated Configurable component objects
        
    Raises:
        Exception: If a component cannot be built from its configuration.
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
    

@app.command("preprocess")
def preprocess_cmd(
    ctx: Context,
    input_path: Path = Option(..., "--input", "-i", help="Input file or directory path"),
    output_path: Optional[Path] = Option(None, "--output", "-o", help="Output file or directory path"),
    component: Optional[List[str]] = Option(None, "--step", "-s", help="Processing step/component to include (can be specified multiple times). Use --step NAME to add a component."),
) -> None:
    """Preprocess images with a chain of preprocessing components.
    
    Examples:
        vistiq preprocess -i input.tif -o output --step DoG --component0-sigma-low 1.0 --component0-sigma-high 12.0
        vistiq preprocess -i input.tif --step DoG --step OtsuThreshold --component0-sigma-low 1.0 --component1-threshold 0.5
    """
    # Build component configs if components are specified
    component_configs = None
    if component:
        auto_register_configurables_by_base_class(Preprocessor)
        
        # Parse component arguments from sys.argv with prefixes
        # We need to filter out component-specific args from sys.argv before passing to argparse
        # to avoid Typer's ambiguity detection
        component_configs = _build_component_configs(component)
        
    
    # Build complete CLIPreprocessorConfig with all fields including CLIAppConfig fields
    logger.info(f"Input path: {type(input_path)}")
    fconfig = FileListConfig(input_paths=input_path)
    flist, metadata = FileList(fconfig).run()
    for f in flist:
        logger.info(f"Found file: {f}")
    if not flist:
        raise ValueError(f"No files found for input path: {input_path}")
    first_file = flist[0]
    logger.info(f"Found {len(flist)} files, only using the first one: {first_file}")
    config = CLIPreprocessorConfig(
        loglevel=ctx.params.get("loglevel", "INFO"),
        device=ctx.params.get("device", "auto"),
        processes=ctx.params.get("processes", 1),
        input_config=ImageLoaderConfig(input_path=first_file),
        output_path=output_path,
        component=component_configs,
    )
    
    # Call run_preprocess with the complete CLIPreprocessorConfig
    run_preprocess(config)


def run_preprocess(config: CLIPreprocessorConfig) -> None:
    """Run the preprocess command.
    
    Processes images through a chain of preprocessing components, handling multiple
    scenes and channels. Saves preprocessed images as OME-TIFF files with metadata.

    Args:
        config: Preprocess configuration containing input/output paths, component
                configurations, and processing parameters.
                
    The function:
    1. Builds component chain from configuration
    2. Loads images from input path (supports multiple scenes)
    3. Processes each channel through the component chain
    4. Saves preprocessed images to output directory with metadata
    """
    logger.info(f"Running preprocess command with config: {config}")
    
    # Auto-register available components
    auto_register_configurables_by_base_class(Preprocessor)
    
    # Build component chain from config
    component_names, built_components = build_component_chain(config.component)
    
    # Get absolute path of input and strip extension for output directory
    input_path = config.input_config.input_path
    input_path_obj = Path(input_path).resolve()
    input_path_str = str(input_path)
    
    # Determine output directory (defaults to current directory if not specified)
    output_base = Path(config.output_path or Path.cwd()).resolve()
    
    scenes = get_scenes(input_path_str)
    for idx, sc in enumerate(scenes):
        logger.info(f"Processing scene: {sc}")
        # Load image with substack slicing if specified
        substack_slices = substack_to_slices(config.input_config.substack if config.input_config else None)
        img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.input_config.rename_channel if config.input_config else None)
        channel_names = metadata["channel_names"]
        
        ishape = str(img.shape).replace(" ", "")
        component_names_str = "-".join(component_names) if component_names else "none"
        output_dir = output_base / input_path_obj.stem / f"{sc}-{ishape}-{component_names_str}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
        img_ch = np.unstack(img, axis=0)
        logger.info(f"Image shape: {img.shape}, channel names: {channel_names}, metadata: {metadata}")
        for im in img_ch:
            logger.debug(f"Channel shape: {im.shape}")

        preprocessed_ch = []
        output_paths = []
        for ch_name, im in zip(channel_names, img_ch):
            logger.info(f"{''.join(metadata["used_scale"]._fields)}")
            # Run components in sequence
            result = im
            for i, component in enumerate(built_components):
                logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()} on channel {ch_name}")
                result = component.run(result, workers=config.processes)
            
            output_path = output_dir / f"Preprocessed_{ch_name}.tif"
            preprocessed_ch.append(result)
            output_paths.append(output_path)
        
        split_channels = config.input_config.split_channels if config.input_config else True
        if split_channels:
            for preprocessed, output_path, ch_name in zip(preprocessed_ch, output_paths, channel_names):
                OmeTiffWriter.save(preprocessed, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(preprocessed.ndim))
        else:
            preprocessed = np.stack(preprocessed_ch, axis=0)
            output_path = output_dir / f"Preprocessed-{"-".join(channel_names)}.tif"
            OmeTiffWriter.save(preprocessed, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=channel_names, dim_order=_infer_dim_order(preprocessed.ndim))

def main() -> None:
    """Entry point for the vistiq CLI using Typer.
    
    This function is called when vistiq is invoked from the command line.
    It delegates to the Typer app which handles argument parsing and command routing.
    
    Before invoking Typer, we filter out component-specific arguments (--component0-*, etc.)
    to prevent Typer from complaining about unknown arguments. These are parsed separately
    by our custom argparse parser in the command functions.
    """
    # Store original argv - we'll need it for parsing component arguments
    _original_argv = sys.argv[:]
    
    # Filter out component-specific arguments before Typer sees them
    # This prevents "No such option" errors for --component0-*, --component1-*, etc.
    filtered_argv = [_original_argv[0]]  # Keep script name
    
    i = 1
    while i < len(_original_argv):
        arg = _original_argv[i]
        # Keep all arguments except component-specific ones (--component0-*, --component1-*, etc.)
        if not (arg.startswith("--component") and len(arg) > 11 and arg[11].isdigit()):
            filtered_argv.append(arg)
            i += 1
        else:
            # Skip component-specific arguments (we'll parse them ourselves in the command function)
            i += 1
            # Skip the value if it's not another option
            if i < len(_original_argv) and not _original_argv[i].startswith("-"):
                i += 1
    
    # Temporarily replace sys.argv for Typer's parsing
    sys.argv = filtered_argv
    
    # Store original argv in a module-level variable so command functions can access it
    import vistiq.cli as cli_module
    cli_module._ORIGINAL_ARGV = _original_argv
    
    try:
        app()
    finally:
        # Restore original argv
        sys.argv = _original_argv


if __name__ == "__main__":
    main()

