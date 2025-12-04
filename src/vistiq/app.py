import argparse
import logging
import numpy as np
import os
import itertools
import sys
from pathlib import Path
from typing import Optional, Literal, List, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from bioio_ome_tiff.writers import OmeTiffWriter
    OME_TIFF_AVAILABLE = True
except ImportError:
    OME_TIFF_AVAILABLE = False
    OmeTiffWriter = None
from .core import ArrayIteratorConfig, cli_config
from .io import DataLoaderConfig, FileList, ImageLoader, ImageLoaderConfig
from .preprocess import DoG, DoGConfig, PreprocessorConfig
from .cli import CLIAppConfig, CLISubcommandConfig, CLIPreprocessorConfig, parse_substack, run_preprocess
from .seg import (
    MicroSAMSegmenter, MicroSAMSegmenterConfig, 
    RegionFilter, RegionFilterConfig, 
    RangeFilter, RangeFilterConfig, 
    RegionAnalyzer, RegionAnalyzerConfig,
    SegmenterConfig, LabellerConfig, ThresholderConfig
)
from .analysis import CoincidenceDetector, CoincidenceDetectorConfig
from .train import TrainerConfig, MicroSAMTrainer, MicroSAMTrainerConfig, DatasetCreator, DatasetCreatorConfig
from .workflow_builder import (
    get_registry,
    ConfigArgumentBuilder,
    WorkflowBuilder,
    auto_register_configurables
)
from .core import Configurable
from .utils import (
    load_mp4,
    check_device,
    load_image,
    get_scenes,
)

logger = logging.getLogger(__name__)


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

def configure_logger(level: str = "INFO", force: bool = False) -> logging.Logger:
    """Configure the logger for use in CLI or interactive environments (Jupyter/IPython).

    This function configures the root logger with appropriate handlers and formatting.
    It works in both command-line applications and interactive environments like
    Jupyter notebooks or IPython.

    Args:
        level (str): The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        force (bool): If True, force reconfiguration even if handlers already exist.
            Useful in Jupyter/IPython where logging might already be configured.

    Returns:
        logging.Logger: The configured logger instance.
    """
    lmap = logging.getLevelNamesMapping()
    level_int = lmap.get(level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Check if we need to configure handlers
    needs_config = force or len(root_logger.handlers) == 0
    
    if needs_config:
        # Use basicConfig with force parameter if available (Python 3.8+)
        # Otherwise, manually configure
        try:
            # Python 3.8+ supports force parameter
            logging.basicConfig(
                level=level_int,
                format="%(asctime)s - %(levelname)s - %(message)s",
                force=force,
            )
        except TypeError:
            # Python < 3.8: manually remove handlers if force=True
            if force:
                root_logger.handlers.clear()
            logging.basicConfig(
                level=level_int,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
    else:
        # Just update the level on existing handlers
        root_logger.setLevel(level_int)
        for handler in root_logger.handlers:
            handler.setLevel(level_int)
            # Update format if it's a StreamHandler
            if isinstance(handler, logging.StreamHandler):
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    return logger


def parse_dimension_range(range_str: str) -> tuple[int, int]:
    """Parse a single dimension range like '4-10' or '5' into start/end indices.
    
    Args:
        range_str: Range string like '4-10' (1-based, inclusive) or '5' (single value).
        
    Returns:
        Tuple of (start, end) as zero-based indices (both inclusive).
        
    Raises:
        ValueError: If range string is invalid.
    """
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
        # Convert to zero-based inclusive indices
        return start - 1, end - 1
    else:
        if not range_str.isdigit():
            raise ValueError(f"Invalid range format: '{range_str}'. Expected 'A-B' or 'A'")
        val = int(range_str)
        if val < 1:
            raise ValueError("Range index is 1-based and must be >= 1")
        # Single value: both start and end are the same (zero-based)
        return val - 1, val - 1


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
        For legacy format, uses None as key: {None: slice(0, 10)} for substack='10'
        Example: 
        - {None: slice(0, 10)} for substack='10' (legacy, first axis)
        - {None: slice(1, 40)} for substack='2-40' (legacy, first axis)
        - {'T': slice(3, 10), 'Z': slice(1, 20)} for substack='T:4-10;Z:2-20' (new)
    """
    if not substack:
        return None
    
    substack = str(substack).strip()
    
    # Check if it's the new format (contains ':' and possibly multiple dimensions)
    if ":" in substack:
        # New format: T:4-10;Z:2-20
        slices_dict = {}
        # Split by semicolon to get dimension:range pairs
        parts = [p.strip() for p in substack.split(";")]
        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid substack format: '{part}'. Expected 'DIM:RANGE'")
            dim_name, range_str = part.split(":", 1)
            dim_name = dim_name.strip().upper()  # Normalize to uppercase
            if not dim_name:
                raise ValueError(f"Invalid dimension name in: '{part}'")
            
            # Parse the range
            start, end = parse_dimension_range(range_str)
            # end is inclusive, so we need end+1 for slice
            slices_dict[dim_name] = slice(start, end + 1)
        
        return slices_dict if slices_dict else None
    else:
        # Legacy format: '10' or '2-40' (applied to first axis)
        # Use None as key to indicate "first dimension"
        start, end = parse_substack(substack)
        if start is None or end is None:
            return None
        
        # end is inclusive, so we need end+1 for slice
        # Use None as key to indicate first dimension
        return {None: slice(start, end + 1)}

class CLISegmentConfig(CLISubcommandConfig):
    """Configuration for the segment subcommand.
    
    Defines parameters for image segmentation including thresholding method,
    connectivity, and area filtering options.
    
    Attributes:
        threshold_method: Method to use for thresholding ("otsu", "local", "niblack", "sauvola").
        block_size: Block size for local thresholding (must be odd).
        connectivity: Connectivity for labeling (1 for 4-connected, 2 for 8-connected).
        min_area: Minimum object area for filtering (None = no minimum).
        max_area: Maximum object area for filtering (None = no maximum).
    """
    subcommand: str = Field(default="segment", description="Subcommand to run")
    components: list[ThresholderConfig | SegmenterConfig | LabellerConfig] = Field(
        default=None, description="Components to use for segmentation"
    )
    model_type: Literal["vit_l_lm", "vit_b_lm", "vit_t_lm", "vit_h_lm"] = Field(
        default="vit_l_lm", description="Model type"
    )
    threshold_method: Literal["otsu", "local", "niblack", "sauvola"] = Field(
        default="otsu", description="Thresholding method"
    )
    block_size: Optional[int] = Field(
        default=51, description="Block size for local thresholding (must be odd)"
    )
    connectivity: int = Field(
        default=1, description="Connectivity for labeling (1 or 2)"
    )
    min_area: Optional[float] = Field(
        default=None, description="Minimum object area for filtering"
    )
    max_area: Optional[float] = Field(
        default=None, description="Maximum object area for filtering"
    )

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate that block_size is odd if provided.
        
        Args:
            v: Block size value to validate.
            
        Returns:
            Validated block size.
            
        Raises:
            ValueError: If block_size is even.
        """
        if v is not None and v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v

    @field_validator("connectivity")
    @classmethod
    def validate_connectivity(cls, v: int) -> int:
        """Validate that connectivity is 1 or 2.
        
        Args:
            v: Connectivity value to validate.
            
        Returns:
            Validated connectivity value.
            
        Raises:
            ValueError: If connectivity is not 1 or 2.
        """
        if v not in (1, 2):
            raise ValueError("connectivity must be 1 or 2")
        return v


class CLIAnalyzeConfig(CLISubcommandConfig):
    """Configuration for the analyze subcommand.
    
    Defines parameters for analyzing segmented images and extracting
    object properties and coordinates.
    
    Attributes:
        include_stats: Whether to include object statistics in analysis.
        include_coords: Whether to include coordinate extraction.
    """
    subcommand: str = Field(default="analyze", description="Subcommand to run")
    components: list[RegionAnalyzerConfig] = Field(
        default=None, description="Components to use for analysis"
    )
    include_stats: bool = Field(
        default=True, description="Include object statistics in analysis"
    )
    include_coords: bool = Field(
        default=True, description="Include coordinate extraction"
    )


class CLICoincidenceConfig(CLISubcommandConfig):
    """Configuration for the coincidence subcommand.
    
    Defines parameters for running the coincidence detection workflow
    with DoG preprocessing, MicroSAM segmentation, and coincidence detection.
    
    Attributes:
        sigma_low: Sigma for lower Gaussian blur in DoG (default: 1.0).
        sigma_high: Sigma for higher Gaussian blur in DoG (default: 12.0).
        normalize: Normalize DoG output to [0, 1] range (default: True).
        area_min: Minimum value for area filter (default: 0.0).
        area_max: Maximum value for area filter (default: inf).
        volume_min: Minimum value for volume filter (default: 0.0).
        volume_max: Maximum value for volume filter (default: inf).
        aspect_ratio_min: Minimum value for aspect ratio filter (default: 0.0).
        aspect_ratio_max: Maximum value for aspect ratio filter (default: 1.0).
        cross_sectional_area_min: Minimum value for cross-sectional area filter (default: 0.0).
        cross_sectional_area_max: Maximum value for cross-sectional area filter (default: inf).
        model_type: MicroSAM model type (default: "vit_l_lm").
        threshold: Threshold for coincidence detection (default: 0.1).
        method: Coincidence detection method: 'iou' or 'dice' (default: "dice").
        mode: Coincidence detection mode: 'bounding_box' or 'outline' (default: "outline").
    """
    subcommand: str = Field(default="coincidence", description="Subcommand to run")
    components: list[CoincidenceDetectorConfig] = Field(
        default=None, description="Components to use for coincidence detection"
    )
    sigma_low: float = Field(
        default=1.0, description="Sigma for lower Gaussian blur in DoG"
    )
    sigma_high: float = Field(
        default=12.0, description="Sigma for higher Gaussian blur in DoG"
    )
    normalize: bool = Field(
        default=True, description="Normalize DoG output to [0, 1] range"
    )
    area_min: float = Field(
        default=0.0, description="Minimum value for area filter"
    )
    area_max: float = Field(
        default=np.inf, description="Maximum value for area filter"
    )
    volume_min: float = Field(
        default=0.0, description="Minimum value for volume filter"
    )
    volume_max: float = Field(
        default=np.inf, description="Maximum value for volume filter"
    )
    aspect_ratio_min: float = Field(
        default=0.0, description="Minimum value for aspect ratio filter"
    )
    aspect_ratio_max: float = Field(
        default=1.0, description="Maximum value for aspect ratio filter"
    )
    cross_sectional_area_min: float = Field(
        default=0.0, description="Minimum value for cross-sectional area filter"
    )
    cross_sectional_area_max: float = Field(
        default=np.inf, description="Maximum value for cross-sectional area filter"
    )
    model_type: Literal["vit_l_lm", "vit_b_lm", "vit_t_lm", "vit_h_lm"] = Field(
        default="vit_l_lm", description="MicroSAM model type"
    )
    threshold: float = Field(
        default=0.1, description="Threshold for coincidence detection"
    )
    method: Literal["iou", "dice"] = Field(
        default="dice", description="Coincidence detection method: 'iou' or 'dice'"
    )
    mode: Literal["bounding_box", "outline"] = Field(
        default="outline", description="Coincidence detection mode: 'bounding_box' or 'outline'"
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate that threshold is between 0.0 and 1.0.
        
        Args:
            v: Threshold value to validate.
            
        Returns:
            Validated threshold value.
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class CLITrainConfig(CLISubcommandConfig):
    """Configuration for the train subcommand.
    
    Defines parameters for training/fine-tuning MicroSAM models.
    
    Attributes:
        model_type: Model type for training (default: "vit_l_lm").
        checkpoint_name: Name for the checkpoint (default: "sam_synthetic").
        export_path: Path to export the trained model (default: "./finetuned_synthetic_model.pth").
        batch_size: Batch size for training (default: 1).
        patch_shape: Shape of patches for training as comma-separated string (default: "1,512,512").
        split_ratio: Ratio for train/validation split (default: 0.8).
        raw_key: Key for raw image data in the data file (default: None).
        label_key: Key for label data in the data file (default: None).
        learning_rate: Learning rate for training (default: 1e-5).
        log_image_interval: Interval for logging images during training (default: 10).
        n_objects_per_batch: Number of objects per batch (default: 10).
        n_iterations: Number of training iterations (default: 10000).
        n_sub_iteration: Number of sub-iterations (default: 8).
        mixed_precision: Whether to use mixed precision training (default: True).
        compile_model: Whether to compile the model (default: False).
        verbose: Whether to print verbose output (default: True).
        image_path: Image file or directory path for training.
        label_path: Label file or directory path corresponding to image_path.
        patterns: Optional comma-separated file patterns for filtering TIFF files in directories (default: None).
    """
    subcommand: str = Field(default="train", description="Subcommand to run")
    components: list[TrainerConfig] = Field(
        default=None, description="Components to use for training"
    )
    model_type: Literal["vit_l_lm", "vit_b_lm", "vit_t_lm", "vit_h_lm"] = Field(
        default="vit_l_lm", description="Model type for training"
    )
    checkpoint_name: str = Field(
        default="sam_synthetic", description="Name for the checkpoint"
    )
    export_path: str = Field(
        default="./finetuned_synthetic_model.pth", description="Path to export the trained model"
    )
    batch_size: int = Field(
        default=1, description="Batch size for training"
    )
    patch_shape: str = Field(
        default="1,512,512", description="Shape of patches for training as comma-separated integers"
    )
    split_ratio: float = Field(
        default=0.8, description="Ratio for train/validation split"
    )
    raw_key: Optional[str] = Field(
        default=None, description="Key for raw image data in the data file"
    )
    label_key: Optional[str] = Field(
        default=None, description="Key for label data in the data file"
    )
    learning_rate: float = Field(
        default=1e-5, description="Learning rate for training"
    )
    log_image_interval: int = Field(
        default=10, description="Interval for logging images during training"
    )
    n_objects_per_batch: int = Field(
        default=10, description="Number of objects per batch"
    )
    n_iterations: int = Field(
        default=10000, description="Number of training iterations"
    )
    n_sub_iteration: int = Field(
        default=8, description="Number of sub-iterations"
    )
    mixed_precision: bool = Field(
        default=True, description="Whether to use mixed precision training"
    )
    compile_model: bool = Field(
        default=False, description="Whether to compile the model"
    )
    verbose: bool = Field(
        default=True, description="Whether to print verbose output"
    )
    image_path: Path = Field(
        description="Image file or directory path for training"
    )
    label_path: Path = Field(
        description="Label file or directory path corresponding to image_path"
    )
    patterns: Optional[str] = Field(
        default=None,
        description="Comma-separated file patterns for filtering TIFF files in directories (e.g., '*_img.tif,*_label.tif')"
    )


class CLIFullConfig(CLISubcommandConfig):
    """Configuration for the full subcommand (segment + analyze).
    
    Combines segmentation and analysis parameters for running the complete
    pipeline in a single command.
    
    Attributes:
        threshold_method: Method to use for thresholding ("otsu", "local", "niblack", "sauvola").
        block_size: Block size for local thresholding (must be odd).
        connectivity: Connectivity for labeling (1 for 4-connected, 2 for 8-connected).
        min_area: Minimum object area for filtering (None = no minimum).
        max_area: Maximum object area for filtering (None = no maximum).
        include_stats: Whether to include object statistics in analysis.
        include_coords: Whether to include coordinate extraction.
    """
    subcommand: str = Field(default="full", description="Subcommand to run")

    threshold_method: Literal["otsu", "local", "niblack", "sauvola"] = Field(
        default="otsu", description="Thresholding method"
    )
    block_size: Optional[int] = Field(
        default=51, description="Block size for local thresholding (must be odd)"
    )
    connectivity: int = Field(
        default=1, description="Connectivity for labeling (1 or 2)"
    )
    min_area: Optional[float] = Field(
        default=None, description="Minimum object area for filtering"
    )
    max_area: Optional[float] = Field(
        default=None, description="Maximum object area for filtering"
    )
    include_stats: bool = Field(
        default=True, description="Include object statistics in analysis"
    )
    include_coords: bool = Field(
        default=True, description="Include coordinate extraction"
    )

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate that block_size is odd if provided.
        
        Args:
            v: Block size value to validate.
            
        Returns:
            Validated block size.
            
        Raises:
            ValueError: If block_size is even.
        """
        if v is not None and v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v

    @field_validator("connectivity")
    @classmethod
    def validate_connectivity(cls, v: int) -> int:
        """Validate that connectivity is 1 or 2.
        
        Args:
            v: Connectivity value to validate.
            
        Returns:
            Validated connectivity value.
            
        Raises:
            ValueError: If connectivity is not 1 or 2.
        """
        if v not in (1, 2):
            raise ValueError("connectivity must be 1 or 2")
        return v


class ComponentAwareHelpAction(argparse._HelpAction):
    """Custom help action that shows component-specific arguments when --component is specified."""
    
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, 
                 help=None, base_class=None, registry=None):
        super().__init__(option_strings, dest, default=default, help=help)
        self.base_class = base_class
        self.registry = registry
    
    def __call__(self, parser, namespace, values, option_string=None):
        # Check if --component was specified in the raw arguments
        components = []
        raw_args = sys.argv[1:]  # Skip script name
        
        # Find the subcommand index
        subcommand_idx = -1
        for i, arg in enumerate(raw_args):
            if arg in ['preprocess', 'segment', 'train', 'workflow', 'analyze', 'coincidence', 'full']:
                subcommand_idx = i
                break
        
        # Only look for --component after the subcommand
        if subcommand_idx >= 0:
            i = subcommand_idx + 1
            while i < len(raw_args):
                if raw_args[i] == '--component':
                    if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('-') and raw_args[i + 1] != '-h':
                        components.append(raw_args[i + 1])
                        i += 2
                    else:
                        i += 1
                elif raw_args[i] == '-h' or raw_args[i] == '--help':
                    break
                else:
                    i += 1
        
        # If components are specified, show component-specific help
        if components and self.registry and self.base_class:
            # Get config classes for the specified components
            component_configs = {}
            for comp_name in components:
                configurable_class = self.registry.get_configurable_class(comp_name)
                if configurable_class:
                    config_class_name = f"{configurable_class.__name__}Config"
                    config_class = self.registry.get_config_class(config_class_name)
                    if config_class:
                        component_configs[comp_name] = config_class
            
            if component_configs:
                # Print standard help first
                parser.print_help()
                
                # Print component-specific information
                print("\n" + "="*70)
                print("Component-specific arguments:")
                print("="*70)
                for comp_name, config_class in component_configs.items():
                    print(f"\n{comp_name} arguments:")
                    if hasattr(config_class, 'model_fields'):
                        # Get base fields to skip (from Configuration base class)
                        base_fields = set()
                        from .core import Configuration
                        if hasattr(Configuration, 'model_fields'):
                            base_fields = set(Configuration.model_fields.keys())
                        
                        for field_name, field_info in config_class.model_fields.items():
                            # Skip base Configuration fields
                            if field_name in base_fields:
                                continue
                            
                            # Check if field should be included in CLI (same logic as ConfigArgumentBuilder)
                            include_in_cli = True
                            
                            # Check class-level CLI config from @cli_config decorator
                            # Check the MRO to find all __cli_config__ attributes in the inheritance chain
                            for cls in config_class.__mro__:
                                if hasattr(cls, '__cli_config__'):
                                    cli_config = cls.__cli_config__
                                    if cli_config.get('include_only') is not None:
                                        include_in_cli = field_name in cli_config['include_only']
                                        break  # include_only takes precedence
                                    elif cli_config.get('exclude') is not None:
                                        if field_name in cli_config['exclude']:
                                            include_in_cli = False
                                            break  # If excluded in any parent, exclude it
                            
                            # Check field-level CLI marker in json_schema_extra
                            if include_in_cli and hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                                cli_flag = field_info.json_schema_extra.get('cli')
                                if cli_flag is False:
                                    include_in_cli = False
                            # Also check FieldInfo.extra (Pydantic v2)
                            elif include_in_cli and hasattr(field_info, 'extra') and isinstance(field_info.extra, dict):
                                cli_flag = field_info.extra.get('cli')
                                if cli_flag is False:
                                    include_in_cli = False
                            
                            if not include_in_cli:
                                continue
                            
                            arg_name = field_name.replace("_", "-")
                            description = getattr(field_info, 'description', None) or field_name
                            default = getattr(field_info, 'default', None)
                            if default is not None and default != ...:
                                default_str = f" (default: {default})"
                            else:
                                default_str = ""
                            
                            # Get field type for better help
                            field_type = getattr(field_info, 'annotation', None)
                            type_str = ""
                            if field_type:
                                try:
                                    # Handle Union types (Python 3.10+)
                                    if hasattr(field_type, '__origin__'):
                                        # Handle Optional, Union, etc.
                                        if field_type.__origin__ is type(None) or field_type.__origin__ is None:
                                            # Optional type
                                            args = getattr(field_type, '__args__', None)
                                            if args and len(args) > 0:
                                                non_none_type = args[0]
                                                if hasattr(non_none_type, '__name__'):
                                                    type_str = f" [{non_none_type.__name__}]"
                                                else:
                                                    type_str = f" [{str(non_none_type)}]"
                                        else:
                                            type_str = f" [{str(field_type)}]"
                                    elif hasattr(field_type, '__name__'):
                                        type_str = f" [{field_type.__name__}]"
                                    elif hasattr(field_type, '__qualname__'):
                                        type_str = f" [{field_type.__qualname__}]"
                                    else:
                                        # Fallback to string representation
                                        type_str = f" [{str(field_type)}]"
                                except Exception:
                                    # If anything goes wrong, just skip the type
                                    type_str = ""
                            
                            print(f"  --{arg_name}{type_str}: {description}{default_str}")
            else:
                # No valid components found, show normal help
                parser.print_help()
        else:
            # No components specified, show normal help
            parser.print_help()
        
        parser.exit()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser from CLIAppConfig.
    
    Args:
        parser: Argument parser to add common arguments to.
    """
    from .workflow_builder import ConfigArgumentBuilder
    # Add CLIAppConfig fields dynamically
    ConfigArgumentBuilder.add_config_arguments(parser, CLIAppConfig, prefix="")


def build_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser with subcommands.

    Dynamically creates subcommand parsers from _SUBCOMMAND_CONFIGS registry.
    Main parser only shows CLIAppConfig options.

    Returns:
        argparse.ArgumentParser: Configured parser for the vistiq CLI.
    """
    parser = argparse.ArgumentParser(
        prog="vistiq",
        description=(
            "Turn complex imaging data into quantitative insight "
            "with modular, multi-step analysis."
        ),
    )

    # Add CLIAppConfig arguments to main parser
    _add_common_args(parser)

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Available commands", 
        required=True,
        metavar="COMMAND"
    )
    
    # Dynamically create subcommand parsers from registry
    from .workflow_builder import ConfigArgumentBuilder
    
    # Track which subcommands we've created
    created_subcommands = set()
    
    for config_class in _SUBCOMMAND_CONFIGS:
        # Get subcommand name from the default value of the subcommand field
        if hasattr(config_class, "model_fields") and "subcommand" in config_class.model_fields:
            field_info = config_class.model_fields["subcommand"]
            default_value = getattr(field_info, "default", None)
            if default_value:
                subcommand_name = default_value
                # Skip if already created
                if subcommand_name in created_subcommands:
                    continue
                created_subcommands.add(subcommand_name)
                
                # Get help text from class docstring (first line)
                help_text = config_class.__doc__.split('\n')[0].strip() if config_class.__doc__ else subcommand_name
                
                # Create subcommand parser
                subcommand_parser = subparsers.add_parser(
                    subcommand_name,
                    help=help_text,
                    add_help=False  # We'll add custom help action if needed
                )
                
                # Add CLIAppConfig arguments to subcommand parser
                _add_common_args(subcommand_parser)
                
                # Add subcommand-specific arguments from config class
                ConfigArgumentBuilder.add_config_arguments(
                    subcommand_parser,
                    config_class,
                    prefix=""
                )
                
                # Add --component argument if this subcommand supports components
                # Use the registry to get available components
                # Capture subcommand_name and components in closure for help action
                help_subcommand_name = subcommand_name
                help_available_components = _SUBCOMMAND_COMPONENT_REGISTRY.get(subcommand_name, None)
                
                if help_available_components:
                    # Build component names list
                    component_names = sorted([cls.__name__.replace('Config', '') for cls in help_available_components])
                    
                    subcommand_parser.add_argument(
                        "-c",
                        "--component",
                        dest="components",
                        action="append",
                        help=f"Component to include in {subcommand_name} pipeline (can be specified multiple times). "
                             f"Example: --component {component_names[0] if component_names else 'ComponentName'}",
                    )
                
                # Add custom help action that appends component list at the end
                # Create a closure to capture the subcommand name and components
                def make_help_action(subcmd_name, available_comps):
                    class SubcommandHelpAction(argparse._HelpAction):
                        def __call__(self, parser, namespace, values, option_string=None):
                            # Print standard help
                            parser.print_help()
                            
                            # Append available components if this subcommand supports them
                            if available_comps:
                                component_names = sorted([cls.__name__.replace('Config', '') for cls in available_comps])
                                print("\n" + "="*70)
                                print(f"Available components for '{subcmd_name}':")
                                print("="*70)
                                for name in component_names:
                                    print(f"  {name}")
                                print()
                            
                            parser.exit()
                    return SubcommandHelpAction
                
                subcommand_parser.add_argument(
                    '-h', '--help',
                    action=make_help_action(help_subcommand_name, help_available_components),
                    help='show this help message and exit'
                )
    
    # Handle workflow subcommand separately (it has special component handling)
    # Only add if not already created by the dynamic loop
    if "workflow" not in created_subcommands:
        workflow_parser = subparsers.add_parser(
            "workflow", help="Build and run modular workflows from CLI-specified components"
        )
        _add_common_args(workflow_parser)
    workflow_parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="single image file or directory with image files to be processed",
    )
    workflow_parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=False,
        help="output file or directory",
    )
    workflow_parser.add_argument(
        "-g",
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="convert loaded images/videos to grayscale before processing",
    )
    workflow_parser.add_argument(
        "-f",
        "--substack",
        dest="substack",
        type=str,
        default=None,
        help="substack to process; examples: '10' or '2-40' (legacy, first axis) or 'T:4-10,Z:2-20' (new format). Default: all frames",
    )
    workflow_parser.add_argument(
        "--component",
        dest="components",
        action="append",
        required=True,
        help="Component to include in workflow (can be specified multiple times). "
             "Available components will be auto-discovered from registered modules. "
             "Example: --component DoG --component OtsuThreshold",
    )

    # Auto-register components and add their arguments dynamically
    auto_register_configurables(Configurable, [
        "vistiq.preprocess",
        "vistiq.seg",
        "vistiq.analysis",
        "vistiq.core"
    ])
    
    registry = get_registry()
    for config_name in registry.list_configs():
        config_class = registry.get_config_class(config_name)
        if config_class:
            # Add arguments for this config class
            # Use component index as prefix if multiple components
            ConfigArgumentBuilder.add_config_arguments(
                workflow_parser,
                config_class,
                prefix=""
            )

    return parser


# Registry of subcommand config classes
# Each config class should have a subcommand field with a default value that matches the CLI subcommand string
_SUBCOMMAND_CONFIGS: list[type[CLISubcommandConfig]] = [
    CLIPreprocessorConfig,
    CLISegmentConfig,
    CLIAnalyzeConfig,
    CLICoincidenceConfig,
    CLITrainConfig,
]

# Dictionary registry mapping subcommand names to lists of allowed component Configuration classes
# This maps each subcommand to the component config types it accepts in its components field
_SUBCOMMAND_COMPONENT_REGISTRY: dict[str, list[type]] = {}

def _build_subcommand_component_registry() -> dict[str, list[type]]:
    """Build a dictionary registry mapping subcommands to their allowed component config types.
    
    For each CLI*Config class, checks the `components` field type annotation to determine
    what component Configuration classes are allowed. Uses the subcommand field to map
    to the subcommand name.
    
    Returns:
        Dictionary mapping subcommand names to lists of allowed component Configuration classes.
    """
    from typing import get_args, get_origin
    from .core import Configuration
    
    registry: dict[str, list[type]] = {}
    
    # Get base class components field for comparison
    base_has_components = hasattr(CLISubcommandConfig, "__annotations__") and "components" in CLISubcommandConfig.__annotations__
    
    for config_class in _SUBCOMMAND_CONFIGS:
        # Get subcommand name
        if not (hasattr(config_class, "model_fields") and "subcommand" in config_class.model_fields):
            continue
        field_info = config_class.model_fields["subcommand"]
        default_value = getattr(field_info, "default", None)
        if not default_value:
            continue
        
        subcommand_name = default_value
        
        # Check if this class explicitly defines components in its annotations (not inherited)
        has_explicit_components = hasattr(config_class, "__annotations__") and "components" in config_class.__annotations__
        
        if not has_explicit_components:
            # This class doesn't explicitly define components - set to empty list
            if subcommand_name not in registry:
                registry[subcommand_name] = []
            continue
        
        # Get the annotation directly from the class
        annotation = config_class.__annotations__["components"]
        
        # Also check model_fields to get the actual field info
        if not (hasattr(config_class, "model_fields") and "components" in config_class.model_fields):
            if subcommand_name not in registry:
                registry[subcommand_name] = []
            continue
        
        components_field = config_class.model_fields["components"]
        
        # Extract the inner type from list[SomeConfig]
        origin = get_origin(annotation)
        if origin is list or (hasattr(origin, "__origin__") and getattr(origin, "__origin__") is list):
            args = get_args(annotation)
            if args:
                component_config_type = args[0]
                
                # Handle Union types (e.g., Optional[SomeConfig] -> SomeConfig, or SegmenterConfig | LabellerConfig)
                component_types = []
                inner_origin = get_origin(component_config_type)
                # Check for Union or | operator (UnionType in Python 3.10+)
                if inner_origin is not None:
                    # It's a Union type
                    union_args = get_args(component_config_type)
                    component_types = [arg for arg in union_args if arg is not type(None)]
                else:
                    # Not a Union, just a single type
                    component_types = [component_config_type]
                
                # Process each type in the Union
                for comp_type in component_types:
                    # Check if it's a Configuration subclass
                    if isinstance(comp_type, type) and issubclass(comp_type, Configuration):
                        if subcommand_name not in registry:
                            registry[subcommand_name] = []
                        if comp_type not in registry[subcommand_name]:  # Avoid duplicates
                            registry[subcommand_name].append(comp_type)
                    else:
                        # Try to resolve string annotations
                        if isinstance(comp_type, str):
                            # Forward reference - try to resolve
                            if hasattr(config_class, "__module__"):
                                try:
                                    import sys
                                    module = sys.modules.get(config_class.__module__)
                                    if module:
                                        resolved_type = getattr(module, comp_type, None)
                                        if resolved_type and isinstance(resolved_type, type) and issubclass(resolved_type, Configuration):
                                            if subcommand_name not in registry:
                                                registry[subcommand_name] = []
                                            if resolved_type not in registry[subcommand_name]:  # Avoid duplicates
                                                registry[subcommand_name].append(resolved_type)
                                except (AttributeError, TypeError):
                                    pass
    
    # Now expand the registry to include all subclasses of the allowed types
    # Get all registered Configuration classes from the ComponentRegistry
    from .workflow_builder import get_registry
    import inspect
    import importlib
    
    # Auto-register components to ensure registry is populated
    auto_register_configurables(Configurable, [
        "vistiq.preprocess",
        "vistiq.seg",
        "vistiq.analysis",
        "vistiq.train",
        "vistiq.core"
    ])
    
    component_registry = get_registry()
    all_registered_configs = {}
    for config_name in component_registry.list_configs():
        config_class = component_registry.get_config_class(config_name)
        if config_class:
            all_registered_configs[config_name] = config_class
    
    # Also scan modules for Configuration subclasses that might not be registered
    # This catches cases like MicroSAMSegmenterConfig that might not follow the naming convention
    all_config_classes = set(all_registered_configs.values())
    modules_to_scan = ["vistiq.preprocess", "vistiq.seg", "vistiq.analysis", "vistiq.train", "vistiq.core"]
    for module_name in modules_to_scan:
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, Configuration) and 
                    obj is not Configuration and
                    obj not in all_config_classes):
                    all_config_classes.add(obj)
        except ImportError:
            pass
    
    # For each subcommand, find all config classes that are subclasses of the allowed types
    expanded_registry: dict[str, list[type]] = {}
    for subcommand_name, allowed_types in registry.items():
        expanded_registry[subcommand_name] = list(allowed_types)  # Start with base types
        
        # Find all subclasses of the allowed types
        for config_class in all_config_classes:
            # Check if this config class is a subclass of any allowed type
            for allowed_type in allowed_types:
                if (isinstance(config_class, type) and 
                    issubclass(config_class, allowed_type) and 
                    config_class is not allowed_type):  # Don't add the base type again
                    if config_class not in expanded_registry[subcommand_name]:
                        expanded_registry[subcommand_name].append(config_class)
                    break  # Only add once per subcommand
    
    return expanded_registry

# Initialize the registry
_SUBCOMMAND_COMPONENT_REGISTRY = _build_subcommand_component_registry()


def args_to_config(args: argparse.Namespace) -> CLIAppConfig | CLISubcommandConfig:
    """Convert argparse namespace to appropriate Pydantic config model.

    Uses a dynamic registry of subcommand config classes to find the matching
    config based on the `subcommand` field default value.

    Args:
        args: Parsed command-line arguments.

    Returns:
        CLIAppConfig or CLISubcommandConfig subclass: Appropriate configuration model instance.
    """
    # Build base_kwargs from CLIAppConfig fields only
    base_kwargs = {
        "loglevel": getattr(args, "loglevel", "INFO").upper(),
        "device": getattr(args, "device", "auto"),
        "processes": getattr(args, "processes", 1),
    }
    
    # Find the matching subcommand config class by checking the default value of the subcommand field
    config_class = None
    for subcommand_config_class in _SUBCOMMAND_CONFIGS:
        # Get the default value of the subcommand field
        if hasattr(subcommand_config_class, "model_fields") and "subcommand" in subcommand_config_class.model_fields:
            field_info = subcommand_config_class.model_fields["subcommand"]
            default_value = getattr(field_info, "default", None)
            if default_value == args.command:
                config_class = subcommand_config_class
                break
    
    # If no matching config found, return base CLIAppConfig
    if config_class is None:
        return CLIAppConfig(**base_kwargs)
    
    # Use ConfigArgumentBuilder to build the config from args
    # This handles field extraction, defaults, and type conversion properly
    from .workflow_builder import ConfigArgumentBuilder
    
    # Build config using ConfigArgumentBuilder (without prefix for subcommand configs)
    config = ConfigArgumentBuilder.build_config_from_args(args, config_class, prefix="")
    
    # Set the subcommand field to match args.command
    config_dict = {"subcommand": args.command}
    
    # Handle special case: output_path defaults to current directory if not provided
    if hasattr(config, "output_path") and getattr(config, "output_path", None) is None:
        if hasattr(args, "output_path") and args.output_path:
            config_dict["output_path"] = Path(args.output_path)
        else:
            config_dict["output_path"] = Path.cwd()
    
    # Update with base_kwargs and subcommand
    # Use model_copy to update immutable Pydantic models
    update_dict = {**base_kwargs, **config_dict}
    if hasattr(config, "model_copy"):
        config = config.model_copy(update=update_dict)
    else:
        # Fallback for older Pydantic versions
        for key, value in update_dict.items():
            setattr(config, key, value)
    
    return config


def run_segment(config: CLISegmentConfig, args: Optional[argparse.Namespace] = None) -> None:
    """Run the segment command.

    Args:
        config: Segment configuration.
        args: Optional parsed command-line arguments (for component-based processing).
    """
    logger.info("Running segment command with config: %s", config)
    
    # Check if components are specified
    components = getattr(args, "components", []) if args else []
    
    if components:
        # Component-based processing (similar to workflow)
        logger.info("Using component-based segmentation")
        
        # Auto-register available components
        from .seg import Segmenter
        auto_register_configurables(Segmenter)
        
        # Build workflow builder
        builder = WorkflowBuilder()
        
        # Build each component
        built_components = []
        for i, component_name in enumerate(components):
            try:
                prefix = f"component{i}." if len(components) > 1 else ""
                component = builder.build_component(component_name, args, prefix=prefix)
                built_components.append(component)
                logger.info(f"Built component {i+1}/{len(components)}: {component_name}")
            except Exception as e:
                logger.error(f"Failed to build component '{component_name}': {e}")
                raise
        
        # Get absolute path of input and strip extension for output directory
        input_path_obj = Path(config.input_path).resolve()
        input_path_str = str(config.input_path)
        
        # Determine output directory (defaults to current directory if not specified)
        output_base = Path(config.output_path or Path.cwd()).resolve()
        
        scenes = get_scenes(input_path_str)
        for idx, sc in enumerate(scenes):
            logger.info(f"Processing scene: {sc}")
            # Load image with substack slicing if specified
            substack_slices = substack_to_slices(config.substack)
            img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.rename_channel)
            channel_names = metadata["channel_names"]
            
            # Process each channel
            for ch_idx, ch_name in enumerate(channel_names):
                logger.info(f"Processing channel: {ch_name}")
                channel_img = img[ch_idx] if len(img.shape) > 2 else img
                
                # Run components in sequence
                processed_img = channel_img
                for component in built_components:
                    processed_img = component.run(processed_img)
                
                # Save output
                output_dir = output_base / f"{input_path_obj.stem}_scene{idx}" / ch_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{input_path_obj.stem}_scene{idx}_{ch_name}_segmented.tif"
                OmeTiffWriter.save(processed_img, str(output_file), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(processed_img.ndim))
                logger.info(f"Saved segmented image to: {output_file}")
    else:
        # Legacy processing (use Segmenter directly)
        logger.info("Using legacy segmentation")
        from .seg import Segmenter, SegmenterConfig
        
        segmenter_config = SegmenterConfig(
            threshold_method=config.threshold_method,
            block_size=config.block_size,
            connectivity=config.connectivity,
            min_area=config.min_area,
            max_area=config.max_area,
        )
        segmenter = Segmenter.from_config(segmenter_config)
        
        # Get absolute path of input and strip extension for output directory
        input_path_obj = Path(config.input_path).resolve()
        input_path_str = str(config.input_path)
        
        # Determine output directory (defaults to current directory if not specified)
        output_base = Path(config.output_path or Path.cwd()).resolve()
        
        scenes = get_scenes(input_path_str)
        for idx, sc in enumerate(scenes):
            logger.info(f"Processing scene: {sc}")
            # Load image with substack slicing if specified
            substack_slices = substack_to_slices(config.substack)
            img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.rename_channel)
            channel_names = metadata["channel_names"]
            
            # Process each channel
            for ch_idx, ch_name in enumerate(channel_names):
                logger.info(f"Processing channel: {ch_name}")
                channel_img = img[ch_idx] if len(img.shape) > 2 else img
                
                # Run segmentation
                segmented = segmenter.run(channel_img)
                
                # Save output
                output_dir = output_base / f"{input_path_obj.stem}_scene{idx}" / ch_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{input_path_obj.stem}_scene{idx}_{ch_name}_segmented.tif"
                OmeTiffWriter.save(segmented, str(output_file), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(segmented.ndim))
                logger.info(f"Saved segmented image to: {output_file}")


def run_analyze(config: CLIAnalyzeConfig) -> None:
    """Run the analyze command.

    Args:
        config: Analyze configuration.
    """
    logger.info("Running analyze command with config: %s", config)
    
    from .analysis import RegionAnalyzer, RegionAnalyzerConfig
    
    # Create analyzer config
    analyzer_config = RegionAnalyzerConfig(
        properties=["centroid", "bbox", "volume", "area", "aspect_ratio", "sphericity"] if config.include_stats else [],
        include_coords=config.include_coords,
    )
    analyzer = RegionAnalyzer.from_config(analyzer_config)
    
    # Get absolute path of input
    input_path_obj = Path(config.input_path).resolve()
    input_path_str = str(config.input_path)
    
    # Determine output directory (defaults to current directory if not specified)
    output_base = Path(config.output_path or Path.cwd()).resolve()
    
    scenes = get_scenes(input_path_str)
    for idx, sc in enumerate(scenes):
        logger.info(f"Processing scene: {sc}")
        # Load image with substack slicing if specified
        substack_slices = substack_to_slices(config.substack)
        img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.rename_channel)
        
        # Analyze each channel
        for ch_idx, ch_name in enumerate(metadata["channel_names"]):
            logger.info(f"Analyzing channel: {ch_name}")
            channel_img = img[ch_idx] if len(img.shape) > 2 else img
            
            # Run analysis
            results = analyzer.run(channel_img)
            
            # Save output
            output_dir = output_base / f"{input_path_obj.stem}_scene{idx}" / ch_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{input_path_obj.stem}_scene{idx}_{ch_name}_analysis.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Saved analysis results to: {output_file}")


def run_train(config: CLITrainConfig, args: Optional[argparse.Namespace] = None) -> None:
    """Convert argparse namespace to appropriate Pydantic config model.

    Uses a dynamic registry of subcommand config classes to find the matching
    config based on the `subcommand` field default value.

    Args:
        args: Parsed command-line arguments.

    Returns:
        CLIAppConfig or CLISubcommandConfig subclass: Appropriate configuration model instance.
    """
    # Build base_kwargs from CLIAppConfig fields only
    base_kwargs = {
        "loglevel": getattr(args, "loglevel", "INFO").upper(),
        "device": getattr(args, "device", "auto"),
        "processes": getattr(args, "processes", 1),
    }
    
    # Find the matching subcommand config class by checking the default value of the subcommand field
    config_class = None
    for subcommand_config_class in _SUBCOMMAND_CONFIGS:
        # Get the default value of the subcommand field
        if hasattr(subcommand_config_class, "model_fields") and "subcommand" in subcommand_config_class.model_fields:
            field_info = subcommand_config_class.model_fields["subcommand"]
            default_value = getattr(field_info, "default", None)
            if default_value == args.command:
                config_class = subcommand_config_class
                break
    
    # If no matching config found, return base CLIAppConfig
    if config_class is None:
        return CLIAppConfig(**base_kwargs)
    
    # Use ConfigArgumentBuilder to build the config from args
    # This handles field extraction, defaults, and type conversion properly
    from .workflow_builder import ConfigArgumentBuilder
    
    # Build config using ConfigArgumentBuilder (without prefix for subcommand configs)
    config = ConfigArgumentBuilder.build_config_from_args(args, config_class, prefix="")
    
    # Set the subcommand field to match args.command
    config_dict = {"subcommand": args.command}
    
    # Handle special case: output_path defaults to current directory if not provided
    if hasattr(config, "output_path") and getattr(config, "output_path", None) is None:
        if hasattr(args, "output_path") and args.output_path:
            config_dict["output_path"] = Path(args.output_path)
        else:
            config_dict["output_path"] = Path.cwd()
    
    # Update with base_kwargs and subcommand
    # Use model_copy to update immutable Pydantic models
    update_dict = {**base_kwargs, **config_dict}
    if hasattr(config, "model_copy"):
        config = config.model_copy(update=update_dict)
    else:
        # Fallback for older Pydantic versions
        for key, value in update_dict.items():
            setattr(config, key, value)
    
    return config


def run_segment(config: CLISegmentConfig, args: Optional[argparse.Namespace] = None) -> None:
    """Run the segment command.

    Args:
        config: Segment configuration.
        args: Optional parsed command-line arguments (for component-based processing).
    """
    logger.info("Running segment command with config: %s", config)
    
    # Check if components are specified
    components = getattr(args, "components", []) if args else []
    
    if components:
        # Component-based processing (similar to workflow)
        logger.info("Using component-based segmentation")
        
        # Auto-register available components
        from .seg import Segmenter
        auto_register_configurables(Segmenter)
        
        # Build workflow builder
        builder = WorkflowBuilder()
        
        # Build each component
        built_components = []
        for i, component_name in enumerate(components):
            try:
                prefix = f"component{i}." if len(components) > 1 else ""
                component = builder.build_component(component_name, args, prefix=prefix)
                built_components.append(component)
                logger.info(f"Built component {i+1}/{len(components)}: {component_name}")
            except Exception as e:
                logger.error(f"Failed to build component '{component_name}': {e}")
                raise
        
        # Get absolute path of input and strip extension for output directory
        input_path_obj = Path(config.input_path).resolve()
        input_path_str = str(config.input_path)
        
        # Determine output directory (defaults to current directory if not specified)
        output_base = Path(config.output_path or Path.cwd()).resolve()
        
        scenes = get_scenes(input_path_str)
        for idx, sc in enumerate(scenes):
            logger.info(f"Processing scene: {sc}")
            # Load image with substack slicing if specified
            substack_slices = substack_to_slices(config.substack)
            img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.rename_channel)
            channel_names = metadata["channel_names"]
            
            ishape = str(img.shape).replace(" ", "")
            component_names = "-".join(components)
            output_dir = output_base / input_path_obj.stem / f"{sc}-{ishape}-{component_names}"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
        
            img_ch = np.unstack(img, axis=0)
            logger.info(f"Image shape: {img.shape}, channel names: {channel_names}, metadata: {metadata}")
            for im in img_ch:
                logger.debug(f"Channel shape: {im.shape}")

            labels_ch = []
            for ch_name, im in zip(channel_names, img_ch):
                logger.info(f"{''.join(metadata["used_scale"]._fields)}")
                # Run components in sequence
                result = im
                for i, component in enumerate(built_components):
                    logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()} on channel {ch_name}")
                    result = component.run(result, workers=config.processes, metadata=metadata)
                
                # Assume result is labels (could be enhanced to handle different output types)
                labels_ch.append(result)
            
            if config.split_channels:
                for labels, ch_name in zip(labels_ch, channel_names):
                    output_path = output_dir / f"Labels_{ch_name}.tif"
                    OmeTiffWriter.save(labels, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(labels.ndim))
            else:
                labels = np.stack(labels_ch, axis=0)
                output_path = output_dir / f"Labels-{"-".join(channel_names)}.tif"
                OmeTiffWriter.save(labels, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=channel_names, dim_order=_infer_dim_order(labels.ndim))
        return
    
    # Legacy behavior: MicroSAM segmentation
    logger.info("Using legacy MicroSAM segmentation")
    
    # Set up configurations
    volume_it_cfg = ArrayIteratorConfig(slice_def=(-3, -2, -1))

    region_analyzer_config = RegionAnalyzerConfig(properties=["centroid", "bbox", "volume", "cross_sectional_area", "aspect_ratio", "sphericity"], iterator_config=volume_it_cfg, output_type="dataframe")
    region_filter_config = RegionFilterConfig(filters=[])
    #    RangeFilter(RangeFilterConfig(attribute="volume", range=(config.volume_min, config.volume_max))),
    #    RangeFilter(RangeFilterConfig(attribute="cross_sectional_area", range=(config.cross_sectional_area_min, config.cross_sectional_area_max))),
    #    RangeFilter(RangeFilterConfig(attribute="aspect_ratio", range=(config.aspect_ratio_min, config.aspect_ratio_max)))
    #])

    # Get absolute path of input and strip extension for output directory
    input_path_obj = Path(config.input_path).resolve()
    input_path_str = str(config.input_path)

    # Determine output directory (defaults to current directory if not specified)
    output_base = Path(config.output_path or Path.cwd()).resolve()
    
    scenes = get_scenes(input_path_str)
    for idx, sc in enumerate(scenes):
        logger.info(f"Processing scene: {sc}")
        # Load image with substack slicing if specified
        substack_slices = substack_to_slices(config.substack)
        img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.rename_channel)
        
        channel_names = metadata["channel_names"]
        # img = img[:, 30:40, ]

        ishape = str(img.shape).replace(" ", "")
        output_dir = output_base / input_path_obj.stem / f"{sc}-{ishape}-microsam-{config.model_type}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
        img_ch = np.unstack(img, axis=0)
        logger.info(f"Image shape: {img.shape}, channel names: {channel_names}, metadata: {metadata}")
        for im in img_ch:
            logger.debug(f"Channel shape: {im.shape}")

        labels_ch = []
        #feature_dfs = {}

        for ch_name, im in zip(channel_names, img_ch):
            logger.debug(f"metadata[used_scale]: {''.join(metadata["used_scale"]._fields)}")
            logger.debug(f"type(im): {type(im)}, shape: {im.shape}, ch_name: {ch_name}")

            # MicroSAMSegmenter with RegionFilter
            region_filter = RegionFilter(region_filter_config)
            region_analyzer = RegionAnalyzer(region_analyzer_config)
            
            # Set up new microsam config for each channel to deal with different embeddings
            microsam_config = MicroSAMSegmenterConfig(
                model_type=config.model_type,
                embedding_path=str(output_dir / f"embeddings_{ch_name}"),
                region_analyzer=region_analyzer,
                region_filter=region_filter,
                do_labels=True,
                do_regions=True
            )

            microsam = MicroSAMSegmenter(microsam_config)
            _, labels, regions = microsam.run(im, metadata=metadata)
            labels_ch.append(labels)
            # feature_dfs[ch_name] = [regions]
            # logger.info (f"Feature DataFrame: {ch_name}: {regions.head()}")
        if config.split_channels:
            for ch_name, labels in zip(channel_names, labels_ch):
                output_path = output_dir / f"Labels_{ch_name}.tif"
                OmeTiffWriter.save(labels, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(labels.ndim))
        else:
            labels = np.stack(labels_ch, axis=0)
            output_path = output_dir / f"Labels-{"-".join(channel_names)}.tif"
            OmeTiffWriter.save(labels, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=channel_names, dim_order=_infer_dim_order(labels.ndim))




def run_analyze(config: CLIAnalyzeConfig) -> None:
    """Run the analyze command.

    Args:
        config: Analyze configuration.
    """
    logger.info("Running analyze command with config: %s", config)
    logger.info("Opening: %s", config.input_path)

    # TODO: Implement analysis logic
    logger.info("Analysis not yet implemented")
    logger.info("Would analyze with stats=%s, coords=%s", config.include_stats, config.include_coords)




def run_train(config: CLITrainConfig, args: Optional[argparse.Namespace] = None) -> None:
    """Run the training workflow.
    
    Args:
        config: Train configuration.
        args: Optional parsed command-line arguments (for component-based processing).
    """
    logger.info(f"Running train command with config: {config}")
    
    # Check if components are specified
    components = getattr(args, "components", []) if args else []
    
    if components:
        # Component-based processing
        logger.info("Using component-based training")
        
        # Auto-register available components
        from .train import Trainer
        auto_register_configurables(Trainer)
        
        # Build workflow builder
        builder = WorkflowBuilder()
        
        # Build each component
        built_components = []
        for i, component_name in enumerate(components):
            try:
                prefix = f"component{i}." if len(components) > 1 else ""
                component = builder.build_component(component_name, args, prefix=prefix)
                built_components.append(component)
                logger.info(f"Built component {i+1}/{len(components)}: {component_name}")
            except Exception as e:
                logger.error(f"Failed to build component '{component_name}': {e}")
                raise
        
        # Run components in sequence
        # Note: Training components may have different input/output requirements
        # This is a basic implementation that may need refinement
        result = None
        for i, component in enumerate(built_components):
            logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()}")
            if i == 0:
                # First component typically takes image_path and label_path
                result = component.run(config.image_path, config.label_path)
            else:
                # Subsequent components take the result from previous component
                result = component.run(result)
        
        logger.info("Component-based training completed")
        return
    
    # Parse patch_shape from string to tuple
    patch_shape = tuple(int(x.strip()) for x in config.patch_shape.split(","))
        
    # Parse patterns if provided
    patterns_tuple = None
    if config.patterns:
        logger.debug(f"Parsing patterns from: {repr(config.patterns)}")
        # Split by comma, but handle quoted strings if needed
        pattern_parts = [p.strip() for p in config.patterns.split(",")]
        logger.debug(f"Split pattern parts: {pattern_parts} (length: {len(pattern_parts)})")
        
        if len(pattern_parts) == 2:
            # Convert empty strings or "*" to None (no filter)
            pattern_a_raw = pattern_parts[0]
            pattern_b_raw = pattern_parts[1]
            
            # Handle empty strings and wildcard
            pattern_a = pattern_a_raw if pattern_a_raw and pattern_a_raw != "*" else None
            pattern_b = pattern_b_raw if pattern_b_raw and pattern_b_raw != "*" else None
            
            patterns_tuple = (pattern_a, pattern_b)
            logger.info(f"Using patterns: image_pattern={repr(pattern_a)}, label_pattern={repr(pattern_b)}")
        elif len(pattern_parts) == 1:
            logger.warning(
                "Patterns must be comma-separated with exactly 2 values (image_pattern,label_pattern). "
                "Got only 1 value: %s. Ignoring patterns.",
                config.patterns
            )
        else:
            logger.warning(
                "Patterns must be comma-separated with exactly 2 values (image_pattern,label_pattern). "
                "Got %d values: %s. Ignoring patterns.",
                len(pattern_parts),
                config.patterns
            )
    else:
        logger.debug("No patterns provided, will match all files")
    
    dc = DatasetCreator(DatasetCreatorConfig(out_path="./training_data", patterns=patterns_tuple, remove_empty_labels=True, random_samples=5, exclude=["checkpoints", "training_data"]))
    img_label_pairs = dc.run(config.image_path, config.label_path)

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
    microsam_trainer_config = MicroSAMTrainerConfig(
        model_type=config.model_type,
        checkpoint_name=config.checkpoint_name,
        export_path=config.export_path,
        batch_size=config.batch_size,
        patch_shape=patch_shape,
        roi_def=np.s_[:, :, :],  # Default: all slices
        device=config.device,
        split_ratio=config.split_ratio,
        raw_key=config.raw_key,
        label_key=config.label_key,
        learning_rate=config.learning_rate,
        log_image_interval=config.log_image_interval,
        n_objects_per_batch=config.n_objects_per_batch,
        n_iterations=config.n_iterations,
        n_sub_iteration=config.n_sub_iteration,
        mixed_precision=config.mixed_precision,
        compile_model=config.compile_model,
        verbose=config.verbose,
    )

    trainer = MicroSAMTrainer(microsam_trainer_config)
    trainer.run(image_paths, label_paths)


def run_coincidence(config: CLICoincidenceConfig) -> None:
    """Run the coincidence detection workflow.
    
    Args:
        config: Coincidence configuration.
    """
    logger.info(f"Running coincidence command with config: {config}")
    
    # Set up configurations
    volume_it_cfg = ArrayIteratorConfig(slice_def=(-3, -2, -1))

    dog_config = DoGConfig(
        sigma_low=config.sigma_low,
        sigma_high=config.sigma_high,
        normalize=config.normalize
    )
    region_analyzer_config = RegionAnalyzerConfig(properties=["centroid", "bbox", "volume", "cross_sectional_area", "aspect_ratio", "sphericity"], iterator_config=volume_it_cfg, output_type="dataframe")
    region_filter_config = RegionFilterConfig(filters=[
        RangeFilter(RangeFilterConfig(attribute="volume", range=(config.volume_min, config.volume_max))),
        RangeFilter(RangeFilterConfig(attribute="cross_sectional_area", range=(config.cross_sectional_area_min, config.cross_sectional_area_max))),
        RangeFilter(RangeFilterConfig(attribute="aspect_ratio", range=(config.aspect_ratio_min, config.aspect_ratio_max)))
    ])
    coincidence_config = CoincidenceDetectorConfig(
        method=config.method,
        mode=config.mode,
        iterator_config=volume_it_cfg,
        threshold=config.threshold
    )

    # Get absolute path of input and strip extension for output directory
    input_path_obj = Path(config.input_path).resolve()
    input_path_str = str(config.input_path)

    # Determine output directory (defaults to current directory if not specified)
    output_base = Path(config.output_path or Path.cwd()).resolve()
    
    scenes = get_scenes(input_path_str)
    for idx, sc in enumerate(scenes):
        logger.info(f"Processing scene: {sc}")
        # Load image with substack slicing if specified
        substack_slices = substack_to_slices(config.substack)
        img, metadata = load_image(input_path_str, scene_index=idx, substack=substack_slices, squeeze=True, rename_channel=config.rename_channel)
        
        channel_names = metadata["channel_names"]
        # img = img[:, 30:40, ]

        ishape = str(img.shape).replace(" ", "")
        output_dir = output_base / input_path_obj.stem / f"{sc}-{ishape}-dog-{config.sigma_low}-{config.sigma_high}-threshold-{config.threshold}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
        img_ch = np.unstack(img, axis=0)
        logger.info(f"Image shape: {img.shape}, channel names: {channel_names}, metadata: {metadata}")
        for im in img_ch:
            logger.debug(f"Channel shape: {im.shape}")

        labels_ch = []
        feature_dfs = {}

        for ch_name, im in zip(channel_names, img_ch):
            logger.info(f"{''.join(metadata["used_scale"]._fields)}")
            # 1. DoG preprocessing
            dog = DoG(dog_config)
            preprocessed = dog.run(im)
            output_path = output_dir / f"Preprocessed_{ch_name}.tif"
            OmeTiffWriter.save(preprocessed, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(preprocessed.ndim))

            # 2. MicroSAMSegmenter with RegionFilter
            region_filter = RegionFilter(region_filter_config)
            region_analyzer = RegionAnalyzer(region_analyzer_config)
            # need to set up new microsam config for each channel to deal with different embeddings
            microsam_config = MicroSAMSegmenterConfig(
                model_type=config.model_type,
                embedding_path=str(output_dir / f"embeddings_{ch_name}"),
                region_analyzer=region_analyzer,
                region_filter=region_filter,
                do_labels=True,
                do_regions=True
            )

            microsam = MicroSAMSegmenter(microsam_config)
            _, labels, regions = microsam.run(preprocessed, metadata=metadata)
            output_path = output_dir / f"Labels_{ch_name}.tif"
            OmeTiffWriter.save(labels, str(output_path), physical_pixel_sizes=metadata["used_scale"], channel_names=[ch_name], dim_order=_infer_dim_order(labels.ndim))
            labels_ch.append(labels)
            feature_dfs[ch_name] = [regions]
            logger.info (f"Feature DataFrame: {ch_name}: {regions.head()}")

            os.makedirs(output_dir / f"commits_{ch_name}", exist_ok=True)

        # 3. CoincidenceDetector
        # Create pairwise combinations of labels
        label_index_combinations = list(itertools.combinations(range(len(labels_ch)), 2))
        logger.info(f"Label index combinations: {label_index_combinations}")
        for idx_combination in label_index_combinations:
            coincidence_detector = CoincidenceDetector(coincidence_config)
            _, dfs = coincidence_detector.run(
                labels_ch[idx_combination[0]], 
                labels_ch[idx_combination[1]], 
                stack_names=(channel_names[idx_combination[0]], channel_names[idx_combination[1]])
            )
            for key, df in dfs.items():
                feature_dfs[key].append(df)
        
        # 4. Save features
        for key, dfs in feature_dfs.items():
            output_path = output_dir / f"Features_{key}.csv"
            logger.info(f"Saving to: {output_path}")
            dfs[0].join(dfs[1:]).to_csv(output_path, index=True)


def run_workflow(config: CLIAppConfig, args: argparse.Namespace) -> None:
    """Run a modular workflow built from CLI-specified components.
    
    Args:
        config: Base application configuration.
        args: Parsed command-line arguments with workflow component specifications.
    """
    # Auto-register available components
    auto_register_configurables(Configurable, [
        "vistiq.preprocess",
        "vistiq.seg",
        "vistiq.analysis",
        "vistiq.core"
    ])
    
    # Get workflow components from args
    components = getattr(args, "components", [])
    if not components:
        logger.error("No components specified for workflow")
        raise ValueError("At least one component must be specified with --component")
    
    # Build workflow builder
    builder = WorkflowBuilder()
    
    # Build each component
    built_components = []
    for i, component_name in enumerate(components):
        try:
            prefix = f"component{i}." if len(components) > 1 else ""
            component = builder.build_component(component_name, args, prefix=prefix)
            built_components.append(component)
            logger.info(f"Built component {i+1}/{len(components)}: {component_name}")
        except Exception as e:
            logger.error(f"Failed to build component '{component_name}': {e}")
            raise
    
    # Load input data
    if config.input_path:
        from .utils import load_image
        logger.info(f"Loading input from: {config.input_path}")
        # Convert substack string to dimension slices if specified
        substack_slices = substack_to_slices(config.substack)
        data = load_image(str(config.input_path), substack=substack_slices, rename_channel=config.rename_channel)
    else:
        logger.error("Input path required for workflow")
        raise ValueError("--input is required")
    
    # Run components in sequence
    result = data
    for i, component in enumerate(built_components):
        logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()}")
        result = component.run(result)
    
    # Save output (defaults to current directory if not specified)
    import numpy as np
    logger.info(f"Saving output to: {config.output_path}")
    np.save(str(config.output_path), result)


def main() -> None:
    """Entry point for the vistiq CLI.

    Parses command-line arguments, configures logging, and invokes the appropriate command.

    Returns:
        None
    """
    parser = build_parser()
    
    # First parse known arguments to get component specifications
    # This allows component-specific arguments to be unknown initially
    args, unknown = parser.parse_known_args()
    
    # If components are specified, dynamically add their arguments and re-parse
    if hasattr(args, 'components') and args.components:
        from .workflow_builder import get_registry, ConfigArgumentBuilder
        
        # Get the appropriate base class based on command
        if args.command == 'preprocess':
            from .preprocess import Preprocessor
            base_class = Preprocessor
        elif args.command == 'segment':
            from .seg import Segmenter
            base_class = Segmenter
        elif args.command == 'train':
            from .train import Trainer
            base_class = Trainer
        else:
            base_class = None
        
        if base_class:
            # Auto-register components
            auto_register_configurables(base_class)
            registry = get_registry()
            
            # Create a temporary parser for component arguments
            component_parser = argparse.ArgumentParser(add_help=False)
            
            # Add arguments for each specified component
            for component_name in args.components:
                configurable_class = registry.get_configurable_class(component_name)
                if configurable_class:
                    actual_name = configurable_class.__name__
                    config_class_name = f"{actual_name}Config"
                    config_class = registry.get_config_class(config_class_name)
                    if config_class:
                        ConfigArgumentBuilder.add_config_arguments(
                            component_parser,
                            config_class,
                            prefix=""
                        )
            
            # Parse unknown arguments with component parser
            component_args, remaining = component_parser.parse_known_args(unknown)
            
            # Merge component arguments into main args namespace
            for key, value in vars(component_args).items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
            
            # Warn about any remaining unknown arguments
            if remaining:
                logger.warning(f"Ignoring unknown arguments: {remaining}")
    
    # If there are still unknown arguments and no components, that's an error
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    # Configure logging as early as possible
    configure_logger(args.loglevel)

    # Convert args to Pydantic config
    try:
        config = args_to_config(args)
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        raise

    # Determine compute device - resolve "auto" to actual device
    if hasattr(config, 'device') and config.device == "auto":
        device_obj = check_device()
        device_str = str(device_obj)  # Convert torch.device to string (e.g., "cuda", "cpu", "mps")
        # Update the config object with the resolved device
        config = config.model_copy(update={"device": device_str})
        logger.info(f"Auto-selected device: {device_str}")
    elif hasattr(config, 'device'):
        logger.info(f"Using specified device: {config.device}")

    logger.info(f"vistiq invoked with command: {args.command}")
    logger.debug(f"Configuration: {config.model_dump()}")

    # Route to appropriate command handler
    if args.command == "preprocess":
        run_preprocess(config, args)  # type: ignore
    elif args.command == "segment":
        run_segment(config, args)  # type: ignore
    elif args.command == "analyze":
        run_analyze(config)  # type: ignore
    elif args.command == "full":
        run_full(config)  # type: ignore
    elif args.command == "coincidence":
        run_coincidence(config)  # type: ignore
    elif args.command == "train":
        run_train(config, args)  # type: ignore
    elif args.command == "workflow":
        run_workflow(config, args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
