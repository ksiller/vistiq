import argparse
import logging
import os
import itertools
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

import numpy as np

from .utils import load_mp4, check_device, load_image, get_scenes, to_tif
from .core import ArrayIteratorConfig
from .preprocess import DoG, DoGConfig
from .seg import MicroSAMSegmenter, MicroSAMSegmenterConfig, RegionFilter, RegionFilterConfig, RangeFilter, RangeFilterConfig
from .analysis import CoincidenceDetector, CoincidenceDetectorConfig
from .workflow_builder import (
    ComponentRegistry,
    get_registry,
    ConfigArgumentBuilder,
    WorkflowBuilder,
    auto_register_configurables
)


logger = logging.getLogger(__name__)


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


def parse_frames(frames: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """Parse frames string into start/end frame indices.

    Args:
        frames: Frame specification string like '10' or '2-40' (1-based, inclusive).

    Returns:
        Tuple of (start_frame, end_frame) as zero-based indices, or (None, None) if not specified.

    Raises:
        ValueError: If frames string is invalid.
    """
    if not frames:
        return None, None

    fs = str(frames).strip()
    if "-" in fs:
        a, b = fs.split("-", 1)
        if not a.isdigit() or not b.isdigit():
            raise ValueError(
                "--frames must be positive integers like '10' or '2-40'"
            )
        a_i = int(a)
        b_i = int(b)
        if a_i < 1 or b_i < 1:
            raise ValueError("--frames indices are 1-based and must be >= 1")
        if b_i < a_i:
            raise ValueError("--frames end must be >= start")
        # Convert to zero-based inclusive indices
        return a_i - 1, b_i - 1
    else:
        if not fs.isdigit():
            raise ValueError("--frames must be a positive integer or a range 'A-B'")
        n = int(fs)
        if n < 1:
            raise ValueError("--frames index is 1-based and must be >= 1")
        start_frame = n - 1
        return start_frame, start_frame


class AppConfig(BaseModel):
    """Base configuration for all vistiq commands.
    
    Provides common configuration options shared across all subcommands,
    including input/output paths, logging, and frame selection.
    
    Attributes:
        loglevel: Logging level for the application.
        input_path: Path to input file or directory.
        output_path: Path to output file or directory.
        grayscale: Whether to convert loaded images/videos to grayscale.
        frames: Frame specification string (e.g., '10' or '2-40').
    """

    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    input_path: Optional[Path] = Field(
        default=None, description="Input file or directory path"
    )
    output_path: Optional[Path] = Field(
        default=None, description="Output file or directory path"
    )
    grayscale: bool = Field(
        default=False, description="Convert loaded images/videos to grayscale"
    )
    frames: Optional[str] = Field(
        default=None, description="Frames to process (e.g., '10' or '2-40')"
    )

    @property
    def start_frame(self) -> Optional[int]:
        """Get start frame index (0-based).
        
        Returns:
            Start frame index (0-based), or None if not specified.
        """
        start, _ = parse_frames(self.frames)
        return start

    @property
    def end_frame(self) -> Optional[int]:
        """Get end frame index (0-based, inclusive).
        
        Returns:
            End frame index (0-based, inclusive), or None if not specified.
        """
        _, end = parse_frames(self.frames)
        return end


class SegmentConfig(AppConfig):
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


class AnalyzeConfig(AppConfig):
    """Configuration for the analyze subcommand.
    
    Defines parameters for analyzing segmented images and extracting
    object properties and coordinates.
    
    Attributes:
        include_stats: Whether to include object statistics in analysis.
        include_coords: Whether to include coordinate extraction.
    """

    include_stats: bool = Field(
        default=True, description="Include object statistics in analysis"
    )
    include_coords: bool = Field(
        default=True, description="Include coordinate extraction"
    )


class PreprocessConfig(AppConfig):
    """Configuration for the preprocess subcommand.
    
    Defines parameters for image preprocessing including denoising and
    Difference of Gaussians (DoG) filtering.
    
    Attributes:
        preprocess_method: Method to use for preprocessing ("dog", "noise2stack", or "none").
        sigma_low: Sigma for lower Gaussian blur in DoG (default: 1.0).
        sigma_high: Sigma for higher Gaussian blur in DoG (default: 5.0).
        mode: Border handling mode for Gaussian filtering (default: "reflect").
        window: Temporal window size for Noise2Stack denoising (default: 5).
        exclude_center: Exclude center frame from Noise2Stack average (default: True).
        normalize: Normalize output to [0, 1] range (default: True).
    """

    preprocess_method: Literal["dog", "noise2stack", "none"] = Field(
        default="dog", description="Preprocessing method"
    )
    sigma_low: float = Field(
        default=1.0, description="Sigma for lower Gaussian blur in DoG"
    )
    sigma_high: float = Field(
        default=5.0, description="Sigma for higher Gaussian blur in DoG"
    )
    mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = Field(
        default="reflect", description="Border handling mode for Gaussian filtering"
    )
    window: int = Field(
        default=5, description="Temporal window size for Noise2Stack denoising (odd recommended)"
    )
    exclude_center: bool = Field(
        default=True, description="Exclude center frame from Noise2Stack average"
    )
    normalize: bool = Field(
        default=True, description="Normalize output to [0, 1] range"
    )

    @field_validator("window")
    @classmethod
    def validate_window(cls, v: int) -> int:
        """Validate that window is >= 1.
        
        Args:
            v: Window size value to validate.
            
        Returns:
            Validated window size.
            
        Raises:
            ValueError: If window is < 1.
        """
        if v < 1:
            raise ValueError("window must be >= 1")
        return v

    @model_validator(mode="after")
    def validate_window_exclude_center(self) -> "PreprocessConfig":
        """Validate that window is >= 2 when exclude_center is True.
        
        Returns:
            Validated configuration.
            
        Raises:
            ValueError: If exclude_center is True and window < 2.
        """
        if self.exclude_center and self.window < 2:
            raise ValueError("window must be >= 2 when exclude_center=True")
        return self


class CoincidenceConfig(AppConfig):
    """Configuration for the coincidence subcommand.
    
    Defines parameters for running the coincidence detection workflow
    with DoG preprocessing, MicroSAM segmentation, and coincidence detection.
    
    Attributes:
        sigma_low: Sigma for lower Gaussian blur in DoG (default: 1.0).
        sigma_high: Sigma for higher Gaussian blur in DoG (default: 12.0).
        normalize: Normalize DoG output to [0, 1] range (default: True).
        area_lower: Lower bound for area filter (default: 100).
        area_upper: Upper bound for area filter (default: 10000).
        model_type: MicroSAM model type (default: "vit_l_lm").
        threshold: Threshold for coincidence detection (default: 0.1).
        method: Coincidence detection method: 'iou' or 'dice' (default: "dice").
        mode: Coincidence detection mode: 'bounding_box' or 'outline' (default: "outline").
    """

    sigma_low: float = Field(
        default=1.0, description="Sigma for lower Gaussian blur in DoG"
    )
    sigma_high: float = Field(
        default=12.0, description="Sigma for higher Gaussian blur in DoG"
    )
    normalize: bool = Field(
        default=True, description="Normalize DoG output to [0, 1] range"
    )
    area_lower: float = Field(
        default=100, description="Lower bound for area filter"
    )
    area_upper: float = Field(
        default=10000, description="Upper bound for area filter"
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


class FullConfig(AppConfig):
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


def build_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser with subcommands.

    Returns:
        argparse.ArgumentParser: Configured parser for the vistiq CLI.
    """
    parser = argparse.ArgumentParser(
        prog="vistiq",
        description=(
            "Turn complex imaging data into actionable, quantitative insight "
            "with modular, multi-step analysis."
        ),
    )

    # Common arguments for all subcommands
    parser.add_argument(
        "-l",
        "--loglevel",
        dest="loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Segment subcommand
    segment_parser = subparsers.add_parser(
        "segment", help="Segment images to identify objects"
    )
    segment_parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="single image file or directory with image files to be processed",
    )
    segment_parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="output file or directory",
    )
    segment_parser.add_argument(
        "-g",
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="convert loaded images/videos to grayscale before processing",
    )
    segment_parser.add_argument(
        "-f",
        "--frames",
        dest="frames",
        type=str,
        default=None,
        help="frames to process; examples: '10' or '2-40' (inclusive). Default: all frames",
    )
    segment_parser.add_argument(
        "--threshold-method",
        dest="threshold_method",
        type=str,
        default="otsu",
        choices=["otsu", "local", "niblack", "sauvola"],
        help="thresholding method to use",
    )
    segment_parser.add_argument(
        "--block-size",
        dest="block_size",
        type=int,
        default=51,
        help="block size for local thresholding (must be odd)",
    )
    segment_parser.add_argument(
        "--connectivity",
        dest="connectivity",
        type=int,
        default=1,
        choices=[1, 2],
        help="connectivity for labeling (1 or 2)",
    )
    segment_parser.add_argument(
        "--min-area",
        dest="min_area",
        type=float,
        default=None,
        help="minimum object area for filtering",
    )
    segment_parser.add_argument(
        "--max-area",
        dest="max_area",
        type=float,
        default=None,
        help="maximum object area for filtering",
    )

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze segmented images"
    )
    analyze_parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="input file or directory with segmented images",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="output file or directory",
    )
    analyze_parser.add_argument(
        "-g",
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="convert loaded images/videos to grayscale before processing",
    )
    analyze_parser.add_argument(
        "-f",
        "--frames",
        dest="frames",
        type=str,
        default=None,
        help="frames to process; examples: '10' or '2-40' (inclusive). Default: all frames",
    )
    analyze_parser.add_argument(
        "--include-stats",
        dest="include_stats",
        action="store_true",
        default=True,
        help="include object statistics in analysis",
    )
    analyze_parser.add_argument(
        "--no-stats",
        dest="include_stats",
        action="store_false",
        help="exclude object statistics from analysis",
    )
    analyze_parser.add_argument(
        "--include-coords",
        dest="include_coords",
        action="store_true",
        default=True,
        help="include coordinate extraction",
    )
    analyze_parser.add_argument(
        "--no-coords",
        dest="include_coords",
        action="store_false",
        help="exclude coordinate extraction",
    )

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess images with denoising and filtering"
    )
    preprocess_parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="single image file or directory with image files to be processed",
    )
    preprocess_parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="output file or directory",
    )
    preprocess_parser.add_argument(
        "-g",
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="convert loaded images/videos to grayscale before processing",
    )
    preprocess_parser.add_argument(
        "-f",
        "--frames",
        dest="frames",
        type=str,
        default=None,
        help="frames to process; examples: '10' or '2-40' (inclusive). Default: all frames",
    )
    preprocess_parser.add_argument(
        "--preprocess-method",
        dest="preprocess_method",
        type=str,
        default="dog",
        choices=["dog", "noise2stack", "none"],
        help="preprocessing method to use (default: dog)",
    )
    preprocess_parser.add_argument(
        "--sigma-low",
        dest="sigma_low",
        type=float,
        default=1.0,
        help="sigma for lower Gaussian blur in DoG (default: 1.0)",
    )
    preprocess_parser.add_argument(
        "--sigma-high",
        dest="sigma_high",
        type=float,
        default=5.0,
        help="sigma for higher Gaussian blur in DoG (default: 5.0)",
    )
    preprocess_parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="reflect",
        choices=["reflect", "constant", "nearest", "mirror", "wrap"],
        help="border handling mode for Gaussian filtering (default: reflect)",
    )
    preprocess_parser.add_argument(
        "--window",
        dest="window",
        type=int,
        default=5,
        help="temporal window size for Noise2Stack denoising (odd recommended, default: 5)",
    )
    preprocess_parser.add_argument(
        "--exclude-center",
        dest="exclude_center",
        action="store_true",
        default=True,
        help="exclude center frame from Noise2Stack average (default: True)",
    )
    preprocess_parser.add_argument(
        "--no-exclude-center",
        dest="exclude_center",
        action="store_false",
        help="include center frame in Noise2Stack average",
    )
    preprocess_parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="normalize output to [0, 1] range (default: True)",
    )
    preprocess_parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="do not normalize output",
    )

    # Full subcommand
    full_parser = subparsers.add_parser(
        "full", help="Run full pipeline (segment + analyze)"
    )
    full_parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="single image file or directory with image files to be processed",
    )
    full_parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="output file or directory",
    )
    full_parser.add_argument(
        "-g",
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="convert loaded images/videos to grayscale before processing",
    )
    full_parser.add_argument(
        "-f",
        "--frames",
        dest="frames",
        type=str,
        default=None,
        help="frames to process; examples: '10' or '2-40' (inclusive). Default: all frames",
    )
    full_parser.add_argument(
        "--threshold-method",
        dest="threshold_method",
        type=str,
        default="otsu",
        choices=["otsu", "local", "niblack", "sauvola"],
        help="thresholding method to use",
    )
    full_parser.add_argument(
        "--block-size",
        dest="block_size",
        type=int,
        default=51,
        help="block size for local thresholding (must be odd)",
    )
    full_parser.add_argument(
        "--connectivity",
        dest="connectivity",
        type=int,
        default=1,
        choices=[1, 2],
        help="connectivity for labeling (1 or 2)",
    )
    full_parser.add_argument(
        "--min-area",
        dest="min_area",
        type=float,
        default=None,
        help="minimum object area for filtering",
    )
    full_parser.add_argument(
        "--max-area",
        dest="max_area",
        type=float,
        default=None,
        help="maximum object area for filtering",
    )
    full_parser.add_argument(
        "--include-stats",
        dest="include_stats",
        action="store_true",
        default=True,
        help="include object statistics in analysis",
    )
    full_parser.add_argument(
        "--no-stats",
        dest="include_stats",
        action="store_false",
        help="exclude object statistics from analysis",
    )
    full_parser.add_argument(
        "--include-coords",
        dest="include_coords",
        action="store_true",
        default=True,
        help="include coordinate extraction",
    )
    full_parser.add_argument(
        "--no-coords",
        dest="include_coords",
        action="store_false",
        help="exclude coordinate extraction",
    )

    # Coincidence subcommand
    coincidence_parser = subparsers.add_parser(
        "coincidence", help="Run coincidence detection workflow with DoG preprocessing and MicroSAM segmentation"
    )
    coincidence_parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="input image file path",
    )
    coincidence_parser.add_argument(
        "--sigma-low",
        dest="sigma_low",
        type=float,
        default=1.0,
        help="sigma for lower Gaussian blur in DoG (default: 1.0)",
    )
    coincidence_parser.add_argument(
        "--sigma-high",
        dest="sigma_high",
        type=float,
        default=12.0,
        help="sigma for higher Gaussian blur in DoG (default: 12.0)",
    )
    coincidence_parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="normalize DoG output to [0, 1] range (default: True)",
    )
    coincidence_parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="disable normalization of DoG output",
    )
    coincidence_parser.add_argument(
        "--area-lower",
        dest="area_lower",
        type=float,
        default=100,
        help="lower bound for area filter (default: 100)",
    )
    coincidence_parser.add_argument(
        "--area-upper",
        dest="area_upper",
        type=float,
        default=10000,
        help="upper bound for area filter (default: 10000)",
    )
    coincidence_parser.add_argument(
        "--model-type",
        dest="model_type",
        type=str,
        default="vit_l_lm",
        choices=["vit_l_lm", "vit_b_lm", "vit_t_lm", "vit_h_lm"],
        help="MicroSAM model type (default: vit_l_lm)",
    )
    coincidence_parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=0.1,
        help="threshold for coincidence detection (default: 0.1)",
    )
    coincidence_parser.add_argument(
        "--method",
        dest="method",
        type=str,
        default="dice",
        choices=["iou", "dice"],
        help="coincidence detection method: 'iou' or 'dice' (default: dice)",
    )
    coincidence_parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="outline",
        choices=["bounding_box", "outline"],
        help="coincidence detection mode: 'bounding_box' or 'outline' (default: outline)",
    )

    # Workflow subcommand - modular workflow builder
    workflow_parser = subparsers.add_parser(
        "workflow", help="Build and run modular workflows from CLI-specified components"
    )
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
        "--frames",
        dest="frames",
        type=str,
        default=None,
        help="frames to process; examples: '10' or '2-40' (inclusive). Default: all frames",
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
    auto_register_configurables([
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


def args_to_config(args: argparse.Namespace) -> AppConfig:
    """Convert argparse namespace to appropriate Pydantic config model.

    Args:
        args: Parsed command-line arguments.

    Returns:
        AppConfig or subclass: Appropriate configuration model instance.
    """
    # Convert Path strings to Path objects
    input_path = Path(args.input_path) if getattr(args, "input_path", None) else None
    output_path = Path(args.output_path) if getattr(args, "output_path", None) else None

    base_kwargs = {
        "loglevel": args.loglevel.upper(),
        "input_path": input_path,
        "output_path": output_path,
        "grayscale": getattr(args, "grayscale", False),
        "frames": getattr(args, "frames", None),
    }

    if args.command == "segment":
        return SegmentConfig(
            **base_kwargs,
            threshold_method=getattr(args, "threshold_method", "otsu"),
            block_size=getattr(args, "block_size", 51),
            connectivity=getattr(args, "connectivity", 1),
            min_area=getattr(args, "min_area", None),
            max_area=getattr(args, "max_area", None),
        )
    elif args.command == "analyze":
        return AnalyzeConfig(
            **base_kwargs,
            include_stats=getattr(args, "include_stats", True),
            include_coords=getattr(args, "include_coords", True),
        )
    elif args.command == "preprocess":
        return PreprocessConfig(
            **base_kwargs,
            preprocess_method=getattr(args, "preprocess_method", "dog"),
            sigma_low=getattr(args, "sigma_low", 1.0),
            sigma_high=getattr(args, "sigma_high", 5.0),
            mode=getattr(args, "mode", "reflect"),
            window=getattr(args, "window", 5),
            exclude_center=getattr(args, "exclude_center", True),
            normalize=getattr(args, "normalize", True),
        )
    elif args.command == "full":
        return FullConfig(
            **base_kwargs,
            threshold_method=getattr(args, "threshold_method", "otsu"),
            block_size=getattr(args, "block_size", 51),
            connectivity=getattr(args, "connectivity", 1),
            min_area=getattr(args, "min_area", None),
            max_area=getattr(args, "max_area", None),
            include_stats=getattr(args, "include_stats", True),
            include_coords=getattr(args, "include_coords", True),
        )
    elif args.command == "coincidence":
        return CoincidenceConfig(
            **base_kwargs,
            sigma_low=getattr(args, "sigma_low", 1.0),
            sigma_high=getattr(args, "sigma_high", 12.0),
            normalize=getattr(args, "normalize", True),
            area_lower=getattr(args, "area_lower", 100),
            area_upper=getattr(args, "area_upper", 10000),
            model_type=getattr(args, "model_type", "vit_l_lm"),
            threshold=getattr(args, "threshold", 0.1),
            method=getattr(args, "method", "dice"),
            mode=getattr(args, "mode", "outline"),
        )
    else:
        return AppConfig(**base_kwargs)


def run_segment(config: SegmentConfig) -> None:
    """Run the segment command.

    Args:
        config: Segment configuration.
    """
    logger.info("Running segment command with config: %s", config)
    logger.info("Opening: %s", config.input_path)
    
    # Load image stack
    image_stack = load_mp4(
        str(config.input_path),
        to_grayscale=config.grayscale,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
    )
    logger.info("Image stack: %s", image_stack.shape)

    # TODO: Implement segmentation logic
    logger.info("Segmentation not yet implemented")
    logger.info("Would segment with method: %s", config.threshold_method)


def run_analyze(config: AnalyzeConfig) -> None:
    """Run the analyze command.

    Args:
        config: Analyze configuration.
    """
    logger.info("Running analyze command with config: %s", config)
    logger.info("Opening: %s", config.input_path)

    # TODO: Implement analysis logic
    logger.info("Analysis not yet implemented")
    logger.info("Would analyze with stats=%s, coords=%s", config.include_stats, config.include_coords)


def run_preprocess(config: PreprocessConfig) -> None:
    """Run the preprocess command.

    Args:
        config: Preprocess configuration.
    """
    logger.info("Running preprocess command with config: %s", config)
    logger.info("Opening: %s", config.input_path)
    
    # Load image stack
    image_stack = load_mp4(
        str(config.input_path),
        to_grayscale=config.grayscale,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
    )
    logger.info("Image stack: %s", image_stack.shape)

    # TODO: Implement preprocessing logic
    logger.info("Preprocessing not yet implemented")
    logger.info("Would preprocess with method: %s", config.preprocess_method)
    if config.preprocess_method == "dog":
        logger.info("DoG parameters: sigma_low=%s, sigma_high=%s, mode=%s", 
                   config.sigma_low, config.sigma_high, config.mode)
    elif config.preprocess_method == "noise2stack":
        logger.info("Noise2Stack parameters: window=%s, exclude_center=%s", 
                   config.window, config.exclude_center)


def run_full(config: FullConfig) -> None:
    """Run the full pipeline (segment + analyze).

    Args:
        config: Full configuration.
    """
    logger.info("Running full pipeline with config: %s", config)
    
    # Run segment first
    segment_config = SegmentConfig(
        loglevel=config.loglevel,
        input_path=config.input_path,
        output_path=config.output_path,  # Could be intermediate path
        grayscale=config.grayscale,
        frames=config.frames,
        threshold_method=config.threshold_method,
        block_size=config.block_size,
        connectivity=config.connectivity,
        min_area=config.min_area,
        max_area=config.max_area,
    )
    run_segment(segment_config)

    # Then run analyze
    analyze_config = AnalyzeConfig(
        loglevel=config.loglevel,
        input_path=config.output_path,  # Use segment output as analyze input
        output_path=config.output_path,
        grayscale=config.grayscale,
        frames=config.frames,
        include_stats=config.include_stats,
        include_coords=config.include_coords,
    )
    run_analyze(analyze_config)


def run_coincidence(config: CoincidenceConfig) -> None:
    """Run the coincidence detection workflow.
    
    Args:
        config: Coincidence configuration.
    """
    logger.info("Running coincidence command with config: %s", config)
    
    # Set up configurations
    dog_config = DoGConfig(
        sigma_low=config.sigma_low,
        sigma_high=config.sigma_high,
        normalize=config.normalize
    )
    region_filter_config = RegionFilterConfig(
        filters=[RangeFilter(RangeFilterConfig(attribute="area", range=(config.area_lower, config.area_upper)))]
    )
    volume_it_cfg = ArrayIteratorConfig(slice_def=(-3, -2, -1))
    coincidence_config = CoincidenceDetectorConfig(
        method=config.method,
        mode=config.mode,
        iterator_config=volume_it_cfg,
        threshold=config.threshold
    )

    # Get absolute path of input and strip extension for output directory
    input_path_obj = Path(config.input_path).resolve()
    input_path_str = str(config.input_path)

    scenes = get_scenes(input_path_str)
    for idx, sc in enumerate(scenes):
        logger.info(f"Processing scene: {sc}")
        # Load image
        img, scale = load_image(input_path_str, scene_index=idx, squeeze=True)
        # img = img[:, 30:40, ]

        output_dir = f"{input_path_obj.with_suffix('')}-{sc}-dog-{config.sigma_low}-{config.sigma_high}-threshold-{config.threshold}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
        img_ch = np.unstack(img, axis=0)
        logger.info(f"Image shape: {img.shape}, scale: {scale}")
        for im in img_ch:
            logger.debug(f"Channel shape: {im.shape}")

        masks_ch = []
        labels_ch = []
        regions_ch = []

        for i, im in enumerate(img_ch):
            # 1. DoG preprocessing
            dog = DoG(dog_config)
            preprocessed = dog.run(im)
            output_path = f"{output_dir}/Preprocessed_Ch{i}.tif"
            to_tif(output_path, preprocessed)

            # 2. MicroSAMSegmenter with RegionFilter
            region_filter = RegionFilter(region_filter_config)
            # need to set up new microsam config for each channel to deal with different embeddings
            microsam_config = MicroSAMSegmenterConfig(
                model_type=config.model_type,
                region_filter=region_filter,
                do_labels=True,
                do_regions=True
            )

            microsam = MicroSAMSegmenter(microsam_config)
            masks, labels, regions = microsam.run(preprocessed)
            output_path = f"{output_dir}/Labels_Ch{i}.tif"
            to_tif(output_path, labels)
            masks_ch.append(masks)
            labels_ch.append(labels)
            regions_ch.append(regions)

        # 3. CoincidenceDetector
        # Create pairwise combinations of labels
        label_index_combinations = list(itertools.combinations(range(len(labels_ch)), 2))
        logger.info(f"Label index combinations: {label_index_combinations}")
        feature_dfs = {}
        for idx_combination in label_index_combinations:
            coincidence_detector = CoincidenceDetector(coincidence_config)
            _, dfs = coincidence_detector.run(
                labels_ch[idx_combination[0]], 
                labels_ch[idx_combination[1]], 
                stack_names=(f"Ch{idx_combination[0]}", f"Ch{idx_combination[1]}")
            )
            for key, df in dfs.items():
                if key not in feature_dfs:
                    feature_dfs[key] = [df]
                else:
                    feature_dfs[key].append(df)
        for key, dfs in feature_dfs.items():
            output_path = f"{output_dir}/Coincidence_{key}.csv"
            logger.info(f"Saving to: {output_path}")
            dfs[0].join(dfs[1:]).to_csv(output_path, index=True)


def run_workflow(config: AppConfig, args: argparse.Namespace) -> None:
    """Run a modular workflow built from CLI-specified components.
    
    Args:
        config: Base application configuration.
        args: Parsed command-line arguments with workflow component specifications.
    """
    # Auto-register available components
    auto_register_configurables([
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
        # For now, assume single image - could be extended for stacks
        data = load_image(str(config.input_path))
    else:
        logger.error("Input path required for workflow")
        raise ValueError("--input is required")
    
    # Run components in sequence
    result = data
    for i, component in enumerate(built_components):
        logger.info(f"Running component {i+1}/{len(built_components)}: {component.name()}")
        result = component.run(result)
    
    # Save output if specified
    if config.output_path:
        import numpy as np
        logger.info(f"Saving output to: {config.output_path}")
        np.save(str(config.output_path), result)
    else:
        logger.warning("No output path specified, results not saved")


def main() -> None:
    """Entry point for the vistiq CLI.

    Parses command-line arguments, configures logging, and invokes the appropriate command.

    Returns:
        None
    """
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging as early as possible
    configure_logger(args.loglevel)

    # Convert args to Pydantic config
    try:
        config = args_to_config(args)
    except Exception as e:
        logger.error("Failed to create configuration: %s", e)
        raise

    # Determine compute device
    device = check_device()
    logger.info("Selected device: %s", device)

    logger.info("vistiq invoked with command: %s", args.command)
    logger.debug("Configuration: %s", config.model_dump())

    # Route to appropriate command handler
    if args.command == "segment":
        run_segment(config)  # type: ignore
    elif args.command == "analyze":
        run_analyze(config)  # type: ignore
    elif args.command == "preprocess":
        run_preprocess(config)  # type: ignore
    elif args.command == "full":
        run_full(config)  # type: ignore
    elif args.command == "coincidence":
        run_coincidence(config)  # type: ignore
    elif args.command == "workflow":
        run_workflow(config, args)
    else:
        logger.error("Unknown command: %s", args.command)
        parser.print_help()
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
