import argparse
import logging
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .utils import load_mp4, check_device


logger = logging.getLogger(__name__)


def configure_logger(level: str = "INFO") -> logging.Logger:
    """Configure the logger.

    Args:
        level (str): The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: The configured logger instance.
    """
    lmap = logging.getLevelNamesMapping()
    level_int = lmap.get(level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_int, format="%(asctime)s - %(levelname)s - %(message)s"
    )
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
    """Base configuration for all vistiq commands."""

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
        """Get start frame index (0-based)."""
        start, _ = parse_frames(self.frames)
        return start

    @property
    def end_frame(self) -> Optional[int]:
        """Get end frame index (0-based, inclusive)."""
        _, end = parse_frames(self.frames)
        return end


class SegmentConfig(AppConfig):
    """Configuration for the segment subcommand."""

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
        if v is not None and v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v

    @field_validator("connectivity")
    @classmethod
    def validate_connectivity(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError("connectivity must be 1 or 2")
        return v


class AnalyzeConfig(AppConfig):
    """Configuration for the analyze subcommand."""

    include_stats: bool = Field(
        default=True, description="Include object statistics in analysis"
    )
    include_coords: bool = Field(
        default=True, description="Include coordinate extraction"
    )


class FullConfig(AppConfig):
    """Configuration for the full subcommand (segment + analyze)."""

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
        if v is not None and v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v

    @field_validator("connectivity")
    @classmethod
    def validate_connectivity(cls, v: int) -> int:
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

    return parser


def args_to_config(args: argparse.Namespace) -> AppConfig:
    """Convert argparse namespace to appropriate Pydantic config model.

    Args:
        args: Parsed command-line arguments.

    Returns:
        AppConfig or subclass: Appropriate configuration model instance.
    """
    # Convert Path strings to Path objects
    input_path = Path(args.input_path) if args.input_path else None
    output_path = Path(args.output_path) if args.output_path else None

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


def main() -> None:
    """Entry point for the vistiq CLI.

    Parses command-line arguments, configures logging, and invokes the appropriate command.

    Returns:
        None
    """
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging as early as possible
    logger_instance = configure_logger(args.loglevel)

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
    elif args.command == "full":
        run_full(config)  # type: ignore
    else:
        logger.error("Unknown command: %s", args.command)
        parser.print_help()
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
