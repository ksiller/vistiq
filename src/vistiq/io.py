from __future__ import annotations

import copy
import fnmatch
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union, Any, List, Literal
import numpy as np
import pandas as pd
import os
from pydantic import Field, field_validator
from bioio import Dimensions,Scale
from bioio_ome_tiff.writers import OmeTiffWriter
from vistiq.core import Configuration, Configurable
from prefect import task
from vistiq.utils import str_to_dict, NamedTuple

logger = logging.getLogger(__name__)

def unstack_image(data: np.ndarray, metadata: dict[str, Any], axis: Union[int, str], key:str="axes") -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Split image data and metadata along a given axis.
    
    Splits the image array along the specified axis and updates metadata accordingly.
    When splitting along the channel axis ("C"), each resulting array gets its own
    channel name from the metadata.
    
    Args:
        data: Image data array to split.
        metadata: Image metadata dictionary containing axes information and other metadata.
        axis: Axis to split along. Can be an integer index or a string axis label (e.g., "C", "T", "Z").
        key: Key in metadata dictionary that contains the axes labels. Default is "axes".
        
    Returns:
        Tuple of (list of split image arrays, list of metadata dictionaries).
        Each metadata dictionary corresponds to one split array. When splitting channels,
        each metadata dict contains a single channel name.
        
    Examples:
        Example 1: Split along channel axis
            Input: data shape (3, 100, 200), metadata["axes"] = ["C", "Y", "X"], 
                   metadata["channel_names"] = ["Red", "Green", "Blue"], axis = "C"
            Output: 3 arrays of shape (100, 200), each with metadata["channel_names"] = ["Red"], ["Green"], ["Blue"]
        
        Example 2: Split along time axis
            Input: data shape (10, 100, 200), metadata["axes"] = ["T", "Y", "X"], axis = "T"
            Output: 10 arrays of shape (100, 200), each with metadata["axes"] = ["Y", "X"]
        
        Example 3: Split along axis by index
            Input: data shape (5, 10, 20), metadata["axes"] = ["Z", "Y", "X"], axis = 0
            Output: 5 arrays of shape (10, 20), each with metadata["axes"] = ["Y", "X"]
    """
    if isinstance(axis, str):
        axis_idx = metadata[key].index(axis)
    else:
        axis_idx = axis
    data = np.unstack(data, axis=axis_idx)
    new_shape = data[0].shape
    logger.info(f"Unstacked image data along {axis} axis with index {axis_idx}. Data shapes: {[data[i].shape for i in range(len(data))]}")
    metadata = copy.deepcopy(metadata)
    splitting_channels = axis_idx == metadata[key].index("C")
    if splitting_channels:
        if "channel_names" in metadata:
            channel_names = metadata["channel_names"]
            assert len(channel_names) == len(data), "Number of channel names must match number of data arrays"
        else:
            channel_names = [f"Channel {i}" for i in range(len(data))]
    if "shape" in metadata:
        metadata["shape"] = new_shape
    if key in metadata:
        axes = [axis for i, axis in enumerate(metadata[key]) if i != axis_idx]
        dim_str = "".join(axes)
        metadata[key] = axes
    else:
        axes = []
        dim_str = ""
    if "scale" in metadata:
        # metadata["scale"] = [axis for i, axis in enumerate(metadata["scale"]) if i != axis_idx]
        metadata["scale"] = Scale(**{s:v if s in axes else None for i, (s,v) in enumerate(metadata["scale"]._asdict().items())})
    if "dims" in metadata:
        metadata["dims"] = Dimensions(axes, new_shape)
    if "dim_order" in metadata:
        metadata["dim_order"] = dim_str
    if splitting_channels:
        new_metadata = [copy.deepcopy(metadata)|{"channel_names": [channel_names[i]]} for i in range(len(data))]
    else:
        new_metadata = [copy.deepcopy(metadata) for _ in range(len(data))]
    return data, new_metadata

def squeeze_image(data: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Squeeze image data and metadata by removing singleton dimensions.
    
    Removes dimensions of size 1 from the image array and updates metadata
    accordingly, including axes labels, shape, scale, and dimension order.
    
    Args:
        data: Image data array to squeeze.
        metadata: Image metadata dictionary containing axes, shape, scale, and other information.
        
    Returns:
        Tuple of (squeezed image data, updated metadata dictionary).
        The metadata is updated to reflect the new shape and axes after removing
        singleton dimensions. If the channel axis is removed, channel_axis is set to None.
        
    Examples:
        Example 1: Remove singleton time dimension
            Input: data shape (1, 10, 20, 30), metadata["axes"] = ["T", "Z", "Y", "X"]
            Output: data shape (10, 20, 30), metadata["axes"] = ["Z", "Y", "X"]
        
        Example 2: Remove multiple singleton dimensions
            Input: data shape (1, 5, 1, 100, 200), metadata["axes"] = ["T", "C", "Z", "Y", "X"]
            Output: data shape (5, 100, 200), metadata["axes"] = ["C", "Y", "X"]
        
        Example 3: No singleton dimensions
            Input: data shape (3, 100, 200), metadata["axes"] = ["C", "Y", "X"]
            Output: data shape (3, 100, 200), metadata["axes"] = ["C", "Y", "X"] (unchanged)
    """
    singular_axes = [index for index, value in enumerate(data.shape) if value == 1]
    data = np.squeeze(data)
    metadata = metadata.copy()
    if "shape" in metadata:
        metadata["shape"] = data.shape
    if "axes" in metadata:
        axes = [axis for i, axis in enumerate(metadata["axes"]) if i not in singular_axes]
        metadata["axes"] = axes
    else:
        axes = []
    if "scale" in metadata:
        # metadata["scale"] = [axis for i, axis in enumerate(metadata["scale"]) if i not in singular_axes]
        metadata["scale"] = Scale(**{s:v if s in axes else None for s,v in metadata["scale"]._asdict().items()})
    if "channel_axis" in metadata:
        if "C" in metadata["axes"]:
            metadata["channel_axis"] = metadata["axes"].index("C")
        else:
            metadata["channel_axis"] = None
    if "dims" in metadata:
        metadata["dims"] = Dimensions("".join(metadata["axes"]), data.shape)
    if "dim_order" in metadata:
        metadata["dim_order"] = "".join(metadata["axes"])
    return data, metadata

def unsqueeze_image(data: np.ndarray, metadata: dict[str, Any], target:Union[str, list[str]], key:str="axes") -> tuple[np.ndarray, dict[str, Any]]:
    """Unsqueeze image data and metadata.
    
    Args:
        data: Image data.
        metadata: Image metadata.
        key: Key to use to look up axes labels in the metadata and to compare with target. Typically keyed as"axes", "dim_order" with "ZYX" or ["Z", "Y", "X"].
        target: Target axes to unsqueeze. Formatted as string "TCYYX" or list ["T", "C", "Y", "Y", "X"]
        
    Examples:
        Example 1: Add time dimension to 3D image
            Input: data shape (10, 20, 30), metadata["axes"] = "ZYX", target = "TZYX"
            Output: data shape (1, 10, 20, 30), metadata["axes"] = "TZYX"
        
        Example 2: Add channel and time dimensions to 2D image
            Input: data shape (100, 200), metadata["axes"] = "YX", target = "TCYX"
            Output: data shape (1, 1, 100, 200), metadata["axes"] = "TCYX"
        
        Example 3: Add missing dimensions to match target
            Input: data shape (5, 10, 20), metadata["axes"] = "CYX", target = "TCZYX"
            Output: data shape (1, 5, 1, 10, 20), metadata["axes"] = "TCZYX"
        
    Returns:
        Tuple of (image data, image metadata).
    """
    if isinstance(target, str):
        target = [axis for axis in target]
    # find the indices of the missing axes in the target
    missing_axes = [i for i, axis in enumerate(target) if axis not in metadata[key]]
    # start from the end of the data array and add the missing axes in reverse order
    missing_axes.sort(reverse=True)
    for idx in missing_axes:
        data = np.expand_dims(data, axis=idx)
    if isinstance(metadata[key], str):
        metadata[key] = "".join(target)
    else:
        metadata[key] = target
    if "shape" in metadata:
        metadata["shape"] = data.shape
    if "scale" in metadata:
        #metadata["scale"] = [axis for i, axis in enumerate(metadata["scale"]) if i != idx]
        metadata["scale"] = Scale(**{s:v if data.shape[i] != 1 else None for i, (s,v) in enumerate(metadata["scale"]._asdict().items())})
    if "channel_axis" in metadata and "C" in target:
        metadata["channel_axis"] = target.index("C")
    if "dims" in metadata:
        metadata["dims"] = Dimensions("".join(target), data.shape)
    if "dim_order" in metadata:
        metadata["dim_order"] = "".join(target)
    return data, metadata

class FileListConfig(Configuration):
    """Configuration for file list operations.
    
    This configuration defines parameters for creating and filtering lists of files,
    including path specifications, include/exclude patterns, and filtering options.
    
    Attributes:
        paths: List of paths (files or directories) to search for files.
        include: Optional string or list of patterns to include files (e.g., '*.tif' or ['*.tif', '*.tiff']).
        exclude: Optional string or list of patterns to exclude files (e.g., '*.tmp' or ['*.tmp', '*.bak']).
    """
    paths: Union[str, Path, List[Union[str, Path]]] = Field(
        description="List of paths (files or directories) to search for files"
    )
    recursive: bool = Field(
        default=True,
        description="Whether to recursively search for files"
    )
    follow_symlinks: bool = Field(
        default=True,
        description="Whether to follow symlinks"
    )
    include: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="String or list of file patterns to include (e.g., '*.tif' or ['*.tif', '*.csv'])"
    )
    exclude: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="String or list of file patterns to exclude (e.g., '*.bak' or ['*.bak', '*.tmp'])"
    )    
    skip_on_error: bool = Field(
        default=True,
        description="Whether to skip paths that do not exist"
    )
    
    @field_validator('paths', mode='before')
    @classmethod
    def normalize_paths(cls, v: Any) -> List[Path]:
        """Normalize paths to always be a list of Path objects.
        
        Converts all strings to Path objects. Does NOT expand '~/' to preserve
        portability across different users. Expansion happens at runtime in the run method.
        
        Args:
            v: Input value (string, Path, list, or None).
            
        Returns:
            List of Path objects (empty list if None). Strings are converted to Path.
        """
        if v is None:
            return []
        
        def to_path(path: Union[str, Path]) -> Path:
            """Convert to Path without expanding ~/."""
            if isinstance(path, str):
                return Path(path)
            elif isinstance(path, Path):
                return path
            else:
                raise ValueError(f"Path must be a string or Path, got {type(path)}")
        
        if isinstance(v, (str, Path)):
            return [to_path(v)]
        if isinstance(v, list):
            return [to_path(p) for p in v]
        raise ValueError(f"paths must be a string, Path, or list of strings/Paths, got {type(v)}")
    
    @field_validator('include', 'exclude', mode='before')
    @classmethod
    def normalize_patterns(cls, v: Any) -> Optional[List[str]]:
        """Normalize string patterns to lists.
        
        Args:
            v: Input value (string, list, or None).
            
        Returns:
            List of patterns or None.
        """
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(f"include/exclude must be a string or list of strings, got {type(v)}")

class FileList(Configurable[FileListConfig]):
    """File list generator for finding and filtering files.
    
    This class provides a configurable interface for creating lists of files
    from specified paths, with support for include/exclude pattern filtering.
    
    Example:
        >>> config = FileListConfig(paths=["/data/images", "/data/labels"], include=["*.tif"], exclude=["*.bak", "*.tmp"])
        >>> fl = FileList(config)
        >>> files = fl.run()
    """
    
    def __init__(self, config: FileListConfig):
        """Initialize the file list generator.
        
        Args:
            config: FileListConfig instance.
        """
        super().__init__(config)
    
    @classmethod
    def from_config(cls, config: FileListConfig) -> "FileList":
        """Create a FileList instance from a configuration.
        
        Args:
            config: FileListConfig instance.
            
        Returns:
            FileList instance.
        """
        return cls(config)
    
    @task(name="FileList.run")
    def run(self) -> list[Path]:
        """Generate a list of files according to the configuration.
        
        Returns:
            List of file paths.
            
        Raises:
            FileNotFoundError: If any specified path does not exist.
            ValueError: If no valid paths are provided.
        """
        all_files = []
        
        # paths is already normalized to a list by the validator
        input_paths = self.config.paths
        
        if not input_paths:
            logger.warning("No input paths specified")
            return []
        
        # Resolve base_path to absolute path for proper relative path resolution
        # Expand ~/ at runtime (not in validator) to preserve portability across users
        base_path = input_paths[0].expanduser().resolve() if input_paths else Path.cwd()
        
        # Collect files from specified paths using glob patterns
        # paths are already normalized to Path objects by the validator
        for path_obj in input_paths:
            # Expand ~/ at runtime (not in validator) to preserve portability across users
            path_obj = path_obj.expanduser()
            
            # If relative, resolve relative to base_path
            if not path_obj.is_absolute():
                path_obj = (base_path.parent if base_path.is_file() else base_path) / path_obj
                path_obj = path_obj.resolve()
            else:
                path_obj = path_obj.resolve()
            
            # Log the resolved path for debugging
            logger.debug(f"Checking path existence: {path_obj} (resolved from {path_obj})")
            
            if not path_obj.exists():
                # Try to provide more helpful error message
                logger.error(f"Path does not exist: {path_obj}")
                logger.error(f"  Original path: {path_obj}")
                logger.error(f"  Resolved path: {path_obj.resolve()}")
                logger.error(f"  Path is absolute: {path_obj.is_absolute()}")
                logger.error(f"  Path parent exists: {path_obj.parent.exists() if path_obj.parent else False}")
                
                if self.config.skip_on_error:
                    logger.warning(f"Path does not exist: {path_obj}. Ignoring.")
                    continue
                else:
                    logger.error(f"Path does not exist: {path_obj}. Stopping file list generation.")
                    raise FileNotFoundError(f"Path does not exist: {path_obj}")
            
            # If it's a file, add it directly (ignore include/exclude patterns)
            # Explicitly specified files should always be included
            if path_obj.is_file():
                all_files.append(path_obj)
            
            # If it's a directory, use glob patterns for efficient search
            elif path_obj.is_dir():
                # Check if follow_symlinks parameter is supported (Python 3.13+)
                import sys
                supports_follow_symlinks = sys.version_info >= (3, 13)
                
                if self.config.include:
                    # Use glob patterns directly for search
                    for pattern in self.config.include:
                        if self.config.recursive:
                            # Recursive search: use **/ pattern
                            glob_pattern = f"**/{pattern}"
                            search_method = path_obj.rglob
                        else:
                            # Non-recursive search: use pattern directly
                            glob_pattern = pattern
                            search_method = path_obj.glob
                        
                        # Conditionally pass follow_symlinks based on Python version
                        if supports_follow_symlinks:
                            file_iter = search_method(glob_pattern, follow_symlinks=self.config.follow_symlinks)
                        else:
                            file_iter = search_method(glob_pattern)
                            # If follow_symlinks=False and not supported, we can't filter symlinks
                            # In older Python versions, symlinks are always followed
                            if not self.config.follow_symlinks:
                                logger.warning("follow_symlinks=False is not supported in Python < 3.13. Symlinks will be followed.")
                        
                        for file_path in file_iter:
                            if file_path.is_file():
                                all_files.append(file_path)
                else:
                    # No include patterns - collect all files
                    if self.config.recursive:
                        search_method = path_obj.rglob
                        glob_pattern = "*"
                    else:
                        search_method = path_obj.glob
                        glob_pattern = "*"
                    
                    # Conditionally pass follow_symlinks based on Python version
                    if supports_follow_symlinks:
                        file_iter = search_method(glob_pattern, follow_symlinks=self.config.follow_symlinks)
                    else:
                        file_iter = search_method(glob_pattern)
                        # If follow_symlinks=False and not supported, we can't filter symlinks
                        # In older Python versions, symlinks are always followed
                        if not self.config.follow_symlinks:
                            logger.warning("follow_symlinks=False is not supported in Python < 3.13. Symlinks will be followed.")
                    
                    for file_path in file_iter:
                        if file_path.is_file():
                            all_files.append(file_path)
        
        if not all_files:
            logger.warning("No files found in specified paths")
            return [], {"total_files": 0, "filtered_files": 0}
        
        # Remove duplicates (in case multiple patterns match the same file)
        all_files = list(dict.fromkeys(all_files))  # Preserves order while removing duplicates
        
        # Apply exclude patterns
        if self.config.exclude:
            all_files = [
                file_path for file_path in all_files
                if not any(fnmatch.fnmatch(file_path.name, pattern) for pattern in self.config.exclude)
            ]
        
        # Sort files for consistent ordering
        all_files.sort()
        
        for f in all_files:
            logger.debug(f"Found file: {f}")
        logger.info(f"Found {len(all_files)} files matching criteria")
        
        return all_files

class DataWriterConfig(Configuration):
    """Configuration for data writing operations.
    
    This configuration defines parameters for writing data to files,
    including path, format, and other writing options.
    """
    path: Union[str, Path] = Field(
        default=".",
        description="Path to output file or directory"
    )
    format: Optional[str] = Field(
        default=None,
        description="Format of the output data (e.g., 'tif', 'png', 'jpg', 'h5', 'zarr')"
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing files"
    )
    
    @field_validator('path', mode='before')
    @classmethod
    def normalize_path(cls, v: Any) -> Path:
        """Normalize path to a Path object.
        
        Converts strings to Path objects. Does NOT expand '~/' to preserve
        portability across different users. Expansion happens at runtime in the run method.
        
        Args:
            v: Input value (string, Path, or None).
            
        Returns:
            Path object. Strings are converted to Path.
        """
        if v is None:
            return Path(".")
        if isinstance(v, str):
            return Path(v)
        if isinstance(v, Path):
            return v
        raise ValueError(f"path must be a string or Path, got {type(v)}")
    #df_writer: Optional[DataFrameWriterConfig] = Field(
    #    default=DataFrameWriterConfig(),
    #    description="Configuration for saving dataframes"
    #)
    #image_writer: Optional[ImageWriterConfig] = Field(
    #    default=ImageWriterConfig(),
    #     description="Configuration for saving images"
    #)

class DataWriter(Configurable[DataWriterConfig]):
    """Data writer for writing data to files.
    
    This class provides a configurable interface for writing data to files,
    with support for various formats and options.
    """
    
    def __init__(self, config: DataWriterConfig):
        """Initialize the data writer.
        
        Args:
            config: DataWriterConfig instance.
        """
        super().__init__(config)
    
    @classmethod
    def from_config(cls, config: DataWriterConfig) -> "DataWriter":
        """Create a DataWriter instance from a configuration.
        
        Args:
            config: DataWriterConfig instance.
        """
        return cls(config)
    
    @task(name="DataWriter.run")
    def run(self, data: Any, path: Path|str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Write data to a file according to the configuration.
        
        Args:
            data: Data to write.
            metadata: Optional metadata to pass to the writer.
        """
        raise NotImplementedError("Subclasses must implement this method")

class DataFrameWriterConfig(DataWriterConfig):
    """Configuration for dataframe writing operations.
    
    This configuration defines parameters for writing dataframes to files,
    including format, compression, and other writing options.
    """
    format: Literal["csv", "parquet"] = Field(
        default="csv",
        description="Format of the output dataframe (e.g., 'csv', 'parquet')"
    )
    save_index: bool = Field(
        default=True,
        description="Whether to save the index of the dataframe"
    )

class DataFrameWriter(DataWriter):
    """DataFrame writer for writing dataframes to files."""

    def __init__(self, config: DataFrameWriterConfig):
        """Initialize the dataframe writer.
        
        Args:
            config: DataFrameWriterConfig instance.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: DataFrameWriterConfig) -> "DataFrameWriter":
        """Create a DataFrameWriter instance from a configuration.
        
        Args:
            config: DataFrameWriterConfig instance.
        """
        return cls(config)

    @task(name="DataFrameWriter.run")
    def run(self, data: pd.DataFrame, path: Path|str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Write dataframe data to a file according to the configuration.
        
        Args:
            data: Dataframe data to write.
            path: Path to output file (expands ~/ at runtime).
            metadata: Optional metadata to pass to the writer.
        """
        # Expand ~/ at runtime (not in validator) to preserve portability across users
        if isinstance(path, str):
            path = Path(path).expanduser()
        else:
            path = path.expanduser()
        
        if os.path.exists(path) and not self.config.overwrite:
            raise FileExistsError(f"File already exists: {path}")
        if self.config.format == "csv":
            data.to_csv(path, index=self.config.save_index)
        elif self.config.format == "parquet":
            data.to_parquet(path, index=self.config.save_index)
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")
        logger.info(f"Saved dataframe to {path}")

class ImageWriterConfig(DataWriterConfig):
    """Configuration for image writing operations.
    
    This configuration defines parameters for writing image data to files,
    including format, compression, and other writing options.
    """
    writer: Optional[OmeTiffWriter] = Field(
        default=None,
        description="OME-TIFF writer"
    )
    format: Optional[Literal["tif", "png", "jpg", "h5", "zarr"]] = Field(
        default="tif",
        description="Format of the output image data (e.g., 'tif', 'png', 'jpg', 'h5', 'zarr')"
    )
    split_channels: bool = Field(
        default=False,
        description="Whether to split channels into separate files after processing"
    )
    extension: str = Field(
        default="tif",
        description="Extension of the output image data (e.g., 'tif', 'png', 'jpg', 'h5', 'zarr')"
    )


# Rebuild models to resolve forward references and Literal types
# This is needed when using Literal types in Pydantic models
DataWriterConfig.model_rebuild()
DataFrameWriterConfig.model_rebuild()
ImageWriterConfig.model_rebuild()


class ImageWriter(DataWriter):
    """Image writer for writing image data to files."""

    def __init__(self, config: ImageWriterConfig):
        """Initialize the image writer.
        
        Args:
            config: ImageWriterConfig instance.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: ImageWriterConfig) -> "ImageWriter":
        """Create an ImageWriter instance from a configuration.
        
        Args:
            config: ImageWriterConfig instance.
        """
        return cls(config)

    @task(name="ImageWriter.run")
    def run(self, data: np.ndarray, path: Path|str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Write image data to a file according to the configuration.
        
        Writes image data to a file using OME-TIFF format. If split_channels is True,
        the image is split along the channel axis and each channel is saved as a
        separate file with the channel name in the filename.
        
        Args:
            data: Image data array to write.
            path: Path to the output file or directory. If split_channels is True,
                  channel names will be added to the filename.
            metadata: Optional metadata dictionary containing:
                - channel_names: List of channel names (required if split_channels is True)
                - dim_order: Dimension order string (e.g., "TCYX", "ZYX")
                - physical_pixel_sizes: Named tuple with Z, Y, X pixel sizes
                - image_names: Optional list of image names
                
        Raises:
            FileExistsError: If the output file already exists and overwrite is False.
            ValueError: If metadata is missing required fields (e.g., channel_names when split_channels is True).
        """
        # Expand ~/ at runtime (not in validator) to preserve portability across users
        if isinstance(path, str):
            path = Path(path).expanduser()
        else:
            path = path.expanduser()
        
        if os.path.exists(path) and not self.config.overwrite:
            raise FileExistsError(f"File already exists: {path}")
        
        logger.info(f"Preparing to save image with metadata: {metadata}")
        from collections import namedtuple
        PhysicalPixelSizes = namedtuple("PhysicalPixelSizes", ["Z", "Y", "X"])
        if self.config.split_channels:
            data, metadata = unstack_image(data, metadata, "C", key="axes")
            for img, m in zip(data, metadata):
                channel_names = m.get("channel_names", [])
                OmeTiffWriter.save(
                    data=img,
                    uri=path.with_suffix(f".{"-".join(channel_names)}.{self.config.extension}"),
                    dim_order=m.get("dim_order", ""),
                    channel_names=m.get("channel_names", None),
                    image_name=m.get("image_names", None),
                    physical_pixel_sizes=PhysicalPixelSizes(*map(abs, m.get("physical_pixel_sizes", None))), 
                )
        else:   
            channel_names = metadata.get("channel_names", [])
            OmeTiffWriter.save(
                data=data,
                uri=path.with_suffix(f".{"-".join(channel_names)}.{self.config.extension}"),
                dim_order=metadata.get("dim_order", ""),
                channel_names=metadata.get("channel_names", []),
                image_name=metadata.get("image_names", ""),
                physical_pixel_sizes=PhysicalPixelSizes(*map(abs, metadata.get("physical_pixel_sizes", None))), 
            )
        logger.info(f"Saved image to {path}")

class DataLoaderConfig(Configuration):
    """Base configuration for data loading operations.
    
    This is a base configuration class that can be extended for specific
    data loading use cases (e.g., image loading, label loading, etc.).
    """
    substack: Optional[str] = Field(
        default=None,
        description="Substack specification string (e.g., 'T:4-10;Z:2-20' or legacy '10' or '2-40')"
    )
    squeeze: bool = Field(
        default=False,
        description="Whether to squeeze singleton dimensions from the loaded image"
    )


class DataLoader(Configurable[DataLoaderConfig]):
    """Base data loader class.
    
    This is a base class that can be extended for specific data loading
    use cases (e.g., image loading, label loading, etc.).
    
    Subclasses should implement the `run()` method to define their specific
    loading behavior.
    """
    
    def __init__(self, config: DataLoaderConfig):
        """Initialize the data loader.
        
        Args:
            config: DataLoaderConfig instance.
        """
        super().__init__(config)
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: DataLoaderConfig) -> "DataLoader":
        """Create a DataLoader instance from a configuration.
        
        Args:
            config: DataLoaderConfig instance.
            
        Returns:
            DataLoader instance.
        """
        pass
    
    @abstractmethod
    def run(self, path: Path|str) -> tuple[Any, dict[str, Any]]:
        """Load data according to the configuration.
        
        Args:
            path: Path to the data file or directory.
            
        Returns:
            Tuple of (data array, metadata dictionary).
            
        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the data cannot be loaded or processed.
        """
        raise NotImplementedError("Subclasses must implement this method")


class ImageLoaderConfig(DataLoaderConfig):
    """Configuration for image loading operations.
    
    This configuration defines parameters for loading image data from files,
    including scene selection, substack slicing, channel renaming, and other
    image loading options.
    
    Attributes:
        scene_index: Optional scene index to load (for multi-scene files).
        grayscale: Whether to convert loaded images to grayscale.
    """
    
    scene_index: Optional[int] = Field(
        default=None,
        description="Scene index to load (for multi-scene files, None = load all)"
    )
    grayscale: bool = Field(
        default=False,
        description="Whether to convert loaded images to grayscale"
    )
    split_channels: bool = Field(
        default=False,
        description="Whether to split channels into separate files after processing"
    )
    rename_channel: Optional[dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping original channel names to new names (format: 'old1:new1;old2:new2')"
    )
    
    @field_validator('rename_channel', mode='before')
    @classmethod
    def parse_rename_channel(cls, v: Any) -> Optional[dict[str, str]]:
        """Parse rename_channel from string format to dictionary.
        
        Uses the utility function from vistiq.utils for consistent parsing.
        """
        return str_to_dict(v, value_type=str)


class ImageLoader(DataLoader):
    """Image loader for image files.
    
    This class provides a configurable interface for loading image data from files,
    with support for scene selection, substack slicing, channel renaming, and
    other image loading options.
    
    Example:
        >>> config = ImageLoaderConfig(scene_index=0, squeeze=True)
        >>> loader = ImageLoader(config)
        >>> image, metadata = loader.run(path="image.tif")
    """
    
    def __init__(self, config: ImageLoaderConfig):
        """Initialize the image loader.
        
        Args:
            config: ImageLoaderConfig instance.
        """
        super().__init__(config)
    
    @classmethod
    def from_config(cls, config: ImageLoaderConfig) -> "ImageLoader":
        """Create an ImageLoader instance from a configuration.
        
        Args:
            config: ImageLoaderConfig instance.
            
        Returns:
            ImageLoader instance.
        """
        return cls(config)
    

    def _load_image(self,
        path: Union[str, Path],
        scene_index: Optional[int] = None,
        reader: Optional[type] = None,
        substack: Optional[dict[str, slice]] = None,
        squeeze: bool = False,
        rename_channel: Optional[dict[str, str]] = None,
    ) -> tuple[np.ndarray, dict]:
        """Load an image file using bioio, optionally selecting a specific scene and applying dimension slicing.
        
        Args:
            path: Path to the image file.
            scene_index: Optional scene index to load. If None, loads the first/default scene.
            reader: Optional reader class to use. If None, bioio will auto-detect the appropriate reader.
            substack: Optional dictionary of dimension slices. Keys should be dimension names ('T', 'Z', 'C', 'Y', 'X')
                    and values should be slice objects. For example: {'T': slice(0, 10), 'Z': slice(5, 15), 'C': slice(0, 2)}.
                    Special case: If None is used as a key, the slice will be applied to the first axis of the image
                    (determined from image metadata). This is used internally for legacy substack format.
            squeeze: If True, remove dimensions of size 1 from the array. Default is False.
            rename_channel: Optional dictionary mapping original channel names to new names.
                        Only channels present in this dictionary will be renamed.
                        Example: {"Red": "Dpn", "Blue": "EDU"}.
            
        Returns:
            tuple[np.ndarray, dict]: Image data as a numpy array and metadata dictionary.
                                    The metadata includes channel_names (with renaming applied if specified).
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the scene index is out of range, dimension names are invalid, or the image cannot be loaded.
            ImportError: If bioio is not installed.
        """
        try:
            from bioio import BioImage
        except ImportError:
            raise ImportError("bioio is required for load_image. Install it with: pip install bioio")
        
        file_path = os.fspath(path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the file with bioio - auto-detect reader or use specified one
        if reader is not None:
            reader_instance = BioImage(str(file_path), reader=reader)
            logger.info("Using specified reader: %s", reader.__name__)
        else:
            reader_instance = BioImage(str(file_path))
        logger.debug(f"reader_instance.metadata: {reader_instance.metadata}")
        dim_str = ("".join([i[0] for i in reader_instance.dims.items()]))

        # If scene_index is provided, validate it
        if scene_index is not None:
            scenes = reader_instance.scenes
            if scenes is None:
                raise ValueError("File has no scenes (scenes is None)")
            
            try:
                num_scenes = len(scenes)
            except (TypeError, AttributeError):
                raise ValueError("Cannot determine number of scenes")
            
            if scene_index < 0 or scene_index >= num_scenes:
                raise ValueError(f"Scene index {scene_index} is out of range (0-{num_scenes-1})")
            
            # Load the specific scene
            reader_instance.set_scene(scene_index)
            logger.debug (f"set scene to {scene_index}")
        
        # Load the image data, passing substack directly to get_image_data
        if substack is not None and len(substack) > 0:
            reader_instance.set_scene(scene_index)
            
            # Handle legacy format: if None is a key, apply to first dimension
            if None in substack:
                # Get the first dimension name from the image metadata
                first_dim = reader_instance.dims.order[0].upper() if reader_instance.dims.order else "T"
                # Move the slice from None key to the actual first dimension
                first_slice = substack.pop(None)
                substack[first_dim] = first_slice
                logger.debug(f"Legacy substack format: applied to first dimension '{first_dim}'")
            
            logger.debug(f"dim_str: {dim_str}, substack: {substack}")
            scene_data = reader_instance.get_image_data(dim_str, **substack) #.squeeze()
            ch_names = reader_instance.channel_names[substack["C"]]
        else:
            scene_data = reader_instance.get_image_data()
            ch_names = reader_instance.channel_names
        
        # Apply channel renaming if specified
        if rename_channel:
            renamed_ch_names = [rename_channel[ch_name] if ch_name in rename_channel.keys() else ch_name for ch_name in ch_names]
            # Preserve original type if possible (tuple -> tuple, list -> list)
            if isinstance(ch_names, tuple):
                ch_names = tuple(renamed_ch_names)
            else:
                ch_names = renamed_ch_names
        
        metadata = {}
        metadata["scene_index"] = scene_index
        metadata["dim_order"] = dim_str
        metadata["axes"] = [
            label.upper() for label in reader_instance.dims.order
        ]  # "t", "c", "y", "x"]
        metadata["channel_names"] = [str(ch) for ch in ch_names]
        metadata["channel_axis"] = metadata["axes"].index("C")
        metadata["shape"] = reader_instance.shape
        metadata["dims"] = reader_instance.dims
        metadata["pixel_unit"] = "um"
        metadata["scale"] = reader_instance.scale
        metadata["physical_pixel_sizes"] = reader_instance.physical_pixel_sizes    
        # Validate the data
        if scene_data is None:
            raise ValueError("Image data is None")
        
        if scene_data.size == 0:
            raise ValueError("Image data is empty (size=0)")
        
        # Apply squeezing if requested
        if squeeze:
            scene_data, metadata = squeeze_image(scene_data, metadata)
        
        logger.info(
            f"Loaded image: {file_path} scene={scene_index if scene_index is not None else 'default'} -> shape={scene_data.shape} dtype={scene_data.dtype}, channel_names={ch_names}",
        )
        
        return scene_data, metadata

    @task(name="ImageLoader.run")
    def run(self, path: Path|str) -> tuple[np.ndarray, dict[str, Any]]:
        """Load image data according to the configuration.
        
        Loads an image file using bioio, optionally selecting a specific scene,
        applying substack slicing, and squeezing singleton dimensions.
        
        Args:
            path: Path to the image file to load.
            
        Returns:
            Tuple of (image array, metadata dictionary). The metadata includes:
                - channel_names: List of channel names (as strings)
                - axes: List of axis labels (e.g., ["T", "C", "Z", "Y", "X"])
                - shape: Shape of the loaded image
                - scale: Scale information for each dimension
                - channel_axis: Index of the channel axis
                - dim_order: Dimension order string
                - physical_pixel_sizes: Physical pixel sizes for Z, Y, X dimensions
                - Other metadata from the image file
                
        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the image cannot be loaded, scene index is out of range,
                       or substack specification is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        logger.info(f"Loading image from: {path}")
        
        # Convert substack string to slices if provided
        substack_slices = None
        if self.config.substack:
            # Import locally to avoid circular dependency with app.py
            from vistiq.app import substack_to_slices
            substack_slices = substack_to_slices(self.config.substack)
        
        # Load the image
        image, metadata = self._load_image(
            path=path,
            scene_index=self.config.scene_index,
            substack=substack_slices,
            squeeze=self.config.squeeze,
            rename_channel=self.config.rename_channel
        )
        
        # Apply grayscale conversion if requested
        if self.config.grayscale and image.ndim > 2:
            # Convert to grayscale by taking mean across channel dimension
            # Assuming channels are in the first dimension
            if image.shape[0] > 1:  # Multiple channels
                image = np.mean(image, axis=0, keepdims=True)
                logger.info("Converted image to grayscale")
                # Update metadata
                if "channel_names" in metadata:
                    metadata["channel_names"] = ["grayscale"]
        
        logger.info(f"Loaded image with shape: {image.shape}, dtype: {image.dtype}")
        
        return image, metadata







