from __future__ import annotations

import fnmatch
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union, Any
import numpy as np
from pydantic import Field, field_validator

from vistiq.core import Configuration, Configurable
from vistiq.utils import load_image, str_to_dict

logger = logging.getLogger(__name__)

class FileListConfig(Configuration):
    """Configuration for file list operations.
    
    This configuration defines parameters for creating and filtering lists of files,
    including path specifications, include/exclude patterns, and filtering options.
    
    Attributes:
        paths: List of paths (files or directories) to search for files.
        include: Optional list of patterns to include files (e.g., ['*.tif', '*.tiff']).
        exclude: Optional list of patterns to exclude files (e.g., ['*.tmp', '*.bak']).
    """
    
    input_paths: list[Union[str, Path]] = Field(
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
    include: Optional[list[str]] = Field(
        default=None,
        description="List of file patterns to include (e.g., ['*.tif', '*.csv'])"
    )
    exclude: Optional[list[str]] = Field(
        default=None,
        description="List of file patterns to exclude (e.g., ['*.bak', '*.tmp'])"
    )
    skip_on_error: bool = Field(
        default=True,
        description="Whether to skip paths that do not exist"
    )

class FileList(Configurable[FileListConfig]):
    """File list generator for finding and filtering files.
    
    This class provides a configurable interface for creating lists of files
    from specified paths, with support for include/exclude pattern filtering.
    
    Example:
        >>> config = FileListConfig(paths=["/data/images", "/data/labels"], include=["*.tif"])
        >>> file_list = FileList(config)
        >>> files, metadata = file_list.run(path="/data")
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
    
    def run(self) -> tuple[list[Path], dict[str, Any]]:
        """Generate a list of files according to the configuration.
        
        Args:
            path: Base path to search for files (if paths in config are relative).
        
        Returns:
            Tuple of (list of file paths, metadata dictionary).
            
        Raises:
            FileNotFoundError: If any specified path does not exist.
            ValueError: If no valid paths are provided.
        """
        all_files = []
        base_path = Path(self.config.input_paths[0]) if self.config.input_paths else Path.cwd()
        
        # Collect files from specified paths using glob patterns
        for path_spec in self.config.input_paths:
            path_obj = Path(path_spec)
            
            # If relative, resolve relative to base_path
            if not path_obj.is_absolute():
                path_obj = base_path / path_obj
            else:
                path_obj = path_obj.resolve()
            
            if not path_obj.exists():
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
                        
                        for file_path in search_method(glob_pattern, follow_symlinks=self.config.follow_symlinks):
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
                    
                    for file_path in search_method(glob_pattern, follow_symlinks=self.config.follow_symlinks):
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
        
        metadata = {
            "total_files": len(all_files),
            "include_patterns": self.config.include,
            "exclude_patterns": self.config.exclude,
            "search_paths": [str(p) for p in self.config.input_paths]
        }
        
        logger.info(f"Found {len(all_files)} files matching criteria")
        
        return all_files, metadata
        
class DataLoaderConfig(Configuration):
    """Base configuration for data loading operations.
    
    This is a base configuration class that can be extended for specific
    data loading use cases (e.g., image loading, label loading, etc.).
    """
    input_path: Union[str, Path] = Field(
        description="Path to input file or directory"
    )
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
    def run(self) -> tuple[Any, dict[str, Any]]:
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
        default=True,
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
    
    def run(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Load image data according to the configuration.
        
        Returns:
            Tuple of (image array, metadata dictionary).
            
        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the image cannot be loaded or processed.
        """
        path = Path(self.config.input_path)
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
        image, metadata = load_image(
            str(path),
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







