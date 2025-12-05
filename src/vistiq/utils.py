from __future__ import annotations

import os
import logging
import fnmatch
from pathlib import Path
from typing import (
    Union,
    Optional,
    List,
    NamedTuple,
    Tuple,
    Dict,
    Pattern,
    Any,
)
from collections import defaultdict

import numpy as np
from pydantic import BaseModel
import pandas as pd

import cv2
import tifffile
import uuid
import torch

try:
    from bioio_ome_tiff.writers import OmeTiffWriter
    OME_TIFF_AVAILABLE = True
except ImportError:
    OME_TIFF_AVAILABLE = False
    OmeTiffWriter = None


logger = logging.getLogger(__name__)


def str_to_dict(s: Optional[Union[str, dict]], value_type: Any = None) -> Optional[dict]:
    """Parse a string in format 'key1:value1;key2:value2' into a dictionary.
    
    This is a generic utility function that can be used in field validators, CLI argument parsing,
    and other places where key-value pair strings need to be converted to dictionaries.
    
    Accepts either:
    - A dictionary (already parsed) - returned as-is (optionally cast)
    - A string in format 'key1:value1;key2:value2' (e.g., 'Red:Dpn;Blue:EDU') - parsed into dict
    - None or empty string - returns None
    
    Args:
        s: Input value (string, dict, or None).
        value_type: Optional type to cast dictionary values to (e.g., int, float, str).
                   If None, values remain as strings.
        
    Returns:
        Dictionary mapping keys to values, or None. Values are cast to value_type if specified.
        
    Raises:
        ValueError: If the string format is invalid or value casting fails.
        TypeError: If value_type is provided but casting fails.
        
    Examples:
        >>> str_to_dict("Red:Dpn;Blue:EDU")
        {'Red': 'Dpn', 'Blue': 'EDU'}
        >>> str_to_dict("a:1;b:2", value_type=int)
        {'a': 1, 'b': 2}
        >>> str_to_dict("x:1.5;y:2.7", value_type=float)
        {'x': 1.5, 'y': 2.7}
        >>> str_to_dict({"Red": "Dpn"})
        {'Red': 'Dpn'}
        >>> str_to_dict(None)
        None
        >>> str_to_dict("")
        None
    """
    if s is None:
        return None
    
    # If already a dict, optionally cast values and return
    if isinstance(s, dict):
        if value_type is None:
            return s
        # Cast all values to the specified type
        return {k: value_type(v) for k, v in s.items()}
    
    # If it's a string, parse it
    if isinstance(s, str):
        if not s.strip():
            return None
        
        result = {}
        pairs = s.split(';')
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            if ':' not in pair:
                raise ValueError(f"Invalid format: '{pair}'. Expected format: 'key:value'")
            parts = pair.split(':', 1)  # Split only on first colon
            if len(parts) != 2:
                raise ValueError(f"Invalid format: '{pair}'. Expected format: 'key:value'")
            key, value = parts[0].strip(), parts[1].strip()
            if not key:
                raise ValueError(f"Empty key in pair: '{pair}'")
            if not value:
                raise ValueError(f"Empty value in pair: '{pair}'")
            
            # Cast value if type is specified
            if value_type is not None:
                try:
                    value = value_type(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to cast value '{value}' to {value_type.__name__}: {e}")
            
            result[key] = value
        
        return result if result else None
    
    # For any other type, raise an error
    raise ValueError(f"Input must be a string or dict, got {type(s).__name__}")


class Configuration(BaseModel):
    """Base configuration class for utility components.
    
    Attributes:
        classname: Optional class name identifier.
    """
    classname: str = None


class ArrayIteratorConfig(Configuration):
    """Configuration for array iteration over 2D slices.
    
    Specifies which axes to keep (not iterate over) when processing arrays.
    Default keeps the last 2 axes (typically spatial dimensions Y, X).
    
    Attributes:
        slice_def: Tuple of axis indices to keep. Negative indices are supported.
                   Default: (-2, -1) keeps last 2 axes, iterates over all others.
    """
    slice_def: tuple[int, ...] = (
        -2,
        -1,
    )

class VolumeStackIteratorConfig(Configuration):
    """Configuration for array iteration over 3D volumes.
    
    Specifies which axes to keep (not iterate over) when processing 3D volumes.
    Default keeps the last 3 axes (typically spatial dimensions Z, Y, X).
    
    Attributes:
        slice_def: Tuple of axis indices to keep. Negative indices are supported.
                   Default: (-3, -2, -1) keeps last 3 axes, iterates over all others.
    """
    slice_def: tuple[int, ...] = (
        -3,
        -2,
        -1,
    )

class ArrayIterator:
    """Iterator over a numpy array with custom slicing.

    Iterates over axes not specified in slice_def, keeping axes in slice_def
    as full slices. Supports negative indices in slice_def.

    Examples:
        For array shape (5, 30, 20):
        - slice_def = (-2, -1): Keep last 2 axes, iterate over axis 0
        - slice_def = (0, 1): Keep first 2 axes, iterate over axis 2
    """

    def __init__(self, arr: np.ndarray, config: ArrayIteratorConfig):
        """Initialize the ArrayIterator.

        Args:
            arr (np.ndarray): The array to iterate over.
            config (ArrayIteratorConfig): The configuration for the iterator.
                slice_def specifies which axes to keep (not iterate over).
                Must contain integers (negative indices supported, e.g., -1 for last axis).
        """
        self.arr = arr
        self.shape = arr.shape
        self.config = config
        self.ndim = len(arr.shape)

        # Special case: empty slice_def means process entire array as single slice
        if len(config.slice_def) == 0:
            # Create a single index tuple with all slices (entire array)
            self.indices = [tuple([slice(None)] * self.ndim)]
            self.index = 0
            return

        # Normalize slice_def: convert negative indices to positive
        normalized_slice_def = tuple(
            axis if axis >= 0 else self.ndim + axis for axis in config.slice_def
        )

        # Validate normalized indices
        for axis in normalized_slice_def:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(
                    f"Invalid axis index {axis} for array with {self.ndim} dimensions"
                )

        # Find axes to iterate over (not in slice_def)
        iter_axes = [i for i in range(self.ndim) if i not in normalized_slice_def]

        # Generate shape for iteration (only over iter_axes)
        iter_shape = tuple(self.shape[axis] for axis in iter_axes)

        # Generate all index tuples for iteration
        self.indices = []
        for iter_indices in np.ndindex(iter_shape):
            # Build full index tuple: specific indices for iter_axes, slice(None) for others
            full_index = [slice(None)] * self.ndim
            for iter_axis_idx, iter_axis in enumerate(iter_axes):
                full_index[iter_axis] = iter_indices[iter_axis_idx]
            self.indices.append(tuple(full_index))

        self.index = 0

    def __iter__(self):
        """Return the iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next slice."""
        if self.index < len(self.indices):
            slice = self.arr[self.indices[self.index]]
            self.index += 1
            return slice
        else:
            raise StopIteration

    def __len__(self) -> int:
        """Return the number of iterations."""
        return len(self.indices)


def create_unique_folder(base_path=".", prefix="", suffix="", exist_ok=True):
    """Create a unique folder within the specified base path using a UUID.
    
    Generates a unique folder name by combining a UUID with optional prefix
    and suffix, ensuring no naming conflicts.
    
    Args:
        base_path: Base directory path where the folder will be created.
                  Defaults to current directory.
        prefix: Optional prefix to prepend to the UUID in the folder name.
        suffix: Optional suffix to append to the UUID in the folder name.
        exist_ok: If True, does not raise error if folder already exists.
                  Defaults to True.
    
    Returns:
        str: Full path to the created unique folder.
    """
    unique_id = (
        uuid.uuid4().hex
    )  # Generate a UUID and get its hexadecimal representation
    dir_name = f"{prefix}{unique_id}{suffix}"
    full_path = os.path.join(base_path, dir_name)

    os.makedirs(full_path, exist_ok=exist_ok)
    return full_path



def masks_to_labels(masks: list[np.ndarray]) -> np.ndarray:
    """Convert a list of masks to a labeled array.
    
    Each mask in the input list is assigned a unique label value. The output
    array has the same shape as the input masks, with each pixel labeled
    according to which mask(s) it belongs to.
    
    Args:
        masks: List of binary masks (boolean or integer arrays). All masks
               must have the same shape.
    
    Returns:
        np.ndarray: Labeled array of integer type. Each mask gets a unique
                   label starting from 1. Pixels that belong to multiple
                   masks will have the sum of their label values.
    """
    labels = np.zeros_like(masks[0]).astype(int)
    for i, mask in enumerate(masks, start=1):
        labels = labels + mask.astype(bool).astype(int)*i
    return labels

def labels_to_mask(labels: np.ndarray) -> list[np.ndarray]:
    """Convert a labeled array to a list of binary masks.
    
    Creates a separate binary mask for each unique label value in the input
    array. Background (label 0) is included as the first mask.
    
    Args:
        labels: Labeled array with integer label values.
    
    Returns:
        list[np.ndarray]: List of boolean masks, one for each unique label
                         value in the input array. Masks are in the order
                         of unique label values found.
    """
    masks = []
    for l_value in np.unique(labels):
        masks.append((labels == l_value).astype(bool))
    return masks

def load_image(
    path: Union[str, os.PathLike],
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
        scene_data = reader_instance.get_image_data(dim_str, **substack).squeeze()
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
    metadata["axes"] = [
        label.upper() for label in reader_instance.dims.order
    ]  # "t", "c", "y", "x"]
    metadata["channel_names"] = ch_names
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
        scene_data = np.squeeze(scene_data)
    
    logger.info(
        f"Loaded image: {file_path} scene={scene_index if scene_index is not None else 'default'} -> shape={scene_data.shape} dtype={scene_data.dtype}, channel_names={ch_names}",
    )
    
    return scene_data, metadata


def get_scenes(
    path: Union[str, os.PathLike],
    reader: Optional[type] = None,
) -> List[Union[str, int]]:
    """Get scene identifiers from an image file using bioio.
    
    Args:
        path: Path to the image file.
        reader: Optional reader class to use. If None, bioio will auto-detect the appropriate reader.
        
    Returns:
        List of scene identifiers. The type depends on the file format:
        - For formats with named scenes: List of strings (scene names/IDs)
        - For formats with indexed scenes: List of integers (scene indices)
        - Empty list if no scenes are found
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If bioio is not installed.
    """
    try:
        from bioio import BioImage
    except ImportError:
        raise ImportError("bioio is required for get_scenes. Install it with: pip install bioio")
    
    file_path = os.fspath(path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the file with bioio - auto-detect reader or use specified one
    if reader is not None:
        reader_instance = BioImage(str(file_path), reader=reader)
        logger.info("Using specified reader: %s", reader.__name__)
    else:
        reader_instance = BioImage(str(file_path))
    
    # Get scenes
    scenes = reader_instance.scenes
    
    if scenes is None:
        #logger.warning("File has no scenes (scenes is None)")
        return []
    
    try:
        # Try to convert to list
        if hasattr(scenes, '__iter__') and not isinstance(scenes, str):
            scene_list = list(scenes)
            logger.info("Found %d scenes in file: %s", len(scene_list), file_path)
            return scene_list
        else:
            # Single scene or string
            return [scenes]
    except (TypeError, AttributeError) as e:
        logger.warning("Could not convert scenes to list: %s", e)
        return []


def load_mp4(
    path: Union[str, os.PathLike],
    to_grayscale: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> np.ndarray:
    """Load an MP4 file into a NumPy array.

    Args:
        path (Union[str, os.PathLike]): Path to the MP4 video file.
        to_grayscale (bool, optional): If True, convert frames to single-channel
            grayscale before stacking. If False, keep color and convert BGR→RGB.
            Defaults to False.
        start_frame (int | None): Optional starting frame index (0-based). If None, starts at 0.
        end_frame (int | None): Optional inclusive ending frame index. If None, reads until EOF.

    Returns:
        np.ndarray: Video data stacked along time axis.
            - If `to_grayscale` is False: shape (T, H, W, C) in RGB order.
            - If `to_grayscale` is True: shape (T, H, W) with uint8 frames.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the video cannot be opened.
    """
    file_path = os.fspath(path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    frames_rgb: list[np.ndarray] = []
    try:
        # Seek to start index when specified
        current_index = 0
        # Normalize and clamp indices
        s_idx = 0 if start_frame is None else max(0, int(start_frame))
        e_idx = None if end_frame is None else int(end_frame)
        if e_idx is not None and e_idx < s_idx:
            raise ValueError("end_frame must be >= start_frame")
        if s_idx > 0:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if e_idx is not None and total > 0 and e_idx >= total:
                e_idx = total - 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(s_idx))
            current_index = s_idx

        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            if e_idx is not None and current_index > e_idx:
                break
            if to_grayscale:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                frames_rgb.append(frame_gray)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            current_index += 1
    finally:
        capture.release()

    if not frames_rgb:
        logger.info("No frames read from: %s", file_path)
        return np.empty((0,), dtype=np.uint8)

    video_array = np.stack(frames_rgb, axis=0)
    logger.info(
        "Loaded MP4: %s -> shape=%s dtype=%s grayscale=%s",
        file_path,
        video_array.shape,
        video_array.dtype,
        to_grayscale,
    )
    return video_array


def check_device() -> torch.device:
    """Determine the best available PyTorch device (cuda, mps, or cpu).

    Returns:
        torch.device: The selected device in priority order cuda > mps > cpu.
    """
    # Prefer CUDA when available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        logger.info("Found CUDA device: %s", name)
        return device

    # Fallback to Apple Metal Performance Shaders on macOS
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Found MPS device")
        return torch.device("mps")

    # Default to CPU
    logger.info("Falling back to CPU device")
    return torch.device("cpu")


def to_mp4(path: Union[str, os.PathLike], stack: np.ndarray, fps: int = 15) -> None:
    """Write an n-dimensional numpy stack to an MP4 file.

    Args:
        path (Union[str, os.PathLike]): Output file path (should end with .mp4).
        stack (np.ndarray): Video stack with time as first axis (T, ..., H, W).
            Supported frame layouts:
              - (T, H, W): single-channel; written as grayscale.
              - (T, H, W, C): RGB(A) with C in {3, 4}; written as color.
        fps (int, optional): Frames per second. Defaults to 15.

    Returns:
        None

    Raises:
        ValueError: If stack dimensionality or shape is not supported.
        RuntimeError: If the video writer cannot be opened.
    """
    file_path = os.fspath(path)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    if stack.ndim < 3:
        raise ValueError("stack must be at least 3D: (T, H, W) or (T, H, W, C)")

    t_dim = stack.shape[0]
    if t_dim == 0:
        logger.info("Empty stack; nothing to write: %s", file_path)
        return

    # Determine frame size and color mode
    color = False
    if stack.ndim == 3:
        height, width = int(stack.shape[-2]), int(stack.shape[-1])
    elif stack.ndim == 4 and stack.shape[-1] in (3, 4):
        height, width = int(stack.shape[-3]), int(stack.shape[-2])
        color = True
    else:
        raise ValueError("Unsupported stack shape; expected (T,H,W) or (T,H,W,C [3|4])")

    # FourCC for mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height), color)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open VideoWriter for: {file_path}")

    try:
        for t in range(t_dim):
            frame = stack[t]
            # Normalize dtype to uint8
            if frame.dtype != np.uint8:
                f32 = frame.astype(np.float32)
                maxv = float(f32.max()) if f32.size > 0 else 0.0
                if maxv > 0:
                    frame_u8 = np.clip(255.0 * (f32 / maxv), 0, 255).astype(np.uint8)
                else:
                    frame_u8 = f32.astype(np.uint8)
            else:
                frame_u8 = frame

            if color:
                # frame_u8 expected as RGB(A); convert to BGR for OpenCV
                if frame_u8.shape[-1] == 4:
                    frame_u8 = frame_u8[..., :3]
                rgb = frame_u8
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            else:
                # Grayscale expected shape (H, W)
                if frame_u8.ndim == 3 and frame_u8.shape[-1] == 1:
                    frame_u8 = frame_u8[..., 0]
                if frame_u8.ndim != 2:
                    raise ValueError("Grayscale frames must be 2D (H, W) per time step")
                writer.write(frame_u8)
    finally:
        writer.release()

    logger.info(
        "Wrote MP4: %s (frames=%d, fps=%d, size=%dx%d, color=%s)",
        file_path,
        t_dim,
        fps,
        width,
        height,
        color,
    )


def to_tif(path: Union[str, os.PathLike], stack: np.ndarray, dim_order: Optional[str] = None) -> None:
    """Write an n-dimensional numpy array as an OME-TIFF file.

    Uses OmeTiffWriter from bioio_ome_tiff.writers to write OME-TIFF format,
    which includes proper metadata and dimension ordering.

    Args:
        path (Union[str, os.PathLike]): Output file path (should end with .tif/.tiff).
        stack (np.ndarray): N-dimensional array. The array dimensions should match
            the dimension order specified (default: inferred from shape).
        dim_order (Optional[str]): Dimension order string (e.g., "TCZYX", "CZYX", "ZYX").
            If None, will be inferred from the array shape. Default: None.

    Returns:
        None

    Raises:
        RuntimeError: If the optional dependency `bioio_ome_tiff` is not available.
    """
    if not OME_TIFF_AVAILABLE:
        raise RuntimeError(
            "bioio_ome_tiff is not available. Please install it with: pip install bioio-ome-tiff"
        )
    
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer dimension order if not provided
    if dim_order is None:
        ndim = stack.ndim
        if ndim == 2:
            dim_order = "YX"
        elif ndim == 3:
            # Common cases: TYX, ZYX, or CYX
            dim_order = "ZYX"  # Default to ZYX for 3D
        elif ndim == 4:
            # Common cases: TZYX, CZYX, or TCYX
            dim_order = "CZYX"  # Default to CZYX for 4D
        elif ndim == 5:
            dim_order = "TCZYX"  # Default to TCZYX for 5D
        else:
            # For other dimensions, use generic labels
            dim_order = "".join([chr(ord('A') + i) for i in range(ndim - 2)]) + "YX"
            logger.warning(
                "Inferred dimension order '%s' for %dD array. Consider specifying dim_order explicitly.",
                dim_order, ndim
            )
    
    # Write using OmeTiffWriter
    OmeTiffWriter.save(stack, str(file_path), dim_order=dim_order)
    logger.info(
        "Wrote OME-TIFF: %s shape=%s dtype=%s dim_order=%s",
        file_path,
        tuple(stack.shape),
        stack.dtype,
        dim_order,
    )


FilePattern = Union[str, Pattern[str]]


def _collect_tif_files(
    root: Path,
    pattern: Optional[FilePattern] = None,
) -> Dict[str, Path]:
    """Collect TIFF files under ``root`` that satisfy an optional filter."""
    if not root.exists():
        logger.warning("Path does not exist: %s", root)
        return {}

    resolved_root = root.resolve()

    def _matches(candidate: Path) -> bool:
        if pattern is None:
            return True
        if hasattr(pattern, "search"):
            return bool(pattern.search(str(candidate)))
        return fnmatch.fnmatch(str(candidate), pattern) or fnmatch.fnmatch(
            candidate.name, pattern
        )

    collected: Dict[str, Path] = {}
    for glob_pattern in ("*.tif", "*.tiff"):
        for file_path in resolved_root.rglob(glob_pattern):
            if not file_path.is_file():
                continue
            if not _matches(file_path):
                continue
            abs_path = file_path.resolve()
            try:
                rel_key = str(abs_path.relative_to(resolved_root))
            except ValueError:
                rel_key = abs_path.as_posix()
            collected[rel_key] = abs_path
    return collected


def find_matching_file_pairs(
    path_a: Path,
    path_b: Path,
    patterns: Optional[Tuple[Optional[FilePattern], Optional[FilePattern]]] = None,
    exclude: Optional[List[str]] = None,
) -> List[Tuple[Path, Path]]:
    """Find matching files between two directory trees.

    The function recursively searches each root for files matching the respective patterns (if provided)
    and optionally filters the results with glob strings or compiled regular expressions.
    
    When patterns are provided, files are matched based on:
    1. Same directory path (relative to their respective roots)
    2. Matching pattern tuple (pattern_a for path_a, pattern_b for path_b)
    3. Common suffix after the pattern prefix (e.g., "Red.tif" in "Preprocessed_Red.tif" and "Labels_Red.tif")
    
    When patterns are not provided, files are paired when they share the same relative path
    (including filename), which is useful when comparing mirrored directory structures.

    Args:
        path_a: First directory tree to search.
        path_b: Second directory tree to search.
        patterns: Optional tuple of filters applied to each directory tree. Each filter
            can be a glob string (e.g., ``"Preprocessed_*.tif"``) or a compiled regular
            expression. Use ``None`` to disable filtering for a side.
        exclude: Optional list of strings. Paths containing any of these strings will be excluded.

    Returns:
        A list of `(file_from_a, file_from_b)` tuples representing matched files.
    """

    pattern_a = patterns[0] if patterns else None
    pattern_b = patterns[1] if patterns else None

    files_a = _collect_tif_files(path_a, pattern_a)
    files_b = _collect_tif_files(path_b, pattern_b)
    
    # Filter out paths that contain any exclude string
    if exclude:
        def _should_exclude(file_path: Path) -> bool:
            """Check if a file path should be excluded.
            
            A path is excluded if any part of the path (as a string) contains
            any of the exclude strings.
            """
            path_str = str(file_path)
            # Check if any exclude string appears anywhere in the path
            for exclude_str in exclude:
                if exclude_str in path_str:
                    return True
            return False
        
        # Filter files_a
        files_a = {rel_path: file_path for rel_path, file_path in files_a.items() 
                   if not _should_exclude(file_path)}
        # Filter files_b
        files_b = {rel_path: file_path for rel_path, file_path in files_b.items() 
                   if not _should_exclude(file_path)}
        logger.debug(f"After exclude filtering: {len(files_a)} files in {path_a} and {len(files_b)} files in {path_b}")
    logger.debug(f"Found {len(files_a)} files in {path_a} and {len(files_b)} files in {path_b}")

    matches: List[Tuple[Path, Path]] = []
    
    if patterns:
        # When patterns are provided, match by directory path and common suffix
        # Group files by their directory path (relative to root)
        dir_files_a: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        dir_files_b: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        
        resolved_a = path_a.resolve()
        resolved_b = path_b.resolve()
        
        # Group files from path_a by directory (relative to path_a)
        for rel_path, file_path in files_a.items():
            try:
                # Get relative directory path within path_a
                rel_dir = str(file_path.parent.relative_to(resolved_a))
            except ValueError:
                # If relative_to fails (e.g., paths on different drives on Windows),
                # use the parent directory as-is (this is a fallback)
                rel_dir = str(file_path.parent)
            # Extract suffix after pattern (for matching)
            file_name = file_path.name
            # Try to extract common suffix by matching the pattern
            if pattern_a and "*" in pattern_a:
                # Pattern has wildcard - extract the suffix part after the wildcard
                pattern_parts = pattern_a.split("*", 1)  # Split only on first *
                pattern_prefix = pattern_parts[0]
                pattern_suffix = pattern_parts[1] if len(pattern_parts) > 1 else ""
                
                # Check if filename matches the pattern structure
                if file_name.startswith(pattern_prefix) and file_name.endswith(pattern_suffix):
                    # Extract the suffix part (everything after the prefix, or use pattern_suffix if it exists)
                    if pattern_suffix:
                        # Use the pattern suffix as the common suffix
                        suffix = pattern_suffix
                    else:
                        # No suffix in pattern, use everything after prefix
                        suffix = file_name[len(pattern_prefix):]
                else:
                    # Pattern doesn't match, use full filename
                    suffix = file_name
            else:
                suffix = file_name
            dir_files_a[rel_dir].append((suffix, file_path))
        
        # Group files from path_b by directory (relative to path_b)
        for rel_path, file_path in files_b.items():
            try:
                # Get relative directory path within path_b
                rel_dir = str(file_path.parent.relative_to(resolved_b))
            except ValueError:
                # If relative_to fails (e.g., paths on different drives on Windows),
                # use the parent directory as-is (this is a fallback)
                rel_dir = str(file_path.parent)
            # Extract suffix after pattern (for matching)
            file_name = file_path.name
            # Try to extract common suffix by matching the pattern
            if pattern_b and "*" in pattern_b:
                # Pattern has wildcard - extract the suffix part after the wildcard
                pattern_parts = pattern_b.split("*", 1)  # Split only on first *
                pattern_prefix = pattern_parts[0]
                pattern_suffix = pattern_parts[1] if len(pattern_parts) > 1 else ""
                
                # Check if filename matches the pattern structure
                if file_name.startswith(pattern_prefix) and file_name.endswith(pattern_suffix):
                    # Extract the suffix part (everything after the prefix, or use pattern_suffix if it exists)
                    if pattern_suffix:
                        # Use the pattern suffix as the common suffix
                        suffix = pattern_suffix
                    else:
                        # No suffix in pattern, use everything after prefix
                        suffix = file_name[len(pattern_prefix):]
                else:
                    # Pattern doesn't match, use full filename
                    suffix = file_name
            else:
                suffix = file_name
            dir_files_b[rel_dir].append((suffix, file_path))
        
        logger.debug(f"Grouped {len(dir_files_a)} directories from {path_a} and {len(dir_files_b)} directories from {path_b}")
        
        # Match files in the same relative directory (within their respective roots) with the same suffix
        # This works even when path_a and path_b are different, as long as they have the same directory structure
        all_dirs = set(dir_files_a.keys()) | set(dir_files_b.keys())
        for rel_dir in all_dirs:
            files_in_a = {suffix: path for suffix, path in dir_files_a.get(rel_dir, [])}
            files_in_b = {suffix: path for suffix, path in dir_files_b.get(rel_dir, [])}
            
            if files_in_a and files_in_b:
                logger.debug(f"Matching files in directory '{rel_dir}': {len(files_in_a)} files in path_a, {len(files_in_b)} files in path_b")
            
            # Match files with the same suffix
            common_suffixes = set(files_in_a.keys()) & set(files_in_b.keys())
            for suffix in common_suffixes:
                matches.append((files_in_a[suffix], files_in_b[suffix]))
                logger.debug(f"Matched: {files_in_a[suffix].name} <-> {files_in_b[suffix].name} (suffix: {suffix})")
    else:
        # When no patterns, match by exact relative path (original behavior)
        for rel_path, file_a in files_a.items():
            partner = files_b.get(rel_path)
            if partner is not None:
                matches.append((file_a, partner))

    return matches