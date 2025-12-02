from __future__ import annotations

import numpy as np
from typing import Optional, Literal, Any
from pydantic import Field, field_validator, model_validator
from scipy.ndimage import uniform_filter1d
from skimage.filters import gaussian
from skimage.transform import resize
from joblib import Parallel, delayed
import logging

from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor, cli_config

logger = logging.getLogger(__name__)

@cli_config(exclude=['output_type'])
class PreprocessorConfig(StackProcessorConfig):
    """Configuration for image preprocessing operations.

    This configuration class defines parameters for preprocessing steps, e.g. normalization, denoising and Difference of Gaussians (DoG)
    filtering.
    """
    normalize: bool = Field(
        default=True, description="Normalize output to [0, 1] range"
    )
    output_type: Literal["stack"] = "stack"
    dtype: Literal[int, np.uint8, np.uint32, np.uint64, float, np.float32, np.float64] | None = Field(default=None, description="dtype of processed stack. If None, same as input dtype.")


class Preprocessor(StackProcessor):
    """Preprocessor for image stacks using denoising and Difference of Gaussians filtering.

    This class provides configurable preprocessing operations including temporal denoising
    and Difference of Gaussians (DoG) filtering for image stacks.
    """

    def __init__(self, config: PreprocessorConfig):
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: PreprocessorConfig) -> "Preprocessor":
        """Create a Preprocessor instance from a configuration.

        Args:
            config: Preprocessing configuration.

        Returns:
            A new Preprocessor instance.
        """
        return cls(config)

    def normalize(self, stack: np.ndarray) -> np.ndarray:
        """Normalize the output of the preprocess chain.

        Args:
            stack: Input image stack.
        """
        stack_min = np.min(stack)
        stack_max = np.max(stack)
        if stack_max > stack_min:
            stack = (stack - stack_min) / (stack_max - stack_min)
        else:
            # All values are the same; set to 0
            stack = np.zeros_like(stack, dtype=np.float32)
        logger.info(f"Normalized stack with shape {stack.shape}, min:max {stack_min}:{stack_max} -> {np.min(stack)}:{np.max(stack)}")
        return stack.astype(np.float32, copy=False)

    def run(self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10, metadata: Optional[dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Run the preprocess chain on an image stack.

        Args:
            stack: Input image stack.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.
        """
        input_dtype = stack.dtype
        logger.info(f"Running preprocessor {self.__class__.__name__}, on stack of type {input_dtype}, {np.issubdtype(input_dtype, np.integer)}")
        preprocessed = super().run(stack, *args, workers=workers, verbose=verbose, metadata=metadata, **kwargs)

        # normalize
        if self.config.normalize:
            preprocessed = self.normalize(preprocessed)
        
        # Determine output type
        if self.config.output_type is not None:
            dtype = self.config.dtype
        else:
            dtype = input_dtype
        
        # Scale to the min/max range of the output type
        if np.issubdtype(dtype, np.integer):
            # For integer types, scale to [type_min, type_max]
            type_info = np.iinfo(dtype)
            type_min = type_info.min
            type_max = type_info.max
            
            if self.config.normalize:
                # Already normalized to [0, 1], just scale to output type range
                preprocessed = preprocessed * (type_max - type_min) + type_min
            else:
                # Get current range and normalize, then scale to output type range
                stack_min = np.min(preprocessed)
                stack_max = np.max(preprocessed)
                if stack_max > stack_min:
                    preprocessed = (preprocessed - stack_min) / (stack_max - stack_min)
                    preprocessed = preprocessed * (type_max - type_min) + type_min
                else:
                    # All values are the same; set to middle of range
                    preprocessed = np.full_like(preprocessed, (type_min + type_max) // 2, dtype=np.float64)
            
            preprocessed = np.clip(preprocessed, type_min, type_max)
        elif np.issubdtype(dtype, np.floating):
            # For float types, if normalized, keep [0, 1] range
            # Otherwise, scale to [0, 1] to match typical float expectations
            if not self.config.normalize:
                stack_min = np.min(preprocessed)
                stack_max = np.max(preprocessed)
                if stack_max > stack_min:
                    preprocessed = (preprocessed - stack_min) / (stack_max - stack_min)
                else:
                    preprocessed = np.zeros_like(preprocessed, dtype=np.float32)
        
        preprocessed = preprocessed.astype(dtype, copy=False)
        
        return preprocessed

class PreprocessChainConfig(Configuration):
    """Configuration for chain of preprocessors.

    This configuration class defines parameters for chaining multiple preprocessors.
    """

    preprocessors: list[PreprocessorConfig] = Field(
        default=[], description="List of preprocessors to apply"
    )


class PreprocessChain(Configurable):
    """Chain of preprocessors for image processing.

    This class chains multiple preprocessors together, applying each one in sequence to the input stack.
    """

    def __init__(self, config: PreprocessChainConfig):
        """Initialize the preprocess chain.

        Args:
            config: Preprocess chain configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: PreprocessChainConfig) -> "PreprocessChain":
        """Create a PreprocessChain instance from a configuration.

        Args:
            config: Preprocess chain configuration.
        """
        return cls(config)


    def run(self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10, metadata: Optional[dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Run the preprocess chain on an image stack.

        Args:
            stack: Input image stack.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.
        """
        return super().run(stack, *args, workers=workers, verbose=verbose)

class DoG(Preprocessor):
    """Difference of Gaussians (DoG) filter for image processing.

    This class provides configurable Difference of Gaussians filtering operations
    for image stacks, which can enhance edges and features by subtracting a
    high-sigma Gaussian blur from a low-sigma Gaussian blur.
    """

    def __init__(self, config: DoGConfig):
        """Initialize the DoG filter.

        Args:
            config: DoG configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: DoGConfig) -> "DoG":
        """Create a DoG instance from a configuration.

        Args:
            config: DoG configuration.

        Returns:
            A new DoG instance.
        """
        return cls(config)

    def _process_slice(
        self, slice: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Compute Difference of Gaussians (DoG) for a single slice.

        Args:
            slice: Input slice.

        Returns:
            Difference of Gaussians (DoG) for the single slice.
        """
        logger.info(f"Processing slice, slice.shape={slice.shape}")
        g_low = gaussian(
            slice,
            sigma=self.config.sigma_low,
            mode=self.config.mode,
            preserve_range=True,
        )
        g_high = gaussian(
            slice,
            sigma=self.config.sigma_high,
            mode=self.config.mode,
            preserve_range=True,
        )
        return g_low - g_high


class DoGConfig(PreprocessorConfig):
    """Configuration for Difference of Gaussians (DoG) filtering operations.

    This configuration class defines parameters for applying Difference of Gaussians
    filtering to image stacks.
    """

    sigma_low: float | tuple[float, ...] = Field(
        default=1.0, description="Sigma for the lower Gaussian blur"
    )
    sigma_high: float | tuple[float, ...] = Field(
        default=5.0, description="Sigma for the higher Gaussian blur"
    )
    mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = Field(
        default="reflect", description="Border handling mode for Gaussian filtering"
    )




class Noise2StackConfig(PreprocessorConfig):
    """Configuration for Noise2Stack-inspired denoising operations.

    This configuration class defines parameters for temporal denoising using
    a Noise2Stack-inspired approach that averages temporal neighbors.
    """

    window: int = Field(
        default=5, description="Temporal window size for denoising (odd recommended)"
    )
    exclude_center: bool = Field(
        default=True, description="Exclude center frame from denoising average"
    )

    @field_validator("window")
    @classmethod
    def validate_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError("window must be >= 1")
        return v

    @model_validator(mode="after")
    def validate_window_exclude_center(self) -> "Noise2StackConfig":
        """Validate that window is >= 2 when exclude_center is True."""
        if self.exclude_center and self.window < 2:
            raise ValueError(
                "window must be >= 2 when exclude_center=True"
            )
        return self


class Noise2Stack(Preprocessor):
    """Noise2Stack-inspired denoiser for image stacks.

    This class provides temporal denoising by averaging neighboring frames in time.
    It implements a simple, non-learning variant inspired by the Noise2Stack idea:
    predict each frame from its neighboring frames in time by computing a temporal
    moving average, optionally excluding the center frame from the average.
    """

    def __init__(self, config: Noise2StackConfig):
        """Initialize the Noise2Stack denoiser.

        Args:
            config: Noise2Stack configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: Noise2StackConfig) -> "Noise2Stack":
        """Create a Noise2Stack instance from a configuration.

        Args:
            config: Noise2Stack configuration.

        Returns:
            A new Noise2Stack instance.
        """
        return cls(config)

    def denoise(
        self, stack: np.ndarray
    ) -> np.ndarray:
        """Denoise an image stack by averaging temporal neighbors (Noise2Stack-inspired).

        This implements a simple, non-learning variant inspired by the Noise2Stack idea:
        predict each frame from its neighboring frames in time. Here we compute a
        temporal moving average over the first axis (time), optionally excluding the
        center frame from the average.

        Args:
            stack: Input stack with time as first axis. Shapes supported:
                (T, H, W) or (T, H, W, C).

        Returns:
            Denoised stack with the same shape and dtype as the input.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if not self.config.denoise:
            return stack

        window = self.config.denoise_window
        exclude_center = self.config.denoise_exclude_center

        if window < 1:
            raise ValueError("denoise_window must be >= 1")
        if exclude_center and window < 2:
            raise ValueError("denoise_window must be >= 2 when denoise_exclude_center=True")

        input_dtype = stack.dtype
        work = stack.astype(np.float32, copy=False)

        # Apply temporal moving average along T axis
        avg = uniform_filter1d(work, size=window, axis=0, mode="nearest")

        if exclude_center:
            # With mode='nearest', uniform_filter1d uses an effective window of exactly `window`.
            # Exclude the center by subtracting the original frame and renormalize.
            denoised = (avg * float(window) - work) / float(window - 1)
        else:
            denoised = avg

        # Cast back to input dtype with clipping for integer types
        if np.issubdtype(input_dtype, np.integer):
            info = np.iinfo(input_dtype)
            denoised = np.clip(denoised, info.min, info.max).astype(input_dtype, copy=False)
        else:
            denoised = denoised.astype(input_dtype, copy=False)

        return denoised

    def run(self, stack: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Denoise an image stack by averaging temporal neighbors.

        This implements a simple, non-learning variant inspired by the Noise2Stack idea:
        predict each frame from its neighboring frames in time. Here we compute a
        temporal moving average over the first axis (time), optionally excluding the
        center frame from the average.

        Args:
            stack: Input stack with time as first axis. Shapes supported:
                (T, H, W) or (T, H, W, C).

        Returns:
            Denoised stack with the same shape and dtype as the input.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        window = self.config.window
        exclude_center = self.config.exclude_center

        if window < 1:
            raise ValueError("window must be >= 1")
        if exclude_center and window < 2:
            raise ValueError("window must be >= 2 when exclude_center=True")

        input_dtype = stack.dtype
        work = stack.astype(np.float32, copy=False)

        # Apply temporal moving average along T axis
        avg = uniform_filter1d(work, size=window, axis=0, mode="nearest")

        if exclude_center:
            # With mode='nearest', uniform_filter1d uses an effective window of exactly `window`.
            # Exclude the center by subtracting the original frame and renormalize.
            denoised = (avg * float(window) - work) / float(window - 1)
        else:
            denoised = avg

        # Cast back to input dtype with clipping for integer types
        if np.issubdtype(input_dtype, np.integer):
            info = np.iinfo(input_dtype)
            denoised = np.clip(denoised, info.min, info.max).astype(input_dtype, copy=False)
        else:
            denoised = denoised.astype(input_dtype, copy=False)

        return denoised


class ResizeConfig(PreprocessorConfig):
    """Configuration for image resizing operations.
    
    This configuration class defines parameters for resizing image stacks.
    """
    
    width: Optional[int] = Field(
        default=None, description="Target width in pixels (None to maintain aspect ratio)"
    )
    height: Optional[int] = Field(
        default=None, description="Target height in pixels (None to maintain aspect ratio)"
    )
    order: int = Field(
        default=1, description="Spline interpolation order (0=nearest, 1=bilinear, 3=cubic)"
    )
    preserve_range: bool = Field(
        default=True, description="Preserve the original value range (True) or normalize to [0, 1] (False)"
    )
    anti_aliasing: bool = Field(
        default=True, description="Apply anti-aliasing when downsampling"
    )
    
    @field_validator("width", "height")
    @classmethod
    def validate_dimension(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("width and height must be >= 1 if specified")
        return v
    
    @field_validator("order")
    @classmethod
    def validate_order(cls, v: int) -> int:
        if v not in [0, 1, 3]:
            raise ValueError("order must be 0 (nearest), 1 (bilinear), or 3 (cubic)")
        return v
    
    @model_validator(mode="after")
    def validate_dimensions(self) -> "ResizeConfig":
        """Validate that at least one dimension is specified."""
        if self.width is None and self.height is None:
            raise ValueError("At least one of width or height must be specified")
        return self


class Resize(Preprocessor):
    """Image resizer for image stacks.
    
    This class provides configurable resizing operations for image stacks,
    supporting both upsampling and downsampling with various interpolation methods.
    """
    
    def __init__(self, config: ResizeConfig):
        """Initialize the resizer.
        
        Args:
            config: Resize configuration.
        """
        super().__init__(config)
    
    @classmethod
    def from_config(cls, config: ResizeConfig) -> "Resize":
        """Create a Resize instance from a configuration.
        
        Args:
            config: Resize configuration.
            
        Returns:
            A new Resize instance.
        """
        return cls(config)
    
    def _process_slice(
        self, slice: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Resize a single slice.
        
        Args:
            slice: Input slice.
            metadata: Optional metadata (unused).
            
        Returns:
            Resized slice.
        """
        original_shape = slice.shape
        target_shape = self.config.output_shape[-len(original_shape):] #or list(original_shape)
        
        # Resize the slice
        resized = resize(
            slice,
            output_shape=tuple(target_shape),
            order=self.config.order,
            preserve_range=self.config.preserve_range,
            anti_aliasing=self.config.anti_aliasing,
        )
        
        # Preserve dtype if preserve_range is True
        if self.config.preserve_range:
            resized = resized.astype(slice.dtype, copy=False)
        
        logger.debug(f"Resized slice from {original_shape} to {resized.shape}")
        return resized

    def run(self, stack: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Resize an image stack.
        
        Args:
            stack: Input stack.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.
        """
        # Determine target shape
        original_shape = stack.shape
        target_shape = list(original_shape)
        if self.config.width is not None and self.config.height is not None:
            # Both specified: use both
            target_shape[-2] = self.config.height  # Y dimension
            target_shape[-1] = self.config.width   # X dimension
        elif self.config.width is not None:
            # Only width specified: maintain aspect ratio
            aspect_ratio = original_shape[-1] / original_shape[-2]
            target_shape[-1] = self.config.width
            target_shape[-2] = int(self.config.width / aspect_ratio)
        elif self.config.height is not None:
            # Only height specified: maintain aspect ratio
            aspect_ratio = original_shape[-1] / original_shape[-2]
            target_shape[-2] = self.config.height
            target_shape[-1] = int(self.config.height * aspect_ratio)
        
        # Update config.output_shape using model_copy to ensure Pydantic validation
        # This allows _reshape_slice_results to use the correct output dimensions
        if hasattr(self.config, 'model_copy'):
            self.config = self.config.model_copy(update={"output_shape": tuple(target_shape)})
        else:
            # Fallback for older Pydantic versions
            self.config.output_shape = tuple(target_shape)
        logger.info(f"RESIZING stack from {original_shape} to {target_shape}")
        return super().run(stack, metadata=metadata, **kwargs)