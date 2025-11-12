from __future__ import annotations

import numpy as np
from typing import Optional, Literal
from pydantic import Field, field_validator, model_validator
from scipy.ndimage import uniform_filter1d
from skimage.filters import gaussian
from joblib import Parallel, delayed
import logging

from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor

logger = logging.getLogger(__name__)


class PreprocessConfig(StackProcessorConfig):
    """Configuration for image preprocessing operations.

    This configuration class defines parameters for denoising and Difference of Gaussians (DoG)
    filtering operations on image stacks.
    """

    normalize: bool = Field(
        default=True, description="Normalize DoG output to [0, 1] range"
    )
    output_type: Literal[int, np.uint8, np.uint32, np.uint64, float, np.float32, np.float64] | None = Field(default=None, description="dtype of processed stack. If None, same as input dtype.")


class Preprocessor(StackProcessor):
    """Preprocessor for image stacks using denoising and Difference of Gaussians filtering.

    This class provides configurable preprocessing operations including temporal denoising
    and Difference of Gaussians (DoG) filtering for image stacks.
    """

    def __init__(self, config: PreprocessConfig):
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: PreprocessConfig) -> "Preprocessor":
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

    def run(self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10) -> np.ndarray:
        """Run the preprocess chain on an image stack.

        Args:
            stack: Input image stack.
        """
        input_dtype = stack.dtype
        logger.info(f"Running preprocessor {self.__class__.__name__}, on stack of type {input_dtype}, {np.issubdtype(input_dtype, np.integer)}")
        preprocessed = super().run(stack, *args, workers=workers, verbose=verbose)

        # normalize
        if self.config.normalize:
            preprocessed = self.normalize(preprocessed)
        
        # Determine output type
        if self.config.output_type is not None:
            out_type = self.config.output_type
        else:
            out_type = input_dtype
        
        # Scale to the min/max range of the output type
        if np.issubdtype(out_type, np.integer):
            # For integer types, scale to [type_min, type_max]
            type_info = np.iinfo(out_type)
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
        elif np.issubdtype(out_type, np.floating):
            # For float types, if normalized, keep [0, 1] range
            # Otherwise, scale to [0, 1] to match typical float expectations
            if not self.config.normalize:
                stack_min = np.min(preprocessed)
                stack_max = np.max(preprocessed)
                if stack_max > stack_min:
                    preprocessed = (preprocessed - stack_min) / (stack_max - stack_min)
                else:
                    preprocessed = np.zeros_like(preprocessed, dtype=np.float32)
        
        preprocessed = preprocessed.astype(out_type, copy=False)
        
        return preprocessed

class PreprocessChainConfig(Configuration):
    """Configuration for chain of preprocessors.

    This configuration class defines parameters for chaining multiple preprocessors.
    """

    preprocessors: list[PreprocessConfig] = Field(
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


    def run(self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10) -> np.ndarray:
        """Run the preprocess chain on an image stack.

        Args:
            stack: Input image stack.
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
        """
        return cls(config)

    def _process_slice(
        self, slice: np.ndarray
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


class DoGConfig(PreprocessConfig):
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




class Noise2StackConfig(PreprocessConfig):
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

    def run(self, stack: np.ndarray) -> np.ndarray:
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

