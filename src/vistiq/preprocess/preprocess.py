from __future__ import annotations

import numpy as np
from typing import Optional, Literal, Any, Union, Sequence
from pydantic import Field, ImportString, field_serializer, field_validator, model_validator
from bioio import Dimensions, Scale
from scipy.ndimage import uniform_filter1d, gaussian_filter, distance_transform_edt
from skimage.exposure import rescale_intensity

# segmentation, draw
from skimage.filters import gaussian
from skimage.transform import resize, rescale
import logging

from vistiq.core import (
    Configurable,
    StackProcessorConfig,
    StackProcessor,
    cli_config,
    generate_name,
)
from prefect import task
from vistiq.utils import ArrayIteratorConfig
from vistiq.workflow import WorkflowConfig, Workflow
logger = logging.getLogger(__name__)


@cli_config(exclude=["output_type"])
class PreprocessorConfig(StackProcessorConfig):
    """Configuration for image preprocessing operations.

    Shared options used by preprocessing operators derived from
    :class:`Preprocessor`.

    Attributes:
        normalize: If ``True``, normalize the processed output to ``[0, 1]``
            before dtype scaling.
        output_type: Output container format. Preprocessors return ``"stack"``.
        dtype: Target dtype of processed output. If ``None``, the input dtype is
            preserved.
    """

    normalize: bool = Field(
        default=False, description="Normalize output to [0, 1] range"
    )
    output_type: Literal["stack"] = "stack"
    dtype: (
        Literal[
            bool,
            int,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            float,
            np.float32,
            np.float64,
        ]
        | None
    ) = Field(
        default=None,
        description="dtype of processed stack. If None, same as input dtype.",
    )

    @field_validator("dtype", mode="before")
    @classmethod
    def _resolve_dtype(cls, value: Any) -> Any:
        if value is None or isinstance(value, type):
            return value
        if value == "bool":
            return bool
        if value == "int":
            return int
        if value == "float":
            return float
        if isinstance(value, str) and hasattr(np, value):
            return getattr(np, value)
        return value

    @field_serializer("dtype", when_used="json")
    def _serialize_dtype(self, value: Any) -> Any:
        if value is None:
            return None
        if value is bool:
            return "bool"
        if value is int:
            return "int"
        if value is float:
            return "float"
        if isinstance(value, type) and issubclass(value, np.generic):
            return value.__name__
        return value


class Preprocessor(StackProcessor):
    """Base class for stack preprocessing operations.

    Runs per-slice processing via :class:`StackProcessor`, then applies optional
    normalization and dtype conversion/scaling in a consistent way.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

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
        logger.info(
            f"Normalized stack with shape {stack.shape}, min:max {stack_min}:{stack_max} -> {np.min(stack)}:{np.max(stack)}"
        )
        return stack.astype(np.float32, copy=False)

    @task(name="Preprocessor.run")
    def run(
        self,
        stack: np.ndarray,
        *args,
        workers: int = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Run preprocessing on an image stack."""
        input_dtype = stack.dtype
        logger.info(
            f"Running preprocessor {self.__class__.__name__}, on stack of type {input_dtype}, {np.issubdtype(input_dtype, np.integer)}"
        )
        preprocessed, updated_metadata = super().run(
            stack, *args, workers=workers, verbose=verbose, metadata=metadata, **kwargs
        )

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
                    preprocessed = np.full_like(
                        preprocessed, (type_min + type_max) // 2, dtype=np.float64
                    )

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

        return (preprocessed, updated_metadata)


class PreprocessFlowConfig(WorkflowConfig):
    """Configuration for sequential preprocessing pipelines.

    Attributes:
        processors: Ordered list of :class:`PreprocessorConfig` subclasses
            (e.g. :class:`RescaleConfig`, :class:`DoGConfig`, :class:`FuncProcessorConfig`).
    """

    processors: list[PreprocessorConfig] = Field(
        default_factory=list,
        description="Preprocessor configs applied in sequence",
    )


class PreprocessFlow(Workflow):
    """Apply multiple preprocessors sequentially to the same stack."""

    def __init__(self, config: PreprocessFlowConfig):
        """Initialize the preprocess flow.

        Args:
            config: Preprocess flow configuration.
        """
        super().__init__(config)
        self.processors: list[Preprocessor] = Configurable.create_many_from_configs(
            self.config.processors,
            expected_type=Preprocessor,
            error_header="Failed to instantiate PreprocessFlow processors",
        )
        logger.info(f"Setting up PreprocessFlow with {",".join([p.__class__.__name__ for p in self.processors])}")

    @classmethod
    def from_config(cls, config: PreprocessFlowConfig) -> "PreprocessFlow":
        """Create a PreprocessFlow instance from a configuration.

        Args:
            config: Preprocess flow configuration.
        """
        return cls(config)

    @task(name="PreprocessFlow._run", task_run_name=generate_name)
    def _run(
        self,
        stack: np.ndarray,
        *args,
        workers: Union[int, list[int]] = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Run configured processing steps in sequence."""
        current_stack = stack
        current_metadata = metadata
        if isinstance(workers, int):
            workers = [workers] * len(self.processors)
        if len(workers) != len(self.processors):
            logger.warning(f"Number of workers ({len(workers)}) must match number of processors ({len(self.processors)}). Updating worker specification to match number of processors.")
            if len(workers) < len(self.processors):
                workers = workers + [-1] * (len(self.processors) - len(workers))
            else:
                workers = workers[:len(self.processors)]

        logger.info(f"Running PreprocessFlow with {len(self.processors)} processors")
        for processor, processor_workers in zip(self.processors, workers):
            result = processor.run(
                current_stack,
                *args,
                workers=processor_workers,
                verbose=verbose,
                metadata=current_metadata,
                **kwargs,
            )

            # Support both APIs: processor.run() may return stack only,
            # or (stack, metadata).
            if isinstance(result, tuple) and len(result) == 2:
                current_stack, current_metadata = result
            else:
                current_stack = result

        return current_stack, current_metadata


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
        logger.debug(f"Processing slice, slice.shape={slice.shape}")
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

#    @task(name="DoG.run")
#    def run(self, *args, **kwargs) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
#        """Run the DoG filter on an image stack.
#
#        Args:
#            *args: Additional arguments to pass to the processor.
#            **kwargs: Additional keyword arguments to pass to the processor.
#
#        Returns:
#            Tuple of (DoG filtered image stack, updated metadata or None).
#        """
#        return super().run(*args, **kwargs)


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
            raise ValueError("window must be >= 2 when exclude_center=True")
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

    def denoise(self, stack: np.ndarray) -> np.ndarray:
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
            raise ValueError(
                "denoise_window must be >= 2 when denoise_exclude_center=True"
            )

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
            denoised = np.clip(denoised, info.min, info.max).astype(
                input_dtype, copy=False
            )
        else:
            denoised = denoised.astype(input_dtype, copy=False)

        return denoised

    @task(name="Noise2Stack.run")
    def run(
        self, stack: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Denoise an image stack by averaging temporal neighbors."""
        window = self.config.window
        exclude_center = self.config.exclude_center

        if window < 1:
            raise ValueError("window must be >= 1")
        if exclude_center and window < 2:
            raise ValueError("window must be >= 2 when exclude_center=True")

        input_dtype = stack.dtype
        work = stack.astype(np.float32, copy=False)

        avg = uniform_filter1d(work, size=window, axis=0, mode="nearest")

        if exclude_center:
            denoised = (avg * float(window) - work) / float(window - 1)
        else:
            denoised = avg

        if np.issubdtype(input_dtype, np.integer):
            info = np.iinfo(input_dtype)
            denoised = np.clip(denoised, info.min, info.max).astype(
                input_dtype, copy=False
            )
        else:
            denoised = denoised.astype(input_dtype, copy=False)

        return (denoised, metadata)


def _target_spatial_shape(
    original_shape: tuple[int, ...],
    *,
    width: Optional[int],
    height: Optional[int],
) -> tuple[int, ...]:
    """Compute output stack shape from optional target width/height."""
    target_shape = list(original_shape)
    if width is not None and height is not None:
        target_shape[-2] = height
        target_shape[-1] = width
    elif width is not None:
        aspect_ratio = original_shape[-1] / original_shape[-2]
        target_shape[-1] = width
        target_shape[-2] = int(width / aspect_ratio)
    elif height is not None:
        aspect_ratio = original_shape[-1] / original_shape[-2]
        target_shape[-2] = height
        target_shape[-1] = int(height * aspect_ratio)
    else:
        raise ValueError("At least one of width or height must be specified")
    return tuple(target_shape)


class UpsampleConfig(PreprocessorConfig):
    """Configuration for label-mask upsampling via signed distance fields.

    Target spatial size follows the same ``width`` / ``height`` convention as
    :class:`ResizeConfig` (Y = height, X = width). Upsampling is applied along
    the spatial axes kept by ``iterator_config.slice_def`` (by default the last
    two axes).
    """

    width: Optional[int] = Field(
        default=None,
        description="Target width in pixels (None to maintain aspect ratio)",
    )
    height: Optional[int] = Field(
        default=None,
        description="Target height in pixels (None to maintain aspect ratio)",
    )
    sigma: float = Field(
        default=1.0, description="Gaussian sigma for smoothing", ge=0.0
    )
    recompute_scale: bool = Field(
        default=True,
        description="Recompute scale/physical dimensions of upsampled pixels/voxels",
    )

    @field_validator("width", "height")
    @classmethod
    def validate_dimension(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("width and height must be >= 1 if specified")
        return v

    @model_validator(mode="after")
    def validate_dimensions(self) -> "UpsampleConfig":
        if self.width is None and self.height is None:
            raise ValueError("At least one of width or height must be specified")
        return self


class Upsample(Preprocessor):
    """SDT-based upsampler for label-mask stacks.

    Each non-zero label is upsampled independently using a signed distance field,
    optional Gaussian smoothing, and thresholding at zero. Intended for segmentation
    label masks, not intensity images.
    """

    def __init__(self, config: UpsampleConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: UpsampleConfig) -> "Upsample":
        return cls(config)

    def _process_slice(
        self, slice: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Upsample a single label slice to the configured spatial size."""
        if self.config.output_shape is None:
            raise ValueError(
                "Upsample.output_shape must be set before processing slices; call run()"
            )
        target_shape = self.config.output_shape[-slice.ndim :]
        scale = tuple(
            target / current for target, current in zip(target_shape, slice.shape)
        )
        upsampled_mask = np.zeros(target_shape, dtype=slice.dtype)

        label_values = np.unique(slice)
        label_values = label_values[label_values != 0]

        for label in label_values:
            mask = slice == label
            dist_inside = distance_transform_edt(mask)
            dist_outside = distance_transform_edt(~mask)
            sdt = dist_inside - dist_outside
            upsampled_sdt = rescale(
                sdt,
                scale=scale,
                order=3,
                anti_aliasing=False,
            )

            if self.config.sigma > 0:
                upsampled_sdt = gaussian_filter(upsampled_sdt, self.config.sigma)

            upsampled_mask[upsampled_sdt > 0] = label

        return upsampled_mask

    @task(name="Upsample.run")
    def run(
        self,
        stack: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Upsample a label stack, preserving label IDs and dtype."""
        original_shape = stack.shape
        target_shape = _target_spatial_shape(
            original_shape,
            width=self.config.width,
            height=self.config.height,
        )
        if hasattr(self.config, "model_copy"):
            self.config = self.config.model_copy(
                update={"output_shape": target_shape}
            )
        else:
            self.config.output_shape = target_shape
        logger.info(f"Upsampling stack from {original_shape} to {target_shape}")

        input_dtype = stack.dtype
        upsampled, updated_metadata = super(Preprocessor, self).run(
            stack, metadata=metadata, **kwargs
        )

        output_dtype = self.config.dtype if self.config.dtype is not None else input_dtype
        if upsampled.dtype != output_dtype:
            upsampled = upsampled.astype(output_dtype, copy=False)
        return upsampled, updated_metadata


class ResizeConfig(PreprocessorConfig):
    """Configuration for image resizing operations.

    This configuration class defines parameters for resizing image stacks.
    """

    width: Optional[int] = Field(
        default=None,
        description="Target width in pixels (None to maintain aspect ratio)",
    )
    height: Optional[int] = Field(
        default=None,
        description="Target height in pixels (None to maintain aspect ratio)",
    )
    order: int = Field(
        default=1,
        description="Spline interpolation order (0=nearest, 1=bilinear, 3=cubic)",
    )
    preserve_range: bool = Field(
        default=True,
        description="Preserve the original value range (True) or normalize to [0, 1] (False)",
    )
    anti_aliasing: bool = Field(
        default=True, description="Apply anti-aliasing when downsampling"
    )
    recompute_scale: bool = Field(
        default=True, description="Recompute scale/physical dimensions of pixels/voxels"
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
        target_shape = self.config.output_shape[
            -len(original_shape) :
        ]  # or list(original_shape)

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

    @task(name="Resize.run")
    def run(
        self, stack: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Resize an image stack."""
        original_shape = stack.shape
        target_shape = _target_spatial_shape(
            original_shape,
            width=self.config.width,
            height=self.config.height,
        )

        # Update config.output_shape using model_copy to ensure Pydantic validation
        # This allows _reshape_slice_results to use the correct output dimensions
        if hasattr(self.config, "model_copy"):
            self.config = self.config.model_copy(
                update={"output_shape": tuple(target_shape)}
            )
        else:
            # Fallback for older Pydantic versions
            self.config.output_shape = tuple(target_shape)
        logger.info(f"Resizing stack from {original_shape} to {target_shape}")
        return super().run(stack, metadata=metadata, **kwargs)


class RescaleConfig(PreprocessorConfig):

    low: float = 0.0
    high: float = 100.0


class Rescale(Preprocessor):

    def __init__(self, config: RescaleConfig):
        super().__init__(config)

    def _process_slice(
        self, slice: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        plow, phigh = np.percentile(slice, (self.config.low, self.config.high))
        logger.debug(plow, phigh)
        scaled = rescale_intensity(
            slice, out_range=self.config.dtype, in_range=(plow, phigh)
        )
        return scaled


class FuncProcessorConfig(PreprocessorConfig):
    """Configuration for applying an arbitrary function per stack slice."""

    iterator_config: ArrayIteratorConfig = Field(
        default_factory=lambda: ArrayIteratorConfig(slice_def=())
    )
    strict_axis: bool = True
    func: ImportString | None = None
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    output_dims: Optional[Union[Sequence[str], dict[str, int]]] = Field(
        default=None,
        description=(
            "Optional output axis layout after processing. A sequence of axis "
            "letters relabels metadata to match the result rank. A mapping of "
            "axis letter to size reshapes the result and updates metadata "
            "(for example {'Z': 99, 'Y': 512, 'X': 512})."
        ),
    )


class FuncProcessor(Preprocessor):
    """Run a configured callable on each iterated slice of the stack."""

    def __init__(self, config: FuncProcessorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: FuncProcessorConfig) -> "FuncProcessor":
        return cls(config)

    @staticmethod
    def _parse_output_dims(
        spec: Union[Sequence[str], dict[str, int]],
    ) -> tuple[list[str], Optional[tuple[int, ...]]]:
        if isinstance(spec, dict):
            axes = list(spec.keys())
            shape = tuple(int(spec[axis]) for axis in axes)
            return axes, shape
        axes = list(spec)
        return axes, None

    def _apply_output_dims(self, array: np.ndarray) -> np.ndarray:
        spec = self.config.output_dims
        if spec is None:
            return array
        axes, shape = self._parse_output_dims(spec)
        if shape is not None:
            if array.size != int(np.prod(shape)):
                raise ValueError(
                    f"output_dims shape {shape} does not match result size "
                    f"{array.size} (shape {array.shape})"
                )
            return array.reshape(shape)
        if len(axes) != array.ndim:
            raise ValueError(
                f"output_dims length {len(axes)} does not match result ndim "
                f"{array.ndim} (shape {array.shape})"
            )
        return array

    def _update_metadata_for_output_dims(
        self,
        metadata: dict[str, Any],
        new_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        spec = self.config.output_dims
        assert spec is not None
        axes, expected_shape = self._parse_output_dims(spec)
        if expected_shape is not None and tuple(new_shape) != expected_shape:
            raise ValueError(
                f"output_dims shape {expected_shape} does not match result shape "
                f"{new_shape}"
            )

        new_metadata = metadata.copy()
        old_axes = list(metadata.get("axes", []))
        new_metadata["axes"] = axes
        new_metadata["shape"] = tuple(new_shape)
        new_metadata["dims"] = Dimensions(axes, tuple(new_shape))
        new_metadata["dim_order"] = "".join(axes)

        if "scale" in new_metadata and new_metadata["scale"] is not None:
            scale_dict = new_metadata["scale"]._asdict()
            for axis in axes:
                if axis not in scale_dict:
                    scale_dict[axis] = None
            new_metadata["scale"] = Scale(**scale_dict)

        if (
            "physical_pixel_sizes" in new_metadata
            and new_metadata["physical_pixel_sizes"] is not None
        ):
            pps = new_metadata["physical_pixel_sizes"]
            if hasattr(pps, "_asdict"):
                pps_dict = pps._asdict()
                new_metadata["physical_pixel_sizes"] = type(pps)(
                    **{axis: pps_dict.get(axis) for axis in axes}
                )

        added_axes = [axis for axis in axes if axis not in old_axes]
        if added_axes:
            logger.info(
                "Applied output_dims %s; added metadata axes: %s",
                spec,
                added_axes,
            )
        return new_metadata

    def _update_metadata(
        self,
        stack,
        results,
        *args,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[dict[str, Any]]:
        if self.config.output_dims is None:
            return super()._update_metadata(
                stack, results, *args, metadata=metadata, **kwargs
            )
        if metadata is None:
            return None
        return self._update_metadata_for_output_dims(metadata, tuple(results.shape))

    def _process_slice(
        self, slice: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        func = self.config.func
        args = self.config.args
        proc_kwargs = self.config.kwargs.copy()
        if "axis" in proc_kwargs:
            axis_spec = proc_kwargs["axis"]
            if isinstance(axis_spec, (int, np.integer)):
                axis_items = (int(axis_spec),)
            elif isinstance(axis_spec, str):
                axis_items = axis_spec
            else:
                axis_items = axis_spec

            axis_indices = tuple(
                (
                    self._axis_index(metadata, letter)
                    if isinstance(letter, str)
                    else letter
                )
                for letter in axis_items
            )
            axis_indices = tuple(i for i in axis_indices if i is not None)
            if len(axis_indices) == 1:
                proc_kwargs["axis"] = axis_indices[0]
            elif len(axis_indices) == 0:
                proc_kwargs.pop("axis")
            else:
                proc_kwargs["axis"] = axis_indices
            logger.info(
                "Mapped axis %s to axis index %s",
                axis_spec,
                proc_kwargs.get("axis"),
            )
        results = func(slice, *args, **proc_kwargs)
        if self.config.dtype is not None and isinstance(results, np.ndarray):
            results = results.astype(self.config.dtype)
            logger.info("Converted results to %s", results.dtype)
        return self._apply_output_dims(results)
