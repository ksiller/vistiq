from typing import Any, Callable, Optional

import numpy as np
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt, field_validator
from skimage.filters import threshold_local, threshold_otsu
from vistiq.core import StackProcessor, StackProcessorConfig


class ThresholderConfig(StackProcessorConfig):
    """Base configuration for thresholding operations.

    This is the base configuration class for all thresholding methods.
    """

    pass


class RangeThresholdConfig(ThresholderConfig):
    """Configuration for range-based thresholding.

    Thresholds images based on a value range, either absolute or relative.

    Attributes:
        threshold: Tuple of (min, max) threshold values.
        relative: If True, threshold values are relative to image min/max.
            If False, threshold values are absolute.
    """

    threshold: tuple[float, float] = (0.0, 1.0)
    relative: bool = True


class LocalThresholdConfig(ThresholderConfig):
    """Configuration for local adaptive thresholding.

    Performs thresholding using local statistics computed in blocks.

    Attributes:
        block_size: Size of the local neighborhood (must be odd).
        offset: Offset value added to the threshold.
        method: Method for computing local threshold ("gaussian", "mean", etc.).
        mode: Border handling mode for the local computation.
        param: Optional parameter function for threshold calculation.
    """

    block_size: Optional[PositiveInt] = 51
    offset: Optional[NonNegativeFloat] = 0.0
    method: Optional[str] = "gaussian"
    mode: Optional[str] = "reflect"
    param: Optional[Callable] = None

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: int) -> int:
        """Validate that block_size is odd.

        Args:
            v: Block size value to validate.

        Returns:
            Validated block size.

        Raises:
            ValueError: If block_size is even.
        """
        if v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v


class OtsuThresholdConfig(ThresholderConfig):
    """Configuration for Otsu's thresholding method.

    Otsu's method automatically determines the optimal threshold value
    by maximizing the between-class variance.
    """

    # hist: Optional[np.ndarray] = None
    # bins: Optional[int] = 256
    pass


class NiblackThresholdConfig(ThresholderConfig):
    """Configuration for Niblack's adaptive thresholding method.

    Niblack's method computes a local threshold based on mean and standard
    deviation in a local window.

    Attributes:
        window_size: Size of the local window (must be odd).
        offset: Offset value added to the threshold.
        sigma: Standard deviation scaling factor.
    """

    window_size: PositiveInt = 51
    offset: NonNegativeFloat = 0.0
    sigma: PositiveFloat = 1.0

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Validate that window_size is odd.

        Args:
            v: Window size value to validate.

        Returns:
            Validated window size.

        Raises:
            ValueError: If window_size is even.
        """
        if v % 2 == 0:
            raise ValueError("window_size must be odd")
        return v


class SauvolaThresholdConfig(ThresholderConfig):
    """Configuration for Sauvola's adaptive thresholding method.

    Sauvola's method is an improvement over Niblack's method that handles
    varying illumination better.

    Attributes:
        window_size: Size of the local window (must be odd).
        offset: Offset value added to the threshold.
        sigma: Standard deviation scaling factor.
    """

    window_size: PositiveInt = 51
    offset: NonNegativeFloat = 0.0
    sigma: PositiveFloat = 1.0

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Validate that window_size is odd.

        Args:
            v: Window size value to validate.

        Returns:
            Validated window size.

        Raises:
            ValueError: If window_size is even.
        """
        if v % 2 == 0:
            raise ValueError("window_size must be odd")
        return v


class Thresholder(StackProcessor):
    """Base class for thresholding operations.

    Converts grayscale images to binary masks using various thresholding methods.
    """

    @classmethod
    def from_config(cls, config: ThresholderConfig) -> "Thresholder":
        """Create a Thresholder instance from a configuration.

        Args:
            config: Thresholder configuration.

        Returns:
            A new Thresholder instance.
        """
        return cls(config)


class RangeThreshold(Thresholder):
    """Thresholder that uses a fixed value range.

    Creates a binary mask by thresholding pixels within a specified range.
    """

    def __init__(self, config: RangeThresholdConfig):
        """Initialize the range thresholder.

        Args:
            config: Range threshold configuration.
        """
        super().__init__(config)

    def _process_slice(
        self, img: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Apply range thresholding to a single slice.

        Args:
            img: Input image slice.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Binary mask where pixels within the threshold range are True.
        """
        th_min, th_max = self.config.threshold
        if th_min is None:
            th_min = np.min(img)
        if th_max is None:
            th_max = np.max(img)
        return (img >= th_min) & (img <= th_max)


class LocalThreshold(Thresholder):
    """Thresholder that uses local adaptive thresholding.

    Computes threshold values locally for each pixel based on neighborhood statistics.
    """

    def __init__(self, config: LocalThresholdConfig):
        """Initialize the local thresholder.

        Args:
            config: Local threshold configuration.
        """
        super().__init__(config)

    def _process_slice(
        self, img: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Apply local adaptive thresholding to a single slice.

        Args:
            img: Input image slice.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Binary mask from local thresholding.
        """
        th = threshold_local(
            img,
            block_size=self.config.block_size,
            offset=self.config.offset,
            method=self.config.method,
            mode=self.config.mode,
            param=self.config.param if self.config.param is not None else None,
        )
        return img > th


class OtsuThreshold(Thresholder):
    """Thresholder that uses Otsu's method.

    Automatically determines the optimal threshold by maximizing between-class variance.
    """

    def __init__(self, config: OtsuThresholdConfig):
        """Initialize the Otsu thresholder.

        Args:
            config: Otsu threshold configuration.
        """
        super().__init__(config)

    def _process_slice(
        self, img: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Apply Otsu's thresholding to a single slice.

        Args:
            img: Input image slice.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Binary mask from Otsu thresholding.
        """
        th = threshold_otsu(img)
        return img > th
