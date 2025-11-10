import numpy as np
from joblib import Parallel, delayed
from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    NonNegativeFloat,
    PositiveInt,
    PositiveFloat,
)
from typing import Optional, List, Tuple, Union, Callable, Any, Literal
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import disk, binary_dilation
from vistiq.analysis import circularity, aspect_ratio, bbox_width, bbox_height
from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor, ChainProcessorConfig, ChainProcessor
from vistiq.workflow import Workflow

from skimage.filters import (
    threshold_otsu,
    threshold_niblack,
    threshold_sauvola,
    threshold_local,
)


def dilate_regions(
    mask: np.ndarray,
    max_area: float,
) -> np.ndarray:
    """Dilate regions in a 2D binary mask to a target maximum area.

    Identifies individual regions in the mask and dilates each region iteratively
    until its area reaches (but does not exceed) the specified maximum area.
    Ensures that dilation does not fuse adjacent regions by preventing overlap
    with other regions (both original and already-dilated).

    Args:
        mask (np.ndarray): 2D binary mask where True pixels represent regions.
        max_area (float): Maximum area for each region after dilation.
            Regions larger than this will be left unchanged.

    Returns:
        np.ndarray: 2D binary mask with dilated regions. Same shape as input.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    if not np.any(mask):
        return mask.copy()

    # Label regions
    labels = sk_label(mask, connectivity=1)
    props = regionprops(labels)

    # Track all regions that have been processed to prevent fusion
    # This mask contains all original regions plus any already-dilated regions
    occupied_mask = np.zeros_like(mask, dtype=bool)
    dilated_mask = np.zeros_like(mask, dtype=bool)

    for prop in props:
        if prop.area > max_area:
            # Already larger than target, add as-is
            region_mask = labels == prop.label
            dilated_mask |= region_mask
            occupied_mask |= region_mask
            continue

        # Create mask for this region
        region_mask = labels == prop.label
        current_area = prop.area

        # Define forbidden area: all other regions (original + already dilated)
        forbidden = occupied_mask & ~region_mask

        # Iteratively dilate with small structuring element until area reaches target
        dilated = region_mask.copy()
        selem = disk(1)  # Use small disk for fine-grained control
        max_iterations = int(np.ceil(max_area))  # Safety limit
        iteration = 0

        while current_area < max_area and iteration < max_iterations:
            try:
                # Dilate the current region
                new_dilated = binary_dilation(dilated, selem)

                # Remove any pixels that would overlap with other regions
                new_dilated = new_dilated & ~forbidden

                # Calculate area after excluding forbidden regions
                new_area = np.sum(new_dilated)

                if new_area > current_area and new_area <= max_area:
                    # Valid dilation step: area increased and still within limit
                    dilated = new_dilated
                    current_area = new_area
                    iteration += 1
                else:
                    # Would exceed target or no more valid dilation possible, stop
                    break
            except (ValueError, IndexError):
                break

        # Add dilated region to output mask and mark as occupied
        dilated_mask |= dilated
        occupied_mask |= dilated

    return dilated_mask


def assign_unique_labels(
    labeled_arrays: list[np.ndarray] | np.ndarray,
) -> np.ndarray | list[np.ndarray]:
    """Assign unique labels across multiple labeled arrays.

    Takes a list of labeled arrays (or a single array) and ensures that labels
    are unique across all arrays by offsetting labels in each subsequent array.

    Args:
        labeled_arrays (list[np.ndarray] | np.ndarray): List of labeled arrays
            or a single labeled array. Each array should have integer labels where
            0 represents background.

    Returns:
        np.ndarray | list[np.ndarray]: Array(s) with unique labels. Returns a single
            array if input is a single array, otherwise returns a list of arrays.
    """
    if isinstance(labeled_arrays, list):
        labeled_arrays = np.stack(labeled_arrays, axis=0)
    if labeled_arrays.ndim == 2:
        # nothing to do
        return labeled_arrays

    # make a copy to avoid modifying the original arrays
    result = labeled_arrays.copy()
    current_max_label = 0

    for i, arr in enumerate(result):
        if arr.size == 0:
            continue

        # Get unique labels (excluding background 0)
        unique_labels = np.unique(arr[arr > 0])

        if len(unique_labels) > 0:
            # Offset all non-zero labels
            for label in unique_labels:
                arr[arr == label] = label + current_max_label

            # Update max label for next iteration
            current_max_label = int(np.max(arr))

    return result


class ThresholderConfig(StackProcessorConfig):
    pass


class RangeThresholderConfig(ThresholderConfig):
    threshold: tuple[float, float] = (0.0, 1.0)
    relative: bool = True


class LocalThresholderConfig(ThresholderConfig):
    block_size: Optional[PositiveInt] = 51
    offset: Optional[NonNegativeFloat] = 0.0
    method: Optional[str] = "gaussian"
    mode: Optional[str] = "reflect"
    param: Optional[Callable] = None

    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v


class OtsuThresholderConfig(ThresholderConfig):
    # hist: Optional[np.ndarray] = None
    # bins: Optional[int] = 256
    pass


class NiblackThresholderConfig(ThresholderConfig):
    window_size: PositiveInt = 51
    offset: NonNegativeFloat = 0.0
    sigma: PositiveFloat = 1.0

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("window_size must be odd")
        return v


class SauvolaThresholderConfig(ThresholderConfig):
    window_size: PositiveInt = 51
    offset: NonNegativeFloat = 0.0
    sigma: PositiveFloat = 1.0

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("window_size must be odd")
        return v


class RangeFilterConfig(Configuration):
    attribute: str = None
    range: Union[tuple[float, float], str] = None


class RangeFilter(Configurable):
    def __init__(self, config: RangeFilterConfig):
        super().__init__(config)

    def min_value(self) -> float:
        return (
            self.config.range[0] if not isinstance(self.config.range, str) else -np.inf
        )

    def max_value(self) -> float:
        return (
            self.config.range[1] if not isinstance(self.config.range, str) else +np.inf
        )

    def discretize(self, target_value: float, tolerance: float) -> None:
        """Discretize the filter."""
        self.config.range = (target_value - tolerance, target_value + tolerance)

    def in_range(self, value: float) -> bool:
        print(
            f"self.config={self.config}, value={value}, min_value={self.min_value()}, max_value={self.max_value()}"
        )
        return value >= self.min_value() and value <= self.max_value()


class RegionFilterConfig(Configuration):
    filters: List[RangeFilterConfig] = None

    @classmethod
    def allowed_attributes(cls) -> List[str]:
        return [
            "area",
            "solidity",
            "aspect_ratio",
            "eccentricity",
        ] + Labeller.extra_properties

    @model_validator(mode="after")
    def validate_filters(self) -> "RegionFilterConfig":
        """Validate that all filter attributes are in the allowed list."""
        if self.filters is None:
            return self

        allowed = self.allowed_attributes()
        for filter_config in self.filters:
            if filter_config.attribute is None:
                continue
            if filter_config.attribute not in allowed:
                raise ValueError(
                    f"Filter attribute '{filter_config.attribute}' is not allowed. "
                    f"Allowed attributes are: {allowed}"
                )
        return self


class RegionFilter(Configurable[RegionFilterConfig]):

    def __init__(self, config: RegionFilterConfig):
        super().__init__(config)
        self.filters = [
            RangeFilter(filter_config) for filter_config in self.config.filters
        ]

    @classmethod
    def from_config(cls, config: RegionFilterConfig) -> "RegionFilter":
        """Create a RegionFilter instance from a configuration.

        Args:
            config: RegionFilter configuration.

        Returns:
            A new RegionFilter instance.
        """
        return cls(config)

    def has_filter(self, attribute: str) -> bool:
        for filter in self.config.filters:
            if filter.attribute == attribute:
                return True
        return False

    def get_filter(self, attribute: str) -> RangeFilter:
        for filter in self.filters:
            if filter.config.attribute == attribute:
                return filter
        raise ValueError(f"Filter for attribute '{attribute}' not found")

    def run(
        self, regions: List["RegionProperties"]
    ) -> Tuple[List["RegionProperties"], List["RegionProperties"]]:
        """Run the filter."""
        accepted_regions = []
        removed_regions = []
        for region in regions:
            for filter in self.filters:
                print(f"checking filter: {filter.config.attribute}")
                if not filter.in_range(getattr(region, filter.config.attribute)):
                    print(
                        f"filter {filter.config.attribute} not in range for region {region.label}"
                    )
                    removed_regions.append(region)
                    break
            print(
                f"filter {filter.config.attribute} in range for region {region.label}"
            )
            accepted_regions.append(region)
        return accepted_regions, removed_regions


class BinaryProcessorConfig(StackProcessorConfig):
    watershed: bool = False
    dilation_target_area: NonNegativeFloat = 0.0


class LabellerConfig(StackProcessorConfig):
    connectivity: PositiveInt = 1


class RelabelerConfig(StackProcessorConfig):
    by_plane: bool = True
    squeeze: bool = True


class Relabeler(StackProcessor):
    def __init__(self, config: RelabelerConfig):
        super().__init__(config)

    def run(
        self,
        labels: List[np.ndarray],
        regions: Optional[List["RegionProperties"]] = None,
    ) -> List[np.ndarray]:
        """Run the relabeler."""
        return assign_unique_labels(labels, regions)

        extra_properties = (
            (
                circularity,
                bbox_width,
                bbox_height,
                aspect_ratio,
            ),
        )


class Labeller(StackProcessor):
    """Labeller workflow step. Labels a binary mask."""

    max_dim_per_process: int = 3
    output_type: Literal["stack", "list"] = "list"
    extra_properties: List[str] = [
        circularity,
        aspect_ratio,
        bbox_width,
        bbox_height,
    ]

    def __init__(self, config: LabellerConfig):
        super().__init__(config)

    def _process_stack(self, mask: np.ndarray) -> np.ndarray:
        """Process the mask."""
        labels = sk_label(mask, connectivity=self.config.connectivity)
        regions = regionprops(labels)
        return labels, regions

    def _process_by_plane(self, mask: np.ndarray) -> np.ndarray:
        """Process the mask by plane."""
        labels = sk_label(mask, connectivity=self.config.connectivity)

        regions = regionprops(labels, extra_properties=Labeller.extra_properties)
        return labels, regions

    def run(self, mask: np.ndarray) -> np.ndarray:
        """Run the labeller."""
        labels, regions = super().run(mask)
        if self.config.by_plane:
            labels = assign_unique_labels(labels)
        return labels, regions


class RegionProcessorConfig(Configuration):
    relabel_by_plane: bool = True
    update_labels: bool = False


class RegionProcessor(Configurable[RegionProcessorConfig]):
    """Region processor workflow step. Processes a list of region properties."""

    def __init__(self, config: RegionProcessorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: RegionProcessorConfig) -> "RegionProcessor":
        """Create a RegionProcessor instance from a configuration.

        Args:
            config: RegionProcessor configuration.

        Returns:
            A new RegionProcessor instance.
        """
        return cls(config)

    def run(self, region: "RegionProperties", label: np.ndarray) -> "RegionProperties":
        """Run the region processor."""
        return region, label


class Thresholder(Configurable[ThresholderConfig]):
    @classmethod
    def from_config(cls, config: ThresholderConfig) -> "Thresholder":
        """Create a Thresholder instance from a configuration.

        Args:
            config: Thresholder configuration.

        Returns:
            A new Thresholder instance.
        """
        return cls(config)


class RangeThresholder(Thresholder):
    def __init__(self, config: RangeThresholderConfig):
        super().__init__(config)

    def run(self, img: np.ndarray) -> np.ndarray:
        """Run the thresholder."""
        th_min, th_max = self.config.threshold
        if th_min is None:
            th_min = np.min(img)
        if th_max is None:
            th_max = np.max(img)
        return (img >= th_min) & (img <= th_max)


class LocalThresholder(Thresholder):
    def __init__(self, config: LocalThresholderConfig):
        super().__init__(config)

    def run(self, img: np.ndarray) -> np.ndarray:
        """Run the thresholder."""
        th = threshold_local(
            img,
            block_size=self.config.block_size,
            offset=self.config.offset,
            method=self.config.method,
            mode=self.config.mode,
            param=self.config.param if self.config.param is not None else None,
        )
        return img > th


class OtsuThresholder(Thresholder):
    def __init__(self, config: OtsuThresholderConfig):
        super().__init__(config)

    def run(self, img: np.ndarray) -> np.ndarray:
        """Run the thresholder."""
        th = threshold_otsu(img)
        return img > th


class BinaryProcessor(Configurable[BinaryProcessorConfig]):
    def __init__(self, config: BinaryProcessorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: BinaryProcessorConfig) -> "BinaryProcessor":
        """Create a BinaryProcessor instance from a configuration.

        Args:
            config: BinaryProcessor configuration.

        Returns:
            A new BinaryProcessor instance.
        """
        return cls(config)

    def run(self, mask: np.ndarray) -> np.ndarray:
        """Run the binary processor."""
        return mask


class Labeller(Configurable[LabellerConfig]):
    def __init__(self, config: LabellerConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: LabellerConfig) -> "Labeller":
        """Create a Labeller instance from a configuration.

        Args:
            config: Labeller configuration.

        Returns:
            A new Labeller instance.
        """
        return cls(config)

    def run(self, mask: np.ndarray) -> np.ndarray:
        """Run the labeller."""
        labels = sk_label(mask, connectivity=self.config.connectivity)
        regions = regionprops(labels)
        return labels, regions


class SegmenterConfig(Configuration):
    thresholder: Optional[ThresholderConfig] = None
    binary_processor: Optional[BinaryProcessorConfig] = None
    labeller: Optional[LabellerConfig] = None
    region_filter: Optional[RegionFilterConfig] = None

    @model_validator(mode="after")
    @classmethod
    def check_labeller(cls, data: Any) -> Any:
        if "region_filter" in data and "labeller" not in data:
            raise ValueError(
                "A labeller must be provided if a region filter is specified"
            )
        return data


class Segmenter(Workflow):
    def __init__(
        self,
        thresholder: Thresholder = None,
        binary_processor: BinaryProcessor = None,
        labeller: Labeller = None,
        region_filter: RegionFilter = None,
        config: SegmenterConfig = None,
    ):
        super().__init__(config)
        self.thresholder = thresholder
        self.binary_processor = binary_processor
        self.labeller = labeller
        self.region_filter = region_filter

    def _labels(self, masks: List[np.ndarray] | np.ndarray) -> np.ndarray:
        """Label the images.

        Args:
            labeled_images (List[np.ndarray]|np.ndarray): List of labeled images or a single labeled image.

        Returns:
            np.ndarray: Labels of the images.
        """

        if isinstance(labeled_images, list):
            labeled_images = np.stack(labeled_images, axis=0)
        if labeled_images.ndim == 2:
            labeled_images = labeled_images[None, ...]
        labels = sk_label(masks, connectivity=sconnectivity)
        return labels

    def _regions(self, labels: np.ndarray) -> List["RegionProperties"]:
        """Get the regions from the labels.

        Args:
            labels (np.ndarray): Labels of the images.

        Returns:
            List[RegionProps]: Regions of the images.
        """
        return regionprops(labels)

    def run(
        self,
        img: np.ndarray,
        include_mask: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List["RegionProperties"]]]:
        """Run the segment step.

        Args:
            img (np.ndarray): Input image.
            include_mask (Optional[np.ndarray]): Optional mask to include in the segmentation.
            exclude_mask (Optional[np.ndarray]): Optional mask to exclude from the segmentation.
            do_regions (bool): Whether to return regions.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, List[RegionProps]]]: Binary mask or tuple of binary mask and regions.
        """
        binary_mask = self.thresholder.run(img)
        if include_mask is not None:
            binary_mask = binary_mask & include_mask
        if exclude_mask is not None:
            binary_mask = binary_mask & ~exclude_mask
        if self.binary_processor is not None:
            binary_mask = self.binary_processor.run(binary_mask)
        if self.region_filter is not None:
            labels, regions = self.labeller.run(binary_mask)
            regions = self.region_filter.run(regions)
        return binary_mask, labels, regions


class IterativeSegmenterConfig(SegmenterConfig):
    iterations: PositiveInt = 10


class IterativeSegmenter(Workflow):
    """Iterative segmentation workflow.

    Args:
        config (IterativeSegmentationConfig): Configuration for the iterative segmentation workflow.
    """

    def __init__(self, config: IterativeSegmenterConfig):
        super().__init__(config)

    def _dilate_regions(self, mask: np.ndarray) -> np.ndarray:
        """Dilate the regions in the mask.

        Args:
            mask (np.ndarray): Mask to dilate.

        Returns:
            np.ndarray: Dilated mask.
        """
        if self.dilate_target_area is not None:
            if mask.ndim == 2:
                return dilate_regions(mask, self.dilate_target_area)
            elif mask.ndim == 3:
                return np.stack(
                    [
                        dilate_regions(m_plane, self.dilate_target_area)
                        for m_plane in mask
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"Mask must be 2D or 3D, got {mask.ndim}D")
        return mask

    def run(
        self,
        img: np.ndarray,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List["RegionProperties"]]]:
        """Run the segmentor.

        Args:
            img (np.ndarray): Input image.
            include_mask (Optional[np.ndarray]): Optional mask to include in the segmentation.
            exclude_mask (Optional[np.ndarray]): Optional mask to exclude from the segmentation.
        """
        include_mask = None
        exclude_mask = None
        regions = []
        masks = []
        labels = []
        for s in self.steps:
            mask, regions, labels = s.run(img, include_mask, exclude_mask)
            dilated_mask = self._dilate_regions(mask)
            if exclude_mask is not None:
                exclude_mask = dilated_mask | exclude_mask
            else:
                exclude_mask = dilated_mask
            regions.append(regions)
            masks.append(dilated_mask)
            labels.append(labels)
        if self.output == "last":
            masks = masks[-1]
            regions = regions[-1]
            labels = labels[-1]
            return masks, labels, regions
        labels = assign_unique_labels(labels)
        if self.output == "stack":
            masks = np.stack(masks, axis=0)
            regions = np.stack(regions, axis=0)
            labels = np.stack(labels, axis=0)
        elif self.output == "combine":
            masks = np.sum(masks, axis=0)
            regions = np.sum(regions, axis=0)
            labels = np.sum(labels, axis=0)
        return masks, labels, regions


class SeriesSegmenterConfig(SegmenterConfig):
    segmenters: List[SegmenterConfig]
    output: str = "stack"

    @field_validator("output")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed_categories = ["stack", "combine", "last", "list"]
        if v not in allowed_categories:
            raise ValueError(
                f"'{v}' is not an allowed category. Must be one of: {allowed_categories}"
            )
        return v


class SeriesSegmenter(Workflow):
    def __init__(self, config: SeriesSegmenterConfig):
        super().__init__(config)

    def run(
        self, img: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, List["RegionProperties"]]]:
        """Run the segmenter."""
        for segmenter in self.config.segmenters:
            mask, regions, labels = segmenter.run(img)
            return mask, regions, labels

