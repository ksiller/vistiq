from functools import cached_property
import numpy as np
from joblib import Parallel, delayed
from pydantic import (
    field_validator,
    model_validator,
    NonNegativeFloat,
    PositiveInt,
    PositiveFloat,
)
import pandas as pd
from typing import Optional, List, Tuple, Union, Callable, Any, Literal, Dict
import logging
from skimage.measure import label as sk_label
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import disk, binary_dilation
from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor, ChainProcessorConfig, ChainProcessor
from vistiq.workflow import Workflow
from vistiq.utils import ArrayIterator, ArrayIteratorConfig

logger = logging.getLogger(__name__)

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


class RangeFilterConfig(Configuration):
    """Configuration for range-based region filtering.
    
    Filters regions based on whether a specified attribute value falls
    within a given range.
    
    Attributes:
        attribute: Name of the region property to filter on.
        range: Tuple of (min, max) values, or "all" to accept all values.
    """
    attribute: str = None
    range: Union[tuple[float, float], str] = None


class RangeFilter(Configurable):
    """Filter that checks if a value falls within a specified range.
    
    This filter can be used to filter regions based on whether a property
    value falls within a min/max range.
    """
    
    def __init__(self, config: RangeFilterConfig):
        """Initialize the range filter.
        
        Args:
            config: Range filter configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: RangeFilterConfig) -> "RangeFilter":
        """Create a RangeFilter instance from a configuration.

        Args:
            config: RangeFilter configuration.

        Returns:
            A new RangeFilter instance.
        """
        return cls(config)    
    
    def min_value(self) -> float:
        """Get the minimum value for the filter range.
        
        Returns:
            Minimum value, or -infinity if range is "all".
        """
        return (
            self.config.range[0] if not isinstance(self.config.range, str) else -np.inf
        )

    def max_value(self) -> float:
        """Get the maximum value for the filter range.
        
        Returns:
            Maximum value, or +infinity if range is "all".
        """
        return (
            self.config.range[1] if not isinstance(self.config.range, str) else +np.inf
        )

    def discretize(self, target_value: float, tolerance: float) -> None:
        """Discretize the filter to a target value with tolerance.
        
        Sets the filter range to (target_value - tolerance, target_value + tolerance).
        
        Args:
            target_value: Center value for the range.
            tolerance: Half-width of the range.
        """
        self.config.range = (target_value - tolerance, target_value + tolerance)

    def in_range(self, value: float) -> bool:
        """Check if a value falls within the filter range.
        
        Args:
            value: Value to check.
            
        Returns:
            True if value is within [min_value, max_value], False otherwise.
        """
        #print(
        #    f"self.config={self.config}, value={value}, min_value={self.min_value()}, max_value={self.max_value()}"
        #)
        return value >= self.min_value() and value <= self.max_value()


class RegionFilterConfig(Configuration):
    """Configuration for region filtering operations.
    
    Filters regions based on multiple criteria using range filters.
    
    Attributes:
        filters: List of RangeFilter instances to apply to regions.
    """
    filters: List[RangeFilter] = []

    @model_validator(mode="after")
    def validate_filters(self) -> "RegionFilterConfig":
        """Validate that all filter attributes are in the allowed list.
        
        Returns:
            Validated configuration.
            
        Raises:
            ValueError: If any filter attribute is not in the allowed properties list.
        """
        if self.filters is None:
            self.filters = []
            return self

        allowed = RegionAnalyzer.allowed_properties()
        for filter in self.filters:
            if filter.config.attribute is None:
                continue
            if filter.config.attribute not in allowed:
                raise ValueError(
                    f"Filter attribute '{filter.config.attribute}' is not allowed. "
                    f"Allowed attributes are: {allowed}"
                )
        return self


class RegionFilter(Configurable[RegionFilterConfig]):
    """Filter that removes regions based on property value ranges.
    
    Applies multiple range filters to a list of regions, removing regions
    that don't pass all filter criteria.
    """

    def __init__(self, config: RegionFilterConfig):
        """Initialize the region filter.
        
        Args:
            config: Region filter configuration.
        """
        super().__init__(config)
        #self.filters = [
        #    RangeFilter(filter_config) for filter_config in self.config.filters
        #]

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
        """Check if a filter exists for the given attribute.
        
        Args:
            attribute: Name of the attribute to check.
            
        Returns:
            True if a filter exists for this attribute, False otherwise.
        """
        for filter in self.config.filters:
            if filter.config.attribute == attribute:
                return True
        return False

    def get_filter(self, attribute: str) -> RangeFilter:
        """Get the filter for a specific attribute.
        
        Args:
            attribute: Name of the attribute.
            
        Returns:
            RangeFilter for the specified attribute.
            
        Raises:
            ValueError: If no filter exists for the attribute.
        """
        for filter in self.config.filters:
            if filter.config.attribute == attribute:
                return filter
        raise ValueError(f"Filter for attribute '{attribute}' not found")

    def run(
        self, regions: List["RegionProperties"]
    ) -> Tuple[List["RegionProperties"], List["RegionProperties"]]:
        """Filter regions based on configured criteria.
        
        Removes regions that don't pass all filter criteria. A region is
        removed if any of its property values fall outside the specified range.
        
        Args:
            regions: List of region properties to filter.
            
        Returns:
            Tuple of (accepted_regions, removed_regions):
            - accepted_regions: Regions that passed all filters.
            - removed_regions: Regions that failed at least one filter.
        """
        if self.config.filters is None or len(self.config.filters) == 0:
            logger.info("RegionFilter: no filters, returning all regions")
            return regions, []
        removed_regions = []
        for region in regions:
            for filter in self.config.filters:
                value = getattr(region, filter.config.attribute)
                if not filter.in_range(value):
                    #logger.debug(
                    #    f"filter {filter.config.attribute} value={value} not in range for region {region.label}"
                    #)
                    removed_regions.append(region)
                    break
        accepted_regions = [region for region in regions if region not in removed_regions]
        logger.info(f"RegionFilter: len(accepted_regions)={len(accepted_regions)}, len(removed_regions)={len(removed_regions)}")
        return accepted_regions, removed_regions


class BinaryProcessorConfig(StackProcessorConfig):
    """Configuration for binary image processing operations.
    
    Attributes:
        watershed: Whether to apply watershed segmentation.
        dilation_target_area: Target area for region dilation (0.0 = no dilation).
    """
    watershed: bool = False
    dilation_target_area: NonNegativeFloat = 0.0


class RelabelerConfig(StackProcessorConfig):
    """Configuration for relabeling operations.
    
    Relabels arrays to ensure unique labels across multiple labeled arrays.
    
    Attributes:
        output_type: Output format ("stack" for stacked array).
        squeeze: Whether to squeeze output dimensions.
    """
    output_type: Literal["stack"] = "stack" # force output type to stack
    squeeze: bool = True # don't squeeze the output


class Relabeler(StackProcessor):
    """Relabeler that ensures unique labels across multiple labeled arrays.
    
    Takes labeled arrays and reassigns labels so that labels are unique
    across all arrays by offsetting labels in each subsequent array.
    """
    
    def __init__(self, config: RelabelerConfig):
        """Initialize the relabeler.
        
        Args:
            config: Relabeler configuration.
        """
        super().__init__(config)

    @classmethod
    def assign_unique_labels(
        cls,
        labeled_arrays: list[np.ndarray] | np.ndarray,
        iterator: Optional[ArrayIterator] = None,
    ) -> tuple[np.ndarray | list[np.ndarray], dict[tuple, list[tuple[int, int]]]]:
        """Assign unique labels across multiple labeled arrays.

        Takes a list of labeled arrays (or a single array) and ensures that labels
        are unique across all arrays by offsetting labels in each subsequent array.

        Args:
            labeled_arrays (list[np.ndarray] | np.ndarray): List of labeled arrays
                or a single labeled array. Each array should have integer labels where
                0 represents background.
            iterator (ArrayIterator, optional): ArrayIterator for flexible iteration.
                Defaults to ArrayIterator with slice_def=[-2,-1] (iterates over first
                axis, keeps last 2 axes).

        Returns:
            tuple[np.ndarray | list[np.ndarray], dict[tuple, list[tuple[int, int]]]]: 
                A tuple containing:
                - Array(s) with unique labels. Returns a single array if input is a single 
                  array, otherwise returns a list of arrays.
                - Dictionary mapping array index tuple (from iterator.indices) to list of 
                  (old_label, new_label) tuples for all label mappings in that slice.
        """
        label_mappings: dict[tuple, list[tuple[int, int]]] = {}
        was_list = isinstance(labeled_arrays, list)
        
        if was_list:
            labeled_arrays = np.stack(labeled_arrays, axis=0)
        if labeled_arrays.ndim == 2:
            # nothing to do, return empty mapping
            return labeled_arrays, {}

        # make a copy to avoid modifying the original arrays
        result = labeled_arrays.copy()
        
        # Create iterator if not provided
        if iterator is None:
            iterator = ArrayIterator(result, ArrayIteratorConfig(slice_def=(-2, -1)))
        else:
            # Create a new iterator with the result array, using the provided iterator's config
            iterator = ArrayIterator(result, iterator.config)
        
        current_max_label = 0

        # Iterate over slices using the iterator
        for index, arr_slice in enumerate(iterator):
            # Use iterator.indices[index] as the key for label_mappings
            index_tuple = iterator.indices[index]
            if arr_slice.size == 0:
                label_mappings[index_tuple] = []
                continue

            # Get unique labels (excluding background 0)
            unique_labels = np.unique(arr_slice[arr_slice > 0])
            label_mappings[index_tuple] = []

            if len(unique_labels) > 0:
                # Offset all non-zero labels and track mappings
                for label in unique_labels:
                    new_label = label + current_max_label
                    arr_slice[arr_slice == label] = new_label
                    label_mappings[index_tuple].append((int(label), int(new_label)))

                # Update max label for next iteration
                current_max_label = int(np.max(arr_slice))
            
            index += 1

        return result, label_mappings

    def _process_slice(self, labels: np.ndarray) -> np.ndarray:
        """Process a single slice by assigning unique labels.
        
        Args:
            labels: Labeled array slice.
            
        Returns:
            Relabeled array with unique labels.
        """
        # For a single slice, no relabeling needed (already unique)
        return labels

    def run(
        self,
        labels: np.ndarray | List[np.ndarray],
    ) -> np.ndarray:
        """Run the relabeler to assign unique labels.
        
        Args:
            labels: Labeled array or list of labeled arrays to relabel.
            
        Returns:
            Relabeled array with unique labels. Same shape as input (stacked if input was a list).
        """
        # Convert list to array if needed
        if isinstance(labels, list):
            labels_array = np.stack(labels, axis=0)
        else:
            labels_array = labels
        
        # Store original shape
        original_shape = labels_array.shape
        
        # Create iterator from config
        iterator = ArrayIterator(labels_array, self.config.iterator_config)
        
        # Use assign_unique_labels with the iterator
        relabeled_labels, _ = self.assign_unique_labels(labels_array, iterator)
        
        # Ensure output has same shape as input
        if relabeled_labels.shape != original_shape:
            # Reshape if needed (shouldn't happen, but just in case)
            relabeled_labels = relabeled_labels.reshape(original_shape)
        
        return relabeled_labels


class LabelRemoverConfig(StackProcessorConfig):
    """Configuration for label removal operations.
    
    This configuration defines how labels should be removed from label arrays.
    """
    iterator_config: ArrayIteratorConfig = ArrayIteratorConfig(slice_def=()) # no slicing, remove all labels
    output_type: Literal["stack"] = "stack" # force output type to stack
    squeeze: bool = False # don't squeeze the output


class LabelRemover(StackProcessor):
    """Remove specified labels from a label array by setting them to background (0).
    
    This class processes label arrays by removing (masking) specified label IDs,
    setting their pixels to background (0).
    """
    
    def __init__(self, config: LabelRemoverConfig):
        """Initialize the label remover.
        
        Args:
            config: Label remover configuration.
        """
        super().__init__(config)
    
    @classmethod
    def from_config(cls, config: LabelRemoverConfig) -> "LabelRemover":
        """Create a LabelRemover instance from a configuration.
        
        Args:
            config: Label remover configuration.
            
        Returns:
            A new LabelRemover instance.
        """
        return cls(config)
    
    def _extract_label_ids(
        self, 
        label_ids: Union[List["RegionProperties"], List[int], np.ndarray]
    ) -> np.ndarray:
        """Extract label IDs from various input formats.
        
        Args:
            label_ids: Can be:
                - List of RegionProperties (extracts .label attribute)
                - List of ints
                - numpy array of ints
                
        Returns:
            numpy array of label IDs to remove.
        """
        if isinstance(label_ids, list) and len(label_ids) > 0:
            # Check if first element is a RegionProperties object
            if hasattr(label_ids[0], 'label'):
                # Extract label attribute from RegionProperties
                return np.array([region.label for region in label_ids], dtype=np.int32)
            else:
                # Assume it's a list of ints
                return np.array(label_ids, dtype=np.int32)
        elif isinstance(label_ids, np.ndarray):
            return label_ids.astype(np.int32)
        else:
            # Empty or None
            return np.array([], dtype=np.int32)
    
    def _process_slice(self, labels: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
        """Process a single slice by removing specified labels.
        
        Args:
            labels: Label array slice.
            label_ids: Array of label IDs to remove.
            
        Returns:
            Processed label array with specified labels set to 0.
        """
        result = labels.copy()
        if len(label_ids) > 0:
            # Create mask for all labels to remove
            mask = np.isin(result, label_ids)
            # Set masked pixels to background (0)
            result[mask] = 0
        return result
    
    def run(
        self, 
        labels: np.ndarray, 
        label_ids: Union[List["RegionProperties"], List[int], np.ndarray],
        workers: int = -1, 
        verbose: int = 10
    ) -> np.ndarray:
        """Remove specified labels from the label array.
        
        Args:
            labels: Input label array.
            label_ids: Labels to remove. Can be:
                - List of RegionProperties (extracts .label attribute)
                - List of ints
                - numpy array of ints
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.
            
        Returns:
            Processed label array with specified labels set to background (0).
        """
        # Extract label IDs from various input formats
        label_ids_array = self._extract_label_ids(label_ids)
        
        if len(label_ids_array) == 0:
            # No labels to remove, return original
            return labels
        
        # Use parent's run method with label_ids as additional argument
        return super().run(labels, label_ids_array, workers=workers, verbose=verbose)


class LabellerConfig(StackProcessorConfig):
    """Configuration for labeling operations.
    
    Labels connected components in binary masks.
    
    Attributes:
        connectivity: Connectivity for labeling (1 for 4-connected, 2 for 8-connected).
        region_filter: Optional region filter to apply after labeling.
        output_type: Output format ("list" for list of arrays).
    """
    connectivity: PositiveInt = 1
    region_filter: Optional[RegionFilter] = None
    output_type: Literal["list"] = "list"


class Labeller(StackProcessor):
    """Labeler that identifies connected components in binary masks.
    
    Labels connected regions in binary masks and optionally filters regions
    based on property criteria.
    """

    test: str = "test"
    max_dim_per_process: int = 3


    def __init__(self, config: LabellerConfig):
        """Initialize the labeller.
        
        Args:
            config: Labeller configuration.
        """
        super().__init__(config)
        # Create a mapping from function names to functions for extra_properties
        #extra_props_map = {func.__name__: func for func in Labeller.extra_properties}
        
        # Determine which extra_properties are needed based on region_filter
        #if self.config.region_filter is not None and self.config.region_filter.config.filters is not None:
        #    # Get attribute names used by filters
        #    filter_attributes = {
        #        filter.config.attribute 
        #        for filter in self.config.region_filter.config.filters 
        #        if filter.config.attribute is not None
        #    }
        #    # Map attribute names to actual callable functions
        #    self.use_extra_properties = [
        #        extra_props_map[attr] 
        #        for attr in filter_attributes 
        #        if attr in extra_props_map
        #    ]
        #else:
        #    # No filter or no filters specified
        #    self.use_extra_properties = None

    #def _process_stack(self, mask: np.ndarray) -> np.ndarray:
    #    """Process the mask."""
    #    labels = sk_label(mask, connectivity=self.config.connectivity)
    #    regions = regionprops(labels)
    #    return labels, regions

    def _process_slice(self, mask: np.ndarray) -> tuple[np.ndarray, List["RegionProperties"]]:
        """Process a single slice by labeling connected components.
        
        Labels connected components in the binary mask and optionally filters
        regions based on configured criteria.
        
        Args:
            mask: Binary mask to label.
            
        Returns:
            Tuple of (labels, regions):
            - labels: Labeled array with unique integer labels for each region.
            - regions: List of region properties for each labeled region.
        """
        labels = sk_label(mask, connectivity=self.config.connectivity)
        if self.config.region_filter is not None and self.config.region_filter.config.filters is not None:
            extra_properties = [filter.config.attribute for filter in self.config.region_filter.config.filters if filter.config.attribute is not None and filter.config.attribute in RegionAnalyzer.extra_properties_funcs().keys()]
        else:
            extra_properties = []
        logger.info(f"extra_properties={extra_properties}")
        ra = RegionAnalyzer(RegionAnalyzerConfig(properties=extra_properties))
        regions = ra.run(labels)
        logger.info(f"Labeller: len(regions)={len(regions)}")

        if self.config.region_filter is not None:
            logger.info(f"Labeller: self.config.region_filter.config={self.config.region_filter.config}")
            # Store original labels before filtering
            original_labels = labels.copy()
            regions, removed_regions = self.config.region_filter.run(regions)
            labels = np.zeros_like(labels)
            for region in regions:
                # Use original_labels to create mask, not the zeroed labels
                region_mask = original_labels == region.label
                labels[region_mask] = region.label
        return labels, regions

    def run(self, mask: np.ndarray, workers: int = -1, verbose: int = 10) -> tuple[np.ndarray, List["RegionProperties"]]:
        """Run the labeller on a binary mask.
        
        Labels connected components in the mask and optionally filters regions.
        
        Args:
            mask: Binary mask to label.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.
            
        Returns:
            Tuple of (labels, regions):
            - labels: Labeled array with unique integer labels.
            - regions: List of region properties.
        """
        labels, regions = super().run(mask, workers=workers, verbose=verbose)
        #print (f"type(labels)={type(labels)}, type(regions)={type(regions)}")
        #if len(labels) > 1:
        #    iterator = ArrayIterator(labels, self.config.iterator_config)
        #    labels, labels_map = assign_unique_labels(labels, iterator)
        #    print (f"type(labels)={type(labels)}, type(labels_map)={type(labels_map)}")
        #    regions = remap_regions(regions, labels_map)
        return labels, regions


class RegionAnalyzer(StackProcessor):
    """Analyzer that extracts region properties from labeled images.
    
    Computes properties for each labeled region, including built-in properties
    from scikit-image and custom extra properties like circularity and aspect ratio.
    """
    
    default_properties: List[str] = ["label", "centroid"]

    def __init__(self, config: "RegionAnalyzerConfig"):
        """Initialize the region analyzer.
        
        Args:
            config: Region analyzer configuration.
        """
        super().__init__(config)

    #@cached_property
    @staticmethod
    def builtin_properties() -> List[str]:
        """Get list of built-in region properties from scikit-image.
        
        Returns:
            List of property names available from regionprops.
        """
        fake_array=np.ones((2,2))
        labels = sk_label(fake_array)
        regions = regionprops(labels)
        return [attr for attr in dir(regions[0]) if not attr.startswith("_")]

    @classmethod
    def extra_properties_funcs(cls) -> Dict[str, Callable]:
        """Get dictionary of custom extra property functions.
        
        Returns:
            Dictionary mapping property names to their computation functions.
        """
        return {
            "circularity": cls.circularity,
            "aspect_ratio": cls.aspect_ratio,
            "bbox_width": cls.bbox_width,
            "bbox_height": cls.bbox_height,
        }

    @staticmethod
    def allowed_properties() -> List[str]:
        """Get list of all allowed property names.
        
        Returns:
            Combined list of built-in and custom property names.
        """
        return RegionAnalyzer.builtin_properties() + list(RegionAnalyzer.extra_properties_funcs().keys())

    def used_extra_properties(self) -> List[str]:
        """Get list of extra properties that are being used.
        
        Returns:
            List of extra property names from config that are custom properties.
        """
        return [prop for prop in self.config.properties if prop in RegionAnalyzer.extra_properties_funcs().keys()]

    def used_extra_properties_funcs(self) -> List[Callable]:
        """Get list of extra property functions that are being used.
        
        Returns:
            List of callable functions for the extra properties being used.
        """
        uep = self.used_extra_properties()
        return [func for k, func in RegionAnalyzer.extra_properties_funcs().items() if k in uep]

    def used_builtin_properties(self) -> List[str]:
        """Get list of built-in properties that are being used.
        
        Returns:
            List of built-in property names from config.
        """
        return [prop for prop in self.config.properties if prop in RegionAnalyzer.builtin_properties()]

    @classmethod
    def from_config(cls, config: "RegionAnalyzerConfig") -> "RegionAnalyzer":
        """Create a RegionAnalyzer instance from a configuration.

        Args:
            config: RegionAnalyzer configuration.

        Returns:
            A new RegionProcessor instance.
        """
        return cls(config)

    @staticmethod
    def circularity(regionmask, intensity_image=None):
        """Compute circularity: 4π * area / perimeter² (perfect circle = 1.0).
        
        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            
        Returns:
            Circularity value (1.0 for perfect circle), or NaN if invalid.
        """
        from skimage.measure import perimeter

        perim = perimeter(regionmask)
        area = np.sum(regionmask)
        if perim > 0:
            return float(4.0 * np.pi * area / (perim**2))
        return float("nan")

    @staticmethod
    def bbox_width(regionmask, intensity_image=None):
        """Compute bounding box width from region mask.

        Uses numpy operations to find column bounds more efficiently.
        
        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            
        Returns:
            Width of the bounding box, or NaN if invalid.
        """
        if not np.any(regionmask):
            return float("nan")
        # Find columns that contain any True values
        cols = np.any(regionmask, axis=0)
        if not np.any(cols):
            return float("nan")
        # bbox format: (min_row, min_col, max_row, max_col)
        # Width = max_col - min_col + 1
        col_indices = np.where(cols)[0]
        return float(col_indices[-1] - col_indices[0] + 1)

    @staticmethod
    def bbox_height(regionmask, intensity_image=None):
        """Compute bounding box height from region mask.

        Uses numpy operations to find row bounds more efficiently.
        
        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            
        Returns:
            Height of the bounding box, or NaN if invalid.
        """
        if not np.any(regionmask):
            return float("nan")
        # Find rows that contain any True values
        rows = np.any(regionmask, axis=1)
        if not np.any(rows):
            return float("nan")
        # Height = max_row - min_row + 1
        row_indices = np.where(rows)[0]
        return float(row_indices[-1] - row_indices[0] + 1)


    @staticmethod
    def aspect_ratio(regionmask, intensity_image=None):
        """Compute aspect ratio: minor_axis_length / major_axis_length.
        
        Computes aspect ratio from the covariance matrix of region coordinates.
        
        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            
        Returns:
            Aspect ratio (minor/major axis), or NaN if invalid.
        """
        coords = np.where(regionmask)
        if len(coords[0]) == 0:
            return float("nan")

        coords_array = np.array([coords[0], coords[1]], dtype=np.float64)
        centroid = np.mean(coords_array, axis=1)
        coords_centered = coords_array - centroid[:, np.newaxis]

        if coords_centered.shape[1] < 2:
            return float("nan")

        cov = np.cov(coords_centered)
        eigenvalues = np.linalg.eigvals(cov)
        if len(eigenvalues) < 2 or eigenvalues[0] <= 0:
            return float("nan")

        # Sort eigenvalues (largest first)
        eigenvalues = np.sort(eigenvalues)[::-1]
        if eigenvalues[1] <= 0:
            return float("nan")

        return float(np.sqrt(eigenvalues[1] / eigenvalues[0]))


    def _process_slice(self, labels: np.ndarray) -> List["RegionProperties"] | pd.DataFrame:
        """Process a single slice to extract region properties.
        
        Args:
            labels: Labeled array slice.
            
        Returns:
            Either a list of RegionProperties or a pandas DataFrame, depending
            on output_type configuration.
            
        Raises:
            ValueError: If output_type is invalid.
        """
        if self.config.output_type == "list":
            results = regionprops(labels, extra_properties=self.used_extra_properties_funcs())
        elif self.config.output_type == "dataframe":
            results = regionprops_table(labels, properties=self.used_builtin_properties(), extra_properties=self.used_extra_properties_funcs())
        else:
            raise ValueError(f"Invalid output type: {self.config.output_type}. Allowed output types are: list, dataframe")
        logger.info(f"Identified {len(results)} regions, return as {self.config.output_type}")
        return results

    def _reshape_slice_results(self, results: list[Any], slice_indices: list[tuple[int,...]], input_shape: tuple[int,...]) -> List["RegionProperties"] | pd.DataFrame:
        """Reshape slice results according to output configuration.
        
        Args:
            results: List of results from each slice.
            slice_indices: List of index tuples for each slice.
            input_shape: Shape of the input array.
            
        Returns:
            Reshaped results according to output_type.
        """
        return super()._reshape_slice_results(results, slice_indices=slice_indices, input_shape=input_shape)
    
    def run(self, labels: np.ndarray, workers: int = -1, verbose: int = 10) -> List["RegionProperties"] | pd.DataFrame:
        """Run the region analyzer on a labeled array.
        
        Args:
            labels: Labeled array to analyze.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.
            
        Returns:
            Region properties as list or DataFrame, depending on output_type.
        """
        results = super().run(labels, workers=workers, verbose=verbose)
        return results

class RegionAnalyzerConfig(StackProcessorConfig):
    """Configuration for region analysis operations.
    
    Attributes:
        output_type: Output format ("list" for RegionProperties list, "dataframe" for pandas DataFrame).
        properties: List of property names to compute. "label" is always included.
    """
    output_type: Literal["list", "dataframe"] = "list"
    properties: List[str] = RegionAnalyzer.default_properties

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, v: List[str]) -> List[str]:
        """Validate that all properties are allowed and include "label".
        
        Args:
            v: List of property names to validate.
            
        Returns:
            Validated list with "label" added if missing.
            
        Raises:
            ValueError: If any property is not in the allowed list.
        """
        if v is None or len(v) == 0:
            return RegionAnalyzer.default_properties
        elif not set(v).issubset(set(RegionAnalyzer.allowed_properties())):
            raise ValueError(f"One or more invalid properties: {v}. Allowed properties are: {RegionAnalyzer.allowed_properties()}")
        if "label" not in v:
            v = ["label"] + v
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

    def _process_slice(self, img: np.ndarray) -> np.ndarray:
        """Apply range thresholding to a single slice.
        
        Args:
            img: Input image slice.
            
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

    def _process_slice(self, img: np.ndarray) -> np.ndarray:
        """Apply local adaptive thresholding to a single slice.
        
        Args:
            img: Input image slice.
            
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

    def _process_slice(self, img: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding to a single slice.
        
        Args:
            img: Input image slice.
            
        Returns:
            Binary mask from Otsu thresholding.
        """
        th = threshold_otsu(img)
        return img > th


class BinaryProcessor(Configurable[BinaryProcessorConfig]):
    """Processor for binary image operations.
    
    Applies post-processing operations to binary masks, such as watershed
    segmentation or region dilation.
    """
    
    def __init__(self, config: BinaryProcessorConfig):
        """Initialize the binary processor.
        
        Args:
            config: Binary processor configuration.
        """
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
        """Run the binary processor on a mask.
        
        Args:
            mask: Binary mask to process.
            
        Returns:
            Processed binary mask.
        """
        return mask




    def _process_slice(self, mask: np.ndarray) -> np.ndarray:
        """Process a single slice of binary mask.
        
        Args:
            mask: Binary mask slice to process.
            
        Returns:
            Processed binary mask.
        """
        # Currently just returns the mask unchanged
        # Can be extended for watershed, dilation, etc.
        return mask


class SegmenterConfig(Configuration):
    """Configuration for segmentation workflow.
    
    Defines the components and options for the segmentation pipeline.
    
    Attributes:
        thresholder: Optional thresholder for converting images to binary masks.
        binary_processor: Optional processor for binary mask post-processing.
        labeller: Optional labeller for identifying connected components.
        region_analyzer: Optional analyzer for extracting region properties.
        region_filter: Optional filter for removing regions based on criteria.
        do_labels: Whether to compute and return labels.
        do_regions: Whether to compute and return region properties.
    """
    thresholder: Optional[Thresholder] = OtsuThreshold(OtsuThresholdConfig())
    binary_processor: Optional[BinaryProcessor] = None
    labeller: Optional[Labeller] = Labeller(LabellerConfig(connectivity=1, region_filter=None, output_type="list"))
    region_analyzer: Optional[RegionAnalyzer] = None #RegionAnalyzer(RegionAnalyzerConfig(output_type="list", properties=RegionAnalyzer.default_properties))
    region_filter: Optional[RegionFilter] = None
    # label_remover: Optional[LabelRemover] = None
    do_labels: bool = True
    do_regions: bool = False

    @model_validator(mode="after")
    @classmethod
    def check_labeller(cls, data: Any) -> Any:
        """Validate that labeller is provided if region_filter is specified.
        
        Args:
            data: Configuration data to validate.
            
        Returns:
            Validated configuration data.
            
        Raises:
            ValueError: If region_filter is specified without a labeller.
        """
        if "region_filter" in data and "labeller" not in data:
            raise ValueError(
                "A labeller must be provided if a region filter is specified"
            )
        return data


class Segmenter(Workflow):
    """Segmentation workflow that combines thresholding, labeling, and region analysis.
    
    Performs a complete segmentation pipeline: thresholding -> binary processing ->
    labeling -> region analysis -> filtering.
    """
    
    def __init__(self, config: SegmenterConfig = None):
        """Initialize the segmenter.
        
        Args:
            config: Segmenter configuration.
        """
        super().__init__(config)


    def run(
        self,
        img: np.ndarray,
        include_mask: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List["RegionProperties"]]]:
        """Run the segment step.

        Args:
            img (np.ndarray): Input image.
            include_mask (Optional[np.ndarray]): Optional mask to include in the segmentation.
            exclude_mask (Optional[np.ndarray]): Optional mask to exclude from the segmentation.
            do_labels (bool): Whether to return labels.
            do_regions (bool): Whether to return regions.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, List[RegionProps]]]: Binary mask or tuple of binary mask and regions.
        """
        # determine whether to compute regions and labels
        do_regions = self.config.do_regions and (self.config.region_analyzer is not None or self.config.region_filter is not None)
        do_labels = do_regions or (self.config.do_labels and self.config.labeller is not None)
        if do_labels and self.config.labeller is None:
            logger.info("Labeller not provided, using default Labeller with connectivity=1 and region_filter=None")
            labeller = Labeller(LabellerConfig(connectivity=1, region_filter=None, output_type="list"))
        else:
            labeller = self.config.labeller
        if do_regions and self.config.region_analyzer is None:
            logger.info(f"RegionAnalyzer not provided, using default RegionAnalyzer with properties: {RegionAnalyzer.default_properties}")
            properties = self.config.region_analyzer.config.properties if self.config.region_analyzer is not None else RegionAnalyzer.default_properties
            region_analyzer = RegionAnalyzer(RegionAnalyzerConfig(output_type="list", properties=properties))
        else:
            region_analyzer = self.config.region_analyzer

        # process the image
        binary_mask = self.config.thresholder.run(img)
        if include_mask is not None:
            binary_mask = binary_mask & include_mask
        if exclude_mask is not None:
            binary_mask = binary_mask & ~exclude_mask
        if self.config.binary_processor is not None:
            binary_mask = self.config.binary_processor.run(binary_mask)
        if do_labels:
            labels, _ = labeller.run(binary_mask)
            iterator_config = self.config.labeller.config.iterator_config
            # update the labels to ensure they are unique across the substacks
            relabeler = Relabeler(RelabelerConfig(iterator_config=iterator_config))
            labels = relabeler.run(labels)
            if do_regions:
                regions = region_analyzer.run(labels)
                if self.config.region_filter is not None:
                    regions, removed_regions = self.config.region_filter.run(regions)
                    # remove the areas in labels corresponding to the removed regions
                    label_remover = LabelRemover(LabelRemoverConfig(iterator_config=ArrayIteratorConfig(slice_def=()), output_type="stack", squeeze=False))
                    labels = label_remover.run(removed_regions)
                return binary_mask, labels, regions
            else:
                logger.info("No regions to compute, returning binary mask and labels")
                return binary_mask, labels
        else:
            logger.info("No labels or regions to compute, returning binary mask")
            return binary_mask


class IterativeSegmenterConfig(SegmenterConfig):
    """Configuration for iterative segmentation workflow.
    
    Attributes:
        iterations: Number of iterations to perform.
    """
    iterations: PositiveInt = 10


class IterativeSegmenter(Workflow):
    """Iterative segmentation workflow that processes regions incrementally.
    
    Performs segmentation iteratively, dilating and excluding processed regions
    in each iteration to avoid re-segmenting the same areas.
    """

    def __init__(self, config: IterativeSegmenterConfig):
        """Initialize the iterative segmenter.
        
        Args:
            config: Iterative segmenter configuration.
        """
        super().__init__(config)

    def _dilate_regions(self, mask: np.ndarray) -> np.ndarray:
        """Dilate regions in the mask to create exclusion zones.

        Args:
            mask: Binary mask containing regions to dilate.

        Returns:
            Dilated mask for use as exclusion mask in next iteration.
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
        """Run the iterative segmentation workflow.

        Performs segmentation iteratively, excluding processed regions in each
        iteration to avoid re-segmenting the same areas.

        Args:
            img: Input image to segment.

        Returns:
            Tuple of (masks, labels, regions) or single values depending on output mode.
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
        labels, _ = Relabeler.assign_unique_labels(labels)
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
    """Configuration for series segmentation workflow.
    
    Runs multiple segmenters in sequence on the same image.
    
    Attributes:
        segmenters: List of segmenter configurations to run in sequence.
        output: Output format ("stack", "combine", "last", or "list").
    """
    segmenters: List[SegmenterConfig]
    output: str = "stack"

    @field_validator("output")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate that output category is allowed.
        
        Args:
            v: Output category string to validate.
            
        Returns:
            Validated output category.
            
        Raises:
            ValueError: If output category is not in allowed list.
        """
        allowed_categories = ["stack", "combine", "last", "list"]
        if v not in allowed_categories:
            raise ValueError(
                f"'{v}' is not an allowed category. Must be one of: {allowed_categories}"
            )
        return v


class SeriesSegmenter(Workflow):
    """Segmentation workflow that runs multiple segmenters in series.
    
    Applies multiple segmenters sequentially to the same image, allowing
    for cascaded or multi-stage segmentation approaches.
    """
    
    def __init__(self, config: SeriesSegmenterConfig):
        """Initialize the series segmenter.
        
        Args:
            config: Series segmenter configuration.
        """
        super().__init__(config)

    def run(
        self, img: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, List["RegionProperties"]]]:
        """Run the series of segmenters on an image.
        
        Args:
            img: Input image to segment.
            
        Returns:
            Results from the segmenters, format depends on output configuration.
        """
        for segmenter in self.config.segmenters:
            mask, regions, labels = segmenter.run(img)
            return mask, regions, labels

class MicroSAMSegmenterConfig(SegmenterConfig):
    """Configuration for MicroSAM segmenter.
    
    Attributes:
        model: MicroSAM model to use.
    """
    model: str = "microsam-large"

class MicroSAMSegmenter(Workflow):
    """Segmenter that uses MicroSAM to segment images.
    
    Args:
        config: MicroSAM segmenter configuration.
    """
    def __init__(self, config: MicroSAMSegmenterConfig):
        super().__init__(config)

    def run(self, img: np.ndarray) -> np.ndarray:
        """Run the MicroSAM segmenter on an image.
        