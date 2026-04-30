import logging
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from prefect import task, flow
from pydantic import field_validator, model_validator, PositiveInt
from skimage.measure import label as sk_label

from vistiq.core import (
    Configuration,
    StackProcessor,
    StackProcessorConfig,
)
from vistiq.utils import ArrayIterator, ArrayIteratorConfig, create_unique_folder
from vistiq.workflow import Workflow

from vistiq.segment._debug import debug_mask_labels
from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig
from vistiq.segment.postprocess import (
    BinaryProcessor,
    BinaryProcessorConfig,
    dilate_regions,
)
from vistiq.segment.select import (
    RegionFilter,
    RegionFilterConfig,
)
from vistiq.segment.threshold import (
    OtsuThreshold,
    OtsuThresholdConfig,
    Thresholder,
)

logger = logging.getLogger(__name__)


class RelabelerConfig(StackProcessorConfig):
    """Configuration for relabeling operations.

    Relabels arrays to ensure unique labels across multiple labeled arrays.

    Attributes:
        output_type: Output format ("stack" for stacked array).
        squeeze: Whether to squeeze output dimensions.
    """

    output_type: Literal["stack"] = "stack"  # force output type to stack
    squeeze: bool = True  # don't squeeze the output


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

    def _process_slice(
        self, labels: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Process a single slice by assigning unique labels.

        Args:
            labels: Labeled array slice.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Relabeled array with unique labels.
        """
        # For a single slice, no relabeling needed (already unique)
        return labels

    @task(name="Relabeler.run")
    def run(
        self,
        labels: np.ndarray | List[np.ndarray],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Run the relabeler to assign unique labels.

        Args:
            labels: Labeled array or list of labeled arrays to relabel.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Relabeled array with unique labels. Same shape as input (stacked if input was a list).
        """
        logger.info("DEBUG: entered Relabeler.run")
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


def remap_labels(
    labels: np.ndarray,
    mapping: Optional[Union[dict[int, int], list[tuple[int, int]]]] = None,
    exclude: Optional[list[int]] = [0],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Remap labels to consecutive positive integers, keeping 0 as background.

    After removing labels, there may be gaps in the label sequence (e.g., labels 1, 3, 5).
    This function remaps them to consecutive integers (1, 2, 3) while preserving 0 as background.

    Args:
        labels: Label array with potentially non-consecutive label IDs.
        mapping: Optional mapping of labels to new labels.
        exclude: Optional list of labels to exclude from remapping.

    Returns:
        Tuple of (remapped_label_array, mapping_list) where mapping_list is a list of
        (old_label, new_label) tuples.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels, sorted=True)
    do_exclude = exclude is not None and len(exclude) > 0
    logger.debug(f"Unique labels: {unique_labels}, exclude={exclude}")
    if len(unique_labels) == 0:
        # No labels to remap, return as-is with empty mapping
        return labels.astype(np.int32), []
    if mapping is None:
        map_from = np.array(unique_labels, dtype=np.int32)
        map_to = np.arange(0, len(unique_labels), dtype=np.int32)

    elif isinstance(mapping, dict):
        mapping = dict(mapping)
        # sort mapping by keys
        mapping = dict(sorted(mapping.items()))
        map_from = np.array(list(mapping.keys()), dtype=np.int32)
        map_to = np.array(list(mapping.values()), dtype=np.int32)
    elif (
        isinstance(mapping, list)
        and isinstance(mapping[0], tuple)
        and len(mapping[0]) == 2
    ):
        mapping = np.array(sorted(mapping), dtype=np.int32)
        map_from = mapping[:, 0]
        map_to = mapping[:, 1]
    else:
        raise ValueError(f"Invalid mapping type: {type(mapping)}")

    # Handle exclude
    if do_exclude:
        map_to = map_to[~np.isin(map_to, exclude)]
        map_from = map_from[~np.isin(map_from, exclude)]

    # Create mapping list for return
    mapping_list = list(zip(map_from, map_to))
    # for pair in mapping_list:
    #    logger.debug(f"Mapping {pair[0]} -> {pair[1]}")

    # vals, inv = np.unique(labels, return_inverse=True)
    # indices_to_replace = np.searchsorted(vals, map_from)
    # vals[indices_to_replace] = map_to
    # results = vals[inv].reshape(labels.shape)

    # Create mapping array: index is old label, value is new label
    # Initialize with identity mapping (each index maps to itself)
    # This ensures labels not in map_from stay unchanged
    max_label = int(np.max(map_from)) if len(map_from) > 0 else 0
    max_input_label = int(np.max(labels)) if len(labels) > 0 else 0
    mapping_size = max(max_label, max_input_label) + 1

    mapping_temp = np.arange(mapping_size, dtype=np.int32)
    mapping_temp[map_from] = map_to
    mapping_temp[0] = (
        0  # Ensure label 0 maps to 0 (background) unless explicitly mapped
    )
    results = mapping_temp[labels]
    return results, mapping_list


def remap_dataframe_labels(
    df: pd.DataFrame,
    mapping: Optional[Union[dict[int, int], list[tuple[int, int]]]] = None,
    exclude: Optional[list[int]] = [0],
    key: Optional[str] = None,
) -> pd.DataFrame:
    """Remap labels in a DataFrame using the remap_labels function.

    This function uses remap_labels to remap labels in a DataFrame column or index.
    It's a convenience wrapper that extracts labels, applies remap_labels, and updates
    the DataFrame.

    Args:
        df: Input DataFrame containing labels to remap.
        mapping: Optional mapping of labels to new labels (passed to remap_labels).
        exclude: Optional list of labels to exclude from remapping (passed to remap_labels).
        key: Column name to update. If None, remap the DataFrame's index.

    Returns:
        DataFrame with remapped labels in the specified column or index.
    """
    df = df.copy()  # Work on a copy to avoid modifying the original

    # Determine where to get labels from and where to write them
    index_name = df.index.name
    if (
        key is None
        or key.lower() == "index"
        or (index_name is not None and index_name.lower() == key.lower())
    ):
        # Use index
        labels = df.index.values
        target_is_index = True
    else:
        # Use specified column
        if key not in df.columns:
            raise ValueError(
                f"Column '{key}' not found in DataFrame. Available columns: {list(df.columns)}"
            )
        labels = df[key].values
        target_is_index = False

    # Use remap_labels to remap the labels
    remapped_labels, _ = remap_labels(labels, mapping=mapping, exclude=exclude)

    # Update DataFrame
    if target_is_index:
        # Remap index - preserve index name if it exists
        index_name = df.index.name
        df.index = remapped_labels
        if index_name is not None:
            df.index.name = index_name
    else:
        # Remap column
        df[key] = remapped_labels

    logger.debug(f"Relabeled DataFrame using remap_labels")
    return df


def remap_regionproperties(
    regions: List["RegionProperties"],
    mapping: Optional[Union[dict[int, int], list[tuple[int, int]]]] = None,
    exclude: Optional[list[int]] = [0],
    key: Optional[str] = None,
) -> List["RegionProperties"]:
    """Remap labels in a list of RegionProperties to consecutive positive integers, keeping 0 as background.

    This function uses remap_labels to remap labels in RegionProperties objects.
    It updates the `label` attribute of each RegionProperties object.

    Args:
        regions: List of RegionProperties objects containing labels to remap.
        mapping: Optional mapping of labels to new labels (passed to remap_labels).
        exclude: Optional list of labels to exclude from remapping (passed to remap_labels).
        key: Optional parameter for API consistency (not used for RegionProperties,
                    which always update the `label` attribute).

    Returns:
        List of RegionProperties objects with remapped labels in their `label` attribute.
    """
    if len(regions) == 0:
        return regions

    # Extract labels from RegionProperties objects
    labels = np.array([region.label for region in regions], dtype=np.int32)

    # Use remap_labels to remap the labels
    remapped_labels, _ = remap_labels(labels, mapping=mapping, exclude=exclude)

    # Update RegionProperties objects' label attribute
    for region, new_label in zip(regions, remapped_labels):
        region.label = int(new_label)

    logger.debug(f"Remapped {len(regions)} RegionProperties labels using remap_labels")
    return regions


def remap_regions(
    regions: Union[List["RegionProperties"], pd.DataFrame],
    mapping: Union[dict[int, int], list[tuple[int, int]]],
    key: Optional[str] = None,
) -> Union[List["RegionProperties"], pd.DataFrame]:
    """Remap labels in RegionProperties or DataFrame using a provided mapping.

    This function is a convenience wrapper that handles both RegionProperties lists
    and DataFrames. It uses remap_regionproperties for RegionProperties and
    remap_dataframe_labels for DataFrames.

    Args:
        regions: Either a list of RegionProperties objects or a pandas DataFrame.
        mapping: Mapping of old labels to new labels. Can be a dict or list of tuples.
        key: Column name to update in DataFrame. If None, remap the DataFrame's index.
                   Ignored for RegionProperties.

    Returns:
        Remapped regions (same type as input).
    """
    if isinstance(regions, pd.DataFrame):
        # Handle DataFrame
        return remap_dataframe_labels(regions, mapping=mapping, key=key)
    elif isinstance(regions, list):
        # Handle list of RegionProperties
        # Note: remap_regionproperties doesn't use exclude when mapping is provided
        return remap_regionproperties(regions, mapping=mapping, exclude=[0], key=key)
    else:
        raise TypeError(
            f"regions must be either List[RegionProperties] or pd.DataFrame, got {type(regions)}"
        )


class LabelRemoverConfig(StackProcessorConfig):
    """Configuration for label removal operations.

    This configuration defines how labels should be removed from label arrays.
    """

    iterator_config: ArrayIteratorConfig = ArrayIteratorConfig(
        slice_def=()
    )  # no slicing, remove all labels
    remap: bool = (
        False  # remap labels to consecutive positive integers (0 is background)
    )
    output_type: Literal["stack"] = "stack"  # force output type to stack
    squeeze: bool = False  # don't squeeze the output


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
        label_ids: Union[List["RegionProperties"], pd.DataFrame, List[int], np.ndarray],
    ) -> np.ndarray:
        """
            Extract label IDs from input formats.

            IMPORTANT:
            Label IDs must match segmentation labels, not DataFrame row indices.

            For DataFrames:
            - Use 'label' column if present
            - Otherwise use index only if named 'label'
            - Do not assume row indices are labels

        Returns:
        np.ndarray of int32 label IDs.
        """
        logger.debug(f"type(label_ids)={type(label_ids)}")

        if isinstance(label_ids, pd.DataFrame):
            if "label" in label_ids.columns:
                return label_ids["label"].astype(np.int32).to_numpy()
            elif label_ids.index.name == "label":
                return label_ids.index.to_numpy(dtype=np.int32)
            else:
                raise ValueError(
                    "LabelRemover received a DataFrame without a 'label' column "
                    "or index named 'label', so labels cannot be extracted safely."
                )

        elif isinstance(label_ids, list) and len(label_ids) > 0:
            if hasattr(label_ids[0], "label"):
                return np.array([region.label for region in label_ids], dtype=np.int32)
            else:
                return np.array(label_ids, dtype=np.int32)

        elif isinstance(label_ids, np.ndarray):
            return label_ids.astype(np.int32)

        else:
            return np.array([], dtype=np.int32)

    def _process_slice(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Process a single slice by removing specified labels.

        Args:
            labels: Label array slice.
            label_ids: Array of label IDs to remove.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Processed label array with specified labels set to 0.
        """
        # Ensure labels is a proper array (not scalar) and convert to writable int32
        labels = np.asarray(labels)
        if labels.ndim == 0:
            # Handle scalar case (shouldn't happen, but be safe)
            return labels.astype(np.int32)

        # Ensure result is a writable array with compatible dtype for assignment
        # Convert to int32 to avoid issues with uint32 assignment
        # Use np.array() with copy=True to ensure we have a proper writable array
        result = np.array(labels, dtype=np.int32, copy=True)
        # Ensure result is writable (set write flag explicitly)
        result.setflags(write=True)

        if len(label_ids) > 0:
            # Ensure label_ids is a proper array
            label_ids = np.asarray(label_ids)
            # Create mask for all labels to remove
            mask = np.isin(result, label_ids)
            # Set masked pixels to background (0)
            result[mask] = 0
            logger.debug(
                f"{len(label_ids)} labels removed, {len(np.unique(labels))} unique labels before removal, {len(np.unique(result))} labels remaining, {len(np.unique(result))} unique labels after removal"
            )
        return result

    @task(name="RegionRemover.run")
    def run(
        self,
        labels: np.ndarray,
        region_properties: Union[
            List["RegionProperties"], pd.DataFrame, List[int], np.ndarray
        ],
        workers: int = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Union[np.ndarray, tuple[np.ndarray, list[tuple[int, int]]]]:
        """Remove specified labels from the label array.

        Args:
            labels: Input label array.
            region_properties: Region properties to remove. Can be:
                - List of RegionProperties (extracts .label attribute)
                - pandas DataFrame with a 'label' column or index
                - List of ints
                - numpy array of ints
            workers: Number of parallel workers (-1 for all cores).
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.
            verbose: Verbosity level for parallel processing.

        Returns:
            If remap=False: Processed label array with specified labels set to background (0).
            If remap=True: Tuple of (processed label array, mapping list) where mapping is
                list of (old_label, new_label) tuples.
        """
        print("DEBUG: entered LabelRemover.run")
        print("DEBUG: region_properties type =", type(region_properties))
        # Extract label IDs from various input formats
        label_ids_array = self._extract_label_ids(region_properties)
        logger.debug(f"label_ids_array={label_ids_array}")

        # Use parent's run method with label_ids as additional argument
        results, _ = super().run(
            labels, label_ids_array, workers=workers, verbose=verbose
        )
        if self.config.remap:
            results, mapping = remap_labels(results)
            # logger.debug(f"Remapping: {[pair for pair in mapping]}")
            # logger.debug(f"Results after removal and remapping: {results}")
            return results, mapping
        else:
            logger.debug(f"Results after removal: {results}")
            return results


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
        # extra_props_map = {func.__name__: func for func in Labeller.extra_properties}

        # Determine which extra_properties are needed based on region_filter
        # if self.config.region_filter is not None and self.config.region_filter.config.filters is not None:
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
        # else:
        #    # No filter or no filters specified
        #    self.use_extra_properties = None

    # def _process_stack(self, mask: np.ndarray) -> np.ndarray:
    #    """Process the mask."""
    #    labels = sk_label(mask, connectivity=self.config.connectivity)
    #    regions = regionprops(labels)
    #    return labels, regions

    def _process_slice(
        self, mask: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> tuple[np.ndarray, List["RegionProperties"]]:
        """Process a single slice by labeling connected components.

        Labels connected components in the binary mask and optionally filters
        regions based on configured criteria.

        Args:
            mask: Binary mask to label.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Tuple of (labels, regions):
            - labels: Labeled array with unique integer labels for each region.
            - regions: List of region properties for each labeled region.
        """
        labels = sk_label(mask, connectivity=self.config.connectivity)
        if (
            self.config.region_filter is not None
            and self.config.region_filter.config.filters is not None
        ):
            extra_properties = [
                filter.config.attribute
                for filter in self.config.region_filter.config.filters
                if filter.config.attribute is not None
                and filter.config.attribute
                in RegionAnalyzer.extra_properties_funcs().keys()
            ]
        else:
            extra_properties = []
        logger.info(f"extra_properties={extra_properties}")
        iterator_config = ArrayIteratorConfig(
            slice_def=self.config.iterator_config.slice_def
        )
        ra = RegionAnalyzer(
            RegionAnalyzerConfig(
                iterator_config=iterator_config, properties=extra_properties
            )
        )
        regions = ra.run(labels)
        logger.info(f"Labeller: len(regions)={len(regions)}")

        if self.config.region_filter is not None:
            logger.info(
                f"Labeller: self.config.region_filter.config={self.config.region_filter.config}"
            )
            # Store original labels before filtering
            original_labels = labels.copy()
            # Flatten regions if it's a list of lists (from iterator processing)
            if isinstance(regions, list) and len(regions) > 0:
                # Check if first element is a list (nested structure from iterator)
                if isinstance(regions[0], list):
                    regions = [region for sublist in regions for region in sublist]
            regions, removed_labels = self.config.region_filter.run(regions)
            labels = np.zeros_like(labels)
            for region in regions:
                # Use original_labels to create mask, not the zeroed labels
                region_mask = original_labels == region.label
                labels[region_mask] = region.label
        return labels, regions

    @task(name="Labeller.run")
    def run(
        self,
        mask: np.ndarray,
        workers: int = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, List["RegionProperties"]]:
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
        # print (f"type(labels)={type(labels)}, type(regions)={type(regions)}")
        # if len(labels) > 1:
        #    iterator = ArrayIterator(labels, self.config.iterator_config)
        #    labels, labels_map = assign_unique_labels(labels, iterator)
        #    print (f"type(labels)={type(labels)}, type(labels_map)={type(labels_map)}")
        #    regions = remap_regions(regions, labels_map)
        return labels, regions


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
    labeller: Optional[Labeller] = Labeller(
        LabellerConfig(connectivity=1, region_filter=None, output_type="list")
    )
    region_analyzer: Optional[RegionAnalyzer] = (
        None  # RegionAnalyzer(RegionAnalyzerConfig(output_type="list", properties=RegionAnalyzer.default_properties))
    )
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
        self.config.do_regions = self.config.do_regions or (
            self.config.region_analyzer is not None
            or self.config.region_filter is not None
        )
        self.config.do_labels = self.config.do_regions or (
            self.config.do_labels and self.config.labeller is not None
        )
        if self.config.do_labels and self.config.labeller is None:
            logger.info(
                "Labeller not provided, using default Labeller with connectivity=1 and region_filter=None"
            )
            self.config.labeller = Labeller(
                LabellerConfig(connectivity=1, region_filter=None, output_type="list")
            )
        if self.config.do_regions and self.config.region_analyzer is None:
            iterator_config = self.config.labeller.config.iterator_config
            # Set properties based on region_filter if present, otherwise use defaults
            if (
                self.config.region_filter is not None
                and self.config.region_filter.config.filters is not None
            ):
                # Extract filter attributes to ensure RegionAnalyzer computes them
                filter_attributes = [
                    filter.config.attribute
                    for filter in self.config.region_filter.config.filters
                    if filter.config.attribute is not None
                ]
                # Combine default properties with filter attributes (avoid duplicates)
                properties = list(RegionAnalyzer.default_properties)
                for attr in filter_attributes:
                    if attr not in properties:
                        properties.append(attr)
            else:
                properties = RegionAnalyzer.default_properties
            logger.info(
                f"RegionAnalyzer not provided, using default RegionAnalyzer with properties: {properties}"
            )
            self.config.region_analyzer = RegionAnalyzer(
                RegionAnalyzerConfig(
                    iterator_config=iterator_config,
                    output_type="list",
                    properties=properties,
                )
            )
        logger.info(f"Segmenter config: {self.config}")

    @task(name="Segmenter.run")
    def run(
        self,
        img: np.ndarray,
        include_mask: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, List["RegionProperties"]],
    ]:
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

        # process the image
        binary_mask = self.config.thresholder.run(img)
        if include_mask is not None:
            binary_mask = binary_mask & include_mask
        if exclude_mask is not None:
            binary_mask = binary_mask & ~exclude_mask
        if self.config.binary_processor is not None:
            binary_mask = self.config.binary_processor.run(binary_mask)
        if self.config.do_labels:
            labels, _ = self.config.labeller.run(binary_mask)
            iterator_config = self.config.labeller.config.iterator_config
            # update the labels to ensure they are unique across the substacks
            relabeler = Relabeler(RelabelerConfig(iterator_config=iterator_config))
            labels = relabeler.run(labels)
            if self.config.do_regions:
                regions = self.config.region_analyzer.run(labels)
                # Flatten regions if it's a list of lists (from iterator processing)
                if isinstance(regions, list) and len(regions) > 0:
                    # Check if first element is a list (nested structure from iterator)
                    if isinstance(regions[0], list):
                        regions = [region for sublist in regions for region in sublist]
                if self.config.region_filter is not None:
                    regions, removed_labels = self.config.region_filter.run(regions)
                    # remove the areas in labels corresponding to the removed regions
                    label_remover = LabelRemover(
                        LabelRemoverConfig(
                            iterator_config=ArrayIteratorConfig(slice_def=()),
                            output_type="stack",
                            squeeze=False,
                        )
                    )
                    labels = label_remover.run(labels, removed_labels)
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

    @task(name="IterativeSegmenter.run")
    def run(
        self, img: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
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

        all_regions = []
        all_masks = []
        all_labels = []

        for s in self.steps:
            mask, step_labels, step_regions = s.run(img, include_mask, exclude_mask)

            dilated_mask = self._dilate_regions(mask)

            if exclude_mask is not None:
                exclude_mask = dilated_mask | exclude_mask
            else:
                exclude_mask = dilated_mask

            all_regions.append(step_regions)
            all_masks.append(dilated_mask)
            all_labels.append(step_labels)

        if self.output == "last":
            masks = all_masks[-1]
            regions = all_regions[-1]
            labels = all_labels[-1]
            return masks, labels, regions

        labels, _ = Relabeler.assign_unique_labels(all_labels)

        if self.output == "stack":
            masks = np.stack(all_masks, axis=0)
            regions = np.stack(all_regions, axis=0)
            labels = np.stack(labels, axis=0)
        elif self.output == "combine":
            masks = np.sum(all_masks, axis=0)
            regions = np.sum(all_regions, axis=0)
            labels = np.sum(labels, axis=0)
        else:
            masks = all_masks
            regions = all_regions
            labels = labels

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

    @task(name="SeriesSegmenter.run")
    def run(
        self, img: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, List["RegionProperties"]]]:
        """Run the series of segmenters on an image.

        Args:
            img: Input image to segment.

        Returns:
            Results from the segmenters, format depends on output configuration.
        """
        all_masks = []
        all_labels = []
        all_regions = []

        for segmenter in self.config.segmenters:
            mask, labels, regions = segmenter.run(img, metadata=metadata, **kwargs)
            all_masks.append(mask)
            all_labels.append(labels)
            all_regions.append(regions)

        if self.config.output == "last":
            return all_masks[-1], all_labels[-1], all_regions[-1]

        all_labels, _ = Relabeler.assign_unique_labels(all_labels)

        if self.config.output == "stack":
            masks = np.stack(all_masks, axis=0)
            labels = np.stack(all_labels, axis=0)
            regions = all_regions
        elif self.config.output == "combine":
            masks = np.sum(all_masks, axis=0)
            labels = np.sum(all_labels, axis=0)
            regions = all_regions
        elif self.config.output == "list":
            masks = all_masks
            labels = all_labels
            regions = all_regions
        else:
            raise ValueError(
                f"Invalid output mode: {self.config.output}. "
                f"Expected one of: stack, combine, last, list"
            )

        return masks, labels, regions


class MicroSAMSegmenterConfig(SegmenterConfig):
    """Configuration for MicroSAM segmenter.

    Attributes:
        model: MicroSAM model to use.
    """

    model_type: str = "vit_l_lm"
    thresholder: Optional[Thresholder] = None
    labeller: Optional[Labeller] = None
    predictor: Optional[Any] = None
    segmenter: Optional[Any] = None
    checkpoint: Optional[str] = None
    embedding_path: Optional[str] = None
    device: Optional[str] = None


class MicroSAMSegmenter(Segmenter):
    """Segmenter that uses MicroSAM to segment images.

    Args:
        config: MicroSAM segmenter configuration.
    """

    def __init__(self, config: MicroSAMSegmenterConfig):
        super().__init__(config)
        try:
            if self.config.checkpoint:
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=self.config.model_type,
                    checkpoint=self.config.checkpoint,
                )
            else:
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=self.config.model_type
                )
        except TypeError:
            # Older micro_sam versions may not accept `checkpoint=...`
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=self.config.model_type
            )
            if self.config.checkpoint:
                raise ValueError(
                    "Checkpoint was provided, but this micro_sam version doesn't support "
                    "passing `checkpoint` to get_predictor_and_segmenter."
                )

        if self.config.predictor is None:
            self.config.predictor = predictor
        if self.config.segmenter is None:
            self.config.segmenter = segmenter
        self.config.do_labels = True
        # self.config.do_regions = self.config.region_analyzer is not None

    @flow(name="MicroSAMSegmenter.run")
    def run(
        self, img: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Run the MicroSAM segmenter on an image.

        Args:
            img: Input image to segment.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Regions: List of regions.
        """
        if self.config.embedding_path is None:
            self.config.embedding_path = create_unique_folder(base_path="embeddings")
        labels = automatic_instance_segmentation(
            predictor=self.config.predictor,
            segmenter=self.config.segmenter,
            input_path=img,
            embedding_path=self.config.embedding_path,
        )
        binary_mask = np.zeros_like(labels).astype(np.bool)
        binary_mask[labels > 0] = True
        if self.config.do_regions:
            regions = self.config.region_analyzer.run(labels, metadata=metadata)
            # Flatten regions if it's a list of lists (from iterator processing)
            if isinstance(regions, list) and len(regions) > 0:
                # Check if first element is a list (nested structure from iterator)
                if isinstance(regions[0], list):
                    regions = [region for sublist in regions for region in sublist]
            if self.config.region_filter is not None:
                regions, removed_labels = self.config.region_filter.run(regions)
                # remove the areas in labels corresponding to the removed regions
                logger.info(
                    f"Starting label removal with {len(removed_labels)} removed regions"
                )
                label_remover = LabelRemover(
                    LabelRemoverConfig(
                        iterator_config=ArrayIteratorConfig(slice_def=()),
                        remap=True,
                        output_type="stack",
                        squeeze=False,
                    )
                )
                labels, newlabels_map = label_remover.run(labels, removed_labels)
                regions = remap_regions(regions, newlabels_map, key="label")
            return binary_mask, labels, regions
        else:
            logger.info("No regions to compute, returning binary mask and labels")
            return binary_mask, labels, None
