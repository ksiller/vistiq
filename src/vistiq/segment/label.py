import logging
import os
import math
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import supervision as sv

from os import PathLike
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from micro_sam.multi_dimensional_segmentation import merge_instance_segmentation_3d

from prefect import task, flow
from pydantic import Field, field_validator, model_validator, PositiveInt
from skimage.measure import label as sk_label

from vistiq.core import (
    Configurable,
    StackProcessor,
    StackProcessorConfig,
    Tiler,
    TilerConfig,
    Untiler,
    UntilerConfig,
    generate_name,
    labels_to_masks,
)
from vistiq.utils import (
    ArrayIterator,
    ArrayIteratorConfig,
    array_content_digest,
    check_device,
    set_fractional_memory,
)
from vistiq.preprocess import (
    Resize, 
    ResizeConfig, 
    UpsampleConfig,
    Upsample,
)

from vistiq.workflow import Workflow, WorkflowConfig

from vistiq.analysis.overlap import OverlapCalculator, MaskOverlapCalculatorConfig
from vistiq.analysis.matrix import group_matrix_indices
from vistiq.constant.matrix import LOWER_ND

from vistiq.segment._debug import debug_mask_labels
from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig
from vistiq.segment.postprocess import (
    BinaryProcessorConfig,
    dilate_regions,
)
from vistiq.segment.select import (
    RegionFilterConfig,
    _filter_config_entry,
)
from vistiq.segment.threshold import (
    OtsuThresholdConfig,
    ThresholderConfig,
)

logger = logging.getLogger(__name__)

def box_iou_batch_3d(
    boxes_true: np.typing.NDArray[np.number],
    boxes_detection: np.typing.NDArray[np.number],
    overlap_metric: Literal["IOU", "IOS"] = "IOU"
) -> np.ndarray[np.float32]:
    """
    Adapted for 3d from https://github.com/roboflow/supervision/blob/develop/src/supervision/detection/utils/iou_and_nms.py
    
    Compute pairwise overlap scores between batches of bounding boxes.

    Supports standard IOU (intersection-over-union) and IOS
    (intersection-over-smaller-area) metrics for all `boxes_true` and
    `boxes_detection` pairs. Returns a matrix of overlap values in range
    `[0, 1]`, matching each box from the first batch to each from the second.

    Args:
        boxes_true: Array of reference boxes in
            shape `(N, 4)` as `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        boxes_detection: Array of detected boxes in
            shape `(M, 4)` as `(x_min, y_min, z_min, x_max, y_max, z_min)`.
        overlap_metric: Overlap type.
            Use `OverlapMetric.IOU` for intersection-over-union,
            `OverlapMetric.IOS` for intersection-over-smaller-area.
            Defaults to `OverlapMetric.IOU`.

    Returns:
        Overlap matrix of shape `(N, M)`, where entry
            `[i, j]` is the overlap between `boxes_true[i]` and
            `boxes_detection[j]`.

    Raises:
        ValueError: If `overlap_metric` is not IOU or IOS.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> import supervision as sv
        >>> boxes_true = np.array([
        ...     [100, 100, 200, 200],
        ...     [300, 300, 400, 400]
        ... ])
        >>> boxes_detection = np.array([
        ...     [150, 150, 250, 250],
        ...     [320, 320, 420, 420]
        ... ])
        >>> sv.box_iou_batch_3d(
        ...     boxes_true, boxes_detection, overlap_metric=sv.OverlapMetric.IOU
        ... )
        array([[0.14285..., 0.        ],
               [0.        , 0.47058...]], dtype=float32)
        >>> sv.box_iou_batch(
        ...     boxes_true, boxes_detection, overlap_metric=sv.OverlapMetric.IOS
        ... )
        array([[0.25, 0.  ],
               [0.  , 0.64]], dtype=float32)

        ```
    """
    #overlap_metric = OverlapMetric.from_value(overlap_metric)
    x_min_true, y_min_true, z_min_true, x_max_true, y_max_true, z_max_true = boxes_true.T
    x_min_det, y_min_det, z_min_det, x_max_det, y_max_det, z_max_det = boxes_detection.T
    count_true, count_det = boxes_true.shape[0], boxes_detection.shape[0]

    if count_true == 0 or count_det == 0:
        return cast(
            np.typing.NDArray[np.float32], np.empty((count_true, count_det), dtype=np.float32)
        )

    x_min_inter = np.empty((count_true, count_det), dtype=np.float32)
    x_max_inter = np.empty_like(x_min_inter)
    y_min_inter = np.empty_like(x_min_inter)
    y_max_inter = np.empty_like(x_min_inter)
    z_min_inter = np.empty_like(x_min_inter)
    z_max_inter = np.empty_like(x_min_inter)

    np.maximum(x_min_true[:, None], x_min_det[None, :], out=x_min_inter)
    np.minimum(x_max_true[:, None], x_max_det[None, :], out=x_max_inter)
    np.maximum(y_min_true[:, None], y_min_det[None, :], out=y_min_inter)
    np.minimum(y_max_true[:, None], y_max_det[None, :], out=y_max_inter)
    np.maximum(z_min_true[:, None], z_min_det[None, :], out=z_min_inter)
    np.minimum(z_max_true[:, None], z_max_det[None, :], out=z_max_inter)

    # we reuse x_max_inter and y_max_inter to store inter_w, inter_h and inter_d
    np.subtract(x_max_inter, x_min_inter, out=x_max_inter)  # inter_w
    np.subtract(y_max_inter, y_min_inter, out=y_max_inter)  # inter_h
    np.subtract(z_max_inter, z_min_inter, out=z_max_inter)  # inter_d
    np.clip(x_max_inter, 0.0, None, out=x_max_inter)
    np.clip(y_max_inter, 0.0, None, out=y_max_inter)
    np.clip(z_max_inter, 0.0, None, out=z_max_inter)

    area_inter = x_max_inter * y_max_inter * z_max_inter # inter_w * inter_h * inter_d

    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true) * (z_max_true - z_min_true)
    area_det = (x_max_det - x_min_det) * (y_max_det - y_min_det)  * (z_max_det - z_min_det)

    if overlap_metric == "IOU":
        area_norm = area_true[:, None] + area_det[None, :] - area_inter
    elif overlap_metric == "IOS":
        area_norm = np.minimum(area_true[:, None], area_det[None, :])
    else:
        raise ValueError(
            f"overlap_metric {overlap_metric} is not supported, "
            "only 'IOU' and 'IOS' are supported"
        )

    out: np.ndarray[np.float32] = np.zeros_like(area_inter, dtype=np.float32)
    np.divide(area_inter, area_norm, out=out, where=area_norm > 0)
    return out


def labels_to_masks(labels):
    label_values = (v for v in np.unique(labels) if v > 0)
    masks = []
    for value in label_values:
        mask = labels == value
        masks.append(mask)
    return np.array(masks)


def group_bboxes(bboxes, divisor=1, threshold=0.5):
    
    def in_groups(item, groups):
        for g in groups:
            if item in g:
                return True
        return False

    bboxes = np.asarray(bboxes)
    if bboxes.size == 0 or bboxes.shape[0] == 0:
        return []
    
    xyxy = np.mod(bboxes, divisor)
    #print (xyxy[:7])
    if len(bboxes[0]) == 4:
        iou_matrix = sv.box_iou_batch(xyxy, xyxy, overlap_metric=sv.OverlapMetric.IOU)
    elif len(bboxes[0]) == 6:
        iou_matrix = box_iou_batch_3d(xyxy, xyxy, overlap_metric="IOU")
    #print (iou_matrix)
    iou_matrix = np.triu(iou_matrix, k=1)
    pairs = np.argwhere(iou_matrix > threshold)
    
    groups = []
    for i, pair in enumerate(pairs):
        p0 = pair[0]
        #print (i, p0, p1, iou_matrix[p0, p1])
        if not in_groups(p0, groups):
            pairs_with_p0 = np.unique(np.array([p for p in pairs if p[0] == p0]).flatten())
            logger.info(f"Creating new group with {pairs_with_p0}")
            groups.append(pairs_with_p0)
    return groups


def label_grouped_mask(mask:np.ndarray, groups:list[np.ndarray], threshold:float=0.5, dtype:str="uint64"):
    labels = []
    for label_value, g in enumerate(groups, 1):
        label_array = (mask[g].mean(axis=0)>threshold) * label_value
        labels.append(label_array)
    labels = np.sum(np.array(labels), axis=0).astype(dtype)
    return labels


def region_bbox_array(results: pd.DataFrame, spatial_ndim: int) -> np.ndarray:
    """Extract bbox rows from a RegionAnalyzer dataframe for :func:`group_bboxes`."""
    if results.empty:
        return np.empty((0, spatial_ndim * 2))

    if spatial_ndim == 3:
        cols = ["bbox-2", "bbox-1", "bbox-0", "bbox-5", "bbox-4", "bbox-3"]
        offset = np.array((0, 0, 0, 1, 1, 1))
    elif spatial_ndim == 2:
        cols = ["bbox-1", "bbox-0", "bbox-3", "bbox-2"]
        offset = np.array((0, 0, 1, 1))
    else:
        raise ValueError(f"Unsupported spatial ndim for bbox grouping: {spatial_ndim}")

    missing = [c for c in cols if c not in results.columns]
    if missing:
        raise KeyError(
            f"Missing bbox columns {missing} in region analysis results; "
            f"available columns: {list(results.columns)}"
        )
    return results[cols].to_numpy() - offset


class SegmenterConfig(StackProcessorConfig):

    pass


class Segmenter(StackProcessor):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

    @task(name="Segmenter.run")
    def run(
        self,
        stack: np.ndarray,
        *args,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[Any, Optional[dict[str, Any]]]:
        labels, metadata = super().run(stack, *args, metadata=metadata, **kwargs)
        # relabeled, label_mappings = Relabeler.assign_unique_labels(
        #    labels, self.config.iterator
        # )
        iterator_config = self.config.iterator_config
        relabeler = Relabeler(RelabelerConfig(iterator_config=iterator_config))
        labels = relabeler.run(labels, metadata=metadata)
        no_labels = len(np.unique(labels)) - 1  # exclude 0 (background)
        if np.max(labels) > no_labels:
            logger.warning(
                f"Segmenter: found {no_labels}; labels do NOT represent consecutive integers"
            )

        return labels, metadata


class MergerConfig(StackProcessorConfig):
    """Configuration for label-stack merging.

    Base configuration for :class:`Merger` and its subclasses. Mergers operate on
    label arrays after slice-wise segmentation, linking instances across the stack
    axis into a single consistent volume.

    Defaults are tuned for processing a full label volume in one pass rather than
    iterating over individual slices.

    Attributes:
        iterator_config: How the input is iterated before merging. Defaults to
            ``slice_def=()``, so the entire array is passed to the merge step.
        output_type: Always ``"stack"``; merged labels are returned as a single
            array.
        squeeze: Whether to remove singleton dimensions from the output. Defaults
            to ``False`` to preserve the input rank.
    """

    iterator_config: ArrayIteratorConfig = ArrayIteratorConfig(
        slice_def=()
    )
    output_type: Literal["stack"] = "stack"
    squeeze: bool = False


class Merger(StackProcessor):
    """Base class for merging per-slice labels into a consistent volume.

    Validates input dimensionality and delegates to :class:`StackProcessor` for
    execution. Subclasses (e.g. :class:`MicroSAMMerger`) implement the actual
    merge logic, typically via a ``merge`` method or ``_process_slice``.

    By default, mergers expect a 3D label array (e.g. ``(Z, Y, X)``) where each
    slice contains independent instance IDs that should be linked across ``Z``.
    Arrays with fewer dimensions than required by the iterator are returned
    unchanged; arrays with unsupported rank are logged and returned unchanged.

    Used as the optional ``merger`` step in :class:`SegmentationFlow`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, config: MergerConfig):
        """Initialize the merger.

        Args:
            config: Merger configuration.
        """
        super().__init__(config)

    def can_merge(self, ndim: int) -> bool:
        """Return whether this merger supports the given array rank.

        Args:
            ndim: Number of dimensions in the label array.

        Returns:
            ``True`` if the array rank matches :attr:`ndims`.
        """
        return ndim == self.ndims

    @property
    def ndims(self) -> int:
        """Number of dimensions the merger expects (default: 3)."""
        return 3

    def merge(self, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Merger is not implemented")

    def _process_slice(self, labels: np.ndarray, *args, metadata: Optional[dict[str, Any]] = None, **kwargs) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        return self.merge(labels)

    @task(name="Merger.run")
    def run(self, labels: Union[np.ndarray, list[np.ndarray]], *args, metadata: Optional[dict[str, Any]] = None, **kwargs) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
        """Merge label arrays when the input rank is supported.

        Accepts a single array or a list of arrays (stacked along axis 0). If the
        input is too low-dimensional for the configured iterator, or its rank does
        not match :attr:`ndims`, the labels are returned without merging.

        Args:
            labels: Label array or list of label arrays to merge.
            metadata: Optional image metadata passed through unchanged.
            *args: Additional positional arguments for :class:`StackProcessor`.
            **kwargs: Additional keyword arguments for :class:`StackProcessor`.

        Returns:
            Tuple of ``(merged_labels, metadata)``.
        """
        logger.info(f"Running Merger with config: {self.config}")
        if isinstance(labels, list):
            labels = np.stack(labels, axis=0)
        slice_ndim = ArrayIterator(labels, self.config.iterator_config).slice_ndim
        if labels.ndim < slice_ndim:
            logger.info(f"{self.name} handles {self.ndims}-dimensional stacks or slices. The slices produced by the iterator have only {slice_ndim} dimensions. Nothing to merge")
            return labels, metadata
        if not self.can_merge(labels.ndim):
            logger.error(f"Merger: cannot merge {labels.ndim}-dimensional stack")
            return labels, metadata
        merged,_  = super().run(labels, *args, metadata=metadata, **kwargs)
        return merged


class MicroSAMMergerConfig(MergerConfig):
    """Configuration for 3D instance merging with micro_sam.

    Extends :class:`MergerConfig` with parameters passed to
    ``micro_sam.multi_dimensional_segmentation.merge_instance_segmentation_3d``.
    Use after per-slice (2D) instance segmentation to link objects across the
    stack axis into a single 3D label volume.

    Attributes:
        beta: Trade-off between overlap and distance when matching instances
            across adjacent slices (higher favors overlap).
        with_background: Whether label 0 is treated as background during merging.
        gap_closing: Whether to close small gaps along the stack axis between
            matched instances.
        min_z_extent: Minimum number of slices an instance must span to be kept.
        verbose: Whether to print progress from the underlying merge routine.
    """

    beta: float = 0.5
    with_background: bool = True
    gap_closing: bool = True
    min_z_extent: int = 10
    verbose: bool = False


class MicroSAMMerger(Merger):
    """Merge 2D instance label stacks into a consistent 3D label volume.

    Wraps ``merge_instance_segmentation_3d`` from micro_sam. Expects a 3D array
    of per-slice instance labels (e.g. shape ``(Z, Y, X)``) produced by a
    segmenter such as :class:`MicroSAMSegmenter`. Instances that correspond to
    the same object in neighboring slices receive the same label ID.

    Typically used as the ``merger`` step in :class:`SegmentationFlow` after
    slice-wise segmentation.
    """

    def __init__(self, config: MicroSAMMergerConfig):
        """Initialize the merger.

        Args:
            config: MicroSAM merger configuration.
        """
        super().__init__(config)

    def merge(self, labels: np.ndarray) -> np.ndarray:
        """Link instances across slices using micro_sam.

        Args:
            labels: 3D label array with per-slice instance IDs.

        Returns:
            Label array with IDs consistent across the stack axis.
        """
        return merge_instance_segmentation_3d(
            labels,
            beta=self.config.beta,
            with_background=self.config.with_background,
            gap_closing=self.config.gap_closing,
            min_z_extent=self.config.min_z_extent
        )

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
            logger.warning(
                f"Relabeler: reshaping output to match input shape {original_shape}"
            )

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
    unique_labels = np.unique(labels)  # sorted=True has been deprecated
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

    @task(name="LabelRemover.run")
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
            return results, None


class LabellerConfig(StackProcessorConfig):
    """Configuration for labeling operations.

    Labels connected components in binary masks.

    Attributes:
        connectivity: Connectivity for labeling (1 for 4-connected, 2 for 8-connected).
        region_filter: Optional region filter to apply after labeling.
        output_type: Output format ("list" for list of arrays).
    """

    connectivity: PositiveInt = 1
    region_filter: Optional[RegionFilterConfig] = None
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
        region_filter_cfg = self.config.region_filter
        if region_filter_cfg is not None and region_filter_cfg.filters:
            extra_funcs = RegionAnalyzer.extra_properties_funcs()
            extra_properties = []
            for f in region_filter_cfg.filters:
                fc = _filter_config_entry(f)
                for attr in fc.attribute_list():
                    if attr in extra_funcs:
                        extra_properties.append(attr)
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

        if region_filter_cfg is not None:
            region_filter = Configurable.create_from_config(region_filter_cfg)
            logger.info(f"Labeller: region_filter.config={region_filter.config}")
            # Store original labels before filtering
            original_labels = labels.copy()
            # Flatten regions if it's a list of lists (from iterator processing)
            if isinstance(regions, list) and len(regions) > 0:
                # Check if first element is a list (nested structure from iterator)
                if isinstance(regions[0], list):
                    regions = [region for sublist in regions for region in sublist]
            regions, removed_labels = region_filter.run(regions)
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
        slice_results, _ = super().run(
            mask, workers=workers, verbose=verbose, metadata=metadata, **kwargs
        )
        labels, regions = slice_results
        # print (f"type(labels)={type(labels)}, type(regions)={type(regions)}")
        # if len(labels) > 1:
        #    iterator = ArrayIterator(labels, self.config.iterator_config)
        #    labels, labels_map = assign_unique_labels(labels, iterator)
        #    print (f"type(labels)={type(labels)}, type(labels_map)={type(labels_map)}")
        #    regions = remap_regions(regions, labels_map)
        return labels, regions


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

    @task(name="IterativeSegmenter._run", task_run_name=generate_name)
    def _run(
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

    @task(name="SeriesSegmenter._run", task_run_name=generate_name)
    def _run(
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
    # segmmentation_mode: Literal["ais", "amg", "apg"] = "ais" # need to upgrade micro_sam to support this
    #predictor: Optional[Any] = None
    #segmenter: Optional[Any] = None
    checkpoint: Optional[str] = None
    embedding_path: Optional[str] = None
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    crop_nms_thresh: float = 0.7
    min_mask_region_area: int = 0
    output_mode: str = "instance_segmentation"
    with_background: bool = True
    device: Optional[str] = None
    device_no: int = 0
    gpu_fraction: float = 1.0
    preferred_backend: Literal["processes", "threads"] = "processes"
    # ndim: Optional[int] = None # image specific, should be set at runtime if needed


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
                    # segmentation_mode=self.config.segmmentation_mode,
                )
            else:
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=self.config.model_type,
                    # segmentation_mode=self.config.segmmentation_mode,
                )
        except TypeError:
            # Older micro_sam versions may not accept `checkpoint=...`
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=self.config.model_type,
                # segmentation_mode=self.config.segmmentation_mode,
            )
            if self.config.checkpoint:
                raise ValueError(
                    "Checkpoint was provided, but this micro_sam version doesn't support "
                    "passing `checkpoint` to get_predictor_and_segmenter."
                )

        self.predictor = predictor
        self.segmenter = segmenter
        # self.config.do_labels = True
        # self.config.do_regions = self.config.region_analyzer is not None

    def _process_slice(
        self, img_slice: np.ndarray, mask_path:Optional[Union[Union[PathLike, str], np.ndarray]] = None,metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        import torch
        if self.config.device is not None:
            device = self.config.device
        else:
            device = check_device()
        with torch.device(device):
            set_fractional_memory(self.config.gpu_fraction, device=device)

            if self.config.embedding_path is None:
                self.config.embedding_path = (
                    os.path.expanduser()
                )  # create_unique_folder(base_path="embeddings")
            arr_hash = array_content_digest(img_slice)
            embedding_path = os.path.join(self.config.embedding_path, arr_hash)
            os.makedirs(embedding_path, exist_ok=True)
            logger.info(
                f"Using {embedding_path} for embeddings. img_slice.shape={img_slice.shape}"
            )

            labels = automatic_instance_segmentation(
                predictor=self.predictor,
                segmenter=self.segmenter,
                input_path=img_slice,
                embedding_path=embedding_path,
                # ndim=self.config.ndim,
                # pred_iou_thresh=self.config.pred_iou_thresh,
                # stability_score_thresh=self.config.stability_score_thresh,
                # box_nms_thresh=self.config.box_nms_thresh,
                # crop_nms_thresh=self.config.crop_nms_thresh,
                # min_mask_region_area=self.config.min_mask_region_area,
                # output_mode=self.config.output_mode,
                # with_background=self.config.with_background,
                # mask_path=mask_path,
                # device=self.config.device,
            )
        
        return labels


class BasicSegmenterConfig(SegmenterConfig):
    """Configuration for a threshold-based segmentation pipeline.

    Defines a simple three-stage pipeline:
    threshold -> optional binary post-processing -> connected-component labeling.

    Attributes:
        thresholder: Converts image intensities into a binary foreground mask.
        binary_processor: Optional morphological cleanup on the binary mask.
        labeller: Connected-component labeler applied to the final mask.
    """

    thresholder: ThresholderConfig = Field(default_factory=OtsuThresholdConfig)
    binary_processor: Optional[BinaryProcessorConfig] = None
    labeller: LabellerConfig = Field(default_factory=LabellerConfig)


class BasicSegmenter(Segmenter):

    def __init__(self, config: BasicSegmenterConfig):
        super().__init__(config)

    def _process_slice(self, slice, metadata: Optional[dict[str, Any]] = None):
        mask = Configurable.create_from_config(self.config.thresholder).run(
            slice, metadata=metadata
        )
        if self.config.binary_processor is not None:
            mask = Configurable.create_from_config(self.config.binary_processor).run(
                mask, metadata=metadata
            )
        return Configurable.create_from_config(self.config.labeller).run(
            mask, metadata=metadata
        )


class SegmentationFlowConfig(WorkflowConfig):
    """Configuration for :class:`SegmentationFlow`.

    Composes a segmenter with optional post-segmentation steps. The segmenter
    (e.g. :class:`MicroSAMSegmenter` or :class:`BasicSegmenter`) is responsible
    for producing label arrays; other fields refine or filter those labels.

    Attributes:
        segmenter: Configuration for the segmenter (e.g.
            :class:`MicroSAMSegmenterConfig`). Instantiated per run in
            :meth:`SegmentationFlow._run`. Its ``iterator_config`` also drives
            relabeling when ``relabeler`` is omitted.
        merger: Optional merger configuration (e.g. :class:`MicroSAMMergerConfig`).
            Skipped when ``None``.
        region_filter: Optional filter configuration. When set, a
            :class:`RegionAnalyzer` and :class:`RegionFilter` are created per
            run with properties required by the filter.
        relabeler: Optional relabeler configuration. When ``None``, a
            :class:`Relabeler` is built from ``segmenter.iterator_config``.
    """

    segmenter: SegmenterConfig = None
    merger: Optional[MergerConfig] = None
    # include_mask_detector: OverlapDetector
    # exclude_mask_detector: OverlapDetector
    region_filter: Optional[RegionFilterConfig] = None
    relabeler: Optional[RelabelerConfig] = None


class SegmentationFlow(Workflow):
    """End-to-end segmentation workflow over an image stack.

    Runs the configured :class:`Segmenter`, optionally merges slice-wise labels
    into a 3D volume, applies include/exclude masks, ensures globally unique
    label IDs via :class:`Relabeler`, and optionally filters regions by measured
    properties.

    Pipeline (when all optional steps are enabled)::

        segmenter -> merger -> mask include/exclude -> relabeler -> region filter

    Example::

        flow = SegmentationFlow(
            SegmentationFlowConfig(
                segmenter=mscfg,
                merger=mmcfg,
                region_filter=rfcfg,
            )
        )
        labels = flow.run(img, metadata=metadata)
    """

    @staticmethod
    def _region_analyzer_for_filter(
        region_filter_config: RegionFilterConfig,
        iterator_config: ArrayIteratorConfig,
    ) -> RegionAnalyzer:
        """Build a :class:`RegionAnalyzer` with properties required by the filter config."""
        filter_attributes: list[str] = []
        for f in region_filter_config.filters or []:
            filter_attributes.extend(_filter_config_entry(f).attribute_list())
        properties = list(RegionAnalyzer.default_properties)
        properties += [attr for attr in filter_attributes if attr not in properties]
        logger.info(
            "Creating RegionAnalyzer for region filter with properties: %s",
            properties,
        )
        return RegionAnalyzer(
            RegionAnalyzerConfig(
                iterator_config=iterator_config,
                output_type="list",
                properties=properties,
            )
        )

    @task(name="SegmentationFlow._run", task_run_name=generate_name)
    def _run(
        self,
        img: np.ndarray,
        *args,
        metadata: Optional[dict[str, Any]] = None,
        include_masks: Optional[list[np.ndarray]] = None,
        exclude_masks: Optional[list[np.ndarray]] = None,
        **kwargs,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Run the segmentation flow on an input image/stack.

        Args:
            img: Input image or stack.
            metadata: Optional metadata describing the input.
            include_masks: Optional masks multiplied into labels to keep only
                selected regions.
            exclude_masks: Optional masks multiplied out of labels to remove
                selected regions.
            *args: Additional positional arguments forwarded to processors.
            **kwargs: Additional keyword arguments forwarded to processors.

        Returns:
            Segmentation label array.
        """
        logging.info(f"SegmentationFlow _run: channel_names={metadata.get('channel_names', None)}, channel_axis={metadata.get('channel_axis', None)}, axes={metadata.get('axes', None)}")

        if self.config.segmenter is None:
            raise ValueError("SegmentationFlowConfig.segmenter is required")

        segmenter = Configurable.create_from_config(self.config.segmenter)
        raw_labels, _ = segmenter.run(img, *args, metadata=metadata, **kwargs)

        if self.config.merger is not None:
            merger = Configurable.create_from_config(self.config.merger)
            labels = merger.run(raw_labels, *args, metadata=metadata, **kwargs)
        else:
            labels = raw_labels

        # mask labels
        if include_masks is not None:
            for mask in include_masks:
                labels = labels * mask
        if exclude_masks is not None:
            for mask in exclude_masks:
                labels = labels * ~mask

        # update the labels to ensure they are unique across the substacks
        l_before = np.unique(labels)
        if self.config.relabeler is not None:
            relabeler = Configurable.create_from_config(self.config.relabeler)
        else:
            relabeler = Relabeler(
                RelabelerConfig(
                    iterator_config=self.config.segmenter.iterator_config,
                )
            )
        labels = relabeler.run(labels, *args, metadata=metadata, **kwargs)
        l_after = np.unique(labels)
        logger.debug(f"Relabeler: l_before == l_after:{l_before == l_after}")

        # filter labels based on region properties
        if self.config.region_filter is not None:
            iterator_config = self.config.segmenter.iterator_config
            region_analyzer = self._region_analyzer_for_filter(
                self.config.region_filter,
                iterator_config,
            )
            regions = region_analyzer.run(labels, *args, metadata=metadata, **kwargs)
            if isinstance(regions, list) and len(regions) > 0:
                # Check if first element is a list (nested structure from iterator)
                if isinstance(regions[0], list):
                    regions = [region for sublist in regions for region in sublist]
            region_filter = Configurable.create_from_config(self.config.region_filter)
            regions, removed_labels = region_filter.run(regions)
            # remove the areas in labels corresponding to the removed regions
            label_remover = LabelRemover(
                LabelRemoverConfig(
                    iterator_config=ArrayIteratorConfig(slice_def=()),
                    remap=True,
                    output_type="stack",
                    squeeze=False,
                )
            )
            labels, _ = label_remover.run(labels, removed_labels)

        return labels


class TiledSegmentationFlowConfig(SegmentationFlowConfig):

    tile_factor: Tuple[int, ...] = (3, 3)
    resize_factor: Tuple[float, ...] = (0.25, 0.25)
    pad_width: Union[int, Tuple[Tuple[int, int]], dict[int, Tuple[int, int]]] = {-2:(0,5), -1:(0,5)}
    iou_threshold: float = 0.5
    consensus_threshold: float = 0.75


class TiledSegmentationFlow(SegmentationFlow):

    def __init__(self, config: TiledSegmentationFlowConfig):
        super().__init__(config)


    @task(name="TiledSegmentationFlow._run", task_run_name=generate_name)
    def _run(
        self,
        stack: np.ndarray,
        *args,
        metadata: Optional[dict[str, Any]] = None,
        config: Optional[TiledSegmentationFlowConfig] = TiledSegmentationFlowConfig(),
        **kwargs,
    ):
        """Run the tiled segmentation flow.

        Args:
            stack: The input stack.
            *args: Additional arguments.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """
        # get original width and height
        orig_width = stack.shape[-1]
        orig_height = stack.shape[-2]

        # resize
        width = stack.shape[-1]*self.config.resize_factor[-1]
        height = stack.shape[-2]*self.config.resize_factor[-2]
        rcfg = ResizeConfig(width=width, height=height)
        r_stack, r_metadata = Resize(rcfg).run(stack, metadata=metadata, **kwargs)

        # tile with padding
        tcfg = TilerConfig(factor=self.config.tile_factor, alt_flip=False, pad_width=self.config.pad_width)
        t_stack,t_metadata = Tiler(tcfg).run(r_stack, *args,metadata=r_metadata, **kwargs)

        # run segmentation on tiled stack
        t_labels = super()._run(t_stack, *args, metadata=t_metadata, **kwargs)

        # convert labels to stack of masks: the stack will have shape (len(t_groups), *t_labels.shape)
        t_masks = labels_to_masks(t_labels)

        # untile: the tiles will be stacked and inserted as new axis 0 in untiled array. The untiled array will have shape (len(t_groups), *t_masks.shape)
        ucfg = UntilerConfig(
            factor = self.config.tile_factor,
            iterator_config = ArrayIteratorConfig(slice_def=())
        )
        untiled,_ = Untiler(ucfg).run(t_masks)
        t_proj =  np.sum(untiled>0, axis=0)>0

        olcfg = MaskOverlapCalculatorConfig(
            annotate=False,
            triangle=LOWER_ND,
        )
        ol_calc = OverlapCalculator(olcfg)
        ol_result = ol_calc.run(t_proj, t_proj)
        iou = ol_calc.format(ol_result, "iou")
        groups = group_matrix_indices(iou, threshold=self.config.iou_threshold)
        logger.info(f"TiledSegmentationFlow: groups={groups}")
        # label grouped mask
        stacks = np.array([(t_proj[g].mean(axis=0)>self.config.consensus_threshold) * (i+1) for i,g in enumerate(groups)])
        labels = np.sum(stacks, axis=0)

        # remove padding
        cropped_height = labels.shape[-2]-self.config.pad_width[-2][1]
        cropped_width = labels.shape[-1]-self.config.pad_width[-1][1]
        cropped_labels = labels[..., 0:cropped_height, 0:cropped_width]

        #ecfg = ResizeConfig(width=orig_width, height=orig_height, anti_aliasing=False, order=0, preserve_range=True, normalize=False, dtype=np.uint16)
        #resized_labels, _ = Resize(ecfg).run(cropped_labels, metadata=r_metadata,  **kwargs)
        ecfg = UpsampleConfig(width=orig_width, height=orig_height, sigma=3.0)
        resized_labels, _ = Upsample(ecfg).run(cropped_labels, metadata=r_metadata, **kwargs)

        if resized_labels.shape != stack.shape:
            logging.error(f"resized_labels.shape: {resized_labels.shape} != stack.shape: {stack.shape}")
            raise ValueError(f"resized_labels.shape: {resized_labels.shape} != stack.shape: {stack.shape}")

        if self.config.region_filter is not None:
            region_analyzer = self._region_analyzer_for_filter(
                self.config.region_filter,
                self.config.segmenter.iterator_config,
            )
            r_results = region_analyzer.run(
                resized_labels, metadata=metadata, **kwargs
            )
            region_filter = Configurable.create_from_config(self.config.region_filter)
            _, removed_labels = region_filter.run(r_results)
            label_remover = LabelRemover(
                LabelRemoverConfig(
                    iterator_config=ArrayIteratorConfig(slice_def=()),
                    remap=True,
                    output_type="stack",
                    squeeze=False,
                )
            )
            resized_labels, _ = label_remover.run(resized_labels, removed_labels)
        return resized_labels
