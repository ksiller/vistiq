from typing import Any, Optional, Tuple

import logging
import numpy as np
import math
from pydantic import NonNegativeFloat
from prefect import task
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import binary_dilation, disk
from vistiq.core import Configurable, StackProcessorConfig

logger = logging.getLogger(__name__)


def crop_to_nonzero(arr, pad=0):
    coords = np.argwhere(arr != 0)
    # Find min/max for each dimension and pad
    min_coords = np.maximum(coords.min(axis=0) - pad, np.zeros(arr.ndim, dtype="int"))
    max_coords = np.minimum(coords.max(axis=0) + 1 + pad, np.array(arr.shape))
    zipped_coords = zip(min_coords, max_coords)
    # Create slices for each dimension
    slices = tuple(slice(start, end) for start, end in zipped_coords)
    return arr[slices], slices


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


class BinaryProcessorConfig(StackProcessorConfig):
    """Configuration for binary image processing operations.

    Attributes:
        watershed: Whether to apply watershed segmentation.
        dilation_target_area: Target area for region dilation (0.0 = no dilation).
    """

    watershed: bool = False
    dilation_target_area: NonNegativeFloat = 0.0


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

    @task(name="BinaryProcessor.run")
    def run(
        self, mask: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Run the binary processor on a mask.

        Args:
            mask: Binary mask to process.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Processed binary mask.
        """
        return mask

    def _process_slice(
        self, mask: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Process a single slice of binary mask.

        Args:
            mask: Binary mask slice to process.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Processed binary mask.
        """
        # Currently just returns the mask unchanged
        # Can be extended for watershed, dilation, etc.
        return mask


class WatershedConfig(BinaryProcessorConfig):
    footprint: Tuple[int, ...] = None
    clear_border: bool = False
    h: float = 1.0
    min_distance: float = 1
    compactness: int = 1


class Watershed(Configurable[WatershedConfig]):

    def __init__(self, config: WatershedConfig):
        """Initialize the watershed processor.

        Args:
            config: Watershed configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: "WatershedConfig") -> "Watershed":
        """Create a RegionAnalyzer instance from a configuration.

        Args:
            config: Watershed configuration.

        Returns:
            A new Watershed processor instance.
        """
        return cls(config)

    def _watershed(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
        h: float,
        min_distance: int,
        compactness: int,
    ):
        """Run the watershed on a binary mask or labeled image.

        Args:
            image: Binary mask to process.

        Returns:
            Processed binary mask.
        """
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from skimage.morphology import h_maxima

        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(image)

        # supress local maxima <h to prevent oversegmentation
        local_max_mask = h_maxima(distance, h=h)
        coords = peak_local_max(
            distance,
            footprint=footprint,
            labels=local_max_mask,
            min_distance=min_distance,
        )

        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image, compactness=compactness)

        return labels.astype("uint32")

    @task(name="Watershed.run")
    def run(self, mask: np.ndarray, metadata: Optional[dict[str, Any]] = None):
        """Run the relabeler to assign unique labels.

        Args:
            mask: Labeled array or list of labeled arrays to relabel.
            metadata: Optional metadata to pass to the processor.

        Returns:
            Relabeled array with unique labels. Same shape as input.
        """
        if (
            metadata is not None
            and "scale" in metadata
            and len(metadata["scale"]) >= mask.ndim
        ):
            min_dist_pixel = int(
                np.min(self.config.min_distance / np.array(metadata["scale"]))
            )
            logger.info(
                f"metadata['scale']={metadata['scale']}, min_distance={self.config.min_distance}, min_dist_pixel={min_dist_pixel}"
            )
        else:
            min_dist_pixel = math.ceil(self.config.min_distance)
            logger.info(f"image scale not defined, min_dist_pixel={min_dist_pixel}")
        clear_border = self.config.clear_border
        footprint = self.config.footprint
        if footprint is None:
            footprint = np.ones((3,) * mask.ndim)
        else:
            footprint = np.ones(footprint[-mask.ndim :])
        logger.info(f"footprint={footprint.shape}")
        # last = int(np.max(mask))
        # logger.info(f"last={last}")
        new_labels = np.zeros(mask.shape, dtype="uint32")
        last_label = 0
        values = [i for i in np.unique(mask) if i != 0]
        for i in values:
            single_inst = np.where(mask == i, 1, 0)
            # for efficiency, crop array to labeled image block
            cropped_inst, slices = crop_to_nonzero(single_inst)
            labels = self._watershed(
                cropped_inst,
                footprint=footprint,
                h=self.config.h,
                min_distance=min_dist_pixel,
                compactness=self.config.compactness,
            )
            logger.info(
                f"Segmenting mask with intensity value={i}; segment values={np.unique(labels)}, last_label={last_label}"
            )
            # assuming anisotropic voxels (x,y,x): (5,1,1)
            # labels = anisotropic_watershed(cropped_inst, sampling=(5.0,1.0,1.0), first_label=first_label)
            labels[labels != 0] += last_label
            new_labels[slices] += labels
            last_label = np.max(new_labels)

        if self.config.clear_border and (len(np.unique(new_labels)) > 2):
            # if np.unique is <=2, it means there's only one region and we should not clear it even if it touches the image border
            # The `clear_border` function takes the labeled image and removes objects that touch the image edge.
            new_labels = segmentation.clear_border(new_labels)

        return new_labels
