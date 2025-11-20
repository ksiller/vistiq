import numpy as np
import pandas as pd
import logging
from typing import Literal, Dict, List, Tuple, Optional, Any
from pydantic import Field, field_validator
from skimage.measure import regionprops
from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor, ChainProcessorConfig, ChainProcessor
from vistiq.workflow import Workflow
from vistiq.utils import ArrayIterator, ArrayIteratorConfig, create_unique_folder

logger = logging.getLogger(__name__)

class CoincidenceDetectorConfig(StackProcessorConfig):
    """Configuration for coincidence detection workflow.
    
    Attributes:
        output_type: Output type ("list" or "stack").
        output: Output fields ("score" or "above_threshold").
        method: Overlap method to use ("iou" or "dice").
        mode: Overlap mode ("box" or "strict").
        threshold: Threshold for the overlap score (must be between 0.0 and 1.0).
    """
    output_type: Literal["list"] = Field(default="list", description="Output type")
    output: List[Literal["score", "above_threshold"]] = Field(default=["score", "above_threshold"], description="Output fields")
    method: Literal["iou", "dice"] = Field(default="iou", description="Overlap method")
    mode: Literal["bounding_box", "outline"] = Field(default="outline", description="Overlap mode")
    threshold: float = Field(default=0.5, description="Threshold for the overlap score")
    
    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate that threshold is between 0.0 and 1.0.
        
        Args:
            v: Threshold value to validate.
            
        Returns:
            Validated threshold value.
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {v}")
        return v


class CoincidenceDetector(StackProcessor):
    """Detector that computes the coincidence/overlap between two labeled imagestacks.

    Args:
        config: Configuration for the coincidence detector.
        
    """
    
    def __init__(self, config: CoincidenceDetectorConfig):
        super().__init__(config)

    def _iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute the Intersection Over Union (IoU) between two binary masks. It's equivalent to the Jaccard index.

        Formula:
            IoU = intersection / union
            intersection = sum(mask1 & mask2)
            union = sum(mask1 | mask2)

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            
        Returns:
            IoU score between 0.0 and 1.0.
        """
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _dice(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute the Dice coefficient between two binary masks. It's equivalent to the F1 score.

        Formula:
            Dice = 2 * intersection / (sum(mask1) + sum(mask2))
            intersection = sum(mask1 & mask2)
            sum_masks = sum(mask1) + sum(mask2)

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            
        Returns:
            Dice coefficient between 0.0 and 1.0.
        """
        intersection = np.sum(mask1 & mask2)
        sum_masks = np.sum(mask1) + np.sum(mask2)
        if sum_masks == 0:
            return 0.0
        return float(2 * intersection / sum_masks)
    
    def _bbox_to_mask(self, bbox: Tuple, shape: Tuple) -> np.ndarray:
        """Create a binary mask from a bounding box.
        
        Args:
            bbox: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                  For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X) in Z-Y-X order.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (Z, Y, X) or (height, width) if using 2D projection.
            
        Returns:
            Binary mask with ones in the bounding box region.
        """
        # Handle both 2D (4 values) and 3D (6 values) bounding boxes
        if len(bbox) == 6:
            # 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            min_Z, min_Y, min_X, max_Z, max_Y, max_X = bbox
            # Check if shape is 3D or 2D
            if len(shape) == 3:
                # Full 3D mask: shape is (Z, Y, X)
                mask = np.zeros(shape, dtype=bool)
                mask[min_Z:max_Z, min_Y:max_Y, min_X:max_X] = True
            else:
                # 2D projection - use only Y and X dimensions
                mask = np.zeros(shape, dtype=bool)
                mask[min_Y:max_Y, min_X:max_X] = True
        else:
            # 2D: (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = bbox
            # Create mask with shape of the full image
            mask = np.zeros(shape, dtype=bool)
            # Set the bounding box region to True
            mask[min_row:max_row, min_col:max_col] = True
        return mask
    
    def _iou_box(self, bbox1: Tuple, bbox2: Tuple, shape: Tuple) -> float:
        """Compute IoU between two bounding boxes by creating masks and using _iou.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice).
            bbox2: Bounding box. Same format as bbox1.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (depth, height, width) or (height, width) if using 2D projection.
            
        Returns:
            IoU score between 0.0 and 1.0.
        """
        mask1 = self._bbox_to_mask(bbox1, shape)
        mask2 = self._bbox_to_mask(bbox2, shape)
        return self._iou(mask1, mask2)
    
    def _dice_box(self, bbox1: Tuple, bbox2: Tuple, shape: Tuple) -> float:
        """Compute Dice coefficient between two bounding boxes by creating masks and using _dice.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice).
            bbox2: Bounding box. Same format as bbox1.
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (depth, height, width) or (height, width) if using 2D projection.
            
        Returns:
            Dice coefficient between 0.0 and 1.0.
        """
        mask1 = self._bbox_to_mask(bbox1, shape)
        mask2 = self._bbox_to_mask(bbox2, shape)
        return self._dice(mask1, mask2)
    
    def _extract_region(self, labels: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Extract a sub-region from labels based on bounding box.
        
        Args:
            labels: Labeled image array.
            bbox: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                  For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X).
                  Note: regionprops returns bboxes in Z-Y-X order for 3D arrays.
                  For array shape (Z, Y, X), the bbox directly maps to array[Z, Y, X].
            
        Returns:
            Extracted sub-region from labels.
        """
        if len(bbox) == 6:
            # 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            # regionprops bbox format for 3D is in Z-Y-X order
            # For array shape (Z, Y, X), directly map: bbox[Z, Y, X] -> array[Z, Y, X]
            min_z, min_y, min_x, max_z, max_y, max_x = bbox
            if labels.ndim == 3:
                return labels[min_z:max_z, min_y:max_y, min_x:max_x]
            else:
                # 2D array, bbox is (min_row, min_col, max_row, max_col)
                min_y, min_x, max_y, max_x = bbox[:4]
                return labels[min_y:max_y, min_x:max_x]
        else:
            # 2D: (min_row, min_col, max_row, max_col)
            min_y, min_x, max_y, max_x = bbox
            return labels[min_y:max_y, min_x:max_x]
    
    def _bbox_to_relative(self, bbox: Tuple, union_bbox: Tuple) -> Tuple:
        """Convert a bounding box to coordinates relative to a union bounding box.
        
        Args:
            bbox: Original bounding box. For 2D: (min_row, min_col, max_row, max_col).
                  For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X) in Z-Y-X order.
            union_bbox: Union bounding box in the same format.
            
        Returns:
            Relative bounding box with coordinates relative to the union bbox origin.
        """
        if len(bbox) == 6:
            # 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            min_Z, min_Y, min_X, max_Z, max_Y, max_X = bbox
            u_min_Z, u_min_Y, u_min_X, u_max_Z, u_max_Y, u_max_X = union_bbox
            return (
                min_Z - u_min_Z,
                min_Y - u_min_Y,
                min_X - u_min_X,
                max_Z - u_min_Z,
                max_Y - u_min_Y,
                max_X - u_min_X
            )
        else:
            # 2D
            min_row, min_col, max_row, max_col = bbox
            u_min_row, u_min_col, u_max_row, u_max_col = union_bbox
            return (
                min_row - u_min_row,
                min_col - u_min_col,
                max_row - u_min_row,
                max_col - u_min_col
            )
    
    def _bboxes_overlap(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if two bounding boxes overlap.
        
        Args:
            bbox1: Bounding box. For 2D: (min_row, min_col, max_row, max_col).
                   For 3D: (min_Z, min_Y, min_X, max_Z, max_Y, max_X) in Z-Y-X order.
            bbox2: Bounding box. Same format as bbox1.
            
        Returns:
            True if bounding boxes overlap, False otherwise.
        """
        if len(bbox1) == 6 and len(bbox2) == 6:
            # 3D bounding boxes: (min_Z, min_Y, min_X, max_Z, max_Y, max_X)
            min_Z1, min_Y1, min_X1, max_Z1, max_Y1, max_X1 = bbox1
            min_Z2, min_Y2, min_X2, max_Z2, max_Y2, max_X2 = bbox2
            
            # Check overlap in all three dimensions (Z, Y, X)
            overlap_Z = not (max_Z1 <= min_Z2 or max_Z2 <= min_Z1)
            overlap_Y = not (max_Y1 <= min_Y2 or max_Y2 <= min_Y1)
            overlap_X = not (max_X1 <= min_X2 or max_X2 <= min_X1)
            
            return overlap_Z and overlap_Y and overlap_X
        else:
            # 2D bounding boxes: (min_row, min_col, max_row, max_col)
            min_row1, min_col1, max_row1, max_col1 = bbox1
            min_row2, min_col2, max_row2, max_col2 = bbox2
            
            # Check overlap in both dimensions
            overlap_row = not (max_row1 <= min_row2 or max_row2 <= min_row1)
            overlap_col = not (max_col1 <= min_col2 or max_col2 <= min_col1)
            
            return overlap_row and overlap_col
    
    def _bbox_union(self, bboxes: List[Tuple]) -> Optional[Tuple]:
        """Compute the bounding box that encompasses a list of bounding boxes.
        
        Works for n-dimensional bounding boxes. Supports two input formats:
        
        1. Flat format: (min_0, min_1, ..., min_{n-1}, max_0, max_1, ..., max_{n-1})
           - 2D: (min_row, min_col, max_row, max_col) = 4 values
           - 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice) = 6 values
        
        2. Tuple-of-tuples format: ((min_0, min_1, ..., min_{n-1}), (max_0, max_1, ..., max_{n-1}))
           - 2D: ((min_row, min_col), (max_row, max_col))
           - 3D: ((min_row, min_col, min_slice), (max_row, max_col, max_slice))
        
        Args:
            bboxes: List of bounding boxes. All bounding boxes must have the same
                   dimensionality and format. Can be:
                   - List of flat tuples: [(min_0, ..., max_0, ...), ...]
                   - List of tuple-of-tuples: [((min_0, ...), (max_0, ...)), ...]
            
        Returns:
            Union bounding box in the same format as input, or None if list is empty.
            The union bounding box contains all input bounding boxes.
            
        Raises:
            ValueError: If bounding boxes have different dimensionalities or formats.
        """
        if not bboxes:
            return None
        
        try:
            bboxes = np.array(bboxes)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"All bounding boxes must have the same dimensionality and format. "
                f"Error converting to numpy array: {e}"
            )
        
        ndim = bboxes.ndim
        
        # Check if input is in tuple-of-tuples format (ndim == 3) or flat format (ndim == 2)
        if ndim == 3:
            # Tuple-of-tuples format: ((min_coords...), (max_coords...))
            # bboxes shape: (N, 2, M) where N is number of bboxes, M is number of dimensions
            union_mins = np.min(bboxes[:, 0, :], axis=0)
            union_maxs = np.max(bboxes[:, 1, :], axis=0)
            return (tuple(union_mins), tuple(union_maxs))
        else:
            # Flat format: (min_0, min_1, ..., min_{M-1}, max_0, max_1, ..., max_{M-1})
            # bboxes shape: (N, 2*M) where N is number of bboxes, M is number of dimensions
            # Need to split each bbox into min and max parts
            num_dims = bboxes.shape[1] // 2
            if bboxes.shape[1] % 2 != 0:
                raise ValueError(
                    f"Flat format bounding boxes must have an even number of elements. "
                    f"Got shape {bboxes.shape[1]}"
                )
            bbox_array = bboxes.reshape((len(bboxes), 2, num_dims))
            union_mins = np.min(bbox_array[:, 0, :], axis=0)
            union_maxs = np.max(bbox_array[:, 1, :], axis=0)
            return tuple(np.concatenate([union_mins, union_maxs]))

    def _process_slice(self, labels1: np.ndarray, labels2: np.ndarray, stack_names:Tuple[str, str] = ["stack_1", "stack_2"], metadata: Optional[dict[str, Any]] = None, **kwargs) -> List[Dict]:
        """Process a single slice of the coincidence detector.
        
        Computes overlap between each region in labels1 with all regions in labels2.
        
        Args:
            labels1: First labeled image (each pixel has an integer label, 0 is background).
            labels2: Second labeled image (each pixel has an integer label, 0 is background).

        Returns:
            List of dictionaries, each containing:
            - region1_label: Label of region in labels1
            - region2_label: Label of region in labels2
            - overlap_score: Computed overlap score (IoU or Dice)
            - above_threshold: Boolean indicating if score >= threshold
        """
        # Get unique labels (excluding background 0)
        unique_labels1 = np.unique(labels1)
        unique_labels1 = unique_labels1[unique_labels1 > 0]
        
        unique_labels2 = np.unique(labels2)
        unique_labels2 = unique_labels2[unique_labels2 > 0]
        
        # If no regions in either image, return empty list
        if len(unique_labels1) == 0 or len(unique_labels2) == 0:
            return []
        
        results = []
        
        # Always get bounding boxes for all regions (regardless of mode)
        # This allows us to check overlap before expensive computations
        props1 = regionprops(labels1)
        props2 = regionprops(labels2)
        bboxes1 = {prop.label: prop.bbox for prop in props1}
        bboxes2 = {prop.label: prop.bbox for prop in props2}
        
        # Debug: Log bbox info for the labels we're comparing
        logger.debug(f"labels1.shape: {labels1.shape}, labels2.shape: {labels2.shape}")
        logger.debug(f"Unique labels1: {unique_labels1[:10]}... (showing first 10)")
        logger.debug(f"Unique labels2: {unique_labels2[:10]}... (showing first 10)")
        
        # Debug: Check a sample bbox to understand the format
        if unique_labels1.size > 0 and unique_labels1[0] in bboxes1:
            sample_bbox = bboxes1[unique_labels1[0]]
            logger.debug(f"Sample bbox for label {unique_labels1[0]}: {sample_bbox}, len: {len(sample_bbox)}")
        
        # Compute overlap for each region pair
        for label1 in unique_labels1:
            # Get bounding box for region1
            bbox1 = bboxes1.get(label1)
            if bbox1 is None:
                continue
            
            for label2 in unique_labels2:
                # Get bounding box for region2
                bbox2 = bboxes2.get(label2)
                if bbox2 is None:
                    continue
                
                # Check if bounding boxes overlap first (early exit optimization)
                if not self._bboxes_overlap(bbox1, bbox2):
                    # No overlap, score is 0 - skip expensive computation
                    logger.debug(f"No overlap between labels {label1} and {label2}, bbox1: {bbox1}, bbox2: {bbox2}")
                    overlap_score = 0.0
                else:
                    # Bounding boxes overlap, compute union bbox for optimized mask extraction
                    union_bbox = self._bbox_union([bbox1, bbox2])
                    if union_bbox is None:
                        logger.debug(f"No union bbox found for labels {label1} and {label2}, bbox1: {bbox1}, bbox2: {bbox2}")
                        overlap_score = 0.0
                    else:
                        # Extract sub-regions from both labels using union bbox
                        logger.debug(f"Union bbox found for labels {label1} and {label2}, union_bbox: {union_bbox}, labels1.shape: {labels1.shape}, labels2.shape: {labels2.shape}")
                        sub_labels1 = self._extract_region(labels1, union_bbox)
                        sub_labels2 = self._extract_region(labels2, union_bbox)
                        logger.debug(f"Extracted sub_labels1.shape: {sub_labels1.shape}, sub_labels2.shape: {sub_labels2.shape}")
                        logger.debug(f"Unique labels in sub_labels1: {np.unique(sub_labels1)}, looking for label {label1}")
                        logger.debug(f"Unique labels in sub_labels2: {np.unique(sub_labels2)}, looking for label {label2}")
                        
                        # Create masks on the smaller sub-regions
                        if self.config.mode == "outline":
                            # Pixel-level overlap on sub-regions
                            mask1 = (sub_labels1 == label1)
                            mask2 = (sub_labels2 == label2)
                            
                            # Debug: Check if masks are empty
                            logger.debug(f"mask1 sum: {np.sum(mask1)}, mask2 sum: {np.sum(mask2)}")
                            if not np.any(mask1):
                                logger.debug(f"Warning: mask1 for label {label1} is empty after extraction. Union bbox: {union_bbox}, bbox1: {bbox1}")
                            if not np.any(mask2):
                                logger.debug(f"Warning: mask2 for label {label2} is empty after extraction. Union bbox: {union_bbox}, bbox2: {bbox2}")
                            
                            if self.config.method == "iou":
                                overlap_score = self._iou(mask1, mask2)
                            else:  # dice
                                overlap_score = self._dice(mask1, mask2)
                        else:  # bounding_box mode
                            # Convert bboxes to relative coordinates within the union bbox
                            rel_bbox1 = self._bbox_to_relative(bbox1, union_bbox)
                            rel_bbox2 = self._bbox_to_relative(bbox2, union_bbox)
                            sub_shape = sub_labels1.shape
                            if self.config.method == "iou":
                                overlap_score = self._iou_box(rel_bbox1, rel_bbox2, sub_shape)
                            else:  # dice
                                overlap_score = self._dice_box(rel_bbox1, rel_bbox2, sub_shape)
                        logger.debug(f"Union bbox found for labels {label1} and {label2}, union_bbox: {union_bbox}, overlap_score: {overlap_score}")
                
                results.append({
                    stack_names[0]: int(label1),
                    stack_names[1]: int(label2),
                    "score": overlap_score,
                    "above_threshold": overlap_score >= self.config.threshold
                })
        
        return results

    def _consolidate_results(self, results: List[Dict], stack_names: Tuple[str, str] = ["stack_1", "stack_2"]) -> Dict[str, pd.DataFrame]:
        """Consolidate the results of the coincidence detector.
        
        Groups results by stack name and label, determining if any overlap for each
        label is above threshold and computing the maximum score.
        
        Args:
            results: List of result dictionaries from _process_slice, each containing:
                - stack_name1: Label ID from first stack
                - stack_name2: Label ID from second stack
                - score: Overlap score
                - above_threshold: Boolean indicating if score >= threshold
            stack_names: List of two stack names. If None, will be inferred from results.
            
        Returns:
            Dictionary keyed by stack name, where each value is a DataFrame with columns:
            - label: Label ID
            - above_threshold: Boolean indicating if ANY overlap for this label is above threshold
            - max_score: Maximum overlap score for this label across all overlaps
        """
        if not results:
            return {
                stack_names[0]: pd.DataFrame(columns=["label", "above_threshold", "max_score"]),
                stack_names[1]: pd.DataFrame(columns=["label", "above_threshold", "max_score"])
            }
               
        # Initialize result structure: {stack_name: {label_id: {'scores': [...], 'bools': [...]}}}
        temp_consolidated: Dict[str, Dict[int, Dict[str, List]]] = {
            stack_names[0]: {},
            stack_names[1]: {}
        }
        
        # Group results by stack and label, collecting both scores and booleans
        for result in results:
            label1 = result[stack_names[0]]
            label2 = result[stack_names[1]]
            score = result["score"]
            above_threshold = result["above_threshold"]
            
            # Add to stack 1 -> stack 2 mapping
            if label1 not in temp_consolidated[stack_names[0]]:
                temp_consolidated[stack_names[0]][label1] = {"scores": [], "bools": []}
            temp_consolidated[stack_names[0]][label1]["scores"].append(score)
            temp_consolidated[stack_names[0]][label1]["bools"].append(above_threshold)
            
            # Add to stack 2 -> stack 1 mapping
            if label2 not in temp_consolidated[stack_names[1]]:
                temp_consolidated[stack_names[1]][label2] = {"scores": [], "bools": []}
            temp_consolidated[stack_names[1]][label2]["scores"].append(score)
            temp_consolidated[stack_names[1]][label2]["bools"].append(above_threshold)
        
        # Build separate DataFrames for each stack
        dataframes = {}
        for stack_name, comp_stack_name in zip(stack_names, stack_names[::-1]):
            rows = []
            for label, data in temp_consolidated[stack_name].items():
                rows.append({
                    "label": label,
                    f"{comp_stack_name} +": any(data["bools"]),
                    f"{self.config.method} {comp_stack_name} +": max(data["scores"]) if data["scores"] else 0.0
                })
            dataframes[stack_name] = pd.DataFrame(rows).set_index("label")
        
        return dataframes

    def run(self, labels1: np.ndarray, labels2: np.ndarray, stack_names: Optional[Tuple[str, str]] = None, metadata: Optional[dict[str, Any]] = None, **kwargs) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run the coincidence detector on a labeled image.
        
        Args:
            labels1: First labeled image.
            labels2: Second labeled image.
            stack_names: List of two stack names. If None, will be inferred from results.
            
        Returns:
            Tuple of (results, consolidated_dfs):
            - results: List of result dictionaries from each slice
            - consolidated_dfs: Dictionary keyed by stack name, where each value is a DataFrame
              with columns: label, above_threshold, max_score
        """
        if stack_names is None or len(stack_names) != 2:
            stack_names = ("stack_1", "stack_2",)        

        results = super().run(labels1, labels2, stack_names, metadata=metadata, **kwargs)
        consolidated_dfs = self._consolidate_results(results, stack_names)
        return results, consolidated_dfs
