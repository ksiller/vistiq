import numpy as np
import pandas as pd
from typing import Literal, Dict, List, Tuple, Optional
from pydantic import Field, field_validator
from skimage.measure import regionprops
from vistiq.core import Configuration, Configurable, StackProcessorConfig, StackProcessor, ChainProcessorConfig, ChainProcessor
from vistiq.workflow import Workflow
from vistiq.utils import ArrayIterator, ArrayIteratorConfig, create_unique_folder

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
                  For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice).
            shape: Shape of the full image. For 2D: (height, width).
                   For 3D: (depth, height, width) or (height, width) if using 2D projection.
            
        Returns:
            Binary mask with ones in the bounding box region.
        """
        # Handle both 2D (4 values) and 3D (6 values) bounding boxes
        if len(bbox) == 6:
            # 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice)
            min_row, min_col, min_slice, max_row, max_col, max_slice = bbox
            # Check if shape is 3D or 2D
            if len(shape) == 3:
                # Full 3D mask
                mask = np.zeros(shape, dtype=bool)
                mask[min_slice:max_slice, min_row:max_row, min_col:max_col] = True
            else:
                # 2D projection - use only spatial dimensions
                mask = np.zeros(shape, dtype=bool)
                mask[min_row:max_row, min_col:max_col] = True
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

    def _process_slice(self, labels1: np.ndarray, labels2: np.ndarray, stack_names:Tuple[str, str] = ["stack_1", "stack_2"]) -> List[Dict]:
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
        
        # Get region properties for bounding boxes if needed
        bboxes1 = None
        bboxes2 = None
        img_shape = None
        if self.config.mode == "bounding_box":
            bboxes1 = {prop.label: prop.bbox for prop in regionprops(labels1)}
            bboxes2 = {prop.label: prop.bbox for prop in regionprops(labels2)}
            # Get image shape for creating masks
            # For 3D images, use full shape; for 2D, use spatial dimensions
            if labels1.ndim == 3:
                img_shape = labels1.shape
            else:
                img_shape = labels1.shape[:2]
        
        # Compute overlap for each region pair
        for label1 in unique_labels1:
            # Create binary mask for region1
            if self.config.mode == "outline":
                mask1 = (labels1 == label1)
            
            for label2 in unique_labels2:
                # Compute overlap based on mode
                if self.config.mode == "outline":
                    # Pixel-level overlap
                    mask2 = (labels2 == label2)
                    if self.config.method == "iou":
                        overlap_score = self._iou(mask1, mask2)
                    else:  # dice
                        overlap_score = self._dice(mask1, mask2)
                else:  # bounding_box mode
                    # Bounding box overlap - create masks and use _iou/_dice
                    if label1 in bboxes1 and label2 in bboxes2:
                        if self.config.method == "iou":
                            overlap_score = self._iou_box(bboxes1[label1], bboxes2[label2], img_shape)
                        else:  # dice
                            overlap_score = self._dice_box(bboxes1[label1], bboxes2[label2], img_shape)
                    else:
                        overlap_score = 0.0
                
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

    def run(self, labels1: np.ndarray, labels2: np.ndarray, stack_names: Optional[Tuple[str, str]] = None) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
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

        results = super().run(labels1, labels2, stack_names)
        consolidated_dfs = self._consolidate_results(results, stack_names)
        return results, consolidated_dfs