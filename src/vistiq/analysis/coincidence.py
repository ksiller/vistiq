import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from prefect import task

from vistiq.core import StackProcessor, StackProcessorConfig
from vistiq.utils import ArrayIteratorConfig, _normalize_stack_names
from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig, bbox_array_from_dataframe
from vistiq.analysis.overlap import (
    IoUMetricsCalculatorConfig,
    MetricsCalculatorConfig,
)

logger = logging.getLogger(__name__)


def _label_ids_and_boxes(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segmentation label ids and axis-aligned boxes ``(N, 2 * ndim)``.

    Uses integer ``label`` values from :class:`RegionAnalyzer` (not ``object_id``,
    which is a hex string). Boxes are in array axis order (min/max pairs per axis).
    """
    labels = np.asarray(labels)

    ra = RegionAnalyzer(
        RegionAnalyzerConfig(
            properties=["label", "bbox"],
            output_type="dataframe",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    table = ra.run(labels)

    if len(table) == 0:
        return np.array([], dtype=np.int64), np.empty(
            (0, 2 * labels.ndim), dtype=np.float32
        )
    if "label" in table.columns:
        label_ids = table["label"].astype(np.int64, copy=False)
    elif table.index.name == "label":
        label_ids = table.index.astype(np.int64, copy=False)
    else:
        raise ValueError(
            "Label DataFrame must have a 'label' column or index named 'label'"
        )
    if labels.ndim not in (2, 3):
        raise ValueError(
            f"Label arrays must be 2D or 3D for region bounding boxes; got {labels.ndim}D"
        )
    boxes = bbox_array_from_dataframe(table)
    if boxes is None:
        raise ValueError(
            "RegionAnalyzer table has no bbox columns; "
            f"available: {list(table.columns)}"
        )
    return label_ids, boxes.astype(np.float32, copy=False)


def _box_overlap_matrix(
    boxes_a: np.typing.NDArray[np.number],
    boxes_b: np.typing.NDArray[np.number],
    metric: MetricsCalculatorConfig,
    *,
    device: Union[str, int, Any, None] = None,
) -> np.ndarray[np.float32]:
    """Pairwise box overlap via :class:`~vistiq.analysis.overlap.OverlapCalculator`."""
    from vistiq.analysis.overlap import BoxOverlapCalculatorConfig, OverlapCalculator

    calc = OverlapCalculator(
        BoxOverlapCalculatorConfig(
            metrics_calculators=[metric],
        )
    )
    result = calc.run(boxes_a, boxes_b, device=device)
    from vistiq.matrix.ops import MatrixFormatter, MatrixFormatterConfig

    return np.asarray(
        MatrixFormatter(
            MatrixFormatterConfig(output_type="np.ndarray", annotate=False)
        ).run(result.metric()),
        dtype=np.float32,
    )


def _label_overlap_matrix(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    metric: MetricsCalculatorConfig,
    *,
    intersection_mode: Literal["linear", "sparse", "auto"] = "auto",
    device: Union[str, int, Any, None] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-volume overlap matrix and aligned label ids via OverlapCalculator."""
    from vistiq.analysis.overlap import (
        LabelBuilder,
        LabelBuilderConfig,
        LabelIntersectionCalculatorConfig,
        LabelOverlapCalculatorConfig,
        OverlapCalculator,
    )

    builder = LabelBuilder(LabelBuilderConfig())
    build_a = builder.run(labels_a)
    build_b = builder.run(labels_b)
    calc = OverlapCalculator(
        LabelOverlapCalculatorConfig(
            metrics_calculators=[metric],
            intersection_calculator=LabelIntersectionCalculatorConfig(
                mode=intersection_mode
            ),
        )
    )
    result = calc.run(labels_a, labels_b, device=device)
    from vistiq.matrix.ops import MatrixFormatter, MatrixFormatterConfig

    scores = np.asarray(
        MatrixFormatter(
            MatrixFormatterConfig(output_type="np.ndarray", annotate=False)
        ).run(result.metric()),
        dtype=np.float32,
    )
    return scores, np.asarray(build_a.label_ids), np.asarray(build_b.label_ids)


class CoincidenceDetectorConfig(StackProcessorConfig):
    """Configuration for coincidence detection workflow.
    
    Attributes:
        output_type: Output type ("list" only).
        output: Output fields ("score" or "above_threshold").
        method: Overlap metric calculator config (e.g. ``IoUMetricsCalculatorConfig``).
        mode: Overlap mode ("bounding_box" or "outline").
        threshold: Threshold for the overlap score (must be between 0.0 and 1.0).
    """
    output_type: Literal["list"] = Field(default="list", description="Output type")
    output: List[Literal["score", "above_threshold"]] = Field(default=["score", "above_threshold"], description="Output fields")
    method: MetricsCalculatorConfig = Field(
        default_factory=IoUMetricsCalculatorConfig,
        description="Overlap metric calculator config",
    )
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

    def _process_slice(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
        stack_names: Tuple[str, str] = ("stack_1", "stack_2"),
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict]:
        """Compute pairwise overlap between regions in two label volumes."""
        labels1 = np.asarray(labels1)
        labels2 = np.asarray(labels2)
        stack_names = _normalize_stack_names(stack_names)
        if labels1.shape != labels2.shape:
            raise ValueError(
                "labels1 and labels2 must have the same shape: "
                f"{labels1.shape} vs {labels2.shape}"
            )

        if self.config.mode == "outline":
            scores, ids1, ids2 = _label_overlap_matrix(
                labels1,
                labels2,
                metric=self.config.method,
            )
        elif self.config.mode == "bounding_box":
            ids1, boxes1 = _label_ids_and_boxes(labels1)
            ids2, boxes2 = _label_ids_and_boxes(labels2)
            if len(ids1) == 0 or len(ids2) == 0:
                return []
            scores = _box_overlap_matrix(
                boxes1, boxes2, metric=self.config.method
            )
        else:
            raise ValueError(f"Unsupported coincidence mode: {self.config.mode!r}")

        if len(ids1) == 0 or len(ids2) == 0:
            return []

        results: List[Dict] = []
        for i, label1 in enumerate(ids1):
            for j, label2 in enumerate(ids2):
                score = float(scores[i, j])
                results.append(
                    {
                        stack_names[0]: int(label1),
                        stack_names[1]: int(label2),
                        "score": score,
                        "above_threshold": score >= self.config.threshold,
                    }
                )
        return results

    def _consolidate_results(self, results: List[Dict], stack_names: Tuple[str, str] = ["stack_1", "stack_2"]) -> Dict[str, pd.DataFrame]:
        """Consolidate the results of the coincidence detector."""
        stack_names = _normalize_stack_names(stack_names)
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
                    f"{self.config.method.name} {comp_stack_name} +": max(data["scores"]) if data["scores"] else 0.0
                })
            dataframes[stack_name] = pd.DataFrame(rows).set_index("label")
        
        return dataframes

    @task(name="CoincidenceDetector.run", tags=["gpu_concurrency_limited"])
    def run(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
        stack_names: Optional[Tuple[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run the coincidence detector on a labeled image."""
        if stack_names is None or len(stack_names) != 2:
            stack_names = ("stack_1", "stack_2")
        else:
            stack_names = _normalize_stack_names(stack_names)

        # labels2 is iterated in lock-step with labels1; stack_names is a plain
        # positional argument passed whole to every slice.
        results, _updated_metadata = super().run(
            labels1, stack_names, coiterate=[labels2], metadata=metadata, **kwargs
        )
        consolidated_dfs = self._consolidate_results(results, stack_names)
        return results, consolidated_dfs
