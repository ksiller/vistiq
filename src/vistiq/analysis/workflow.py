from __future__ import annotations

import itertools
import logging
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Any, Optional, Union, Literal    
from pydantic import Field
from prefect import task, unmapped

from vistiq.analysis import OverlapResult
from vistiq.analysis.coincidence import CoincidenceDetectorConfig
from vistiq.analysis.distance import DistanceCalculatorConfig
from vistiq.analysis.matrix import MatrixAggregatorConfig
from vistiq.analysis.overlap import OverlapCalculatorConfig, region_map_from_dataframe
from vistiq.core import ArrayIteratorConfig, Configurable, generate_name
from vistiq.segment import MatrixFilterConfig
from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig
from vistiq.utils import resolve_futures
from vistiq.workflow import Workflow, WorkflowConfig

logger = logging.getLogger(__name__)


def _stack_name(metadata: Optional[list[dict[str, Any]]], index: int) -> str:
    if metadata is not None:
        try:
            return metadata[index]["channel_names"][0]
        except (KeyError, IndexError, TypeError):
            pass
    return f"stack_{index}"


def _pair_stack_names(
    metadata: Optional[list[dict[str, Any]]],
    pair: tuple[int, int],
) -> tuple[str, str]:
    return _stack_name(metadata, pair[0]), _stack_name(metadata, pair[1])


def _spacing_for_labels(
    metadata: Optional[dict[str, Any]],
    labels: np.ndarray,
) -> Optional[tuple[float, ...]]:
    if metadata is None:
        return None
    scale = metadata.get("scale")
    if scale is None:
        return None
    return tuple(scale[-labels.ndim :])


def _pair_key(stage: str, stack_names: tuple[str, str]) -> str:
    return f"{stage}: {stack_names[0]} vs {stack_names[1]}"


def _as_numpy(result: ArrayLike) -> np.ndarray:
    """Convert filter/aggregator outputs to host numpy for pandas labeling."""
    try:
        import torch

        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(result)


class AnalysisFlowConfig(WorkflowConfig):
    """Configuration for the analysis workflow."""

    region_analyzer: RegionAnalyzerConfig = Field(
        default_factory=lambda: RegionAnalyzerConfig(
            properties=["centroid"],
            output_type="dataframe",
            index_on="object_id",
            map_axes=True,
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    distance_calculator: Optional[DistanceCalculatorConfig] = None
    coincidence_detector: Optional[CoincidenceDetectorConfig] = None
    overlap_calculator: Optional[OverlapCalculatorConfig] = None
    overlap_filter: Optional[MatrixFilterConfig] = None
    overlap_aggregator: Optional[MatrixAggregatorConfig] = None
    auto_join: Optional[bool] = True
    pairing_mode: Literal["combinations", "permutations", "product"] = "permutations"


class AnalysisFlow(Workflow):
    """Workflow that runs region analysis and optional pairwise overlap/coincidence."""

    def __init__(self, config: AnalysisFlowConfig):
        super().__init__(config)

    def _pair_indices(self, n_items: int) -> list[tuple[int, int]]:
        if self.config.pairing_mode == "combinations":
            return list(itertools.combinations(range(n_items), 2))
        elif self.config.pairing_mode == "permutations":
            return list(itertools.permutations(range(n_items), 2))
        elif self.config.pairing_mode == "product":
            return list(itertools.product(range(n_items), repeat=2))
        else:
            raise ValueError(f"Invalid pairing mode: {self.config.pairing_mode}")

    def _region_analyzer_config(self) -> RegionAnalyzerConfig:
        required_properties = ["label", "object_id", "centroid"]
        if self.config.overlap_calculator is not None:
            required_properties.append("bbox")

        output_type = "dataframe"
        index_on = "object_id"
        basecfg = self.config.region_analyzer
        if basecfg is not None:
            properties = list(set(basecfg.properties + required_properties))
            return basecfg.model_copy(
                update={
                    "properties": properties,
                    "output_type": output_type,
                    "index_on": index_on,
                }
            )
        return RegionAnalyzerConfig(
            properties=required_properties,
            index_on=index_on,
            output_type=output_type,
        )

    @task(name="AnalysisFlow._to_dataframe", task_run_name=generate_name)
    def _to_dataframe(self, result: ArrayLike, overlap_result: OverlapResult) -> Union[pd.DataFrame, ArrayLike]:
        oc_cfg = self.config.overlap_calculator
        filter_cfg = self.config.overlap_filter
        if (
            oc_cfg.output_type == "dataframe"
            and filter_cfg.output in ("masked_values", "mask")
            and overlap_result.annotations is not None
        ):
            row, col = overlap_result.annotations
            return pd.DataFrame(_as_numpy(result), index=row, columns=col)
        return result

    @task(name="AnalysisFlow._to_dataseries", task_run_name=generate_name)
    def _to_dataseries(self, result: ArrayLike, overlap_result: OverlapResult, axis: int, series_name: str) -> Union[pd.Series, ArrayLike]:
        oc_cfg = self.config.overlap_calculator
        filter_cfg = self.config.overlap_filter
        if (
            oc_cfg.output_type == "dataframe"
            and filter_cfg.output in ("masked_values", "mask")
            and overlap_result.annotations is not None
        ):
            row, col = overlap_result.annotations
            values = _as_numpy(result).ravel()
            axis = int(axis)
            # axis=0 collapses rows → one value per column; axis=1 collapses columns → one per row
            index = col if axis == 0 else row
            if len(values) != len(index):
                raise ValueError(
                    f"aggregated result length {len(values)} does not match "
                    f"annotation length {len(index)} for axis={axis}"
                )
            return pd.Series(values, index=index, name=series_name)
        return result

    def _run(
        self,
        labels: list[np.ndarray],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Run analysis on labeled image stacks of equal shape."""
        if len(labels) == 0:
            raise ValueError("No labels provided")
        if any(label.shape != labels[0].shape for label in labels):
            raise ValueError("All labels must have the same shape")
        if metadata is not None and len(metadata) != len(labels):
            raise ValueError(
                "Number of metadata sets must match number of labeled image stacks"
            )

        results: dict[str, Any] = {}
        pair_indices = self._pair_indices(len(labels))

        racfg = self._region_analyzer_config()
        ra = RegionAnalyzer(racfg)
        r_results = ra.run.map(labels, metadata=metadata)
        for index, r_result in enumerate(r_results):
            results[f"region_analyzer: {_stack_name(metadata, index)}"] = r_result

        if self.config.overlap_calculator is not None and pair_indices:
            resolved_tables = resolve_futures(
                {str(index): r_result for index, r_result in enumerate(r_results)}
            )
            region_tables = [resolved_tables[str(index)] for index in range(len(labels))]

            l1 = [labels[i] for i, _ in pair_indices]
            l2 = [labels[j] for _, j in pair_indices]
            stack_names = [_pair_stack_names(metadata, pair) for pair in pair_indices]
            region_maps = [
                (
                    region_map_from_dataframe(
                        region_tables[i],
                        axes=metadata[i].get("axes") if metadata else None,
                    ),
                    region_map_from_dataframe(
                        region_tables[j],
                        axes=metadata[j].get("axes") if metadata else None,
                    ),
                )
                for i, j in pair_indices
            ]
            spacings = [
                _spacing_for_labels(metadata[i] if metadata else None, labels[i])
                for i, _ in pair_indices
            ]

            logger.info("Setting up overlap calculator for pairs: %s", stack_names)
            oc = Configurable.create_from_config(self.config.overlap_calculator)
            overlap_results = oc.run.map(
                l1,
                l2,
                region_map=region_maps,
                spacing=spacings,
            )
            formatted_results = oc.format.map(overlap_results)
            for stack_name, formatted_result in zip(stack_names, formatted_results):
                results[_pair_key("overlap", stack_name)] = formatted_result

            if self.config.overlap_filter is not None:
                overlap_filter = Configurable.create_from_config(
                    self.config.overlap_filter
                )
                overlap_matrices = oc.matrix.map(overlap_results)
                filtered_results = overlap_filter.run.map(overlap_matrices)
                labeled_filtered = self._to_dataframe.map(filtered_results, overlap_results)
                for stack_name, lf_result in zip(stack_names, labeled_filtered):
                    results[_pair_key("overlap_filtered", stack_name)] = lf_result

                if self.config.overlap_aggregator is not None:
                    overlap_aggregator = Configurable.create_from_config(
                        self.config.overlap_aggregator
                    )
                    agg_results = overlap_aggregator.run.map(filtered_results)
                    axis = self.config.overlap_aggregator.axis
                    operation = self.config.overlap_aggregator.operation
                    series_names = [
                        f"{operation} {a} vs {b}" for a, b in stack_names
                    ]
                    labeled_agg = self._to_dataseries.map(
                        agg_results,
                        overlap_results,
                        axis=unmapped(axis),
                        series_name=series_names,
                    )
                    for stack_name, la_result in zip(stack_names, labeled_agg):
                        results[_pair_key("overlap_aggregated", stack_name)] = la_result

        if self.config.coincidence_detector is not None and pair_indices:
            l1 = [labels[i] for i, _ in pair_indices]
            l2 = [labels[j] for _, j in pair_indices]
            stack_names = [_pair_stack_names(metadata, pair) for pair in pair_indices]
            logger.info("Setting up coincidence detector for pairs: %s", stack_names)
            cd = Configurable.create_from_config(self.config.coincidence_detector)
            c_results = cd.run.map(l1, l2, stack_names=stack_names)
            for stack_name, c_result in zip(stack_names, c_results):
                results[_pair_key("coincidence", stack_name)] = c_result

        resolved = resolve_futures(results)

        ra_keys = sorted(key for key in resolved if key.startswith("region_analyzer:"))
        if ra_keys:
            resolved["region_analyzer_all"] = pd.concat(
                [resolved[key] for key in ra_keys],
                ignore_index=False, # keep object_id as index
            )

        if self.config.auto_join:
            resolved = self._auto_join(resolved)
        return resolved

    def _auto_join(self, resolved: dict[str, Any]) -> dict[str, Any]:
        """Auto-join the resolved results."""
        return resolved
