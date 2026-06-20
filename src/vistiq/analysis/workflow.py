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
from vistiq.analysis.overlap import OverlapCalculatorConfig, region_map_from_dataframe
from vistiq.analysis.coincidence import CoincidenceDetectorConfig
from vistiq.analysis.distance import DistanceCalculatorConfig
from vistiq.analysis.matrix import (
    HierarchicalMatrix,
    HierarchicalMatrixConfig,
    MatrixAggregatorConfig,
    MatrixCombiner,
    MatrixCombinerConfig,
)
from vistiq.analysis.spatial import (
    KnnAnalysis,
    KnnAnalysisConfig,
    RnnAnalysis,
    RnnAnalysisConfig,
    SpatialScopeConfig,
)
from vistiq.constant.matrix import UPPER
from vistiq.core import ArrayIteratorConfig, Configurable, generate_name
from vistiq.graph import (
    GraphBuilderConfig,
    GraphExporter,
    GraphExporterConfig,
    NXGraphBuilderConfig,
    NXGraphQuery,
    NXGraphQueryConfig,
    graph_to_dataframe,
    resolve_subtree_origins,
    subtree_origin_key,
)
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
    matrix_combiner: Optional[MatrixCombinerConfig] = Field(
        default_factory=lambda: MatrixCombinerConfig(
            fill_value=float("nan"),
            symmetrize=True,
            triangle=UPPER,
        )
    )
    hierarchical_matrix: Optional[HierarchicalMatrixConfig] = Field(
        default_factory=lambda: HierarchicalMatrixConfig(orphan_strategy="drop")
    )
    graph_builder: Optional[GraphBuilderConfig] = Field(
        default_factory=NXGraphBuilderConfig
    )
    graph_query: Optional[NXGraphQueryConfig] = Field(
        default_factory=lambda: NXGraphQueryConfig(
            attributes=["descendant_counts", "ancestor_lineage"],
            filter_attribute="channel",
            include_attributes=["label", "channel"],
            lineage_value_attribute="label",
            output_type="dataframe",
        )
    )
    graph_exporter: Optional[GraphExporterConfig] = Field(
        default_factory=GraphExporterConfig
    )
    spatial_graph_query: Optional[NXGraphQueryConfig] = Field(
        default_factory=lambda: NXGraphQueryConfig(
            attributes=["neighbor_summary"],
            include_attributes=[],
            weight_attribute="distance",
            output_type="dataframe",
        )
    )
    knn_analysis: Optional[KnnAnalysisConfig] = None
    rnn_analysis: Optional[RnnAnalysisConfig] = None
    spatial_scope: SpatialScopeConfig = Field(default_factory=SpatialScopeConfig)
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
        if self.config.hierarchical_matrix is not None:
            required_properties.append(self.config.hierarchical_matrix.rank_attribute)
        if self.config.knn_analysis is not None:
            required_properties.append("centroid")

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

    def _concat_region_tables(
        self, measurements: dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        ra_keys = sorted(key for key in measurements if key.startswith("region_analyzer:"))
        if not ra_keys:
            return None
        frames: list[pd.DataFrame] = []
        for key in ra_keys:
            frame = measurements[key]
            if "channel" not in frame.columns:
                frame = frame.assign(channel=key.removeprefix("region_analyzer:").strip())
            frames.append(frame)
        return pd.concat(frames, ignore_index=False)

    def _build_region_analyzer_all(
        self, measurements: dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        graph = measurements.get("containment_graph")
        if graph is not None and self.config.graph_exporter is not None:
            return GraphExporter(self.config.graph_exporter).run(graph)
        if graph is not None:
            return graph_to_dataframe(graph)
        return self._concat_region_tables(measurements)

    @staticmethod
    def _join_derived_columns(
        base: pd.DataFrame, derived: list[pd.DataFrame]
    ) -> pd.DataFrame:
        result = base
        for frame in derived:
            new_cols = frame.columns.difference(result.columns)
            if len(new_cols):
                result = result.join(frame[new_cols], how="left")
        return result

    @staticmethod
    def _finalize_region_analyzer_all(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop non-region rows and normalize ``channel`` for downstream grouping."""
        if "channel" not in frame.columns:
            return frame
        present = frame["channel"].notna()
        if not present.all():
            frame = frame.loc[present].copy()
        elif not frame["channel"].map(type).eq(str).all():
            frame = frame.copy()
        else:
            return frame
        frame["channel"] = frame["channel"].map(str)
        return frame

    def _hierarchical_analysis(
        self,
        ios_matrices: list[Any],
        regions: pd.DataFrame,
        stack_names: list[str],
    ) -> dict[str, Any]:
        if not ios_matrices:
            return {}
        if (
            self.config.matrix_combiner is None
            or self.config.hierarchical_matrix is None
            or self.config.graph_builder is None
        ):
            return {}

        ios_global = MatrixCombiner(self.config.matrix_combiner).run(ios_matrices)
        hm = HierarchicalMatrix(self.config.hierarchical_matrix).run(
            ios_global, regions
        )
        builder = Configurable.create_from_config(self.config.graph_builder)
        dag = builder.run(hm.matrix, hm.regions, annotations=None)
        output: dict[str, Any] = {
            "ios_global": ios_global,
            "containment_graph": dag,
        }

        gqcfg = self.config.graph_query
        if gqcfg is None:
            return output

        if gqcfg.filter_value is not None:
            query_configs = [gqcfg]
        elif "channel" in regions.columns:
            query_configs = [
                gqcfg.model_copy(update={"filter_value": channel})
                for channel in sorted(regions["channel"].dropna().unique())
            ]
        else:
            query_configs = [
                gqcfg.model_copy(update={"filter_value": name})
                for name in stack_names
            ]

        filter_values = [cfg.filter_value for cfg in query_configs]
        logger.info(f"Running graph query for filter values: {filter_values}")
        gq = NXGraphQuery(gqcfg.model_copy(update={"filter_value": None}))
        gq_results = list(
            gq.run.map(unmapped(dag), node=None, filter_value=filter_values)
        )
        counts_frames = [
            frame
            for frame in resolve_futures(
                list(gq.format.map(gq_results, unmapped("descendant_counts")))
            )
            if isinstance(frame, pd.DataFrame) and not frame.empty
        ]
        lineage_frames = [
            frame
            for frame in resolve_futures(
                list(gq.format.map(gq_results, unmapped("ancestor_lineage")))
            )
            if isinstance(frame, pd.DataFrame) and not frame.empty
        ]

        graph_parts = [
            frame
            for frame in (
                pd.concat(counts_frames, axis=0) if counts_frames else None,
                pd.concat(lineage_frames, axis=0) if lineage_frames else None,
            )
            if frame is not None and not frame.empty
        ]
        if not graph_parts:
            return output

        graph_df = pd.concat(graph_parts, axis=1)
        graph_df = graph_df.loc[:, ~graph_df.columns.duplicated()]
        output["hierarchical_analysis"] = graph_df
        return output

    def _spatial_analysis(
        self,
        containment_graph: Any,
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if containment_graph is None:
            return {}
        if self.config.knn_analysis is None and self.config.rnn_analysis is None:
            return {}

        scope = self.config.spatial_scope
        origins = resolve_subtree_origins(
            containment_graph,
            match=scope.match,
            exclude=scope.exclude,
            auto_root=scope.auto_root,
        )
        axes = tuple(metadata[0].get("axes", ())) if metadata else None
        gqcfg = self.config.spatial_graph_query
        index = (self.config.graph_exporter or GraphExporterConfig()).index
        output: dict[str, Any] = {}
        knn_mapped: Optional[list[Any]] = None
        knn_results: Optional[list[Any]] = None

        if self.config.knn_analysis is not None:
            knn = KnnAnalysis(self.config.knn_analysis)
            knn_mapped = list(
                knn.run.map(
                    unmapped(containment_graph),
                    node=origins,
                    axes=unmapped(axes),
                )
            )
            if len(origins) > 1:
                output["knn_analysis"] = {
                    subtree_origin_key(origin): result
                    for origin, result in zip(origins, knn_mapped)
                }
            else:
                output["knn_analysis"] = knn_mapped[0]

            if gqcfg is not None:
                knn_results = resolve_futures(knn_mapped)
                gq = NXGraphQuery(
                    gqcfg.model_copy(
                        update={
                            "output_index": index,
                            "group_attribute": self.config.knn_analysis.grouping_attribute,
                            "neighbor_analysis": "knn",
                            "neighbor_k": self.config.knn_analysis.k,
                        }
                    )
                )
                gq_results = list(
                    gq.run.map([result.graph for result in knn_results])
                )
                frames = [
                    frame
                    for frame in resolve_futures(
                        list(gq.format.map(gq_results, unmapped("neighbor_summary")))
                    )
                    if isinstance(frame, pd.DataFrame) and not frame.empty
                ]
                if len(origins) > 1:
                    frames = [
                        frame.assign(spatial_origin=subtree_origin_key(origin))
                        for frame, origin in zip(frames, origins)
                    ]
                if frames:
                    output["spatial_analysis"] = pd.concat(frames, axis=0)

        if self.config.rnn_analysis is not None:
            rnn = RnnAnalysis(self.config.rnn_analysis)
            if knn_mapped is not None:
                knn_results = knn_results or resolve_futures(knn_mapped)
                distance_inputs = [
                    result.distance_matrix for result in knn_results
                ]
            else:
                distance_inputs = [None] * len(origins)
            rnn_mapped = list(
                rnn.run.map(
                    unmapped(containment_graph),
                    node=origins,
                    distance_matrix=distance_inputs,
                    axes=unmapped(axes),
                )
            )
            if len(origins) > 1:
                output["rnn_analysis"] = {
                    subtree_origin_key(origin): result
                    for origin, result in zip(origins, rnn_mapped)
                }
            else:
                output["rnn_analysis"] = rnn_mapped[0]

            if gqcfg is not None:
                rnn_results = resolve_futures(rnn_mapped)
                gq = NXGraphQuery(
                    gqcfg.model_copy(
                        update={
                            "output_index": index,
                            "group_attribute": self.config.rnn_analysis.grouping_attribute,
                            "neighbor_analysis": "rnn",
                            "neighbor_radius": self.config.rnn_analysis.radius,
                        }
                    )
                )
                gq_results = list(
                    gq.run.map([result.graph for result in rnn_results])
                )
                frames = [
                    frame
                    for frame in resolve_futures(
                        list(gq.format.map(gq_results, unmapped("neighbor_summary")))
                    )
                    if isinstance(frame, pd.DataFrame) and not frame.empty
                ]
                if len(origins) > 1:
                    frames = [
                        frame.assign(spatial_origin=subtree_origin_key(origin))
                        for frame, origin in zip(frames, origins)
                    ]
                if frames:
                    output["rnn_spatial_analysis"] = pd.concat(frames, axis=0)

        return output


    def _run(
        self,
        labels: list[np.ndarray],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Run analysis on labeled image stacks of equal shape."""
        logger.info(f"Running analysis flow with config: {self.config}")
        logger.info(f"Number of labels: {len(labels)}")
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

        measurements = resolve_futures(results)

        stack_names = [_stack_name(metadata, index) for index in range(len(labels))]
        regions = self._concat_region_tables(measurements)
        if regions is not None:
            ios_matrices = [
                matrix
                for key, matrix in measurements.items()
                if key.startswith("overlap_filtered:")
            ]
            measurements = {
                **measurements,
                **self._hierarchical_analysis(
                    ios_matrices, regions, stack_names
                ),
            }
            measurements = {
                **measurements,
                **self._spatial_analysis(
                    measurements.get("containment_graph"),
                    metadata,
                ),
            }

        if self.config.auto_join:
            measurements = self._auto_join(resolve_futures(measurements))

        return resolve_futures(measurements)

    def _auto_join(self, measurements: dict[str, Any]) -> dict[str, Any]:
        """Build region_analyzer_all and return a new measurements dict."""
        base = self._build_region_analyzer_all(measurements)
        if base is None:
            return measurements

        derived: list[pd.DataFrame] = []
        for key in (
            "hierarchical_analysis",
            "spatial_analysis",
            "rnn_spatial_analysis",
        ):
            frame = measurements.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                derived.append(frame)

        drop_keys = {
            "hierarchical_analysis",
            "spatial_analysis",
            "rnn_spatial_analysis",
        }
        joined = self._join_derived_columns(base, derived)
        return {
            **{key: value for key, value in measurements.items() if key not in drop_keys},
            "region_analyzer_all": self._finalize_region_analyzer_all(joined),
        }


AnalysisFlowConfig.model_rebuild()
