import logging
from typing import List, Optional, Tuple, Union, final, Literal, Any

import numpy as np
import pandas as pd
import torch
from prefect import task
from pydantic import field_validator, model_validator

from vistiq.core import Configurable, Configuration, generate_name
from vistiq.matrix.types import ArrayBackend
from vistiq.utils import convert_array_like
from vistiq.segment.analysis import (
    RegionAnalyzer,
    dataframe_to_numpy,
    region_to_numpy,
    regions_to_numpy,
)

logger = logging.getLogger(__name__)


def _value_column_count(data: Union[np.ndarray, "torch.Tensor"]) -> int:
    """Return the number of value columns represented by *data*."""
    if data.ndim <= 1:
        return 1
    return int(data.shape[-1])


class FilterConfig(Configuration):
    """Shared settings for a single :class:`Filter`.

    Attributes:
        attribute: Region property name, list of property names, or list of
            column indices used to select values from tabular input. When empty
            or unset, conversion utilities use all available columns/attributes
            (:func:`~vistiq.segment.analysis.dataframe_to_numpy`,
            :func:`~vistiq.segment.analysis.regions_to_numpy`).
        axis: Optional axis tuple (reserved for axis-aware filters).
        strict: When ``True``, :meth:`Filter.run` raises if the number of
            configured attributes does not match the width of the value array.
        preferred_input_type: Value array backend passed to
            :meth:`Filter._convert_input` (``"np.ndarray"`` or ``"torch.Tensor"``).
            Torch-backed filters should set this and implement
            :meth:`accept_indices` against :class:`torch.Tensor`.
    """
    attribute: Optional[Union[str, List[str]]] = None
    axis: Optional[Union[int, tuple[int, ...]]] = None
    strict: bool = True
    preferred_input_type: ArrayBackend = "np.ndarray"

    def attribute_list(self) -> List[str]:
        """Return configured attribute selector(s) as a list.

        A single string attribute becomes a one-element list. An attribute
        already given as a list is returned unchanged. ``None`` means all
        attributes (empty list).
        """
        attribute = self.attribute
        if attribute is None:
            return []
        return attribute if isinstance(attribute, list) else [attribute]

def _reject_configurable_filter_entries(filters: Any) -> Any:
    """Reject :class:`Configurable` instances in a ``filters`` field value."""
    if filters is None:
        return []
    if not isinstance(filters, list):
        return filters
    for index, entry in enumerate(filters):
        if isinstance(entry, Configurable):
            raise ValueError(
                f"filters[{index}] must be a FilterConfig subclass, not "
                f"{type(entry).__name__}; pass configuration objects only and "
                "instantiate filters via Filter.create_from_config()."
            )
    return filters


def _filter_config_entry(entry: FilterConfig) -> FilterConfig:
    """Return a validated filter config entry."""
    if isinstance(entry, Configurable):
        raise TypeError(
            f"Expected FilterConfig, got {type(entry).__name__}; "
            "use Filter.create_from_config() at runtime."
        )
    return entry


def _accept_index_set(
    indices: Union[np.ndarray, "torch.Tensor", tuple[Any, ...]],
) -> set[int]:
    """Normalize :meth:`Filter.accept_indices` output to a set of row indices."""
    if isinstance(indices, tuple):
        indices = indices[0]
    return set(np.asarray(indices, dtype=np.int64).tolist())


class Filter(Configurable[FilterConfig]):
    """Base class for selecting rows or regions by property values.

    Subclasses implement :meth:`accept_indices` (or override :meth:`accept` where
    allowed) to define acceptance criteria. :meth:`run` normalizes list,
    :class:`~pandas.Series`, :class:`~pandas.DataFrame`, or array input to a
    value array, then returns the subset of values that pass the filter.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, config: FilterConfig):
        """Initialize the filter.

        Args:
            config: Filter configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: FilterConfig) -> "Filter":
        """Create a Filter instance from a configuration.

        Args:
            config: Filter configuration.

        Returns:
            A new Filter instance.
        """
        return cls(config)

    def accept_indices(
        self, values: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Return indices of *values* that satisfy this filter.

        Subclasses must implement this. The return type depends on the
        subclass (for example a 1-D index array or the tuple returned by
        :func:`numpy.where`).

        Args:
            values: Value array in :attr:`FilterConfig.preferred_input_type`
                (already normalized by :meth:`run` or :meth:`_convert_input`).

        Raises:
            NotImplementedError: On the base class.
        """
        raise NotImplementedError("Filter.accept_indices is not implemented")

    @final
    def accept(
        self, values: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Return the subset of *values* at indices accepted by this filter.

        Delegates to :meth:`accept_indices`, then indexes *values* with the
        result. This is the array-selection API used by :meth:`run`.

        Args:
            values: Value array to filter.

        Returns:
            Filtered value array (fancy-indexed from *values*).
        """
        indices = self.accept_indices(values)
        return values[indices]

    def _convert_input(
        self,
        data: Union[np.ndarray, "torch.Tensor", List[float], List["RegionProperties"], pd.Series, pd.DataFrame],
        dtype: ArrayBackend = "np.ndarray",
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Normalize supported containers to the requested array backend."""
        if isinstance(data, torch.Tensor):
            return convert_array_like(data, dtype=dtype)

        if isinstance(data, pd.DataFrame):
            data = dataframe_to_numpy(data, attributes=self.config.attribute_list())
            if data is None:
                data = np.array([])
        elif isinstance(data, list) and data and hasattr(data[0], "label"):
            data = regions_to_numpy(data, attributes=self.config.attribute_list())
        elif isinstance(data, pd.Series):
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data)

        return convert_array_like(data, dtype=dtype, device=device)

    @final
    def run(
        self,
        data: Union[np.ndarray, "torch.Tensor", List[float], List["RegionProperties"], pd.Series, pd.DataFrame],
        *args: Any,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Extract property values from *data* and return those that pass.

        DataFrames are reduced with
        :func:`~vistiq.segment.analysis.dataframe_to_numpy`; lists of
        :class:`RegionProperties` are converted via
        :func:`~vistiq.segment.analysis.regions_to_numpy`.
        When ``strict`` is enabled, the number of configured attributes must
        match ``data.shape[-1]`` (the last axis of the value matrix).

        Args:
            data: Region table, value sequence, or raw array.

        Returns:
            Accepted values in the configured backend
            (:attr:`FilterConfig.preferred_input_type`).

        Raises:
            ValueError: When ``strict`` is ``True`` and attribute count does not
                match data width.
        """
        data = self._convert_input(
            data, *args, dtype=self.config.preferred_input_type, device=device
        )
        attribute_list = self.config.attribute_list()
        if (
            attribute_list
            and len(attribute_list) != _value_column_count(data)
        ):
            # some filter attributes are missing
            if self.config.strict:
                raise ValueError(f"Length of attribute list {attribute_list} and data shape do not match; Filter failed.")
            else:
                logger.warning(f"Length of attribute list {attribute_list} and data shape do not match; Strict mode is disabled, ignoring filters.")
        return self.accept(data)


class FilterOpsConfig(Configuration):
    """Configuration for :class:`FilterOps`.

    Attributes:
        filters: :class:`FilterConfig` entries combined by *operation*.
        operation: ``"and"`` (intersection), ``"or"`` (union), ``"xor"``
            (symmetric difference), or ``"not"`` (complement; exactly one filter).
    """
    filters: List[FilterConfig] = []
    operation: Literal["and", "or", "xor", "not"] = "and"

    @field_validator("filters", mode="before")
    @classmethod
    def _filters_must_be_configs(cls, filters: Any) -> Any:
        return _reject_configurable_filter_entries(filters)

class FilterOps(Configurable[FilterOpsConfig]):
    """Combine several :class:`Filter` criteria with a logical operation."""

    def __init__(self, config: FilterOpsConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: FilterOpsConfig) -> "FilterOps":
        """Create a FilterOps instance from a configuration."""
        return cls(config)

    def run(
        self,
        data: Union[np.ndarray, List[float], List["RegionProperties"], pd.Series, pd.DataFrame],
        *args: Any,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Return values whose indices pass the combined filter criteria."""
        filters = [
            Filter.create_from_config(entry) for entry in (self.config.filters or [])
        ]
        if not filters:
            return np.asarray([])

        if self.config.operation == "not" and len(filters) != 1:
            raise ValueError("FilterOps operation 'not' requires exactly one filter")

        values = filters[0]._convert_input(
            data,
            *args,
            dtype=filters[0].config.preferred_input_type,
            device=device,
        )
        if values.ndim == 0:
            n = 1
        else:
            n = int(values.shape[0])

        index_sets = [_accept_index_set(f.accept_indices(values)) for f in filters]
        operation = self.config.operation
        if operation == "and":
            selected = set.intersection(*index_sets) if index_sets else set()
        elif operation == "or":
            selected = set.union(*index_sets) if index_sets else set()
        elif operation == "xor":
            selected: set[int] = set()
            for index_set in index_sets:
                selected ^= index_set
        elif operation == "not":
            selected = set(range(n)) - index_sets[0]
        else:
            raise ValueError(f"Unsupported FilterOps operation: {operation!r}")

        if not selected:
            empty: Union[np.ndarray, torch.Tensor]
            if isinstance(values, torch.Tensor):
                empty = values.new_empty((0,) + values.shape[1:])
            else:
                empty = np.empty((0,) + values.shape[1:], dtype=values.dtype)
            return empty
        order = sorted(selected)
        return values[order]


from vistiq.matrix.select import (  # noqa: E402
    MatrixFilter,
    MatrixFilterConfig,
    TopKFilter,
    TopKFilterConfig,
    ValueFilter,
    ValueFilterConfig,
)

class MinFilterConfig(FilterConfig):
    """Configuration for :class:`MinFilter`.

    Attributes:
        minimum: Lower bound; a value must meet this floor to count as passing.
        operator: ``"gte"`` or ``"gt"`` comparison against ``minimum``.
    """
    minimum: float = None
    operator: Literal["gte", "gt"] = "gte"


class MinFilter(Filter):
    """Keep value rows where the attribute meets a minimum threshold.

    Designed for use with :class:`RegionFilter` on scalar or multi-column
    region property arrays from :func:`~vistiq.segment.analysis.region_to_numpy`.
    """

    def __init__(self, config: MinFilterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MinFilterConfig) -> "MinFilter":
        return cls(config)

    def accept_indices(self, values: np.ndarray) -> np.ndarray:
        """Return indices of values that meet ``minimum``."""
        minimum, operator = self.config.minimum, self.config.operator
        if operator == "gte":
            return np.where(values >= minimum)
        if operator == "gt":
            return np.where(values > minimum)
        raise ValueError(f"Invalid operator: {operator}")


class MaxFilterConfig(FilterConfig):
    """Configuration for :class:`MaxFilter`.

    Attributes:
        maximum: Upper bound; a value must meet this ceiling to count as passing.
        operator: ``"lte"`` or ``"lt"`` comparison against ``maximum``.
    """
    maximum: float = None
    operator: Literal["lte", "lt"] = "lte"


class MaxFilter(Filter):
    """Keep value rows where the attribute meets a maximum threshold.

    Designed for use with :class:`RegionFilter` on scalar or multi-column
    region property arrays from :func:`~vistiq.segment.analysis.region_to_numpy`.
    """

    def __init__(self, config: MaxFilterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MaxFilterConfig) -> "MaxFilter":
        return cls(config)

    def accept_indices(self, values: np.ndarray) -> np.ndarray:
        """Return indices of values that meet ``maximum``."""
        maximum, operator = self.config.maximum, self.config.operator
        if operator == "lte":
            return np.where(values <= maximum)
        if operator == "lt":
            return np.where(values < maximum)
        raise ValueError(f"Invalid operator: {operator}")


class RangeFilterConfig(FilterConfig):
    """Configuration for range-based region filtering.

    Filters regions based on whether a specified attribute value falls
    within a given range.

    Attributes:
        attribute: Name of the region property to filter on.
        range: Tuple of (min, max) values, or "all" to accept all values.
    """

    range: Union[tuple[float, float], str] = None


class RangeFilter(Filter):
    """Keep values inside a closed interval ``[min, max]``."""

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


    def accept_indices(self, values: np.ndarray) -> np.ndarray:
        """Return indices of values in ``[min_value(), max_value()]``."""
        lo, hi = self.min_value(), self.max_value()
        return np.where((values >= lo) & (values <= hi))[0]

class RegionFilterConfig(Configuration):
    """Configuration for :class:`RegionFilter`.

    Attributes:
        filters: Ordered list of :class:`FilterConfig` entries. All filters must
            pass (logical AND) for a region to be kept.
    """

    filters: List[FilterConfig] = []

    @field_validator("filters", mode="before")
    @classmethod
    def _filters_must_be_configs(cls, filters: Any) -> Any:
        return _reject_configurable_filter_entries(filters)

    @model_validator(mode="after")
    def validate_filters(self) -> "RegionFilterConfig":
        """Ensure every filter attribute is an allowed region property name.

        Each filter's :meth:`~FilterConfig.attribute_list` entries are checked with
        :meth:`~vistiq.segment.analysis.RegionAnalyzer.is_allowed_filter_attribute`.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If any attribute is not an allowed property or mapped
                column name.
        """
        if self.filters is None:
            self.filters = []
            return self

        for filter in self.filters:
            fc = _filter_config_entry(filter)
            attributes = fc.attribute_list()
            if attributes is None or len(attributes) == 0:
                continue
            if not all(RegionAnalyzer.is_allowed_filter_attribute(attribute) for attribute in attributes):
                raise ValueError(f"One or multiple filter attributes {attributes} are not allowed. Use a region property from {RegionAnalyzer.allowed_properties()} or a mapped axis column (e.g. 'cross_sectional_area-xy', 'centroid-y', 'bbox-start-z').")
        return self


class RegionFilter(Configurable[RegionFilterConfig]):
    """Drop regions that fail any configured :class:`Filter`.

    Accepts either a list of :class:`RegionProperties` or a region
    :class:`~pandas.DataFrame` (for example from :class:`RegionAnalyzer` with
    ``output_type="dataframe"``). Filters are applied in order; a region must
    satisfy every filter to be kept.
    """

    def __init__(self, config: RegionFilterConfig):
        """Initialize the region filter.

        Args:
            config: Region filter configuration.
        """
        super().__init__(config)
        # self.filters = [
        #    RangeFilter(filter_config) for filter_config in self.config.filters
        # ]

    @classmethod
    def from_config(cls, config: RegionFilterConfig) -> "RegionFilter":
        """Create a RegionFilter instance from a configuration.

        Args:
            config: RegionFilter configuration.

        Returns:
            A new RegionFilter instance.
        """
        return cls(config)

    @staticmethod
    def _mask_from_indices(indices: np.ndarray, index: pd.Index) -> pd.Series:
        """Build a per-row boolean mask from an index array."""
        if isinstance(indices, tuple):
            indices = indices[0]
        mask = pd.Series(False, index=index)
        if len(indices) > 0:
            mask.iloc[list(indices)] = True
        return mask

    def has_filter(self, attribute: str) -> bool:
        """Return whether any filter targets *attribute*.

        Matches a string ``attribute`` field exactly, or membership in a
        list-valued ``attribute``.
        """
        for entry in self.config.filters or []:
            fc = _filter_config_entry(entry)
            attr = fc.attribute
            if attr == attribute:
                return True
            if isinstance(attr, list) and attribute in attr:
                return True
        return False

    def get_filter(self, attribute: str) -> Filter:
        """Return the first :class:`Filter` whose config targets *attribute*.

        Raises:
            ValueError: If no filter matches.
        """
        for entry in self.config.filters or []:
            fc = _filter_config_entry(entry)
            attr = fc.attribute
            if attr == attribute or (isinstance(attr, list) and attribute in attr):
                return Filter.create_from_config(entry)
        raise ValueError(f"Filter for attribute '{attribute}' not found")

    def get_attribute_names(self) -> List[Union[str, List[str]]]:
        """Return each filter's ``attribute`` (string or list, unflattened)."""
        if not self.config.filters:
            return []
        names: List[Union[str, List[str]]] = []
        for entry in self.config.filters:
            fc = _filter_config_entry(entry)
            if fc.attribute is not None:
                names.append(fc.attribute)
        return names

    @task(name="RegionFilter.run")
    def run(self, regions: Union[List["RegionProperties"], pd.DataFrame]) -> Tuple[
        Union[List["RegionProperties"], pd.DataFrame],
        Union[List[int], np.ndarray],
    ]:
        """Keep regions that pass every configured filter.

        **DataFrame path:** AND-combines per-row boolean masks from each
        filter's :meth:`~Filter.accept_indices` on columns selected by
        :func:`~vistiq.segment.analysis.dataframe_to_numpy`. Filters whose columns are absent
        from the table are skipped. ``removed_labels`` is an ``int32`` NumPy
        array from the ``label`` column or index.

        **List path:** Iterates filters outermost; for each
        :class:`RegionProperties`, evaluates :meth:`~Filter.accept_indices` on
        :func:`~vistiq.segment.analysis.region_to_numpy`. Already-removed labels are skipped.
        ``removed_labels`` is a sorted list of integer label ids.

        Args:
            regions: :class:`RegionProperties` list or region property table.

        Returns:
            ``(accepted_regions, removed_labels)`` — same container type as
            *regions* for accepted rows/objects; label ids for removed regions.
        """
        logger.info(f"Running {type(self).__name__} with config: {self.config}")
        if self.config.filters is None or len(self.config.filters) == 0:
            logger.info("RegionFilter: no filters, returning all regions")
            empty_removed: Union[List[int], np.ndarray] = (
                np.array([], dtype=np.int32)
                if isinstance(regions, pd.DataFrame)
                else []
            )
            return regions, empty_removed

        # Handle DataFrame input
        if isinstance(regions, pd.DataFrame):
            logger.info("Applying RegionFilter to a DataFrame")
            mask = pd.Series(True, index=regions.index)
            for f in self.config.filters:
                filter = Filter.create_from_config(f)
                values = dataframe_to_numpy(
                    regions, attributes=filter.config.attribute_list()
                )
                if values is None:
                    continue
                indices = filter.accept_indices(values)
                mask &= self._mask_from_indices(indices, regions.index)

            accepted_regions = regions.loc[mask]
            removed_index = regions.index[~mask]

            if len(removed_index) > 0:
                if "label" in regions.columns:
                    removed_labels = (
                        regions.loc[removed_index, "label"]
                        .astype(np.int32)
                        .to_numpy()
                    )
                elif regions.index.name == "label":
                    removed_labels = np.asarray(removed_index, dtype=np.int32)
                else:
                    raise ValueError(
                        "RegionFilter received a DataFrame without a 'label' column "
                        "or index named 'label', so filtered rows cannot be mapped "
                        "back to segmentation labels safely."
                    )
            else:
                removed_labels = np.array([], dtype=np.int32)

            logger.info(
                f"RegionFilter: len(accepted_regions)={len(accepted_regions)}, "
                f"len(removed_labels)={len(removed_labels)}"
            )
            return accepted_regions, removed_labels

        logger.info("Applying RegionFilter to a list")

        # Handle list of RegionProperties input
        removed_labels: set[int] = set()
        for f in self.config.filters:
            filter = Filter.create_from_config(f)
            for region in regions:
                if region.label in removed_labels:
                    continue
                values = filter._convert_input(
                    region_to_numpy(
                        region, attributes=filter.config.attribute_list()
                    ),
                    dtype=filter.config.preferred_input_type,
                )
                indices = filter.accept_indices(values)
                if isinstance(indices, tuple):
                    indices = indices[0]
                if len(indices) == 0:
                    removed_labels.add(region.label)
        # Compare by label instead of using 'in' to avoid triggering RegionProperties.__eq__
        # which would compute all properties including ones not requested (like eccentricity)
        accepted_regions = [
            region for region in regions if region.label not in removed_labels
        ]
        logger.info(
            f"RegionFilter: len(accepted_regions)={len(accepted_regions)}, len(removed_labels)={len(removed_labels)}"
        )
        return accepted_regions, sorted(removed_labels)
