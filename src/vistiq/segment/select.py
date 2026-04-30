import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from prefect import task
from pydantic import model_validator

from vistiq.core import Configurable, Configuration
from vistiq.segment.analysis import RegionAnalyzer

logger = logging.getLogger(__name__)


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

    def get_attribute_names(self) -> List[str]:
        """Get list of attribute names from all filters.

        Returns:
            List of attribute names from all filters in the configuration.
            Returns empty list if no filters are configured or if filters have no attributes.
        """
        if self.config.filters is None or len(self.config.filters) == 0:
            return []
        return [
            filter.config.attribute
            for filter in self.config.filters
            if filter.config.attribute is not None
        ]

    @task(name="RegionFilter.run")
    def run(self, regions: Union[List["RegionProperties"], pd.DataFrame]) -> Tuple[
        Union[List["RegionProperties"], pd.DataFrame],
        Union[List["RegionProperties"], pd.DataFrame],
    ]:
        """Filter regions based on configured criteria.

        Removes regions that don't pass all filter criteria. A region is
        removed if any of its property values fall outside the specified range.

        Args:
            regions: List of region properties or pandas DataFrame to filter.

        Returns:
            Tuple of (accepted_regions, removed_labels):
            - accepted_regions: Regions that passed all filters.
            - removed_labels: Region labels that failed at least one filter.
            Returns the same type as input (list or DataFrame).
        """
        logger.info(f"Running {type(self).__name__} with config: {self.config}")
        if self.config.filters is None or len(self.config.filters) == 0:
            logger.info("RegionFilter: no filters, returning all regions")
            return regions, ([] if isinstance(regions, list) else pd.DataFrame())

        # Handle DataFrame input
        if isinstance(regions, pd.DataFrame):
            logger.info(f"Applying RegionFilter to a DataFrame")
            removed_indices = []

            for idx, row in regions.iterrows():
                for filter in self.config.filters:
                    if filter.config.attribute not in row.index:
                        logger.warning(
                            f"Attribute '{filter.config.attribute}' not found in DataFrame columns"
                        )
                        continue

                    value = row[filter.config.attribute]
                    if not filter.in_range(value):
                        removed_indices.append(idx)
                        break

            accepted_regions = regions.drop(index=removed_indices)

            if removed_indices:
                if "label" in regions.columns:
                    removed_labels = (
                        regions.loc[removed_indices, "label"]
                        .astype(np.int32)
                        .to_numpy()
                    )
                elif regions.index.name == "label":
                    removed_labels = np.asarray(removed_indices, dtype=np.int32)
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
        else:
            logger.info(f"Applying RegionFilter to a list")

        # Handle list of RegionProperties input
        # removed_labels = []
        removed_labels = set()  # Use set for O(1) lookup by label
        for region in regions:
            for filter in self.config.filters:
                value = getattr(region, filter.config.attribute)
                if not filter.in_range(value):
                    # logger.debug(
                    #    f"filter {filter.config.attribute} value={value} not in range for region {region.label}"
                    # )
                    # removed_labels.append(region)
                    removed_labels.add(region.label)
                    break
        # Compare by label instead of using 'in' to avoid triggering RegionProperties.__eq__
        # which would compute all properties including ones not requested (like eccentricity)
        accepted_regions = [
            region for region in regions if region.label not in removed_labels
        ]
        logger.info(
            f"RegionFilter: len(accepted_regions)={len(accepted_regions)}, len(removed_labels)={len(removed_labels)}"
        )
        return accepted_regions, sorted(removed_labels)
