import logging
import uuid
from functools import wraps
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from prefect import task
from skimage.measure import label as sk_label, regionprops, regionprops_table, perimeter

from vistiq.core import StackProcessor, StackProcessorConfig
from vistiq.segment._debug import debug_mask_labels
from vistiq.utils import ArrayIteratorConfig, axis_labels_from_metadata

logger = logging.getLogger(__name__)


def _normalize_attribute_list(
    attributes: Optional[Union[str, List[str]]],
) -> List[str]:
    """Normalize *attributes* to a (possibly empty) list of selectors."""
    if attributes is None:
        return []
    return attributes if isinstance(attributes, list) else [attributes]


def _reorder_columns_by_axes(
    columns: list[str], axes: Sequence[str]
) -> Optional[list[str]]:
    """Order *columns* by axis suffix; suffix matching is case-insensitive."""
    suffix_to_col = {str(col).rsplit("-", 1)[-1].lower(): str(col) for col in columns}
    ordered: list[str] = []
    for axis in axes:
        col = suffix_to_col.get(str(axis).lower())
        if col is None:
            return None
        ordered.append(col)
    if len(ordered) != len(columns):
        return None
    return ordered


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _is_scalar(value: Any) -> bool:
    """Return whether *value* is a single filterable scalar (not vector/dict)."""
    return not isinstance(value, (tuple, list, np.ndarray, dict))


def _region_attribute_raw(region: Any, col: str) -> Any:
    """Read an attribute without scalar coercion (for vector flattening)."""
    if col in region.__dict__:
        return region.__dict__[col]
    if col != RegionAnalyzer.base_property_name(col):
        return RegionAnalyzer.get_region_attribute(region, col)
    return getattr(region, col)


def _region_attribute_cell(region: Any, col: str, scalar_only: bool = True) -> Any:
    """Read one attribute from *region*.

    Args:
        region: :class:`RegionProperties` instance.
        col: Attribute or mapped column name.
        scalar_only: When ``True``, use :meth:`RegionAnalyzer.get_region_attribute`
            and raise if the resolved value is a vector or dict. When ``False``,
            return the raw property (vectors are flattened later).

    Raises:
        ValueError: If ``scalar_only`` is ``True`` and the value is not scalar.
        AttributeError: If the attribute cannot be resolved.
    """
    if scalar_only:
        value = RegionAnalyzer.get_region_attribute(region, col)
        if not _is_scalar(value):
            raise ValueError(
                f"Attribute '{col}' is not scalar (got {type(value).__name__!r}); "
                "use mapped component names (e.g. 'centroid-y') or explicit scalar columns."
            )
        return value
    try:
        return _region_attribute_raw(region, col)
    except AttributeError as exc:
        raise AttributeError(
            f"'{type(region).__name__}' object has no attribute '{col}'"
        ) from exc


def _attribute_as_1d(region: Any, col: str, *, scalar_only: bool) -> np.ndarray:
    """Read one attribute and return its values as a 1-D array."""
    value = _region_attribute_cell(region, col, scalar_only=scalar_only)
    return np.atleast_1d(np.asarray(value)).ravel()


def region_column_names(
    regions: List[Any],
    attributes: Optional[Union[str, List[str]]] = None,
    scalar_only: bool = False,
) -> List[str]:
    """Resolve attribute column names for :class:`RegionProperties` input.

    When *attributes* is set, those names are returned unchanged. When
    *attributes* is ``None`` or empty, uses ``_vistiq_property_names`` attached
    by :class:`RegionAnalyzer` (if present), plus any other materialized
    ``region.__dict__`` keys. Each candidate must resolve on at least one
    region; with ``scalar_only=True``, vector properties are omitted.

    Args:
        regions: Region objects used for auto-discovery when *attributes* is
            unset.
        attributes: Explicit attribute name(s) to select.
        scalar_only: When auto-discovering columns, include only scalar
            properties if ``True``; include vectors too if ``False``.

    Returns:
        Sorted list of column names.
    """
    cols = _normalize_attribute_list(attributes)
    if cols:
        return cols
    if not regions:
        return []

    candidates: set[str] = set()
    for region in regions:
        configured = region.__dict__.get("_vistiq_property_names")
        if configured:
            candidates.update(configured)
        for key in region.__dict__:
            if not key.startswith("_"):
                candidates.add(key)

    names: set[str] = set()
    for name in candidates:
        for region in regions:
            try:
                if scalar_only:
                    value = RegionAnalyzer.get_region_attribute(region, name)
                else:
                    value = _region_attribute_raw(region, name)
                if _is_scalar(value) or not scalar_only:
                    names.add(name)
                    break
            except (AttributeError, ValueError):
                continue
    return sorted(names)


def region_to_numpy(
    region: "RegionProperties",
    attributes: Optional[Union[str, List[str]]] = None,
    *,
    scalar_only: bool = False,
    cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract property values from one :class:`RegionProperties` object.

    This is the primitive used by :func:`regions_to_numpy`. Each requested
    column is read via :meth:`RegionAnalyzer.get_region_attribute`, then
    raveled and concatenated left-to-right into a single 1-D feature vector.

    Args:
        region: Region object to read.
        attributes: Attribute name(s) to select. When unset, keys from
            ``region.__dict__`` are auto-discovered on this object only.
        scalar_only: When ``True``, vector or dict attributes raise
            :class:`ValueError` at read time; when auto-discovering columns,
            only scalar keys are listed. When ``False``, vectors are flattened
            into the output (e.g. ``centroid`` contributes ``ndim`` values).
        cols: Pre-resolved column names. When provided, *attributes* is not
            used for column resolution (used internally by
            :func:`regions_to_numpy`).

    Returns:
        1-D NumPy array of flattened values, or an empty array when no columns
        resolve.
    """
    if cols is None:
        cols = region_column_names([region], attributes, scalar_only=scalar_only)
    if not cols:
        return np.array([])
    return np.concatenate(
        [_attribute_as_1d(region, col, scalar_only=scalar_only) for col in cols]
    )


def regions_to_numpy(
    regions: List["RegionProperties"],
    attributes: Optional[Union[str, List[str]]] = None,
    scalar_only: bool = False,
) -> np.ndarray:
    """Extract property values from a list of :class:`RegionProperties` objects.

    Column names are resolved once across *regions*, then each region is
    converted with :func:`region_to_numpy` and stacked with :func:`numpy.vstack`.

    Mapped names (for example ``cross_sectional_area-xy``) are resolved through
    :meth:`RegionAnalyzer.get_region_attribute`. When *attributes* is unset,
    keys from ``region.__dict__`` are auto-discovered; set ``scalar_only=True``
    to omit vector properties, or pass vector names explicitly with
    ``scalar_only=False`` to flatten them.

    Args:
        regions: Region objects to convert.
        attributes: Attribute name(s) to select. When unset, keys are
            auto-discovered from the union of all regions.
        scalar_only: Forwarded to :func:`region_column_names` and
            :func:`region_to_numpy`.

    Returns:
        ``(n_regions,)`` when there is a single scalar column; otherwise
        ``(n_regions, n_features)`` where *n_features* is the total flattened
        width of all requested columns. Empty inputs yield ``(0, 0)`` or
        ``(n_regions, 0)`` when no columns resolve.
    """
    if not regions:
        return np.empty((0, 0))
    cols = region_column_names(regions, attributes, scalar_only=scalar_only)
    if not cols:
        return np.empty((len(regions), 0))
    rows = [
        region_to_numpy(
            region,
            scalar_only=scalar_only,
            cols=cols,
        )
        for region in regions
    ]
    matrix = np.vstack(rows)
    if len(cols) == 1 and matrix.shape[1] == 1:
        return matrix[:, 0]
    return matrix


def dataframe_to_numpy(
    df: pd.DataFrame,
    attributes: Optional[Union[str, List[str]]] = None,
    strict: bool = True,
    axes: Optional[Sequence[str]] = None,
    reset_index: bool = True,
) -> Optional[np.ndarray]:
    """Select DataFrame column(s) and return them as a NumPy array.

    Companion to :func:`regions_to_numpy` for tabular region output from
    :class:`RegionAnalyzer` (``output_type="dataframe"``).

    When *attributes* is ``None`` or empty, returns the full table via
    :meth:`~pandas.DataFrame.to_numpy`. String selectors name columns; integer
    selectors choose by position via ``iloc``.

    When *axes* is set (e.g. ``metadata["axes"]``), prefix-matched columns are
    reordered by axis suffix with case-insensitive matching (``"Z"`` matches
    ``centroid-z``, ``bbox-start-z``, …). If reordering fails, columns keep
    dataframe order and a warning is logged.

    When *reset_index* is ``True`` (default), a non-default index (e.g.
    ``label`` from :class:`RegionAnalyzer`) is promoted to columns before
    selection. Plain :class:`~pandas.RangeIndex` frames are left unchanged.

    Args:
        df: Region property table.
        attributes: Column name(s) or integer position(s) to select.
        strict: When ``True`` (default), string selectors must match column
            names exactly. When ``False``, string selectors match any column
            whose name starts with the selector (useful for prefixes such as
            ``centroid`` matching ``centroid-y``, ``centroid-z``, …).
        axes: Optional axis names for reordering prefix-matched columns.
        reset_index: Promote a named/non-range index to columns before select.

    Returns:
        NumPy array of selected values, or ``None`` when no matching columns
        remain. A single selected column yields ``(n_rows,)``; multiple columns
        yield ``(n_rows, n_cols)``.

    Raises:
        ValueError: If attribute entries are neither strings nor integers.
    """
    if reset_index and not (
        isinstance(df.index, pd.RangeIndex) and df.index.name is None
    ):
        df = df.reset_index()
    attribute_list = _normalize_attribute_list(attributes)
    if not attribute_list:
        return df.to_numpy()
    if isinstance(attribute_list[0], str):
        if strict:
            existing = [col for col in attribute_list if col in df.columns]
        else:
            existing = [
                col
                for col in df.columns
                if any(str(col).startswith(attr) for attr in attribute_list)
            ]
        if not existing:
            return None
        if axes is not None:
            reordered = _reorder_columns_by_axes([str(col) for col in existing], axes)
            if reordered is None:
                logger.warning(
                    "Could not reorder columns for axes %s; using dataframe order. "
                    "Available columns: %s",
                    list(axes),
                    list(df.columns),
                )
            else:
                existing = reordered
        if len(existing) == 1:
            return df[existing[0]].to_numpy()
        return df[existing].to_numpy()
    if isinstance(attribute_list[0], int):
        existing = [col for col in attribute_list if col < df.shape[1]]
        missing = [col for col in attribute_list if col not in existing]
        if missing:
            logger.warning(
                "Attribute index(es) %s out of range for DataFrame; skipping filter",
                missing,
            )
        if not existing:
            return None
        selected = df.iloc[:, existing] if len(existing) > 1 else df.iloc[:, existing[0]]
        return selected.to_numpy()
    raise ValueError(f"Invalid attribute list type: {type(attribute_list[0])}")


def bbox_array_from_dataframe(
    df: pd.DataFrame,
    *,
    bbox_cols: Optional[Sequence[str]] = None,
    axes: Optional[Sequence[str]] = None,
    reset_index: bool = True,
) -> Optional[np.ndarray]:
    """Extract bbox bounds as ``(n_rows, 2 * ndim)`` from a RegionAnalyzer table.

    Supports mapped columns ``bbox-start-{axis}`` / ``bbox-end-{axis}`` and
    unmapped ``bbox-0`` … ``bbox-{2 * ndim - 1}`` layouts.
    """
    if reset_index and not (
        isinstance(df.index, pd.RangeIndex) and df.index.name is None
    ):
        df = df.reset_index()
    if bbox_cols is not None:
        if all(col in df.columns for col in bbox_cols):
            return df[list(bbox_cols)].to_numpy(dtype=np.float64)
        return None

    columns = [str(col) for col in df.columns]
    if any(col.lower().startswith("bbox-start-") for col in columns):
        starts = dataframe_to_numpy(
            df,
            attributes=["bbox-start"],
            strict=False,
            axes=axes,
            reset_index=False,
        )
        ends = dataframe_to_numpy(
            df,
            attributes=["bbox-end"],
            strict=False,
            axes=axes,
            reset_index=False,
        )
        if starts is None or ends is None:
            return None
        return np.hstack([_as_2d(starts), _as_2d(ends)])

    unmapped = sorted(
        (
            col
            for col in df.columns
            if str(col).startswith("bbox-") and str(col).split("-")[-1].isdigit()
        ),
        key=lambda col: int(str(col).split("-")[-1]),
    )
    if unmapped:
        return df[unmapped].to_numpy(dtype=np.float64)
    return None


class RegionAnalyzer(StackProcessor):
    """Extract region properties from labeled images.

    Wraps :func:`skimage.measure.regionprops` (or ``regionprops_table``) and
    adds vistiq-specific identifiers, optional axis-aware column names, and
    custom shape metrics. Iteration over the label stack is controlled by
    :attr:`RegionAnalyzerConfig.iterator_config` (inherited from
    :class:`~vistiq.core.StackProcessor`).

    Mandatory outputs
        Every region includes ``label``, a unique ``object_id``, a shared
        ``slice_id`` per iterator slice, and a shared ``stack_id`` per
        :meth:`run` call (:attr:`mandatory_properties`).

    Custom extra properties
        :meth:`extra_properties_funcs` registers ``circularity``, ``sphericity``,
        ``aspect_ratio``, ``cross_sectional_area``, and ``volume``.

    Area vs volume
        scikit-image ``area`` on ``ndim >= 3`` label slices is voxel count (volume).
        :meth:`_relabel_area_as_volume` exposes it as ``volume`` in outputs. For
        2D slices the column/attribute remains ``area``. regionprops runs with
        the original signed spacing (needed for coordinates such as centroid);
        :meth:`_ensure_positive_extent_values` then makes area/volume/cross-section
        non-negative.

    Vector properties (``cross_sectional_area``, ``aspect_ratio``)
        For ``ndim > 2`` these return tuples — one entry per orthogonal plane
        (``aspect_ratio`` also appends an overall value). For 2D slices they
        return a single scalar.

        * **DataFrame** — set ``map_axes=True`` to rename ``prop-0`` columns
          to axis or plane labels (``centroid-y``, ``cross_sectional_area-xy``,
          ``aspect_ratio`` for the overall component).
        * **List** — include mapped names in ``properties`` (e.g.
          ``cross_sectional_area-xy``); :meth:`_expand_mapped_property_attributes`
          attaches scalar components for :class:`~vistiq.segment.select.RegionFilter`.
        * :meth:`get_region_attribute` resolves mapped names and treats bare
          ``aspect_ratio`` as the overall scalar; bare ``cross_sectional_area``
          requires a plane suffix.

    Slice annotations
        When ``slice_annotations`` appears in ``properties``, per-slice axis
        indices from the stack iterator are attached as dataframe columns or list
        attributes. Keys use lowercase axis labels (``c``, ``z``, …).

    Metadata
        Optional. ``metadata['scale']`` supplies voxel spacing; ``metadata['axes']``
        (or ``axis`` / ``dim_order``) drives axis renaming. If metadata is
        ``None`` or lacks axes, generic labels ``axis_0``, ``axis_1``, … are
        used.

    Attributes:
        mandatory_properties: Always attached — ``label``, ``object_id``,
            ``slice_id``, ``stack_id``.
        default_properties: ``mandatory_properties`` plus ``centroid``.
        postcomputed_properties: Set after regionprops; not passed to scikit-image
            as ``extra_properties`` (includes ``slice_annotations``).
    """

    mandatory_properties: ClassVar[tuple[str, ...]] = (
        "label",
        "object_id",
        "slice_id",
        "stack_id",
    )
    default_properties: ClassVar[tuple[str, ...]] = mandatory_properties + ("centroid",)
    postcomputed_properties: ClassVar[frozenset[str]] = frozenset(
        {"object_id", "slice_id", "stack_id", "slice_annotations", "channel"}
    )

    @classmethod
    def ensure_mandatory_properties(cls, properties: List[str]) -> List[str]:
        """Return *properties* with :attr:`mandatory_properties` present (stable order)."""
        props = list(properties)
        for name in cls.mandatory_properties:
            if name not in props:
                if name == "label":
                    props.insert(0, name)
                elif "label" in props:
                    props.insert(props.index("label") + 1, name)
                else:
                    props.insert(0, name)
        return props

    def __init__(self, config: "RegionAnalyzerConfig"):
        """Initialize the region analyzer.

        Args:
            config: Region analyzer configuration.
        """
        super().__init__(config)

    # @cached_property
    @staticmethod
    def builtin_properties() -> List[str]:
        """Get list of built-in region properties from scikit-image.

        Returns:
            List of property names available from regionprops.
        """
        fake_array = np.ones((2, 2))
        labels = sk_label(fake_array)
        regions = regionprops(labels)
        return sorted([attr for attr in dir(regions[0]) if not attr.startswith("_")])

    @classmethod
    def extra_properties_funcs(cls) -> Dict[str, Callable]:
        """Get dictionary of custom extra property functions.

        Returns:
            Dictionary mapping property names to their computation functions.
        """
        return {
            "circularity": cls.circularity,
            "sphericity": cls.sphericity,
            "aspect_ratio": cls.aspect_ratio,
            "cross_sectional_area": cls.cross_sectional_area,
            "volume": cls.volume,
        }

    @staticmethod
    def allowed_properties() -> List[str]:
        """Get list of all allowed property names.

        Returns:
            Combined list of built-in and custom property names.
        """
        return sorted(
            RegionAnalyzer.builtin_properties()
            + list(RegionAnalyzer.extra_properties_funcs().keys())
            + list(RegionAnalyzer.postcomputed_properties)
        )

    @classmethod
    def is_allowed_property_name(cls, name: Optional[str]) -> bool:
        """Whether *name* is a valid base property or mapped column identifier.

        Accepts scikit-image property names, custom extras, postcomputed names
        (``slice_annotations``), and ``map_axes`` forms such as ``centroid-y``,
        ``bbox-start-z``, ``cross_sectional_area-xy``, and ``aspect_ratio-0``.
        """
        if name is None:
            return True

        allowed = set(cls.allowed_properties())
        if name in allowed:
            return True

        parts = name.split("-")
        if (
            len(parts) == 3
            and parts[0] == "bbox"
            and parts[1] in ("start", "end")
            and parts[2]
            and "bbox" in allowed
        ):
            return True

        if "-" in name:
            prop, suffix = name.rsplit("-", 1)
            if prop in allowed and suffix:
                return True

        return False

    @classmethod
    def is_allowed_filter_attribute(cls, attribute: Optional[str]) -> bool:
        """Alias for :meth:`is_allowed_property_name` (RangeFilter compatibility)."""
        return cls.is_allowed_property_name(attribute)

    @classmethod
    def base_property_name(cls, name: str) -> str:
        """Map a property or mapped column name to its regionprops base name.

        Examples: ``cross_sectional_area-xy`` → ``cross_sectional_area``,
        ``bbox-start-z`` → ``bbox``, ``centroid-y`` → ``centroid``.
        """
        if name in cls.allowed_properties():
            return name

        parts = name.split("-")
        if (
            len(parts) == 3
            and parts[0] == "bbox"
            and parts[1] in ("start", "end")
            and "bbox" in cls.allowed_properties()
        ):
            return "bbox"

        if "-" in name:
            prop, _suffix = name.rsplit("-", 1)
            if prop in cls.allowed_properties():
                return prop

        return name

    @classmethod
    def component_from_mapped_name(
        cls,
        name: str,
        base_value: Any,
        slice_axis_labels: List[str],
    ) -> Any:
        """Extract one scalar component from a vector base property value.

        Args:
            name: Mapped property name (e.g. ``cross_sectional_area-xy``).
            base_value: Value from regionprops (tuple, scalar, or array).
            slice_axis_labels: Axis labels for the analyzed slice dimensions.

        Returns:
            Scalar component, or ``None`` if *name* does not match *base_value*.
        """
        if name == cls.base_property_name(name):
            return base_value

        ndim = len(slice_axis_labels)
        axis_names = [str(axis).lower() for axis in slice_axis_labels]
        plane_pairs = cls.cross_sectional_area_plane_indices(ndim)
        base = cls.base_property_name(name)

        if base == "bbox":
            parts = name.split("-")
            if len(parts) == 3 and parts[1] in ("start", "end") and parts[2] in axis_names:
                idx = axis_names.index(parts[2])
                bbox = tuple(base_value)
                if parts[1] == "start":
                    return bbox[idx]
                return bbox[idx + ndim]
            return None

        suffix = name.rsplit("-", 1)[1]

        if base in ("cross_sectional_area", "aspect_ratio"):
            if isinstance(base_value, (tuple, list)) and suffix.isdigit():
                idx = int(suffix)
                if base == "aspect_ratio" and idx == len(plane_pairs) and ndim > 2:
                    return base_value[idx]
                if 0 <= idx < len(base_value):
                    return base_value[idx]
            if ndim == 2 and len(plane_pairs) == 1:
                plane = cls.cross_sectional_area_plane_label(slice_axis_labels, 0, 1)
                if plane == suffix:
                    if isinstance(base_value, (tuple, list)):
                        return base_value[0]
                    return base_value
            if isinstance(base_value, (tuple, list)):
                for i, (axis_i, axis_j) in enumerate(plane_pairs):
                    plane = cls.cross_sectional_area_plane_label(
                        slice_axis_labels, axis_i, axis_j
                    )
                    if plane == suffix:
                        return base_value[i]
            return None

        if isinstance(base_value, (tuple, list, np.ndarray)) and suffix in axis_names:
            return base_value[axis_names.index(suffix)]

        return None

    @classmethod
    def _region_slice_axes(cls, region: Any) -> Optional[List[str]]:
        """Return slice axis labels stored on *region*, if available."""
        slice_axes = region.__dict__.get("_vistiq_slice_axes")
        if slice_axes is not None:
            return list(slice_axes)
        if hasattr(region, "image") and region.image is not None:
            ndim = region.image.ndim
            return [f"axis_{i}" for i in range(ndim)]
        return None

    @classmethod
    def _scalar_from_base_property(cls, name: str, value: Any) -> Any:
        """Return a scalar when *name* is an unmapped vector extra property.

        Raises:
            AttributeError: If *name* is ``cross_sectional_area`` but *value* is
                a multi-component tuple (use e.g. ``cross_sectional_area-xy``).
        """
        if not isinstance(value, (tuple, list, np.ndarray)):
            return value
        if name == "aspect_ratio":
            # 3D+ tuples are (per-plane..., overall); overall is always last.
            return value[-1]
        if name == "cross_sectional_area":
            raise AttributeError(
                "cross_sectional_area is a per-plane vector; use a mapped name "
                "such as 'cross_sectional_area-xy'"
            )
        return value

    def _expand_mapped_property_attributes(
        self,
        regions: List[Any],
        labels_ndim: int,
        metadata: Optional[dict[str, Any]],
    ) -> List[Any]:
        """Attach mapped property scalars and slice axis labels to each region.

        Always sets ``_vistiq_slice_axes`` on each region. Names in
        ``config.properties`` that differ from their base property (e.g.
        ``cross_sectional_area-xy``) are resolved and stored on ``region.__dict__``
        for :meth:`get_region_attribute` and :class:`~vistiq.segment.select.RegionFilter`.
        """
        mapped_names = {
            name
            for name in self.config.properties
            if name != RegionAnalyzer.base_property_name(name)
        }
        slice_axes = self._slice_axis_labels(labels_ndim, metadata)
        for region in regions:
            region.__dict__["_vistiq_slice_axes"] = slice_axes
            for name in mapped_names:
                base = RegionAnalyzer.base_property_name(name)
                try:
                    base_value = getattr(region, base)
                except AttributeError:
                    continue
                component = RegionAnalyzer.component_from_mapped_name(
                    name, base_value, slice_axes
                )
                if component is not None:
                    region.__dict__[name] = component
        return regions

    @classmethod
    def get_region_attribute(cls, region: Any, name: str) -> Any:
        """Read a property from a region, including mapped axis/plane names.

        Resolution order: expanded ``__dict__`` entry, mapped name via
        :meth:`component_from_mapped_name`, then base ``getattr``. Vector extras
        are unpacked where defined (overall ``aspect_ratio``; plane-specific
        ``cross_sectional_area`` requires a suffix).

        Args:
            region: A :class:`skimage.measure.RegionProperties` instance.
            name: Base or mapped property name.

        Returns:
            Property value (scalar for filtered attributes).

        Raises:
            AttributeError: If the property or component cannot be resolved.
        """
        if name in region.__dict__:
            val = region.__dict__[name]
            if name == cls.base_property_name(name):
                return cls._scalar_from_base_property(name, val)
            return val

        base = cls.base_property_name(name)
        if name != base:
            slice_axes = cls._region_slice_axes(region)
            if slice_axes is None:
                raise AttributeError(
                    f"'{type(region).__name__}' object has no attribute '{name}'"
                )
            try:
                base_value = getattr(region, base)
            except AttributeError as exc:
                raise AttributeError(
                    f"'{type(region).__name__}' object has no attribute '{name}'"
                ) from exc
            component = cls.component_from_mapped_name(name, base_value, slice_axes)
            if component is None:
                raise AttributeError(
                    f"'{type(region).__name__}' object has no attribute '{name}'"
                )
            return component

        try:
            value = getattr(region, name)
        except AttributeError as exc:
            raise AttributeError(
                f"'{type(region).__name__}' object has no attribute '{name}'"
            ) from exc
        return cls._scalar_from_base_property(name, value)

    def used_extra_properties(self) -> List[str]:
        """Get list of extra properties that are being used.

        Returns:
            List of extra property names from config that are custom properties.
        """
        bases: List[str] = []
        seen: set[str] = set()
        for prop in self.config.properties:
            base = RegionAnalyzer.base_property_name(prop)
            if (
                base in RegionAnalyzer.extra_properties_funcs()
                and base not in RegionAnalyzer.postcomputed_properties
                and base not in seen
            ):
                seen.add(base)
                bases.append(base)
        return sorted(bases)

    def used_extra_properties_funcs(
        self, spacing: Optional[Tuple[float, ...]] = None
    ) -> List[Callable]:
        """Get list of extra property functions that are being used.

        Args:
            spacing: Optional spacing tuple to pass to extra_properties functions.

        Returns:
            List of callable functions for the extra properties being used.
            Functions are wrapped to include spacing if provided.
        """
        uep = self.used_extra_properties()
        base_funcs = {
            k: func
            for k, func in RegionAnalyzer.extra_properties_funcs().items()
            if k in uep
        }

        # Wrap functions to pass spacing if provided
        wrapped_funcs = []
        for prop_name, func in base_funcs.items():
            if spacing is not None:
                # Check if function accepts spacing parameter
                import inspect

                sig = inspect.signature(func)
                if "spacing" in sig.parameters:
                    # Create a named wrapper function that passes spacing
                    # scikit-image calls: func(regionmask, intensity_image)
                    # We need to call: func(regionmask, intensity_image, spacing=spacing)
                    def make_wrapper(f, prop_n, sp):
                        @wraps(f)
                        def wrapper(regionmask, intensity_image=None):
                            return f(regionmask, intensity_image, spacing=sp)

                        # Set the function name to match the property name
                        wrapper.__name__ = prop_n
                        return wrapper

                    wrapped_funcs.append(make_wrapper(func, prop_name, spacing))
                else:
                    # Function doesn't accept spacing, use as-is
                    wrapped_funcs.append(func)
            else:
                wrapped_funcs.append(func)

        return wrapped_funcs

    def used_builtin_properties(self) -> List[str]:
        """Get list of built-in properties that are being used.

        Returns:
            List of built-in property names from config.
        """
        bases: List[str] = []
        seen: set[str] = set()
        builtin = set(RegionAnalyzer.builtin_properties())
        for prop in self.config.properties:
            base = RegionAnalyzer.base_property_name(prop)
            if base in builtin and base not in seen:
                seen.add(base)
                bases.append(base)
        return bases

    @classmethod
    def from_config(cls, config: "RegionAnalyzerConfig") -> "RegionAnalyzer":
        """Create a RegionAnalyzer instance from a configuration.

        Args:
            config: RegionAnalyzer configuration.

        Returns:
            A new RegionAnalyzer instance.
        """
        return cls(config)

    @staticmethod
    def new_id() -> str:
        """Return a new unique id (thread/process safe via :func:`uuid.uuid4`)."""
        return uuid.uuid4().hex

    @classmethod
    def new_object_id(cls) -> str:
        """Return a new unique object id."""
        return cls.new_id()

    @classmethod
    def new_slice_id(cls) -> str:
        """Return a new unique slice id."""
        return cls.new_id()

    @classmethod
    def new_stack_id(cls) -> str:
        """Return a new unique stack id."""
        return cls.new_id()

    def _assign_object_ids(
        self, results: List[Any] | pd.DataFrame
    ) -> List[Any] | pd.DataFrame:
        """Attach a unique ``object_id`` to each region after regionprops."""
        if isinstance(results, list):
            for region in results:
                setattr(region, "object_id", self.new_object_id())
        elif isinstance(results, pd.DataFrame):
            n = len(results)
            if n:
                results = results.copy()
                results["object_id"] = [self.new_object_id() for _ in range(n)]
        return results

    def _set_result_index(self, results: pd.DataFrame) -> pd.DataFrame:
        """Index the result table by :attr:`~RegionAnalyzerConfig.index_on`."""
        index_col = self.config.index_on
        if results.index.name == index_col:
            return results
        if index_col not in results.columns:
            raise ValueError(
                f"Cannot index RegionAnalyzer output on {index_col!r}; "
                f"available columns: {list(results.columns)}"
            )
        return results.set_index(index_col)

    def _assign_stack_and_slice_ids(
        self,
        results: List[Any] | pd.DataFrame,
        stack_id: str,
        slice_id: str,
    ) -> List[Any] | pd.DataFrame:
        """Attach shared ``stack_id`` and ``slice_id`` to each region in a slice."""
        if isinstance(results, list):
            for region in results:
                setattr(region, "stack_id", stack_id)
                setattr(region, "slice_id", slice_id)
        elif isinstance(results, pd.DataFrame):
            n = len(results)
            if n:
                results = results.copy()
                results["stack_id"] = stack_id
                results["slice_id"] = slice_id
        return results

    def _assign_property_names(
        self, results: List[Any] | pd.DataFrame
    ) -> List[Any] | pd.DataFrame:
        """Record configured output property names on list regions for discovery."""
        if not isinstance(results, list):
            return results
        names = set(self.config.properties)
        for region in results:
            for key in region.__dict__:
                if not key.startswith("_"):
                    names.add(key)
        property_names = sorted(names)
        for region in results:
            region.__dict__["_vistiq_property_names"] = property_names
        return results

    def _slice_axis_labels(
        self,
        labels_ndim: int,
        metadata: Optional[dict[str, Any]],
    ) -> List[str]:
        """Axis labels for the slice dimensions passed to regionprops.

        Uses ``metadata['axes']`` (via :func:`~vistiq.utils.axis_labels_from_metadata`)
        and ``iterator_config.slice_def`` to select labels for the analyzed
        sub-array. Falls back to ``axis_0``, ``axis_1``, … when metadata is
        ``None`` or labels cannot be mapped.
        """
        axis_labels = axis_labels_from_metadata(metadata)
        if not axis_labels:
            return [f"axis_{i}" for i in range(labels_ndim)]

        stack_ndim = len(axis_labels)
        slice_def = self.config.iterator_config.slice_def
        if len(slice_def) == 0:
            if labels_ndim <= stack_ndim:
                return list(axis_labels[:labels_ndim])
            return [f"axis_{i}" for i in range(labels_ndim)]

        normalized = tuple(
            axis if axis >= 0 else stack_ndim + axis for axis in slice_def
        )
        mapped = [axis_labels[i] for i in normalized if i < len(axis_labels)]
        if len(mapped) != labels_ndim:
            logger.warning(
                "slice_def maps to %d axis labels but slice has %d dimensions; "
                "falling back to generic axis names",
                len(mapped),
                labels_ndim,
            )
            return [f"axis_{i}" for i in range(labels_ndim)]
        return mapped

    @staticmethod
    def map_dataframe_axis_columns(
        df: pd.DataFrame,
        slice_axis_labels: List[str],
    ) -> pd.DataFrame:
        """Rename ``property-N`` columns using slice axis labels.

        ``bbox`` mins/maxes become ``bbox-start-{axis}`` / ``bbox-end-{axis}``.
        For ``ndim > 2``, ``cross_sectional_area`` and ``aspect_ratio`` tuple
        columns map to plane labels (``cross_sectional_area-xy``, …); the last
        ``aspect_ratio`` component becomes ``aspect_ratio``. Other vector
        properties map ``prop-i`` → ``prop-{axis}``.

        Args:
            df: DataFrame from ``regionprops_table``.
            slice_axis_labels: Labels for each dimension of the analyzed slice.

        Returns:
            DataFrame with renamed columns (unchanged if nothing to rename).
        """
        ndim = len(slice_axis_labels)
        if ndim == 0:
            return df

        axis_names = [str(axis).lower() for axis in slice_axis_labels]
        plane_pairs = RegionAnalyzer.cross_sectional_area_plane_indices(ndim)
        rename: dict[str, str] = {}
        for col in df.columns:
            if "-" not in col:
                continue
            prop, suffix = col.rsplit("-", 1)
            if not suffix.isdigit():
                continue
            i = int(suffix)
            if prop == "bbox":
                if i < ndim:
                    rename[col] = f"bbox-start-{axis_names[i]}"
                elif i < 2 * ndim:
                    rename[col] = f"bbox-end-{axis_names[i - ndim]}"
            elif prop == "cross_sectional_area" and ndim > 2 and i < len(plane_pairs):
                axis_i, axis_j = plane_pairs[i]
                plane = RegionAnalyzer.cross_sectional_area_plane_label(
                    slice_axis_labels, axis_i, axis_j
                )
                rename[col] = f"cross_sectional_area-{plane}"
            elif prop == "aspect_ratio" and ndim > 2:
                if i < len(plane_pairs):
                    axis_i, axis_j = plane_pairs[i]
                    plane = RegionAnalyzer.cross_sectional_area_plane_label(
                        slice_axis_labels, axis_i, axis_j
                    )
                    rename[col] = f"aspect_ratio-{plane}"
                elif i == len(plane_pairs):
                    rename[col] = "aspect_ratio"
            elif i < ndim:
                rename[col] = f"{prop}-{axis_names[i]}"

        if rename:
            df = df.rename(columns=rename)
        return df

    def _assign_slice_annotations(
        self,
        results: List[Any] | pd.DataFrame,
        slice_annotations: Optional[dict[str, Any]],
    ) -> List[Any] | pd.DataFrame:
        """Attach per-slice axis indices as columns (dataframe) or attributes (list).

        Axis keys are normalized to lowercase (``c``, ``z``, …) to match ``map_axes``.
        Values are stored as ``int64`` (dataframe columns) or Python ``int`` (list attr).
        """
        if not slice_annotations or "slice_annotations" not in self.config.properties:
            return results

        normalized = {str(k).lower(): int(v) for k, v in slice_annotations.items()}

        if isinstance(results, pd.DataFrame):
            results = results.copy()
            n = len(results)
            for key, value in normalized.items():
                if n:
                    results[key] = np.full(n, value, dtype=np.int64)
                else:
                    results[key] = pd.Series(dtype=np.int64)
        elif isinstance(results, list):
            for region in results:
                setattr(region, "slice_annotations", dict(normalized))
        return results

    @staticmethod
    def _positive_extent_value(value: Any) -> Any:
        """Return a non-negative scalar or tuple of scalars (area/volume/cross-section)."""
        if isinstance(value, (tuple, list)):
            return type(value)(float(abs(v)) for v in value)
        return float(abs(value))

    @classmethod
    def _ensure_positive_extent_values(
        cls, results: List[Any] | pd.DataFrame
    ) -> List[Any] | pd.DataFrame:
        """Ensure area, volume, and cross-sectional area measurements are non-negative."""
        if isinstance(results, pd.DataFrame):
            results = results.copy()
            for col in results.columns:
                if col in ("area", "volume") or col.startswith("cross_sectional_area"):
                    results[col] = results[col].abs()
            return results

        for region in results:
            for prop in ("area", "volume", "cross_sectional_area"):
                if not hasattr(region, prop):
                    continue
                val = getattr(region, prop)
                if isinstance(val, (tuple, list, np.ndarray)):
                    if prop == "cross_sectional_area":
                        continue
                    val = cls._positive_extent_value(val)
                else:
                    val = cls._positive_extent_value(val)
                region.__dict__[prop] = val
            for key, val in list(region.__dict__.items()):
                if key.startswith("cross_sectional_area-"):
                    region.__dict__[key] = cls._positive_extent_value(val)
        return results

    @staticmethod
    def _relabel_area_as_volume(
        results: List[Any] | pd.DataFrame, labels_ndim: int
    ) -> List[Any] | pd.DataFrame:
        """Rename regionprops ``area`` to ``volume`` when the label slice is 3D+.

        scikit-image ``area`` counts voxels for ``ndim >= 3``; expose it as
        ``volume`` in outputs. No-op for 2D slices or when ``volume`` is already
        present (e.g. from the custom :meth:`volume` extra property).
        """
        if labels_ndim < 3:
            return results

        if isinstance(results, pd.DataFrame):
            if "area" not in results.columns:
                return results
            if "volume" in results.columns:
                return results.drop(columns=["area"])
            return results.rename(columns={"area": "volume"})

        for region in results:
            if hasattr(region, "area"):
                region.__dict__["volume"] = float(abs(region.area))
        return results

    @staticmethod
    def circularity(regionmask, intensity_image=None, spacing=None):
        """Compute circularity: 4π * area / perimeter² (perfect circle = 1.0).

        This function is for 2D regions only. For 3D regions, use sphericity.

        Args:
            regionmask: Binary mask of the region (2D).
            intensity_image: Optional intensity image (not used).
            spacing: Optional spacing tuple for anisotropic voxels (not used for circularity).

        Returns:
            Circularity value (1.0 for perfect circle), or NaN if invalid.
        """
        if regionmask.ndim == 3:
            return float("nan")

        perim = perimeter(regionmask)
        area = np.sum(regionmask)
        if perim > 0:
            return float(4.0 * np.pi * area / (perim**2))
        return float("nan")

    @staticmethod
    def sphericity(regionmask, intensity_image=None, spacing=None):
        """Compute sphericity: π^(1/3) * (6*volume)^(2/3) / surface_area (perfect sphere = 1.0).

        This function is for 3D regions only. For 2D regions, use circularity.

        Args:
            regionmask: Binary mask of the region (3D).
            intensity_image: Optional intensity image (not used).
            spacing: Optional spacing tuple for anisotropic voxels.
                    Used to compute surface area accurately.

        Returns:
            Sphericity value (1.0 for perfect sphere), or NaN if invalid.
        """
        if regionmask.ndim == 2:
            return float("nan")

        volume = np.sum(regionmask)
        if volume == 0:
            return float("nan")

        # Compute surface area using marching cubes
        try:
            # Try different possible import names for marching cubes
            try:
                from skimage.measure import marching_cubes
            except ImportError:
                try:
                    from skimage.measure import marching_cubes_lewiner as marching_cubes
                except ImportError:
                    # If marching_cubes is not available, return NaN
                    return float("nan")

            if spacing is not None and len(spacing) == 3:
                verts, faces, normals, values = marching_cubes(
                    regionmask, spacing=spacing
                )
            else:
                verts, faces, normals, values = marching_cubes(regionmask)

            # Calculate surface area from mesh
            # Surface area is sum of areas of all triangular faces
            if len(faces) == 0:
                return float("nan")

            # Compute area of each triangular face
            face_areas = []
            for face in faces:
                v0, v1, v2 = verts[face]
                # Area = 0.5 * ||(v1-v0) × (v2-v0)||
                cross = np.cross(v1 - v0, v2 - v0)
                area = 0.5 * np.linalg.norm(cross)
                face_areas.append(area)

            surface_area = sum(face_areas)

            if surface_area > 0:
                # Sphericity = π^(1/3) * (6*volume)^(2/3) / surface_area
                sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
                return float(sphericity)
            return float("nan")
        except (ValueError, RuntimeError, ImportError):
            # marching_cubes may fail for degenerate cases or not be available
            return float("nan")

    @staticmethod
    def _aspect_ratio_for_axes(
        regionmask: np.ndarray,
        axis_indices: Tuple[int, ...],
        spacing: Optional[np.ndarray] = None,
    ) -> float:
        """Aspect ratio from the covariance matrix along selected axes."""
        coords = np.where(regionmask)
        if len(coords[0]) == 0 or len(axis_indices) < 2:
            return float("nan")

        coords_rows: List[np.ndarray] = []
        for axis in axis_indices:
            row = coords[axis].astype(np.float64)
            if spacing is not None and spacing.size > axis:
                row = row * float(spacing[axis])
            coords_rows.append(row)

        coords_array = np.array(coords_rows, dtype=np.float64)
        centroid = np.mean(coords_array, axis=1)
        coords_centered = coords_array - centroid[:, np.newaxis]
        if coords_centered.shape[1] < len(axis_indices):
            return float("nan")

        cov = np.cov(coords_centered)
        if np.ndim(cov) < 2:
            return float("nan")

        eigenvalues = np.linalg.eigvals(cov)
        if len(eigenvalues) < 2 or np.any(eigenvalues <= 0):
            return float("nan")

        eigenvalues = np.sort(eigenvalues)[::-1]
        return float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))

    @staticmethod
    def aspect_ratio(regionmask, intensity_image=None, spacing=None):
        """Aspect ratio per orthogonal plane plus an all-axis value (3D+ only).

        For ``ndim > 2`` (e.g. ZYX) returns ``(yz, xz, xy, overall)``. For 2D
        slices returns a single scalar. scikit-image expands 3D+ tuples as
        ``aspect_ratio-0``..``aspect_ratio-N``; the last index is renamed to
        ``aspect_ratio`` when ``map_axes`` is enabled.

        Args:
            regionmask: Binary mask of the region (2D or 3D).
            intensity_image: Optional intensity image (not used).
            spacing: Optional per-axis spacing for the slice dimensions.

        Returns:
            Scalar for 2D, or tuple of in-plane ratios plus all-dimension ratio.
        """
        ndim = regionmask.ndim
        if ndim < 2:
            return float("nan")

        spacing_arr: Optional[np.ndarray] = None
        if spacing is not None:
            spacing_arr = np.abs(np.asarray(spacing[:ndim], dtype=np.float64))

        if ndim == 2:
            return RegionAnalyzer._aspect_ratio_for_axes(
                regionmask, (0, 1), spacing=spacing_arr
            )

        values: List[float] = []
        for axis_i, axis_j in RegionAnalyzer.cross_sectional_area_plane_indices(ndim):
            values.append(
                RegionAnalyzer._aspect_ratio_for_axes(
                    regionmask, (axis_i, axis_j), spacing=spacing_arr
                )
            )
        values.append(
            RegionAnalyzer._aspect_ratio_for_axes(
                regionmask, tuple(range(ndim)), spacing=spacing_arr
            )
        )
        return tuple(values)

    @staticmethod
    def cross_sectional_area_plane_indices(ndim: int) -> List[Tuple[int, int]]:
        """Return in-plane axis index pairs ``(i, j)`` with ``i < j``."""
        return [(i, j) for i in range(ndim) for j in range(i + 1, ndim)]

    @staticmethod
    def cross_sectional_area_plane_label(
        axis_labels: List[str], axis_i: int, axis_j: int
    ) -> str:
        """Build a plane label from two axis names (e.g. ``'xy'`` from Y and X)."""
        names = sorted(
            [str(axis_labels[axis_i]).lower(), str(axis_labels[axis_j]).lower()]
        )
        return "".join(names)

    @staticmethod
    def cross_sectional_area(regionmask, intensity_image=None, spacing=None):
        """Maximum cross-sectional area for each orthogonal plane (3D+ only).

        For each in-plane axis pair, sum the mask within that plane, then take
        the maximum over the reduced array (all dimensions orthogonal to the plane).
        For ``ndim > 2`` (e.g. ZYX) returns one value per plane (yz, xz, xy).
        For 2D slices returns a single scalar. scikit-image expands 3D+ tuples as
        ``cross_sectional_area-0``, etc.; :meth:`map_dataframe_axis_columns`
        renames these when ``map_axes`` is on.

        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            spacing: Optional per-axis spacing for the slice dimensions.

        Returns:
            Scalar for 2D, or tuple of cross-sectional areas per plane.
        """
        ndim = regionmask.ndim
        if ndim < 2:
            return float("nan")

        spacing_arr: Optional[np.ndarray] = None
        if spacing is not None:
            spacing_arr = np.abs(np.asarray(spacing[:ndim], dtype=np.float64))

        if ndim == 2:
            pixel_count = float(np.sum(regionmask))
            if spacing_arr is not None and spacing_arr.size > 1:
                pixel_count *= float(np.abs(spacing_arr[0] * spacing_arr[1]))
            return pixel_count

        areas: List[float] = []
        for axis_i, axis_j in RegionAnalyzer.cross_sectional_area_plane_indices(ndim):
            plane_axes = (axis_i, axis_j)
            projected = np.sum(regionmask, axis=plane_axes)
            pixel_count = float(np.max(projected))
            if spacing_arr is not None and spacing_arr.size > axis_j:
                pixel_count *= float(np.abs(spacing_arr[axis_i] * spacing_arr[axis_j]))
            areas.append(pixel_count)
        return tuple(areas)

    @staticmethod
    def volume(regionmask, intensity_image=None, spacing=None):
        """Compute volume: sum of all pixels/voxels in the region mask.

        This is equivalent to regionprops.area, which computes the number of
        pixels (or voxels for 3D) in the region. For 3D regions, this represents
        volume rather than area.

        If spacing is provided, the volume accounts for anisotropic voxel sizes
        by multiplying the pixel count by the product of spacing values.

        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            spacing: Optional spacing tuple for anisotropic voxels.
                    If provided, volume = pixel_count * product(spacing).

        Returns:
            Volume as a float. If spacing is provided, returns physical volume.
            Otherwise, returns number of pixels/voxels.
        """
        if regionmask.ndim == 2:
            return float("nan")

        pixel_count = float(np.sum(regionmask))

        if spacing is not None:
            spacing_arr = np.abs(np.asarray(spacing[: regionmask.ndim], dtype=np.float64))
            voxel_volume = float(np.prod(spacing_arr))
            return pixel_count * voxel_volume

        return pixel_count

    @staticmethod
    def _channel_names_string(
        channel_names: Optional[Union[List[str], str]],
    ) -> Optional[str]:
        """Format ``metadata['channel_names']`` as a comma-separated string."""
        if channel_names is None:
            return None
        if isinstance(channel_names, str):
            return channel_names or None
        names = [str(name) for name in channel_names if str(name)]
        if not names:
            return None
        return ",".join(names)

    def _assign_channel_names(
        self,
        results: List[Any] | pd.DataFrame,
        channel_names: Optional[Union[List[str], str]],
        channel_col: Optional[str] = "channel",
    ) -> List[Any] | pd.DataFrame:
        """Attach ``metadata['channel_names']`` to each region as *channel_col*."""
        channel_str = self._channel_names_string(channel_names)
        if channel_str is None:
            return results
        if isinstance(results, list):
            for region in results:
                setattr(region, channel_col, channel_str)
        elif isinstance(results, pd.DataFrame) and len(results):
            results = results.copy()
            results[channel_col] = channel_str
        return results

    def _process_slice(
        self,
        labels: np.ndarray,
        slice_annotations: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        stack_id: Optional[str] = None,
        **kwargs,
    ) -> List["RegionProperties"] | pd.DataFrame:
        """Process one label slice and return region measurements.

        Args:
            labels: Labeled array for one iterator step (2D plane or ND sub-volume).
            slice_annotations: Optional axis→index map from the stack iterator.
            metadata: Optional stack metadata; ``scale`` sets spacing, ``axes``
                drives ``map_axes`` renaming. Safe to pass ``None``.
            stack_id: Optional stack identifier; a new id is generated if omitted.
            **kwargs: Forwarded from :class:`StackProcessor` (ignored here).

        Returns:
            List of :class:`RegionProperties` or a DataFrame indexed by
            :attr:`~RegionAnalyzerConfig.index_on`, depending on
            :attr:`RegionAnalyzerConfig.output_type`.

        Raises:
            ValueError: If ``output_type`` is not ``list`` or ``dataframe``.
        """
        if metadata is None or metadata.get("scale", None) is None:
            spacing = None
        else:
            spacing = metadata.get("scale", None)
        if spacing is not None:
            spacing = spacing[-labels.ndim :]
        debug_mask_labels("RegionAnalyzer._process_slice", labels)
        channel_names = metadata.get("channel_names", None) if metadata else None
        logger.info(
            f"RegionAnalyzer: Applying scale: {spacing}, labels.shape={labels.shape}, "
            f"channel_names={channel_names}"
        )

        # Get extra_properties functions with spacing wrapped in
        extra_props_funcs = self.used_extra_properties_funcs(spacing=spacing)

        if self.config.output_type == "list":
            results = regionprops(
                labels, extra_properties=extra_props_funcs, spacing=spacing
            )
            results = self._expand_mapped_property_attributes(
                results, labels.ndim, metadata
            )
        elif self.config.output_type == "dataframe":
            results = pd.DataFrame(
                regionprops_table(
                    labels,
                    properties=self.used_builtin_properties(),
                    extra_properties=extra_props_funcs,
                    spacing=spacing,
                )
            )
            if self.config.map_axes:
                slice_axes = self._slice_axis_labels(labels.ndim, metadata)
                results = self.map_dataframe_axis_columns(results, slice_axes)
        else:
            raise ValueError(
                f"Invalid output type: {self.config.output_type}. Allowed output types are: list, dataframe"
            )

        results = self._relabel_area_as_volume(results, labels.ndim)
        results = self._ensure_positive_extent_values(results)
        results = self._assign_channel_names(results, channel_names)

        stack_id = stack_id or self.new_stack_id()
        slice_id = self.new_slice_id()
        results = self._assign_object_ids(results)
        results = self._assign_stack_and_slice_ids(results, stack_id, slice_id)

        if slice_annotations:
            results = self._assign_slice_annotations(results, slice_annotations)

        if self.config.output_type == "list":
            results = self._assign_property_names(results)
        elif isinstance(results, pd.DataFrame) and len(results):
            results = self._set_result_index(results)

        if isinstance(results, list):
            logger.debug(
                "DEBUG RegionAnalyzer labels:", [r.label for r in results[:10]]
            )
        elif isinstance(results, pd.DataFrame):
            if "label" in results.columns:
                logger.debug(
                    "DEBUG RegionAnalyzer labels:", results["label"].tolist()[:10]
                )
            elif results.index.name == "label":
                logger.debug(
                    "DEBUG RegionAnalyzer labels:", results.index.tolist()[:10]
                )
            else:
                logger.debug("DEBUG RegionAnalyzer result type:", type(results))
        else:
            logger.debug("DEBUG RegionAnalyzer result type:", type(results))

        preview = results.head() if hasattr(results, "head") else results[:5]
        logger.info(
            f"Identified {len(results)} regions, return as {self.config.output_type}"
        )
        return results

    def _reshape_slice_results(
        self,
        results: list[Any],
        slice_indices: list[tuple[int, ...]],
        input_shape: tuple[int, ...],
        output_shape: Optional[tuple[int, ...]] = None,
    ) -> List["RegionProperties"] | pd.DataFrame:
        """Concatenate per-slice tables, preserving :attr:`~RegionAnalyzerConfig.index_on`."""
        if self.config.output_type != "dataframe":
            return super()._reshape_slice_results(
                results,
                slice_indices=slice_indices,
                input_shape=input_shape,
                output_shape=output_shape,
            )

        index_col = self.config.index_on
        frames = []
        for frame in results:
            if hasattr(frame, "index") and frame.index.name == index_col:
                frames.append(frame.reset_index())
            else:
                frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) and index_col in combined.columns:
            combined = combined.set_index(index_col)
        return combined

    def _reshape_slice_results_OBSOLETE(
        self,
        results: list[Any],
        slice_indices: list[tuple[int, ...]],
        input_shape: tuple[int, ...],
    ) -> List["RegionProperties"] | pd.DataFrame:
        """Reshape slice results according to output configuration.
        Args:
            results: List of results from each slice.
            slice_indices: List of index tuples for each slice.
            input_shape: Shape of the input array.

        Returns:
            Reshaped results according to output_type.
        """
        return super()._reshape_slice_results(
            results, slice_indices=slice_indices, input_shape=input_shape
        )

    @task(name="RegionAnalyzer.run")
    def run(
        self,
        labels: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> List["RegionProperties"] | pd.DataFrame:
        """Analyze labeled regions over the stack iterator.

        Args:
            labels: Labeled array (any dimensionality supported by the iterator).
            metadata: Optional stack metadata (spacing, axes, etc.). May be ``None``.
            **kwargs: Passed to :class:`StackProcessor` (``workers``, ``verbose``,
                optional ``stack_id``).

        Returns:
            Region properties as a list or DataFrame per ``output_type``.
        """
        logger.debug("DEBUG: entered RegionAnalyzer.run")
        logger.debug("DEBUG: labels shape =", getattr(labels, "shape", None))
        # debug_mask_labels("RegionAnalyzer.run", labels)
        stack_id = kwargs.pop("stack_id", None) or self.new_stack_id()
        results, _ = super().run(
            labels, metadata=metadata, stack_id=stack_id, **kwargs
        )
        logger.debug(f"RegionAnalyzer.run(): Results = {results}")
        return results


class RegionAnalyzerConfig(StackProcessorConfig):
    """Configuration for :class:`RegionAnalyzer`.

    Inherits stack iteration settings from :class:`~vistiq.core.StackProcessorConfig`
    (``iterator_config``, ``batch_size``, ``preferred_backend``, etc.).

    Attributes:
        output_type: ``"list"`` returns :class:`RegionProperties` objects (for
            filtering and downstream Python code). ``"dataframe"`` returns a
            pandas table indexed by :attr:`index_on`.
        index_on: Column to use as the DataFrame index when ``output_type`` is
            ``"dataframe"``. ``"label"`` (default) or ``"object_id"``.
        properties: Names to compute. Built-in scikit-image properties, custom
            extras (:meth:`RegionAnalyzer.extra_properties_funcs`), postcomputed
            ``slice_annotations``, and mapped identifiers (``centroid-y``,
            ``cross_sectional_area-xy``, ``bbox-end-z``, …) are accepted.
            ``label``, ``object_id``, ``slice_id``, and ``stack_id`` are always
            injected (:meth:`RegionAnalyzer.ensure_mandatory_properties`).
        map_axes: When ``True`` and ``output_type="dataframe"``, apply
            :meth:`RegionAnalyzer.map_dataframe_axis_columns` after
            ``regionprops_table``. List output ignores this flag — list mapped
            names via ``properties`` instead.
        expand_coordinates: Reserved for future coordinate expansion; not
            implemented yet.
        iterator_config: :class:`~vistiq.utils.ArrayIteratorConfig` defining
            ``slice_def``. ``None`` or ``()`` analyzes the whole volume.
            Length-1 ``slice_def`` is invalid. Dimensionality also drives
            automatic ``area``↔``volume`` and ``circularity``↔``sphericity``
            substitution (:meth:`validate_properties_iterator`).

    Note:
        ``area``/``volume`` and ``circularity``/``sphericity`` are swapped
        automatically when ``slice_def`` implies 2D vs 3D slice geometry.
    """

    output_type: Literal["list", "dataframe"] = "list"
    index_on: Literal["object_id", "label"] = "label"
    properties: List[str] = Field(
        default_factory=lambda: list(RegionAnalyzer.default_properties)
    )
    map_axes: Optional[bool] = False
    expand_coordinates: Optional[bool] = False

    @field_validator("iterator_config")
    @classmethod
    def validate_iterator_config(cls, v: ArrayIteratorConfig) -> ArrayIteratorConfig:
        """Validate ``slice_def``: ``None``, ``()`` (whole volume), or length >= 2."""
        slice_def = v.slice_def
        if slice_def is None or len(slice_def) == 0:
            return v
        if len(slice_def) == 1:
            raise ValueError(
                "iterator_config.slice_def must be None, empty () for whole-volume "
                "analysis, or keep at least two axes; "
                f"got {slice_def!r} (length 1)"
            )
        return v

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, v: List[str]) -> List[str]:
        """Validate property names and ensure mandatory outputs are present.

        Args:
            v: List of property names to validate.

        Returns:
            Validated list including mandatory properties.

        Raises:
            ValueError: If any property is not in the allowed list.
        """
        if v is None or len(v) == 0:
            v = list(RegionAnalyzer.default_properties)
        else:
            invalid = [
                prop
                for prop in v
                if not RegionAnalyzer.is_allowed_property_name(prop)
            ]
            if invalid:
                raise ValueError(
                    f"One or more invalid properties: {invalid}. "
                    f"Use names from {RegionAnalyzer.allowed_properties()} "
                    f"or mapped axis columns (e.g. 'cross_sectional_area-xy')."
                )
        return RegionAnalyzer.ensure_mandatory_properties(v)

    @model_validator(mode="after")
    def validate_properties_iterator(self) -> "RegionAnalyzerConfig":
        """Adjust area/volume and circularity/sphericity for slice dimensionality.

        When ``slice_def`` spans three or more axes, ``area`` is replaced by
        ``volume`` and ``circularity`` by ``sphericity`` (and vice versa for
        lower-dimensional slices). Whole-volume analysis (``slice_def=()``) is
        handled at runtime in :meth:`RegionAnalyzer._relabel_area_as_volume`
        based on the actual label slice dimensionality.

        Returns:
            Validated configuration.
        """
        props = (
            list(self.properties)
            if self.properties
            else list(RegionAnalyzer.default_properties)
        )

        has_area = "area" in props
        has_volume = "volume" in props
        slice_def = self.iterator_config.slice_def
        slice_def_len = len(slice_def) if slice_def is not None else 0

        if has_area and slice_def_len >= 3:
            props = [p for p in props if p != "area"] + ["volume"]
        if has_volume and slice_def_len < 3:
            props = [p for p in props if p != "volume"] + ["area"]

        has_circularity = "circularity" in props
        has_sphericity = "sphericity" in props

        if has_circularity and slice_def_len >= 3:
            props = [p for p in props if p != "circularity"] + ["sphericity"]
        if has_sphericity and slice_def_len < 3:
            props = [p for p in props if p != "sphericity"] + ["circularity"]

        props = RegionAnalyzer.ensure_mandatory_properties(props)
        if props != self.properties:
            object.__setattr__(self, "properties", props)
        return self
