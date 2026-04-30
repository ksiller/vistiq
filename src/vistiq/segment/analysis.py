import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import field_validator, model_validator
from prefect import task
from skimage.measure import label as sk_label, regionprops, regionprops_table

from vistiq.core import StackProcessor, StackProcessorConfig
from vistiq.segment._debug import debug_mask_labels

logger = logging.getLogger(__name__)


class RegionAnalyzer(StackProcessor):
    """Analyzer that extracts region properties from labeled images.

    Computes properties for each labeled region, including built-in properties
    from scikit-image and custom extra properties like circularity and aspect ratio.
    """

    default_properties: List[str] = ["label", "centroid"]

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
        )

    def used_extra_properties(self) -> List[str]:
        """Get list of extra properties that are being used.

        Returns:
            List of extra property names from config that are custom properties.
        """
        return sorted(
            [
                prop
                for prop in self.config.properties
                if prop in RegionAnalyzer.extra_properties_funcs().keys()
            ]
        )

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
        return [
            prop
            for prop in self.config.properties
            if prop in RegionAnalyzer.builtin_properties()
        ]

    @classmethod
    def from_config(cls, config: "RegionAnalyzerConfig") -> "RegionAnalyzer":
        """Create a RegionAnalyzer instance from a configuration.

        Args:
            config: RegionAnalyzer configuration.

        Returns:
            A new RegionProcessor instance.
        """
        return cls(config)

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
        from skimage.measure import perimeter

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
    def aspect_ratio(regionmask, intensity_image=None, spacing=None):
        """Compute aspect ratio: minor_axis_length / major_axis_length.

        Computes aspect ratio from the covariance matrix of region coordinates.
        Works for both 2D and 3D regions. The aspect ratio is the ratio of the
        smallest to largest eigenvalue of the covariance matrix.

        If spacing is provided, coordinates are scaled by spacing to account for
        anisotropic voxels.

        Args:
            regionmask: Binary mask of the region (2D or 3D).
            intensity_image: Optional intensity image (not used).
            spacing: Optional spacing tuple for anisotropic voxels.

        Returns:
            Aspect ratio (minor/major axis), or NaN if invalid.
        """
        coords = np.where(regionmask)
        if len(coords[0]) == 0:
            return float("nan")

        ndim = regionmask.ndim

        # Create coordinate array for all dimensions
        coords_array = np.array([coords[i] for i in range(ndim)], dtype=np.float64)

        # Scale coordinates by spacing if provided (broadcasting)
        if spacing is not None and len(spacing) >= ndim:
            spacing_array = np.array(spacing[:ndim], dtype=np.float64)
            coords_array = coords_array * spacing_array[:, np.newaxis]

        centroid = np.mean(coords_array, axis=1)
        coords_centered = coords_array - centroid[:, np.newaxis]

        if coords_centered.shape[1] < ndim:
            return float("nan")

        cov = np.cov(coords_centered)
        eigenvalues = np.linalg.eigvals(cov)
        if len(eigenvalues) < ndim or np.any(eigenvalues <= 0):
            return float("nan")

        # Sort eigenvalues (largest first)
        eigenvalues = np.sort(eigenvalues)[::-1]
        # Aspect ratio: smallest / largest eigenvalue
        return float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))

    @staticmethod
    def cross_sectional_area(regionmask, intensity_image=None, spacing=None):
        """Compute the maximum cross-sectional area in the xy plane of the region.

        Args:
            regionmask: Binary mask of the region.
            intensity_image: Optional intensity image (not used).
            spacing: Optional spacing tuple for anisotropic voxels.
        Returns:
            Maximum cross-sectional area as a float.
        """
        # Sum along the last two spatial dimensions (typically Y and X)
        pixel_count = float(np.max(np.sum(regionmask, axis=(-2, -1))))

        if spacing is not None:
            # Calculate physical area accounting for pixel spacing
            pixel_area = np.abs(np.prod(spacing[-2]))
            return pixel_count * pixel_area

        return pixel_count

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
        pixel_count = float(np.sum(regionmask))

        if spacing is not None:
            # Calculate physical volume accounting for voxel spacing
            voxel_volume = np.abs(np.prod(spacing))
            return pixel_count * voxel_volume

        return pixel_count

    def _process_slice(
        self, labels: np.ndarray, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> List["RegionProperties"] | pd.DataFrame:
        """Process a single slice to extract region properties.

        Args:
            labels: Labeled array slice.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Either a list of RegionProperties or a pandas DataFrame, depending
            on output_type configuration.

        Raises:
            ValueError: If output_type is invalid.
        """
        if metadata is None or metadata.get("scale", None) is None:
            spacing = None
        else:
            spacing = metadata.get("scale", None)
        if spacing is not None:
            spacing = spacing[-labels.ndim :]
        debug_mask_labels("RegionAnalyzer._process_slice", labels)
        logger.info(f"RegionAnalyzer: Applying scale: {spacing}")

        # Get extra_properties functions with spacing wrapped in
        extra_props_funcs = self.used_extra_properties_funcs(spacing=spacing)

        if self.config.output_type == "list":
            results = regionprops(
                labels, extra_properties=extra_props_funcs, spacing=spacing
            )
        elif self.config.output_type == "dataframe":
            results = pd.DataFrame(
                regionprops_table(
                    labels,
                    properties=self.used_builtin_properties(),
                    extra_properties=extra_props_funcs,
                    spacing=spacing,
                )
            ).set_index("label")
        else:
            raise ValueError(
                f"Invalid output type: {self.config.output_type}. Allowed output types are: list, dataframe"
            )

        if isinstance(results, list):
            logger.debug(
                "DEBUG RegionAnalyzer labels:", [r.label for r in results[:10]]
            )
        elif hasattr(results, "columns") and "label" in results.columns:
            logger.debug("DEBUG RegionAnalyzer labels:", results["label"].tolist()[:10])
        else:
            logger.debug("DEBUG RegionAnalyzer result type:", type(results))

        preview = results.head() if hasattr(results, "head") else results[:5]
        logger.info(
            f"Identified {len(results)} regions, return as {self.config.output_type}"
        )
        return results

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
        workers: int = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> List["RegionProperties"] | pd.DataFrame:
        """Run the region analyzer on a labeled array.

        Args:
            labels: Labeled array to analyze.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.

        Returns:
            Region properties as list or DataFrame, depending on output_type.
        """
        logger.debug("DEBUG: entered RegionAnalyzer.run")
        logger.debug("DEBUG: labels shape =", getattr(labels, "shape", None))
        debug_mask_labels("RegionAnalyzer.run", labels)
        results = super().run(
            labels, workers=workers, verbose=verbose, metadata=metadata, **kwargs
        )
        logger.debug(f"RegionAnalyzer.run(): Results = {results}")
        return results[0]


class RegionAnalyzerConfig(StackProcessorConfig):
    """Configuration for region analysis operations.

    Attributes:
        output_type: Output format ("list" for RegionProperties list, "dataframe" for pandas DataFrame).
        properties: List of property names to compute. "label" is always included.
    """

    output_type: Literal["list", "dataframe"] = "list"
    properties: List[str] = RegionAnalyzer.default_properties

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, v: List[str]) -> List[str]:
        """Validate that all properties are allowed and include "label".

        Args:
            v: List of property names to validate.

        Returns:
            Validated list with "label" added if missing.

        Raises:
            ValueError: If any property is not in the allowed list.
        """
        if v is None or len(v) == 0:
            return RegionAnalyzer.default_properties
        elif not set(v).issubset(set(RegionAnalyzer.allowed_properties())):
            raise ValueError(
                f"One or more invalid properties: {v}. Allowed properties are: {RegionAnalyzer.allowed_properties()}"
            )
        if "label" not in v:
            v = ["label"] + v
        return v

    @model_validator(mode="after")
    def validate_properties_iterator(self) -> "RegionAnalyzerConfig":
        """Validate properties based on iterator configuration.

        Ensures mutually exclusive properties are handled correctly:
        - "area" and "volume": If iterator_config.slice_def has length < 3: use "area", otherwise use "volume"
        - "circularity" and "sphericity": If iterator_config.slice_def has length < 3: use "circularity", otherwise use "sphericity"

        Returns:
            Validated configuration.
        """
        if self.properties is None or len(self.properties) == 0:
            self.properties = RegionAnalyzer.default_properties

        # Check if both "area" and "volume" are present
        has_area = "area" in self.properties
        has_volume = "volume" in self.properties

        slice_def_len = len(self.iterator_config.slice_def)

        # Handle area/volume mutual exclusivity
        if has_area and slice_def_len >= 3:
            self.properties = [p for p in self.properties if p != "area"] + ["volume"]
        if has_volume and slice_def_len < 3:
            self.properties = [p for p in self.properties if p != "volume"] + ["area"]

        # Handle circularity/sphericity mutual exclusivity
        has_circularity = "circularity" in self.properties
        has_sphericity = "sphericity" in self.properties

        if has_circularity and slice_def_len >= 3:
            self.properties = [p for p in self.properties if p != "circularity"] + [
                "sphericity"
            ]
        if has_sphericity and slice_def_len < 3:
            self.properties = [p for p in self.properties if p != "sphericity"] + [
                "circularity"
            ]

        return self
