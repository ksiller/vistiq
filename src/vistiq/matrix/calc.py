"""Pairwise matrix calculators over two aligned collections."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
from prefect import task

from vistiq.core import Configurable, Configuration, generate_name
from vistiq.matrix.types import ArrayBackend, MatrixData
from vistiq.utils import convert_array_like, resolve_torch_device

logger = logging.getLogger(__name__)


class MatrixCalculatorConfig(Configuration):
    """Configuration for :class:`MatrixCalculator`."""

    preferred_input_type: ArrayBackend = "np.ndarray"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class MatrixCalculator(Configurable[MatrixCalculatorConfig]):
    """Base class for pairwise matrix calculations over two point collections."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, config: MatrixCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixCalculatorConfig) -> "MatrixCalculator":
        return cls(config)

    def _calculate(
        self,
        points1: Union[torch.Tensor, np.ndarray],
        points2: Union[torch.Tensor, np.ndarray],
        *,
        device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Compute a pairwise matrix between two point collections."""
        raise NotImplementedError("Subclasses must implement this method")

    def _convert_input(
        self,
        points1: Union[torch.Tensor, np.ndarray],
        points2: Union[torch.Tensor, np.ndarray],
        *,
        device: Optional[torch.device] = None,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """Normalize point arrays to the configured backend."""
        dtype = self.config.preferred_input_type
        return (
            convert_array_like(points1, dtype=dtype, device=device),
            convert_array_like(points2, dtype=dtype, device=device),
        )

    def _apply_spacing(
        self,
        points1: Union[torch.Tensor, np.ndarray],
        points2: Union[torch.Tensor, np.ndarray],
        *,
        spacing: Optional[tuple[float, ...]] = None,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """Scale point coordinates by per-axis *spacing*."""
        if spacing is None:
            return points1, points2
        if points1.ndim != 2 or points2.ndim != 2:
            raise ValueError(
                "points1 and points2 must be 2-D (n_points, n_dims) when spacing is set"
            )
        if len(spacing) != points1.shape[1] or len(spacing) != points2.shape[1]:
            raise ValueError(
                f"Spacing length ({len(spacing)}) must match point dimension "
                f"({points1.shape[1]})"
            )
        spacing_arr = convert_array_like(spacing, dtype="np.ndarray")
        if isinstance(points1, torch.Tensor):
            spacing_scale = convert_array_like(
                spacing_arr,
                dtype="torch.Tensor",
                device=points1.device,
            ).to(dtype=points1.dtype)
            points1 = points1 * spacing_scale.unsqueeze(0)
            points2 = points2 * spacing_scale.unsqueeze(0)
        else:
            points1 = points1 * spacing_arr[np.newaxis, :]
            points2 = points2 * spacing_arr[np.newaxis, :]
        return points1, points2

    @task(name="MatrixCalculator.run", task_run_name=generate_name)
    def run(
        self,
        points1: Union[torch.Tensor, np.ndarray],
        points2: Union[torch.Tensor, np.ndarray],
        spacing: Optional[tuple[float, ...]] = None,
        point_annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> MatrixData:
        """Perform matrix calculation on two collections of points."""
        device = resolve_torch_device(
            device,
            preferred_input_type=self.config.preferred_input_type,
            preferred_device=self.config.preferred_device,
        )
        points1, points2 = self._convert_input(points1, points2, device=device)
        points1, points2 = self._apply_spacing(points1, points2, spacing=spacing)
        logger.info(
            f"MatrixCalculator.run: device={device}, points1.shape={points1.shape}, points1.dtype={getattr(points1, 'dtype', type(points1))}, "
            f"points2.shape={points2.shape}, points2.dtype={getattr(points2, 'dtype', type(points2))}, spacing={spacing}"
        )
        raw_results = self._calculate(points1, points2, device=device)
        annotations: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        if point_annotations is not None:
            rows, cols = point_annotations
            if isinstance(raw_results, torch.Tensor):
                shape = tuple(raw_results.shape)
            else:
                shape = np.asarray(raw_results).shape
            if len(rows) != shape[0] or len(cols) != shape[1]:
                raise ValueError(
                    "point_annotations must have the same number of rows and "
                    "columns as the results"
                )
            annotations = (
                tuple(str(value) for value in rows),
                tuple(str(value) for value in cols),
            )
        return MatrixData(matrix=raw_results, annotations=annotations)
