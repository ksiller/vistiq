import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from prefect import task
from pydantic import model_validator

from vistiq.matrix.types import MatrixData
from vistiq.core import Configuration, Configurable, generate_name
from vistiq.utils import convert_array_like, resolve_torch_device

logger = logging.getLogger(__name__)


class MatrixCalculatorConfig(Configuration):
    """Configuration for :class:`MatrixCalculator`."""

    preferred_input_type: Literal["numpy", "torch.Tensor"] = "numpy"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class MatrixCalculator(Configurable):
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
        points1: Union["torch.Tensor", np.ndarray],
        points2: Union["torch.Tensor", np.ndarray],
        spacing: Optional[Union[dict[str, float], tuple[float, ...]]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> Union["torch.Tensor", np.ndarray]:
        """Compute a pairwise matrix between two point collections.

        Args:
            points1: First collection of points.
            points2: Second collection of points.
            spacing: Spacing applied in :meth:`_convert_input`, if provided.
            device: Resolved torch device from :meth:`run`.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _convert_input(
        self,
        points1: Union["torch.Tensor", np.ndarray],
        points2: Union["torch.Tensor", np.ndarray],
        *,
        spacing: Optional[tuple[float, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[
        Union["torch.Tensor", np.ndarray],
        Union["torch.Tensor", np.ndarray],
        Optional[np.ndarray],
    ]:
        """Normalize point arrays to the configured backend and apply spacing."""
        if spacing is not None:
            if points1.ndim != 2 or points2.ndim != 2:
                raise ValueError(
                    "points1 and points2 must be 2-D (n_points, n_dims) when spacing is set"
                )
            if len(spacing) != points1.shape[1] or len(spacing) != points2.shape[1]:
                raise ValueError(
                    f"Spacing length ({len(spacing)}) must match point dimension "
                    f"({points1.shape[1]})"
                )

        dtype = self.config.preferred_input_type
        points1 = convert_array_like(points1, dtype=dtype, device=device)
        points2 = convert_array_like(points2, dtype=dtype, device=device)
        spacing_arr: Optional[np.ndarray] = None
        if spacing is not None:
            spacing_arr = convert_array_like(spacing, dtype="numpy")
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
        return points1, points2, spacing_arr

    @task(name="MatrixCalculator.run", task_run_name=generate_name)
    def run(
        self,
        points1: Union["torch.Tensor", np.ndarray],
        points2: Union["torch.Tensor", np.ndarray],
        spacing: Optional[tuple[float, ...]] = None,
        point_annotations: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> MatrixData:
        """Perform matrix calculation on two collections of points.

        Args:
            points1: First collection of points.
            points2: Second collection of points.
            spacing: Per-axis scale applied before calculation.
            point_annotations: Optional row/column labels stored on the result.
            device: Optional runtime torch device override. When omitted,
                :attr:`MatrixCalculatorConfig.preferred_device` is resolved
                via :func:`~vistiq.utils.resolve_torch_device`.
        """
        device = resolve_torch_device(
            device,
            preferred_input_type=self.config.preferred_input_type,
            preferred_device=self.config.preferred_device,
        )
        points1, points2, spacing_arr = self._convert_input(
            points1, points2, spacing=spacing, device=device
        )
        logger.info(
            f"MatrixCalculator.run: device={device}, points1.shape={points1.shape}, points1.dtype={getattr(points1, 'dtype', type(points1))}, "
            f"points2.shape={points2.shape}, points2.dtype={getattr(points2, 'dtype', type(points2))}, spacing={spacing}"
        )
        raw_results = self._calculate(
            points1, points2, spacing=spacing_arr, device=device
        )
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
            annotations = (tuple(str(value) for value in rows), tuple(str(value) for value in cols))
        return MatrixData(matrix=raw_results, annotations=annotations)


class DistanceCalculatorConfig(MatrixCalculatorConfig):
    """Configuration for :class:`DistanceCalculator`."""

    method: Literal["euclidean", "hamming", "chebyshev", "minkowski"] = "euclidean"
    minkowski_p: float = 3.0
    preferred_input_type: Literal["numpy", "torch.Tensor"] = "torch.Tensor"

    @model_validator(mode="after")
    def _require_torch_input(self) -> "DistanceCalculatorConfig":
        if self.preferred_input_type != "torch.Tensor":
            raise ValueError(
                "DistanceCalculator requires preferred_input_type='torch.Tensor'"
            )
        return self


class DistanceCalculator(MatrixCalculator):
    """Pairwise distance matrix calculator backed by :func:`torch.cdist`."""

    _CDIST_P_BY_METHOD: dict[str, float] = {
        "euclidean": 2.0,
        "hamming": 0.0,
        "chebyshev": float("inf"),
    }

    def __init__(self, config: DistanceCalculatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: DistanceCalculatorConfig) -> "DistanceCalculator":
        return cls(config)

    def _cdist_p(self) -> float:
        """Map configured method name to ``torch.cdist``'s numeric ``p`` argument."""
        if self.config.method == "minkowski":
            return self.config.minkowski_p
        return self._CDIST_P_BY_METHOD[self.config.method]

    def _calculate(
        self,
        points1: Union["torch.Tensor", np.ndarray],
        points2: Union["torch.Tensor", np.ndarray],
        spacing: Optional[Union[dict[str, float], tuple[float, ...]]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> "torch.Tensor":
        """Compute pairwise distances with PyTorch."""
        if not isinstance(points1, torch.Tensor):
            points1 = convert_array_like(
                points1,
                dtype="torch.Tensor",
                device=device,
            )
        if not isinstance(points2, torch.Tensor):
            points2 = convert_array_like(
                points2,
                dtype="torch.Tensor",
                device=device,
            )
        distances = torch.cdist(points1, points2, p=self._cdist_p())
        logger.info("DistanceCalculator._calculate: distances.shape=%s", distances.shape)
        return distances
