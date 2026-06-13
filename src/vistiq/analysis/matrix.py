"""Unary matrix operations on analysis results (e.g. distance matrices)."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import numpy as np
import torch
from prefect import task
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vistiq.constant.matrix import FULL
from vistiq.core import Configurable, Configuration, generate_name
from vistiq.utils import convert_array_like, prepare_matrix_values, resolve_torch_device

logger = logging.getLogger(__name__)

def group_matrix_indices(matrix, threshold=0.5):
    """Group row/column indices with pairwise overlap > threshold."""
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    # Fill upper triangle from lower (self-comparison matrices are often lower-tri)
    sym = np.array(m, copy=True)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sym[i, j], sym[j, i]
            if np.isnan(a) and np.isnan(b):
                continue
            sym[i, j] = sym[j, i] = np.nanmax([a, b])
    adj = (sym > threshold) & ~np.eye(n, dtype=bool)
    _, labels = connected_components(csr_matrix(adj.astype(int)), directed=False)
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, []).append(idx)
    return [sorted(g) for g in groups.values()]

class MatrixAggregatorConfig(Configuration):
    """Configuration for :class:`MatrixAggregator`.

    Attributes:
        operation: One of ``"min"``, ``"max"``, ``"mean"``, ``"sum"``,
            ``"median"``, or ``"count"``.
        axis: Axis along which to aggregate (required).
        ignore_nan: When ``True``, NaN entries are excluded from aggregation.
        triangle: Bitmask of selectable regions on square 2-D matrices; see
            :mod:`vistiq.constant.matrix`.
        preferred_input_type: Backend for :meth:`MatrixAggregator.run`.
        preferred_device: Torch device when ``preferred_input_type`` is
            ``"torch.Tensor"``; ``None`` selects automatically.
    """

    operation: Literal["min", "max", "mean", "sum", "median", "count"] = "mean"
    axis: Optional[int] = 0
    ignore_nan: bool = True
    triangle: int = FULL
    preferred_input_type: Literal["numpy", "torch.Tensor"] = "torch.Tensor"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None


class MatrixAggregator(Configurable[MatrixAggregatorConfig]):
    """Aggregate values in a matrix along a configured axis."""

    def __init__(self, config: MatrixAggregatorConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixAggregatorConfig) -> "MatrixAggregator":
        return cls(config)

    @task(name="MatrixAggregator.run", task_run_name=generate_name)
    def run(
        self,
        data: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Reduce *data* along :attr:`~MatrixAggregatorConfig.axis`."""
        as_numpy = isinstance(data, np.ndarray)
        if (
            device is None
            and self.config.preferred_device is None
            and isinstance(data, torch.Tensor)
        ):
            device = data.device
        else:
            device = resolve_torch_device(
                device,
                preferred_input_type=self.config.preferred_input_type,
                preferred_device=self.config.preferred_device,
            )
        values = convert_array_like(
            data,
            dtype=self.config.preferred_input_type,
            device=device,
        )
        if not isinstance(values, torch.Tensor):
            values = convert_array_like(values, dtype="torch.Tensor", device=device)
        result = self._aggregate(values)
        if as_numpy or self.config.preferred_input_type == "numpy":
            return result.detach().cpu().numpy()
        return result

    def _resolve_axis(self, values: torch.Tensor) -> int:
        axis = self.config.axis
        if axis is None:
            raise ValueError("MatrixAggregatorConfig.axis must be set")
        if isinstance(axis, tuple):
            raise ValueError(
                f"MatrixAggregator does not support tuple axis {axis!r}"
            )
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError(
                f"axis {self.config.axis!r} is out of bounds for ndim={values.ndim}"
            )
        return axis

    def _prepare_values(
        self, values: torch.Tensor, exclude: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_matrix_values(
            values,
            exclude,
            ignore_nan=self.config.ignore_nan,
            triangle=self.config.triangle,
        )

    def _aggregate(self, values: torch.Tensor) -> torch.Tensor:
        axis = self._resolve_axis(values)
        operation = self.config.operation
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        nan = torch.tensor(float("nan"), dtype=values.dtype, device=values.device)

        if operation == "count":
            _, valid = self._prepare_values(values, zero)
            return valid.sum(dim=axis)

        if operation == "sum":
            prepared, _ = self._prepare_values(values, zero)
            return prepared.sum(dim=axis)

        if operation == "mean":
            prepared, valid = self._prepare_values(values, zero)
            counts = valid.sum(dim=axis)
            sums = prepared.sum(dim=axis)
            return torch.where(counts > 0, sums / counts.to(values.dtype), nan)

        if operation == "min":
            fill = torch.full((), float("inf"), dtype=values.dtype, device=values.device)
            prepared, valid = self._prepare_values(values, fill)
            result = torch.min(prepared, dim=axis).values
            return torch.where(valid.any(dim=axis), result, nan)

        if operation == "max":
            fill = torch.full((), float("-inf"), dtype=values.dtype, device=values.device)
            prepared, valid = self._prepare_values(values, fill)
            result = torch.max(prepared, dim=axis).values
            return torch.where(valid.any(dim=axis), result, nan)

        if operation == "median":
            prepared, valid = self._prepare_values(values, nan)
            result = torch.nanmedian(prepared, dim=axis).values
            return torch.where(valid.any(dim=axis), result, nan)

        raise ValueError(f"Invalid operation: {operation}")
