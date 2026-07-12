"""Torch-backed filters for selecting entries from labeled matrices."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from prefect import task

from vistiq.core import Configurable, Configuration, generate_name
from vistiq.matrix.mask import prepare_matrix_values
from vistiq.matrix.types import ArrayBackend, FULL, MatrixData
from vistiq.utils import convert_array_like, resolve_torch_device

if TYPE_CHECKING:
    from skimage.measure import RegionProperties

logger = logging.getLogger(__name__)


def _value_column_count(data: Union[np.ndarray, torch.Tensor]) -> int:
    """Return the number of value columns represented by *data*."""
    if data.ndim <= 1:
        return 1
    return int(data.shape[-1])


class MatrixFilterConfig(Configuration):
    """Shared settings for torch-backed matrix filters.

    Attributes:
        attribute: Optional attribute selector (reserved for tabular inputs).
        axis: Optional axis tuple (reserved for axis-aware filters).
        strict: When ``True``, :meth:`MatrixFilter.run` raises if attribute
            count does not match data width.
        ignore_nan: When ``True``, NaN entries are excluded from selection.
        triangle: Bitmask of selectable regions on square 2-D matrices
            (row index = ``i``, column index = ``j``). Atomic flags combine with
            bitwise OR; named presets are sums of atoms:

            - ``DIAGONAL`` (``1``): ``i == j``
            - ``LOWER_ND`` (``2``): ``i > j``
            - ``UPPER_ND`` (``4``): ``i < j``
            - ``LOWER`` (``3``): ``DIAGONAL | LOWER_ND`` → ``i >= j``
            - ``UPPER`` (``5``): ``DIAGONAL | UPPER_ND`` → ``i <= j``
            - ``OFF_DIAGONAL`` (``6``): ``LOWER_ND | UPPER_ND`` → ``i != j``
            - ``FULL`` (``7``): ``DIAGONAL | LOWER_ND | UPPER_ND`` → entire matrix
        output: Result shape from :meth:`MatrixFilter.run` / :meth:`MatrixFilter.apply`:

            - ``"indices"`` — integer index coordinates (default).
            - ``"values"`` — selected matrix entries as a 1-D
              :class:`torch.Tensor`.
            - ``"mask"`` — boolean :class:`torch.Tensor` with ``True`` at
              selected positions.
            - ``"masked_values"`` — same shape as the input matrix; selected
              entries keep their value, all other entries are ``NaN``.
    """

    attribute: Optional[Union[str, List[str]]] = None
    axis: Optional[Union[int, tuple[int, ...]]] = None
    strict: bool = True
    preferred_input_type: Literal["torch.Tensor"] = "torch.Tensor"
    preferred_device: Optional[Literal["cuda", "mps", "cpu"]] = None
    ignore_nan: bool = True
    triangle: int = FULL
    output: Literal["indices", "values", "mask", "masked_values"] = "indices"

    def attribute_list(self) -> List[str]:
        """Return configured attribute selector(s) as a list."""
        attribute = self.attribute
        if attribute is None:
            return []
        return attribute if isinstance(attribute, list) else [attribute]


class MatrixFilter(Configurable[MatrixFilterConfig]):
    """Base class for filters that operate on :class:`torch.Tensor` matrices.

    Subclasses implement :meth:`accept_indices` against tensors already
    converted via :meth:`_convert_input`. Shared masking for NaNs and triangle
    regions lives here.
    """

    def __init__(self, config: MatrixFilterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: MatrixFilterConfig) -> "MatrixFilter":
        return cls(config)

    def _convert_input(
        self,
        data: Union[np.ndarray, torch.Tensor, List[float], pd.Series, pd.DataFrame],
        *,
        dtype: ArrayBackend = "torch.Tensor",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Normalize supported containers to a torch tensor."""
        if isinstance(data, torch.Tensor):
            return convert_array_like(data, dtype=dtype, device=device)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif isinstance(data, pd.Series):
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data)
        return convert_array_like(data, dtype=dtype, device=device)

    def apply(self, values: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        """Select from *values* and return the configured output."""
        return self._format_output(values, self.accept_indices(values))

    def _wrap_matrix_data_output(
        self,
        data: MatrixData,
        coords: np.ndarray,
        raw: Union[np.ndarray, torch.Tensor],
    ) -> Union[MatrixData, np.ndarray, torch.Tensor]:
        """Attach annotations when filtering :class:`MatrixData` input."""
        from vistiq.matrix.types import (
            annotations_at_coords,
            composite_matrix_annotations,
        )

        output = self.config.output
        if output in ("masked_values", "mask"):
            return MatrixData(matrix=raw, annotations=data.annotations)
        if output == "indices":
            return raw
        if output == "values":
            if data.annotations is None:
                return MatrixData(matrix=raw, annotations=None)
            projected = annotations_at_coords(data.annotations, coords)
            return MatrixData(
                matrix=raw,
                annotations=composite_matrix_annotations(projected),
            )
        return raw

    def _format_output(
        self, values: torch.Tensor, indices: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        """Map raw index coordinates to the configured output representation."""
        if self.config.output == "indices":
            return indices

        if indices.size == 0:
            if self.config.output == "mask":
                return torch.zeros(values.shape, dtype=torch.bool, device=values.device)
            if self.config.output == "masked_values":
                return torch.full_like(values, float("nan"))
            return torch.tensor([], dtype=values.dtype, device=values.device)

        if indices.ndim == 1:
            idx = torch.as_tensor(indices, dtype=torch.long, device=values.device)
            if self.config.output == "values":
                return values[idx]
            if self.config.output == "masked_values":
                result = torch.full_like(values, float("nan"))
                result[idx] = values[idx]
                return result
            mask = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
            mask[idx] = True
            return mask

        idx_cols = tuple(
            torch.as_tensor(indices[:, dim], dtype=torch.long, device=values.device)
            for dim in range(indices.shape[1])
        )
        if self.config.output == "values":
            return values[idx_cols]
        if self.config.output == "masked_values":
            result = torch.full_like(values, float("nan"))
            result[idx_cols] = values[idx_cols]
            return result
        mask = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
        mask[idx_cols] = True
        return mask

    @task(name="MatrixFilter.run", task_run_name=generate_name)
    def run(
        self,
        data: Union[
            MatrixData,
            np.ndarray,
            torch.Tensor,
            List[float],
            List[RegionProperties],
            pd.Series,
            pd.DataFrame,
        ],
        *args: Any,
        device: Optional[torch.device] = None,
    ) -> Union[MatrixData, np.ndarray, torch.Tensor]:
        """Normalize *data* to a tensor and return the configured output."""
        matrix_data = data if isinstance(data, MatrixData) else None
        source = data.matrix if matrix_data is not None else data
        device = resolve_torch_device(
            device,
            preferred_input_type=self.config.preferred_input_type,
            preferred_device=self.config.preferred_device,
        )
        values = self._convert_input(
            source, dtype=self.config.preferred_input_type, device=device
        )
        attribute_list = self.config.attribute_list()
        if attribute_list and len(attribute_list) != _value_column_count(values):
            if self.config.strict:
                raise ValueError(
                    f"Length of attribute list {attribute_list} and data shape "
                    "do not match; Filter failed."
                )
            logger.warning(
                "Length of attribute list %s and data shape do not match; "
                "strict mode is disabled, ignoring filters.",
                attribute_list,
            )
        coords = self.accept_indices(values)
        raw = self._format_output(values, coords)
        if matrix_data is not None:
            return self._wrap_matrix_data_output(matrix_data, coords, raw)
        return raw

    def _prepare_values(
        self, values: torch.Tensor, exclude: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return *(masked values, validity mask)* for matrix selection."""
        return prepare_matrix_values(
            values,
            exclude,
            ignore_nan=self.config.ignore_nan,
            triangle=self.config.triangle,
        )

    @staticmethod
    def _filter_flat_indices(
        flat_idx: np.ndarray, valid: torch.Tensor
    ) -> np.ndarray:
        """Drop flat indices that point at masked positions."""
        if flat_idx.size == 0:
            return flat_idx
        valid_flat = valid.reshape(-1).cpu().numpy()
        return flat_idx[valid_flat[flat_idx]]

    def _filter_coords(
        self, coords: np.ndarray, valid: torch.Tensor
    ) -> np.ndarray:
        """Drop coordinate rows that point at masked positions."""
        if coords.size == 0:
            return coords
        valid_np = valid.cpu().numpy()
        kept = [coord for coord in coords if valid_np[tuple(coord)]]
        return np.asarray(kept, dtype=np.int64)

    def accept_indices(self, values: torch.Tensor) -> np.ndarray:
        """Return indices of matrix elements selected by this filter."""
        raise NotImplementedError("MatrixFilter.accept_indices is not implemented")


class TopKFilterConfig(MatrixFilterConfig):
    """Configuration for :class:`TopKFilter`."""

    k: int = 1
    axis: Optional[int] = 1
    largest: bool = True
    sort: bool = False


class TopKFilter(MatrixFilter):
    """Select indices of the top-*k* values along an axis (via :func:`torch.topk`)."""

    def __init__(self, config: TopKFilterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: TopKFilterConfig) -> "TopKFilter":
        return cls(config)

    def _exclude_value(self, values: torch.Tensor) -> torch.Tensor:
        """Sentinel written over masked positions so :func:`torch.topk` skips them."""
        fill = float("-inf") if self.config.largest else float("inf")
        return torch.full((), fill, dtype=values.dtype, device=values.device)

    def _effective_k(
        self, valid: torch.Tensor, axis: Optional[int], ndim: int
    ) -> int:
        """Cap *k* by the number of selectable elements along *axis*."""
        if axis is None or ndim == 1:
            count = int(valid.sum().item())
        else:
            counts = valid.sum(dim=axis)
            count = int(counts.max().item()) if counts.numel() else 0
        return min(self.config.k, count) if count > 0 else 0

    def accept_indices(self, values: torch.Tensor) -> np.ndarray:
        """Return index coordinates of the top-*k* elements in *values*."""
        if values.ndim == 0 or values.numel() == 0:
            return np.array([], dtype=np.int64)

        prepared, valid = self._prepare_values(values, self._exclude_value(values))
        axis = self.config.axis
        global_select = values.ndim == 1 or axis is None

        if global_select:
            k = self._effective_k(valid, axis=None, ndim=values.ndim)
            if k <= 0:
                return np.array([], dtype=np.int64)
            flat = prepared.reshape(-1)
            _, idx = torch.topk(
                flat,
                k=k,
                largest=self.config.largest,
                sorted=self.config.sort,
            )
            flat_idx = self._filter_flat_indices(
                idx.detach().cpu().numpy(), valid
            )
            if values.ndim == 1:
                return flat_idx
            return self._filter_coords(
                np.column_stack(
                    np.unravel_index(
                        flat_idx, tuple(int(s) for s in values.shape)
                    )
                ),
                valid,
            )

        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError(
                f"axis {self.config.axis!r} is out of bounds for ndim={values.ndim}"
            )

        k = self._effective_k(valid, axis=axis, ndim=values.ndim)
        if k <= 0:
            return np.array([], dtype=np.int64)

        _, indices = torch.topk(
            prepared,
            k=k,
            dim=axis,
            largest=self.config.largest,
            sorted=self.config.sort,
        )
        indices_np = indices.detach().cpu().numpy()
        coords: list[list[int]] = []
        for position, selected in np.ndenumerate(indices_np):
            coord = list(position)
            coord[axis] = int(selected)
            coords.append(coord)
        return self._filter_coords(np.asarray(coords, dtype=np.int64), valid)


class ValueFilterConfig(MatrixFilterConfig):
    """Configuration for :class:`ValueFilter`."""

    ref_value: Optional[float] = None
    operator: Literal["<", "<=", ">", ">=", "==", "!="] = "<="


class ValueFilter(MatrixFilter):
    """Keep matrix entries that satisfy a scalar threshold comparison."""

    def __init__(self, config: ValueFilterConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: ValueFilterConfig) -> "ValueFilter":
        return cls(config)

    def _value_mask(self, values: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of elements passing the configured comparison."""
        ref_value = self.config.ref_value
        if ref_value is None:
            raise ValueError("ValueFilterConfig.ref_value must be set")
        operator = self.config.operator
        if operator == ">":
            return values > ref_value
        if operator == ">=":
            return values >= ref_value
        if operator == "<":
            return values < ref_value
        if operator == "<=":
            return values <= ref_value
        if operator == "==":
            return values == ref_value
        if operator == "!=":
            return values != ref_value
        raise ValueError(f"Invalid operator: {operator}")

    def accept_indices(self, values: torch.Tensor) -> np.ndarray:
        """Return coordinates of elements that pass the threshold and validity mask."""
        passed = self._value_mask(values)
        exclude = torch.zeros((), dtype=values.dtype, device=values.device)
        _, valid = self._prepare_values(values, exclude)
        passed &= valid
        if values.ndim == 1:
            return torch.where(passed)[0].detach().cpu().numpy().astype(np.int64)
        coords = torch.stack(torch.where(passed), dim=1)
        return coords.detach().cpu().numpy().astype(np.int64)
