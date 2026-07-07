"""Triangle masking and validity preparation for matrix operations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from vistiq.matrix.types import DIAGONAL, FULL, LOWER_ND, UPPER_ND


def triangle_valid_mask_numpy(
    values: np.ndarray, flags: int
) -> Optional[np.ndarray]:
    """Boolean mask of allowed triangle regions; ``None`` if unrestricted or non-square."""
    if flags == FULL:
        return None
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return None

    n = values.shape[0]
    row_idx = np.arange(n)[:, None]
    col_idx = np.arange(n)[None, :]
    valid = np.zeros((n, n), dtype=bool)
    if flags & DIAGONAL:
        valid |= row_idx == col_idx
    if flags & LOWER_ND:
        valid |= row_idx > col_idx
    if flags & UPPER_ND:
        valid |= row_idx < col_idx
    return valid


def triangle_valid_mask(
    values: torch.Tensor, flags: int
) -> Optional[torch.Tensor]:
    """Return a boolean mask of allowed triangle regions, or ``None`` if unrestricted."""
    if flags == FULL:
        return None
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return None

    n = values.shape[0]
    row_idx = torch.arange(n, device=values.device).unsqueeze(1)
    col_idx = torch.arange(n, device=values.device).unsqueeze(0)

    valid = torch.zeros((n, n), dtype=torch.bool, device=values.device)
    if flags & DIAGONAL:
        valid |= row_idx == col_idx
    if flags & LOWER_ND:
        valid |= row_idx > col_idx
    if flags & UPPER_ND:
        valid |= row_idx < col_idx
    return valid


def mask_triangle(values: np.ndarray, flags: int, fill_value: float) -> np.ndarray:
    """Zero out (or fill) cells outside the configured triangle region."""
    mask = triangle_valid_mask_numpy(values, flags)
    if mask is None:
        return values
    result = np.array(values, copy=True)
    result[~mask] = fill_value
    return result


def prepare_matrix_values(
    values: torch.Tensor,
    exclude: torch.Tensor,
    *,
    ignore_nan: bool,
    triangle: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return *(masked values, validity mask)* for matrix reduction or selection."""
    valid = torch.ones(values.shape, dtype=torch.bool, device=values.device)
    if ignore_nan:
        valid &= ~torch.isnan(values)
    triangle_mask = triangle_valid_mask(values, triangle)
    if triangle_mask is not None:
        valid &= triangle_mask
    prepared = torch.where(valid, values, exclude)
    return prepared, valid
