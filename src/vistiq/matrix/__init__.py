"""Matrix types, masking, selection, and operations.

``vistiq.matrix`` owns labeled numeric data and generic matrix algebra:

- :class:`~vistiq.matrix.types.MatrixData` — ndarray/tensor + axis labels
- :class:`~vistiq.matrix.calc.MatrixCalculator` — pairwise calculators
  (e.g. :class:`~vistiq.matrix.calc.DistanceCalculator`)
- :class:`~vistiq.matrix.select.MatrixFilter` — threshold / top-k selection
- :class:`~vistiq.matrix.ops.MatrixFormatter` — export to DataFrame/ndarray
- :class:`~vistiq.matrix.ops.MatrixCombiner` / :class:`~vistiq.matrix.ops.MatrixAggregator`

Domain-specific matrix **producers** (overlap IoU/IoS/Dice for boxes, masks,
and labels) live in :mod:`vistiq.analysis.overlap` because they encode imaging
geometry, not generic matrix math. Graph structure and hierarchy inference live
in :mod:`vistiq.graph`.
"""

from vistiq.matrix.mask import prepare_matrix_values, triangle_valid_mask
from vistiq.matrix.types import (
    DIAGONAL,
    FULL,
    LOWER,
    LOWER_ND,
    OFF_DIAGONAL,
    UPPER,
    UPPER_ND,
    AnnotationFactory,
    ArrayBackend,
    MatrixAnnotations,
    MatrixArray,
    MatrixContainer,
    MatrixData,
    MatrixFormatOutput,
    annotations_at_coords,
    as_matrix_data,
    composite_matrix_annotations,
    default_matrix_annotations,
    matrix_to_numpy,
    normalize_coords,
    resolve_matrix_annotations,
    square_matrix,
)

_CALC_EXPORTS = frozenset(
    {
        "DistanceCalculator",
        "DistanceCalculatorConfig",
        "MatrixCalculator",
        "MatrixCalculatorConfig",
    }
)

_SELECT_EXPORTS = frozenset(
    {
        "MatrixFilter",
        "MatrixFilterConfig",
        "TopKFilter",
        "TopKFilterConfig",
        "ValueFilter",
        "ValueFilterConfig",
    }
)

_OPS_EXPORTS = frozenset(
    {
        "MatrixAggregator",
        "MatrixAggregatorConfig",
        "MatrixCombiner",
        "MatrixCombinerConfig",
        "MatrixFormatter",
        "MatrixFormatterConfig",
        "group_matrix_indices",
        "upper_triangle_adjacency",
    }
)


def __getattr__(name: str):
    if name in _CALC_EXPORTS:
        from vistiq.matrix import calc

        return getattr(calc, name)
    if name in _SELECT_EXPORTS:
        from vistiq.matrix import select

        return getattr(select, name)
    if name in _OPS_EXPORTS:
        from vistiq.matrix import ops

        return getattr(ops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DIAGONAL",
    "FULL",
    "LOWER",
    "LOWER_ND",
    "OFF_DIAGONAL",
    "UPPER",
    "UPPER_ND",
    "AnnotationFactory",
    "ArrayBackend",
    "DistanceCalculator",
    "DistanceCalculatorConfig",
    "MatrixAggregator",
    "MatrixAggregatorConfig",
    "MatrixAnnotations",
    "MatrixArray",
    "MatrixCalculator",
    "MatrixCalculatorConfig",
    "MatrixCombiner",
    "MatrixCombinerConfig",
    "MatrixContainer",
    "MatrixData",
    "MatrixFilter",
    "MatrixFilterConfig",
    "MatrixFormatOutput",
    "MatrixFormatter",
    "MatrixFormatterConfig",
    "TopKFilter",
    "TopKFilterConfig",
    "ValueFilter",
    "ValueFilterConfig",
    "annotations_at_coords",
    "as_matrix_data",
    "composite_matrix_annotations",
    "default_matrix_annotations",
    "group_matrix_indices",
    "matrix_to_numpy",
    "normalize_coords",
    "prepare_matrix_values",
    "resolve_matrix_annotations",
    "square_matrix",
    "triangle_valid_mask",
    "upper_triangle_adjacency",
]
