"""Matrix types, masking, and operations."""

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
        "MatrixCalculator",
        "MatrixCalculatorConfig",
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
    "MatrixFormatOutput",
    "MatrixFormatter",
    "MatrixFormatterConfig",
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
