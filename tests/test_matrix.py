"""Tests for vistiq.matrix."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vistiq.matrix import (
    MatrixAggregator,
    MatrixAggregatorConfig,
    MatrixCombiner,
    MatrixCombinerConfig,
    MatrixData,
    MatrixFormatter,
    MatrixFormatterConfig,
    annotations_at_coords,
    as_matrix_data,
    composite_matrix_annotations,
    normalize_coords,
)
from vistiq.matrix.types import FULL, OFF_DIAGONAL, UPPER


class TestMatrixAggregator:
    def test_sum_axis_0(self):
        data = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        result = MatrixAggregator(
            MatrixAggregatorConfig(operation="sum", axis=0)
        ).run(data)
        assert isinstance(result.matrix, torch.Tensor)
        expected = torch.tensor([3.0, 4.0, 5.0])
        torch.testing.assert_close(result.matrix, expected)

    def test_mean_ignores_nan(self):
        data = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
        result = MatrixAggregator(
            MatrixAggregatorConfig(operation="mean", axis=1)
        ).run(data)
        torch.testing.assert_close(result.matrix[0], torch.tensor(2.0))
        torch.testing.assert_close(result.matrix[1], torch.tensor(5.0))

    def test_count_with_triangle_mask(self):
        data = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        result = MatrixAggregator(
            MatrixAggregatorConfig(
                operation="count", axis=1, triangle=OFF_DIAGONAL
            )
        ).run(data)
        assert result.matrix.tolist() == [2.0, 2.0, 2.0]

    def test_numpy_input_returns_numpy(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = MatrixAggregator(
            MatrixAggregatorConfig(operation="sum", axis=0)
        ).run(data)
        assert isinstance(result.matrix, np.ndarray)
        np.testing.assert_allclose(result.matrix, [4.0, 6.0])


class TestMatrixCombiner:
    def test_combine_blocks(self):
        import pandas as pd

        a = pd.DataFrame([[1.0, 2.0]], index=["x"], columns=["y", "z"])
        b = pd.DataFrame([[3.0]], index=["y"], columns=["x"])
        combiner = MatrixCombiner(
            MatrixCombinerConfig(fill_value=0.0, symmetrize=False, triangle=FULL)
        )
        formatter = MatrixFormatter(MatrixFormatterConfig())
        result = formatter.run(combiner.run([a, b]))
        assert result.loc["x", "y"] == 1.0
        assert result.loc["y", "x"] == 3.0

    def test_upper_triangle_mask(self):
        import pandas as pd

        a = pd.DataFrame([[1.0]], index=["x"], columns=["y"])
        b = pd.DataFrame([[2.0]], index=["y"], columns=["x"])
        combiner = MatrixCombiner(
            MatrixCombinerConfig(
                fill_value=0.0,
                symmetrize=True,
                triangle=UPPER,
            )
        )
        formatter = MatrixFormatter(MatrixFormatterConfig())
        result = formatter.run(combiner.run([a, b]))
        assert result.loc["x", "y"] == 2.0
        assert result.loc["y", "x"] == 0.0
        assert result.loc["x", "x"] == 0.0

    def test_off_diagonal_excludes_self_pairs(self):
        import pandas as pd

        block = pd.DataFrame([[0.5]], index=["a"], columns=["a"])
        combiner = MatrixCombiner(
            MatrixCombinerConfig(fill_value=0.0, symmetrize=True, triangle=OFF_DIAGONAL)
        )
        formatter = MatrixFormatter(MatrixFormatterConfig())
        result = formatter.run(combiner.run([block]))
        assert result.loc["a", "a"] == 0.0

    def test_matrix_data_input(self):
        data = as_matrix_data(
            __import__("pandas").DataFrame([[1.0]], index=["x"], columns=["y"])
        )
        combiner = MatrixCombiner(
            MatrixCombinerConfig(fill_value=0.0, symmetrize=False)
        )
        result = combiner.run([data])
        assert result.annotations == (("x",), ("y",))

    def test_as_matrix_data_series_uses_index(self):
        import pandas as pd

        series = pd.Series([1.0, 2.0], index=["a", "b"])
        data = as_matrix_data(series)
        assert data.shape == (2,)
        assert data.annotations == (("a", "b"),)

    def test_as_matrix_data_ignores_annotations_for_pandas(self):
        import pandas as pd

        frame = pd.DataFrame([[1.0]], index=["x"], columns=["y"])
        series = pd.Series([1.0], index=["x"])
        override = (("wrong",), ("labels",))
        assert as_matrix_data(frame, annotations=override).annotations == (
            ("x",),
            ("y",),
        )
        assert as_matrix_data(series, annotations=override).annotations == (("x",),)


class TestMatrixAnnotationProjection:
    def test_normalize_coords_1d(self):
        coords = np.array([0, 2, 4], dtype=np.int64)
        normalized = normalize_coords(coords)
        assert normalized.shape == (3, 1)
        np.testing.assert_array_equal(normalized[:, 0], coords)

    def test_annotations_at_coords_2d(self):
        annotations = (("r0", "r1", "r2"), ("c0", "c1"))
        coords = np.array([[0, 1], [2, 0]], dtype=np.int64)
        projected = annotations_at_coords(annotations, coords)
        assert projected == (("r0", "r2"), ("c1", "c0"))

    def test_composite_matrix_annotations(self):
        projected = (("r0", "r2"), ("c1", "c0"))
        composite = composite_matrix_annotations(projected)
        assert composite == (("r0|c1", "r2|c0"),)

    def test_composite_single_axis_passthrough(self):
        projected = (("a", "b", "c"),)
        composite = composite_matrix_annotations(projected)
        assert composite == projected


class TestMatrixFormatter:
    def test_dataframe_export(self):
        import pandas as pd

        data = MatrixData(
            matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
            annotations=(("r0", "r1"), ("c0", "c1")),
        )
        frame = MatrixFormatter(MatrixFormatterConfig()).run(data)
        assert isinstance(frame, pd.DataFrame)
        assert frame.index.tolist() == ["r0", "r1"]
        assert frame.columns.tolist() == ["c0", "c1"]

    def test_numpy_export_without_annotations(self):
        data = MatrixData(matrix=np.array([[1.0, 2.0], [3.0, 4.0]]))
        matrix = MatrixFormatter(
            MatrixFormatterConfig(output_type="np.ndarray", annotate=False)
        ).run(data)
        np.testing.assert_allclose(matrix, data.matrix)

    def test_series_export_for_1d(self):
        import pandas as pd

        data = MatrixData(matrix=np.array([1.0, 2.0]), annotations=(("a", "b"),))
        series = MatrixFormatter(MatrixFormatterConfig()).run(data)
        assert isinstance(series, pd.Series)
        assert series.index.tolist() == ["a", "b"]
