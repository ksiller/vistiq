"""Tests for vistiq.analysis.matrix."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vistiq.analysis.matrix import (
    MatrixAggregator,
    MatrixAggregatorConfig,
    MatrixCombiner,
    MatrixCombinerConfig,
)
from vistiq.constant.matrix import FULL, OFF_DIAGONAL, UPPER


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
        assert isinstance(result, torch.Tensor)
        expected = torch.tensor([3.0, 4.0, 5.0])
        torch.testing.assert_close(result, expected)

    def test_mean_ignores_nan(self):
        data = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
        result = MatrixAggregator(
            MatrixAggregatorConfig(operation="mean", axis=1)
        ).run(data)
        torch.testing.assert_close(result[0], torch.tensor(2.0))
        torch.testing.assert_close(result[1], torch.tensor(5.0))

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
        assert result.tolist() == [2.0, 2.0, 2.0]

    def test_numpy_input_returns_numpy(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = MatrixAggregator(
            MatrixAggregatorConfig(operation="sum", axis=0)
        ).run(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [4.0, 6.0])


class TestMatrixCombiner:
    def test_combine_blocks(self):
        import pandas as pd

        a = pd.DataFrame([[1.0, 2.0]], index=["x"], columns=["y", "z"])
        b = pd.DataFrame([[3.0]], index=["y"], columns=["x"])
        result = MatrixCombiner(
            MatrixCombinerConfig(fill_value=0.0, symmetrize=False, triangle=FULL)
        ).run([a, b])
        assert result.loc["x", "y"] == 1.0
        assert result.loc["y", "x"] == 3.0

    def test_upper_triangle_mask(self):
        import pandas as pd

        a = pd.DataFrame([[1.0]], index=["x"], columns=["y"])
        b = pd.DataFrame([[2.0]], index=["y"], columns=["x"])
        result = MatrixCombiner(
            MatrixCombinerConfig(
                fill_value=0.0,
                symmetrize=True,
                triangle=UPPER,
            )
        ).run([a, b])
        assert result.loc["x", "y"] == 2.0
        assert result.loc["y", "x"] == 0.0
        assert result.loc["x", "x"] == 0.0

    def test_off_diagonal_excludes_self_pairs(self):
        import pandas as pd

        block = pd.DataFrame([[0.5]], index=["a"], columns=["a"])
        result = MatrixCombiner(
            MatrixCombinerConfig(fill_value=0.0, symmetrize=True, triangle=OFF_DIAGONAL)
        ).run([block])
        assert result.loc["a", "a"] == 0.0
