"""Tests for vistiq.analysis.distance."""

import pytest

torch = pytest.importorskip("torch")

from vistiq.matrix.calc import DistanceCalculator, DistanceCalculatorConfig
from vistiq.matrix.types import MatrixData


class TestDistanceCalculator:
    def test_manhattan_config_accepts_method(self):
        cfg = DistanceCalculatorConfig(method="manhattan")
        assert cfg.method == "manhattan"

    def test_manhattan_pairwise_distances(self):
        points = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
        result = DistanceCalculator(
            DistanceCalculatorConfig(method="manhattan")
        ).run(points, points, device=torch.device("cpu"))
        assert isinstance(result, MatrixData)
        expected = torch.tensor(
            [
                [0.0, 3.0, 4.0],
                [3.0, 0.0, 3.0],
                [4.0, 3.0, 0.0],
            ]
        )
        torch.testing.assert_close(result.matrix, expected)

    def test_cdist_p_manhattan(self):
        calc = DistanceCalculator(DistanceCalculatorConfig(method="manhattan"))
        assert calc._cdist_p() == 1.0
