import logging
from typing import Literal, Optional, Union

import numpy as np
import torch
from pydantic import model_validator

from vistiq.matrix.calc import MatrixCalculator, MatrixCalculatorConfig
from vistiq.matrix.types import ArrayBackend

logger = logging.getLogger(__name__)


class DistanceCalculatorConfig(MatrixCalculatorConfig):
    """Configuration for :class:`DistanceCalculator`."""

    method: Literal["euclidean", "manhattan", "hamming", "chebyshev", "minkowski"] = "euclidean"
    minkowski_p: float = 3.0
    preferred_input_type: ArrayBackend = "torch.Tensor"

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
        "manhattan": 1.0,
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
        points1: Union[torch.Tensor, np.ndarray],
        points2: Union[torch.Tensor, np.ndarray],
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute pairwise distances with PyTorch."""
        del device
        distances = torch.cdist(points1, points2, p=self._cdist_p())
        logger.info("DistanceCalculator._calculate: distances.shape=%s", distances.shape)
        return distances
