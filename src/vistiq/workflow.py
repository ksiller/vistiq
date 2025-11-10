import numpy as np
from typing import List
from pydantic import BaseModel

from vistiq.core import Configuration, StackProcessor, StackProcessorConfig


class WorkflowStepConfig(Configuration):
    pass


class WorkflowConfig(Configuration):
    step_configs: List[WorkflowStepConfig]


class BaseClass:
    def __init__(self, config: BaseModel):
        self.config = config

    def from_config(self, config: BaseModel) -> "BaseClass":
        raise NotImplementedError("Subclasses must implement this method")

    def name(self) -> str:
        return type(self).__name__

    def __str__(self) -> str:
        return f"{self.name()} with config: {self.config}"

    def __repr__(self) -> str:
        return f"{self.name()}({self.config})"

    def get_config(self) -> BaseModel:
        return self.config

    def set_config(self, config: BaseModel):
        self.config = config

    def run(self, img: np.ndarray, workers: int = -1, verbose: int = 10) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")


class WorkflowStep(BaseClass):
    def __init__(self, config: WorkflowStepConfig):
        super().__init__(config)

    def run(
        self, img: np.ndarray, *args, workers: int = -1, verbose: int = 10
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")


class Workflow(BaseClass):
    def __init__(self, config: WorkflowConfig):
        super().__init__(config)

