import numpy as np
from typing import List, Optional, Any
from pydantic import BaseModel
import logging

from vistiq.core import Configuration, StackProcessor, StackProcessorConfig

logger = logging.getLogger(__name__)


class WorkflowStepConfig(Configuration):
    """Configuration for a single workflow step.
    
    Base configuration class for workflow steps. Subclasses should define
    step-specific configuration parameters.
    """
    pass


class WorkflowConfig(Configuration):
    """Configuration for a complete workflow.
    
    Defines a sequence of workflow steps to be executed in order.
    
    Attributes:
        step_configs: List of workflow step configurations to execute.
    """
    step_configs: List[WorkflowStepConfig]


class BaseClass:
    """Base class for workflow components.
    
    Provides common functionality for workflow steps and workflows,
    including configuration management and execution interface.
    
    Attributes:
        config: Configuration model instance.
    """
    
    def __init__(self, config: BaseModel):
        """Initialize the base class.
        
        Args:
            config: Configuration model instance.
        """
        self.config = config

    def from_config(self, config: BaseModel) -> "BaseClass":
        """Create an instance from a configuration model.
        
        Args:
            config: Configuration model instance.
            
        Returns:
            A new instance of the class.
            
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def name(self) -> str:
        """Get the name of this class.
        
        Returns:
            The class name as a string.
        """
        return type(self).__name__

    def __str__(self) -> str:
        """String representation of the object.
        
        Returns:
            A string describing the object and its configuration.
        """
        return f"{self.name()} with config: {self.config}"

    def __repr__(self) -> str:
        """Developer-friendly representation.
        
        Returns:
            A string representation suitable for debugging.
        """
        return f"{self.name()}({self.config})"

    def get_config(self) -> BaseModel:
        """Get the current configuration.
        
        Returns:
            The current configuration model instance.
        """
        return self.config

    def set_config(self, config: BaseModel):
        """Set a new configuration.
        
        Args:
            config: New configuration model instance.
        """
        self.config = config

    def run(self, img: np.ndarray, workers: int = -1, verbose: int = 10, metadata: Optional[dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Run the workflow component on an image.
        
        Args:
            img: Input image array.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for processing.
            metadata: Optional metadata dictionary.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Processed image array.
            
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")


class WorkflowStep(BaseClass):
    """A single step in a workflow.
    
    Represents one processing step that can be part of a larger workflow.
    Subclasses should implement the run method to define the step's behavior.
    """
    
    def __init__(self, config: WorkflowStepConfig):
        """Initialize the workflow step.
        
        Args:
            config: Workflow step configuration.
        """
        super().__init__(config)

    def run(
        self, img: np.ndarray, *args, workers: int = -1, verbose: int = 10, metadata: Optional[dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Run the workflow step on an image.
        
        Args:
            img: Input image array.
            *args: Additional positional arguments.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for processing.
            metadata: Optional metadata dictionary.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Processed image array.
            
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")


class Workflow(BaseClass):
    """A complete workflow composed of multiple steps.
    
    Executes a sequence of workflow steps in order, passing the output
    of each step as input to the next step.
    """
    
    def __init__(self, config: WorkflowConfig):
        """Initialize the workflow.
        
        Args:
            config: Workflow configuration containing step configurations.
        """
        super().__init__(config)

