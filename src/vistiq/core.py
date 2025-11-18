from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, Any, Optional, Tuple, Union
from pydantic import BaseModel, PositiveInt
#from pydantic.dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
import logging

from vistiq.utils import ArrayIterator, ArrayIteratorConfig

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound=BaseModel)

#@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Configuration(BaseModel):
    """Base configuration class for all vistiq components.

    This class serves as the foundation for all configuration models in the vistiq package.
    Subclasses should extend this to define specific configuration parameters.
    """

    classname: str | None = None

    class Config:
        """Pydantic configuration.
        
        Configures Pydantic model behavior:
        - extra: "forbid" prevents extra fields from being accepted
        - validate_assignment: Validates fields when assigned
        - arbitrary_types_allowed: Allows arbitrary types (e.g., numpy arrays)
        """

        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
        arbitrary_types_allowed=True


class Configurable(ABC, Generic[ConfigType]):
    """Abstract base class for configurable objects.

    This class provides a standard interface for objects that can be configured
    with a Pydantic configuration model. Subclasses should implement the abstract
    methods to define their behavior.

    Type Parameters:
        ConfigType: The type of configuration model this object uses.
    """

    def __init__(self, config: ConfigType):
        """Initialize the configurable object.

        Args:
            config: Configuration model instance.
        """
        self.config = config

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigType) -> "Configurable[ConfigType]":
        """Create an instance from a configuration model.

        Args:
            config: Configuration model instance.

        Returns:
            A new instance of the configurable object.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_config(self) -> ConfigType:
        """Get the current configuration.

        Returns:
            The current configuration model instance.
        """
        return self.config

    #def set_config(self, config: ConfigType) -> None:
    #    """Set a new configuration.
    #
    #    Args:
    #        config: New configuration model instance.
    #    """
    #    self.config = config

    def name(self) -> str:
        """Get the name of this configurable object.

        Returns:
            The class name of this object.
        """
        return type(self).__name__

    def __str__(self) -> str:
        """String representation of the configurable object.

        Returns:
            A string describing the object and its configuration.
        """
        return f"{self.name()} with config: {self.config}"

    def __repr__(self) -> str:
        """Developer-friendly representation of the configurable object.

        Returns:
            A string representation suitable for debugging.
        """
        return f"{self.name()}({self.config})"


class StackProcessorConfig(Configuration):
    """Configuration for stack processing operations.

    This configuration defines how image stacks should be processed, including
    iteration strategy, output format, and result reshaping options.
    
    Attributes:
        iterator_config: Configuration for array iteration (which axes to iterate over).
        output_type: Output format ("stack" for single array, "list" for list of results).
        squeeze: Whether to squeeze singleton dimensions from output.
    """

    iterator_config: ArrayIteratorConfig = ArrayIteratorConfig(slice_def=(-2, -1))
    batch_size: PositiveInt = 1
    tile_shape: Optional[Tuple[int, int]] = None
    output_type: Literal["stack", "list"] = "stack"
    squeeze: bool = True


class StackProcessor(Configurable):
    """Processor for image stacks with configurable iteration and output handling.

    This class processes image stacks by iterating over slices according to the
    iterator configuration, processing each slice in parallel, and reshaping
    results according to the output configuration.
    """

    output_options: Literal["stack", "list"] = "stack"
    max_dim_per_process: int = -1

    def __init__(self, config: StackProcessorConfig):
        """Initialize the stack processor.

        Args:
            config: Stack processor configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: StackProcessorConfig) -> "StackProcessor":
        """Create a StackProcessor instance from a configuration.

        Args:
            config: Stack processor configuration.

        Returns:
            A new StackProcessor instance.
        """
        return cls(config)

    def run(
        self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10
    ) -> np.ndarray | tuple[Any,...]:
        """Run the stack processor on an image.

        Args:
            stack: Input stack array.
            *args: Additional arguments to pass to _process_slice.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.

        Returns:
            Processed result array or tuple of lists depending on output_type.
        """
        logger.info(f"Running {type(self).__name__} with config: {self.config}")
        iterator = ArrayIterator(stack, self.config.iterator_config)
        n_iterations = len(iterator)
        if n_iterations == 1:
            results = self._process_slice(
                stack, *args
            )
        else:
            results = Parallel(n_jobs=workers, verbose=verbose)(
                delayed(self._process_slice)(
                    stack_slice, *args)
                for stack_slice in iterator
            )
            # reshape results
            results = self._reshape_slice_results(results, slice_indices=iterator.indices, input_shape=stack.shape)
        return results

    def _process_slice(
        self, slice: np.ndarray, *args) -> np.ndarray | tuple[Any,...]:
        """Process a single slice of the image stack.

        This method must be implemented by subclasses to define the actual
        processing logic for each slice.

        Args:
            slice: Single slice from the image stack.
            *args: Additional arguments.
            workers: Number of workers (for nested parallelism).
            verbose: Verbosity level.

        Returns:
            Processed slice result.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        logger.info(f"Processing slice with shape {slice.shape}")
        raise NotImplementedError("Subclasses must implement this method")

    def _reshape_slice_results(self, results: list[Any], slice_indices: list[tuple[int,...]], input_shape: tuple[int,...]) -> np.ndarray | tuple[Any,...]:
        """Reshape the results of slice processing according to output configuration.
        
        Reshapes the list of slice results into the desired output format:
        - "stack": Stacks results into a single array matching input shape
        - "list": Returns as list or tuple of lists (for tuple results)
        
        Args:
            results: List of results from processing each slice.
            slice_indices: List of index tuples for each slice (not used currently).
            input_shape: Shape of the input array.
            
        Returns:
            Reshaped results as array or tuple/list depending on output_type.
        """
        if self.config.output_type == "stack":
            if not isinstance(results, np.ndarray):
                results = np.stack(results, axis=0)
            results = results.reshape(input_shape)
            if self.config.squeeze:
                results = results.squeeze()
            logger.info(f"Reshaped results to shape {results.shape}")
        elif self.config.output_type == "list":
            if self.config.squeeze:
                results = [
                    result for result in results if result is not None
                ]
            # convert list of tuples to tuple of lists [(r)]
            if isinstance(results[0], tuple):
                logger.info(f"Reshaping results, type(results)={type(results)}")
                zipped_elements = zip(*results)
                list_of_lists = [list(item) for item in zipped_elements]
                results = tuple(list_of_lists)
            else:
                logger.info(f"Not reshaping results, type(results)={type(results)}")
            #    results = [list(item) for item in results]
        return results

class ChainProcessorConfig(Configuration):
    """Configuration for chain processor operations.

    This configuration defines how multiple stack processors should be chained together.
    
    Attributes:
        processors: List of configurable processors to apply in sequence.
    """

    processors: list[Configurable[Configuration]] = []


class ChainProcessor(Configurable[ChainProcessorConfig]):
    """Processor for chaining multiple stack processors.

    This class chains multiple stack processors together, applying each one in sequence to the input stack.
    """

    def __init__(self, config: ChainProcessorConfig):
        """Initialize the chain processor.

        Args:
            config: Chain processor configuration.
        """
        super().__init__(config)
        self.processors = [processor.from_config(processor.config) for processor in config.processors]

    @classmethod
    def from_config(cls, config: ChainProcessorConfig) -> "ChainProcessor":
        """Create a ChainProcessor instance from a configuration.   

        Args:
            config: Chain processor configuration.

        Returns:
            A new ChainProcessor instance.
        """
        return cls(config)

    def run(
        self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10
    ) -> np.ndarray:        
        """Run the chain processor on an image stack.

        Args:
            stack: Input image stack.
            *args: Additional arguments to pass to each processor.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.

        Returns:
            Processed result array or tuple of lists depending on output_type.
        """
        result = stack
        for processor in self.config.processors:
            result = processor.run(result, *args, workers=workers, verbose=verbose)
        return result