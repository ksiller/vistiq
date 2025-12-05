from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, Any, Optional, Tuple, Union, TYPE_CHECKING
from pydantic import BaseModel, PositiveInt, Field, field_validator
#from pydantic.dataclasses import dataclass
import numpy as np
import pandas as pd
# Lazy import joblib to avoid slow initialization at module import time
import logging
from prefect import task
from functools import wraps
from joblib import Parallel, delayed
from bioio import Dimensions, Scale
from vistiq.utils import ArrayIterator, ArrayIteratorConfig

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="Configuration")

if TYPE_CHECKING:
    from typing import Type


def cli_field(
    default: Any = ...,
    *,
    default_factory: Any = None,
    alias: str | None = None,
    alias_priority: int | None = None,
    validation_alias: str | None = None,
    serialization_alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    examples: list[Any] | None = None,
    exclude: bool | None = None,
    discriminator: str | None = None,
    json_schema_extra: dict[str, Any] | None = None,
    frozen: bool | None = None,
    validate_default: bool | None = None,
    repr: bool = True,
    init_var: bool | None = None,
    kw_only: bool | None = None,
    pattern: str | None = None,
    strict: bool | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    **extra: Any,
) -> Any:
    """Create a Field that is explicitly marked for CLI argument generation.
    
    This is a convenience wrapper around Pydantic's Field that marks the field
    as available for CLI argument generation. By default, all Configuration fields
    are included in CLI arguments unless marked with cli=False.
    
    This function is useful for explicitly marking fields that should be exposed
    as CLI arguments, though it's not strictly necessary since fields are included
    by default.
    
    To exclude a field from CLI arguments, use:
        Field(..., json_schema_extra={'cli': False})
    
    Args:
        All arguments are the same as Pydantic's Field, with the addition that
        the field will be explicitly marked for CLI exposure.
        
    Returns:
        A FieldInfo object marked for CLI exposure.
        
    Example:
        class MyConfig(Configuration):
            # This field will be available as --my-arg
            my_arg: str = cli_field(default="value", description="My argument")
            
            # This field will NOT be available as CLI argument
            internal_field: str = Field(default="internal", json_schema_extra={'cli': False})
    """
    # Add CLI marker to json_schema_extra
    if json_schema_extra is None:
        json_schema_extra = {}
    json_schema_extra['cli'] = True
    
    return Field(
        default=default,
        default_factory=default_factory,
        alias=alias,
        alias_priority=alias_priority,
        validation_alias=validation_alias,
        serialization_alias=serialization_alias,
        title=title,
        description=description,
        examples=examples,
        exclude=exclude,
        discriminator=discriminator,
        json_schema_extra=json_schema_extra,
        frozen=frozen,
        validate_default=validate_default,
        repr=repr,
        init_var=init_var,
        kw_only=kw_only,
        pattern=pattern,
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        min_length=min_length,
        max_length=max_length,
        **extra
    )


def cli_config(*, exclude: list[str] | None = None, include_only: list[str] | None = None):
    """Class decorator to mark Configuration fields for CLI argument generation.
    
    This decorator provides an elegant way to control which fields from a Configuration
    class are exposed as CLI arguments. By default, all fields are included unless
    specified otherwise.
    
    Args:
        exclude: List of field names to exclude from CLI arguments.
        include_only: List of field names to include in CLI arguments (excludes all others).
                     Mutually exclusive with exclude.
    
    Returns:
        A class decorator function.
        
    Example:
        @cli_config(exclude=['internal_field', 'computed_value'])
        class MyConfig(Configuration):
            public_field: str = Field(default="value")
            internal_field: str = Field(default="internal")  # Won't appear in CLI
            computed_value: int = Field(default=0)  # Won't appear in CLI
            
        @cli_config(include_only=['width', 'height'])
        class ResizeConfig(Configuration):
            width: int = Field(default=256)  # Will appear in CLI
            height: int = Field(default=256)  # Will appear in CLI
            internal: str = Field(default="")  # Won't appear in CLI
    """
    if exclude and include_only:
        raise ValueError("Cannot specify both 'exclude' and 'include_only'")
    
    def decorator(cls):
        """Apply CLI configuration to the class."""
        # Validate that cls is a Configuration subclass at runtime
        # This avoids forward reference issues during class definition
        # We check after the class is fully defined
        import sys
        current_module = sys.modules.get(cls.__module__)
        if current_module:
            # Check if this is Configuration itself or a subclass
            ConfigBase = getattr(current_module, 'Configuration', None)
            if ConfigBase is not None and cls is not ConfigBase:
                # Only validate if it's not Configuration itself
                if not issubclass(cls, ConfigBase):
                    raise TypeError(f"@cli_config can only be applied to Configuration subclasses, got {cls}")
        # If Configuration isn't available yet (forward reference), we'll skip validation
        # The class will be validated when it's actually used
        
        # Store CLI configuration as class attribute
        # This will be checked by ConfigArgumentBuilder when building arguments
        cls.__cli_config__ = {
            'exclude': exclude,
            'include_only': include_only
        }
        
        return cls
    
    return decorator

#@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
@cli_config(exclude=['classname', 'package', 'version', 'command_group'])
class Configuration(BaseModel):
    """Base configuration class for all vistiq components.

    This class serves as the foundation for all configuration models in the vistiq package.
    Subclasses should extend this to define specific configuration parameters.
    """

    classname: str | None = None
    package: str | None = None
    version: str | None = None
    command_group: str | None = None

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
        # Update config with class metadata using Pydantic's model_copy
        # Get version from package if class doesn't have it
        version = getattr(type(self), '__version__', None)
        if version is None:
            # Try to get version from the package
            try:
                from vistiq import __version__ as package_version
                version = package_version
            except ImportError:
                version = None
        
        # Use model_copy to create updated config (Pydantic v2 way)
        if hasattr(self.config, 'model_copy'):
            self.config = self.config.model_copy(update={
                "classname": type(self).__name__,
                "package": type(self).__module__,
                "version": version
            })
        else:
            # Fallback: directly assign if model_copy not available
            self.config.classname = type(self).__name__
            self.config.package = type(self).__module__
            self.config.version = version

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigType) -> "Configurable[ConfigType]":
        """Create an instance from a configuration model. Realizes an instance with concrete classname, package, and version to capture specific runtime and support reproducibility.

        Args:
            config: Configuration model instance.

        Returns:
            A new instance of the configurable object.
        """
        return cls(config)

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

    @task(name="Configurable.run")
    def run(self, *args, **kwargs) -> Any:
        """Run the configurable object.

        Args:
            *args: Additional arguments to pass to the run method.
            **kwargs: Additional keyword arguments to pass to the run method.

        Returns:
            The result of the run method.
        """
        raise NotImplementedError("Subclasses must implement this method")

class StackProcessorConfig(Configuration):
    """Configuration for stack processing operations.

    This configuration defines how image stacks should be processed, including
    iteration strategy, output format, and result reshaping options.
    
    Attributes:
        iterator_config: Configuration for array iteration (which axes to iterate over).
        output_type: Output format ("stack" for single array, "list" for list of results).
        squeeze: Whether to squeeze singleton dimensions from output.
        split_channels: Whether to split channels into separate files after processing.
        rename_channel: Dictionary mapping original channel names to new names. Can be provided as a string
            in format 'old1:new1;old2:new2' (e.g., 'Red:Dpn;Blue:EDU') which will be automatically parsed.
    """

    iterator_config: ArrayIteratorConfig = ArrayIteratorConfig(slice_def=(-2, -1))
    batch_size: PositiveInt = Field(default=10, description="Number of slices to process in parallel")
    tile_shape: Optional[Tuple[int, int]] = None
    output_type: Literal["stack", "list", "dataframe"] = "stack"
    output_shape: Optional[Tuple[int,...]] = None
    squeeze: bool = True
    split_axis: Optional[Union[int, str]] = Field(default=None, description="Split stack along specified axis into separate files after processing")
    split_channels: bool = Field(
        default=False, 
        description="Split channels into separate files after processing"
    )
    rename_channel: Optional[dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping original channel names to new names (format: 'old1:new1;old2:new2')"
    )
    
    @field_validator('rename_channel', mode='before')
    @classmethod
    def parse_rename_channel(cls, v: Any) -> Optional[dict[str, str]]:
        """Parse rename_channel from string format to dictionary.
        
        Uses the utility function from vistiq.utils for consistent parsing.
        """
        from vistiq.utils import str_to_dict
        return str_to_dict(v, value_type=str)


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

    @task(name="StackProcessor.run")
    def run(
        self, stack: np.ndarray, *args, workers: int = 1, verbose: int = 10, metadata: Optional[dict[str, Any]] = None, **kwargs) -> tuple[Any, Optional[dict[str, Any]]]:
        """Run the stack processor on an image.

        Args:
            stack: Input stack array.
            *args: Additional arguments to pass to _process_slice.
            workers: Number of parallel workers (1 for single process, -1 for all cores).
            verbose: Verbosity level for parallel processing.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Tuple of (processed result array or tuple of lists depending on output_type, updated metadata or None).
        """
        logger.info(f"Running {type(self).__name__} with config: {self.config}")
        logger.info(f"StackProcessor.run: received workers={workers} (type: {type(workers)})")
        # Ensure workers is a valid positive integer (not -1 which means "all cores")
        iterator = ArrayIterator(stack, self.config.iterator_config)
        n_iterations = len(iterator)
        if n_iterations == 1:
            results = self._process_slice(
                stack, *args, metadata=metadata, **kwargs
            )
        else:
            logger.info(f"Using Parallel with n_jobs={workers} for {n_iterations} iterations")
            results = Parallel(n_jobs=workers, verbose=verbose, batch_size=self.config.batch_size)(
                delayed(self._process_slice)(
                    stack_slice, *args, metadata=metadata, **kwargs)
                for stack_slice in iterator
            )
            # reshape results
            results = self._reshape_slice_results(results, slice_indices=iterator.indices, input_shape=stack.shape, output_shape=self.config.output_shape)
        updated_metadata = self._update_metadata(stack, results, *args, metadata=metadata, **kwargs)
        return (results, updated_metadata)

    def _update_after_resize(self, orig_shape, new_shape, new_metadata: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        change = np.array(orig_shape) / np.array(new_shape)
        logger.info(f"Updating metadata with new shape ratio: {change}")
        
        # Update shape if present
        if "shape" in new_metadata:
            new_metadata["shape"] = tuple(new_shape)
        
        # Get axes labels to map change array indices to axis letters
        axes = new_metadata.get("axes", [])
        
        # Update dims if present
        if "dims" in new_metadata:
            if axes:
                # Use axes list directly (Dimensions accepts list or string)
                new_metadata["dims"] = Dimensions(axes, tuple(new_shape))
            else:
                # Try to preserve existing dims structure if possible
                if hasattr(new_metadata["dims"], "order"):
                    dim_order = new_metadata["dims"].order
                    new_metadata["dims"] = Dimensions(dim_order, tuple(new_shape))
                else:
                    # Fallback: create default dimension order
                    dim_order = "".join([chr(ord('A') + i) for i in range(len(new_shape))])
                    new_metadata["dims"] = Dimensions(dim_order, tuple(new_shape))
        
        if not axes:
            logger.warning("No axes information in metadata, cannot update scale and physical_pixel_sizes")
            return new_metadata
        
        # Update scale: multiply each axis's scale value by the corresponding change ratio
        if "scale" in new_metadata and new_metadata["scale"] is not None:
            scale_dict = new_metadata["scale"]._asdict()
            # Map each dimension in change to its axis letter and update the scale
            for i, axis_letter in enumerate(axes):
                if i < len(change) and axis_letter in scale_dict and scale_dict[axis_letter] is not None:
                    scale_dict[axis_letter] = float(scale_dict[axis_letter] * change[i])
            new_metadata["scale"] = Scale(**scale_dict)
        
        # Update physical_pixel_sizes: multiply each axis's pixel size by the corresponding change ratio
        pps_dict = None
        if "physical_pixel_sizes" in new_metadata and new_metadata["physical_pixel_sizes"] is not None:
            pps = new_metadata["physical_pixel_sizes"]
            # Handle both namedtuple and other types
            if hasattr(pps, '_asdict'):
                pps_dict = pps._asdict()
                # Map each dimension in change to its axis letter and update the pixel sizes
                for i, axis_letter in enumerate(axes):
                    if i < len(change) and axis_letter in pps_dict and pps_dict[axis_letter] is not None:
                        pps_dict[axis_letter] = float(pps_dict[axis_letter] * change[i])
                # Reconstruct the namedtuple with the same type
                new_metadata["physical_pixel_sizes"] = type(pps)(**pps_dict)
            elif hasattr(pps, '_fields'):
                # It's a namedtuple, use _fields to get field names
                pps_dict = pps._asdict()
                for i, axis_letter in enumerate(axes):
                    if i < len(change) and axis_letter in pps_dict and pps_dict[axis_letter] is not None:
                        pps_dict[axis_letter] = float(pps_dict[axis_letter] * change[i])
                new_metadata["physical_pixel_sizes"] = type(pps)(**pps_dict)
            else:
                logger.warning(f"Cannot update physical_pixel_sizes: unsupported type {type(pps)}")
            
        return new_metadata

    def _update_metadata(self, stack, results, *args, metadata: Optional[dict[str, Any]] = None, **kwargs) -> Optional[dict[str, Any]]:
        """Update the metadata with the new shape.

        Args:
            stack: Original input stack.
            results: Processed results.
            *args: Additional arguments.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.

        Returns:
            Updated metadata or None.
        """
        if metadata is None:
            return None
        # Store original metadata for comparison
        original_metadata = metadata.copy()
        new_metadata = metadata.copy()
        new_metadata = self._update_after_resize(stack.shape, results.shape, new_metadata)
        
        # Compare and log changed key:value pairs
        changed_keys = []
        for key in set(list(original_metadata.keys()) + list(new_metadata.keys())):
            original_value = original_metadata.get(key)
            new_value = new_metadata.get(key)
            if original_value != new_value:
                changed_keys.append((key, original_value, new_value))
        
        if changed_keys:
            logger.info(f"Metadata updated in {type(self).__name__}: {len(changed_keys)} key(s) changed")
            for key, old_val, new_val in changed_keys:
                logger.info(f"  {key}: {old_val} -> {new_val}")
        else:
            logger.debug(f"Metadata unchanged in {type(self).__name__}")
        
        return new_metadata

    def _process_slice(
        self, slice: np.ndarray, *args, metadata: Optional[dict[str, Any]] = None, **kwargs) -> np.ndarray | tuple[Any,...]:
        """Process a single slice of the image stack.

        This method must be implemented by subclasses to define the actual
        processing logic for each slice.

        Args:
            slice: Single slice from the image stack.
            *args: Additional arguments.
            workers: Number of workers (for nested parallelism).
            verbose: Verbosity level.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.
        Returns:
            Processed slice result.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        logger.info(f"Processing slice with shape {slice.shape}")
        raise NotImplementedError("Subclasses must implement this method")

    def _reshape_slice_results(self, results: list[Any], slice_indices: list[tuple[int,...]], input_shape: tuple[int,...], output_shape: Optional[tuple[int,...]] = None) -> np.ndarray | tuple[Any,...]:
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
            target_shape = output_shape if output_shape is not None else input_shape
            results = results.reshape(target_shape)
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
        elif self.config.output_type == "dataframe":
            results = pd.concat(results, ignore_index=True)
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

    @task(name="ChainProcessor.run")
    def run(
        self, stack: np.ndarray, *args, workers: int = -1, verbose: int = 10, metadata: Optional[dict[str, Any]] = None, **kwargs ) -> np.ndarray:
        """Run the chain processor on an image stack.

        Args:
            stack: Input image stack.
            *args: Additional arguments to pass to each processor.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for parallel processing.
            metadata: Optional metadata to pass to the processor.
            **kwargs: Additional keyword arguments to pass to the processor.
        Returns:
            Processed result array or tuple of lists depending on output_type.
        """
        result = stack
        for processor in self.config.processors:
            result = processor.run(result, *args, workers=workers, verbose=verbose, metadata=metadata, **kwargs)
        return result