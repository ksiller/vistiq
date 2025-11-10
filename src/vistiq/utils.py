import numpy as np
from pydantic import BaseModel


class Configuration(BaseModel):
    classname: str = None


class ArrayIteratorConfig(Configuration):
    slice_def: tuple[int, ...] = (
        -2,
        -1,
    )


class ArrayIterator:
    """Iterator over a numpy array with custom slicing.

    Iterates over axes not specified in slice_def, keeping axes in slice_def
    as full slices. Supports negative indices in slice_def.

    Examples:
        For array shape (5, 30, 20):
        - slice_def = (-2, -1): Keep last 2 axes, iterate over axis 0
        - slice_def = (0, 1): Keep first 2 axes, iterate over axis 2
    """

    def __init__(self, arr: np.ndarray, config: ArrayIteratorConfig):
        """Initialize the ArrayIterator.

        Args:
            arr (np.ndarray): The array to iterate over.
            config (ArrayIteratorConfig): The configuration for the iterator.
                slice_def specifies which axes to keep (not iterate over).
                Must contain integers (negative indices supported, e.g., -1 for last axis).
        """
        self.arr = arr
        self.shape = arr.shape
        self.config = config
        self.ndim = len(arr.shape)

        # Normalize slice_def: convert negative indices to positive
        normalized_slice_def = tuple(
            axis if axis >= 0 else self.ndim + axis for axis in config.slice_def
        )

        # Validate normalized indices
        for axis in normalized_slice_def:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(
                    f"Invalid axis index {axis} for array with {self.ndim} dimensions"
                )

        # Find axes to iterate over (not in slice_def)
        iter_axes = [i for i in range(self.ndim) if i not in normalized_slice_def]

        # Generate shape for iteration (only over iter_axes)
        iter_shape = tuple(self.shape[axis] for axis in iter_axes)

        # Generate all index tuples for iteration
        self.indices = []
        for iter_indices in np.ndindex(iter_shape):
            # Build full index tuple: specific indices for iter_axes, slice(None) for others
            full_index = [slice(None)] * self.ndim
            for iter_axis_idx, iter_axis in enumerate(iter_axes):
                full_index[iter_axis] = iter_indices[iter_axis_idx]
            self.indices.append(tuple(full_index))

        self.index = 0

    def __iter__(self):
        """Return the iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next slice."""
        if self.index < len(self.indices):
            slice = self.arr[self.indices[self.index]]
            self.index += 1
            return slice
        else:
            raise StopIteration

    def __len__(self) -> int:
        """Return the number of iterations."""
        return len(self.indices)

