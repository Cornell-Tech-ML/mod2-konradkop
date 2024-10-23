from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function to apply.
            start (float, optional): The initial value for the reduction. Defaults to 0.0.


        Returns:
        -------
            A collection of tensor functions

        """
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.sub_zip = ops.zip(operators.sub)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.gt_zip = ops.zip(operators.gt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce
            start: initial value where the function starts

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """Applies a mapping function to each element of an input storage
        and stores the results in an output storage. This function
        supports broadcasting to accommodate different input and output shapes.

        Args:
        ----
            out (Storage): The output storage where results will be stored,
                         structured according to `out_shape` and accessed using `out_strides`.
            out_shape (Shape): The shape of the output tensor, defining the dimensions of `out`.
            out_strides (Strides): The strides for accessing elements in the output storage,
                                  specifying the step sizes for each dimension in `out_shape`.
            in_storage (Storage): The input storage containing the elements to be mapped,
                                structured according to `in_shape` and accessed using `in_strides`.
            in_shape (Shape): The shape of the input tensor, defining the dimensions of `in_storage`.
            in_strides (Strides): The strides for accessing elements in the input storage,
                                specifying the step sizes for each dimension in `in_shape`.

        Returns:
        -------
        The function works by:
        1. Initializing index arrays for input and output positions.
        2. Determining the total number of output elements based on its shape.
        3. Iterating through each position in the output array:
           - Converting the current output index to the corresponding input index using broadcasting.
           - Calculating the memory positions in the input and output storage.
           - Applying a specified mapping function (fn) to the input element and storing the result in the output storage.

        Note: The mapping function (fn) must be defined in the outer scope of this function.

        """
        # Initialize indices
        in_index = np.zeros(len(in_shape), dtype=int)
        out_index = np.zeros(len(out_shape), dtype=int)

        # Determine the total number of output elements
        out_size = 1
        for i in out_shape:
            out_size *= i

        # Iterate through each position in the output array
        for i in range(out_size):
            # Convert the current position in the output array to an index
            to_index(i, out_shape, out_index)

            # Map the output index to the corresponding input index using broadcasting
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get the position in the input storage
            in_position = index_to_position(in_index, in_strides)

            # Get the position in the output array
            out_position = index_to_position(out_index, out_strides)

            # Apply the function on the input element and assign the result to the output array
            out[out_position] = fn(in_storage[in_position])

    return _map

    # TODO: Implement for Task 2.3.
    # raise NotImplementedError("Need to implement for Task 2.3")


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Initialize indices to traverse the arrays
        out_index = np.zeros(len(out_shape), dtype=int)  # Index for the output array
        a_in = np.zeros(
            len(a_shape), dtype=int
        )  # Index for array 'a' (after broadcasting)
        b_in = np.zeros(
            len(b_shape), dtype=int
        )  # Index for array 'b' (after broadcasting)

        # Iterate over every element in the output array
        numpyArray = np.prod(out_shape)  # np.prod computes the total number of elements
        numpyArray = numpyArray.astype(int)
        for i in range(numpyArray):
            # Convert flat index 'i' into multi-dimensional index for 'out'
            to_index(i, out_shape, out_index)

            # Compute the linear index in 'out' using strides
            index = index_to_position(out_index, out_strides)

            # Handle broadcasting for array 'a'
            broadcast_index(out_index, out_shape, a_shape, a_in)
            a_value = a_storage[
                index_to_position(a_in, a_strides)
            ]  # Get the value from 'a'

            # Handle broadcasting for array 'b'
            broadcast_index(out_index, out_shape, b_shape, b_in)
            b_value = b_storage[
                index_to_position(b_in, b_strides)
            ]  # Get the value from 'b'

            # Perform the operation (fn) on the elements from 'a' and 'b', and store in 'out'
            out[index] = fn(a_value, b_value)
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        """Reduces a tensor along a specified dimension by applying a reduction function
        to the elements of the input storage and storing the result in the output storage.

        Args:
        ----
            out (Storage): The output storage where the reduced values will be stored,
                         structured according to `out_shape` and accessed using `out_strides`.
            out_shape (Shape): The shape of the output tensor, defining the dimensions of `out`.
            out_strides (Strides): The strides for accessing elements in the output storage,
                                  specifying the step sizes for each dimension in `out_shape`.
            a_storage (Storage): The input storage containing the elements to be reduced,
                               structured according to `a_shape` and accessed using `a_strides`.
            a_shape (Shape): The shape of the input tensor, defining the dimensions of `a_storage`.
            a_strides (Strides): The strides for accessing elements in the input storage,
                               specifying the step sizes for each dimension in `a_shape`.
            reduce_dim (int): The dimension along which to perform the reduction operation.

        Returns:
        -------
        The function works by:
        1. Initializing an index array to keep track of positions in the input tensor.
        2. Iterating over each position in the output tensor.
        3. For each output position, iterating over the elements along the specified `reduce_dim`:
           - Modifying the output index to access the corresponding slice in the input tensor.
           - Applying the reduction function to accumulate values from the input tensor.
        4. Storing the accumulated reduced value back into the output storage.

        Note: The reduction function (fn) must be defined in the outer scope of this function.

        """
        out_index = np.zeros(len(a_shape), dtype=int)

        # Iterate over all elements of the output array
        for output_idx in range(len(out)):
            # Convert the linear index `output_idx` to the multi-dimensional `out_index` in `out_shape`
            to_index(output_idx, out_shape, out_index)

            # Get the corresponding flat index in the output array using strides
            out_pos = index_to_position(out_index, out_strides)

            # Initialize the result for this position with the current value in `out`
            reduced_value = out[out_pos]

            # Loop over the dimension that we are reducing (`reduce_dim`)
            for input_idx in range(a_shape[reduce_dim]):
                # Modify `out_index` along the `reduce_dim` to access the appropriate slice
                a_index = out_index.copy()
                a_index[reduce_dim] = input_idx

                # Get the position in `a_storage` based on `a_index` and `a_strides`
                a_pos = index_to_position(a_index, a_strides)

                # Apply the reduction function to accumulate values
                reduced_value = fn(a_storage[a_pos], reduced_value)

            # Store the reduced value back into the output array
            out[out_pos] = reduced_value
            # TODO: Implement for Task 2.3.
            # raise NotImplementedError("Need to implement for Task 2.3")

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
