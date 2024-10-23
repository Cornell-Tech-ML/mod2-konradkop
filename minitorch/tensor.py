"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    GT,
    Add,
    Sub,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the gradient tracking requirement for the tensor.

        Args:
        ----
            x (bool): If True, enables gradient tracking. If False, disables it.

        This method initializes a history for tracking operations
        if gradient tracking is enabled.

        """
        self.history = History()

    def zero_grad_(self) -> None:
        """Resets the gradient of the tensor to None.

        This method is typically called before performing a new backward
        pass to ensure that gradients from previous iterations do not accumulate.
        """
        self.grad = None

    def requires_grad(self) -> bool:
        """Checks if gradient tracking is enabled for the tensor.

        Returns
        -------
            bool: True if gradient tracking is enabled, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor to create.
                If None, the tensor will have the same shape as the instance.

        Returns:
        -------
            Tensor: A new tensor initialized with zeros and the specified shape.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is constant (i.e., does not require gradients).

        Returns
        -------
            bool: True if the tensor is constant, False otherwise.

        A tensor is considered constant if it has no history of operations
        that would require gradient computation.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Retrieves the input variables (parents) of the current variable.

        Returns
        -------
            Iterable[Variable]: An iterable of input variables that contributed to
            the current variable's value.

        Raises
        ------
            AssertionError: If the variable's history is not set, indicating that
            there are no parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for backpropagation.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to the loss.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A generator yielding tuples of input variables
            and their corresponding gradients.

        Raises:
        ------
            AssertionError: If the history or required attributes are not present, or if
            the number of gradients does not match the number of inputs.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute gradients.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the output with respect
            to the loss. If None, defaults to a tensor of ones with the same shape
            as the current tensor (must be a scalar if shape is not (1,)).

        Raises:
        ------
            AssertionError: If grad_output is None and the tensor is not a scalar.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Performs element-wise division of the current tensor by another tensor.

        Args:
        ----
            b (TensorLike): The tensor to divide by.

        Returns:
        -------
            Tensor: A new tensor resulting from the element-wise division.

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Performs element-wise division of another tensor by the current tensor.

        Args:
        ----
            b (TensorLike): The tensor to be divided by the current tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from the element-wise division.

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # TODO: Implement for Task 2.3.
    # konrad work starts here

    @property
    def dims(self) -> int:
        """Property to access the number of dimensions of the tensor.

        Returns
        -------
            int: The number of dimensions (axes) of the tensor.

        """
        # Access the `dims` attribute of the internal `_tensor` object
        return self._tensor.dims

    @property
    def size(self) -> int:
        """Property to access the total number of elements in the tensor.

        Returns
        -------
            int: The total number of elements in the tensor.

        """
        # Access the `size` attribute of the internal `_tensor` object
        return self._tensor.size

    def __mul__(self, t2: TensorLike) -> Tensor:
        """Multiplies the current tensor with another tensor (t2).
        Uses the Mul operation for element-wise multiplication.
        """
        return Mul.apply(self, self._ensure_tensor(t2))

    def __add__(self, t2: TensorLike) -> Tensor:
        """Adds the current tensor to another tensor (t2).
        Uses the Add operation for element-wise addition.
        """
        return Add.apply(self, self._ensure_tensor(t2))

    def __lt__(self, t2: TensorLike) -> Tensor:
        """Compares the current tensor to another tensor (t2) to check if it is less than t2.
        Returns a tensor of boolean values.
        """
        return LT.apply(self, self._ensure_tensor(t2))

    def __gt__(self, t2: TensorLike) -> Tensor:
        """Compares the current tensor to another tensor (t2) to check if it is greater than t2.
        Returns a tensor of boolean values.
        """
        return GT.apply(self, self._ensure_tensor(t2))

    def __sub__(self, t2: TensorLike) -> Tensor:
        """Subtracts another tensor (t2) from the current tensor.
        Uses the Sub operation for element-wise subtraction.
        """
        return Sub.apply(self, self._ensure_tensor(t2))

    def __neg__(self) -> Tensor:
        """Negates the current tensor, effectively multiplying it by -1.
        Uses the Neg operation.
        """
        return Neg.apply(self)

    def log(self) -> Tensor:
        """Computes the natural logarithm of the current tensor element-wise.
        Uses the Log operation.
        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential of the current tensor element-wise.
        Uses the Exp operation.
        """
        return Exp.apply(self)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid activation function to the current tensor element-wise.
        Uses the Sigmoid operation.
        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU (Rectified Linear Unit) activation function to the current tensor element-wise.
        Uses the ReLU operation.
        """
        return ReLU.apply(self)

    def __eq__(self, t2: TensorLike) -> Tensor:
        """Compares the current tensor to another tensor (t2) to check for equality.
        Returns a tensor of boolean values.
        """
        return EQ.apply(self, self._ensure_tensor(t2))

    def all(self, dim: Optional[Tensor] = None) -> Tensor:
        """Returns a tensor indicating whether all elements along the specified dimension (dim) are True.
        If dim is None, checks across the entire tensor.
        Uses the All operation.
        """
        if dim is not None:
            return All.apply(self, self._ensure_tensor(dim))
        else:
            return All.apply(self.view(self.size), self._ensure_tensor(0))

    def is_close(self, t2: Tensor) -> Tensor:
        """Checks if the elements of the current tensor are close to those of another tensor (t2)
        within a tolerance.
        Returns a tensor of boolean values.
        """
        return IsClose.apply(self, t2)

    def __rmul__(self, t2: TensorLike) -> Tensor:
        """Right-multiplication of the current tensor with another tensor (t2).
        This allows for the syntax `t2 * self`.
        """
        return self * t2

    def __radd__(self, t2: TensorLike) -> Tensor:
        """Right-addition of the current tensor with another tensor (t2).
        This allows for the syntax `t2 + self`.
        """
        return self + t2

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum of elements in the current tensor along the specified dimension (dim).
        If dim is None, computes the sum across all elements.
        Uses the Sum operation.
        """
        input_tensor = self.contiguous()

        if dim is None:
            reshaped_tensor = input_tensor.view(self.size)
            zero_tensor = self._ensure_tensor(0)
            return Sum.apply(reshaped_tensor, zero_tensor)

        else:
            dim_tensor = self._ensure_tensor(dim)

        return Sum.apply(input_tensor, dim_tensor)
    

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of elements in the current tensor along the specified dimension (dim).
        If dim is None, computes the mean across all elements.
        Uses the sum to calculate the mean.
        """
        if dim is not None:
            sum_result = self.sum(dim)
            dim_size = self.shape[dim]
            mean_result = sum_result / dim_size
            return mean_result
        else:
            total_sum = self.sum()
            total_size = self.size
            mean_result = total_sum / total_size
            return mean_result

    def permute(self, *dim: Optional[int]) -> Tensor:
        """Permutes the dimensions of the current tensor according to the specified order (dim).
        Returns a new tensor with the dimensions rearranged.
        """
        return Permute.apply(self, tensor(list(dim)))

    def view(self, *dim: Optional[int]) -> Tensor:
        """Reshapes the current tensor to the specified dimensions (dim) without changing its data.
        Returns a new tensor with the specified shape.
        """
        return View.apply(self, tensor(list(dim)))
