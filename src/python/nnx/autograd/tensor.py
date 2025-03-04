"""Tensor object to use for computations like forward/backward pass."""

import numpy as np


class Tensor:
    """Resembles a data structure to carry the data, gradients and topology.

    Inspired by Andrej Karpathy https://github.com/karpathy/micrograd
    """

    def __init__(self, data: np.ndarray, *, requires_grad: bool = False) -> None:
        """C'tor of Tensor.

        Args:
            data: contents associated with the tensor.
            requires_grad: whether we want to store gradients infomation.

        """
        self._data = data
        self._requires_grad = requires_grad
        self._grad: np.ndarray | None = None
        self._backward = (
            lambda: None
        )  # A function that computes the gradient propagation.
        self._prev = set()  # A set of Tensors that were used to compute this one.

    @property
    def data(self) -> np.ndarray:
        """Exposure of internal data."""
        return self._data

    @property
    def requires_grad(self) -> bool:
        """Read only property of whether the Tensor requires a grad."""
        return self._requires_grad

    @property
    def grad(self) -> np.ndarray | None:
        """Return the gradients."""
        return self._grad

    @grad.setter
    def grad(self, value: np.ndarray | None) -> None:
        self._grad = value

    @property
    def prev(self) -> set:
        """Return the previous nodes."""
        return self._prev

    @prev.setter
    def prev(self, value: set) -> None:
        self._prev = value

    @property
    def T(self) -> "Tensor":  # noqa: N802
        """Transpose the given tensor.

        Returns:
            Transposed version of the tensor.

        """
        result = Tensor(self.data.T, requires_grad=self.requires_grad)

        if self.requires_grad:
            result.prev = {self}

            def _backward() -> None:
                if result.grad is not None:
                    # Gradient of transpose is just the transpose of the gradient
                    grad = result.grad.T
                    self.grad = grad if self.grad is None else self.grad + grad

            result.register_backward(_backward)

        return result

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Overloading for matrix multiplication.

        Args:
            other: The tensor to multiply with.

        Returns:
            Result of matrix multiplication.

        """
        result = Tensor(
            self.data @ other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
        )

        if self.requires_grad or other.requires_grad:
            result.prev = {self, other}

            def _backward() -> None:
                if result.grad is not None:
                    if self.requires_grad:
                        # Gradient with respect to self: dL/dA = dL/dC @ B.T
                        grad = result.grad @ other.data.T
                        self.grad = grad if self.grad is None else self.grad + grad

                    if other.requires_grad:
                        # Gradient with respect to other: dL/dB = A.T @ dL/dC
                        grad = self.data.T @ result.grad
                        other.grad = grad if other.grad is None else other.grad + grad

            result.register_backward(_backward)

        return result

    def __add__(self, other: "Tensor") -> "Tensor":
        """Overloading for addition operation.

        Returns:
            A newly created Tensor after addition.

        """
        result = Tensor(
            self.data + other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
        )

        if self.requires_grad or other.requires_grad:
            result.prev = {self, other}

            def _backward() -> None:
                if result.grad is not None:
                    if self.requires_grad:
                        self.grad = (
                            result.grad
                            if self.grad is None
                            else self.grad + result.grad
                        )

                    if other.requires_grad:
                        other.grad = (
                            result.grad
                            if other.grad is None
                            else other.grad + result.grad
                        )

            result.register_backward(_backward)

        return result

    def register_backward(self, func: callable) -> None:
        """Register the closure to compute backward pass."""
        self._backward = func

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Compute the backward pass."""
        if grad is None:
            grad = np.ones_like(self._data)
        # Accumulate gradients.
        self._grad = grad if self._grad is None else self._grad + grad
        self._backward()
        for t in self._prev:
            t.backward()
