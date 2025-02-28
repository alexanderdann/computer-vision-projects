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
