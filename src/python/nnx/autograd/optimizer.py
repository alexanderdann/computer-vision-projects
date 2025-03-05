"""Contains units which are used for optimising the networks."""


import numpy as np

from nnx.autograd.tensor import Tensor


class SGD:
    """Implements the Stochastic Gradient Descent."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        clip_value: float = 1.0,
    ) -> None:
        """Initialize SGD optimizer with gradient clipping."""
        self.parameters = parameters
        self.lr = lr
        self.clip_value = clip_value

    def step(self) -> None:
        """Execute an optimisation step."""
        for param in self.parameters:
            if param.grad is not None:
                # Handle broadcasting
                grad = param.grad
                data = param.data
                if param.grad.shape != data.shape:
                    axes = tuple(range(len(param.grad.shape) - len(data.shape)))
                    grad = np.sum(param.grad, axis=axes)

                    # Handle case where dimensions are aligned but sizes differ
                    if grad.shape != data.shape:
                        grad = np.reshape(grad, data.shape)

                clipped_grad = np.clip(grad, -self.clip_value, self.clip_value)

                data -= self.lr * clipped_grad

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        for param in self.parameters:
            param.grad = None
