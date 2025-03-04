"""Activation functions for neural networks."""

import numpy as np

from nnx.autograd.layers import Layer
from nnx.autograd.tensor import Tensor


class ReLU(Layer):
    """Implements the Rectified Linear Unit activation function."""

    def __init__(self) -> None:
        """C'tor of ReLU."""
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:  # noqa: PLR6301
        """Forward pass of ReLU activation.

        Args:
            inputs: Tensor of input values.

        Returns:
            Tensor with ReLU activation applied.

        """
        outputs = Tensor(
            np.maximum(0, inputs.data),
            requires_grad=inputs.requires_grad,
        )

        if inputs.requires_grad:
            outputs.prev = {inputs}

            def _backward() -> None:
                if outputs.grad is not None:
                    # Derivative of ReLU: 1 if x > 0, 0 otherwise
                    grad = (inputs.data > 0).astype(np.float64) * outputs.grad
                    inputs.grad = grad if inputs.grad is None else inputs.grad + grad

            outputs.register_backward(_backward)

        return outputs


class Softmax(Layer):
    """Implements the Softmax activation function."""

    def __init__(self, axis: int = -1) -> None:
        """C'tor of Softmax.

        Args:
            axis: The axis along which to apply softmax.

        """
        super().__init__()
        self._axis = axis

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of Softmax activation.

        Args:
            inputs: Tensor of input values.

        Returns:
            Tensor with Softmax activation applied.

        """
        # Shifting for numerical stability by the max.
        # This does not alter the results as it cancels out in numerator and denominator.
        shifted = inputs.data - np.max(inputs.data, axis=self._axis, keepdims=True)
        exp_values = np.exp(shifted)
        softmax_output = exp_values / np.sum(exp_values, axis=self._axis, keepdims=True)

        outputs = Tensor(
            softmax_output,
            requires_grad=inputs.requires_grad,
        )

        if inputs.requires_grad:
            outputs.prev = {inputs}

            def _backward() -> None:
                if outputs.grad is not None:
                    # The Jacobian of softmax is complex
                    # For each sample in the batch, compute proper Jacobian
                    batch_size = inputs.data.shape[0]
                    dx = np.zeros_like(inputs.data)

                    for batch_idx in range(batch_size):
                        softmax = softmax_output[batch_idx]
                        dout = outputs.grad[batch_idx]

                        # Reshaping for matrix operations
                        softmax_reshaped = softmax.reshape(-1, 1)
                        dout_reshaped = dout.reshape(-1, 1)

                        # Jacobian of softmax: diag(softmax) - softmax * softmax^T
                        jacobian = np.diagflat(softmax) - np.dot(softmax_reshaped, softmax_reshaped.T)

                        # Apply chain rule: dx = J * dout
                        dx[batch_idx] = np.dot(jacobian, dout_reshaped).reshape(softmax.shape)

                    inputs.grad = dx if inputs.grad is None else inputs.grad + dx

            outputs.register_backward(_backward)

        return outputs
