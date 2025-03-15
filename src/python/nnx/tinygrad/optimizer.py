"""Contains units which are used for optimising the networks."""

import numpy as np

from nnx.tinygrad.tensor import Tensor


class SGD:
    """Implements a Stochastic Gradient Descent."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        clip_value: float = 1.0,
    ) -> None:
        """Initialize SGD optimizer with gradient clipping.

        Raises:
            ValueError: in the case of invalid paameters.

        """
        self._parameters = parameters
        self._lr = lr

        self._clip_value = clip_value

        for name, val in [("Momentum", momentum), ("Dampening", dampening)]:
            if not (0 <= momentum < 1):
                msg = f"{name} must be in [0, 1), got {val}"
                raise ValueError(msg)

        self._momentum = momentum
        self._dampening = dampening
        self._weight_decay = weight_decay
        self._momentum_terms = []

    def step(self) -> None:
        """Execute an optimisation step."""
        for idx, param in enumerate(self._parameters):
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

                clipped_grad = np.clip(grad, -self._clip_value, self._clip_value)

                clipped_grad = self._weight_decay * grad

                if self._momentum:
                    if idx >= len(self._momentum_terms):
                        self._momentum_terms.append(clipped_grad.copy())

                    else:
                        self._momentum_terms[idx] = (
                            self._momentum * self._momentum_terms[idx]
                            + (1 - self._dampening) * clipped_grad
                        )

                    clipped_grad = self._momentum_terms[idx]

                data -= self._lr * clipped_grad

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        for param in self._parameters:
            param.grad = None


class AdamW:
    """Implements the AdamW as in the paper https://arxiv.org/pdf/1711.05101."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
        clip_value: float = 1.0,
    ) -> None:
        """C'tor of the AdamW.

        Args:
            parameters: list of parameters which need to be updated.
            lr: learning rate.
            betas: tuple containing the first and second order momentum.
            weight_decay: the weight decay to be applied to the weights.
            clip_value: clipping the gradients by value.

        Raises:
            ValueError: malformated betas.

        """
        expected_beta_count = 2
        self._parameters = parameters
        self._lr = lr
        self._clip_value = clip_value
        if len(betas) != expected_beta_count:
            msg = f"Betas needs to be tuple of exactly two values, got {betas}."
            raise ValueError(msg)

        self._beta1, self._beta2 = betas
        self._weight_decay = weight_decay

        self._iteration = 0
        self._epsilon = 1e-5

        self._fo_history = []
        self._so_history = []

    def step(self) -> None:
        """Execute an optimisation step."""
        self._iteration += 1
        for idx, param in enumerate(self._parameters):
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

                # Initialize moments if they don't exist yet
                # Using a hashmap would be more elegant but in this simple setting
                # it is okay
                if idx >= len(self._fo_history):
                    self._fo_history.append(np.zeros_like(grad))
                    self._so_history.append(np.zeros_like(grad))

                self._fo_history[idx] = (
                    self._beta1 * self._fo_history[idx] + (1 - self._beta1) * grad
                )
                self._so_history[idx] = (
                    self._beta2 * self._so_history[idx] + (1 - self._beta2) * grad**2
                )

                scaled_fo_momentum = self._fo_history[idx] / (
                    1 - self._beta1**self._iteration
                )
                scaled_so_momentum = self._so_history[idx] / (
                    1 - self._beta2**self._iteration
                )

                denom = np.sqrt(scaled_so_momentum) + self._epsilon
                update = scaled_fo_momentum / denom

                if self._clip_value > 0:  # One could also add clipping by norm
                    update = np.clip(update, -self._clip_value, self._clip_value)

                data -= self._lr * (scaled_fo_momentum / denom)

                if self._weight_decay > 0:
                    data -= self._lr * self._weight_decay * data

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        for param in self._parameters:
            param.grad = None
