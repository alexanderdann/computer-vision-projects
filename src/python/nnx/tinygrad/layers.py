"""Meta class for layers to import."""

from abc import abstractmethod
from typing import Callable

import numpy as np

from nnx import tinygrad
from nnx.tinygrad.tensor import Tensor


def sanity_checks(func: Callable) -> Callable:
    """Check the input to have correct properties.

    Args:
        func: a Callable used to evaluate np arrays.

    Returns:
        The wrapper function.

    """

    def _wrapper(self, inputs: Tensor) -> Tensor:
        expected_dim = 4
        if inputs.data.ndim != expected_dim:
            msg = f"Data needs to have ndim of 4, got {inputs.data.ndim}"
            raise RuntimeError(msg)

        _, _, _, num_channels = inputs.data.shape

        if hasattr(self, "_in_dim") and (num_channels != self._in_dim):
            msg = (
                "Expected number of channels",
                f"to be {self._in_dim} and not {num_channels}.",
            )
            raise RuntimeError(msg)

        return func(self, inputs)

    return _wrapper


class Layer:
    """Meta class for all layers."""

    def __init__(self) -> None:
        """C'tor of Layer."""
        self._parameters: list[Tensor] = []

    @property
    def parameters(self) -> list[Tensor]:
        """Returns the parameters of the layer."""
        return self._parameters

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Logic of the forward pass."""

    # @sanity_checks
    def __call__(self, inputs: Tensor) -> Tensor:
        """Wrap the forward call.

        Returns:
            The result after the forward pass.

        """
        return self.forward(inputs)


class Sequential(Layer):
    """Sequentially applies a list of layers."""

    def __init__(self, *layers: Layer) -> None:
        """Initialize with a sequence of layers."""
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers in sequence.

        Returns:
            The transformed entry based on the defined chain of transforms.

        """
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self) -> list[Tensor]:
        """Get all parameters from all layers."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters)
            else:
                print(f"Skpping layer {layer}")
        return params

    def eval(self) -> "Sequential":
        """Set the model to eval mode.

        Returns:
            The model in eval mode.

        """
        for param in self.parameters:
            param.requires_grad = False

        return self

    def train(self) -> "Sequential":
        """Set the model to train mode.

        Returns:
            The model in train mode.

        """
        for param in self.parameters:
            param.requires_grad = True

        return self


class Conv2D(Layer):
    """Resembles a 2D Convolution."""

    def __init__(  # noqa: PLR0913
        self,
        kernel_size: tuple[int, int],
        in_dim: int,
        out_dim: int,
        padding: int = 0,
        stride: int = 1,
        *,
        initialiser: Callable,
        bias: bool = True,
    ) -> None:
        """C'tor of Conv2D.

        Raises:
            ValueError: for an invalid kernel size.

        Args:
            kernel_size: spatial dimension of the kernel.
            in_dim: count of input neurons.
            out_dim: count of output neurons.
            padding: constant padding to the sides in pixel.
            stride: step size for the convolution operation.
            initialiser: callable to initialise layers.
            bias: whether we want to use the bias term.

        """
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._padding = padding
        self._stride = stride

        expected_size = 2
        if len(kernel_size) != expected_size:
            msg = (
                "Kernel size is expected to be tuple of length two",
                f", not {len(kernel_size)}",
            )
            raise ValueError(msg)

        weights, bias_ = initialiser(
            in_dim,
            out_dim,
            size=(out_dim, *kernel_size, in_dim),
        )
        self._weights = Tensor(weights, requires_grad=True)
        self._parameters.append(self._weights)

        self._bias = None
        if bias:
            self._bias = Tensor(bias_, requires_grad=True)
            self._parameters.append(self._bias)

    @staticmethod
    def _conv2d_forward(  # noqa: PLR0914
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray | None,
        padding: int,
        stride: int,
    ) -> np.ndarray:
        """Calculate spatial convolution on inputs.

        Returns:
            Transformation applied to the image.

        """
        num_samples, height, width, _ = inputs.shape
        output_filters, kernel_height, kernel_width, _ = weights.shape

        out_height = (height + 2 * padding - kernel_height) // stride + 1
        out_width = (width + 2 * padding - kernel_width) // stride + 1

        if padding:
            inputs = np.pad(
                inputs,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="constant",
            )

        # Initialize the output tensor.
        outputs = np.zeros((num_samples, out_height, out_width, output_filters))

        for sample_id in range(num_samples):
            for pixel_h in range(out_height):
                for pixel_w in range(out_width):
                    for filter_id in range(output_filters):
                        start_h = pixel_h * stride
                        start_w = pixel_w * stride

                        patch = inputs[
                            sample_id,
                            start_h : start_h + kernel_height,
                            start_w : start_w + kernel_width,
                            :,
                        ]

                        # Compute with explicit floating-point handling
                        patch_product = patch * weights[filter_id, :, :, :]
                        # Check for NaN/Inf after multiplication
                        if (
                            np.isnan(patch_product).any()
                            or np.isinf(patch_product).any()
                        ):
                            # Use clipping instead of raising error to continue training
                            patch_product = np.clip(patch_product, -1e10, 1e10)

                        results = np.sum(patch_product)

                        if bias is not None:
                            results += bias[filter_id].item()

                        outputs[sample_id, pixel_h, pixel_w, filter_id] = results

        return outputs

    @staticmethod
    def _conv2d_backward(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0917
        inputs: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray | None,
        grad_output: np.ndarray,
        padding: int,
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_samples, in_height, in_width, in_channels = inputs.shape
        out_channels, kernel_height, kernel_width, _ = weights.shape
        _, out_height, out_width, _ = grad_output.shape

        # Initialize gradients
        dx = np.zeros_like(inputs)
        dweight = np.zeros_like(weights)
        dbias = np.zeros_like(bias) if bias is not None else None

        if padding:
            inputs = np.pad(
                inputs,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="constant",
            )

        # Gradients with respect to bias
        # This is basically just a spatial summation of all samples
        if bias is not None:
            dbias = np.sum(grad_output, axis=(0, 1, 2)).reshape(out_channels, 1)

        # Gradients with respect to weights
        # Fairly simple since a single weight only contributes along a matching index
        # Given y(i, j) = ∑_{p'}∑_{q'} x(i+p, j+q)*w(p,q) and L(y(i, j))
        # -> Derivative for a single w(m, n) only valid for x(i+m, j+n)
        # Hence this is nothing else than elemntwise multiplication
        for sample_id in range(num_samples):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    start_h = h_out * stride
                    start_w = w_out * stride

                    input_patch = inputs[
                        sample_id,
                        start_h : start_h + kernel_height,
                        start_w : start_w + kernel_width,
                        :,
                    ]

                    for c_out in range(out_channels):
                        # Calculate with stability checks
                        patch_grad = (
                            input_patch * grad_output[sample_id, h_out, w_out, c_out]
                        )
                        # Check for numerical issues
                        if np.isnan(patch_grad).any() or np.isinf(patch_grad).any():
                            patch_grad = np.clip(patch_grad, -1e10, 1e10)

                        dweight[c_out] += patch_grad

        # Gradients with respect to weights
        # Since the formula from above contains a `x(i+p, j+q)`, the only addition is
        # that we need to take care of the additions. These just lead to a flipped
        # kernel in the end.
        pad_h = kernel_height - 1
        pad_w = kernel_width - 1
        grad_output_padded = np.pad(
            grad_output,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
        )

        # Rotate the weights (equal to flipping along both spatial dimensions)
        # But also take into account that the in_channels and out_channels changed
        flipped_weights = np.flip(np.flip(weights, axis=1), axis=2)
        flipped_weights = np.transpose(flipped_weights, (3, 1, 2, 0))

        for sample_id in range(num_samples):
            for h_in in range(in_height):
                for w_in in range(in_width):
                    for c_in in range(in_channels):
                        # For each input element, convolve the corresponding region
                        # in grad_output with the flipped weights

                        # Extract the patch from padded grad_output
                        grad_patch = grad_output_padded[
                            sample_id,
                            h_in : h_in + kernel_height,
                            w_in : w_in + kernel_width,
                            :,
                        ]

                        # Dot product with the flipped weights for this input channel
                        patch_product = grad_patch * flipped_weights[c_in]
                        # Check for numerical issues
                        if (
                            np.isnan(patch_product).any()
                            or np.isinf(patch_product).any()
                        ):
                            patch_product = np.clip(patch_product, -1e10, 1e10)

                        dx[sample_id, h_in, w_in, c_in] = np.sum(patch_product)

        # Remove padding if it was applied
        if padding:
            dx = dx[:, padding:-padding, padding:-padding, :]

        return dx, dweight, dbias

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the transformation given the inputs.

        Args:
            inputs: Tensor which needs to be transformed.

        Returns:
            Transformed Tensor.

        """
        outputs: np.ndarray = self._conv2d_forward(
            inputs=inputs.data,
            weights=self._weights.data,
            bias=self._bias.data if self._bias is not None else None,
            padding=self._padding,
            stride=self._stride,
        )

        outputs: Tensor = Tensor(
            outputs,
            requires_grad=inputs.requires_grad or self._weights.requires_grad,
        )

        if inputs.requires_grad or self._weights.requires_grad:
            outputs.prev = {inputs, self._weights}
            if self._bias is not None:
                outputs.prev.add(self._bias)

            def _backward() -> None:
                if outputs.grad is not None:
                    dx, dweight, dbias = self._conv2d_backward(
                        inputs.data,
                        self._weights.data,
                        self._bias.data if self._bias is not None else None,
                        outputs.grad,
                        self._padding,
                        self._stride,
                    )

                    if inputs.requires_grad:
                        inputs.grad = dx if inputs.grad is None else inputs.grad + dx

                    if self._weights.requires_grad:
                        self._weights.grad = (
                            dweight
                            if self._weights.grad is None
                            else self._weights.grad + dweight
                        )

                    if (
                        self._bias is not None
                        and self._bias.requires_grad
                        and dbias is not None
                    ):
                        self._bias.grad = (
                            dbias
                            if self._bias.grad is None
                            else self._bias.grad + dbias
                        )

            outputs.register_backward(_backward)

        return outputs


class Reshape(Layer):
    """Layer to reshape tensor while preserving gradient flow."""

    def __init__(self, target_shape) -> None:
        """Initialize with target shape.

        If an element of target_shape is -1, it will be inferred from the input.

        Args:
            target_shape: Shape to reshape to. Can include -1 for inference.

        """
        super().__init__()
        self.target_shape = target_shape

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass to reshape tensor.

        Args:
            inputs: Input tensor to reshape

        Returns:
            Reshaped tensor with gradient connections preserved

        """
        # Store original shape for backward pass
        original_shape = inputs.data.shape

        # Perform reshape
        reshaped_data = inputs.data.reshape(self.target_shape)
        outputs = Tensor(reshaped_data, requires_grad=inputs.requires_grad)

        if inputs.requires_grad:
            # Connect the computational graph
            outputs.prev = {inputs}

            def _backward() -> None:
                if outputs.grad is not None:
                    # Reshape gradient back to original shape
                    grad = outputs.grad.reshape(original_shape)
                    inputs.grad = grad if inputs.grad is None else inputs.grad + grad

            outputs.register_backward(_backward)

        return outputs


class Linear(Layer):
    """Implements an linear transformation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        initialiser: Callable,
        bias: bool = True,
    ) -> None:
        """C'tor of the Linear layer.

        Args:
            in_dim: count of input neurons.
            out_dim: count of output neurons.
            initialiser: callable to initialise layers.
            bias: whether we want to use the bias term.

        """
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim

        weights, bias_ = initialiser(
            in_dim,
            out_dim,
            size=(out_dim, in_dim),
        )

        self._weights = Tensor(weights, requires_grad=True)
        self._parameters.append(self._weights)

        self._bias = None
        if bias:
            self._bias = Tensor(bias_, requires_grad=True)
            self._parameters.append(self._bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the transformation given the inputs.

        Args:
            inputs: Tensor which needs to be transformed.

        Returns:
            Transformed Tensor.

        """
        outputs: Tensor = inputs @ self._weights.T

        if self._bias is not None:
            outputs += self._bias

        return outputs


class Dropout(Layer):
    """Regularistion measure by randomly deactivating neurons."""

    def __init__(self, p: float = 0.5) -> None:
        """C'tor of Dropout.

        Raises:
            ValueError: probability which exceeds the valid range.

        Args:
            p: probability of a neuron be inactive.

        """
        super().__init__()

        if not (0 < p < 1):
            msg = f"Probability needs to be inside (0, 1). Got {p}."
            raise ValueError(msg)

        self._p = p

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply dropout to the inputs.

        Args:
            inputs: tensor which needs to be transformed.

        Returns:
            A transformed version of the inputs.

        """
        if not inputs.requires_grad:
            return inputs

        mask = tinygrad.rng.binomial(1, 1 - self._p, inputs.shape)

        outputs = Tensor(
            inputs.data * mask / (1 - self._p),
            requires_grad=inputs.requires_grad,
        )

        outputs.prev = {inputs}

        def _backward() -> None:
            if outputs.grad is not None:
                grad = outputs.grad * mask / (1 - self._p)
                inputs.grad = grad if inputs.grad is None else inputs.grad + grad

        outputs.register_backward(_backward)

        return outputs


class LayerNorm(Layer):
    """Implements Layer Normalization.

    Normalizes the input across the features dimension (last dimension)
    with learnable affine parameters (gamma and beta).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm with shape parameters.

        Args:
            normalized_shape: Size of the feature dimension to normalize across
            eps: Small constant for numerical stability

        """
        super().__init__()
        self._normalized_shape = normalized_shape
        self._eps = eps

        self._gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self._beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

        self._parameters.append(self._gamma)
        self._parameters.append(self._beta)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply layer normalization to inputs.

        Args:
            inputs: Tensor which needs to be transformed.

        Returns:
            Normalized tensor of same shape with scaling and shifting applied

        Raises:
            ValueError: If the last dimension of inputs doesn't match normalized_shape

        """
        if inputs.data.shape[-1] != self._normalized_shape:
            msg = (
                f"Last dimension of inputs {inputs.data.shape[-1]} "
                f"doesn't match normalized_shape {self._normalized_shape}"
            )
            raise ValueError(msg)

        # Calculate mean and variance along the feature dimension (last dimension)
        # Keep dimensions for proper broadcasting
        mean = np.mean(inputs.data, axis=-1, keepdims=True)
        var = np.var(inputs.data, axis=-1, keepdims=True)

        x_norm = (inputs.data - mean) / np.sqrt(var + self._eps)

        # Scale and shift (broadcast gamma and beta across batch dimensions)
        outputs_data = self._gamma.data * x_norm + self._beta.data

        outputs = Tensor(
            outputs_data,
            requires_grad=(
                inputs.requires_grad
                or self._gamma.requires_grad
                or self._beta.requires_grad
            ),
        )

        if outputs.requires_grad:
            outputs.prev = {inputs, self._gamma, self._beta}

            def _backward() -> None:
                if outputs.grad is not None:
                    # Cache values for reuse in gradients
                    inv_std = 1.0 / np.sqrt(var + self._eps)

                    if self._gamma.requires_grad:
                        gamma_grad = np.sum(
                            outputs.grad * x_norm,
                            axis=tuple(range(outputs.grad.ndim - 1)),
                        )
                        self._gamma.grad = (
                            gamma_grad
                            if self._gamma.grad is None
                            else self._gamma.grad + gamma_grad
                        )

                    if self._beta.requires_grad:
                        beta_grad = np.sum(
                            outputs.grad,
                            axis=tuple(range(outputs.grad.ndim - 1)),
                        )
                        self._beta.grad = (
                            beta_grad
                            if self._beta.grad is None
                            else self._beta.grad + beta_grad
                        )

                    if inputs.requires_grad:
                        num_channels = inputs.data.shape[-1]
                        dx_norm = outputs.grad * self._gamma.data

                        dvar = (
                            -0.5
                            * np.sum(
                                dx_norm * (inputs.data - mean),
                                axis=-1,
                                keepdims=True,
                            )
                            * inv_std**3
                        )
                        dmean = -np.sum(
                            dx_norm * inv_std,
                            axis=-1,
                            keepdims=True,
                        ) + dvar * (-2.0 / num_channels) * np.sum(
                            inputs.data - mean,
                            axis=-1,
                            keepdims=True,
                        )

                        dx = (
                            dx_norm * inv_std
                            + dvar * (2.0 / num_channels) * (inputs.data - mean)
                            + dmean * (1.0 / num_channels)
                        )
                        inputs.grad = dx if inputs.grad is None else inputs.grad + dx

            outputs.register_backward(_backward)

        return outputs
