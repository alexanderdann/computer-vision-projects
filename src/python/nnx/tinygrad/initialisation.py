"""Functionality used to initialise layers wit specfic statistical properties."""

import numpy as np

from nnx.autograd import rng


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    size: list[int],
    gain: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize the weights using the Xavier Uniform.

    Details can be found in  https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf.

    Args:
        fan_in: the amount of input neurons.
        fan_out: the amout of output neurons.
        size: dimensions of the data to initialise.
        gain: optional factor which can also be applied in PyTorch.
        rng: random number generator object.

    Returns:
        A tuple containing the weights and bias initialized as per Xavier Uniform.

    """
    limit = gain * np.sqrt(6 / (fan_in + fan_out))
    weights = rng.uniform(-limit, limit, size=size)
    bias = np.zeros((fan_out,))

    return weights, bias
