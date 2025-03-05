"""Implements loss functions."""

import numpy as np

from nnx.autograd.tensor import Tensor


def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy loss for classification.

    Args:
        predictions: Model predictions (after softmax)
        targets: One-hot encoded target labels

    Returns:
        Loss value as a Tensor with gradient connections preserved

    """
    epsilon = 1e-10
    log_probs = np.log(predictions.data + epsilon)
    loss_val = -np.sum(targets.data * log_probs) / targets.data.shape[0]
    loss = Tensor(loss_val, requires_grad=predictions.requires_grad)

    if predictions.requires_grad:
        loss.prev = {predictions}

        def _backward() -> None:
            batch_size = predictions.data.shape[0]
            grad = (predictions.data - targets.data) / batch_size
            predictions.grad = (
                grad if predictions.grad is None else predictions.grad + grad
            )

        loss.register_backward(_backward)

    return loss
