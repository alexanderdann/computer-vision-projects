"""Definiton of some types to make typing more apparent and easier to read."""

from typing import TypeVar

import torch

DatasetT = TypeVar("DatasetT")
TorchDictT = TypeVar(
    "TorchDictT",
    bound=dict[
        str,
        torch.Tensor | str,
        list[torch.Tensor] | str,
        list[list[torch.Tensor]],
    ],
)
