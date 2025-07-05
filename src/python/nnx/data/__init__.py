"""Imports related to data topics within nnx."""

from enum import Enum, auto

import numpy as np

rng: np.random.Generator = np.random.default_rng(seed=None)


class LoadingMode(Enum):
    """Defines how dataset samples are loaded."""

    ON_DEMAND = "on_demand"  # Load each sample when requested
    PRELOAD_RAM = "preload_ram"  # Preload as much as possible data into RAM


class DataSplit(Enum):
    """Defines the split of the dataset."""

    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()


__all__ = ["DataSplit", "LoadingMode", "rng"]
