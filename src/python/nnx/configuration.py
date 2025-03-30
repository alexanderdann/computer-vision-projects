"""Logic to make units configurable."""

from dataclasses import dataclass
from typing import TypeVar

import gin

T = TypeVar("T")


def configurable(cls: T) -> T:
    """Combine gin.configurable and dataclass for configuration.

    Returns:
        A gin configurable dataclass.

    """
    decorated_cls = dataclass(cls)  # Apply dataclass first
    return gin.configurable(decorated_cls)  # Then make it gin configurable
