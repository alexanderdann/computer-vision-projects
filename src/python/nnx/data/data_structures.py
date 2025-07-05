"""Units of data which are used through out the repo.

Since such endeavours can get quickly very complex and thousands of samples are used,
we can use pydantic to validate that each sample fulfills our needs.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class AnnotationSample(BaseModel):
    """Data structure to actively validate training data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    bitmasks: list[np.ndarray]

    @field_validator("image")
    def image_checker(cls, v: np.ndarray) -> np.ndarray:  # noqa: N805, needs to cls
        """Check that image has expected properties.

        Raises:
            ValueError: in case data deviates from expected properties.

        Returns:
            Same array as at input.

        """
        if v.min() < 0:
            msg = "Input image is smaller than 0. We need it to be within [0, 1]"
            raise ValueError(msg)

        if v.max() > 1:
            msg = "Input image is greater than 1. We need it to be within [0, 1]"
            raise ValueError(msg)

        return v

    @field_validator("bitmasks")
    def masks_checker(cls, v: list[np.ndarray]) -> list[np.ndarray]:  # noqa: N805, needs to cls
        """Check that masks have expected properties.

        Raises:
            ValueError: in case data deviates from expected properties.

        Returns:
            Same list of arrays as at input.

        """
        for v_ in v:
            if v_.min() < 0:
                msg = "Mask is smaller than 0. We need it to be within [0, 1]"
                raise ValueError(msg)

            if v_.max() > 1:
                msg = "Mask is greater than 1. We need it to be within [0, 1]"
                raise ValueError(msg)

            set_of_v = set(np.unique(v_))
            excessive_v = set_of_v - {0, 1}

            if len(excessive_v) > 0:
                msg = f"Mask needs to binary, contains unexpected {excessive_v}."
                raise ValueError(msg)

        return v


class ImagePrompt(BaseModel):
    """Represents a single image prompt."""

    x: float
    y: float
    height: int
    width: int
    positive: bool

    @field_validator("x", "y")
    def image_checker(cls, v: float) -> float:  # noqa: N805, needs to cls
        """Check that the coordinates have expected properties.

        Raises:
            ValueError: in case data deviates from expected properties.

        Returns:
            Same array as at input.

        """
        if not (0 <= v <= 1):
            msg = "Coordinate needs to be within [0, 1]"
            raise ValueError(msg)

        return v


class SAMSample(AnnotationSample):
    """Single sample to be used with SAM familiy.

    Theoretically one could add all prompts here. We only use points.
    """

    points: list[list[ImagePrompt]]  # each mask gets multiple prompts
