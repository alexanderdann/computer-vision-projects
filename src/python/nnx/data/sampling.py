"""Funtionality related to sampling contents from data such as Prompts."""

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

import nnx
import nnx.data
from nnx.data.data_structures import ImagePrompt


def sample_positive_prompt(mask: np.ndarray) -> ImagePrompt:
    """Sample a single positive point prompt, given the mask.

    Args:
        mask: binary masks representing the area of interest.

    Returns:
        The corresponding ImagePrompt for the sampled position.

    """
    dist: np.ndarray = distance_transform_edt(mask)
    prob: np.ndarray = dist / dist.sum()

    flat_prob: np.ndarray = prob.flatten()
    indices: np.ndarray = np.arange(len(flat_prob))

    # Sample an index according to the probability distribution
    sampled_index = nnx.data.rng.choice(indices, p=flat_prob)

    y, x = np.unravel_index(sampled_index, prob.shape)

    height, width = prob.shape[:2]

    return ImagePrompt(
        x=y / height,
        y=x / width,
        height=height,
        width=width,
        positive=True,
    )


def sample_negative_prompt(
    mask: np.ndarray,
    spatial_constraint: np.ndarray | None = None,
    boundary_width: int = 10,
    safety_margin: int = 5,
) -> ImagePrompt | None:
    """Sample a single negative point prompt, given the mask.

    Args:
        mask: binary masks representing the area of interest.
        spatial_constraint: region in the mask where it is not allowed to draw
            a negative sample. This is crucial for disconnected masks within the
            same class.
        boundary_width: the actual width of the boundary around the safe region to
            draw a negative sample.
        safety_margin: labels can have noise, so we should not sample too close to
            the given binary mask. This parameters influences how large we want this
            margin to be.

    Returns:
        The corresponding ImagePrompt for the sampled position.

    """
    boundary = binary_dilation(mask, iterations=safety_margin)
    safe_boundary = binary_dilation(boundary, iterations=boundary_width)
    safe_boundary[safe_boundary == boundary] = 0

    if spatial_constraint is not None:
        safe_boundary[spatial_constraint] = 0

    if not np.any(safe_boundary):
        safe_boundary = boundary

    # Sample uniformly from boundary
    indices = np.where(safe_boundary)
    idx = nnx.data.rng.integers(0, len(indices[0]), endpoint=False)
    y, x = indices[0][idx], indices[1][idx]

    height, width = mask.shape[:2]

    return ImagePrompt(
        x=y / height,
        y=x / width,
        height=height,
        width=width,
        positive=False,
    )


def sample_prompts(
    mask: np.ndarray,
    spatial_constraint: np.ndarray | None = None,
) -> list[ImagePrompt]:
    """Sample prompts.

    Args:
        mask: binary mask which contains the class of relevance.
        spatial_constraint: whether some region of the image should not
            be used for generation of the prompts.

    Returns:
        List containing the generated prompts.

    """
    ccount, connected_components = cv2.connectedComponents(
        mask.astype(np.uint8),
        connectivity=8,
    )

    max_ccount = 2
    if ccount > max_ccount:  # when having disconnected masks for the same class
        cindices = set(np.unique(connected_components))

        data = []
        for idx in cindices:
            if idx:  # skip background
                cmask = np.zeros_like(connected_components)
                cmask[connected_components == idx] = 1
                no_go = connected_components != 0

                data.extend(sample_prompts(cmask, no_go))

        return data

    data = [sample_positive_prompt(mask=mask)]

    half = 0.5
    third = 1 / 3

    if nnx.data.rng.random() > half:
        data.append(
            sample_positive_prompt(mask=mask),
        )

    if nnx.data.rng.random() > third:
        data.append(
            sample_negative_prompt(mask=mask, spatial_constraint=spatial_constraint),
        )

    return data
