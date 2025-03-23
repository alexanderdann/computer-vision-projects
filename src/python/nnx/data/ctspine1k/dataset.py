"""All components related to loading CTSpine1K data."""

from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import cv2
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from torch.utils.data import Dataset

import nnx.data
from nnx.data.data_structures import AnnotationSample, ImagePrompt, SAMSample

BOUND = dict[str, torch.Tensor | str, list[torch.Tensor] | str, list[list[torch.Tensor]]]  # noqa: E501
SAMTorchSample = TypeVar("SAMTorchSample", bound=BOUND)


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


class CTSpine1K:
    """Dataloading iterator for the CTSpine1K dataset."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        volumetric: bool = False,
    ) -> None:
        """C'tor of CTSpine1K dataset.

        Args:
            cache_dir: points to the directory where the data is downloaded.
            volumetric: whether we want to use 3D or 2D data.

        """
        self._volumetric: bool = volumetric

        if self.volumetric:
            msg = "Volumetric processing currently not supported"
            raise NotImplementedError(msg)

        self._lookup = self._validate_check(cache_dir=cache_dir)

        print(f"Found {len(self)} samples.")

    @property
    def volumetric(self) -> bool:
        """Mode indicating whether we use 3D or 2D data."""
        return self._volumetric

    def __len__(self) -> int:
        """Length attribute of the class.

        Returns:
            Return the amount of samples based on mode.

        """
        if self.volumetric:
            return len(self._lookup)

        return sum(self._lookup.values())

    def _validate_check(self, cache_dir: Path) -> dict:
        candidates = {
            file.parent
            for file in cache_dir.rglob("*")
            if "nii.gz" in file.as_uri().lower()
        }
        samples = {}

        for candidate in candidates:
            file_path = candidate / "image.nii.gz"
            segmentation_path = candidate / "segmentation.nii.gz"

            if (file_path).is_file() and (segmentation_path).is_file():
                image_slices = self._get_sample_length(file_path)
                segmentation_slices = self._get_sample_length(segmentation_path)
                if image_slices != segmentation_slices:
                    msg = (
                        f"Image slices ({len(image_slices)}) are not equal to "
                        "segmentation slices ({len(segmentation_slices)})"
                    )
                    raise ValueError(msg)

                samples[candidate] = image_slices

        return samples

    @lru_cache(maxsize=1)  # since it does not change  # noqa: B019
    def _sorted_lookup(self) -> list[Path]:
        return sorted(self._lookup.keys())

    @staticmethod
    def _get_sample_length(file_path: Path) -> int:
        expected_ndim = 3
        img = nib.load(file_path)  # this only loads the header
        shape = img.shape
        assert len(shape) == expected_ndim
        return shape[2]

    @staticmethod
    def _sliced_sample(path: Path, slice_idx: int, axis: int = 2) -> np.ndarray:
        img = nib.load(path)  # this only loads the header
        shape = img.shape

        if slice_idx < 0 or slice_idx >= shape[axis]:
            msg = f"Slice index {slice_idx} out of bounds (0-{shape[axis] - 1})"
            raise ValueError(msg)

        # Create slicer to extract only the slice we want
        slicer = [slice(None)] * len(shape)
        slicer[axis] = slice(slice_idx, slice_idx + 1)

        data_slice = img.dataobj[tuple(slicer)]
        return np.squeeze(data_slice).copy()

    @staticmethod
    def _volumetric_sample(path: Path) -> None:
        msg = "Volumetric processing currently not supported"
        raise NotImplementedError(msg)

    def _resolve_index(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._volumetric:
            path: Path = self._sorted_lookup[index]
            return (
                self._volumetric_sample(path / "image.nii.gz"),
                self._volumetric_sample(path / "segmentation.nii.gz"),
            )

        lower: int = 0

        for path, value in self._lookup.items():
            upper = lower + value

            if lower <= index < upper:
                return (
                    self._sliced_sample(path / "image.nii.gz", index - lower),
                    self._sliced_sample(path / "segmentation.nii.gz", index - lower),
                )

            lower = upper

        msg = f"Trying access out of bound entry for dataset with {len(self)}."
        raise ValueError(msg)

    def __getitem__(self, index: int) -> AnnotationSample:
        """Fetch a single item given the index.

        Args:
            index: number representing the given entry.

        Returns:
            A single training sample as a tuple of input and output for training.

        """
        input_image, annotation = self._resolve_index(index)

        input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX) / 255

        bitmasks = []
        for c_idx in set(np.unique(annotation)):
            if c_idx:  # we skip background
                bitmask = np.zeros_like(annotation)
                bitmask[annotation == c_idx] = 1
                bitmask = binary_erosion(bitmask, iterations=1)

                if np.sum(bitmask) < 1:
                    # after erosion we have an empty mask
                    continue

                bitmasks.append(bitmask.astype(np.float32))

        return AnnotationSample(image=input_image, bitmasks=bitmasks)


class SAMAdapter(Dataset):
    """Wraps the CTSpine1K dataset to enable training compliant with SAM familiy."""

    def __init__(
        self,
        dataset: CTSpine1K,
        image_size: int,
    ) -> None:
        """C'tor of SAMAdapter.

        Args:
            dataset: class which forwards samples of the CTSpine1K dataset.
            image_size: size of the image. Since this is a single int we assume
                a quadratic image.

        """
        super().__init__()
        self._dataset = dataset
        self._image_size = image_size

    def _convert_point_prompts(
        self,
        prompts: list[ImagePrompt],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert point prompts into expected format.

        Returns:
            Tuple containing the prompts (3D) and labels (2D).

        """
        coordinates, labels = [], []
        for prompt in prompts:
            coordinates.append(
                [prompt.y * self._image_size, prompt.x * self._image_size],
            )
            labels.append(prompt.positive)

        return torch.Tensor(coordinates)[None], torch.Tensor(labels)[None]

    def __len__(self) -> int:
        """Forward the same length as of the initial dataset.

        Returns:
            Number of samples in dataset.

        """
        return len(self._dataset)

    @staticmethod
    def collate_fn(samples: list[SAMSample]) -> SAMTorchSample:
        """Collate function to deal with variable shapes.

        Returns:
            Dictionary containing all elements. To access a single batch
            just access the modality you are interested and the batch count.

        """
        images = torch.stack([torch.Tensor(s.image).permute(2, 0, 1) for s in samples])

        mask_tensors = []
        point_coords = []
        point_labels = []

        for sample in samples:
            sample_masks = [torch.tensor(mask)[None] for mask in sample.bitmasks]
            mask_tensors.append(sample_masks)

            sample_coords = []
            sample_labels = []

            for mask_points in sample.points:
                if mask_points:
                    coords = [[p.y * p.width, p.x * p.height] for p in mask_points]
                    coords = torch.Tensor(coords)[None]  # Shape -> [1, num_points, 2]

                    labels = [1 if p.positive else 0 for p in mask_points]
                    labels = torch.Tensor(labels)[None]  # Shape ->  [1, num_points]
                else:
                    coords = torch.zeros((1, 0, 2))
                    labels = torch.zeros((1, 0))

                sample_coords.append(coords)
                sample_labels.append(labels)

            point_coords.append(sample_coords)
            point_labels.append(sample_labels)

        return {
            "images": images,
            "masks": mask_tensors,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    def __getitem__(self, idx: int) -> SAMSample:
        """Fetch a single SAMSample.

        Mind that in the case where we have no bitmasks,
        there is also nothing to finetune for us.

        Returns:
            A SAMSample containing the image, masks and prompts.

        """
        sample: AnnotationSample = self._dataset[idx]

        image_rgb = [sample.image[:, :, None] for _ in range(3)]
        image_rgb = np.concatenate(image_rgb, axis=-1)
        points = [sample_prompts(mask) for mask in sample.bitmasks]

        return SAMSample(
            image=image_rgb,
            bitmasks=sample.bitmasks,
            points=points,
        )
