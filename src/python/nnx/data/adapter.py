"""Collection of Adapters to combine different sources through a single class."""

import numpy as np
import torch
from torch.utils.data import Dataset

from nnx.data.data_structures import AnnotationSample, ImagePrompt, SAMSample
from nnx.data.sampling import sample_prompts
from nnx.data.typing_ import DatasetT, TorchDictT


class SAMAdapter(Dataset):
    """Wraps the CTSpine1K dataset to enable training compliant with SAM familiy."""

    def __init__(
        self,
        dataset: DatasetT,
        image_size: int | tuple[int, int],
    ) -> None:
        """C'tor of SAMAdapter.

        Raises:
            ValueError: for the case of an invalid image_size.

        Args:
            dataset: class which forwards samples of the CTSpine1K dataset.
            image_size: size of the image. Since this is a single int we assume
                a quadratic image.

        """
        super().__init__()
        self._dataset = dataset

        if isinstance(image_size, tuple):
            if image_size[0] != image_size[1]:
                msg = f"For SAM we need images to be quadratic. Not {image_size}"
                raise ValueError(msg)

            image_size = image_size[0]

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
    def collate_fn(samples: list[SAMSample]) -> TorchDictT:
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
