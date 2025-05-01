"""Wrapper to load the actual data using Python."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from download import download_from_google_drive


class CTSpine1K:
    """Dataloading iterator for the CTSpine1K dataset."""

    def __init__(
        self,
        cache_dir: Path,
        split: Literal["training", "validation", "test"],
        *,
        volumetric: bool = False,
        download: bool = False,
        **download_kwargs: dict,
    ) -> None:
        """C'tor of CTSpine1K dataset.

        Args:
            cache_dir: points to the directory where the data is downloaded.
            split: what split of the data should be used for this instance.
            volumetric: whether we want to use 3D or 2D data.
            download: if set to True the whole data is first fetched from
                Google Drive before it is accessible via this wrapper. Data
                is written to the same folder as specified by cache_dir.
                If the data was already loaded once, you can set this to False
                and we assume all data can be found in cache_dir.
            download_kwargs: keyword arguments which can be passed to the
               `download_from_google_drive` call. In most cases the default
               parameters should be sufficient.

        """
        if download:
            download_from_google_drive(cache_dir, **download_kwargs)

        self._volumetric: bool = volumetric
        self._split: str = split

        self._loaded_split = self._load_split(self._split)

        self._lookup = self._validate_check(
            cache_dir=cache_dir,
            files=self._loaded_split,
        )

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

        return sum(elem[2] for elem in self._lookup.values())

    @staticmethod
    def _load_split(split: str) -> list[str]:
        """Load the training, validation and test split.

        Currently this component assumes that the names always come in pairs
        with identical names in the data and labels. The only difference is the
        additional '_labels' after the name off the file.

        Raises:
            ValueError: split string needs to be one of
                [training, validation, test], otherwise
                the error is thrown.

        Returns:
            List containing the names of the files for the
            relevant split.

        """
        split_file = Path(__file__).parent / "data" / "data_split.txt"
        split_data = split_file.read_text()
        split_list = split_data.split("\n")

        if split.lower() == "hnscc":
            pattern = re.compile(r"HN_P\d{3}\.nii\.gz")
            return [f for f in split_list if re.match(pattern, f)]

        train_ident = split_list.index("trainset:")
        test_public_ident = split_list.index("test_public:")
        test_private_ident = split_list.index("test_private:")

        if split == "training":
            return split_list[train_ident + 1 : test_public_ident]

        if split == "validation":
            return split_list[test_public_ident + 1 : test_private_ident]

        if split == "test":
            return split_list[test_private_ident + 1 :]

        msg = (
            f"Got an invalid split: {split}. "
            "Ensure value is one of training, validation, test"
        )
        raise ValueError(msg)

    def _validate_check(self, cache_dir: Path, files: list[str]) -> dict:
        stems = [file.split(".nii.gz")[0] for file in files]

        data_candidates = [
            file
            for file in sorted(cache_dir.rglob("data/*/*"))
            if "nii.gz" in file.as_uri().lower()
        ]
        label_candidates = [
            file
            for file in sorted(cache_dir.rglob("label/*/*"))
            if "nii.gz" in file.as_uri().lower()
        ]

        if len(data_candidates) != len(label_candidates):
            msg = (
                "Data and labels mismatch. Data is probably not downloaded fully. ",
                f"Got {len(data_candidates)} vs {len(label_candidates)}",
            )
            raise RuntimeError(msg)

        pairs = zip(data_candidates, label_candidates, strict=True)
        pairs_lookup = {pair[0].name.split(".")[0]: pair for pair in pairs}
        samples = {}

        for stem in stems:
            file_path, segmentation_path = pairs_lookup[stem]

            if stem not in segmentation_path.name:
                msg = (
                    "Naming convention seems invalid or violated. ",
                    "Ensure all data was downloaded successfully.",
                )
                raise RuntimeError(msg)

            if not ((file_path).is_file() and (segmentation_path).is_file()):
                msg = (
                    "Data is not a file. Ensure all data was downloaded successfully."
                    f"Failed for {file_path} and {segmentation_path}."
                )
                raise RuntimeError(msg)

            image_slices = self._get_sample_length(file_path)
            segmentation_slices = self._get_sample_length(segmentation_path)
            if image_slices != segmentation_slices:
                msg = (
                    f"Invalid assumptions for {file_path} and {segmentation_path}."
                    f"Image slices ({image_slices}) are not equal to "
                    f"segmentation slices ({segmentation_slices})."
                )
                raise ValueError(msg)

            samples[stem] = (file_path, segmentation_path, image_slices)

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
        volume = nib.load(path)

        return volume.get_fdata()

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Fetch a single item given the index.

        Args:
            index: number representing the given entry.

        Raises:
            IndexError: accessing entries which are out of range.

        Returns:
            A single training sample as a tuple of input and output for training.

        """
        if self._volumetric:
            key = self._sorted_lookup()[index]
            path, labels, _ = self._lookup[key]
            return (
                self._volumetric_sample(path),
                self._volumetric_sample(labels),
            )

        lower: int = 0

        for path, labels, value in self._lookup.values():
            upper = lower + value

            if lower <= index < upper:
                return (
                    self._sliced_sample(path, index - lower),
                    self._sliced_sample(labels, index - lower),
                )

            lower = upper

        msg = f"Trying access out of bound entry for dataset with {len(self)}."
        raise IndexError(msg)
