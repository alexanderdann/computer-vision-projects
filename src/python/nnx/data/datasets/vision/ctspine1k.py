"""All components related to loading CTSpine1K data."""

from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

from nnx.data import DataSplit, LoadingMode
from nnx.data.data_structures import AnnotationSample


class CTSpine1K:
    """Dataloading iterator for the CTSpine1K dataset."""

    def __init__(
        self,
        cache_dir: Path,
        loading_mode: LoadingMode = LoadingMode.ON_DEMAND,
        *,
        split: DataSplit = DataSplit.TRAINING,
        volumetric: bool = False,
    ) -> None:
        """C'tor of CTSpine1K dataset.

        Args:
            cache_dir: points to the directory where the data is downloaded.
            loading_mode: loading logic of data
            split: what part of the data we want to use.
            volumetric: whether we want to use 3D or 2D data.

        """
        self._loading_mode: LoadingMode = loading_mode
        self._volumetric: bool = volumetric

        if self.volumetric:
            msg = "Volumetric processing currently not supported"
            raise NotImplementedError(msg)

        self._lookup = self._validate_check(cache_dir=cache_dir, split=split)

        self._loaded_volumes = None
        if self._loading_mode == LoadingMode.PRELOAD_RAM:
            self._loaded_volumes: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
            self._preload_all_volumes()

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

        return sum(sample[2] for sample in self._lookup)

    @staticmethod
    def _load_split(file: str, split: DataSplit) -> tuple[list[str]]:
        """Load the training, validation and test split.

        Currently this component assumes that the names always come in pairs
        with identical names in the data and labels. The only difference is the
        additional '_labels' after the name off the file.

        Raises:
            ValueError: In the case where we get a split object we cannot interpret.

        Returns:
            Tuples containg the lists for each split.

        """
        split_file = Path(file)
        split_data = split_file.read_text()
        split_list = split_data.split("\n")

        train_ident = split_list.index("trainset:")
        test_public_ident = split_list.index("test_public:")
        test_private_ident = split_list.index("test_private:")

        if split == DataSplit.TRAINING:
            return split_list[
                train_ident + 1 : test_public_ident - 1
            ]  # removing white space

        if split == DataSplit.VALIDATION:
            return split_list[
                test_public_ident + 1 : test_private_ident - 1
            ]  # removing white space

        if split == DataSplit.TEST:
            return split_list[test_private_ident + 1 :]

        msg = "Got invalid DataSplit."
        raise ValueError(msg)

    def _validate_check(self, cache_dir: Path, split: DataSplit) -> dict:
        """Validate that all data is downloaded which fulfills our assumptions.

        The idea is that due to the naming of the files we will have a perfect match
        The first half contains the input nii.gz and the second half has the
        corresponding files with the additonal '_seg.nii.gz'.

        Raises:
            ValueError: In the case where we cannot find the data we want.

        Returns:
            Lookup with pairs to easily work with.

        """
        dirs = {dir_.stem for dir_ in cache_dir.iterdir()}
        if not dirs.issuperset({"raw_data", "metadata"}):
            msg = (
                f"Cannot find the 'raw_data' or 'metadata' directory in {cache_dir!s}. "
                "Ensure you point to the toplevel of the dataset."
            )
            raise ValueError(msg)

        meta_dir = cache_dir / "metadata"
        data_dir = cache_dir / "raw_data"

        files_of_interest = self._load_split(meta_dir / "data_split.txt", split=split)

        table = []
        for subset in ["COLONOG", "COVID-19", "HNSCC-3DCT-RT", "MSD-T10"]:
            for file in (data_dir / "volumes" / subset).glob("*.nii.gz"):
                # We cannot just stem due to '.gz' being dropped in this operation
                if str(file.parts[-1]) not in files_of_interest:
                    continue

                # mind that stem is returned without .gz so we add it
                stem = file.stem.replace(".nii", "_seg.nii.gz")
                segmentation = data_dir / "labels" / subset / stem

                if not segmentation.is_file():
                    msg = f"Could not find {segmentation!s}"
                    raise ValueError(msg)

                image_slices = self._get_sample_length(file)
                segmentation_slices = self._get_sample_length(segmentation)

                if image_slices != segmentation_slices:
                    msg = (
                        f"Invalid assumptions for {file!s} and {segmentation!s}."
                        f"Image slices ({image_slices}) are not equal to "
                        f"segmentation slices ({segmentation_slices})."
                    )
                    raise ValueError(msg)

                table.append((file, segmentation, image_slices))

        return table

    def _preload_all_volumes(self) -> None:
        progress = tqdm(self._lookup)

        for inputs, labels, _ in progress:
            if inputs in self._loaded_volumes:
                return

            img_vol = nib.load(inputs).get_fdata()
            seg_vol = nib.load(labels).get_fdata()

            # Store in memory
            self._loaded_volumes[str(inputs)] = (img_vol, seg_vol)

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
            inputs, labels, _ = self._lookup[index]
            return (
                self._volumetric_sample(inputs),
                self._volumetric_sample(labels),
            )

        lower: int = 0

        for inputs, labels, slices in self._lookup:  # faster when using binary search
            upper = lower + slices

            if lower <= index < upper:
                resolved_index = index - lower

                if self._loaded_volumes is not None:
                    raw_ct, seg_ct = self._loaded_volumes[str(inputs)]

                    image = np.squeeze(raw_ct[:, :, resolved_index]).copy()
                    seg_ct = np.squeeze(seg_ct[:, :, resolved_index]).copy()

                    return (image, seg_ct)

                return (
                    self._sliced_sample(inputs, resolved_index),
                    self._sliced_sample(labels, resolved_index),
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

        input_image = input_image.astype(np.float32)
        input_image = cv2.normalize(input_image, None, 0, 1, cv2.NORM_MINMAX)
        input_image = np.clip(input_image, 0, 1)  # for numerical instability safety

        bitmasks = []
        for c_idx in set(np.unique(annotation)):
            if c_idx:  # we skip background
                bitmask = np.zeros_like(annotation)
                bitmask[annotation == c_idx] = 1

                bitmasks.append(bitmask.astype(np.float32))

        return AnnotationSample(image=input_image, bitmasks=bitmasks)
