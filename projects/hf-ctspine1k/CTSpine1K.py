"""Wrapper to load the actual data using Python."""

from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import datasets
import nibabel as nib
import numpy as np
from datasets import DownloadManager
from huggingface_hub import HfApi

_CITATION = """
@misc{deng2024ctspine1klargescaledatasetspinal,
      title={CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation in Computed Tomography},
      author={Yang Deng and Ce Wang and Yuan Hui and Qian Li and Jun Li and Shiwei Luo and Mengke Sun and Quan Quan and Shuxin Yang and You Hao and Pengbo Liu and Honghu Xiao and Chunpeng Zhao and Xinbao Wu and S. Kevin Zhou},
      year={2024},
      eprint={2105.14711},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2105.14711},
}
"""  # noqa: E501

_DESCRIPTION = """
Spine-related diseases have high morbidity and cause a huge burden of social cost.
Spine imaging is an essential tool for noninvasively visualizing and assessing spinal
pathology. Segmenting vertebrae in computed tomography (CT) images is the basis of
quantitative medical image analysis for clinical diagnosis and surgery planning of
spine diseases. Current publicly available annotated datasets on spinal vertebrae are
small in size. Due to the lack of a large-scale annotated spine image dataset, the
mainstream deep learning-based segmentation methods, which are data-driven, are heavily
restricted. In this paper, we introduce a large-scale spine CT dataset, called CTSpine1K
curated from multiple sources for vertebra segmentation, which contains 1,005 CT volumes
with over 11,100 labeled vertebrae belonging to different spinal conditions. Based on
this dataset, we conduct several spinal vertebrae segmentation experiments to set the
first benchmark. We believe that this large-scale dataset will facilitate further
research in many spine-related image analysis tasks, including but not limited to
vertebrae segmentation, labeling, 3D spine reconstruction from biplanar radiographs,
image super-resolution, and enhancement.
"""

_HOMEPAGE = "https://github.com/MIRACLE-Center/CTSpine1K"

_LICENSE = "CC-BY-NC-SA"


HF_API = HfApi()

_URLS = HF_API.list_repo_files(
    "alexanderdann/CTSpine1K",
    repo_type="dataset",
)


class CTSpine1KBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for the dataset CTSpine1K."""

    def __init__(
        self,
        *,
        volumetric: bool,
        **kwargs: dict,
    ) -> None:
        """C'tor if the CTSpine1KBuilderConfig.

        Args:
            volumetric: whether we want to use 3D or 2D data.
            kwargs: parameters which can be used to overwrite the BuilderConfig
                and are forwarded to the super class call.

        """
        super().__init__(**kwargs)

        self.citation = _CITATION
        self.homepage = _HOMEPAGE
        self.license = _LICENSE

        self.volumetric = volumetric


class CTSpine1K(datasets.GeneratorBasedBuilder):
    """Dataloading generator for the CTSpine1K dataset."""

    BUILDER_CONFIGS: ClassVar[list[CTSpine1KBuilderConfig]] = [
        CTSpine1KBuilderConfig(
            name="3d",
            volumetric=True,
            description="3D volumes of CT spine scans",
            version=datasets.Version("1.0.0"),
        ),
        CTSpine1KBuilderConfig(
            name="2d",
            volumetric=False,
            description="2D axial slices of CT spine scans",
            version=datasets.Version("1.0.0"),
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.volumetric:
            features = datasets.Features(
                {
                    "image": datasets.Array3D(
                        shape=(None, 512, 512),
                        dtype="float32",
                    ),
                    "segmentation": datasets.Array3D(
                        shape=(None, 512, 512),
                        dtype="int32",
                    ),
                    "patient_id": datasets.Value("string"),
                    "index": datasets.Value("int32"),
                },
            )
        else:
            features = datasets.Features(
                {
                    "image": datasets.Array2D(shape=(512, 512), dtype="float32"),
                    "segmentation": datasets.Array2D(shape=(512, 512), dtype="int32"),
                    "patient_id": datasets.Value("string"),
                    "index": datasets.Value("int32"),
                    "slice_index": datasets.Value("int32"),
                },
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            # Add citation and other metadata here
        )

    def _split_generators(
        self,
        dl_manager: DownloadManager,
    ) -> list[datasets.GeneratorBasedBuilder]:
        split_file_idx = _URLS.index("metadata/data_split.txt")
        downloaded_files = dl_manager.download(_URLS)

        training, validation, test = self._load_split(downloaded_files[split_file_idx])

        lookup = self._validate_check(downloaded_files)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"pairs": [lookup[stem] for stem in training]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"pairs": [lookup[stem] for stem in validation]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"pairs": [lookup[stem] for stem in test]},
            ),
        ]

    @staticmethod
    def _load_split(file: str) -> tuple[list[str]]:
        """Load the training, validation and test split.

        Currently this component assumes that the names always come in pairs
        with identical names in the data and labels. The only difference is the
        additional '_labels' after the name off the file.

        Returns:
            Tuples containg the lists for each split.

        """
        split_file = Path(file)
        split_data = split_file.read_text()
        split_list = split_data.split("\n")

        train_ident = split_list.index("trainset:")
        test_public_ident = split_list.index("test_public:")
        test_private_ident = split_list.index("test_private:")

        training = split_list[
            train_ident + 1 : test_public_ident - 1
        ]  # removing white space
        validation = split_list[
            test_public_ident + 1 : test_private_ident - 1
        ]  # removing white space
        test = split_list[test_private_ident + 1 :]

        return training, validation, test

    def _validate_check(self, files: list[str]) -> dict:
        ffiles = sorted(file for file in files if "nii.gz" in file)

        data_candidates = []
        label_candidates = []

        for file in ffiles:
            if "volumes" in file:
                data_candidates.append(file)

                # construction check to ensure data exists
                label = file.split(".nii.gz")
                label = label[0] + "_seg.nii.gz"
                label = label.replace("volumes", "labels")
                label_candidates.append(label)

        if len(data_candidates) != len(label_candidates):
            msg = (
                "Data and labels mismatch. Data is probably not downloaded fully. ",
                f"Got {len(data_candidates)} vs {len(label_candidates)}",
            )
            raise RuntimeError(msg)

        pairs: list[tuple[str, str]] = zip(
            data_candidates,
            label_candidates,
            strict=True,
        )
        lookup = {}
        for input_, label in pairs:
            file_path, segmentation_path = Path(input_), Path(label)

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
                    f"Detected files with different slice count for {file_path} "
                    f"and {segmentation_path}. Image slices ({image_slices}) are not "
                    f"equal to segmentation slices ({segmentation_slices})."
                )
                raise ValueError(msg)

            # convert 'some/path/to/file/dummy_123.nii.gz' to dummy_123.nii.gz
            stem = file_path.name
            if stem.removesuffix(".nii.gz") not in label:
                msg = f"Mismatch in sorted pairs. {input_} vs. {label}"
                raise ValueError(msg)

            lookup[stem] = (file_path, segmentation_path)

        return lookup

    @staticmethod
    def _get_sample_length(file_path: Path) -> int:
        expected_ndim = 3
        img = nib.load(file_path)  # this only loads the header
        shape = img.shape
        assert len(shape) == expected_ndim

        return shape[2]

    @staticmethod
    def _volumetric_sample(path: Path) -> np.ndarray:
        volume = nib.load(path)
        volume = volume.get_fdata()
        return np.transpose(volume, (2, 0, 1))

    def _generate_examples(self, pairs: list[tuple[Path, Path]]) -> Generator:
        for pair_idx, (volume_path, label_path) in enumerate(pairs):
            patient_id = Path(volume_path.stem).stem
            image = self._volumetric_sample(volume_path)
            segmentation = self._volumetric_sample(label_path).astype(np.uint32)

            if self.config.volumetric:
                yield (
                    patient_id,
                    {
                        "image": image,
                        "segmentation": segmentation,
                        "patient_id": patient_id,
                        "index": pair_idx,
                    },
                )
            else:
                for idx in range(image.shape[0]):  # iterate over axial slices
                    yield (
                        patient_id + f"_{idx}",
                        {
                            "image": image[idx],
                            "segmentation": segmentation[idx],
                            "patient_id": patient_id,
                            "index": pair_idx,
                            "slice_index": idx,
                        },
                    )
