# CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation in Computed Tomography

![License](https://img.shields.io/badge/License-CC--BY--NC--SA-blue.svg)
![Dataset Size](https://img.shields.io/badge/Dataset%20Size-150GB-green.svg)
![CT Volumes](https://img.shields.io/badge/CT%20Volumes-1005-orange.svg)
![Vertebrae](https://img.shields.io/badge/Labeled%20Vertebrae-11100+-red.svg)

## Overview

CTSpine1K is a large publicly available annotated spine CT dataset for vertebra segmentation research. This comprehensive dataset addresses the critical need for large-scale annotated spine image data in deep learning-based medical image analysis.

**Key Features:**
- **1,005 CT volumes** with expert annotations
- **Over 11,100 labeled vertebrae** across different spinal conditions
- **Multiple pathological cases** including sacral lumbarization and lumbar sacralization
- **Diverse data sources** ensuring robust model generalization
- **Standardized NIfTI format** for easy integration with medical imaging pipelines
- **Flexible loading options** supporting both 2D slice-based and 3D volumetric analysis

## Dataset Statistics

| Split | CT Volumes | Description |
|-------|------------|-------------|
| **Train** | 610 | Training set for model development |
| **Validation** | 197 | Validation set for hyperparameter tuning |
| **Test** | 198 | Test set for final evaluation |
| **Total** | **1,005** | Complete dataset |

**Technical Specifications:**
- **Image Format:** NIfTI (.nii.gz)
- **Spatial Dimensions:** 512 × 512 pixels
- **Axial Slices:** Variable per CT volume (typically 100-512 slices)
- **Raw Dataset Size:** ~150GB
- **Arrow Dataset Size:** ~1100GB
- **Annotation Format:** Dense segmentation masks with vertebrae-specific labels

## Installation

Install the required dependencies:

```bash
pip install datasets nibabel
```

## Quick Start

### Loading the Dataset

There are two ways to use the dataset: **Raw** format or **Arrow** format. Raw refers to downloading the original `.nii.gz` files using the Hugging Face Hub. The other option converts files to Apache Arrow format, enabling all advanced features Hugging Face offers with seamless PyTorch or TensorFlow integration. 

#### Choose Raw if:

You need specific files only
You have existing NIfTI processing pipelines
You want maximum control over data loading
You're doing exploratory analysis

#### Choose Arrow if:

You want seamless PyTorch/TensorFlow integration
You need fast I/O with intelligent caching
You prefer standardized dataset interfaces
You're training models repeatedly on the same data

> [!TIP] 
> When in doubt, start with the Arrow option — if you decide it's not needed, you can always access the raw files from the cache since the data downloads anyway.

### Option 1: Raw

This approach downloads the dataset files as you see them on Hugging Face. This is useful when you need only parts of the data, want to explore the structure, or already have logic to read and process `.nii.gz` files. 



This offers maximum flexibility since you can also easily filter specific files from `urls` and download these. Mind to convert these to actual urls before downloading.

```python
from huggingface_hub import HfApi

hf_api = HfApi()

urls = hf_api.list_repo_files(
    "alexanderdann/CTSpine1K",
    repo_type="dataset",
)
```

### Option 2: Arrow

This option offers all the benefits of Hugging Face's datasets library. Apache Arrow is a columnar memory format optimized for analytics workloads, providing significant I/O acceleration through efficient data layouts and zero-copy operations. Once converted to Arrow format, the dataset supports fast random access, intelligent caching, and seamless integration with ML frameworks.
The conversion process automatically handles data splitting, type optimization, and creates efficient indexes for rapid access patterns—particularly beneficial when training models that repeatedly access the same data.

> [!WARNING]
> **Memory Warning for 3D mode:** Set `writer_batch_size=1` if you have limited RAM. Use tools such as `htop` on Linux to supervise the procedure and how it behaves.

During the initial Arrow conversion, the system collects multiple samples in memory before writing them as optimized chunks to disk. This batching improves I/O performance but can consume significant memory. If experiencing memory issues, `use writer_batch_size=1` to process one sample at a time. 

> [!NOTE]
> The `trust_remote_code=True` option is needed for the script at https://huggingface.co/datasets/alexanderdann/CTSpine1K/blob/main/CTSpine1K.py to run.

```python
from datasets import load_dataset

# Load 3D volumetric data
dataset_3d = load_dataset(
    'alexanderdann/CTSpine1K',
    name="3d",
    trust_remote_code=True,
    writer_batch_size=5, # see the warning above
)

# Load 2D slice-based data
dataset_2d = load_dataset(
    'alexanderdann/CTSpine1K',
    name="2d",
    trust_remote_code=True,
)

# Access training, validation, and test splits
train_data = dataset_3d["train"]
val_data = dataset_3d["validation"]
test_data = dataset_3d["test"]
```


### Data Structure

Each sample contains:
- **`image`**: Raw CT volume/slice as numpy array (float32)
- **`segmentation`**: Corresponding segmentation mask (int32)
- **`patient_id`**: Unique identifier for the CT scan

**File Organization:**
```
rawdata/
├── volumes/          # Original CT scans
│   └── [dataset]/    # Organized by source dataset
│       └── *.nii.gz
└── labels/           # Segmentation masks
    └── [dataset]/    # Organized by source dataset  
        └── *_seg.nii.gz
```

## Data Sources

CTSpine1K is curated from four established medical imaging datasets to ensure diversity and clinical relevance:

1. **CT COLONOGRAPHY** - Abdominal CT scans from colonography screening studies
2. **HNSCC-3DCT-RT** - High-resolution CT scans from head-and-neck cancer patients
3. **Medical Segmentation Decathlon (Task 3)** - Liver CT scans with visible spine regions  
4. **COVID-19 CT Dataset** - Chest CT scans from COVID-19 studies

This multi-source approach provides:
- **Anatomical diversity** across different body regions
- **Pathological variation** including normal and diseased spines
- **Technical diversity** from different scanners and protocols
- **Demographic representation** across multiple patient populations

For detailed information about each source dataset and acquisition protocols, please refer to our [research paper](https://arxiv.org/abs/2105.14711).

## Vertebrae Labeling

The dataset includes comprehensive vertebrae annotations covering the full spinal column. For specific details about the labeling schema, vertebrae classes, and handling of pathological cases (L6 vertebrae in sacral lumbarization and lumbar sacralization), please consult the original paper.

**Special Cases:**
- **Sacral Lumbarization:** 24 cases with detailed annotations
```
liver_106_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0004_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0067_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0149_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0167_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0175_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0189_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0215_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0261_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0267_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0344_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0401_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0587_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0666_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0672_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0699_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0737_seg.nii.gz
verse506_CT-iso_seg.nii.gz
verse519_CT-iso_seg.nii.gz
verse532_seg.nii.gz
verse539_CT-iso_seg.nii.gz
verse542_CT-iso_seg.nii.gz
verse565_CT-iso_seg.nii.gz
verse586_CT-iso_seg.nii.gz
verse619_CT-iso_seg.nii.gz
```
- **Lumbar Sacralization:** 13 cases with expert labeling
```
liver_83_seg.nii.gz
liver_93_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0064_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0104_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0107_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0110_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0537_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0554_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0555_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0615_seg.nii.gz
1.3.6.1.4.1.9328.50.4.0721_seg.nii.gz
verse100_seg.nii.gz
verse584_seg.nii.gz
verse594_seg.nii.gz
```
- **Pathological Variants:** All visible vertebrae annotated regardless of anatomical variations

## Usage Modes

### 3D Volumetric Mode (`name="3d"`)
- Returns complete CT volumes and segmentation masks
- Ideal for 3D architectures and volumetric analysis
- Memory-intensive but preserves spatial context

### 2D Slice Mode (`name="2d"`)  
- Returns individual axial slices
- Memory-efficient for large-scale experiments
- Compatible with 2D architectures
- Automatic slice indexing across the dataset

## Known Issues & Limitations

Arrow format requires ~1TB compared to ~150GB for raw files (6-7x increase)
Initial conversion to Arrow format can take mutltiple hours depending on hardware.

### Workarounds:

- Use `keep_in_memory=False` when loading to avoid RAM limitations
- Consider using only specific splits: `load_dataset(..., split="train")`
- For limited storage, use the raw format with custom PyTorch/TensorFlow data loaders

## Future Enhancements

- **Metadata Integration:** Addition of clinical metadata and imaging parameters from accompanying spreadsheet files 
- **Reduce storage requirements for Arrow approach:** As stated in the limitations one could further optimise storage, for instance by using smaller data formats (float32 instead of float64 for CTs)

## Citation

If you use the CTSpine1K dataset in your research, please cite our paper:

```bibtex
@misc{deng2024ctspine1klargescaledatasetspinal,
      title={CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation in Computed Tomography},
      author={Yang Deng and Ce Wang and Yuan Hui and Qian Li and Jun Li and Shiwei Luo and Mengke Sun and Quan Quan and Shuxin Yang and You Hao and Pengbo Liu and Honghu Xiao and Chunpeng Zhao and Xinbao Wu and S. Kevin Zhou},
      year={2024},
      eprint={2105.14711},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2105.14711},
}
```

## License

This dataset is released under the **Creative Commons Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA)** license. 

- ✅ **Academic and research use** is permitted
- ✅ **Derivative works** are allowed with proper attribution  
- ❌ **Commercial use** is prohibited
- 📋 **Share-alike** requirement for derivative works

## Links & Resources

- 📄 **Paper:** [CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation](https://arxiv.org/abs/2105.14711)
- 💾 **Dataset:** [Hugging Face Hub](https://huggingface.co/datasets/alexanderdann/CTSpine1K)
- 🔧 **Original Repository:** [GitHub - MIRACLE-Center/CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K)
- 📊 **Paper Collection:** [Papers with Code - CTSpine1K](https://paperswithcode.com/dataset/ctspine1k)

## Acknowledgments

We thank all the original data providers and authors for the contribution of this dataset.

---

**For questions, issues, or contributions, please refer to the original repository or contact the dataset maintainers.**