"""Installation script with dependencies."""

from setuptools import find_packages, setup

setup(
    name="nnx",
    version="0.1",
    python_requires=">=3.12",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    install_requires=[
        "numpy==1.26.4",
        "ruff==0.9.7",
        "tqdm==4.67.1",
        "matplotlib==3.10.1",
        "plotly==6.0.1",
        "nibabel==5.3.2",
        "opencv-python==4.11.0.86",
        "pydantic==2.10.6",
        "scipy==1.15.0",
        "nbformat==5.10.4",  # Needed for rendering of plotly
        "wandb==0.19.8",
        "gin_config==0.5.0",
        "termcolor==2.5.0",
    ],
    extras_require={
        "torch": ["torch==2.6.0", "torchvision==0.21.0"],
    },
)
