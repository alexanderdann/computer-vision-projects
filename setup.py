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
        "torch==2.2.0",
        "torchvision==0.17.0",
    ],
)
