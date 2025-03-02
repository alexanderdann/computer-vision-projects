"""Installation script with dependencies."""

from setuptools import find_packages, setup

setup(
    name="nnx",
    version="0.1",
    python_requires=">=3.12",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    install_requires=["numpy==2.2.0", "ruff==0.9.7", "tqdm==2.2.3"],
)
