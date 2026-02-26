from setuptools import setup, find_packages

setup(
    name="skin-cancer-detection",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "segmentation-models-pytorch>=0.3",
        "albumentations>=1.3",
        "numpy",
        "Pillow",
        "PyYAML",
        "tqdm",
        "matplotlib",
        "scikit-learn",
        "wandb",
    ],
)
