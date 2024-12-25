from setuptools import setup, find_packages

setup(
    name="MLOps",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub",
    ],
    description="Utilities for MLOps workflows",
    author="YifeiDevs",
    url="https://github.com/YifeiDevs/MLOps",
)
