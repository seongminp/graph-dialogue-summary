#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dsum",
    version="1.0.0",
    description="Dialogue summarization",
    author="ActionPower",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["networkx", "nltk", "datasets", "absl-py", "rouge-score"],
)
