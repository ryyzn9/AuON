#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for AuON optimizer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="auon",
    version="0.1.0",
    author="AuON Team",
    description="Alternative Unit-norm momentum-updates by Normalized nonlinear scaling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://ryyzn9.github.io/A-Survey-For-Linear-time-Orthogonal-Optimizer/",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "huggingface_hub>=0.21.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "ruff>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "auon-train=scripts.train:main",
            "auon-compare=scripts.compare_optimizers:main",
        ],
    },
)
