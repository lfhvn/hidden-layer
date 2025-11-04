"""
Setup script for Hidden Layer research lab.

This makes the harness importable from projects.
"""

from setuptools import find_packages, setup

setup(
    name="hidden-layer",
    version="0.2.0",
    description="Hidden Layer Research Lab - AI Research Infrastructure",
    author="Hidden Layer Lab",
    packages=find_packages(include=["harness", "harness.*", "shared", "shared.*"]),
    python_requires=">=3.10",
    install_requires=[
        "anthropic>=0.18.0",
        "openai>=1.0.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "mlx": [
            "mlx>=0.4.0",
            "mlx-lm>=0.4.0",
        ],
        "introspection": [
            "numpy>=1.24.0",
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
