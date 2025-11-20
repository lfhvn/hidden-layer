"""
Setup for ACE (Agentic Context Engineering) package.
"""

from setuptools import setup, find_packages

setup(
    name="ace",
    version="0.1.0",
    description="Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models",
    author="Hidden Layer Lab",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0",
    ],
)
