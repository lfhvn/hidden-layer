"""
Setup script for Hidden Layer research lab.

This makes the harness and research packages importable.

Directory Naming Convention:
- Dash-named dirs (e.g., multi-agent/) contain project files (README, config, notebooks)
- Underscore-named dirs (e.g., multi_agent/) are Python packages for imports
- Use underscore names for imports: `from communication.multi_agent import ...`
"""

from setuptools import find_packages, setup

setup(
    name="hidden-layer",
    version="0.3.0",
    description="Hidden Layer Research Lab - AI Research Infrastructure",
    author="Hidden Layer Lab",
    packages=find_packages(
        include=[
            # Core infrastructure
            "harness",
            "harness.*",
            "shared",
            "shared.*",
            "mlx_lab",
            "mlx_lab.*",
            "ai_research_aggregator",
            "ai_research_aggregator.*",
            # Research areas - Python package names (underscore)
            "communication",
            "communication.*",
            "theory_of_mind",
            "theory_of_mind.*",
            "alignment",
            "alignment.*",
            "representations",
            "representations.*",
            "memory",
            "memory.*",
            # Platform projects
            "agentmesh",
            "agentmesh.*",
        ]
    ),
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
        "web": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlx-lab=mlx_lab.cli:main",
            "ai-digest=ai_research_aggregator.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
