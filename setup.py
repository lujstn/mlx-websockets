#!/usr/bin/env python3
"""Setup script for MLX WebSockets - provides compatibility for various installers."""

from setuptools import find_packages, setup

# Read the contents of README.md
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies only (dev dependencies are in pyproject.toml)
requirements = [
    "mlx>=0.15.0",
    "mlx-lm>=0.15.0",
    "mlx-vlm>=0.0.6",
    "websockets>=12.0",
    "Pillow>=10.3.0",
    "numpy>=1.24.0",
    "rich>=13.0.0",
    "psutil>=5.9.0",
]

setup(
    name="mlx-websockets",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    author="Lucas Johnston Kurilov",
    author_email="code@lucasco.de",
    description="WebSocket streaming server for MLX models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lujstn/mlx-websockets",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mlx=mlx_websockets.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
