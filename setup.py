from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README_PATH = ROOT / "README.md"

if README_PATH.exists():
    long_description = README_PATH.read_text(encoding="utf-8")
    long_description_content_type = "text/markdown"
else:
    long_description = "Lightweight experiment tracker with git-native snapshots and a local dashboard."
    long_description_content_type = "text/plain"

setup(
    name="waddle",
    version="1.0.0",
    description="WaddleML: local Weights & Biases — git-native ML experiment tracker with DuckDB.",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author="Waddle contributors",
    python_requires=">=3.9",
    packages=find_packages(include=["waddle", "waddle.*"]),
    include_package_data=True,
    package_data={"waddle": ["static/*"]},
    install_requires=[
        "duckdb>=0.9.0",
        "starlette>=0.27.0",
        "uvicorn>=0.23.0",
    ],
    extras_require={
        "system": ["psutil>=5.9.0"],
        "gpu": ["pynvml>=11.5.0"],
        "all": ["psutil>=5.9.0", "pynvml>=11.5.0"],
    },
    entry_points={"console_scripts": ["waddle=waddle.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
