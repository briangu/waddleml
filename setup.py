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
    version="0.1.0",
    description="Lightweight run logger with git snapshots and a local dashboard.",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author="Waddle contributors",
    python_requires=">=3.8",
    packages=find_packages(include=["waddle", "waddle.*"]),
    include_package_data=True,
    package_data={"waddle": ["static/*"]},
    entry_points={"console_scripts": ["waddle=waddle.waddle_cli:main"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Build Tools",
    ],
)
