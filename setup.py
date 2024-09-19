import os
from setuptools import setup, find_packages

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

# use the requirements.txt to install dependencies
with open(os.path.join(directory, 'requirements.txt')) as f:
    required_packages = f.read().splitlines()

setup(
    name='waddleml',
    packages=find_packages(),
    version='0.1.2',
    description='WaddleML is a machine learning stats tracker built on DuckDB',
    author='Brian Guarraci',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=required_packages,
    python_requires='<=3.12,>=3.8',
    include_package_data=True,
    zip_safe=False,
    test_suite='tests',
    scripts=['scripts/waddle']
)
