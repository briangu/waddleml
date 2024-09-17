
![Unit Tests](https://github.com/briangu/waddleml/workflows/Unit%20Tests/badge.svg)
[![Last Commit](https://img.shields.io/github/last-commit/briangu/waddleml)](https://img.shields.io/github/last-commit/briangu/waddleml)
[![Dependency Status](https://img.shields.io/librariesio/github/briangu/waddleml)](https://libraries.io/github/briangu/waddleml)
[![Open Issues](https://img.shields.io/github/issues-raw/briangu/waddleml)](https://github.com/briangu/waddleml/issues)
[![Repo Size](https://img.shields.io/github/repo-size/briangu/waddleml)](https://img.shields.io/github/repo-size/briangu/waddleml)
[![GitHub star chart](https://img.shields.io/github/stars/briangu/waddleml?style=social)](https://star-history.com/#briangu/waddleml)

[![Release Notes](https://img.shields.io/github/release/briangu/waddleml)](https://github.com/briangu/waddleml/releases)
[![Downloads](https://static.pepy.tech/badge/waddleml/month)](https://pepy.tech/project/waddleml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# WaddleML

A machine learning stats tracker built on DuckDB.

# Quickstart

1. Install WaddleML.

`pip3 install waddleml`

2. Try the sample code

```python
import waddle
import random  # Just for simulating metrics, replace with real model code.

# Initialize a new run with waddle.init, specifying the project name
run = waddle.init(project="hello-world")

# Save model inputs, hyperparameters, and metadata in run.config
config = run.config
config.learning_rate = 0.01
config.batch_size = 32
config.model_type = "simple_CNN"  # Example configuration

# Simulate model training and log metrics
for epoch in range(10):
    # Simulate a training loss (replace with actual model code)
    train_loss = random.uniform(0.8, 0.4)  # Example of loss decreasing

    # Log the loss metric to Waddle
    run.log({"epoch": epoch, "loss": train_loss})

    # Optionally, log other metrics like accuracy, learning rate, etc.
    if epoch % 2 == 0:
        accuracy = random.uniform(0.6, 0.9)  # Simulate accuracy
        run.log({"epoch": epoch, "accuracy": accuracy})

# Once training is done, mark the run as finished
run.finish()
```

3. Visualize the results.

