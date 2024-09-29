
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

WaddleML is an open-source variant of Weights and Biases (WandB) designed to provide flexible, offline-capable experiment tracking for machine learning projects. It employs a logging strategy, where training jobs write recording events to disk as log files. A local instance of Waddle may either operate as a relay forwarding traffic to a centarl Waddle server or creates local web server that allows users to track their experiments without requiring any online services.

# Key Components

1. WaddleLogger (Client-Side)
    * Logging Events to Disk:
        * The WaddleLogger class is integrated into training scripts to log metrics, parameters, and system information.
        * It writes log entries as JSON files to a designated log folder (log_root).
        * Captures system information such as CPU, memory, and optionally GPU metrics using NVML if available.
        * Can log GPU metrics periodically in a separate thread.
    * Run Information:
        * Captures experiment context for reproducability.
        * Collects and logs run-specific information, including start time, CLI parameters, Python version, OS info, CPU info, total memory, GPU info, code snapshot, and Git commit hash.

2. WaddleServer (Server-Side)
    * Database Initialization:
        * Initializes a DuckDB database to store project information, run information, and log entries.
        * The database schema includes tables for project_info, run_info, and logs.
        * Folder Watching and Log Ingestion:
    * Watches the specified log folder for new log files using a background thread.
        * Processes new log files by ingesting log entries into the DuckDB database.
        * Deletes log files after processing to avoid reprocessing.
    * Web Server and API:
        * Uses FastAPI to provide a web interface and API endpoints.
        * Endpoints include:
            * POST /ingest: Ingests a log entry directly via HTTP.
            * GET /info: Retrieves project information.
            * GET /runs: Retrieves a list of runs with their start times.
            * GET /run/{run_id}: Retrieves log entries for a specific run.
            * WebSocket endpoint /ws for real-time updates to connected clients.
    * Web Interface:
        * Serves an HTML template (index.html) that provides a user interface for viewing experiments.

3. Operation Modes
    * Local Proxy Mode (Offline Mode):
        * The WaddleServer runs locally, watching log files written by WaddleLogger.
        * Users can track experiments without an internet connection.
        * Optionally, users can sync data to a central server by running a sync operation.
    * Relay to Central Server:
        * Local instances can forward log entries to a centralized WaddleML server.
        * Facilitates centralized tracking of experiments from multiple machines or users.
        * The central server aggregates data from multiple sources.

# Workflow

1. Logging Experiments
    * In the training script, the user initializes WaddleLogger, specifying the project name and other configurations.
    * During training, metrics and other relevant data are logged using the log method of WaddleLogger.
    * Log entries are saved as JSON files in the local log folder.
2. Data Ingestion
    * The WaddleServer watches the log folder for new log files.
    * Upon detecting new files, it ingests the log entries into the DuckDB database.
    * After ingestion, log files are deleted to prevent reprocessing.
3. Data Access and Visualization
    * Users access the local web server provided by WaddleServer to view experiment data.
    * The web interface allows users to browse runs, view metrics, and analyze results.
    * Real-time updates are provided via WebSockets for an interactive experience.

# Advantages of WaddleML

* Offline Capability:
    * Users can track experiments without relying on external services or internet connectivity.
    * Ideal for secure environments or situations where data privacy is critical.
* Flexibility in Deployment:
    * Can operate entirely locally or integrate with a central server for collaborative experiment tracking.
    * Supports syncing of local data to a central server when desired.
* Simplicity and Lightweight Design:
    * Uses DuckDB, a lightweight SQL OLAP database, for efficient data storage and querying.
    * Relies on standard formats (JSON) for log files, making it easy to inspect and debug.

# Conclusion

    WaddleML offers a practical and flexible solution for experiment tracking in machine learning projects, particularly suited for environments where offline operation or data privacy is important. By leveraging a "ticker plant" strategy and standard technologies like DuckDB and FastAPI, it provides a lightweight yet powerful platform for monitoring and analyzing experiments.

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

