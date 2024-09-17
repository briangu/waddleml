import duckdb
import psutil
import platform
import argparse
import json
import sys
import os
import subprocess
import threading
import time
from datetime import datetime
from pynvml import *
from typing import Any, Dict, Optional


class WaddleLogger:
    def __init__(self, db_path, id=None, config=None, use_gpu_metrics=True):
        self.db_path = db_path
        self.id = id or datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(db_path)
        self.config: argparse.Namespace = config
        self.use_gpu_metrics = use_gpu_metrics

        # Create tables for different logging categories
        self.con.execute('''
            CREATE TABLE IF NOT EXISTS run_info (
                id VARCHAR,
                start_time TIMESTAMP,
                cli_params JSON,
                python_version VARCHAR,
                os_info VARCHAR,
                cpu_info VARCHAR,
                total_memory DOUBLE,
                gpu_info JSON,
                code BLOB,
                git_hash VARCHAR,
                timestamp TIMESTAMP,
                PRIMARY KEY (id)
            );
        ''')

        # Modified logs table without value_type field
        self.con.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id VARCHAR,
                step INTEGER,
                category VARCHAR,
                name VARCHAR,
                value_double DOUBLE,
                value_string VARCHAR,
                value_blob BLOB,
                timestamp TIMESTAMP
            );
        ''')

        # Log initial system, CLI parameters, and code
        self.log_run_info()

    def log_run_info(self):
        # Get system information
        python_version = sys.version
        os_info = platform.platform()
        cpu_info = platform.processor()
        total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB

        # Retrieve Git commit hash if available
        git_hash = self.get_git_commit_hash()

        # Read the running script (if accessible)
        code_data = self.get_running_script_code()

        # Prepare the CLI parameters as JSON
        cli_params_json = json.dumps(sys.argv)

        # Insert into the database
        self.con.execute('''
            INSERT INTO run_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.id, datetime.now(), cli_params_json, python_version,
            os_info, cpu_info, total_memory, json.dumps(self.get_gpu_system_metrics()),
            code_data, git_hash, datetime.now()
        ))

    def get_git_commit_hash(self):
        """
        Retrieves the current Git commit hash if the script is in a Git repository.
        """
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
        except Exception:
            git_hash = None
        return git_hash

    def get_running_script_code(self):
        """
        Reads and returns the content of the running script.
        """
        try:
            with open(__file__, 'r') as f:
                return f.read().encode('utf-8')  # Encode to binary for storage
        except Exception as e:
            print(f"Could not read the script: {e}")
            return b''  # Return an empty byte if reading fails

    def get_gpu_system_metrics(self):
        """
        Collects system information about GPU utilization, temperature, etc.
        """
        gpu_metrics = []
        if self.use_gpu_metrics:
            gpu_count = nvmlDeviceGetCount()
            for i in range(gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                gpu_metrics.append({
                    'gpu_index': i,
                    'name': nvmlDeviceGetName(handle).decode('utf-8'),
                    'temperature': nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
                    'memory_used': nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3),  # GB
                    'memory_total': nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3),  # GB
                    'utilization': nvmlDeviceGetUtilizationRates(handle).gpu
                })
        return gpu_metrics

    def log(self, data: Dict[str, Any], step: Optional[int], category='default', timestamp=None):
        """
        Logs data into DuckDB.
        :param data: A dictionary of key-value pairs to log.
        :param step: The training step or time step associated with the log.
        :param category: The category of the data.
        :param timestamp: Optional timestamp; if not provided, current time is used.
        """
        timestamp = timestamp or datetime.now()
        for name, value in data.items():
            value_double = None
            value_string = None
            value_blob = None

            if isinstance(value, (int, float)):
                value_double = value
            elif isinstance(value, str):
                value_string = value
            elif isinstance(value, (bytes, bytearray)):
                value_blob = value
            else:
                # For other types, store as JSON string
                value_string = json.dumps(value)

            # Insert into the logs table
            self.con.execute('''
                INSERT INTO logs (
                    id, step, category, name, value_double, value_string, value_blob, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.id, step or 0, category, name,
                value_double, value_string, value_blob, timestamp
            ))

    def log_gpu_metrics_periodically(self, interval=60):
        """
        Periodically logs GPU system metrics every `interval` seconds in a separate thread.
        """
        def log_metrics():
            while True:
                gpu_metrics = self.get_gpu_system_metrics()
                for metric in gpu_metrics:
                    data = {}
                    for key, value in metric.items():
                        if key != 'gpu_index' and key != 'name':  # Skip index and name in logs
                            data[f'gpu_{metric["gpu_index"]}_{key}'] = value
                    self.log(data=data, step=None, category='gpu_system', timestamp=datetime.now())
                time.sleep(interval)

        # Create and start a thread for logging GPU metrics
        logging_thread = threading.Thread(target=log_metrics, daemon=True)
        logging_thread.start()

config: argparse.Namespace = argparse.Namespace()
run: WaddleLogger = None

def _assign_config(app_config):
    global config
    config = app_config

def init(project, db_root='.waddle', config=None, use_gpu_metrics=True, gpu_metrics_interval=60):
    global run

    # assign the passed config to the global config
    config = config or argparse.Namespace()
    _assign_config(config)

    db_path = os.path.join(db_root, f"{project}.db")
    if use_gpu_metrics:
        # Check if the pynvml package is available
        try:
            nvmlInit()
        except Exception as e:
            print(f"Could not initialize NVML: {e}")
            use_gpu_metrics = False
    run = WaddleLogger(db_path, use_gpu_metrics=use_gpu_metrics, config=config)
    if use_gpu_metrics and gpu_metrics_interval > 0:
        run.log_gpu_metrics_periodically(interval=gpu_metrics_interval)
    return run

def finish():
    pass

def log(category, data, step, timestamp=None):
    if run is None:
        raise ValueError("WaddleLogger is not initialized. Please call `init()` first.")
    if not isinstance(data, dict):
        raise ValueError("The data must be a dictionary.")
    run.log(data=data, step=step, category=category, timestamp=timestamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, default='experiment')
    args = parser.parse_args()

    # Initialize WaddleLogger and start periodic GPU logging in a thread
    init(project="test", config=args)

    # Simulate the rest of your ML code here
    # The GPU logging will run in the background
    for step in range(10):
        log_data = {'loss': 0.01 * step}
        log(category='model', data=log_data, step=step)
        time.sleep(5)  # Simulating training steps

    # Wait for the logging thread to finish
    time.sleep(10)
    print("Done!")

    # dump the logs to a CSV file
    run.con.execute('COPY logs TO \'logs.csv\' (FORMAT CSV, HEADER);')
