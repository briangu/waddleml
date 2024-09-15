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
import base64
import argparse

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

        self.con.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id VARCHAR,
                step INTEGER,
                category VARCHAR,
                name VARCHAR,
                value DOUBLE,
                timestamp TIMESTAMP
            );
        ''')

        self.con.execute('''
            CREATE TABLE IF NOT EXISTS blobs (
                id VARCHAR,
                step INTEGER,
                category VARCHAR,
                file_name VARCHAR,
                file_data BLOB,
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
            base64.b64encode(code_data).decode('utf-8'), git_hash, datetime.now()
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
                    'name': nvmlDeviceGetName(handle),
                    'temperature': nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
                    'memory_used': nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3),  # GB
                    'memory_total': nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3),  # GB
                    'utilization': nvmlDeviceGetUtilizationRates(handle).gpu
                })
        return gpu_metrics

    def log(self, category, name, data, step, is_blob=False, timestamp=None):
        """
        Logs data into DuckDB.
        :param category: The category of the log (e.g., model, system, etc.).
        :param name: The name of the data being logged (e.g., loss, accuracy, etc.).
        :param data: The actual data to log.
        :param step: The training step or time step associated with the log.
        :param is_blob: Whether the data is a blob (binary file) or regular log data.
        """
        timestamp = timestamp or datetime.now()
        if is_blob:
            # Store the binary blob data
            file_name = os.path.basename(name)
            file_data_base64 = base64.b64encode(data).decode('utf-8')  # Store as base64 encoded data

            self.con.execute(f'''
                INSERT INTO blobs VALUES (
                    '{self.id}', {step}, '{category}', '{file_name}', '{file_data_base64}', '{timestamp}'
                );
            ''')
        else:
            # Store regular log data (numbers, strings, etc.)
            print(f"Logging: {category}, {name}, {data}, {step or 0}, {timestamp}")
            self.con.execute(f'''
                INSERT INTO logs VALUES (
                    '{self.id}', {step or 0}, '{category}', '{name}', {data}, '{timestamp}'
                );
            ''')

    def log_gpu_metrics_periodically(self, interval=60):
        """
        Periodically logs GPU system metrics every `interval` seconds in a separate thread.
        """
        def log_metrics():
            while True:
                gpu_metrics = self.get_gpu_system_metrics()
                for metric in gpu_metrics:
                    for key, value in metric.items():
                        if key != 'gpu_index' and key != 'name':  # Skip index and name in logs
                            self.log(category='gpu_system', name=f'gpu_{metric["gpu_index"]}_{key}', data=value, step=None, timestamp=datetime.now())
                time.sleep(interval)

        # Create and start a thread for logging GPU metrics
        logging_thread = threading.Thread(target=log_metrics, daemon=True)
        logging_thread.start()

config: argparse.Namespace = argparse.Namespace()
run: WaddleLogger = None

def _assign_config(app_config):
    global config
    config = app_config

# Initialize the logger (like wandb.init)
def init(project, db_root='waddle', config=None, use_gpu_metrics=True, gpu_metrics_interval=60):
    global run

    # assign the passed config to the global config
    _assign_config(config)

    db_path = os.path.join(db_root, f"{project}.db")
    if use_gpu_metrics:
        # Check if the pynvml package is available
        try:
            nvmlInit()
        except Exception as e:
            print(f"Could not initialize NVML: {e}")
            use_gpu_metrics = False
    run = WaddleLogger(db_path, use_gpu_metrics=use_gpu_metrics)
    if use_gpu_metrics and gpu_metrics_interval > 0:
        run.log_gpu_metrics_periodically(interval=gpu_metrics_interval)
    return run

def log(category, data, step, timestamp=None):
    if run is None:
        raise ValueError("WaddleLogger is not initialized. Please call `init()` first.")
    if not isinstance(data, dict):
        raise ValueError("The data must be a dictionary.")
    for key, value in data.items():
        run.log(category, key, value, step, False, timestamp)

def log_blob(category, name, data, step, timestamp=None):
    if run is None:
        raise ValueError("WaddleLogger is not initialized. Please call `init()` first.")
    run.log(category, name, data, step, True, timestamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, default='experiment')
    args = parser.parse_args()

    # Initialize WaddleLogger and start periodic GPU logging in a thread
    init(project="test", config=args)

    # Simulate the rest of your ML code here
    # The GPU logging will run in the background
    for step in range(10):
        run.log(category='model', name='loss', data=0.01 * step, step=step)
        time.sleep(5)  # Simulating training steps

    # Wait for the logging thread to finish
    time.sleep(10)
    print("Done!")

    # dump the logs to a CSV file
    run.con.execute('COPY logs TO \'logs.csv\' (FORMAT CSV, HEADER);')
