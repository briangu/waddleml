# waddle_logger.py

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
import uuid

class WaddleLogger:
    def __init__(self, log_root, project, name=None, config=None, use_gpu_metrics=True):
        self.log_root = log_root
        self.project = project
        self.name = name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.id = f"{self.project}_{self.name}"
        self.log_path = os.path.join(log_root, "logs")
        os.makedirs(self.log_path, exist_ok=True)
        self.config: argparse.Namespace = config
        self.use_gpu_metrics = use_gpu_metrics

        # Log initial system, CLI parameters, and code
        self.log_run_info()

    def _get_file_prefix(self):
        return f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"

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

        # Prepare the GPU info
        gpu_info = json.dumps(self.get_gpu_system_metrics())

        # Prepare the run info dictionary
        run_info = {
            "run_info": {
                'project': self.project,
                'name': self.name,
                'start_time': datetime.now().isoformat(),
                'cli_params': cli_params_json,
                'python_version': python_version,
                'os_info': os_info,
                'cpu_info': cpu_info,
                'total_memory': total_memory,
                'gpu_info': gpu_info,
                'code': code_data.decode('utf-8'),
                'git_hash': git_hash,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Write run_info to a file in the log folder
        run_info_file = os.path.join(self.log_path, f'{self._get_file_prefix()}.run_info.json')
        with open(run_info_file, 'w') as f:
            json.dump(run_info, f)

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
            try:
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
            except Exception as e:
                print(f"Error getting GPU metrics: {e}")
        return gpu_metrics

    def log(self, data: Dict[str, Any], step: Optional[int], category='default', timestamp=None):
        """
        Writes log entries as JSON files to the log folder (solo mode) or sends via REST API (distributed mode).
        """
        timestamp = timestamp or datetime.now().isoformat()
        for name, value in data.items():
            log_entry = {
                'run': self.name,
                'project': self.project,
                'step': step or 0,
                'category': category,
                'name': name,
                'timestamp': timestamp
            }
            if isinstance(value, (int, float)):
                log_entry['value_double'] = value
            elif isinstance(value, bool):
                log_entry['value_bool'] = value
            elif isinstance(value, (list,dict)):
                log_entry['value_json'] = json.dumps(value)
            elif isinstance(value, bytes):
                log_entry['value_blob'] = value.encode('base64')
            else:
                log_entry['value_string'] = str(value)
            # Write to local folder
            filename = f"{self._get_file_prefix()}.json"
            temp_filename = f"{filename}.tmp"
            filepath = os.path.join(self.log_path, temp_filename)
            # Write to a temporary file
            with open(filepath, 'w') as f:
                json.dump(log_entry, f)
            # Rename to the final filename to ensure atomicity
            final_filepath = os.path.join(self.log_path, filename)
            os.rename(filepath, final_filepath)

    def log_gpu_metrics_periodically(self, interval=60):
        """
        Periodically logs GPU system metrics every `interval` seconds in a separate thread.
        """
        def log_metrics():
            while True:
                gpu_metrics_list = self.get_gpu_system_metrics()
                for metric in gpu_metrics_list:
                    data = {}
                    for key, value in metric.items():
                        if key != 'gpu_index' and key != 'name':  # Skip index and name in logs
                            data[f'gpu_{metric["gpu_index"]}_{key}'] = value
                    self.log(data=data, step=None, category='gpu_system', timestamp=datetime.now().isoformat())
                time.sleep(interval)

        # Create and start a thread for logging GPU metrics
        logging_thread = threading.Thread(target=log_metrics, daemon=True)
        logging_thread.start()

# Global variables and functions
config: argparse.Namespace = argparse.Namespace()
run: WaddleLogger = None
server_process: Optional[subprocess.Popen] = None

def _assign_config(app_config):
    global config
    config = app_config

def init(project, log_root='.waddle/logs', config=None, use_gpu_metrics=True, gpu_metrics_interval=60):
    global run
    global server_process

    os.makedirs(log_root, exist_ok=True)

    # Assign the passed config to the global config
    config = config or argparse.Namespace()
    _assign_config(config)

    if use_gpu_metrics:
        # Check if the pynvml package is available
        try:
            nvmlInit()
        except Exception as e:
            print(f"Could not initialize NVML: {e}")
            use_gpu_metrics = False

    run = WaddleLogger(log_root=log_root, project=project, use_gpu_metrics=use_gpu_metrics, config=config)

    print("Waddle Logger initialized.")
    print("Run ID:", run.id)

    if use_gpu_metrics and gpu_metrics_interval > 0:
        run.log_gpu_metrics_periodically(interval=gpu_metrics_interval)

    return run

def finish():
    global server_process
    if server_process is not None:
        server_process.terminate()
        server_process.wait()
        server_process = None

def log(category, data, step, timestamp=None):
    if run is None:
        raise ValueError("WaddleLogger is not initialized. Please call `init()` first.")
    if not isinstance(data, dict):
        raise ValueError("The data must be a dictionary.")
    run.log(data=data, step=step, category=category, timestamp=timestamp)

