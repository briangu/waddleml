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
    def __init__(self, db_root, project, id=None, config=None, use_gpu_metrics=True, server_url=None):
        self.db_root = db_root
        self.project = project
        self.id = id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_folder = os.path.join(db_root, self.project, self.id)
        os.makedirs(self.log_folder, exist_ok=True)
        self.config: argparse.Namespace = config
        self.use_gpu_metrics = use_gpu_metrics
        self.server_url = server_url

        # Log initial system, CLI parameters, and code
        self.log_run_info()

    def _get_file_prefix(self):
        return os.path.join(self.log_folder, f"{int(time.time() * 1000)}_{uuid.uuid4().hex}")

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
            'id': self.id,
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

        # Write run_info to a file in the log folder
        run_info_file = os.path.join(self.log_folder, f'{self._get_file_prefix()}.run_info.json')
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
                'id': self.id,
                'project': self.project,
                'step': step or 0,
                'category': category,
                'name': name,
                'value': value,
                'timestamp': timestamp
            }
            # Write to local folder
            filename = f"{self._get_file_prefix()}.json"
            temp_filename = f"{filename}.tmp"
            filepath = os.path.join(self.log_folder, temp_filename)
            # Write to a temporary file
            with open(filepath, 'w') as f:
                json.dump(log_entry, f)
            # Rename to the final filename to ensure atomicity
            final_filepath = os.path.join(self.log_folder, filename)
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

def init(project, log_root='.waddle/logs', config=None, use_gpu_metrics=True, gpu_metrics_interval=60, server_url=None, server_port=8000, server_bind="127.0.0.1"):
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

    if server_url is None:
        # Start the waddle server as a subprocess
        server_cmd = ['waddle', '--mode', 'server', '--server-port', str(server_port), '--server-bind', server_bind, '--db-root', log_root, '--project', project, '--watch-folder', os.path.join(log_root, project)]
        server_process = subprocess.Popen(server_cmd)
        # Allow the server some time to start
        time.sleep(2)
        server_url = 'http://localhost:8000'
    else:
        if not server_url:
            raise ValueError("In distributed mode, server_url must be specified.")

    run = WaddleLogger(db_root=log_root, project=project, use_gpu_metrics=use_gpu_metrics, config=config, server_url=server_url)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, default='experiment')
    parser.add_argument('--mode', type=str, choices=['solo', 'distributed'], default='solo')
    parser.add_argument('--server-url', type=str, default=None)
    args = parser.parse_args()

    # Initialize WaddleLogger
    init(project=args.project_name, config=args, mode=args.mode, server_url=args.server_url)

    # Simulate the rest of your ML code here
    # The GPU logging will run in the background
    try:
        for step in range(10):
            log_data = {'loss': 0.01 * step}
            log(category='model', data=log_data, step=step)
            time.sleep(5)  # Simulating training steps
    finally:
        # Finish up and terminate the server process
        finish()

    print("Done!")
