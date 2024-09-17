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
import uuid
import glob
import multiprocessing

class WaddleLogger:
    def __init__(self, db_root, project, id=None, config=None, use_gpu_metrics=True):
        self.db_root = db_root
        self.project = project
        self.id = id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_folder = os.path.join(db_root, self.project, self.id)
        os.makedirs(self.log_folder, exist_ok=True)
        self.config: argparse.Namespace = config
        self.use_gpu_metrics = use_gpu_metrics

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
        run_info_file = os.path.join(self.log_folder, 'run_info.json')
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
                        'name': nvmlDeviceGetName(handle).decode('utf-8'),
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
        Writes log entries as JSON files to the log folder.
        :param data: A dictionary of key-value pairs to log.
        :param step: The training step or time step associated with the log.
        :param category: The category of the data.
        :param timestamp: Optional timestamp; if not provided, current time is used.
        """
        timestamp = timestamp or datetime.now().isoformat()
        for name, value in data.items():
            log_entry = {
                'id': self.id,
                'step': step or 0,
                'category': category,
                'name': name,
                'value': value,
                'timestamp': timestamp
            }
            # Generate a unique filename
            filename = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
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
db_writer_process: Optional[multiprocessing.Process] = None

def _assign_config(app_config):
    global config
    config = app_config

def init(project, db_root='.waddle', config=None, use_gpu_metrics=True, gpu_metrics_interval=60):
    global run
    global db_writer_process

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

    run = WaddleLogger(db_root=db_root, project=project, use_gpu_metrics=use_gpu_metrics, config=config)
    if use_gpu_metrics and gpu_metrics_interval > 0:
        run.log_gpu_metrics_periodically(interval=gpu_metrics_interval)

    # Start the database writer process if not already running
    pid_file = os.path.join(db_root, 'db_writer.pid')
    launch_process = True
    if os.path.exists(pid_file):
        # Check if the process is running
        with open(pid_file, 'r') as f:
            pid = int(f.read())
        if psutil.pid_exists(pid):
            print("Database writer process is already running.")
            launch_process = False
        else:
            print("Found stale PID file. Starting a new database writer process.")
            os.remove(pid_file)
    if launch_process:
        # Start the database writer process
        db_writer_process = multiprocessing.Process(target=database_writer_process, args=(db_root, project))
        db_writer_process.start()
        # Write the PID to the pid file
        with open(pid_file, 'w') as f:
            f.write(str(db_writer_process.pid))

    return run

def finish():
    global db_writer_process
    if db_writer_process is not None:
        print("Terminating database writer process.")
        db_writer_process.terminate()
        db_writer_process.join()
        db_writer_process = None
        # Remove the PID file
        pid_file = os.path.join(run.db_root, 'db_writer.pid')
        if os.path.exists(pid_file):
            os.remove(pid_file)

def log(category, data, step, timestamp=None):
    if run is None:
        raise ValueError("WaddleLogger is not initialized. Please call `init()` first.")
    if not isinstance(data, dict):
        raise ValueError("The data must be a dictionary.")
    run.log(data=data, step=step, category=category, timestamp=timestamp)

def database_writer_process(db_root, project):
    """
    Process that continuously reads log entries from the log folders and writes them to the database.
    """
    db_path = os.path.join(db_root, f"{project}.db")
    con = duckdb.connect(db_path)

    # Create tables if they don't exist
    con.execute('''
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

    con.execute('''
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

    try:
        while True:
            log_folders = glob.glob(os.path.join(db_root, project, '*'))

            for log_folder in log_folders:
                run_id = os.path.basename(log_folder)
                run_info_file = os.path.join(log_folder, 'run_info.json')
                if os.path.exists(run_info_file):
                    # Read run_info and insert into the database if not already inserted
                    try:
                        con.execute("SELECT 1 FROM run_info WHERE id = ?", (run_id,))
                        if not con.fetchall():
                            with open(run_info_file, 'r') as f:
                                run_info = json.load(f)
                            # Insert into run_info table
                            con.execute('''
                                INSERT INTO run_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                run_info['id'], run_info['start_time'], run_info['cli_params'], run_info['python_version'],
                                run_info['os_info'], run_info['cpu_info'], run_info['total_memory'], run_info['gpu_info'],
                                run_info['code'].encode('utf-8'), run_info['git_hash'], run_info['timestamp']
                            ))
                    except Exception as e:
                        print(f"Error inserting run_info for {run_id}: {e}")

                # Process log entries
                log_files = glob.glob(os.path.join(log_folder, '*.json'))
                for log_file in log_files:
                    if os.path.basename(log_file) == 'run_info.json':
                        continue
                    try:
                        with open(log_file, 'r') as f:
                            log_entry = json.load(f)
                        # Determine the type of the value and insert accordingly
                        value = log_entry['value']
                        value_double = None
                        value_string = None
                        value_blob = None

                        if isinstance(value, (int, float)):
                            value_double = value
                        elif isinstance(value, str):
                            value_string = value
                        elif isinstance(value, (list, dict)):
                            # Convert lists and dicts to JSON string
                            value_string = json.dumps(value)
                        elif isinstance(value, bytes):
                            value_blob = value
                        else:
                            # For other types, store as string
                            value_string = json.dumps(value)

                        con.execute('''
                            INSERT INTO logs (
                                id, step, category, name, value_double, value_string, value_blob, timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            log_entry['id'], log_entry['step'], log_entry['category'], log_entry['name'],
                            value_double, value_string, value_blob, log_entry['timestamp']
                        ))
                        # Delete the log file after processing
                        os.remove(log_file)
                    except Exception as e:
                        print(f"Error processing log file {log_file}: {e}")

            time.sleep(1)  # Sleep for a short time before checking again
    except KeyboardInterrupt:
        print("Database writer process terminated.")
    finally:
        con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, default='experiment')
    args = parser.parse_args()

    # Initialize WaddleLogger and start the database writer process
    init(project=args.project_name, config=args)

    # Simulate the rest of your ML code here
    # The GPU logging will run in the background
    try:
        for step in range(10):
            log_data = {'loss': 0.01 * step}
            log(category='model', data=log_data, step=step)
            time.sleep(5)  # Simulating training steps
    finally:
        # Finish up and terminate the database writer process
        finish()

    print("Done!")

    # The database should now be up to date
