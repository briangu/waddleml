import argparse
import asyncio
import glob
import json
import os
import threading
import time
from contextlib import asynccontextmanager
import logging

import duckdb
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import math

logger = logging.getLogger(__name__)

waddle_server_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global waddle_server_instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root', type=str, default='.waddle')
    parser.add_argument('--project', type=str, default='experiment')
    parser.add_argument('--watch-folder', type=str, default=None)
    args, _ = parser.parse_known_args()

    loop = asyncio.get_running_loop()
    waddle_server_instance = WaddleServer(db_root=args.db_root, project=args.project, watch_folder=args.watch_folder, loop=loop)
    yield
    if waddle_server_instance:
        waddle_server_instance.stop()

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connection established: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed: {websocket.client}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to client {connection.client}: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

class WaddleServer:
    def __init__(self, db_root, project, watch_folder=None, loop=None, watch_delay=10):
        self.db_root = db_root
        self.project = project
        self.db_path = os.path.join(db_root, f"{project}.db")
        self.watch_folder = watch_folder
        self.loop = loop
        self.watch_delay = watch_delay

        self.con = duckdb.connect(self.db_path)

        # Create tables if they don't exist
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
                value_double DOUBLE,
                value_string VARCHAR,
                value_blob BLOB,
                timestamp TIMESTAMP
            );
        ''')

        if self.watch_folder:
            # Start the folder-watching thread
            self.watching = True
            self.watch_thread = threading.Thread(target=self.watch_folder_for_logs, daemon=True)
            self.watch_thread.start()

    def stop(self):
        if self.watch_folder:
            self.watching = False
            self.watch_thread.join()

    def watch_folder_for_logs(self):
        try:
            while self.watching:
                logger.info(f"{time.time()} Watching folder for logs...")

                # Get all directories recursively
                log_folders = glob.glob(os.path.join(self.watch_folder, '**'), recursive=True)
                log_folders = [f for f in log_folders if os.path.isdir(f)]

                for log_folder in log_folders:
                    logger.info(f"Processing log folder: {log_folder}")

                    # Process log entries recursively
                    log_files = glob.glob(os.path.join(log_folder, '**', '*.json'), recursive=True)
                    for log_file in log_files:
                        try:
                            logging.info(f"Processing log file: {log_file}")
                            with open(log_file, 'r') as f:
                                log_entry = json.load(f)
                            if os.path.basename(log_file) == 'run_info.json':
                                self.ingest_log_entry({'run_info': log_entry})
                            else:
                                # Ingest log entry
                                self.ingest_log_entry(log_entry)
                            # Delete the log file after processing
                            os.remove(log_file)
                        except Exception as e:
                            logger.error(f"Error processing log file {log_file}: {e}")

                time.sleep(self.watch_delay)
        except Exception as e:
            logger.error(f"Error in watch_folder_for_logs: {e}")

    def ingest_log_entry(self, log_entry):
        try:
            # Handle run_info separately
            if 'run_info' in log_entry:
                run_info = log_entry['run_info']
                run_id = run_info['id']
                self.con.execute("SELECT 1 FROM run_info WHERE id = ?", (run_id,))
                if not self.con.fetchall():
                    # Insert into run_info table
                    self.con.execute('''
                        INSERT INTO run_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        run_info['id'],
                        run_info['start_time'],
                        run_info['cli_params'],
                        run_info['python_version'],
                        run_info['os_info'],
                        run_info['cpu_info'],
                        run_info['total_memory'],
                        run_info['gpu_info'],
                        run_info['code'].encode('utf-8'),
                        run_info['git_hash'],
                        run_info['timestamp']
                    ))
                return

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

            self.con.execute('''
                INSERT INTO logs (
                    id, step, category, name, value_double, value_string, value_blob, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_entry['id'], log_entry['step'], log_entry['category'], log_entry['name'],
                value_double, value_string, value_blob, log_entry['timestamp']
            ))

            # Broadcast the new log entry to connected WebSocket clients
            asyncio.run_coroutine_threadsafe(manager.broadcast(log_entry), self.loop)

        except Exception as e:
            logger.error(f"Error ingesting log entry: {e}")
            raise

@app.post("/ingest")
async def ingest_log(log_entry: dict):
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        waddle_server_instance.ingest_log_entry(log_entry)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=['float', 'double']).columns
    for col in float_cols:
        df[col] = df[col].apply(lambda x: x if isinstance(x, float) and math.isfinite(x) else None)
    #convert timestamps to ISO format
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    print(df)
    del df['value_blob']  # Remove the blob column
    return df

@app.get("/data")
async def get_data(history: int = 2000):
    history = min(max(history, 1), 2000)  # Limit history to between 1 and 1000
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        df = waddle_server_instance.con.execute("SELECT * FROM logs").fetchdf()
        df = sanitize_df(df)
        df = df[-history:]
        x = df.to_dict(orient='records')
        return x
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep the connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def main(port=8000, bind="127.0.0.1", log_level="critical"):
    logging.getLogger("uvicorn.error").handlers = []
    logging.getLogger("uvicorn.error").propagate = False

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level)
    logger.setLevel(numeric_level)

    logger.info("Starting Waddle server.")
    print("Starting Waddle server.")
    logger.info(f"Listening at http://{bind}:{port}")
    print(f"Listening at http://{bind}:{port}")
    uvicorn.run("waddle.server:app", host=bind, port=port, log_level=log_level, lifespan="on")
    logger.info("Waddle server stopped.")
    print("Waddle server stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-bind", type=str, default="127.0.0.1")
    parser.add_argument("--log-level", type=str, default="critical")
    args = parser.parse_args()
    main(port=args.server_port, bind=args.server_bind, log_level=args.log_level)
