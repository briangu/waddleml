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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

waddle_server_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global waddle_server_instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root', type=str, default='.waddle')
    parser.add_argument('--project', type=str, default='experiment')
    parser.add_argument('--log-root', type=str, default=os.path.join('.waddle', 'logs'))
    args, _ = parser.parse_known_args()

    loop = asyncio.get_running_loop()
    waddle_server_instance = WaddleServer(db_root=args.db_root, project=args.project, log_root=args.log_root, loop=loop)
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
    def __init__(self, db_root, project, log_root=None, loop=None, watch_delay=10):
        self.project = project
        self.db_path = os.path.join(db_root, f"waddle.db")
        self.log_root = log_root or os.path.join(db_root, "logs")
        self.loop = loop
        self.watch_delay = watch_delay

        self.con = duckdb.connect(self.db_path)

        # add project info to the database
        self.con.execute('''
            CREATE SEQUENCE IF NOT EXISTS seq_project_info START 1;
            CREATE TABLE IF NOT EXISTS project_info (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_project_info'),
                name VARCHAR,
                timestamp TIMESTAMP,
                data JSON,
                UNIQUE (name)
            );
        ''')

        # insert the default project info
        self.con.execute('INSERT OR IGNORE INTO project_info (name, timestamp, data) VALUES (?, ?, ?)', (project, datetime.now(), {}))

        # Create tables if they don't exist
        self.con.execute('''
            CREATE SEQUENCE IF NOT EXISTS seq_run_info START 1;
            CREATE TABLE IF NOT EXISTS run_info (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_run_info'),
                project_id INTEGER,
                name VARCHAR,
                start_time TIMESTAMP,
                data JSON,
                UNIQUE (project_id, name),
                FOREIGN KEY (project_id) REFERENCES project_info(id)
            );
        ''')

        self.con.execute('''
            CREATE SEQUENCE IF NOT EXISTS seq_logs START 1;
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_logs'),
                run_id INTEGER,
                step INTEGER,
                category VARCHAR,
                name VARCHAR,
                timestamp TIMESTAMP,
                data JSON,
                FOREIGN KEY (run_id) REFERENCES run_info(id)
            );
        ''')

        if self.log_root:
            # Start the folder-watching thread
            self.watching = True
            self.watch_thread = threading.Thread(target=self._watch_for_logs, daemon=True)
            self.watch_thread.start()

    def stop(self):
        if self.log_root:
            self.watching = False
            self.watch_thread.join()

    def _watch_for_logs(self):
        try:
            while self.watching:
                logger.info(f"{time.time()} Watching folder for logs...")

                # Get all directories recursively
                log_folders = glob.glob(os.path.join(self.log_root, '**'), recursive=True)
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
                                # Ingest log entry
                                self._ingest_log_entry(log_entry)
                            # Delete the log file after processing
                            os.remove(log_file)
                        except Exception as e:
                            logger.error(f"Error processing log file {log_file}: {e}")

                time.sleep(self.watch_delay)
        except Exception as e:
            logger.error(f"Error in log_root_for_logs: {e}")

    def _insert_project_info(self, project_info):
        self.con.execute("SELECT id FROM project_info WHERE name = ?", (project_info['name'],))
        project_id = self.con.fetchall()
        if not project_id:
            self.con.execute('INSERT INTO project_info (name, timestamp, data) VALUES (?, ?, ?)', (
                project_info['name'],
                project_info['timestamp'],
                project_info
            ))
            project_id = self.con.execute("SELECT currval('seq_project_info')").fetchone()[0]
        else:
            project_id = project_id[0][0]
        return project_id

    def _insert_run_info(self, project_id, run_info):
        run_name = run_info['name']
        self.con.execute("SELECT id FROM run_info WHERE project_id =? AND name = ?", (project_id, run_name,))
        run_id = self.con.fetchall()
        if not run_id:
            # Insert into run_info table
            self.con.execute('INSERT INTO run_info (project_id, name, start_time, data) VALUES (?, ?, ?, ?)', (
                project_id,
                run_info['name'],
                run_info['start_time'],
                run_info,
            ))
            run_id = self.con.execute("SELECT currval('seq_run_info')").fetchone()[0]
        else:
            run_id = run_id[0][0]
        return run_id

    def _insert_log_entry(self, run_id, log_entry):
        self.con.execute('INSERT INTO logs (run_id, step, category, name, timestamp, data) VALUES (?, ?, ?, ?, ?, ?)', (
            run_id,
            log_entry['step'],
            log_entry['category'],
            log_entry['name'],
            log_entry['timestamp'],
            log_entry
        ))
        log_id = self.con.execute("SELECT currval('seq_logs')").fetchone()[0]
        return log_id

    def _parse_scoped_run_name(self, run_name):
        run_name = run_name.split('/')
        if len(run_name) == 1:
            project_name = "default"
            run_name = run_name[0]
        elif len(run_name) == 2:
            project_name, run_name = run_name
        else:
            raise ValueError(f"Invalid run name: {run_name}")
        return project_name, run_name

    @lru_cache
    def _resolve_project_and_run(self, project_name, run_name):
        # Resolve project and run names into IDs and insert into the database if not already present
        project_id = self._insert_project_info({"name": project_name, "timestamp": datetime.now()})
        run_id = self._insert_run_info(project_id, {"name": run_name, "start_time": datetime.now()})
        return project_id, run_id

    def _ingest_log_entry(self, log_entry):
        try:
            # Handle run_info separately
            if 'run_info' in log_entry:
                run_info = log_entry['run_info']
                project_name = run_info['project']
                project_id = self._insert_project_info({"name": project_name, "timestamp": run_info['start_time']})
                self._insert_run_info(project_id, log_entry['run_info'])
                return

            # Resolve project and run names into IDs and insert into the database if not already present
            scoped_run_name = log_entry.get('run') or log_entry.get('id')
            project_name, run_name = self._parse_scoped_run_name(scoped_run_name)
            project_id, run_id = self._resolve_project_and_run(project_name, run_name)
            log_id = self._insert_log_entry(run_id, log_entry)
            if 'run_name' in log_entry:
                del log_entry['run_name']
            log_entry['project_id'] = project_id
            log_entry['run_id'] = run_id
            log_entry['id'] = log_id

            # Broadcast the new log entry to connected WebSocket clients
            asyncio.run_coroutine_threadsafe(manager.broadcast(log_entry), self.loop)

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error ingesting log entry: {e}")
            raise

@app.post("/ingest")
async def ingest_log(log_entry: dict):
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        waddle_server_instance._ingest_log_entry(log_entry)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    # coalesce the value_double and value_string columns into a value column
    # df['value'] = df['value_double'].combine_first(df['value_string'])

    for i, row in df.iterrows():
        if not pd.isna(row['value_double']):
            df.at[i, 'value'] = row['value_double']
        else:
            df.at[i, 'value'] = row['value_string']

    # drop the value_double and value_string columns
    df = df.drop(columns=['value_double', 'value_string'])

    # Convert timestamps to ISO format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')

    df = df.sort_values(by='step')

    return df


@app.get("/projects")
async def get_info():
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        df = waddle_server_instance.con.execute("SELECT * FROM project_info").fetchdf()
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs")
async def get_runs():
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        df = waddle_server_instance.con.execute("SELECT id,start_time,data FROM run_info ORDER BY start_time DESC").fetchdf()
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df.to_dict(orient='records')
    except Exception as e:
        import traceback
        print(e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run/{run_id}")
async def get_run(run_id: str, history: int = 10):
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        # first get the list of names
        df = waddle_server_instance.con.execute("SELECT DISTINCT name FROM logs WHERE id = ?", (run_id,)).fetchdf()
        # for each name then get the history of that name and collect it in a list
        data = []
        for name in df['name']:
            df = waddle_server_instance.con.execute("SELECT id,step,category,name,value_double,value_string,timestamp FROM logs WHERE id = ? AND name = ? ORDER BY step DESC LIMIT ?", (run_id,name,history,)).fetchdf()
            clean_df = sanitize_df(df).to_dict(orient='records')
            data.extend(clean_df)
        logger.info(f"Returning {len(data)} records for run {run_id}")
        return data
    except Exception as e:
        import traceback
        print(e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/data")
# async def get_data(history: int = 2000):
#     history = min(max(history, 1), 2000)  # Limit history to between 1 and 1000
#     if not waddle_server_instance:
#         raise HTTPException(status_code=500, detail="Server not initialized")
#     try:
#         df = waddle_server_instance.con.execute("SELECT * FROM logs").fetchdf()
#         df = sanitize_df(df)
#         df = df[-history:]
#         x = df.to_dict(orient='records')
#         return x
#     except Exception as e:
#         import traceback
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# get static assets
@app.get("/static/{filename}")
async def get_static(filename: str):
    filename = os.path.basename(filename)
    filepath = os.path.join(os.path.dirname(__file__), "static", filename)
    return FileResponse(filepath)


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
