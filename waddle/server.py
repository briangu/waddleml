import argparse
import asyncio
import glob
import json
import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import duckdb
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from datetime import datetime
from functools import lru_cache

import websockets.legacy
import websockets.legacy.client


logger = logging.getLogger(__name__)

waddle_server_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global waddle_server_instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root', type=str, default='.waddle')
    parser.add_argument('--log-root', type=str, default=os.path.join('.waddle', 'logs'))
    parser.add_argument('--peer', type=str, action='append', default=[], help='URL of peer WaddleServer to connect to.')
    args, _ = parser.parse_known_args()

    loop = asyncio.get_running_loop()
    waddle_server_instance = WaddleServer(db_root=args.db_root, log_root=args.log_root, loop=loop, peers=args.peer)
    await waddle_server_instance.start()
    try:
        yield
    except Exception as e:
        logger.error(f"Error in lifespan context manager: {e}")
    finally:
        await waddle_server_instance.stop()

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
    def __init__(self, db_root, log_root=None, loop=None, peers=None, watch_delay=10):
        self.db_path = os.path.join(db_root, f"waddle.db")
        self.log_root = log_root or os.path.join(db_root, "logs")
        self.loop = loop or asyncio.get_event_loop()
        self.watch_delay = watch_delay

        self.peers = peers or []
        self.peer_tasks = []
        self.peer_ws_clients = {}

        self.con = duckdb.connect(self.db_path)

        # Initialize database tables
        self._initialize_database()

        # For log watching
        self.watching = True

    def _initialize_database(self):
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

    async def start(self):
        # Start log watching
        self.watch_task = asyncio.create_task(self.watch_for_logs())

        # Start peer synchronization
        self.sync_task = asyncio.create_task(self.sync_with_peers())

    async def stop(self):
        # Stop log watching
        self.watching = False
        if hasattr(self, 'watch_task'):
            await self.watch_task

        # Disconnect from peers
        for task in self.peer_tasks:
            task.cancel()
        await asyncio.gather(*self.peer_tasks, return_exceptions=True)

    async def watch_for_logs(self):
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

            await asyncio.sleep(self.watch_delay)

    def _insert_project_info(self, project_info):
        self.con.execute("SELECT id FROM project_info WHERE name = ?", (project_info['name'],))
        project_id = self.con.fetchone()
        if not project_id:
            self.con.execute('INSERT INTO project_info (name, timestamp, data) VALUES (?, ?, ?)', (
                project_info['name'],
                project_info['timestamp'],
                project_info
            ))
            project_id = self.con.execute("SELECT currval('seq_project_info')").fetchone()[0]
        else:
            project_id = project_id[0]
        return project_id

    def _insert_run_info(self, project_id, run_info):
        run_name = run_info.get('name') or run_info.get('id')
        self.con.execute("SELECT id FROM run_info WHERE project_id = ? AND name = ?", (project_id, run_name,))
        run_id = self.con.fetchone()
        if not run_id:
            # Insert into run_info table
            self.con.execute('INSERT INTO run_info (project_id, name, start_time, data) VALUES (?, ?, ?, ?)', (
                project_id,
                run_name,
                run_info.get('start_time') or run_info.get('timestamp') or datetime.now(),
                run_info,
            ))
            run_id = self.con.execute("SELECT currval('seq_run_info')").fetchone()[0]
        else:
            run_id = run_id[0]
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

    @lru_cache
    def _resolve_project_and_run(self, project_name, run_name, timestamp=None):
        timestamp = timestamp or datetime.now()
        # Resolve project and run names into IDs and insert into the database if not already present
        project_id = self._insert_project_info({"name": project_name, "timestamp": timestamp})
        run_id = self._insert_run_info(project_id, {"name": run_name, "start_time": timestamp})
        return project_id, run_id

    def _ingest_log_entry(self, log_entry):
        try:
            # Handle run_info separately
            if 'run_info' in log_entry:
                run_info = log_entry['run_info']
                project_name = run_info['project']
                project_id = self._insert_project_info({"name": project_name, "timestamp": run_info.get('start_time') or run_info.get('timestamp') or datetime.now()})
                self._insert_run_info(project_id, log_entry['run_info'])
                return

            # Resolve project and run names into IDs and insert into the database if not already present
            run_name = log_entry.get('run') or log_entry.get('id') or log_entry.get('run_id')
            project_name = log_entry.get('project') or "default"
            project_id, run_id = self._resolve_project_and_run(project_name, run_name, log_entry.get('timestamp'))
            log_id = self._insert_log_entry(run_id, log_entry)
            if 'run_name' in log_entry:
                del log_entry['run_name']
            log_entry['project_id'] = project_id
            log_entry['run_id'] = run_id
            log_entry['id'] = log_id

            # Broadcast the new log entry to connected WebSocket clients
            asyncio.run_coroutine_threadsafe(manager.broadcast({"command": "LOG", "data": log_entry}), self.loop)

        except Exception as e:
            print("Error ingesting log entry:", log_entry)
            import traceback
            traceback.print_exc()
            logger.error(f"Error ingesting log entry: {e}")
            raise

    async def sync_with_peers(self):
        for peer_url in self.peers:
            task = asyncio.create_task(self.sync_with_peer(peer_url))
            self.peer_tasks.append(task)

    async def sync_with_peer(self, peer_url):
        try:
            logger.info(f"Starting synchronization with peer: {peer_url}")

            ws_url = peer_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
            logger.info(f"Connecting to peer WebSocket at {ws_url}")

            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Connected to peer WebSocket at {ws_url}")

                self.peer_ws_clients[peer_url] = websocket

                # Initial synchronization
                await websocket.send_json({"command": "GET_PROJECTS"})

                async for message in websocket:
                    message = json.loads(message)
                    response = await self.handle_ws_message(message, websocket, peer_url)
                    if response:
                        await websocket.send_json(response)

        except asyncio.CancelledError:
            logger.info(f"Sync task with peer {peer_url} has been cancelled.")
        except Exception as e:
            logger.error(f"Error syncing with peer {peer_url}: {e}")
            # Optionally implement retry logic here

    async def handle_ws_message(self, message, websocket, peer_url=None):
        try:
            if message['command'] == "GET_PROJECTS":
                projects = await get_project_info()
                return {"command": "PROJECTS", "data": projects}

            elif message['command'] == "GET_RUNS":
                runs = await get_project_runs(message['project_id'])
                return {"command": "RUNS", "project_id": message['project_id'], "data": runs}

            elif message['command'] == "GET_RUN":
                run = await get_run(message['project_id'], message['run_id'], from_step=message.get('from_step', 0))
                return {"command": "RUN", "project_id": message['project_id'], "run_id": message['run_id'], "data": run}

            elif message['command'] == "LOG":
                self._ingest_log_entry(message['data'])
                return None

            elif message['command'] == "PROJECTS":
                peer_projects = message['data']
                logger.info(f"Received projects from peer: {peer_projects}")

                # Fetch local projects
                local_projects = await get_project_info()
                local_project_names = set(p['name'] for p in local_projects)

                # Determine which projects to sync
                for project in peer_projects:
                    project_name = project['name']
                    if project_name not in local_project_names:
                        logger.info(f"Synchronizing project: {project_name}")
                        project_id = project['id']

                        # Send GET_RUNS command over websocket
                        await websocket.send_json({"command": "GET_RUNS", "project_id": project_id})
                logger.info(f"Initial synchronization with peer {peer_url} completed.")

            elif message['command'] == "RUNS":
                project_id = message['project_id']
                peer_runs = message['data']

                # Fetch local runs for this project
                local_runs = await get_project_runs(project_id)
                local_run_ids = {run['id']: run['steps'] for run in local_runs}

                # Determine which runs to sync
                for run in peer_runs:
                    run_id = run['id']
                    if run_id not in local_run_ids:
                        logger.info(f"Synchronizing run: {run_id} for project: {project_id}")

                        # Send GET_RUN command over websocket
                        await websocket.send_json({
                            "command": "GET_RUN",
                            "project_id": project_id,
                            "run_id": run_id,
                            "from_step": local_run_ids.get(run_id, 0)
                        })

            elif message['command'] == "RUN":
                project_id = message['project_id']
                run_id = message['run_id']
                run_logs = message['data']

                # Ingest run logs
                for log_entry in run_logs:
                    self._ingest_log_entry(log_entry)
                logger.info(f"Synchronized run {run_id} for project {project_id}")

            else:
                logger.warning(f"Unknown command received: {message['command']}")

        except Exception as e:
            logger.error(f"Error handling message {message}: {e}")
            # Optionally send an error message back
            if websocket:
                await websocket.send_json({"command": "ERROR", "message": str(e)})


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamps to ISO format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')

    df = df.sort_values(by='step')

    return df


@app.get("/projects")
async def get_project_info():
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        df = waddle_server_instance.con.execute("SELECT * FROM project_info").fetchdf()
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/runs")
async def get_project_runs(project_id: int):
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        df = waddle_server_instance.con.execute("""
            SELECT
                id,
                project_id,
                start_time,
                data,
                (SELECT COUNT(id) FROM logs WHERE run_id=id) as steps
            FROM run_info
            WHERE project_id = ? AND steps > 0
            ORDER BY start_time DESC
        """, (project_id,)).fetchdf()
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df.to_dict(orient='records')
    except Exception as e:
        import traceback
        print(e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/runs/{run_id}")
async def get_run(project_id: int, run_id: int, history: Optional[int] = 10, from_step: Optional[int] = None):
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        # first get the list of names
        df = waddle_server_instance.con.execute("SELECT DISTINCT name FROM logs WHERE run_id = ?", (run_id,)).fetchdf()
        # for each name then get the history of that name and collect it in a list
        data = []
        for name in df['name']:
            if from_step:
                df_logs = waddle_server_instance.con.execute("""
                    SELECT * FROM logs
                    WHERE run_id = ? AND name = ? AND step >= ?
                    ORDER BY step DESC
                """, (run_id, name, from_step)).fetchdf()
            else:
                df_logs = waddle_server_instance.con.execute("""
                    SELECT * FROM logs
                    WHERE run_id = ? AND name = ?
                    ORDER BY step DESC LIMIT ?
                """, (run_id, name, history)).fetchdf()
            clean_df = sanitize_df(df_logs)
            data.extend(clean_df.to_dict(orient='records'))
        logger.info(f"Returning {len(data)} records for run {run_id}")
        return data
    except Exception as e:
        import traceback
        print(e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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
            msg = await websocket.receive_json()
            response = await waddle_server_instance.handle_ws_message(msg, websocket)
            if response:
                await websocket.send_json(response)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def main(port=8000, bind="127.0.0.1", log_level="critical", peers=[]):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-bind", type=str, default="127.0.0.1")
    parser.add_argument("--log-level", type=str, default="critical")
    parser.add_argument("--peer", type=str, action='append', default=[], help='URL of peer WaddleServer to connect to.')
    parser.add_argument('--db-root', type=str, default='.waddle')
    parser.add_argument('--log-root', type=str, default=os.path.join('.waddle', 'logs'))
    args = parser.parse_args()
    main(port=args.server_port, bind=args.server_bind, log_level=args.log_level, peers=args.peer)
