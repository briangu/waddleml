import argparse
import asyncio
import json
import os
import logging
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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import websockets.legacy
import websockets.legacy.client


logger = logging.getLogger(__name__)

waddle_server_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global waddle_server_instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root', type=str, default='.waddle')
    parser.add_argument('--log-root', type=str, default=None, help="Directory to watch for log files.")
    parser.add_argument('--watch-delay', type=int, default=10, help="Delay in seconds between watching for new log files.")
    parser.add_argument('--peer', type=str, action='append', default=[], help='URL of peer WaddleServer to source data from.')
    args, _ = parser.parse_known_args()

    loop = asyncio.get_running_loop()
    waddle_server_instance = WaddleServer(db_root=args.db_root, log_root=args.log_root, loop=loop, peers=args.peer, watch_delay=args.watch_delay)
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

class LogFileEventHandler(FileSystemEventHandler):
    def __init__(self, server: 'WaddleServer'):
        super().__init__()
        self.server = server

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.json'):
            try:
                with open(event.src_path, 'r') as f:
                    log_entry = json.load(f)
                    self.server._ingest_log_entry(log_entry)
                os.remove(event.src_path)
            except Exception as e:
                logger.error(f"Error processing log file {event.src_path}: {e}")

class WaddleServer:
    def __init__(self, db_root, log_root=None, loop=None, peers=None, watch_delay=10):
        self.db_path = os.path.join(db_root, f"waddle.db")
        self.log_root = log_root
        self.loop = loop or asyncio.get_event_loop()
        self.watch_delay = watch_delay
        self.watching_logs = self.log_root is not None

        self.peers = peers or []
        self.peer_tasks = []
        self.peer_ws_clients = {}

        self._initialize_database()

    def _initialize_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.con = duckdb.connect(self.db_path)

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
        # Start log watching using watchdog
        self.watch_task = asyncio.create_task(self.watch_for_logs())

        # Start peer synchronization
        self.sync_task = asyncio.create_task(self.sync_with_peers())

    async def stop(self):
        # Stop log watching
        self.watching_logs = False
        if hasattr(self, 'watch_task'):
            await self.watch_task

        # Disconnect from peers
        for task, peer in zip(self.peer_tasks, self.peer_ws_clients.values()):
            task.cancel()
            await peer.close()
        await asyncio.gather(*self.peer_tasks, return_exceptions=True)

    async def watch_for_logs(self):
        if not self.watching_logs:
            return

        event_handler = LogFileEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.log_root, recursive=True)
        observer.start()

        try:
            while self.watching_logs:
                await asyncio.sleep(self.watch_delay)
        finally:
            observer.stop()
            observer.join()

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
                websocket: websockets.legacy.client.WebSocketClientProtocol = websocket
                websocket.send_json = lambda x: websocket.send(json.dumps(x))

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
                runs = await get_project_runs(message['project_id'], convert_timestamps=True)
                return {"command": "RUNS", "project_id": message['project_id'], "data": runs}

            elif message['command'] == "GET_RUN":
                run = await get_run(message['project_id'], message['run_id'], from_step=message.get('from_step', 0), convert_timestamps=True)
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
                    self._ingest_log_entry(json.loads(log_entry['data']))
                logger.info(f"Synchronized run {run_id} for project {project_id}")

            else:
                logger.warning(f"Unknown command received: {message['command']}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error handling message {message}: {e}")
            # Optionally send an error message back
            if websocket:
                await websocket.send_json({"command": "ERROR", "message": str(e)})

@app.get("/projects")
async def get_project_info():
    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        df = waddle_server_instance.con.execute("SELECT * FROM project_info").fetchdf()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/runs")
async def get_project_runs(project_id: int, convert_timestamps: Optional[bool] = True):
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
        if convert_timestamps:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df.to_dict(orient='records')
    except Exception as e:
        import traceback
        print(e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/runs/{run_id}")
async def get_run(
    project_id: int,
    run_id: int,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    history: Optional[int] = 10,
    from_step: Optional[int] = None,
    convert_timestamps: Optional[bool] = True
):
    """
    Returns log data for the specified run.

    Logic priority:
      1) If start_date or end_date is provided, filter by timestamp >= start_date, <= end_date.
      2) Else if from_step is provided, filter logs by step >= from_step.
      3) Else fallback to limiting logs by `history` (the most recent N entries).
    """

    if not waddle_server_instance:
        raise HTTPException(status_code=500, detail="Server not initialized")
    try:
        # 0) First get the distinct chart names for this run.
        name_df = waddle_server_instance.con.execute(
            "SELECT DISTINCT name FROM logs WHERE run_id = ?",
            (run_id,)
        ).fetchdf()

        if name_df.empty:
            return []  # no logs for this run

        # Prepare date range filters if provided
        date_filters = []
        date_params = []

        # We'll interpret start_date and end_date as inclusive bounds.
        # Adjust as needed, e.g. end_date + " 23:59:59" for day-based inclusivity.
        if start_datetime:
            # Parse start_datetime (accepts ISO 8601 or date-time strings)
            try:
                start_dt = pd.to_datetime(start_datetime, errors='raise')
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid start_datetime format (expected ISO format YYYY-MM-DDTHH:MM:SS)."
                )
            # Use Python datetime for binding to TIMESTAMP parameter
            # filter on timestamp, casting the parameter to TIMESTAMP for safety
            date_filters.append("timestamp >= CAST(? AS TIMESTAMP)")
            date_params.append(start_dt.to_pydatetime())

        if end_datetime:
            # Parse end_datetime (accepts ISO 8601 or date-only strings)
            try:
                end_dt = pd.to_datetime(end_datetime, errors='raise')
                # normalize to end of day (23:59:59) for date-bound queries
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid end_datetime format (expected ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
                )
            # Use Python datetime for binding to TIMESTAMP parameter
            # filter on timestamp, casting the parameter to TIMESTAMP for safety
            date_filters.append("timestamp <= CAST(? AS TIMESTAMP)")
            date_params.append(end_dt.to_pydatetime())

        # We'll accumulate all log rows in a list of dicts
        data = []

        # 1) Loop over each chart name to get logs
        for name in name_df['name']:
            # Build base query
            base_query = "SELECT * FROM logs WHERE run_id = ? AND name = ?"
            query_params = [run_id, name]

            # 1a) If date filters are present, that takes priority
            if date_filters:
                # e.g.: SELECT * FROM logs WHERE run_id=? AND name=? AND timestamp>=? AND timestamp<=? ORDER BY step DESC
                for f in date_filters:
                    base_query += f" AND {f}"
                query_params.extend(date_params)
                base_query += " ORDER BY step DESC"

            # 1b) Otherwise, check if from_step is provided
            elif from_step is not None:
                # e.g.: SELECT * FROM logs WHERE run_id=? AND name=? AND step>=? ORDER BY step DESC
                base_query += " AND step >= ? ORDER BY step DESC"
                query_params.append(from_step)

            # 1c) Otherwise fallback to the `history` limit
            else:
                # e.g.: SELECT * FROM logs WHERE run_id=? AND name=? ORDER BY step DESC LIMIT ?
                base_query += " ORDER BY step DESC LIMIT ?"
                query_params.append(history)

            # Execute query
            df = waddle_server_instance.con.execute(base_query, query_params).fetchdf()

            # 2) Convert timestamps to ISO format if requested
            if convert_timestamps and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') \
                                     .dt.strftime('%Y-%m-%dT%H:%M:%S')

            # 3) Sort logs in ascending order by step before returning (so the chart can process them sequentially)
            if not df.empty and 'step' in df.columns:
                df = df.sort_values(by='step', ascending=True)

            # 4) Merge into final result
            data.extend(df.to_dict(orient='records'))

        logger.info(f"Returning {len(data)} records for run {run_id}")
        return data

    except Exception as e:
        import traceback
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
