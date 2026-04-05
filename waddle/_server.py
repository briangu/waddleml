"""Starlette app + routes + WebSocket for the Waddle dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, FileResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ._dashboard_api import DashboardAPI

_ws_clients: Set[WebSocket] = set()


def create_app(db_path: str = None, static_dir: str = None, api: DashboardAPI = None) -> Starlette:
    if api is None:
        api = DashboardAPI(db_path)
    if static_dir is None:
        static_dir = str(Path(__file__).parent / "static")
    static = Path(static_dir)

    async def index(request: Request) -> Response:
        index_file = static / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>waddle dashboard</h1><p>index.html not found</p>", status_code=404)

    async def list_runs(request: Request) -> JSONResponse:
        params = request.query_params
        runs = api.list_runs(
            project=params.get("project"),
            status=params.get("status"),
            sort=params.get("sort", "started_at"),
            order=params.get("order", "desc"),
            limit=int(params.get("limit", "200")),
            offset=int(params.get("offset", "0")),
        )
        return JSONResponse(runs)

    async def get_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        data = api.get_run(run_id)
        if not data:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(data)

    async def get_metrics(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        key = request.query_params.get("key")
        limit = int(request.query_params.get("limit", "5000"))
        data = api.get_metrics(run_id, key=key, limit=limit)
        return JSONResponse(data)

    async def compare_runs(request: Request) -> JSONResponse:
        body = await request.json()
        run_ids = body.get("run_ids", [])
        data = api.compare_runs(run_ids)
        return JSONResponse(data)

    async def delete_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        ok = api.delete_run(run_id)
        if ok:
            return JSONResponse({"ok": True})
        return JSONResponse({"error": "delete failed"}, status_code=500)

    async def metric_keys(request: Request) -> JSONResponse:
        keys = api.metric_keys_global()
        return JSONResponse(keys)

    async def metric_summary(request: Request) -> JSONResponse:
        key = request.query_params.get("key", "")
        limit = int(request.query_params.get("limit", "20"))
        data = api.metric_summary(key, limit=limit)
        return JSONResponse(data)

    async def static_file(request: Request) -> Response:
        file_path = static / request.path_params.get("path", "")
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return JSONResponse({"error": "not found"}, status_code=404)

    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        _ws_clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            _ws_clients.discard(websocket)

    routes = [
        Route("/", index),
        Route("/api/runs", list_runs),
        Route("/api/runs/{run_id}", get_run),
        Route("/api/runs/{run_id}/metrics", get_metrics),
        Route("/api/runs/{run_id}", delete_run, methods=["DELETE"]),
        Route("/api/compare", compare_runs, methods=["POST"]),
        Route("/api/metric-keys", metric_keys),
        Route("/api/metric-summary", metric_summary),
        Route("/static/{path:path}", static_file),
        WebSocketRoute("/ws", ws_endpoint),
    ]

    return Starlette(routes=routes)


async def broadcast_ws(message: dict) -> None:
    text = json.dumps(message)
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)
