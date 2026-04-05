"""Run class with metric batching, context manager, and atexit support."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

from ._db import WaddleDB


class Run:
    def __init__(
        self,
        db: WaddleDB,
        run_id: str,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        repo_id: Optional[str] = None,
        commit_sha: Optional[str] = None,
        system_metrics: bool = True,
    ):
        self._db = db
        self.id = run_id
        self.project = project
        self.name = name or run_id[:8]
        self.commit_sha = commit_sha
        self._step = 0
        self._finished = False
        self._sysmon: Any = None

        # create run record
        env = {
            "python": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "argv": sys.argv,
        }
        config_json = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        env_json = json.dumps(env, ensure_ascii=False, sort_keys=True)
        db.execute(
            """INSERT INTO runs (id, project, repo_id, commit_sha, name, status,
                                 started_at, env, config, notes)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            [run_id, project, repo_id, commit_sha, self.name,
             "running", time.time(), env_json, config_json, None],
        )

        # log config as params
        if config:
            for k, v in config.items():
                self.log_param(k, v)

        # log tags
        if tags:
            for k, v in tags.items():
                self.log_tag(k, v)

        # start system metrics
        if system_metrics:
            self._start_sysmetrics()

        # register atexit
        atexit.register(self._atexit)

    def _start_sysmetrics(self) -> None:
        try:
            from ._sysmetrics import SystemMonitor
            self._sysmon = SystemMonitor(self)
            self._sysmon.start()
        except Exception:
            pass

    def _atexit(self) -> None:
        if not self._finished:
            self.finish(status="aborted")

    def serve_dashboard(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the dashboard web server in a background thread.

        Shares the same DB connection — no file locking issue.
        """
        import threading
        import uvicorn
        from ._dashboard_api import DashboardAPI
        from ._server import create_app

        api = DashboardAPI(db=self._db)
        app = create_app(api=api)

        # Capture the event loop so log() can push WebSocket updates
        self._ws_loop = None

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            self._ws_loop = loop
            config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
            server = uvicorn.Server(config)
            loop.run_until_complete(server.serve())

        threading.Thread(target=_run, daemon=True).start()
        print(f"Dashboard at http://{host}:{port}")

    # ---- logging ----

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step + 1
        ts = time.time()
        for key, value in metrics.items():
            self._db.execute(
                "INSERT INTO metrics (run_id, key, step, ts, value) VALUES ($1, $2, $3, $4, $5)",
                [self.id, key, step, ts, float(value)],
            )
            self._broadcast_metric(key, step, ts, float(value))

    def _broadcast_metric(self, key: str, step: int, ts: float, value: float) -> None:
        """Push metric to WebSocket clients (non-blocking)."""
        loop = getattr(self, '_ws_loop', None)
        if loop is None:
            return
        import asyncio
        from ._server import broadcast_ws
        msg = {"type": "metric", "run_id": self.id, "key": key, "step": step, "ts": ts, "value": value}
        asyncio.run_coroutine_threadsafe(broadcast_ws(msg), loop)

    def log_param(self, key: str, value: Any) -> None:
        self._db.execute(
            """INSERT INTO params (run_id, key, value) VALUES ($1, $2, $3)
               ON CONFLICT (run_id, key) DO UPDATE SET value = EXCLUDED.value""",
            [self.id, key, json.dumps(value, ensure_ascii=False)],
        )

    def log_tag(self, key: str, value: Any) -> None:
        self._db.execute(
            """INSERT INTO tags (run_id, key, value) VALUES ($1, $2, $3)
               ON CONFLICT (run_id, key) DO UPDATE SET value = EXCLUDED.value""",
            [self.id, key, json.dumps(value, ensure_ascii=False)],
        )

    def log_metric(self, key: str, step: int, value: float, ts: Optional[float] = None) -> None:
        if ts is None:
            ts = time.time()
        self._db.execute(
            "INSERT INTO metrics (run_id, key, step, ts, value) VALUES ($1, $2, $3, $4, $5)",
            [self.id, key, step, ts, float(value)],
        )

    def log_artifact(
        self,
        name: str,
        path: Optional[str] = None,
        kind: str = "file",
        inline: bool = False,
    ) -> str:
        aid = uuid.uuid4().hex
        created = time.time()
        uri = None
        blob = None
        sha_hex = None
        size = None
        if path:
            uri = os.path.abspath(path)
            with open(path, "rb") as f:
                data = f.read()
            sha_hex = hashlib.sha256(data).hexdigest()
            size = len(data)
            if inline:
                blob = data
        else:
            sha_hex = hashlib.sha256(b"").hexdigest()
        self._db.execute(
            """INSERT INTO artifacts (id, run_id, name, kind, created_at, uri, sha256, size_bytes, inline_bytes)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
            [aid, self.id, name, kind, created, uri, sha_hex, size, blob],
        )
        return aid

    # ---- lifecycle ----

    def finish(self, status: str = "completed") -> None:
        if self._finished:
            return
        self._finished = True
        if self._sysmon:
            self._sysmon.stop()
        self._db.execute(
            "UPDATE runs SET status = $1, ended_at = $2 WHERE id = $3",
            [status, time.time(), self.id],
        )

    # ---- context manager ----

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.finish(status="failed" if exc else "completed")
        from . import _state
        _state.set_active_run(None)
