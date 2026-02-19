"""WaddleDB — DuckDB-backed storage for runs, metrics, params, tags, artifacts."""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

import duckdb

from ._schema import SCHEMA_DDL
from ._types import RepoInfo


def _now() -> float:
    return time.time()


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


class WaddleDB:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        _ensure_dir(self.path)
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.path)
        self._init()

    def _init(self) -> None:
        for stmt in SCHEMA_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)

    # ---- thread-safe execute helpers ----

    def execute(self, sql: str, params: Any = None) -> duckdb.DuckDBPyConnection:
        with self._lock:
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)

    def fetchone(self, sql: str, params: Any = None) -> Optional[tuple]:
        with self._lock:
            if params:
                return self._conn.execute(sql, params).fetchone()
            return self._conn.execute(sql).fetchone()

    def fetchall(self, sql: str, params: Any = None) -> List[tuple]:
        with self._lock:
            if params:
                return self._conn.execute(sql, params).fetchall()
            return self._conn.execute(sql).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- repos ----

    def upsert_repo(
        self,
        name: str,
        path: str,
        origin_url: Optional[str],
        default_branch: str = "main",
    ) -> RepoInfo:
        abs_path = os.path.abspath(path)
        row = self.fetchone("SELECT id FROM repos WHERE name = $1", [name])
        if row:
            rid = row[0]
            self.execute(
                """UPDATE repos SET path = $1, origin_url = $2, default_branch = $3
                   WHERE id = $4""",
                [abs_path, origin_url, default_branch, rid],
            )
        else:
            rid = uuid.uuid4().hex
            self.execute(
                """INSERT INTO repos (id, name, path, origin_url, default_branch, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                [rid, name, abs_path, origin_url, default_branch, _now()],
            )
        return RepoInfo(rid, name, abs_path, origin_url, default_branch)

    def get_repo(self, name: str) -> RepoInfo:
        row = self.fetchone("SELECT id, name, path, origin_url, default_branch FROM repos WHERE name = $1", [name])
        if not row:
            raise KeyError(f"repo not found: {name}")
        return RepoInfo(row[0], row[1], row[2], row[3], row[4] or "main")

    # ---- commits ----

    def record_commit(self, repo_id: str, commit_sha: str, repo_path: Optional[str] = None) -> None:
        """Record a commit if not already present. Optionally fetches metadata from git."""
        row = self.fetchone(
            "SELECT 1 FROM commits WHERE repo_id = $1 AND commit_sha = $2",
            [repo_id, commit_sha],
        )
        if row:
            return
        msg = author = tree = None
        when_ts = None
        if repo_path:
            try:
                from ._git import sh
                msg = sh(repo_path, "log", "-1", "--pretty=%s", commit_sha).strip()
                author = sh(repo_path, "log", "-1", "--pretty=%an", commit_sha).strip()
                when = sh(repo_path, "log", "-1", "--pretty=%ct", commit_sha).strip()
                tree = sh(repo_path, "rev-parse", f"{commit_sha}^{{tree}}").strip()
                when_ts = float(when) if when else None
            except Exception:
                pass
        self.execute(
            """INSERT INTO commits (repo_id, commit_sha, tree_sha, author, author_time, message)
               VALUES ($1, $2, $3, $4, $5, $6)
               ON CONFLICT DO NOTHING""",
            [repo_id, commit_sha, tree, author, when_ts, msg],
        )
