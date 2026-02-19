"""Read-only DuckDB queries for the dashboard API."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import duckdb


class DashboardAPI:
    def __init__(self, db_path: str):
        self._db_path = db_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path, read_only=True)

    def list_runs(
        self,
        project: Optional[str] = None,
        status: Optional[str] = None,
        sort: str = "started_at",
        order: str = "desc",
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            sql = "SELECT id, project, name, status, started_at, ended_at, commit_sha, entry FROM runs"
            conditions = []
            params: list = []
            if project:
                conditions.append(f"project = ${len(params) + 1}")
                params.append(project)
            if status:
                conditions.append(f"status = ${len(params) + 1}")
                params.append(status)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            allowed_sort = {"started_at", "ended_at", "name", "status", "project"}
            col = sort if sort in allowed_sort else "started_at"
            direction = "ASC" if order.lower() == "asc" else "DESC"
            sql += f" ORDER BY {col} {direction}"

            sql += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
            params.extend([limit, offset])

            rows = conn.execute(sql, params).fetchall()
            cols = ["id", "project", "name", "status", "started_at", "ended_at", "commit_sha", "entry"]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            conn.close()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, project, name, status, started_at, ended_at, commit_sha, entry, env, config, notes FROM runs WHERE id = $1",
                [run_id],
            ).fetchone()
            if not row:
                return {}
            cols = ["id", "project", "name", "status", "started_at", "ended_at", "commit_sha", "entry", "env", "config", "notes"]
            run = dict(zip(cols, row))

            # parse JSON fields
            for field in ("env", "config"):
                if run[field] and isinstance(run[field], str):
                    try:
                        run[field] = json.loads(run[field])
                    except (json.JSONDecodeError, TypeError):
                        pass

            params = conn.execute("SELECT key, value FROM params WHERE run_id = $1", [run_id]).fetchall()
            param_dict = {}
            for k, v in params:
                try:
                    param_dict[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    param_dict[k] = v

            tags = conn.execute("SELECT key, value FROM tags WHERE run_id = $1", [run_id]).fetchall()
            tag_dict = {}
            for k, v in tags:
                try:
                    tag_dict[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    tag_dict[k] = v

            artifacts = conn.execute(
                "SELECT id, name, kind, created_at, uri, sha256, size_bytes FROM artifacts WHERE run_id = $1",
                [run_id],
            ).fetchall()
            art_cols = ["id", "name", "kind", "created_at", "uri", "sha256", "size_bytes"]
            art_list = [dict(zip(art_cols, a)) for a in artifacts]

            metric_keys = conn.execute(
                "SELECT DISTINCT key FROM metrics WHERE run_id = $1 ORDER BY key",
                [run_id],
            ).fetchall()

            return {
                "run": run,
                "params": param_dict,
                "tags": tag_dict,
                "artifacts": art_list,
                "metric_keys": [r[0] for r in metric_keys],
            }
        finally:
            conn.close()

    def get_metrics(
        self,
        run_id: str,
        key: Optional[str] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            if key:
                rows = conn.execute(
                    "SELECT key, step, ts, value FROM metrics WHERE run_id = $1 AND key = $2 ORDER BY step ASC LIMIT $3",
                    [run_id, key, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, step, ts, value FROM metrics WHERE run_id = $1 ORDER BY key, step ASC LIMIT $2",
                    [run_id, limit],
                ).fetchall()
            cols = ["key", "step", "ts", "value"]
            return [dict(zip(cols, r)) for r in rows]
        finally:
            conn.close()

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        if not run_ids:
            return {"runs": [], "param_diff": {}, "metrics": {}}
        conn = self._connect()
        try:
            placeholders = ", ".join(f"${i + 1}" for i in range(len(run_ids)))

            # run info
            rows = conn.execute(
                f"SELECT id, project, name, status, started_at, ended_at, commit_sha, entry FROM runs WHERE id IN ({placeholders})",
                run_ids,
            ).fetchall()
            cols = ["id", "project", "name", "status", "started_at", "ended_at", "commit_sha", "entry"]
            runs = [dict(zip(cols, r)) for r in rows]

            # param diff
            param_rows = conn.execute(
                f"SELECT run_id, key, value FROM params WHERE run_id IN ({placeholders}) ORDER BY key",
                run_ids,
            ).fetchall()
            all_params: Dict[str, Dict[str, Any]] = {}
            for rid, k, v in param_rows:
                if rid not in all_params:
                    all_params[rid] = {}
                try:
                    all_params[rid][k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    all_params[rid][k] = v

            # metric keys
            metric_keys = conn.execute(
                f"SELECT DISTINCT key FROM metrics WHERE run_id IN ({placeholders}) ORDER BY key",
                run_ids,
            ).fetchall()

            # fetch metrics for each run
            metrics: Dict[str, List[Dict[str, Any]]] = {}
            for mk in metric_keys:
                k = mk[0]
                series_rows = conn.execute(
                    f"SELECT run_id, step, value FROM metrics WHERE key = $1 AND run_id IN ({placeholders}) ORDER BY step",
                    [k] + run_ids,
                ).fetchall()
                metrics[k] = [{"run_id": r[0], "step": r[1], "value": r[2]} for r in series_rows]

            return {
                "runs": runs,
                "params": all_params,
                "metrics": metrics,
            }
        finally:
            conn.close()

    def delete_run(self, run_id: str) -> bool:
        conn = duckdb.connect(self._db_path)
        try:
            conn.execute("DELETE FROM metrics WHERE run_id = $1", [run_id])
            conn.execute("DELETE FROM artifacts WHERE run_id = $1", [run_id])
            conn.execute("DELETE FROM tags WHERE run_id = $1", [run_id])
            conn.execute("DELETE FROM params WHERE run_id = $1", [run_id])
            result = conn.execute("DELETE FROM runs WHERE id = $1", [run_id])
            return True
        except Exception:
            return False
        finally:
            conn.close()
