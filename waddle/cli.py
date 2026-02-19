#!/usr/bin/env python3
"""Waddle CLI: init, serve, ls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

GITIGNORE_LINES = [".waddle/"]


# ---------- init ----------

def cmd_init(a: argparse.Namespace) -> int:
    root = Path(a.path or ".").resolve()
    waddle_dir = root / ".waddle"
    waddle_dir.mkdir(parents=True, exist_ok=True)
    print(f"created {waddle_dir}/")

    gi = root / ".gitignore"
    txt = gi.read_text(encoding="utf-8") if gi.exists() else ""
    changed = False
    for line in GITIGNORE_LINES:
        if line not in txt:
            txt += "\n" + line
            changed = True
    if changed:
        gi.write_text(txt + "\n", encoding="utf-8")
        print(f"updated {gi}")

    print("initialized .waddle/")
    return 0


# ---------- ls ----------

def cmd_ls(a: argparse.Namespace) -> int:
    db_path = _find_db(a.db)
    if not db_path:
        print("no .waddle/waddle.duckdb found", file=sys.stderr)
        return 1

    import duckdb
    conn = duckdb.connect(db_path, read_only=True)
    try:
        limit = a.limit or 20
        sql = "SELECT id, project, name, status, started_at, ended_at, commit_sha FROM runs ORDER BY started_at DESC LIMIT $1"
        rows = conn.execute(sql, [limit]).fetchall()
        if not rows:
            print("no runs found")
            return 0

        # header
        print(f"{'ID':>8}  {'Project':<15} {'Name':<20} {'Status':<10} {'Duration':>10} {'Commit':>8}")
        print("-" * 85)
        for row in rows:
            rid, project, name, status, started, ended, commit = row
            duration = ""
            if started and ended:
                secs = ended - started
                if secs < 60:
                    duration = f"{secs:.1f}s"
                else:
                    duration = f"{secs / 60:.1f}m"
            elif started:
                duration = "running"
            commit_str = (commit or "")[:8]
            print(f"{rid[:8]}  {(project or ''):<15} {(name or ''):<20} {(status or ''):<10} {duration:>10} {commit_str:>8}")
    finally:
        conn.close()
    return 0


# ---------- serve ----------

def cmd_serve(a: argparse.Namespace) -> int:
    db_path = _find_db(a.db)
    if not db_path:
        print("no .waddle/waddle.duckdb found. run a training script with waddle.init() first.", file=sys.stderr)
        return 1

    static_dir = a.static_dir or str((Path(__file__).parent / "static").resolve())

    try:
        from ._server import create_app
        import uvicorn
    except ImportError as e:
        print(f"Dashboard requires starlette and uvicorn: {e}", file=sys.stderr)
        return 1

    print(f"[waddle] serving {db_path}")
    app = create_app(db_path, static_dir)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


# ---------- helpers ----------

def _find_db(explicit: str | None = None) -> str | None:
    """Find the DuckDB file. Checks explicit path, then cwd, then walks up to git root."""
    if explicit and Path(explicit).exists():
        return str(Path(explicit).resolve())

    # check cwd
    local = Path.cwd() / ".waddle" / "waddle.duckdb"
    if local.exists():
        return str(local)

    # walk up to find .waddle/
    p = Path.cwd()
    for _ in range(10):
        candidate = p / ".waddle" / "waddle.duckdb"
        if candidate.exists():
            return str(candidate)
        parent = p.parent
        if parent == p:
            break
        p = parent

    return None


# ---------- parser ----------

def build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="waddle", description="WaddleML: local experiment tracker")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Initialize .waddle/ directory")
    pi.add_argument("--path", help="project root (default: cwd)")
    pi.set_defaults(func=cmd_init)

    pl = sub.add_parser("ls", help="List recent runs")
    pl.add_argument("--db", help="path to waddle.duckdb")
    pl.add_argument("-n", "--limit", type=int, default=20, help="max runs to show")
    pl.set_defaults(func=cmd_ls)

    ps = sub.add_parser("serve", help="Start the dashboard server")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--db", help="path to waddle.duckdb")
    ps.add_argument("--static-dir")
    ps.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    args = build().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
