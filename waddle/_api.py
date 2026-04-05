"""Module-level wandb-style API: init, log, finish, log_artifact, log_param, log_tag."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from ._db import WaddleDB
from ._run import Run
from . import _state


def init(
    project: str = "default",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    system_metrics: bool = True,
) -> Run:
    """Initialize a new run.

    Works anywhere. If inside a git repo, automatically captures the commit SHA
    and repo info. If not, the run still works — just without git metadata.
    """
    repo_id: Optional[str] = None
    commit_sha: Optional[str] = None

    # try to detect git repo (optional)
    from ._git import detect_repo_root, get_origin, detect_default_branch, auto_snapshot
    repo_root = detect_repo_root(os.getcwd())

    if repo_root:
        # we're in a git repo — capture info as a bonus
        if db_path is None:
            db_path = os.path.join(repo_root, ".waddle", "waddle.duckdb")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db = WaddleDB(db_path)

        origin = get_origin(repo_root)
        branch = detect_default_branch(repo_root)
        repo_name = os.path.basename(repo_root)
        repo = db.upsert_repo(repo_name, repo_root, origin, branch)
        repo_id = repo.id

        commit_sha = auto_snapshot(repo_root)
        if commit_sha:
            db.record_commit(repo.id, commit_sha, repo_root)
    else:
        # no git — just use a local .waddle/ in cwd
        if db_path is None:
            db_path = os.path.join(os.getcwd(), ".waddle", "waddle.duckdb")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db = WaddleDB(db_path)

    run_id = uuid.uuid4().hex
    run = Run(
        db=db,
        run_id=run_id,
        project=project,
        name=name,
        config=config,
        tags=tags,
        repo_id=repo_id,
        commit_sha=commit_sha,
        system_metrics=system_metrics,
    )
    _state.set_active_run(run)
    return run


def log(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the active run."""
    run = _state.get_active_run()
    if run is None:
        raise RuntimeError("No active run. Call waddle.init() first.")
    run.log(metrics, step=step)


def log_param(key: str, value: Any) -> None:
    run = _state.get_active_run()
    if run is None:
        raise RuntimeError("No active run. Call waddle.init() first.")
    run.log_param(key, value)


def log_tag(key: str, value: Any) -> None:
    run = _state.get_active_run()
    if run is None:
        raise RuntimeError("No active run. Call waddle.init() first.")
    run.log_tag(key, value)


def log_artifact(name: str, path: Optional[str] = None, kind: str = "file", inline: bool = False) -> str:
    run = _state.get_active_run()
    if run is None:
        raise RuntimeError("No active run. Call waddle.init() first.")
    return run.log_artifact(name, path, kind, inline)


def finish() -> None:
    """Finish the active run."""
    run = _state.get_active_run()
    if run is None:
        return
    run.finish()
    _state.set_active_run(None)


def serve_dashboard(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the dashboard server in a background thread, sharing the active run's DB."""
    run = _state.get_active_run()
    if run is None:
        raise RuntimeError("No active run. Call waddle.init() first.")
    run.serve_dashboard(host=host, port=port)
