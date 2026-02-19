"""Tests for the wandb-style Python API (waddle.init / waddle.log / waddle.finish)."""

import json
import os
import subprocess
from pathlib import Path

import pytest

import waddle
from waddle._db import WaddleDB
from waddle import _state


def _init_git_repo(base: Path) -> Path:
    repo_path = base / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "waddle@example.com"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.name", "Waddle Tester"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (repo_path / "train.py").write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_path


@pytest.fixture(autouse=True)
def _cleanup_state():
    """Reset global state after each test."""
    yield
    _state.set_active_run(None)


def test_init_and_log_with_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)

    run = waddle.init(
        project="test-project",
        name="run-1",
        config={"lr": 0.01, "epochs": 100},
        tags={"model": "resnet"},
        system_metrics=False,
    )
    assert run is not None
    assert run.commit_sha is not None  # git detected
    assert _state.get_active_run() is run

    waddle.log({"loss": 0.5, "acc": 0.9}, step=0)
    waddle.log({"loss": 0.3, "acc": 0.95})  # auto-increments step

    waddle.finish()
    assert _state.get_active_run() is None

    db = run._db
    run_row = db.fetchone("SELECT project, name, status, commit_sha FROM runs WHERE id = $1", [run.id])
    assert run_row[0] == "test-project"
    assert run_row[1] == "run-1"
    assert run_row[2] == "completed"
    assert run_row[3] is not None

    param = db.fetchone("SELECT value FROM params WHERE run_id = $1 AND key = 'lr'", [run.id])
    assert json.loads(param[0]) == 0.01

    tag = db.fetchone("SELECT value FROM tags WHERE run_id = $1 AND key = 'model'", [run.id])
    assert json.loads(tag[0]) == "resnet"

    metrics = db.fetchall("SELECT key, step, value FROM metrics WHERE run_id = $1 ORDER BY step", [run.id])
    assert len(metrics) == 4
    loss_rows = [m for m in metrics if m[0] == "loss"]
    assert loss_rows[0][2] == pytest.approx(0.5)
    assert loss_rows[1][2] == pytest.approx(0.3)


def test_init_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """waddle.init() should work in a directory that is NOT a git repo."""
    no_git_dir = tmp_path / "no_git"
    no_git_dir.mkdir()
    monkeypatch.chdir(no_git_dir)

    run = waddle.init(project="no-git", name="run-nogit", system_metrics=False)
    assert run is not None
    assert run.commit_sha is None  # no git

    waddle.log({"loss": 0.42})
    waddle.finish()

    db = run._db
    run_row = db.fetchone("SELECT project, name, status, repo_id, commit_sha FROM runs WHERE id = $1", [run.id])
    assert run_row[0] == "no-git"
    assert run_row[1] == "run-nogit"
    assert run_row[2] == "completed"
    assert run_row[3] is None
    assert run_row[4] is None

    metric = db.fetchone("SELECT value FROM metrics WHERE run_id = $1 AND key = 'loss'", [run.id])
    assert metric[0] == pytest.approx(0.42)


def test_context_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)

    with waddle.init(project="ctx", system_metrics=False) as run:
        waddle.log({"loss": 1.0})
        run_id = run.id

    assert _state.get_active_run() is None
    db_path = os.path.join(str(repo_path), ".waddle", "waddle.duckdb")
    db = WaddleDB(db_path)
    row = db.fetchone("SELECT status FROM runs WHERE id = $1", [run_id])
    assert row[0] == "completed"


def test_context_manager_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)

    with pytest.raises(ValueError):
        with waddle.init(project="err", system_metrics=False) as run:
            run_id = run.id
            raise ValueError("boom")

    db_path = os.path.join(str(repo_path), ".waddle", "waddle.duckdb")
    db = WaddleDB(db_path)
    row = db.fetchone("SELECT status FROM runs WHERE id = $1", [run_id])
    assert row[0] == "failed"


def test_context_manager_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Context manager works without git too."""
    no_git = tmp_path / "nogit"
    no_git.mkdir()
    monkeypatch.chdir(no_git)

    with waddle.init(project="nogit-ctx", system_metrics=False) as run:
        waddle.log({"x": 1.0})
        run_id = run.id

    assert _state.get_active_run() is None
    db = run._db
    row = db.fetchone("SELECT status FROM runs WHERE id = $1", [run_id])
    assert row[0] == "completed"


def test_log_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)
    artifact_file = repo_path / "model.pt"
    artifact_file.write_bytes(b"model weights")

    run = waddle.init(project="art", system_metrics=False)
    aid = waddle.log_artifact("model.pt", str(artifact_file), kind="model", inline=True)
    waddle.finish()

    db = run._db
    row = db.fetchone("SELECT name, kind, sha256, size_bytes FROM artifacts WHERE id = $1", [aid])
    assert row[0] == "model.pt"
    assert row[1] == "model"
    assert row[3] == len(b"model weights")


def test_log_without_init_raises():
    _state.set_active_run(None)
    with pytest.raises(RuntimeError, match="No active run"):
        waddle.log({"loss": 0.5})


def test_log_param_and_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    no_git = tmp_path / "nogit"
    no_git.mkdir()
    monkeypatch.chdir(no_git)

    run = waddle.init(project="pt", system_metrics=False)
    waddle.log_param("batch_size", 32)
    waddle.log_tag("experiment", "baseline")
    waddle.finish()

    db = run._db
    p = db.fetchone("SELECT value FROM params WHERE run_id = $1 AND key = 'batch_size'", [run.id])
    assert json.loads(p[0]) == 32
    t = db.fetchone("SELECT value FROM tags WHERE run_id = $1 AND key = 'experiment'", [run.id])
    assert json.loads(t[0]) == "baseline"
