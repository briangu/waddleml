import json
import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from waddle import WaddleDB, execute_commit
import waddle.waddle_cli as waddle_cli


def _init_git_repo(base: Path) -> Path:
    repo_path = base / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "waddle@example.com"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.name", "Waddle Tester"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runner = """
def waddle_main(run, argv):
    if run is not None:
        run.log_param("argv_len", len(argv))
        run.log_metric("argv_len_metric", 0, float(len(argv)))
        run.log_tag("executed", True)
"""
    (repo_path / "runner.py").write_text(runner.lstrip(), encoding="utf-8")
    (repo_path / "README.md").write_text("sample repo for waddle tests\n", encoding="utf-8")
    subprocess.run(["git", "add", "runner.py", "README.md"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_path


def _new_db(tmp_path: Path) -> WaddleDB:
    db_path = tmp_path / "waddle.sqlite"
    return WaddleDB(str(db_path))


def _connect_rows(db: WaddleDB):
    conn = sqlite3.connect(db.path)
    conn.row_factory = sqlite3.Row
    return conn


def test_upsert_and_get_repo(tmp_path: Path):
    db = _new_db(tmp_path)
    repo_path = _init_git_repo(tmp_path)
    repo_info = db.upsert_repo("main", str(repo_path), "https://example.com/repo.git", default_branch="dev")
    fetched = db.get_repo("main")
    assert fetched == repo_info
    assert fetched.path == str(repo_path.resolve())
    assert fetched.default_branch == "dev"


def test_ensure_commit_records_and_auto_commits(tmp_path: Path):
    db = _new_db(tmp_path)
    repo_path = _init_git_repo(tmp_path)
    repo = db.upsert_repo("main", str(repo_path), None)
    first_sha = db.ensure_commit(repo)
    runner_file = repo_path / "runner.py"
    runner_file.write_text(runner_file.read_text(encoding="utf-8") + "\n# change\n", encoding="utf-8")
    new_sha = db.ensure_commit(repo, auto_commit=True)
    assert new_sha != first_sha
    log = subprocess.run(["git", "log", "-1", "--pretty=%s"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert log.stdout.strip().startswith("waddle: auto snapshot")
    with _connect_rows(db) as conn:
        row = conn.execute("SELECT commit_sha FROM commits WHERE commit_sha=?", (new_sha,)).fetchone()
        assert row is not None


def test_run_context_persists_logs(tmp_path: Path):
    db = _new_db(tmp_path)
    repo_path = _init_git_repo(tmp_path)
    repo = db.upsert_repo("main", str(repo_path), None)
    sha = db.ensure_commit(repo)
    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("artifact body", encoding="utf-8")
    with db.run("project", repo, sha, "runner:waddle_main", name="test-run", env={"env": "value"}, notes="note") as run:
        run.log_param("learning_rate", 0.1)
        run.log_tag("stage", "dev")
        run.log_metric("loss", 1, 0.5)
        run.log_artifact("art", str(artifact_file), inline=True)
    with _connect_rows(db) as conn:
        run_row = conn.execute("SELECT status, env_json, notes FROM runs").fetchone()
        assert run_row["status"] == "completed"
        assert json.loads(run_row["env_json"]) == {"env": "value"}
        assert run_row["notes"] == "note"
        param = conn.execute("SELECT value FROM params WHERE key='learning_rate'").fetchone()
        assert json.loads(param["value"]) == pytest.approx(0.1)
        tag = conn.execute("SELECT value FROM tags WHERE key='stage'").fetchone()
        assert json.loads(tag["value"]) == "dev"
        metric = conn.execute("SELECT key, value FROM metrics WHERE key='loss'").fetchone()
        assert metric["value"] == pytest.approx(0.5)
        artifact = conn.execute("SELECT name, inline_bytes FROM artifacts").fetchone()
        assert artifact["name"] == "art"
        assert artifact["inline_bytes"] is not None


def test_execute_commit_runs_entry(tmp_path: Path):
    db = _new_db(tmp_path)
    repo_path = _init_git_repo(tmp_path)
    repo = db.upsert_repo("main", str(repo_path), None)
    sha = db.ensure_commit(repo)
    with db.run("project", repo, sha, "runner:waddle_main") as run:
        execute_commit(str(repo_path), sha, "runner:waddle_main", ["hello"], run)
    with _connect_rows(db) as conn:
        params = conn.execute("SELECT key, value FROM params WHERE key='argv_len'").fetchone()
        assert params is not None
        assert json.loads(params["value"]) == 1
        metric = conn.execute("SELECT key, value FROM metrics WHERE key='argv_len_metric'").fetchone()
        assert metric is not None
        assert metric["value"] == pytest.approx(1.0)
        tag = conn.execute("SELECT key, value FROM tags WHERE key='executed'").fetchone()
        assert json.loads(tag["value"]) is True


def test_cmd_run_auto_links_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = _init_git_repo(tmp_path)
    examples_dir = repo_path / "examples"
    examples_dir.mkdir()
    sample_src = Path(__file__).parent / "examples" / "ml_sample.py"
    examples_dir.joinpath("ml_sample.py").write_text(sample_src.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["git", "add", "examples/ml_sample.py"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "commit", "-m", "add ml sample"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    monkeypatch.chdir(repo_path)
    args = SimpleNamespace(
        repo="main",
        entry="examples.ml_sample",
        project=None,
        name=None,
        notes=None,
        no_auto_commit=False,
        commit_message=None,
        http=None,
        ws=None,
        host="127.0.0.1",
        static_dir=None,
        entry_argv=["--epochs", "1"],
    )

    rc = waddle_cli.cmd_run(args)
    assert rc == 0

    db_path = repo_path / ".waddle" / "waddle.sqlite"
    assert db_path.exists()
    db = WaddleDB(str(db_path))
    linked = db.get_repo("main")
    assert linked.path == str(repo_path.resolve())


def test_cmd_run_workspace_snapshot_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_path = _init_git_repo(tmp_path)
    examples_dir = repo_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "__init__.py").write_text("", encoding="utf-8")
    sample_src = Path(__file__).parent / "examples" / "ml_sample.py"
    examples_dir.joinpath("ml_sample.py").write_text(sample_src.read_text(encoding="utf-8"), encoding="utf-8")
    # Make the git directory read-only so auto-commit cannot create index.lock.
    git_dir = repo_path / ".git"
    original_mode = git_dir.stat().st_mode & 0o777
    os.chmod(git_dir, 0o555)

    monkeypatch.chdir(repo_path)
    args = SimpleNamespace(
        repo="main",
        entry="examples.ml_sample",
        project=None,
        name=None,
        notes=None,
        no_auto_commit=False,
        commit_message=None,
        http=None,
        ws=None,
        host="127.0.0.1",
        static_dir=None,
        entry_argv=["--epochs", "1"],
    )

    try:
        rc = waddle_cli.cmd_run(args)
    finally:
        os.chmod(git_dir, original_mode)
    assert rc == 0

    db_path = repo_path / ".waddle" / "waddle.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT value FROM tags WHERE key='waddle_workspace_snapshot'").fetchone()
    conn.close()
    assert row is not None
    assert json.loads(row["value"]) is True
