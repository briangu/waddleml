"""Core WaddleDB tests — repos, commits, basic storage."""

import json
import os
import subprocess
from pathlib import Path

import duckdb
import pytest

from waddle import WaddleDB


def _init_git_repo(base: Path) -> Path:
    repo_path = base / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "waddle@example.com"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.name", "Waddle Tester"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (repo_path / "README.md").write_text("sample repo for waddle tests\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_path


def _new_db(tmp_path: Path) -> WaddleDB:
    db_path = tmp_path / "waddle.duckdb"
    return WaddleDB(str(db_path))


def test_upsert_and_get_repo(tmp_path: Path):
    db = _new_db(tmp_path)
    repo_path = _init_git_repo(tmp_path)
    repo_info = db.upsert_repo("main", str(repo_path), "https://example.com/repo.git", default_branch="dev")
    fetched = db.get_repo("main")
    assert fetched == repo_info
    assert fetched.path == str(repo_path.resolve())
    assert fetched.default_branch == "dev"


def test_record_commit(tmp_path: Path):
    db = _new_db(tmp_path)
    repo_path = _init_git_repo(tmp_path)
    repo = db.upsert_repo("main", str(repo_path), None)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True,
                         stdout=subprocess.PIPE, text=True).stdout.strip()

    db.record_commit(repo.id, sha, str(repo_path))
    row = db.fetchone("SELECT commit_sha, author FROM commits WHERE commit_sha = $1", [sha])
    assert row is not None
    assert row[0] == sha
    assert row[1] == "Waddle Tester"


def test_db_schema_creates_all_tables(tmp_path: Path):
    db = _new_db(tmp_path)
    tables = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")
    table_names = {t[0] for t in tables}
    assert {"repos", "commits", "runs", "params", "tags", "metrics", "artifacts"} <= table_names


def test_runs_allow_null_repo_and_commit(tmp_path: Path):
    """Runs should work without git — repo_id and commit_sha can be NULL."""
    db = _new_db(tmp_path)
    db.execute(
        """INSERT INTO runs (id, project, repo_id, commit_sha, name, status, started_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7)""",
        ["run1", "test", None, None, "no-git-run", "running", 1000.0],
    )
    row = db.fetchone("SELECT id, repo_id, commit_sha FROM runs WHERE id = 'run1'")
    assert row[0] == "run1"
    assert row[1] is None
    assert row[2] is None
