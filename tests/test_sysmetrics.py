"""Tests for system metrics collection."""

import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import waddle
from waddle import _state
from waddle._sysmetrics import SystemMonitor


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
    yield
    _state.set_active_run(None)


def test_sysmetrics_with_mock_psutil(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that system metrics are collected with mocked psutil."""
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)

    run = waddle.init(project="sysmetrics-test", system_metrics=False)

    # Create a monitor with mocked psutil
    mock_psutil = MagicMock()
    mock_psutil.cpu_percent.return_value = 42.5
    mock_mem = MagicMock()
    mock_mem.percent = 65.0
    mock_mem.used = 8 * (1024 ** 3)  # 8 GB
    mock_psutil.virtual_memory.return_value = mock_mem

    monitor = SystemMonitor(run, interval=0.1)
    monitor._has_psutil = True

    with patch.dict("sys.modules", {"psutil": mock_psutil}):
        monitor.start()
        time.sleep(0.3)
        monitor.stop()

    # Check system metrics were written
    rows = run._db.fetchall(
        "SELECT key, value FROM metrics WHERE run_id = $1 AND key LIKE 'system/%' ORDER BY key",
        [run.id],
    )
    keys = {r[0] for r in rows}
    assert "system/cpu_percent" in keys
    assert "system/memory_percent" in keys
    assert "system/memory_used_gb" in keys

    waddle.finish()


def test_sysmetrics_graceful_without_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that system metrics gracefully degrade without psutil/pynvml."""
    repo_path = _init_git_repo(tmp_path)
    monkeypatch.chdir(repo_path)

    run = waddle.init(project="no-deps", system_metrics=False)

    monitor = SystemMonitor(run, interval=0.1)
    monitor._has_psutil = False
    monitor._has_pynvml = False

    # start() should return immediately when no capabilities
    monitor.start()
    assert monitor._thread is None
    monitor.stop()

    waddle.finish()
