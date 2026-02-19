"""Git helper utilities — detection and snapshot only."""

from __future__ import annotations

import subprocess
import time
from typing import Optional


def sh(repo_path: str, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def is_dirty(repo_path: str) -> bool:
    out = sh(repo_path, "status", "--porcelain=v1").strip()
    return len(out) > 0


def get_origin(repo_path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def detect_repo_root(path: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        root = proc.stdout.strip()
        return root or None
    except Exception:
        return None


def detect_default_branch(repo_path: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        branch = proc.stdout.strip()
        return branch or "main"
    except Exception:
        return "main"


def get_head_sha(repo_path: str) -> Optional[str]:
    try:
        return sh(repo_path, "rev-parse", "HEAD").strip()
    except Exception:
        return None


def auto_snapshot(repo_path: str) -> Optional[str]:
    """Auto-commit dirty tree and return HEAD sha. Returns None on failure."""
    try:
        if is_dirty(repo_path):
            sh(repo_path, "add", "-A")
            sh(repo_path, "commit", "-m", f"waddle: auto snapshot {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return get_head_sha(repo_path)
    except Exception:
        return get_head_sha(repo_path)
