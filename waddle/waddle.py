
from __future__ import annotations
import os, json, sqlite3, time, uuid, subprocess, sys, base64, hashlib, tempfile, shutil
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from pathlib import Path

def _now() -> float: return time.time()

def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

@dataclass
class RepoInfo:
    id: str
    name: str
    path: str
    origin_url: Optional[str]
    default_branch: str

class WaddleDB:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        _ensure_dir(self.path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        c = self._conn.cursor()
        c.execute("PRAGMA foreign_keys=ON;")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS projects(
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS repos(
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            origin_url TEXT,
            default_branch TEXT,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS commits(
            repo_id TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            tree_sha TEXT,
            author TEXT,
            author_time REAL,
            message TEXT,
            PRIMARY KEY(repo_id, commit_sha),
            FOREIGN KEY(repo_id) REFERENCES repos(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            repo_id TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            commit_sha TEXT NOT NULL,
            entry TEXT NOT NULL,
            name TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            started_at REAL NOT NULL,
            ended_at REAL,
            env_json TEXT,
            notes TEXT,
            FOREIGN KEY(repo_id, commit_sha) REFERENCES commits(repo_id, commit_sha) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS params(
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY(run_id, key)
        );
        CREATE TABLE IF NOT EXISTS tags(
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY(run_id, key)
        );
        CREATE TABLE IF NOT EXISTS metrics(
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            ts REAL NOT NULL,
            key TEXT NOT NULL,
            value REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifacts(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            created_at REAL NOT NULL,
            uri TEXT,
            sha256 TEXT,
            inline_bytes BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_run_key_step ON metrics(run_id, key, step);
        """)
        self._conn.commit()

    # ---------------- Projects / Repos ----------------
    def _get_or_create_project(self, name: str) -> str:
        c = self._conn.cursor()
        r = c.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
        if r: return r["id"]
        pid = uuid.uuid4().hex
        c.execute("INSERT INTO projects(id, name, created_at) VALUES(?,?,?)", (pid, name, _now()))
        self._conn.commit()
        return pid

    def upsert_repo(self, name: str, path: str, origin_url: Optional[str], default_branch: str = "main") -> RepoInfo:
        c = self._conn.cursor()
        r = c.execute("SELECT id FROM repos WHERE name=?", (name,)).fetchone()
        rid = r["id"] if r else uuid.uuid4().hex
        c.execute("""INSERT OR REPLACE INTO repos(id, name, path, origin_url, default_branch, created_at)
                     VALUES(?,?,?,?,?,COALESCE((SELECT created_at FROM repos WHERE id=?), ?))""",
                  (rid, name, os.path.abspath(path), origin_url, default_branch, rid, _now()))
        self._conn.commit()
        return RepoInfo(rid, name, os.path.abspath(path), origin_url, default_branch)

    def get_repo(self, name: str) -> RepoInfo:
        c = self._conn.cursor()
        r = c.execute("SELECT * FROM repos WHERE name=?", (name,)).fetchone()
        if not r: raise KeyError(f"repo not found: {name}")
        return RepoInfo(r["id"], r["name"], r["path"], r["origin_url"], r["default_branch"] or "main")

    # ---------------- Git helpers ----------------
    @staticmethod
    def _sh(repo_path: str, *args: str) -> str:
        res = subprocess.run(["git", *args], cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
        return res.stdout

    @staticmethod
    def _dirty(repo_path: str) -> bool:
        out = WaddleDB._sh(repo_path, "status", "--porcelain=v1").strip()
        return len(out) > 0

    def ensure_commit(self, repo: RepoInfo, auto_commit: bool = True, message: Optional[str] = None) -> str:
        if auto_commit and self._dirty(repo.path):
            self._sh(repo.path, "add", "-A")
            msg = message or f"waddle: auto snapshot {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self._sh(repo.path, "commit", "-m", msg)
        sha = self._sh(repo.path, "rev-parse", "HEAD").strip()
        # record commit metadata if missing
        c = self._conn.cursor()
        row = c.execute("SELECT 1 FROM commits WHERE repo_id=? AND commit_sha=?", (repo.id, sha)).fetchone()
        if not row:
            try:
                msg = self._sh(repo.path, "log", "-1", "--pretty=%s", sha).strip()
                author = self._sh(repo.path, "log", "-1", "--pretty=%an", sha).strip()
                when = self._sh(repo.path, "log", "-1", "--pretty=%ct", sha).strip()
                tree = self._sh(repo.path, "rev-parse", f"{sha}^{{tree}}").strip()
                when_ts = float(when) if when else None
            except Exception:
                msg = None; author = None; when_ts = None; tree = None
            c.execute("""INSERT OR IGNORE INTO commits(repo_id, commit_sha, tree_sha, author, author_time, message)
                         VALUES(?,?,?,?,?,?)""", (repo.id, sha, tree, author, when_ts, msg))
            self._conn.commit()
        return sha

    # ---------------- Run context ----------------
    class Run:
        def __init__(self, conn: sqlite3.Connection, run_id: str):
            self._conn = conn; self.run_id = run_id
        def _update_status(self, status: str):
            c = self._conn.cursor()
            if status in ("completed","failed","aborted"):
                c.execute("UPDATE runs SET status=?, ended_at=? WHERE id=?", (status, time.time(), self.run_id))
            else:
                c.execute("UPDATE runs SET status=? WHERE id=?", (status, self.run_id))
            self._conn.commit()
        # logging
        def log_param(self, key: str, value: Any):
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO params(run_id, key, value) VALUES(?,?,?)",
                      (self.run_id, key, json.dumps(value, ensure_ascii=False)))
            self._conn.commit()
        def log_tag(self, key: str, value: Any):
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO tags(run_id, key, value) VALUES(?,?,?)",
                      (self.run_id, key, json.dumps(value, ensure_ascii=False)))
            self._conn.commit()
        def log_metric(self, key: str, step: int, value: float, ts: Optional[float] = None):
            if ts is None: ts = time.time()
            c = self._conn.cursor()
            c.execute("INSERT INTO metrics(run_id, step, ts, key, value) VALUES(?,?,?,?,?)",
                      (self.run_id, step, ts, key, float(value)))
            self._conn.commit()
        def log_artifact(self, name: str, path: Optional[str] = None, kind: str = "file", inline: bool = False):
            c = self._conn.cursor()
            aid = uuid.uuid4().hex; created = time.time()
            uri=None; blob=None; sha=None
            if path:
                uri = os.path.abspath(path)
                with open(path,"rb") as f: sha = hashlib.sha256(f.read()).hexdigest()
                if inline:
                    with open(path,"rb") as f: blob = f.read()
            else:
                sha = hashlib.sha256(b"").hexdigest()
            c.execute("""INSERT INTO artifacts(id, run_id, name, kind, created_at, uri, sha256, inline_bytes)
                         VALUES(?,?,?,?,?,?,?,?)""",
                      (aid, self.run_id, name, kind, created, uri, sha, blob))
            self._conn.commit(); return aid

    def run(self, project: str, repo: RepoInfo, commit_sha: str, entry: str,
            name: Optional[str] = None, env: Optional[Dict[str, Any]] = None, notes: Optional[str] = None):
        c = self._conn.cursor()
        r = c.execute("SELECT 1 FROM commits WHERE repo_id=? AND commit_sha=?", (repo.id, commit_sha)).fetchone()
        if not r:
            c.execute("INSERT OR IGNORE INTO commits(repo_id, commit_sha) VALUES(?,?)", (repo.id, commit_sha))
        pid = self._get_or_create_project(project)
        rid = uuid.uuid4().hex
        c.execute("""INSERT INTO runs(id, project_id, repo_id, commit_sha, entry, name, status, started_at, env_json, notes)
                     VALUES(?,?,?,?,?,?,?,?,?,?)""",
                  (rid, pid, repo.id, commit_sha, entry, name, "running", _now(), json.dumps(env or {}, ensure_ascii=False, sort_keys=True), notes))
        self._conn.commit()
        run = WaddleDB.Run(self._conn, rid)
        class _Ctx:
            def __enter__(self): return run
            def __exit__(self, exc_type, exc, tb):
                run._update_status("failed" if exc else "completed")
        return _Ctx()

def execute_commit(repo_path: str, commit_sha: str, entry: str, argv: Optional[List[str]] = None, run_obj: Optional[WaddleDB.Run] = None):
    import importlib, types
    tmp = tempfile.mkdtemp(prefix="waddle_wt_")
    try:
        subprocess.run(["git", "worktree", "add", "--detach", tmp, commit_sha], cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.path.insert(0, tmp)
        if ":" in entry: mod_name, func_name = entry.split(":",1)
        else: mod_name, func_name = entry, None
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name) if func_name else getattr(mod, "waddle_main", None)
        if fn is None or not callable(fn):
            raise RuntimeError(f"No runnable entry found for {entry}. Provide 'module:func' or export 'waddle_main'.")
        try:
            fn(run_obj, argv or [])
        except TypeError:
            fn(run_obj)
    finally:
        try:
            sys.path.remove(tmp)
        except Exception:
            pass
        try:
            subprocess.run(["git", "worktree", "remove", "--force", tmp], cwd=repo_path, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
