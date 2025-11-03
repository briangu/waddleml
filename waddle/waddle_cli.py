
#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, json, sqlite3, urllib.request, urllib.parse, shutil, time, threading, selectors, struct, socket
from pathlib import Path
from typing import Optional, Dict, Any, List
from http.server import BaseHTTPRequestHandler, HTTPServer

from waddle import WaddleDB, RepoInfo, execute_commit

# ---------- config ----------
def read_config(path="waddle.json") -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}

def write_json(path: str, obj: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

DEFAULT_CFG = {
  "db": ".waddle/waddle.sqlite",
  "remote": { "name": "default", "url": "", "token": "" },
  "run": { "project": "demo", "entry": "examples.example_train", "repo": "main", "auto_commit": True }
}
GITIGNORE = [".waddle/*.sqlite", ".waddle/*.db", ".waddle/*.wal", ".waddle/*.shm", ".waddle/tmp/", ".waddle/cache/"]

# ---------- init ----------
def cmd_init(a):
    root = Path(a.path or ".").resolve()
    (root/".waddle").mkdir(parents=True, exist_ok=True)
    cfg = root/"waddle.json"
    if not cfg.exists() or a.force:
        write_json(str(cfg), DEFAULT_CFG)
        print(f"wrote {cfg}")
    gi = root/".gitignore"
    txt = gi.read_text(encoding="utf-8") if gi.exists() else ""
    changed = False
    for line in GITIGNORE:
        if line not in txt: txt += ("\n"+line); changed = True
    if changed: gi.write_text(txt+"\n", encoding="utf-8"); print(f"updated {gi}")
    else: print("gitignore ok")
    print("initialized .waddle/")
    return 0

# ---------- repo link ----------
def get_origin(repo_path: str) -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(["git","config","--get","remote.origin.url"], cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return out.stdout.strip() if out.returncode==0 else None
    except Exception:
        return None

def _detect_repo_root(path: str) -> Optional[str]:
    """
    Attempt to discover the git repository root for the provided path.
    Returns an absolute path or None if detection fails.
    """
    from subprocess import run, PIPE
    try:
        proc = run(["git", "rev-parse", "--show-toplevel"], cwd=path, stdout=PIPE, stderr=PIPE, text=True, check=True)
        root = proc.stdout.strip()
        return root or None
    except Exception:
        return None

def _detect_default_branch(repo_path: str) -> str:
    """
    Determine the current branch name to use as the default branch for the repo link.
    Falls back to 'main' when the branch cannot be determined.
    """
    from subprocess import run, PIPE
    try:
        proc = run(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo_path, stdout=PIPE, stderr=PIPE, text=True, check=True)
        branch = proc.stdout.strip()
        return branch or "main"
    except Exception:
        return "main"

def _execute_workspace(repo_path: str, entry: str, argv: List[str], run_obj):
    """
    Execute the provided entry directly from the current working tree without creating a git worktree.
    """
    import importlib
    repo_path = os.path.abspath(repo_path)
    argv = argv or []
    sys.path.insert(0, repo_path)
    try:
        if ":" in entry:
            mod_name, func_name = entry.split(":", 1)
        else:
            mod_name, func_name = entry, None
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name) if func_name else getattr(mod, "waddle_main", None)
        if fn is None or not callable(fn):
            raise RuntimeError(f"No runnable entry found for {entry}. Provide 'module:func' or export 'waddle_main'.")
        try:
            fn(run_obj, argv)
        except TypeError:
            fn(run_obj)
    finally:
        try:
            sys.path.remove(repo_path)
        except ValueError:
            pass

def cmd_repo_link(a):
    cfg = read_config()
    db = WaddleDB(cfg.get("db", ".waddle/waddle.sqlite"))
    origin = get_origin(a.path)
    info = db.upsert_repo(a.name, a.path, origin, a.default_branch or "main")
    print(json.dumps(info.__dict__, indent=2))
    return 0

# ---------- Dashboard server ----------
class MiniWSServer:
    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    def __init__(self, host: str, port: int):
        self.host=host; self.port=port
        self.sel=selectors.DefaultSelector()
        self.srv=None
        self.clients=set()
        self.running=False
    def start(self):
        self.running=True
        self.srv=socket.socket(socket.AF_INET, socket.SOCK_STREAM); self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        self.srv.bind((self.host,self.port)); self.srv.listen(100); self.srv.setblocking(False)
        self.sel.register(self.srv, selectors.EVENT_READ, self._accept)
        threading.Thread(target=self._loop, daemon=True).start()
    def stop(self):
        self.running=False
        try: self.sel.close()
        except Exception: pass
        try: self.srv.close()
        except Exception: pass
        for c in list(self.clients):
            try: c.close()
            except Exception: pass
        self.clients.clear()
    def _loop(self):
        while self.running:
            for key,mask in self.sel.select(timeout=0.5):
                cb=key.data
                try: cb(key.fileobj)
                except Exception:
                    try: self.sel.unregister(key.fileobj)
                    except Exception: pass
                    try: key.fileobj.close()
                    except Exception: pass
    def _accept(self, sock):
        conn,addr=sock.accept(); conn.setblocking(True)
        try:
            req=conn.recv(2048).decode("utf-8","ignore")
            if "Upgrade: websocket" not in req: conn.close(); return
            key=None
            for line in req.split("\r\n"):
                if line.lower().startswith("sec-websocket-key:"):
                    key=line.split(":",1)[1].strip(); break
            if not key: conn.close(); return
            import hashlib, base64
            accept=base64.b64encode(hashlib.sha1((key+self.GUID).encode()).digest()).decode()
            resp="HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "+accept+"\r\n\r\n"
            conn.sendall(resp.encode()); conn.setblocking(False)
            self.sel.register(conn, selectors.EVENT_READ, self._drain)
            self.clients.add(conn)
        except Exception:
            try: conn.close()
            except Exception: pass
    def _drain(self, conn):
        try:
            data=conn.recv(4096)
            if not data: raise RuntimeError("closed")
        except Exception:
            try: self.sel.unregister(conn)
            except Exception: pass
            try: conn.close()
            except Exception: pass
            if conn in self.clients: self.clients.remove(conn)
    def broadcast(self, text: str):
        payload=text.encode("utf-8"); b1=0x81
        l=len(payload)
        if l<=125: header=bytes([b1,l])
        elif l<1<<16: header=bytes([b1,126])+struct.pack("!H",l)
        else: header=bytes([b1,127])+struct.pack("!Q",l)
        frame=header+payload
        dead=[]
        for c in list(self.clients):
            try: c.sendall(frame)
            except Exception: dead.append(c)
        for c in dead:
            try: self.sel.unregister(c)
            except Exception: pass
            try: c.close()
            except Exception: pass
            if c in self.clients: self.clients.remove(c)

class API:
    def __init__(self, db_path: str): self.db_path=os.path.abspath(db_path)
    def _connect(self):
        uri=f"file:{self.db_path}?mode=ro"
        c=sqlite3.connect(uri, uri=True); c.row_factory=sqlite3.Row; return c
    def list_runs(self, limit=200):
        with self._connect() as c:
            cur=c.cursor()
            cur.execute("SELECT id, name, status, started_at, ended_at, commit_sha, entry FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]
    def get_run(self, run_id: str):
        with self._connect() as c:
            cur=c.cursor(); cur.execute("SELECT * FROM runs WHERE id=?", (run_id,)); r=cur.fetchone()
            if not r: return {}
            cur.execute("SELECT key, value FROM params WHERE run_id=?", (run_id,)); params={row["key"]: json.loads(row["value"]) for row in cur.fetchall()}
            cur.execute("SELECT key, value FROM tags WHERE run_id=?", (run_id,)); tags={row["key"]: json.loads(row["value"]) for row in cur.fetchall()}
            cur.execute("SELECT name, kind, created_at, uri, sha256 FROM artifacts WHERE run_id=?", (run_id,)); arts=[dict(row) for row in cur.fetchall()]
            return {"run": dict(r), "params": params, "tags": tags, "artifacts": arts}
    def get_metrics(self, run_id: str, key: Optional[str], limit: int=2000):
        with self._connect() as c:
            cur=c.cursor()
            if key:
                cur.execute("SELECT rowid, key, step, ts, value FROM metrics WHERE run_id=? AND key=? ORDER BY step ASC LIMIT ?", (run_id, key, limit))
            else:
                cur.execute("SELECT rowid, key, step, ts, value FROM metrics WHERE run_id=? ORDER BY key, step ASC LIMIT ?", (run_id, limit))
            return [dict(r) for r in cur.fetchall()]

class Handler(BaseHTTPRequestHandler):
    api: API = None
    static_dir: str = None
    def _ok(self, code=200, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
    def log_message(self, *args, **kwargs): return
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs, unquote
        u=urlparse(self.path); path=u.path; qs=parse_qs(u.query)
        try:
            if path=="/":
                # serve index.html
                p=Path(self.static_dir)/"index.html"
                self._ok(200,"text/html; charset=utf-8"); self.wfile.write(p.read_bytes()); return
            if path.startswith("/static/"):
                file_path = Path(self.static_dir)/path[len("/static/"):]
                if not file_path.exists():
                    self._ok(404); self.wfile.write(b'{"error":"not found"}'); return
                # naive content type
                ctype="text/plain"
                if str(file_path).endswith(".js"): ctype="application/javascript"
                if str(file_path).endswith(".css"): ctype="text/css"
                self._ok(200, ctype); self.wfile.write(file_path.read_bytes()); return
            if path=="/api/runs":
                self._ok(); self.wfile.write(json.dumps(self.api.list_runs()).encode()); return
            if path.startswith("/api/runs/"):
                segs=path.strip("/").split("/")
                run_id=segs[2] if len(segs)>=3 else None
                if segs[-1]=="metrics":
                    key=qs.get("key",[None])[0]; limit=int(qs.get("limit",["2000"])[0])
                    self._ok(); self.wfile.write(json.dumps(self.api.get_metrics(run_id, key, limit)).encode()); return
                else:
                    self._ok(); self.wfile.write(json.dumps(self.api.get_run(run_id)).encode()); return
            self._ok(404); self.wfile.write(b'{"error":"not found"}')
        except Exception as e:
            self._ok(500); self.wfile.write(json.dumps({"error": str(e)}).encode())

def serve_dashboard(db_path: str, host: str, port: int, ws_port: Optional[int], static_dir: str):
    api = API(db_path); Handler.api = api; Handler.static_dir = static_dir
    httpd = HTTPServer((host, port), Handler)
    ws = None
    if ws_port:
        ws = MiniWSServer(host, ws_port); ws.start(); print(f"[waddle] WS at ws://{host}:{ws_port}")
    print(f"[waddle] dashboard at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if ws: ws.stop()
        httpd.server_close()

# ---------- run (with live broadcast) ----------
def cmd_run(a):
    cfg = read_config()
    dbp = cfg.get("db", ".waddle/waddle.sqlite")
    db = WaddleDB(dbp)
    repo_name = a.repo or cfg.get("run",{}).get("repo","main")
    try:
        repo = db.get_repo(repo_name)
    except KeyError:
        repo_hint = getattr(a, "repo_path", None) or cfg.get("run",{}).get("repo_path") or os.getcwd()
        repo_root = _detect_repo_root(repo_hint)
        if not repo_root:
            print(f"unable to locate git repository for '{repo_name}'. run 'waddle repo-link --name {repo_name} --path /path/to/repo' to configure it.", file=sys.stderr)
            return 2
        origin = get_origin(repo_root)
        default_branch = _detect_default_branch(repo_root)
        repo = db.upsert_repo(repo_name, repo_root, origin, default_branch)
        print(f"[waddle] linked repo '{repo_name}' at {repo.path} (default branch {repo.default_branch})")
    workspace_mode = False
    try:
        commit = db.ensure_commit(repo, auto_commit=(a.no_auto_commit is False), message=a.commit_message)
    except RuntimeError as err:
        if a.no_auto_commit:
            raise
        workspace_mode = True
        print(f"[waddle] auto-commit failed ({err}); proceeding with workspace snapshot", file=sys.stderr)
        commit = db.ensure_commit(repo, auto_commit=False)
    entry = a.entry or cfg.get("run",{}).get("entry")
    if not entry:
        print("entry required (module or module:func)", file=sys.stderr); return 2
    entry_args = list(a.entry_argv or [])
    if entry_args and entry_args[0] == "--":
        entry_args = entry_args[1:]

    # Optional servers for live viewing
    ws = None
    if a.ws:
        ws = MiniWSServer(a.host, a.ws); ws.start(); print(f"[waddle] WS at ws://{a.host}:{a.ws}")
    httpd = None
    if a.http:
        static_dir = a.static_dir or str((Path(__file__).parent/"static").resolve())
        threading.Thread(target=serve_dashboard, args=(dbp, a.host, a.http, None, static_dir), daemon=True).start()
        print(f"[waddle] HTTP at http://{a.host}:{a.http}")

    env = {"python": sys.version, "platform": sys.platform, "argv": sys.argv, "repo": repo.__dict__, "commit": commit}
    with db.run(project=a.project or cfg.get("run",{}).get("project","default"),
                repo=repo, commit_sha=commit, entry=entry, name=a.name, env=env, notes=a.notes) as run:
        if workspace_mode:
            run.log_tag("waddle_workspace_snapshot", True)
        # Wrap run to broadcast live metrics
        class BRun:
            def __init__(self, r): self._r=r
            def __getattr__(self, n): return getattr(self._r, n)
            def log_metric(self, key, step, value, ts=None):
                self._r.log_metric(key, step, value, ts)
                if ws:
                    ws.broadcast(json.dumps({"type":"metric","run_id":self._r.run_id,"key":key,"step":step,"value":value,"ts":time.time()}))
            def log_param(self, key, value):
                self._r.log_param(key, value)
                if ws: ws.broadcast(json.dumps({"type":"param","run_id":self._r.run_id,"key":key,"value":value}))
            def log_tag(self, key, value):
                self._r.log_tag(key, value)
                if ws: ws.broadcast(json.dumps({"type":"tag","run_id":self._r.run_id,"key":key,"value":value}))
        try:
            if workspace_mode:
                _execute_workspace(repo.path, entry, entry_args, run_obj=BRun(run))
            else:
                execute_commit(repo.path, commit, entry, argv=entry_args, run_obj=BRun(run))
        except Exception as e:
            run._update_status("failed"); 
            raise
        finally:
            if ws: ws.broadcast(json.dumps({"type":"status","status":"completed","run_id":run.run_id}))

    print(json.dumps({"ok": True, "repo": repo.name, "commit": commit}))
    return 0

# ---------- serve (standalone dashboard) ----------
def cmd_serve(a):
    cfg = read_config(); dbp = cfg.get("db", ".waddle/waddle.sqlite")
    static_dir = a.static_dir or str((Path(__file__).parent/"static").resolve())
    serve_dashboard(dbp, a.host, a.port, a.ws, static_dir)
    return 0

# ---------- push/pull (code + optional DB) ----------
def _remote_url(base: str, path: str, query: Dict[str,str]) -> str:
    q = urllib.parse.urlencode(query); return base.rstrip("/") + path + ("?"+q if q else "")

def cmd_push(a):
    cfg = read_config()
    db_path = a.db or cfg.get("db",".waddle/waddle.sqlite")
    db = WaddleDB(db_path)
    repo = db.get_repo(a.repo or cfg.get("run",{}).get("repo","main"))
    import subprocess
    print("[push] git …"); subprocess.run(["git","push", a.remote or "origin", a.ref or repo.default_branch], cwd=repo.path, check=False)

    remote = cfg.get("remote",{}); 
    if a.url: remote["url"]=a.url
    if a.name: remote["name"]=a.name
    if a.token: remote["token"]=a.token
    if remote.get("url"):
        print("[push] state …")
        url = _remote_url(remote["url"], "/upload_db", {"name": remote.get("name","default")})
        req = urllib.request.Request(url, data=Path(db_path).read_bytes(), method="POST")
        if remote.get("token"): req.add_header("X-Waddle-Token", remote["token"])
        req.add_header("Content-Type","application/octet-stream")
        with urllib.request.urlopen(req) as resp:
            print(resp.read().decode())
    else:
        print("[push] no remote state configured; skipped DB upload")
    return 0

def cmd_pull(a):
    cfg = read_config()
    db_path = a.db or cfg.get("db",".waddle/waddle.sqlite")
    db = WaddleDB(db_path)
    repo = db.get_repo(a.repo or cfg.get("run",{}).get("repo","main"))
    import subprocess
    subprocess.run(["git","pull", a.remote or "origin", a.ref or repo.default_branch], cwd=repo.path, check=False)
    remote = cfg.get("remote",{})
    if a.url: remote["url"]=a.url
    if a.name: remote["name"]=a.name
    if a.token: remote["token"]=a.token
    if remote.get("url"):
        url = _remote_url(remote["url"], "/download_db", {"name": remote.get("name","default")})
        req = urllib.request.Request(url, method="GET")
        if remote.get("token"): req.add_header("X-Waddle-Token", remote["token"])
        try:
            with urllib.request.urlopen(req) as resp:
                if resp.status==200:
                    Path(db_path).write_bytes(resp.read())
                    print("[pull] updated DB from remote")
                else:
                    print("[pull] remote DB not found")
        except Exception as e:
            print(f"[pull] remote error: {e}")
    else:
        print("[pull] no remote state configured; skipped DB download")
    return 0

# ---------- parser ----------
def build():
    p=argparse.ArgumentParser(prog="waddle", description="Waddle: Git-only code + local dashboard")
    sub=p.add_subparsers(dest="cmd", required=True)

    pi=sub.add_parser("init"); pi.add_argument("--path"); pi.add_argument("--force", action="store_true"); pi.set_defaults(func=cmd_init)
    pl=sub.add_parser("repo-link"); pl.add_argument("--name", required=True); pl.add_argument("--path", required=True); pl.add_argument("--default-branch", default="main"); pl.set_defaults(func=cmd_repo_link)

    pr=sub.add_parser("run"); 
    pr.add_argument("--repo"); pr.add_argument("--entry", help="module[:func]"); pr.add_argument("--project"); pr.add_argument("--name"); pr.add_argument("--notes"); 
    pr.add_argument("--no-auto-commit", action="store_true"); pr.add_argument("--commit-message");
    pr.add_argument("--http", type=int, help="dashboard port (serve during run)"); pr.add_argument("--ws", type=int, help="websocket port (broadcast during run)"); pr.add_argument("--host", default="127.0.0.1"); pr.add_argument("--static-dir");
    pr.add_argument("entry_argv", nargs=argparse.REMAINDER); pr.set_defaults(func=cmd_run)

    ps=sub.add_parser("serve"); ps.add_argument("--host", default="127.0.0.1"); ps.add_argument("--port", type=int, default=8080); ps.add_argument("--ws", type=int, help="also start WS for external broadcasters"); ps.add_argument("--static-dir"); ps.set_defaults(func=cmd_serve)

    ppush=sub.add_parser("push"); ppush.add_argument("--db"); ppush.add_argument("--repo"); ppush.add_argument("--remote"); ppush.add_argument("--ref"); ppush.add_argument("--url"); ppush.add_argument("--name"); ppush.add_argument("--token"); ppush.set_defaults(func=cmd_push)
    ppull=sub.add_parser("pull"); ppull.add_argument("--db"); ppull.add_argument("--repo"); ppull.add_argument("--remote"); ppull.add_argument("--ref"); ppull.add_argument("--url"); ppull.add_argument("--name"); ppull.add_argument("--token"); ppull.set_defaults(func=cmd_pull)

    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    args = build().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
