# waddle/ — Package Internals

## Module Map

| Module | Purpose |
|--------|---------|
| `__init__.py` | Public API: `init`, `log`, `finish`, `log_artifact`, `log_param`, `log_tag` |
| `_api.py` | Module-level API — manages global active run |
| `_run.py` | `Run` class — metric batching, context manager, atexit |
| `_state.py` | Thread-safe global run state |
| `_db.py` | `WaddleDB` — DuckDB connection + thread-safe queries |
| `_schema.py` | DuckDB DDL |
| `_git.py` | Git detection + auto-snapshot (optional, never required) |
| `_sysmetrics.py` | `SystemMonitor` background thread |
| `_types.py` | `RepoInfo` dataclass |
| `_dashboard_api.py` | Read-only queries for dashboard API |
| `_server.py` | Starlette app + WebSocket |
| `cli.py` | CLI: `init`, `ls`, `serve` |

## Data Flow

```
waddle.init() → _api.py → _db.py (DuckDB) → _run.py (Run object)
                  ↓                              ↓
             _git.py (optional)           _sysmetrics.py
             auto-detect + snapshot       background thread

waddle.log()  → _state.py (active run) → _run.py → DuckDB metrics table
waddle.serve  → cli.py → _server.py → _dashboard_api.py → DuckDB (read-only)
```
