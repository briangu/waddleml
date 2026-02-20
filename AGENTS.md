# Repository Guidelines

## Project Structure & Module Organization

Core package resides in `waddle/` with a modular architecture:

| Module | Purpose |
|--------|---------|
| `__init__.py` | Public API: `init`, `log`, `finish`, `log_artifact`, `log_param`, `log_tag` |
| `_api.py` | Module-level API — manages global active run |
| `_run.py` | `Run` class — metric batching, context manager, atexit |
| `_state.py` | Thread-safe global run state |
| `_db.py` | `WaddleDB` — DuckDB connection + thread-safe queries |
| `_schema.py` | DuckDB DDL (7 tables: repos, commits, runs, params, tags, metrics, artifacts) |
| `_git.py` | Git detection + auto-snapshot (optional, never required) |
| `_sysmetrics.py` | `SystemMonitor` background thread (CPU/mem/GPU) |
| `_types.py` | `RepoInfo` dataclass |
| `_dashboard_api.py` | Read-only queries for the dashboard API |
| `_server.py` | Starlette app + WebSocket live updates |
| `cli.py` | CLI entry point: `init`, `ls`, `serve` |
| `static/index.html` | Dashboard frontend (vanilla JS + Plotly.js) |

Examples live in `examples/` and tests in `tests/`. Runtime artifacts (`.waddle/waddle.duckdb`) are generated during runs and should remain untracked.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate` — set up a local Python 3.9+ environment.
- `pip install -e ".[all]"` — install the package in editable mode with all optional deps (psutil, pynvml).
- `pytest tests/` — run the test suite.
- `python examples/quickstart.py` — emit sample runs for manual verification.
- `waddle ls` — list recent runs in terminal.
- `waddle serve` — start the dashboard at http://127.0.0.1:8080.

## Coding Style & Naming Conventions

Follow PEP 8 with 4-space indentation, snake_case for functions and modules, CamelCase for public classes (`Run`, `WaddleDB`, `SystemMonitor`). Internal modules use a leading underscore (`_api.py`, `_run.py`, etc.). Keep functions cohesive and favor explicit type hints on public surfaces.

## Testing Guidelines

Tests live under `tests/` (`test_waddle.py`, `test_api.py`, `test_sysmetrics.py`). Run with `pytest tests/`. Use temporary directories and disposable DuckDB paths for isolation. Tests should be deterministic and offline-safe. Mock external dependencies like psutil/pynvml where needed.

## Commit & Pull Request Guidelines

Keep commit subjects short and imperative, under 72 characters. Bundle related changes together. Pull requests should summarize intent, list verification steps, and include screenshots for UI changes.

## Configuration Notes

All data is stored locally in `.waddle/waddle.duckdb` (single DuckDB file, no server process). The `.waddle/` directory is created automatically by `waddle.init()` or `waddle init`. Git integration is optional — waddle works anywhere, with or without a git repo.
