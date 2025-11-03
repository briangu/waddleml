# Repository Guidelines

## Project Structure & Module Organization
Core package resides in `waddle/`: `waddle/waddle.py` implements the `WaddleLogger`, `waddle/server.py` runs the FastAPI + DuckDB server, and `templates/` plus `static/` hold UI assets. The CLI entry point lives in `scripts/waddle`, and sample jobs are under `examples/` for quick smoke tests. Runtime artifacts (`.waddle/logs`, `.waddle/waddle.db`) are generated during runs and should remain untracked.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` — set up a local Python 3.8–3.12 environment.
- `pip install -r requirements.txt && pip install -e .` — install dependencies and the package in editable mode.
- `python -m uvicorn waddle.server:app --reload --host 127.0.0.1 --port 8000` — start the dev server and UI.
- `python examples/hello.py` — emit sample logs to `.waddle/logs` for manual verification.
- `python scripts/waddle --server-port 8000 --log-root .waddle/logs` — run the CLI wrapper against a custom log directory.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions and modules, and CamelCase for public classes (e.g., `WaddleLogger`). Keep functions cohesive and favor explicit type hints when adding new public surfaces. Update docstrings or README sections whenever you introduce new flags or environment variables.

## Testing Guidelines
Add tests under `tests/` mirroring the package layout (e.g., `tests/test_server.py`). Run `python -m unittest discover tests` locally; if you rely on `pytest`, include it in dev setup instructions. Cover logging, ingestion, and WebSocket flows whenever they change, using temporary directories (`tempfile.TemporaryDirectory`) and DuckDB connections tied to disposable paths. Aim for deterministic, offline-safe tests that leave no files behind.

## Commit & Pull Request Guidelines
Keep commit subjects short and imperative, matching the existing history (`Fix UI data loading logic`, `fix html`) and stay under 72 characters. Bundle related changes together, and include a short body when context matters. Pull requests should summarize intent, list verification steps (tests, example scripts, manual server checks), and link GitHub issues or discussion threads. Attach screenshots or console snippets when UI or CLI output changes.

## Configuration Notes
Default logs collect under `.waddle/logs` and the DuckDB file lives at `.waddle/waddle.db`; adjust with `--log-root` or `--db-root` when deploying. Use `--peer` flags to register upstream servers, and document any new configuration combinations before release.
