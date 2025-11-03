# Waddle

Lightweight, git-native experiment logging with a local dashboard that you can run on any machine without external services. Waddle captures metrics, parameters, tags, and artifacts for each run, storing everything in a small SQLite database that mirrors the exact commit that produced the results.

## Features
- Git-aware snapshots: auto-commit (or workspace fallback) before every run to guarantee reproducibility.
- Local storage: runs, metrics, and artifacts live in `.waddle/` via SQLite and the filesystem—no cloud dependency.
- Live dashboard: pure HTML/JS frontend served with the `waddle` CLI, plus optional WebSocket streaming for real-time charts.
- Simple Python API: use `run.log_metric`, `run.log_param`, `run.log_tag`, and `run.log_artifact` inside your training scripts.
- Remote sync hooks: `waddle push` / `waddle pull` ship run data to a custom HTTP endpoint when you're ready to share.

## Repository Layout

```
waddleml/
├── waddle/             # Package code and CLI entry point
│   ├── waddle.py       # Core DB + execution helpers
│   ├── waddle_cli.py   # `waddle` console script (init/run/serve/push/pull)
│   └── static/         # Dashboard assets (HTML, JS, CSS)
├── examples/           # Sample jobs to exercise logging
├── tests/              # Unit tests (run with `python -m unittest discover tests`)
├── setup.py            # Package metadata (`entry_points={"console_scripts": ["waddle=..."]}`)
└── README.md           # You are here
```

Runtime artifacts such as `.waddle/waddle.sqlite`, `.waddle/logs/`, and temporary worktrees are generated on demand and should remain untracked.

## Getting Started

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies in editable mode
```bash
pip install -r requirements.txt
pip install -e .
```

Python 3.8–3.12 is supported.

### 3. Initialize Waddle in your project
```bash
waddle init                  # or `python -m waddle.waddle_cli init`
waddle repo-link --name main --path /path/to/your/repo
```

This writes `waddle.json`, ensures `.waddle/` exists, and appends the necessary ignore rules to `.gitignore`.

## Running Jobs

### Log and stream a training script
```bash
waddle run \
  --repo main \
  --entry examples.ml_sample \
  --project demo \
  --name "linear-regression" \
  --http 8080 \
  --ws 8081 \
  -- --epochs 40 --learning-rate 0.03
```

- `--entry` accepts `module` or `module:callable`. If the callable is omitted, Waddle looks for `waddle_main(run, argv)` inside the module.
- `--http` and `--ws` spin up a local dashboard for the lifetime of the run, so a single command handles logging and live visualization. Visit `http://127.0.0.1:8080` while the job is running to inspect metrics and metadata as they stream in.
- Waddle automatically creates a commit before execution. Opt out with `--no-auto-commit` to run against the current workspace snapshot.

Need the dashboard up persistently (for multiple runs or passive viewing)? Launch it once with `waddle serve --host 127.0.0.1 --port 8080 --ws 8081`, then point subsequent `waddle run` invocations at the same `--http/--ws` ports.

### Programmatic instrumentation

Inside your script you receive a `run` object that exposes the logging API:

```python
def waddle_main(run, argv):
    # Log configuration
    run.log_param("epochs", 40)
    run.log_tag("model", "linear_regression")

    # Training loop
    for step, loss in enumerate(train()):
        run.log_metric("loss", step, loss)

    # Persist an artifact
    run.log_artifact("weights.json", "./weights.json", kind="model", inline=True)
```

If you call the module directly (outside of Waddle) you can pass `None` for `run` in your own entry point to keep the function reusable.

## Dashboard and API

The dashboard is a vanilla HTML+Canvas app bundled in `waddle/static/`. When the server is running you get:
- `GET /` – main UI
- `GET /api/runs` – list of runs (`[{id, name, status, started_at, ended_at, commit_sha, entry}]`)
- `GET /api/runs/<id>` – detailed run metadata, params, tags, artifacts
- `GET /api/runs/<id>/metrics?key=<metric>&limit=2000` – paginated metric samples
- `WS ws://<host>:<port>` – optional WebSocket stream for live updates (`type: metric|param|tag|status`)

You can point external tooling at these endpoints for custom dashboards or automation.

## Remote Sync (optional)

Configure a remote collector in `waddle.json` and use:
- `waddle push` – upload the current `.waddle/` database and referenced artifacts to a remote HTTP endpoint.
- `waddle pull` – fetch run history from the remote into your local store.

The default configuration ships with placeholders; populate `remote.url` and `remote.token` if you implement a server.

## Examples

- `python -m examples.ml_sample --epochs 50` – run the toy regression script without logging.
- `waddle run --repo main --entry examples.ml_sample -- --epochs 50` – same script but instrumented, with metrics and artifacts recorded automatically.

Use these as templates for your own training jobs or ingestion pipelines.

## Development Workflow

- Format/type check: keep code PEP 8 compliant and add concise comments where logic is non-obvious.
- Tests: run `python -m unittest discover tests` before submitting changes. Tests rely on temporary directories and leave no files behind.
- Packaging: `setup.py` exposes the `waddle` console script via `entry_points`; ensure this README stays up to date when adding new CLI flags or environment variables.

## Troubleshooting

- **Git auto-commit fails**: Waddle falls back to a workspace snapshot. You can keep the snapshot by checking the run tag `waddle_workspace_snapshot` or enable `--no-auto-commit` to skip commit attempts entirely.
- **Dashboard shows no data**: Verify the run pointed at the correct database (`waddle.json` → `db` path) and that you're serving the same file the run wrote to.
- **Missing repository link**: Re-run `waddle repo-link --name <name> --path <git root>`; each linked repo is stored in the SQLite registry.

## License

Waddle is released under the MIT License. See `LICENSE` for details.
