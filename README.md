# WaddleML

A lightweight ML experiment tracker with a local dashboard. Think **local Weights & Biases** — no cloud, no account, no config, no git required. Just `pip install` and start logging.

```python
import waddle

with waddle.init(project="my-project", config={"lr": 0.01, "epochs": 100}):
    for epoch in range(100):
        loss = train_one_epoch()
        waddle.log({"loss": loss, "acc": accuracy})
```

Then view everything:

```bash
waddle serve
# open http://127.0.0.1:8080
```

## Features

- **Wandb-style API** — `waddle.init()`, `waddle.log()`, `waddle.finish()` with auto-incrementing steps, context manager, and atexit handler.
- **Works anywhere** — no git required. Use in Jupyter, Colab, Docker, or plain scripts. If you happen to be in a git repo, waddle auto-captures the commit as a bonus.
- **DuckDB storage** — fast, single-file database in `.waddle/waddle.duckdb`. No server process needed.
- **System metrics** — optional background thread captures CPU, memory, and GPU utilization.
- **Rich dashboard** — Plotly.js charts, sortable run table, per-run tabs, and multi-run comparison.
- **Three-command CLI** — `waddle init`, `waddle ls`, `waddle serve`. That's it.

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Instrument your training script

```python
import waddle

with waddle.init(
    project="cifar10",
    name="resnet-baseline",
    config={"lr": 0.001, "batch_size": 64, "epochs": 50},
    tags={"model": "resnet18"},
):
    for epoch in range(50):
        loss, acc = train_epoch()
        waddle.log({"loss": loss, "acc": acc})

    waddle.log_artifact("model.pt", "checkpoints/best.pt", kind="model")
```

### 3. View results

```bash
waddle ls              # quick look in the terminal
waddle serve           # full dashboard at http://127.0.0.1:8080
```

## Python API

### `waddle.init(...) -> Run`

Start a new run. If inside a git repo, auto-captures the commit. If not, works fine without it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str` | `"default"` | Project name (groups runs) |
| `name` | `str` | `None` | Human-readable run name |
| `config` | `dict` | `None` | Hyperparameters — auto-logged as params |
| `tags` | `dict` | `None` | Categorical labels |
| `system_metrics` | `bool` | `True` | Collect CPU/mem/GPU in background |
| `db_path` | `str` | `None` | Override DuckDB path |

Returns a `Run` that works as a context manager.

### `waddle.log(metrics, step=None)`

Log a dictionary of metrics. Step auto-increments if omitted.

```python
waddle.log({"loss": 0.5, "acc": 0.9})        # step 0
waddle.log({"loss": 0.3, "acc": 0.95})       # step 1
waddle.log({"loss": 0.1}, step=100)           # explicit step
```

### `waddle.log_param(key, value)` / `waddle.log_tag(key, value)`

Log individual parameters or tags after init.

### `waddle.log_artifact(name, path=None, kind="file", inline=False)`

Log an output file. If `path` is given, records its location (and optionally stores its contents in DuckDB when `inline=True`).

### `waddle.finish()`

End the active run. Called automatically with `with waddle.init(...)` or at process exit.

## CLI

```
waddle init [--path PATH]                 # create .waddle/ and .gitignore entry
waddle ls [-n 20] [--db PATH]            # list recent runs in terminal
waddle serve [--host HOST] [--port PORT]  # start dashboard
```

### `waddle ls`

```
$ waddle ls
      ID  Project          Name                  Status     Duration   Commit
-------------------------------------------------------------------------------------
a1b2c3d4  hp-sweep         lr=0.1                completed       0.2s  f3e2d1a0
e5f6a7b8  hp-sweep         lr=0.05               completed       0.2s  f3e2d1a0
c9d0e1f2  quickstart       c9d0e1f2              completed       0.1s
```

Runs without git show no commit — that's fine.

## Examples

Four examples, from minimal to full-featured:

### 1. Quickstart — minimal

```bash
python examples/quickstart.py
```

20 lines. Shows `init`, `log`, `log_param`, `log_tag`.

### 2. Linear Regression — full instrumentation

```bash
python examples/linear_regression.py --epochs 100 --lr 0.03
```

Per-epoch metrics, evaluation, model artifact.

### 3. Hyperparameter Sweep — compare runs

```bash
python examples/hyperparameter_sweep.py
waddle serve  # select runs → "Compare Selected"
```

4 runs with different learning rates. Compare overlaid loss curves and parameter diffs.

### 4. Classification — different model type

```bash
python examples/classification.py --epochs 200
```

Binary classification with a perceptron. Loss, accuracy, learned parameters.

## Dashboard

The dashboard at `http://localhost:8080` provides:

- **Run table** — sortable, filterable, multi-select for comparison
- **Single run view** — tabs: Overview (params/tags/env/git), Metrics (Plotly charts), System (CPU/mem/GPU), Artifacts
- **Run comparison** — overlaid metric traces + parameter diff table
- **Live updates** — WebSocket auto-reconnect

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/runs` | GET | List runs (`?project=`, `?status=`, `?sort=`, `?limit=`) |
| `/api/runs/{id}` | GET | Run detail + params + tags + artifacts + metric keys |
| `/api/runs/{id}/metrics` | GET | Metric time series (`?key=`, `?limit=`) |
| `/api/runs/{id}` | DELETE | Delete a run |
| `/api/compare` | POST | Compare runs (`{"run_ids": [...]}`) |
| `/ws` | WS | Live metric streaming |

## Git Integration (Optional)

When you run `waddle.init()` inside a git repository:
- Auto-commits dirty working tree before the run
- Captures the commit SHA and links it to the run
- Records commit metadata (author, message, tree)
- Shows commit info in the dashboard

When not in a git repo, everything works the same — you just don't get commit tracking.

## System Metrics

When `system_metrics=True` (default), a background thread samples every 5s:

| Metric | Source |
|--------|--------|
| `system/cpu_percent` | psutil |
| `system/memory_percent` | psutil |
| `system/memory_used_gb` | psutil |
| `system/gpu0_util_percent` | pynvml |
| `system/gpu0_memory_used_gb` | pynvml |
| `system/gpu0_temp_c` | pynvml |

Missing deps are silently skipped.

## Dependencies

**Required:** `duckdb`, `starlette`, `uvicorn`

**Optional:** `psutil` (CPU/mem), `pynvml` (GPU)

```bash
pip install -e ".[all]"    # everything
```

## Project Structure

```
waddle/
    __init__.py          # Public API: init, log, finish, ...
    _api.py              # Module-level functions
    _run.py              # Run class
    _state.py            # Global run state
    _db.py               # WaddleDB (DuckDB)
    _schema.py           # DDL
    _git.py              # Git detection (optional)
    _sysmetrics.py       # System monitor thread
    _types.py            # RepoInfo dataclass
    _dashboard_api.py    # Dashboard queries
    _server.py           # Starlette app
    cli.py               # CLI: init, ls, serve
    static/index.html    # Dashboard frontend
```

## License

MIT
