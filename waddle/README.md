
# Waddle — Local Dashboard + Live Metrics (No external JS deps)

What’s new
- **Local dashboard server** (`waddle serve`) with a **vanilla JS Canvas chart** (no Chart.js, no React).
- **Live metrics relay**: `waddle run --ws PORT` broadcasts JSON events; the dashboard listens and updates charts in real time.
- Still Git‑only for code (thin snapshots). Waddle stores only runs/metrics/artifacts referencing `(repo, commit)`.

Quick start
```bash
# 1) Init and link a repo (if not done)
python waddle_cli.py init
python waddle_cli.py repo-link --name main --path /path/to/repo

# 2) Start dashboard (http on 8080, ws on 8081)
python waddle_cli.py serve --port 8080 --ws 8081

# 3) Run a job and stream live
python waddle_cli.py run --repo main --entry examples.example_train:main --http 8080 --ws 8081 -- --steps 50
# Or run without starting HTTP (if already serving) and only broadcast WS:
python waddle_cli.py run --repo main --entry examples.example_train:main --ws 8081 -- --steps 50

# Open: http://127.0.0.1:8080
```

Endpoints
- `GET /` → dashboard
- `GET /api/runs` → [{id, name, status, started_at, ended_at, commit_sha, entry}]
- `GET /api/runs/<id>` → run + params/tags/artifacts
- `GET /api/runs/<id>/metrics?key=loss&limit=2000` → metrics
- WS (if enabled): JSON lines — `{type:"metric"|"param"|"tag"|"status", ...}`

Notes
- The chart is a **pure Canvas** line plot; trivially extensible for multiple series.
- When you run with `--http`, the CLI spins a dashboard thread (read‑only) so you can watch the current run live.
- You can keep a long‑lived `waddle serve` running, and any `waddle run --ws PORT` will stream to it.
