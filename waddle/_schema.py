"""DuckDB DDL for the Waddle schema."""

SCHEMA_DDL = """\
CREATE TABLE IF NOT EXISTS repos (
    id VARCHAR PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    path VARCHAR NOT NULL,
    origin_url VARCHAR,
    default_branch VARCHAR DEFAULT 'main',
    created_at DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS commits (
    repo_id VARCHAR NOT NULL REFERENCES repos(id),
    commit_sha VARCHAR NOT NULL,
    tree_sha VARCHAR,
    author VARCHAR,
    author_time DOUBLE,
    message VARCHAR,
    PRIMARY KEY (repo_id, commit_sha)
);

CREATE TABLE IF NOT EXISTS runs (
    id VARCHAR PRIMARY KEY,
    project VARCHAR NOT NULL DEFAULT 'default',
    repo_id VARCHAR,
    commit_sha VARCHAR,
    name VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'running',
    started_at DOUBLE NOT NULL,
    ended_at DOUBLE,
    env JSON,
    config JSON,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS params (
    run_id VARCHAR NOT NULL REFERENCES runs(id),
    key VARCHAR NOT NULL,
    value JSON NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS tags (
    run_id VARCHAR NOT NULL REFERENCES runs(id),
    key VARCHAR NOT NULL,
    value JSON NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id VARCHAR NOT NULL,
    key VARCHAR NOT NULL,
    step INTEGER NOT NULL,
    ts DOUBLE NOT NULL,
    value DOUBLE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key, step);

CREATE TABLE IF NOT EXISTS artifacts (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs(id),
    name VARCHAR NOT NULL,
    kind VARCHAR NOT NULL DEFAULT 'file',
    created_at DOUBLE NOT NULL,
    uri VARCHAR,
    sha256 VARCHAR,
    size_bytes BIGINT,
    inline_bytes BLOB
);
"""
