"""SQLite storage for reports. One table; findings and frame names are JSON text."""

import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "saans.db"
UPLOADS_DIR = DATA_DIR / "uploads"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    media_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    findings TEXT NOT NULL,
    frames TEXT NOT NULL
)
"""


def load_rules():
    return json.loads((DATA_DIR / "grap_rules.json").read_text())


def connect():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def _to_dict(row):
    report = dict(row)
    report["findings"] = json.loads(report["findings"])
    report["frames"] = json.loads(report["frames"])
    return report


def save_report(report):
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reports VALUES "
            "(:id, :created_at, :lat, :lng, :media_type, :summary, :findings, :frames)",
            {**report, "findings": json.dumps(report["findings"]), "frames": json.dumps(report["frames"])},
        )


def list_reports():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM reports ORDER BY created_at DESC").fetchall()
    return [_to_dict(r) for r in rows]


def get_report(report_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return _to_dict(row) if row else None
