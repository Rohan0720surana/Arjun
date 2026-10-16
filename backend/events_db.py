"""
SQLite: incidents, zones, settings, counts and the audit trail.

One file, ``data/arjun.db``, no server to install and nothing to configure —
which matters more than it sounds for a system meant to run on a spare machine
in a security office.

What the schema deliberately does *not* contain is as important as what it
does. There is no table of people, no identity, no descriptor, no face, no
name. An event records a camera, a time, a rule, a zone and a track id that
was invented seconds earlier and is meaningless by the next morning. If this
database leaked in full, what the reader would learn is that somebody fell
over near the library at 18:42 — not who.

The ``audit`` table is the other half of the privacy claim: every time a human
unlocks a raw view or opens an incident clip, that is itself an event, written
here, and shown on the dashboard's privacy panel. The watchers are watched.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import config

_LOCK = threading.RLock()
_CONN: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    camera        TEXT    NOT NULL,
    type          TEXT    NOT NULL,
    severity      TEXT    NOT NULL DEFAULT 'info',
    message       TEXT    NOT NULL DEFAULT '',
    zone_id       INTEGER,
    zone_name     TEXT,
    track_id      INTEGER,
    meta          TEXT    NOT NULL DEFAULT '{}',
    clip_path     TEXT,
    clip_state    TEXT    NOT NULL DEFAULT 'none',
    snapshot_path TEXT,
    acknowledged  INTEGER NOT NULL DEFAULT 0,
    simulated     INTEGER NOT NULL DEFAULT 0,
    notified      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_cam ON events(camera, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, ts DESC);

CREATE TABLE IF NOT EXISTS zones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera      TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    points_json TEXT    NOT NULL,
    params_json TEXT    NOT NULL DEFAULT '{}',
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_zones_cam ON zones(camera);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    action TEXT NOT NULL,
    camera TEXT NOT NULL DEFAULT '',
    actor  TEXT NOT NULL DEFAULT 'staff',
    detail TEXT NOT NULL DEFAULT '',
    ok     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts DESC);

CREATE TABLE IF NOT EXISTS counts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    camera    TEXT NOT NULL,
    zone_id   INTEGER,
    zone_name TEXT NOT NULL DEFAULT '',
    count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_counts_ts ON counts(ts DESC);

CREATE TABLE IF NOT EXISTS occupancy (
    camera    TEXT NOT NULL,
    zone_id   INTEGER NOT NULL,
    in_count  INTEGER NOT NULL DEFAULT 0,
    out_count INTEGER NOT NULL DEFAULT 0,
    updated   REAL NOT NULL,
    PRIMARY KEY (camera, zone_id)
);
"""


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #

def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (once) the shared connection.

    One connection guarded by a lock, rather than one per thread: the write
    volume here is a handful of rows a minute, and a single connection makes
    "did that event actually land" trivially answerable.
    """
    global _CONN
    with _LOCK:
        if _CONN is not None:
            return _CONN
        target = Path(path) if path else config.DB_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target), check_same_thread=False,
                               timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        conn.commit()
        _CONN = conn
        return conn


def init(path: Path | None = None) -> sqlite3.Connection:
    return connect(path)


def close() -> None:
    global _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.commit()
            _CONN.close()
            _CONN = None


def _exec(sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
    conn = connect()
    with _LOCK:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def _query(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    conn = connect()
    with _LOCK:
        return conn.execute(sql, params).fetchall()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    if "meta" in d and isinstance(d["meta"], str):
        try:
            d["meta"] = json.loads(d["meta"])
        except json.JSONDecodeError:
            d["meta"] = {}
    if "ts" in d:
        d["iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(d["ts"]))
        d["clock"] = time.strftime("%H:%M:%S", time.localtime(d["ts"]))
    return d


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

def add_event(
    camera: str,
    etype: str,
    severity: str,
    message: str,
    ts: float | None = None,
    zone_id: int | None = None,
    zone_name: str | None = None,
    track_id: int | None = None,
    meta: dict[str, Any] | None = None,
    simulated: bool = False,
    clip_state: str = "none",
) -> int:
    cur = _exec(
        """INSERT INTO events (ts, camera, type, severity, message, zone_id,
                               zone_name, track_id, meta, simulated, clip_state)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (ts if ts is not None else time.time(), camera, etype, severity,
         message, zone_id, zone_name, track_id,
         json.dumps(meta or {}, default=str), 1 if simulated else 0, clip_state),
    )
    return int(cur.lastrowid or 0)


def set_event_clip(event_id: int, clip_path: str | None,
                   state: str = "ready") -> None:
    _exec("UPDATE events SET clip_path = ?, clip_state = ? WHERE id = ?",
          (clip_path, state, event_id))


def set_event_snapshot(event_id: int, snapshot_path: str) -> None:
    _exec("UPDATE events SET snapshot_path = ? WHERE id = ?",
          (snapshot_path, event_id))


def set_event_notified(event_id: int, channels: Iterable[str]) -> None:
    _exec("UPDATE events SET notified = ? WHERE id = ?",
          (",".join(sorted(set(channels))), event_id))


def acknowledge(event_id: int, value: bool = True) -> None:
    _exec("UPDATE events SET acknowledged = ? WHERE id = ?",
          (1 if value else 0, event_id))


def get_event(event_id: int) -> dict[str, Any] | None:
    rows = _query("SELECT * FROM events WHERE id = ?", (event_id,))
    return row_to_dict(rows[0]) if rows else None


def list_events(
    limit: int = 120,
    camera: str | None = None,
    etype: str | None = None,
    severity: str | None = None,
    since: float | None = None,
    until: float | None = None,
    with_clip: bool | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM events WHERE 1=1"
    params: list[Any] = []
    if camera:
        sql += " AND camera = ?"
        params.append(camera)
    if etype:
        sql += " AND type = ?"
        params.append(etype)
    if severity:
        sql += " AND severity = ?"
        params.append(severity)
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    if until is not None:
        sql += " AND ts <= ?"
        params.append(until)
    if with_clip is True:
        sql += " AND clip_path IS NOT NULL"
    if search:
        sql += " AND (message LIKE ? OR type LIKE ? OR zone_name LIKE ?)"
        like = f"%{search}%"
        params += [like, like, like]
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(int(max(1, min(limit, 2000))))
    return [d for d in (row_to_dict(r) for r in _query(sql, params)) if d]


def related_events(event_id: int, window_s: float = 180.0,
                   limit: int = 40) -> list[dict[str, Any]]:
    """Everything on the same camera around the same time.

    This is what turns a single alert into an incident report: the fall at
    18:42 is far more legible beside the crowd warning that preceded it.
    """
    base = get_event(event_id)
    if base is None:
        return []
    return list_events(limit=limit, camera=base["camera"],
                       since=base["ts"] - window_s,
                       until=base["ts"] + window_s)


def counts_by_type(since: float | None = None) -> list[dict[str, Any]]:
    sql = "SELECT type, COUNT(*) AS n FROM events"
    params: list[Any] = []
    if since is not None:
        sql += " WHERE ts >= ?"
        params.append(since)
    sql += " GROUP BY type ORDER BY n DESC"
    return [dict(r) for r in _query(sql, params)]


def counts_by_hour(since: float, camera: str | None = None) -> list[dict[str, Any]]:
    sql = ("SELECT CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER)"
           " AS hour, COUNT(*) AS n FROM events WHERE ts >= ?")
    params: list[Any] = [since]
    if camera:
        sql += " AND camera = ?"
        params.append(camera)
    sql += " GROUP BY hour ORDER BY hour"
    return [dict(r) for r in _query(sql, params)]


def today_start() -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0,
                        lt.tm_wday, lt.tm_yday, lt.tm_isdst))


def summary(since: float | None = None) -> dict[str, Any]:
    start = today_start() if since is None else since
    rows = _query(
        """SELECT severity, COUNT(*) AS n FROM events
           WHERE ts >= ? GROUP BY severity""", (start,))
    by_sev = {r["severity"]: r["n"] for r in rows}
    total = sum(by_sev.values())
    unack = _query("SELECT COUNT(*) AS n FROM events WHERE ts >= ? AND "
                   "acknowledged = 0", (start,))[0]["n"]
    clips = _query("SELECT COUNT(*) AS n FROM events WHERE clip_path IS NOT NULL"
                   )[0]["n"]
    return {
        "since": start,
        "total": total,
        "critical": by_sev.get("critical", 0),
        "warning": by_sev.get("warning", 0),
        "info": by_sev.get("info", 0),
        "unacknowledged": unack,
        "clips": clips,
        "by_type": counts_by_type(start),
    }


def purge_events(before: float) -> int:
    cur = _exec("DELETE FROM events WHERE ts < ?", (before,))
    return cur.rowcount or 0


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #

def add_zone(camera: str, name: str, ztype: str, points: list[list[float]],
             params: dict[str, Any] | None = None) -> int:
    cur = _exec(
        """INSERT INTO zones (camera, name, type, points_json, params_json,
                              enabled, created_at)
           VALUES (?,?,?,?,?,1,?)""",
        (camera, name, ztype, json.dumps(points),
         json.dumps(params or {}), time.time()),
    )
    return int(cur.lastrowid or 0)


def update_zone(zone_id: int, **fields: Any) -> bool:
    allowed = {"name", "type", "enabled"}
    sets, params = [], []
    for key, value in fields.items():
        if key == "points":
            sets.append("points_json = ?")
            params.append(json.dumps(value))
        elif key == "params":
            sets.append("params_json = ?")
            params.append(json.dumps(value))
        elif key in allowed:
            sets.append(f"{key} = ?")
            params.append(int(value) if key == "enabled" else value)
    if not sets:
        return False
    params.append(zone_id)
    cur = _exec(f"UPDATE zones SET {', '.join(sets)} WHERE id = ?", params)
    return (cur.rowcount or 0) > 0


def delete_zone(zone_id: int) -> bool:
    cur = _exec("DELETE FROM zones WHERE id = ?", (zone_id,))
    _exec("DELETE FROM occupancy WHERE zone_id = ?", (zone_id,))
    return (cur.rowcount or 0) > 0


def zone_rows(camera: str | None = None) -> list[sqlite3.Row]:
    if camera is None:
        return _query("SELECT * FROM zones ORDER BY camera, id")
    return _query("SELECT * FROM zones WHERE camera = ? ORDER BY id", (camera,))


def zone_count(camera: str | None = None) -> int:
    if camera is None:
        return _query("SELECT COUNT(*) AS n FROM zones")[0]["n"]
    return _query("SELECT COUNT(*) AS n FROM zones WHERE camera = ?",
                  (camera,))[0]["n"]


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

def set_setting(key: str, value: Any) -> None:
    _exec("INSERT INTO settings (key, value) VALUES (?,?) "
          "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
          (key, json.dumps(value)))


def get_setting(key: str, default: Any = None) -> Any:
    rows = _query("SELECT value FROM settings WHERE key = ?", (key,))
    if not rows:
        return default
    try:
        return json.loads(rows[0]["value"])
    except json.JSONDecodeError:
        return default


def all_settings() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in _query("SELECT key, value FROM settings"):
        try:
            out[row["key"]] = json.loads(row["value"])
        except json.JSONDecodeError:
            continue
    return out


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

def add_audit(action: str, camera: str = "", detail: str = "",
              actor: str = "staff", ok: bool = True) -> int:
    cur = _exec(
        "INSERT INTO audit (ts, action, camera, actor, detail, ok) "
        "VALUES (?,?,?,?,?,?)",
        (time.time(), action, camera, actor, detail, 1 if ok else 0))
    return int(cur.lastrowid or 0)


def list_audit(limit: int = 80) -> list[dict[str, Any]]:
    rows = _query("SELECT * FROM audit ORDER BY ts DESC LIMIT ?",
                  (int(max(1, min(limit, 500))),))
    return [d for d in (row_to_dict(r) for r in rows) if d]


def audit_summary() -> dict[str, Any]:
    total = _query("SELECT COUNT(*) AS n FROM audit")[0]["n"]
    unlocks = _query("SELECT COUNT(*) AS n FROM audit WHERE action LIKE "
                     "'raw%' OR action LIKE 'clip%'")[0]["n"]
    denied = _query("SELECT COUNT(*) AS n FROM audit WHERE ok = 0")[0]["n"]
    last = _query("SELECT * FROM audit ORDER BY ts DESC LIMIT 1")
    return {"entries": total, "unlocks": unlocks, "denied": denied,
            "last": row_to_dict(last[0]) if last else None}


# --------------------------------------------------------------------------- #
# Counts and occupancy
# --------------------------------------------------------------------------- #

def record_count(camera: str, zone_id: int | None, zone_name: str,
                 count: int, ts: float | None = None) -> None:
    _exec("INSERT INTO counts (ts, camera, zone_id, zone_name, count) "
          "VALUES (?,?,?,?,?)",
          (ts if ts is not None else time.time(), camera, zone_id, zone_name,
           int(count)))


def count_series(since: float, camera: str | None = None,
                 zone_id: int | None = None,
                 buckets: int = 60) -> list[dict[str, Any]]:
    """Average occupancy per time bucket, for the analytics chart."""
    sql = "SELECT ts, camera, zone_id, zone_name, count FROM counts WHERE ts >= ?"
    params: list[Any] = [since]
    if camera:
        sql += " AND camera = ?"
        params.append(camera)
    if zone_id is not None:
        sql += " AND zone_id = ?"
        params.append(zone_id)
    sql += " ORDER BY ts"
    rows = _query(sql, params)
    if not rows:
        return []
    start, end = rows[0]["ts"], max(rows[-1]["ts"], rows[0]["ts"] + 1)
    width = (end - start) / max(1, buckets)
    acc: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        b = min(buckets - 1, int((r["ts"] - start) / width))
        acc.setdefault((b, r["zone_name"]), []).append(float(r["count"]))
    out = []
    for (b, name), values in sorted(acc.items()):
        out.append({"ts": start + b * width, "zone": name,
                    "count": round(sum(values) / len(values), 2)})
    return out


def purge_counts(before: float) -> int:
    cur = _exec("DELETE FROM counts WHERE ts < ?", (before,))
    return cur.rowcount or 0


def bump_occupancy(camera: str, zone_id: int, direction: str) -> dict[str, int]:
    column = "in_count" if direction == "in" else "out_count"
    _exec(f"""INSERT INTO occupancy (camera, zone_id, {column}, updated)
              VALUES (?,?,1,?)
              ON CONFLICT(camera, zone_id) DO UPDATE SET
                  {column} = {column} + 1, updated = excluded.updated""",
          (camera, zone_id, time.time()))
    return occupancy(camera, zone_id)


def occupancy(camera: str, zone_id: int) -> dict[str, int]:
    rows = _query("SELECT in_count, out_count FROM occupancy "
                  "WHERE camera = ? AND zone_id = ?", (camera, zone_id))
    if not rows:
        return {"in": 0, "out": 0, "inside": 0}
    i, o = int(rows[0]["in_count"]), int(rows[0]["out_count"])
    return {"in": i, "out": o, "inside": max(0, i - o)}


def all_occupancy() -> list[dict[str, Any]]:
    rows = _query("SELECT * FROM occupancy")
    out = []
    for r in rows:
        i, o = int(r["in_count"]), int(r["out_count"])
        out.append({"camera": r["camera"], "zone_id": r["zone_id"],
                    "in": i, "out": o, "inside": max(0, i - o)})
    return out


def reset_occupancy(camera: str | None = None) -> None:
    if camera:
        _exec("DELETE FROM occupancy WHERE camera = ?", (camera,))
    else:
        _exec("DELETE FROM occupancy")


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #

def database_report() -> dict[str, Any]:
    """Shown on the privacy panel: exactly what is being kept."""
    size = config.DB_PATH.stat().st_size if config.DB_PATH.exists() else 0
    return {
        "path": str(config.DB_PATH),
        "megabytes": round(size / 1e6, 2),
        "events": _query("SELECT COUNT(*) AS n FROM events")[0]["n"],
        "zones": _query("SELECT COUNT(*) AS n FROM zones")[0]["n"],
        "audit_entries": _query("SELECT COUNT(*) AS n FROM audit")[0]["n"],
        "count_samples": _query("SELECT COUNT(*) AS n FROM counts")[0]["n"],
        "stores_images": False,
        "stores_identities": False,
    }
