"""
The FastAPI application: routes, WebSocket hub, and the wiring between
the camera threads and the browser.

Two WebSockets, for a reason. ``/ws/video/{camera}`` carries binary JPEG frames
for one tile, which means the browser can open one socket per visible tile and
close it the moment that tile is hidden — a camera nobody is looking at costs
nothing to display. ``/ws/events`` carries JSON: alerts the instant they fire,
plus a status tick once a second.

Camera threads must never block on the event loop, so everything they produce
crosses into async through a plain thread-safe queue that one background task
drains. There is no ``run_coroutine_threadsafe`` from a worker thread anywhere
in this file, and no lock a camera can wait on.

**Raw access.** Three things are gated behind the admin PIN: the raw view of a
camera, the incident clips, and nothing else. Unlocking mints a short-lived
token, every unlock and every clip fetch is written to the audit table, and the
raw frame is not even JPEG-encoded until a valid token asks for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import secrets
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import (Body, FastAPI, HTTPException, Query, Request, Response,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles

import config
from backend import ai_assistant, alerts, events_db, privacy
from backend.alerts import AlertManager
from backend.camera_manager import CameraManager
from backend.pipeline import Pipeline
from backend.vision import model_store, simscene
from backend.zone_store import (LINE_TYPES, POLYGON_TYPES, Zone, ZoneSet,
                                validate_points)

# --------------------------------------------------------------------------- #
# Process-wide state
# --------------------------------------------------------------------------- #

manager = CameraManager()
pipelines: dict[str, Pipeline] = {}
alert_manager: AlertManager | None = None

#: Camera threads push here; one async task drains it to the browsers.
_outbox: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=512)

#: Live dashboard sockets.
_event_sockets: set[WebSocket] = set()

#: token -> expiry. Minted by /api/unlock, required for raw views and clips.
_admin_tokens: dict[str, float] = {}

_started_at = time.time()
_model_report: dict[str, bool] = {}


# --------------------------------------------------------------------------- #
# Admin tokens
# --------------------------------------------------------------------------- #

def _mint_token() -> str:
    token = secrets.token_urlsafe(24)
    _admin_tokens[token] = time.time() + config.ADMIN_SESSION_TTL_S
    return token


def _valid_token(token: str | None) -> bool:
    if not token:
        return False
    expiry = _admin_tokens.get(token)
    if expiry is None:
        return False
    if expiry < time.time():
        _admin_tokens.pop(token, None)
        return False
    return True


def _token_from(request: Request | None = None,
                token: str | None = None) -> str | None:
    if token:
        return token
    if request is not None:
        return (request.headers.get("x-admin-token")
                or request.query_params.get("token"))
    return None


def _require_admin(request: Request, token: str | None, action: str,
                   camera: str = "", detail: str = "") -> str:
    """Check the token, and write the attempt to the audit log either way."""
    actual = _token_from(request, token)
    if not _valid_token(actual):
        events_db.add_audit(action, camera, detail or "rejected: no valid PIN",
                            ok=False)
        raise HTTPException(status_code=401,
                            detail="Enter the staff PIN to view this.")
    return actual or ""


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #

def zones_for(camera: str) -> ZoneSet:
    return ZoneSet([Zone.from_row(r) for r in events_db.zone_rows(camera)])


def _seed_simulated_zones(camera: str, source: str) -> int:
    """Give a simulated camera the zones its scene was written around.

    Without this, a first run on a laptop with no camera shows a live dashboard
    where half the rules can never fire, because zone rules need zones and
    nobody has drawn any yet. Only ever applied to ``sim:`` sources, and only
    when that camera has no zones at all.
    """
    if not source.lower().startswith("sim:"):
        return 0
    if events_db.zone_count(camera):
        return 0
    scene = source.split(":", 1)[1].strip().lower()
    seeds = simscene.DEFAULT_ZONES.get(scene, [])
    for z in seeds:
        events_db.add_zone(camera, z["name"], z["type"], z["points"],
                           z.get("params"))
    return len(seeds)


# --------------------------------------------------------------------------- #
# Bridging camera threads to the event loop
# --------------------------------------------------------------------------- #

def _publish(message: dict[str, Any]) -> None:
    """Called from camera and notification threads. Never blocks."""
    try:
        _outbox.put_nowait(message)
    except queue.Full:
        # A dashboard that cannot keep up must not slow a camera down.
        pass


async def _broadcast(message: dict[str, Any]) -> None:
    if not _event_sockets:
        return
    text = json.dumps(message, default=str)
    dead: list[WebSocket] = []
    for ws in list(_event_sockets):
        try:
            await ws.send_text(text)
        except Exception:                                   # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        _event_sockets.discard(ws)


async def _outbox_pump() -> None:
    """Drain the thread-safe queue into the sockets."""
    while True:
        drained = 0
        while drained < 64:
            try:
                await _broadcast(_outbox.get_nowait())
                drained += 1
            except queue.Empty:
                break
        await asyncio.sleep(0.05)


async def _status_tick() -> None:
    """One status frame a second, so the header stays honest."""
    while True:
        try:
            await _broadcast({"kind": "status", "data": _status_payload()})
        except Exception:                                   # noqa: BLE001
            pass
        await asyncio.sleep(1.0)


async def _count_sampler() -> None:
    """Record crowd-zone occupancy for the analytics chart."""
    while True:
        interval = max(2.0, config.thresholds().crowd_sample_interval_s)
        await asyncio.sleep(interval)
        try:
            for name, worker in manager.workers.items():
                if worker.state != "live":
                    continue
                counts = worker.meta.get("zone_counts") or {}
                if not counts:
                    continue
                zset = zones_for(name)
                for zone in zset.of("crowd"):
                    events_db.record_count(name, zone.id, zone.name,
                                           int(counts.get(zone.id, 0)))
        except Exception:                                   # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Status payload
# --------------------------------------------------------------------------- #

def _status_payload() -> dict[str, Any]:
    totals = manager.totals()
    cameras = manager.status()
    for cam in cameras:
        pipe = pipelines.get(cam["name"])
        if pipe is not None:
            cam["zones"] = len(pipe.zones())
            cam["models"] = pipe._load_status
    return {
        "time": time.time(),
        "clock": time.strftime("%H:%M:%S"),
        "date": time.strftime("%a %d %b %Y"),
        "uptime_s": round(time.time() - _started_at, 1),
        "demo_mode": config.demo_mode(),
        "cameras": cameras,
        "totals": totals,
        "today": events_db.summary(),
        "alerts": alert_manager.status() if alert_manager else {},
    }


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global alert_manager, _model_report

    print(f"\n  {config.APP_NAME} — {config.APP_SUBTITLE}")
    print(f"  {config.APP_MOTTO}\n")

    events_db.init()
    pruned = privacy.prune_clips()
    if pruned["removed"]:
        print(f"  Retention: removed {pruned['removed']} expired clip(s)")

    stored = events_db.get_setting("demo_mode")
    if stored is not None:
        config.set_demo_mode(bool(stored))

    _model_report = model_store.preflight()

    alert_manager = AlertManager(broadcast=_publish)
    alert_manager.start()

    for name, source in config.CAMERA_SOURCES:
        seeded = _seed_simulated_zones(name, source)
        recorder = alert_manager.recorder_for(name)
        pipe = Pipeline(
            camera=name,
            zone_provider=zones_for,
            alert_sink=lambda cands, ctx: alert_manager.submit(cands, ctx),
            highlight_source=lambda cam, now: alert_manager.highlights(cam, now),
            snapshot_sink=lambda fired, img: alert_manager.attach_snapshots(
                fired, img),
        )
        pipelines[name] = pipe
        manager.add(name, source, processor=pipe, recorder=recorder)
        note = f" (+{seeded} default zones)" if seeded else ""
        print(f"  Camera: {name:<24} {source}{note}")

    # Restore the entry/exit tallies so a restart does not empty the building.
    for row in events_db.all_occupancy():
        pipe = pipelines.get(row["camera"])
        if pipe is None:
            continue
        for det in pipe.detectors:
            if det.name == "line_counter":
                det.seed(row["zone_id"], row["in"], row["out"])

    manager.start_all()

    tasks = [asyncio.create_task(_outbox_pump()),
             asyncio.create_task(_status_tick()),
             asyncio.create_task(_count_sampler())]

    print(f"\n  Dashboard: http://localhost:{config.PORT}")
    print(f"  Privacy:   raw footage on disk = 0 hours "
          f"({len(list(config.CLIPS_DIR.glob('*.mp4')))} incident clips)\n")

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        manager.stop_all()
        if alert_manager:
            alert_manager.stop()
        events_db.close()


app = FastAPI(title=f"{config.APP_NAME} — {config.APP_SUBTITLE}",
              version=config.APP_VERSION, lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Pages and static files
# --------------------------------------------------------------------------- #

if config.FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)),
              name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = config.FRONTEND_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>ARJUN</h1><p>frontend/index.html is missing.</p>",
                            status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


# --------------------------------------------------------------------------- #
# Configuration and status
# --------------------------------------------------------------------------- #

@app.get("/api/config")
async def api_config() -> dict[str, Any]:
    payload = config.as_dict()
    payload["models"] = _model_report
    payload["scenes"] = simscene.available()
    payload["channels"] = alerts.channel_report()
    payload["assistant"] = ai_assistant.status()
    return payload


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return _status_payload()


@app.get("/api/detectors")
async def api_detectors(camera: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, pipe in pipelines.items():
        if camera and name != camera:
            continue
        out[name] = pipe.detector_status()
    return out


@app.post("/api/demo")
async def api_demo(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    on = bool(payload.get("on", not config.demo_mode()))
    config.set_demo_mode(on)
    events_db.set_setting("demo_mode", on)
    events_db.add_audit("demo_mode", detail=f"demo mode {'on' if on else 'off'}")
    _publish({"kind": "demo", "data": {"on": on}})
    t = config.thresholds()
    return {"demo_mode": on, "loiter_seconds": t.loiter_seconds,
            "abandoned_seconds": t.abandoned_seconds,
            "fall_down_seconds": t.fall_down_seconds}


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

@app.get("/api/events")
async def api_events(
    limit: int = Query(120, ge=1, le=1000),
    camera: str | None = None,
    type: str | None = None,
    severity: str | None = None,
    since: float | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    rows = events_db.list_events(limit=limit, camera=camera, etype=type,
                                 severity=severity, since=since, search=search)
    for row in rows:
        spec = config.alert_spec(row["type"])
        row["title"] = spec.title
        row["icon"] = spec.icon
        row["has_clip"] = bool(row.get("clip_path"))
        row["snapshot"] = (f"/api/snapshot/{row['id']}"
                           if row.get("snapshot_path") else None)
    return {"events": rows, "count": len(rows)}


@app.get("/api/events/{event_id}")
async def api_event(event_id: int) -> dict[str, Any]:
    row = events_db.get_event(event_id)
    if row is None:
        raise HTTPException(404, "No such event")
    spec = config.alert_spec(row["type"])
    row["title"] = spec.title
    row["icon"] = spec.icon
    row["explanation"] = spec.detail
    row["snapshot"] = (f"/api/snapshot/{event_id}"
                       if row.get("snapshot_path") else None)
    row["clip"] = f"/api/clip/{event_id}" if row.get("clip_path") else None
    row["related"] = events_db.related_events(event_id)
    return row


@app.post("/api/events/{event_id}/ack")
async def api_ack(event_id: int,
                  payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    value = bool(payload.get("acknowledged", True))
    events_db.acknowledge(event_id, value)
    _publish({"kind": "alert.update",
              "data": {"id": event_id, "acknowledged": value}})
    return {"ok": True, "id": event_id, "acknowledged": value}


@app.get("/api/snapshot/{event_id}")
async def api_snapshot(event_id: int) -> Response:
    """The anonymous skeleton render. Deliberately needs no PIN."""
    row = events_db.get_event(event_id)
    if row is None or not row.get("snapshot_path"):
        raise HTTPException(404, "No snapshot for that event")
    path = Path(row["snapshot_path"])
    if not path.exists():
        raise HTTPException(404, "Snapshot file is gone")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/clip/{event_id}")
async def api_clip(event_id: int, request: Request,
                   token: str | None = None) -> Response:
    """Raw incident footage. PIN required, and the access is logged."""
    row = events_db.get_event(event_id)
    if row is None or not row.get("clip_path"):
        raise HTTPException(404, "No clip for that event")
    _require_admin(request, token, "clip_view", row["camera"],
                   f"event {event_id}")
    path = config.CLIPS_DIR / str(row["clip_path"])
    if not path.exists():
        raise HTTPException(404, "Clip file is gone")
    events_db.add_audit("clip_view", row["camera"],
                        f"event {event_id} ({row['type']})")
    _publish({"kind": "audit", "data": {"action": "clip_view",
                                        "camera": row["camera"],
                                        "detail": f"event {event_id}"}})
    return FileResponse(path, media_type="video/mp4",
                        headers={"Accept-Ranges": "bytes"})


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #

@app.get("/api/zones")
async def api_zones(camera: str | None = None) -> dict[str, Any]:
    rows = [Zone.from_row(r).to_dict() for r in events_db.zone_rows(camera)]
    return {"zones": rows, "types": config.ZONE_TYPES}


@app.post("/api/zones")
async def api_zone_create(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    camera = str(payload.get("camera", "")).strip()
    name = str(payload.get("name", "")).strip() or "Zone"
    ztype = str(payload.get("type", "restricted")).strip()
    if camera not in manager.workers:
        raise HTTPException(400, f"No camera called {camera!r}")
    if ztype not in POLYGON_TYPES + LINE_TYPES:
        raise HTTPException(400, f"Unknown zone type {ztype!r}")
    try:
        points = validate_points(ztype, payload.get("points"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    params = payload.get("params") or {}
    zone_id = events_db.add_zone(camera, name, ztype, points, params)
    _invalidate(camera)
    events_db.add_audit("zone_create", camera, f"{ztype} '{name}'")
    return {"ok": True, "id": zone_id}


@app.patch("/api/zones/{zone_id}")
async def api_zone_update(zone_id: int,
                          payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("name", "enabled", "params"):
        if key in payload:
            fields[key] = payload[key]
    if "points" in payload:
        ztype = str(payload.get("type", "restricted"))
        try:
            fields["points"] = validate_points(ztype, payload["points"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if not events_db.update_zone(zone_id, **fields):
        raise HTTPException(404, "No such zone, or nothing to change")
    for cam in pipelines:
        _invalidate(cam)
    return {"ok": True, "id": zone_id}


@app.delete("/api/zones/{zone_id}")
async def api_zone_delete(zone_id: int) -> dict[str, Any]:
    if not events_db.delete_zone(zone_id):
        raise HTTPException(404, "No such zone")
    for cam in pipelines:
        _invalidate(cam)
    events_db.add_audit("zone_delete", detail=f"zone {zone_id}")
    return {"ok": True}


def _invalidate(camera: str) -> None:
    pipe = pipelines.get(camera)
    if pipe is not None:
        pipe.invalidate_zones()


# --------------------------------------------------------------------------- #
# Camera control
# --------------------------------------------------------------------------- #

@app.post("/api/camera/{camera}/flag")
async def api_flag(camera: str,
                   payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    worker = manager.get(camera)
    if worker is None:
        raise HTTPException(404, "No such camera")
    key = str(payload.get("key", ""))
    value = bool(payload.get("value", False))
    if not worker.set_flag(key, value):
        raise HTTPException(400, f"No such flag {key!r}")
    events_db.add_audit("camera_flag", camera, f"{key} = {value}")
    _publish({"kind": "camera", "data": worker.status()})
    return {"ok": True, "flags": worker.flags.to_dict()}


@app.get("/api/frame/{camera}")
async def api_frame(camera: str) -> Response:
    """A still of the privacy view — what the zone editor draws on."""
    worker = manager.get(camera)
    if worker is None:
        raise HTTPException(404, "No such camera")
    jpeg = worker.display_jpeg()
    if jpeg is None:
        raise HTTPException(503, "That camera has not produced a frame yet")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/unlock")
async def api_unlock(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Exchange the staff PIN for a short-lived token."""
    pin = str(payload.get("pin", ""))
    camera = str(payload.get("camera", ""))
    if not secrets.compare_digest(pin, config.ADMIN_PIN):
        events_db.add_audit("raw_unlock", camera, "rejected: wrong PIN",
                            ok=False)
        _publish({"kind": "audit", "data": {"action": "raw_unlock",
                                            "camera": camera, "ok": False,
                                            "detail": "wrong PIN"}})
        raise HTTPException(403, "That PIN is not right.")
    token = _mint_token()
    events_db.add_audit("raw_unlock", camera,
                        f"granted for {config.ADMIN_SESSION_TTL_S:.0f}s")
    events_db.add_event(camera or "—", "raw_view", "info",
                        f"Raw view unlocked on {camera or 'the system'} by "
                        f"staff PIN.", meta={"ttl_s": config.ADMIN_SESSION_TTL_S},
                        clip_state="none")
    _publish({"kind": "audit", "data": {"action": "raw_unlock",
                                        "camera": camera, "ok": True}})
    if camera:
        worker = manager.get(camera)
        if worker is not None:
            worker.request_raw(config.ADMIN_SESSION_TTL_S)
    return {"ok": True, "token": token,
            "expires_in": config.ADMIN_SESSION_TTL_S}


@app.post("/api/lock/{camera}")
async def api_lock(camera: str) -> dict[str, Any]:
    worker = manager.get(camera)
    if worker is None:
        raise HTTPException(404, "No such camera")
    worker.revoke_raw()
    events_db.add_audit("raw_lock", camera, "raw view closed")
    return {"ok": True}


@app.get("/api/raw/{camera}")
async def api_raw(camera: str, request: Request,
                  token: str | None = None) -> Response:
    """A single raw frame. PIN required; logged; never cached."""
    worker = manager.get(camera)
    if worker is None:
        raise HTTPException(404, "No such camera")
    _require_admin(request, token, "raw_view", camera)
    worker.request_raw(min(20.0, config.ADMIN_SESSION_TTL_S))
    jpeg = worker.raw_jpeg()
    if jpeg is None:
        # The encoder only starts once a token asks; give it a frame or two.
        await asyncio.sleep(0.35)
        jpeg = worker.raw_jpeg()
    if jpeg is None:
        raise HTTPException(503, "No raw frame available yet")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/mjpeg/{camera}")
async def api_mjpeg(camera: str) -> StreamingResponse:
    """The privacy view as MJPEG — a fallback for phones and old browsers."""
    worker = manager.get(camera)
    if worker is None:
        raise HTTPException(404, "No such camera")

    async def frames():
        interval = 1.0 / max(1.0, config.STREAM_FPS)
        while True:
            jpeg = worker.display_jpeg()
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(interval)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------- #
# Analytics, privacy panel, audit
# --------------------------------------------------------------------------- #

@app.get("/api/analytics")
async def api_analytics(hours: float = Query(12.0, ge=0.1, le=720.0)
                        ) -> dict[str, Any]:
    since = time.time() - hours * 3600
    return {
        "since": since,
        "by_type": events_db.counts_by_type(since),
        "by_hour": events_db.counts_by_hour(since),
        "crowd": events_db.count_series(since),
        "summary": events_db.summary(since),
        "occupancy": events_db.all_occupancy(),
        "titles": {k: v.title for k, v in config.ALERT_SPECS.items()},
    }


@app.get("/api/privacy")
async def api_privacy() -> dict[str, Any]:
    """Everything the privacy panel shows. This is the pitch, in JSON."""
    buffers = alert_manager.buffer_report() if alert_manager else []
    return {
        "storage": privacy.storage_report(),
        "database": events_db.database_report(),
        "buffers": buffers,
        "buffer_seconds": config.RAW_BUFFER_SECONDS,
        "audit": events_db.audit_summary(),
        "recent_audit": events_db.list_audit(limit=30),
        "guarantees": [
            {"claim": "Hours of continuous footage stored",
             "value": "0",
             "why": "Nothing in the codebase can write continuous video. The "
                    "only writer of pixels is the incident recorder, and it "
                    "only runs when a detector fires."},
            {"claim": "Faces stored, anywhere, ever",
             "value": "0",
             "why": "The dashboard is rendered from geometry onto a blank "
                    "canvas. The camera frame is never a layer in it."},
            {"claim": "Named or re-identifiable people",
             "value": "0",
             "why": "Track numbers are invented when someone appears and "
                    "destroyed seconds after they leave. There is no "
                    "appearance descriptor and no face recognition."},
            {"claim": "Seconds of raw video held in memory, per camera",
             "value": str(int(config.RAW_BUFFER_SECONDS)),
             "why": "A fixed-length ring buffer, continuously overwritten. "
                    "Pull the power out and it is gone."},
            {"claim": "Raw views and clip opens recorded in the audit log",
             "value": "all of them",
             "why": "Unlocking a raw feed is itself an event, with a time and "
                    "a camera, shown on this panel."},
        ],
    }


@app.get("/api/audit")
async def api_audit(limit: int = Query(80, ge=1, le=500)) -> dict[str, Any]:
    return {"audit": events_db.list_audit(limit), **events_db.audit_summary()}


# --------------------------------------------------------------------------- #
# Assistant
# --------------------------------------------------------------------------- #

@app.post("/api/ask")
async def api_ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    camera = payload.get("camera") or None
    if not question:
        raise HTTPException(400, "Ask a question first")
    answer = await asyncio.to_thread(ai_assistant.ask, question, None, camera)
    events_db.add_audit("ask", camera or "", question[:160])
    return answer.to_dict()


@app.post("/api/report/{event_id}")
async def api_report(event_id: int) -> dict[str, Any]:
    answer = await asyncio.to_thread(ai_assistant.incident_report, event_id)
    events_db.add_audit("report", detail=f"event {event_id} via {answer.engine}")
    return answer.to_dict()


@app.get("/api/report/{event_id}/download")
async def api_report_download(event_id: int) -> Response:
    answer = await asyncio.to_thread(ai_assistant.incident_report, event_id)
    name = f"arjun-incident-{event_id:05d}.md"
    return Response(
        content=answer.text, media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'})


# --------------------------------------------------------------------------- #
# WebSockets
# --------------------------------------------------------------------------- #

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    _event_sockets.add(ws)
    try:
        await ws.send_text(json.dumps({"kind": "hello",
                                       "data": _status_payload()}, default=str))
        while True:
            # The client says nothing; this is how we notice it has gone.
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:                                       # noqa: BLE001
        pass
    finally:
        _event_sockets.discard(ws)


@app.websocket("/ws/video/{camera}")
async def ws_video(ws: WebSocket, camera: str) -> None:
    """Binary JPEG frames for one tile, paced to STREAM_FPS."""
    await ws.accept()
    worker = manager.get(camera)
    if worker is None:
        await ws.close(code=1008)
        return

    interval = 1.0 / max(1.0, config.STREAM_FPS)
    last_sent = 0.0
    try:
        while True:
            await asyncio.sleep(interval)
            if worker.last_frame_at <= last_sent:
                continue            # nothing new; do not resend the same bytes
            jpeg = worker.display_jpeg()
            if jpeg is None:
                continue
            last_sent = worker.last_frame_at
            await ws.send_bytes(jpeg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:                                       # noqa: BLE001
        pass


@app.websocket("/ws/raw/{camera}")
async def ws_raw(ws: WebSocket, camera: str) -> None:
    """Raw frames for one tile. Requires a valid token on the query string."""
    await ws.accept()
    worker = manager.get(camera)
    token = ws.query_params.get("token")
    if worker is None or not _valid_token(token):
        events_db.add_audit("raw_stream", camera, "rejected: no valid PIN",
                            ok=False)
        await ws.close(code=1008)
        return

    events_db.add_audit("raw_stream", camera, "raw stream opened")
    _publish({"kind": "audit", "data": {"action": "raw_stream",
                                        "camera": camera, "ok": True}})
    interval = 1.0 / max(1.0, config.STREAM_FPS)
    try:
        while True:
            await asyncio.sleep(interval)
            if not _valid_token(token):
                break
            # Keep the encoder alive only while this socket is open.
            worker.request_raw(5.0)
            jpeg = worker.raw_jpeg()
            if jpeg is not None:
                await ws.send_bytes(jpeg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:                                       # noqa: BLE001
        pass
    finally:
        worker.revoke_raw()
        events_db.add_audit("raw_stream", camera, "raw stream closed")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail, "status": exc.status_code},
                        status_code=exc.status_code)
