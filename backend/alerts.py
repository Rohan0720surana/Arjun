"""
What happens after a rule says "I think something happened".

Every detector produces :class:`~backend.detectors.base.Candidate` objects and
stops there. This module turns a candidate into an incident exactly once, and
does the same five things every time:

1.  **Cooldown.** One person falling produces a candidate on every frame for as
    long as they are on the floor. Without suppression that is sixty Telegram
    messages a second. Cooldowns are keyed by camera + rule + track + zone, so
    two different people falling are two incidents, while one person falling is
    one.
2.  **Persist.** A row in SQLite, with the rule's own numbers attached, so the
    report generator and the assistant can explain *why* it fired.
3.  **Record.** The incident recorder is asked for a clip: ten seconds before,
    five after. This is the only moment in the entire system that raw pixels
    are allowed onto disk.
4.  **Show.** Pushed to every connected dashboard over the WebSocket
    immediately — before the snapshot and well before the clip — because the
    first second is the one that matters.
5.  **Tell.** Telegram, and WhatsApp if configured, with a *skeleton* snapshot
    attached. The notification never contains a camera image. A guard reading
    their phone sees the geometry of what happened and nothing about who.

Notifications go out on their own thread. A slow network must not be able to
stall a camera.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

import config
from backend import events_db
from backend.detectors.base import Candidate
from backend.privacy import IncidentRecorder
from backend.state import CameraContext

Broadcaster = Callable[[dict[str, Any]], None]


@dataclass
class FiredAlert:
    """A candidate that survived the cooldown and became an incident."""

    event_id: int
    camera: str
    type: str
    severity: str
    title: str
    message: str
    ts: float
    track_id: int | None = None
    zone_id: int | None = None
    zone_name: str | None = None
    simulated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    clip_state: str = "none"
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        spec = config.alert_spec(self.type)
        return {
            "id": self.event_id,
            "camera": self.camera,
            "type": self.type,
            "title": self.title,
            "icon": spec.icon,
            "severity": self.severity,
            "message": self.message,
            "ts": self.ts,
            "clock": time.strftime("%H:%M:%S", time.localtime(self.ts)),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.ts)),
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "simulated": self.simulated,
            "silent": spec.silent,
            "meta": self.meta,
            "clip_state": self.clip_state,
            "snapshot": (f"/api/snapshot/{self.event_id}"
                         if self.snapshot_path else None),
        }


class AlertManager:
    """One per process. Thread-safe: detectors call in from camera threads."""

    def __init__(self, broadcast: Broadcaster | None = None) -> None:
        self.broadcast = broadcast
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._global_cooldowns: dict[str, float] = {}
        #: (camera, track_id) -> (until, severity), for the renderer.
        self._highlights: dict[tuple[str, int], tuple[float, str]] = {}
        self._recorders: dict[str, IncidentRecorder] = {}
        self._notify_q: "queue.Queue[FiredAlert]" = queue.Queue(maxsize=200)
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self.sent = {"telegram": 0, "whatsapp": 0, "failed": 0}
        self.last_error = ""
        self.counters: dict[str, Any] = {}

    # -- wiring --------------------------------------------------------------- #

    def recorder_for(self, camera: str) -> IncidentRecorder:
        with self._lock:
            rec = self._recorders.get(camera)
            if rec is None:
                rec = IncidentRecorder(camera, on_ready=self._on_clip_ready)
                self._recorders[camera] = rec
            return rec

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._notify_loop,
                                            name="alerts::notify", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    # -- the main entry point ---------------------------------------------------- #

    def submit(self, candidates: Iterable[Candidate],
               ctx: CameraContext) -> list[FiredAlert]:
        """Apply cooldowns, persist, record, broadcast. Returns what fired."""
        fired: list[FiredAlert] = []
        now = ctx.now
        t = config.thresholds()

        for cand in candidates:
            severity = cand.resolved_severity()
            key = f"{ctx.camera}:{cand.cooldown_key()}"
            gkey = f"{ctx.camera}:{cand.type}"

            with self._lock:
                last = self._cooldowns.get(key, 0.0)
                if now - last < t.cooldown_s:
                    continue
                # A global per-rule cooldown stops a swarm of near-identical
                # alerts, but it must never silence a *critical* incident
                # involving a different person: two people falling is two
                # emergencies.
                if severity != "critical":
                    glast = self._global_cooldowns.get(gkey, 0.0)
                    if now - glast < t.cooldown_global_s:
                        continue
                self._cooldowns[key] = now
                self._global_cooldowns[gkey] = now

            spec = cand.spec
            meta = cand.payload()
            meta.setdefault("rule", cand.type)
            if ctx.simulated:
                meta["simulated_feed"] = True

            event_id = events_db.add_event(
                camera=ctx.camera, etype=cand.type, severity=severity,
                message=cand.message, ts=now,
                zone_id=cand.zone.id if cand.zone else None,
                zone_name=cand.zone.name if cand.zone else None,
                track_id=cand.track_id, meta=meta, simulated=ctx.simulated,
                clip_state="pending" if spec.clip else "none",
            )

            alert = FiredAlert(
                event_id=event_id, camera=ctx.camera, type=cand.type,
                severity=severity, title=spec.title, message=cand.message,
                ts=now, track_id=cand.track_id,
                zone_id=cand.zone.id if cand.zone else None,
                zone_name=cand.zone.name if cand.zone else None,
                simulated=ctx.simulated, meta=meta,
                clip_state="pending" if spec.clip else "none",
            )

            # Highlight the subject on the privacy view for a few seconds.
            if cand.track_id is not None:
                with self._lock:
                    self._highlights[(ctx.camera, cand.track_id)] = (
                        now + cand.highlight_for,
                        cand.highlight or severity,
                    )

            if spec.clip:
                rec = self._recorders.get(ctx.camera)
                if rec is not None:
                    started = rec.trigger(
                        key=f"{cand.type}:{cand.track_id}",
                        event_ref=str(event_id),
                        meta={"camera": ctx.camera, "title": spec.title,
                              "severity": severity, "simulated": ctx.simulated},
                    )
                    if not started:
                        alert.clip_state = "merged"
                        events_db.set_event_clip(event_id, None, "merged")
                else:
                    alert.clip_state = "none"
                    events_db.set_event_clip(event_id, None, "none")

            # Entry/exit tallies belong to the room, not to this process:
            # persist them so a restart does not empty the building.
            if cand.type == "line_cross" and cand.zone is not None:
                direction = str(meta.get("direction", "in"))
                try:
                    events_db.bump_occupancy(ctx.camera, cand.zone.id, direction)
                except Exception as exc:                    # noqa: BLE001
                    self.last_error = f"occupancy: {exc}"

            self._emit("alert", alert.to_dict())
            fired.append(alert)

        return fired

    # -- snapshots ------------------------------------------------------------------ #

    def attach_snapshots(self, fired: list[FiredAlert],
                         display: np.ndarray) -> None:
        """Save the *anonymous* view as the evidence image for these alerts.

        This image is a render of skeletons and zones. It is safe to put in a
        chat message, safe to forward, and safe to keep — which is precisely
        why the notification carries this and never the camera frame.
        """
        if not fired:
            return
        ok, buf = cv2.imencode(".jpg", display,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            return
        blob = buf.tobytes()
        for alert in fired:
            path = config.SNAPSHOT_DIR / f"event-{alert.event_id}.jpg"
            try:
                path.write_bytes(blob)
            except OSError as exc:
                self.last_error = f"snapshot write failed: {exc}"
                continue
            alert.snapshot_path = str(path)
            events_db.set_event_snapshot(alert.event_id, str(path))
            self._emit("alert.update", {
                "id": alert.event_id,
                "snapshot": f"/api/snapshot/{alert.event_id}",
            })
            if self._should_notify(alert.severity):
                try:
                    self._notify_q.put_nowait(alert)
                except queue.Full:
                    self.last_error = "notification queue full"

    @staticmethod
    def _should_notify(severity: str) -> bool:
        return (config.severity_rank(severity)
                >= config.severity_rank(config.NOTIFY_MIN_SEVERITY))

    # -- clips ------------------------------------------------------------------------ #

    def _on_clip_ready(self, event_ref: str, path: Path,
                       meta: dict[str, Any]) -> None:
        try:
            event_id = int(event_ref)
        except ValueError:
            return
        events_db.set_event_clip(event_id, path.name, "ready")
        self._emit("alert.update", {
            "id": event_id,
            "clip_state": "ready",
            "clip": f"/api/clip/{event_id}",
            "clip_meta": meta,
        })

    # -- highlights --------------------------------------------------------------------- #

    def highlights(self, camera: str, now: float) -> dict[int, str]:
        with self._lock:
            out: dict[int, str] = {}
            for (cam, tid), (until, sev) in list(self._highlights.items()):
                if until < now:
                    del self._highlights[(cam, tid)]
                elif cam == camera:
                    out[tid] = sev
            return out

    # -- notifications ------------------------------------------------------------------- #

    def _notify_loop(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self._notify_q.get(timeout=0.5)
            except queue.Empty:
                continue
            channels: list[str] = []
            try:
                if send_telegram(alert):
                    channels.append("telegram")
                    self.sent["telegram"] += 1
                if send_whatsapp(alert):
                    channels.append("whatsapp")
                    self.sent["whatsapp"] += 1
            except Exception as exc:                        # noqa: BLE001
                self.sent["failed"] += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            if channels:
                events_db.set_event_notified(alert.event_id, channels)

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self.broadcast is None:
            return
        try:
            self.broadcast({"kind": kind, "data": payload})
        except Exception as exc:                            # noqa: BLE001
            self.last_error = f"broadcast: {exc}"

    # -- introspection --------------------------------------------------------------------- #

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = sum(r.pending_count() for r in self._recorders.values())
            written = sum(r.written for r in self._recorders.values())
        return {
            "channels": {
                "telegram": bool(config.TELEGRAM_BOT_TOKEN
                                 and config.TELEGRAM_CHAT_ID),
                "whatsapp": bool(config.TWILIO_ACCOUNT_SID
                                 and config.TWILIO_AUTH_TOKEN
                                 and config.TWILIO_WHATSAPP_TO),
            },
            "sent": dict(self.sent),
            "queued": self._notify_q.qsize(),
            "clips_pending": pending,
            "clips_written": written,
            "min_severity": config.NOTIFY_MIN_SEVERITY,
            "last_error": self.last_error,
        }

    def buffer_report(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"camera": cam, **rec.buffer.stats()}
                    for cam, rec in self._recorders.items()]


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #

def _caption(alert: FiredAlert) -> str:
    spec = config.alert_spec(alert.type)
    icon = {"critical": "[!]", "warning": "[-]", "info": "[i]"}.get(
        alert.severity, "[i]")
    lines = [
        f"{icon} {spec.title} — {alert.severity.upper()}",
        f"Camera: {alert.camera}",
    ]
    if alert.zone_name:
        lines.append(f"Zone: {alert.zone_name}")
    lines.append(f"Time: {time.strftime('%d %b %H:%M:%S', time.localtime(alert.ts))}")
    lines.append(f"Detail: {alert.message}")
    if alert.simulated:
        lines.append("NOTE: simulated feed (demonstration)")
    lines.append("")
    lines.append(f"{config.APP_NAME} — image shows skeletons only, never faces.")
    return "\n".join(lines)


#: Why each channel last failed. A silent False is fine in the frame loop and
#: useless when somebody is standing in front of a judge asking why their
#: phone did not buzz, so the reason is kept here and surfaced by
#: ``run.py --test-alert`` and the dashboard's integrations strip.
LAST_ERROR: dict[str, str] = {"telegram": "", "whatsapp": ""}


def send_telegram(alert: FiredAlert) -> bool:
    """Push to Telegram. Returns False (quietly) when it is not configured."""
    token, chat = config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID
    if not token or not chat:
        LAST_ERROR["telegram"] = "not configured"
        return False
    import requests

    caption = _caption(alert)
    base = f"https://api.telegram.org/bot{token}"
    try:
        if alert.snapshot_path and Path(alert.snapshot_path).exists():
            with open(alert.snapshot_path, "rb") as fh:
                resp = requests.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": chat, "caption": caption[:1024]},
                    files={"photo": ("skeleton.jpg", fh, "image/jpeg")},
                    timeout=12,
                )
        else:
            resp = requests.post(f"{base}/sendMessage",
                                 data={"chat_id": chat, "text": caption[:4000]},
                                 timeout=12)
        if resp.status_code == 200:
            LAST_ERROR["telegram"] = ""
            return True
        # Telegram explains itself properly; pass that on rather than a code.
        try:
            detail = resp.json().get("description", resp.text[:200])
        except Exception:                                   # noqa: BLE001
            detail = resp.text[:200]
        LAST_ERROR["telegram"] = f"HTTP {resp.status_code}: {detail}"
        return False
    except Exception as exc:                                # noqa: BLE001
        LAST_ERROR["telegram"] = f"{type(exc).__name__}: {exc}"
        return False


def send_whatsapp(alert: FiredAlert) -> bool:
    """Push to WhatsApp through the Twilio sandbox, if credentials exist."""
    sid, token = config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN
    to, frm = config.TWILIO_WHATSAPP_TO, config.TWILIO_WHATSAPP_FROM
    if not (sid and token and to and frm):
        LAST_ERROR["whatsapp"] = "not configured"
        return False
    import requests

    data = {"From": frm, "To": to, "Body": _caption(alert)[:1500]}
    # Twilio fetches media over the public internet, so an image can only be
    # attached when the dashboard is reachable from outside this machine.
    if config.PUBLIC_BASE_URL and alert.snapshot_path:
        data["MediaUrl"] = (f"{config.PUBLIC_BASE_URL.rstrip('/')}"
                            f"/api/snapshot/{alert.event_id}")
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data=data, auth=(sid, token), timeout=12)
        if resp.status_code in (200, 201):
            LAST_ERROR["whatsapp"] = ""
            return True
        try:
            detail = resp.json().get("message", resp.text[:200])
        except Exception:                                   # noqa: BLE001
            detail = resp.text[:200]
        LAST_ERROR["whatsapp"] = f"HTTP {resp.status_code}: {detail}"
        return False
    except Exception as exc:                                # noqa: BLE001
        LAST_ERROR["whatsapp"] = f"{type(exc).__name__}: {exc}"
        return False


def channel_report() -> dict[str, Any]:
    """What the dashboard shows in the integrations strip."""
    return {
        "telegram": {
            "configured": bool(config.TELEGRAM_BOT_TOKEN
                               and config.TELEGRAM_CHAT_ID),
            "hint": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env",
            "last_error": LAST_ERROR["telegram"],
        },
        "whatsapp": {
            "configured": bool(config.TWILIO_ACCOUNT_SID
                               and config.TWILIO_AUTH_TOKEN
                               and config.TWILIO_WHATSAPP_TO),
            "hint": "Set the TWILIO_* keys in .env (sandbox is fine)",
            "last_error": LAST_ERROR["whatsapp"],
        },
        "claude": {
            "configured": bool(config.CLAUDE_API_KEY),
            "hint": "Set CLAUDE_API_KEY in .env for written answers and reports",
        },
    }
