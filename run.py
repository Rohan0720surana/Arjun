#!/usr/bin/env python3
"""
ARJUN — one command starts everything.

    python run.py                      cameras from .env, dashboard on :8000
    python run.py --demo               short timers, for a live demonstration
    python run.py --cameras "Hall=0"   override the camera list for one run
    python run.py --sim                ignore .env and run the scripted scenes
    python run.py --check              verify the install and exit

The first run downloads about 30 MB of pretrained weights into ``models/``.
After that it works offline.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BANNER = r"""
    _    ____      _ _   _ _   _
   / \  |  _ \    | | | | | \ | |     ARJUN
  / _ \ | |_) |_  | | | | |  \| |     Campus Guardian
 / ___ \|  _ <| |_| | |_| | |\  |
/_/   \_\_| \_\\___/ \___/|_| \_|     "Arjuna saw only the eye of the bird."
"""

REQUIRED = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("dotenv", "python-dotenv"),
]
OPTIONAL = [
    ("ultralytics", "ultralytics", "detection and pose — the system needs this "
                                   "for real cameras"),
    ("mediapipe", "mediapipe", "the SOS hand gesture and exam-mode head turn"),
    ("anthropic", "anthropic", "written answers and incident reports"),
    ("requests", "requests", "Telegram and WhatsApp alerts"),
]


def _check_imports() -> list[str]:
    missing: list[str] = []
    for module, package in REQUIRED:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def _report_optional() -> None:
    for module, package, why in OPTIONAL:
        try:
            __import__(module)
        except ImportError:
            print(f"  ! {package} is not installed — {why} will be disabled.")


def _ensure_env() -> None:
    """First run convenience: copy .env.example to .env if it is missing."""
    env, example = ROOT / ".env", ROOT / ".env.example"
    if env.exists() or not example.exists():
        return
    env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print("  Created .env from .env.example — edit it to add cameras or keys.")


def _open_browser_later(url: str, delay: float = 2.0) -> None:
    def go() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:                                   # noqa: BLE001
            pass
    threading.Thread(target=go, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="ARJUN — privacy-first campus safety")
    parser.add_argument("--host", default=None, help="bind address")
    parser.add_argument("--port", type=int, default=None, help="port")
    parser.add_argument("--cameras", default=None,
                        help='override CAMERA_SOURCES, e.g. "Hall=0, Gate=sim:gate"')
    parser.add_argument("--sim", action="store_true",
                        help="run the three scripted scenes and nothing else")
    parser.add_argument("--demo", action="store_true",
                        help="start in demo mode (short timers)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--check", action="store_true",
                        help="verify the installation and exit")
    parser.add_argument("--test-alert", action="store_true",
                        help="send yourself one real alert, then exit")
    parser.add_argument("--reload", action="store_true",
                        help="reload on code changes (development)")
    args = parser.parse_args()

    print(BANNER)

    missing = _check_imports()
    if missing:
        print("  Missing required packages: " + ", ".join(missing))
        print("\n  Install everything with:\n")
        print("      python -m pip install -r requirements.txt\n")
        return 1
    _report_optional()

    _ensure_env()

    # Environment overrides have to happen before config is imported, because
    # config reads them once at import time.
    if args.sim:
        os.environ["CAMERA_SOURCES"] = ("Atrium=sim:atrium, "
                                        "Exam Hall=sim:exam, Gate=sim:gate")
    elif args.cameras:
        os.environ["CAMERA_SOURCES"] = args.cameras
    if args.demo:
        os.environ["DEMO_MODE"] = "1"
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.no_browser:
        os.environ["OPEN_BROWSER"] = "0"

    import config

    if args.check:
        return _run_check(config)

    if args.test_alert:
        return _run_test_alert(config)

    host, port = config.HOST, config.PORT
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    if config.OPEN_BROWSER and not args.reload:
        _open_browser_later(f"http://{shown}:{port}")

    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=args.reload,
                log_level="warning", access_log=False)
    return 0


def _run_test_alert(config) -> int:
    """Fire one genuine alert and report what each channel did with it.

    Deliberately not a "send a test message" helper. It builds a fallen
    skeleton, renders it through the real privacy renderer, and pushes it
    through the real AlertManager — the same cooldowns, the same database
    write, the same snapshot, the same notification thread a real incident
    uses. If this arrives on your phone, a real fall will too, and it will
    look exactly like this.
    """
    import time

    from backend import alerts as alerts_mod
    from backend import events_db
    from backend.alerts import AlertManager
    from backend.camera_manager import CameraFlags
    from backend.detectors.base import Candidate
    from backend.privacy import PrivacyRenderer
    from backend.state import CameraContext, TrackStore
    from backend.vision import synth
    from backend.world import Detection, FrameWorld
    from backend.zone_store import Zone, ZoneSet

    camera = "Test Camera"
    telegram_ready = bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)
    whatsapp_ready = bool(config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN
                          and config.TWILIO_WHATSAPP_TO)

    print("  Sending one real alert through the full path.\n")
    if not telegram_ready and not whatsapp_ready:
        print("  No push channel is configured, so this will reach the")
        print("  database and the dashboard but no phone.")
        print("  Run  python setup_alerts.py  to connect Telegram.\n")

    events_db.init()
    manager = AlertManager(broadcast=lambda _msg: None)
    manager.start()

    # A person on the floor, built from the same parametric model the tests use.
    width, height = 640, 360
    kps = synth.lying(width * 0.45, height * 0.86, height * 0.42, angle_deg=86)
    person = Detection(track_id=7, label="person", conf=0.91,
                       box=synth.box_for(kps), keypoints=kps)
    world = FrameWorld(camera=camera, ts=time.time(), frame_idx=1,
                       width=width, height=height, persons=[person])

    zone = Zone(id=1, camera=camera, name="West stairwell", ztype="loiter",
                points=[[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]])
    zones = ZoneSet([zone])

    tracks = TrackStore()
    tracks.update(world)
    ctx = CameraContext(camera=camera, world=world, tracks=tracks, zones=zones,
                        flags=CameraFlags(), now=world.ts)
    ctx.compute_memberships()

    candidate = Candidate(
        type="fall",
        message=("Person #7 dropped 31% of their height in under 0.6s, torso "
                 "86 deg from vertical, and has been down 2.4s in West "
                 "stairwell."),
        track_id=7, zone=zone, highlight="critical",
        meta={"drop_body_heights": 0.31, "torso_angle_deg": 86.0,
              "down_seconds": 2.4, "test_alert": True},
    )

    fired = manager.submit([candidate], ctx)
    if not fired:
        print("  The alert was suppressed by a cooldown. Wait a minute and "
              "try again.")
        return 1

    alert = fired[0]
    snapshot = PrivacyRenderer().render(
        world, zones=zones, highlights={7: "critical"},
        hud={"fps": 14.0, "demo": config.demo_mode()})
    manager.attach_snapshots(fired, snapshot)

    print(f"  Event ARJUN-{alert.event_id:05d} written to the database")
    print(f"  Skeleton snapshot rendered ({snapshot.shape[1]}x{snapshot.shape[0]})")

    if not telegram_ready and not whatsapp_ready:
        print("\n  Nothing to send. The alert is on the dashboard timeline.")
        return 0

    print("  Handing it to the notification thread…", end="", flush=True)
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if manager.sent["telegram"] or manager.sent["whatsapp"] \
                or manager.sent["failed"] or alerts_mod.LAST_ERROR["telegram"]:
            break
        print(".", end="", flush=True)
        time.sleep(1.0)
    print()

    ok = True
    print()
    for name, ready in (("Telegram", telegram_ready),
                        ("WhatsApp", whatsapp_ready)):
        key = name.lower()
        if not ready:
            print(f"    skip {name}: not configured")
            continue
        if manager.sent[key]:
            print(f"    ok   {name}: delivered — check your phone")
        else:
            ok = False
            reason = alerts_mod.LAST_ERROR[key] or "no response yet"
            print(f"    FAIL {name}: {reason}")
            print(_telegram_hint(reason) if key == "telegram" else "")

    manager.stop()
    print()
    if ok:
        print("  That is exactly what a real fall will look like: the camera")
        print("  name, the zone, the time, the measurements, and a skeleton —")
        print("  never a photograph of the person.")
    return 0 if ok else 1


def _telegram_hint(reason: str) -> str:
    low = reason.lower()
    if "chat not found" in low:
        return ("         The chat ID is wrong, or you have not messaged the "
                "bot yet.\n         Open the bot in Telegram, press Start, "
                "then re-run setup_alerts.py.")
    if "unauthorized" in low or "401" in low:
        return ("         The bot token is wrong. Re-run setup_alerts.py and "
                "paste it again.")
    if "bot was blocked" in low:
        return "         You have blocked the bot in Telegram. Unblock it."
    if "not configured" in low:
        return "         Run  python setup_alerts.py  to connect Telegram."
    return ("         Check the machine is online, then re-run "
            "setup_alerts.py.")


def _run_check(config) -> int:
    """A dry run: models, cameras, database, codecs, detectors."""
    from backend.vision import model_store

    print("  Checking everything ARJUN needs.\n")
    ok = True

    warning = config.synced_storage_warning()
    if warning:
        print("  Storage")
        print(warning, end="\n\n")

    print("  Pretrained models")
    report = model_store.preflight(verbose=True)
    for key, present in report.items():
        print(f"    {'ok  ' if present else 'MISS'} {key}")
        if not present and key in ("detection", "pose"):
            ok = False

    print("\n  Detection rules")
    from backend.detectors.base import UNAVAILABLE, build_all
    rules = build_all()
    print(f"    ok   {len(rules)} rules loaded")
    for ref, reason in UNAVAILABLE.items():
        ok = False
        print(f"    FAIL {ref}: {reason}")

    print("\n  Database")
    from backend import events_db
    events_db.init()
    report_db = events_db.database_report()
    print(f"    ok   {report_db['path']} ({report_db['events']} events, "
          f"{report_db['zones']} zones)")

    print("\n  Video encoding (for incident clips)")
    import cv2
    import numpy as np
    import tempfile
    codec_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        for fourcc in config.CLIP_CODECS:
            path = os.path.join(tmp, f"probe_{fourcc}.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                     12, (320, 180))
            if writer.isOpened():
                for _ in range(6):
                    writer.write(np.zeros((180, 320, 3), np.uint8))
                writer.release()
                print(f"    ok   {fourcc}"
                      + ("  (browser-playable H.264)" if fourcc == "avc1" else
                         "  (fallback; may not play in a browser)"))
                codec_ok = True
                break
            writer.release()
    if not codec_ok:
        ok = False
        print("    FAIL no usable codec — incident clips will not be written")

    print("\n  Cameras")
    from backend.vision.sources import build_source
    for name, source in config.CAMERA_SOURCES:
        src = build_source(source)
        opened = src.open()
        if opened:
            got, frame = src.read()
            shape = "no frame" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}"
            print(f"    ok   {name:<22} {source}  ({shape})")
        else:
            print(f"    WARN {name:<22} {source}  — {src.last_error}")
        src.release()

    print("\n  Integrations")
    for key, value in config.as_dict()["integrations"].items():
        print(f"    {'on ' if value else 'off'}  {key}"
              + ("" if value else "  (feature degrades, nothing breaks)"))

    print("\n  " + ("Everything essential is in place."
                    if ok else "Something essential is missing — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
