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

    host, port = config.HOST, config.PORT
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    if config.OPEN_BROWSER and not args.reload:
        _open_browser_later(f"http://{shown}:{port}")

    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=args.reload,
                log_level="warning", access_log=False)
    return 0


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
