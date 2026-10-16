"""
Pretrained weight management.

The repository ships no model files — a clone is a few hundred kilobytes. The
four weight files ARJUN needs (about 30 MB in total) are fetched into
``models/`` the first time they are asked for and reused forever after.

Everything in here is deliberately forgiving. If the machine is offline and a
weight is missing, we raise :class:`ModelUnavailable` with a message that tells
the user exactly which file to drop into ``models/`` by hand. The caller then
disables that one capability and the rest of the system carries on — a campus
with no internet should still get fall detection from a cached model.
"""

from __future__ import annotations

import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import config

_LOCK = threading.Lock()

#: Ultralytics publishes its weights as GitHub release assets.
_YOLO_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

#: Google publishes MediaPipe task bundles on their model CDN.
_MP_URLS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
}

#: A sanity floor — a truncated download is worse than no download, because it
#: fails later and more confusingly.
_MIN_BYTES = 200_000


class ModelUnavailable(RuntimeError):
    """A weight file is neither on disk nor downloadable."""


def _download(url: str, dest: Path, label: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  · fetching {label} ...", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 18)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        print(" failed")
        raise ModelUnavailable(
            f"could not download {label}.\n"
            f"    Put the file in {dest.parent} by hand and restart:\n"
            f"    {url}\n"
            f"    (reason: {type(exc).__name__}: {exc})"
        ) from exc

    size = tmp.stat().st_size
    if size < _MIN_BYTES:
        tmp.unlink(missing_ok=True)
        print(" truncated")
        raise ModelUnavailable(
            f"{label} downloaded only {size} bytes, which cannot be right. "
            "Check the network and try again."
        )
    tmp.replace(dest)
    print(f" ok ({size / 1e6:.1f} MB)")


def _ensure(filename: str, url: str) -> Path:
    dest = config.MODELS_DIR / filename
    with _LOCK:
        if dest.exists() and dest.stat().st_size >= _MIN_BYTES:
            return dest
        if dest.exists():          # a half-written file from a killed run
            dest.unlink(missing_ok=True)
        _download(url, dest, filename)
    return dest


def ensure_yolo(weights: str) -> Path:
    """Return a local path for a YOLOv8 ``.pt`` file, downloading if needed.

    An absolute or relative path that already exists is passed straight
    through, so you can point ``YOLO_DET_WEIGHTS`` at your own fine-tuned
    model without touching this code.
    """
    given = Path(weights)
    if given.exists():
        return given
    name = given.name
    local = config.MODELS_DIR / name
    if local.exists() and local.stat().st_size >= _MIN_BYTES:
        return local
    return _ensure(name, _YOLO_BASE + name)


def ensure_mp_task(filename: str) -> Path:
    """Return a local path for a MediaPipe ``.task`` bundle."""
    if filename not in _MP_URLS:
        raise ModelUnavailable(f"unknown MediaPipe task bundle: {filename}")
    return _ensure(filename, _MP_URLS[filename])


def preflight(verbose: bool = True) -> dict[str, bool]:
    """Fetch everything up front so the first frame is not stalled by a download.

    Returns a map of capability -> available, and never raises: a missing
    weight disables one feature rather than the application.
    """
    report: dict[str, bool] = {}

    def attempt(key: str, fn, *args) -> None:
        try:
            fn(*args)
            report[key] = True
        except ModelUnavailable as exc:
            report[key] = False
            if verbose:
                print(f"  ! {key} unavailable: {exc}")

    if verbose:
        print("Checking pretrained models")
    attempt("detection", ensure_yolo, config.YOLO_DET_WEIGHTS)
    if config.POSE_BACKEND == "yolo":
        attempt("pose", ensure_yolo, config.YOLO_POSE_WEIGHTS)
    elif config.POSE_BACKEND == "mediapipe":
        attempt("pose", ensure_mp_task, config.MP_POSE_TASK)
    else:
        report["pose"] = False
    attempt("hands", ensure_mp_task, config.MP_HAND_TASK)
    attempt("face", ensure_mp_task, config.MP_FACE_TASK)
    return report
