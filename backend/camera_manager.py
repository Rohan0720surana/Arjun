"""
One thread per camera, and nothing shared that could bring the others down.

Each :class:`CameraWorker` owns its capture source, its own copies of the
neural networks (ByteTrack keeps its state inside the model object, so the
models cannot be shared between streams anyway), its rolling raw buffer and its
detector state. A worker that throws is caught, logged, marked ``error`` and
restarted by its own loop; the manager above it never notices, and the other
cameras never stutter.

The worker also decides *not* to do work. Three small economies matter on a
laptop CPU:

* The raw frame is only JPEG-encoded when somebody with the PIN is actually
  watching the raw view. The rest of the time the raw pixels exist as a numpy
  array in a ring buffer and nothing else.
* Queued frames are dropped rather than processed. A safety system showing
  three-second-old footage is worse than one showing fewer frames per second.
* Capture is paced to ``TARGET_FPS`` so a 60 fps webcam does not burn four
  cores to produce frames nobody looks at.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

import cv2
import numpy as np

import config
from backend.vision.sources import FrameSource, build_source, classify


# --------------------------------------------------------------------------- #
# Per-camera settings that staff can flip from the dashboard
# --------------------------------------------------------------------------- #

@dataclass
class CameraFlags:
    enabled: bool = True
    paused: bool = False
    exam_mode: bool = False
    heatmap: bool = False
    after_hours_override: bool = False   # demo: pretend it is night
    detectors_off: bool = False          # pure viewer, no rules

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class ProcessedFrame:
    """What a processor hands back for one frame."""

    display: np.ndarray                      # the privacy view, BGR
    people: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class Processor(Protocol):
    """The pipeline, from the worker's point of view."""

    def process(self, worker: "CameraWorker", frame: np.ndarray,
                frame_idx: int, ts: float) -> ProcessedFrame: ...

    def reset(self, worker: "CameraWorker") -> None: ...


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #

class CameraWorker(threading.Thread):

    def __init__(
        self,
        name: str,
        source: str,
        processor: Processor | None = None,
        recorder: Any | None = None,
        on_frame: Callable[["CameraWorker", ProcessedFrame], None] | None = None,
    ) -> None:
        super().__init__(name=f"cam::{name}", daemon=True)
        self.cam_name = name
        self.source_str = source
        self.kind = classify(source)
        self.processor = processor
        self.recorder = recorder          # privacy.IncidentRecorder or None
        self.on_frame = on_frame

        self.flags = CameraFlags()
        self.simulated = self.kind == "sim"

        self._src: FrameSource | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Published state, read by the API and the WebSocket hub.
        self.state = "starting"           # starting|live|reconnecting|error|paused
        self.last_error = ""
        self.frame_idx = 0
        self.fps = 0.0
        self.people = 0
        self.started_at = time.time()
        self.last_frame_at = 0.0
        self.width = 0
        self.height = 0
        self.reconnects = 0
        self.meta: dict[str, Any] = {}

        self._display_jpeg: bytes | None = None
        self._raw_jpeg: bytes | None = None
        self._display_frame: np.ndarray | None = None

        # Raw pixels are only encoded while this is in the future.
        self._raw_wanted_until = 0.0

    # -- published state ---------------------------------------------------- #

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.cam_name,
                "source": self._redacted_source(),
                "kind": self.kind,
                "simulated": self.simulated,
                "state": self.state,
                "error": self.last_error,
                "fps": round(self.fps, 1),
                "people": self.people,
                "frames": self.frame_idx,
                "reconnects": self.reconnects,
                "uptime_s": round(time.time() - self.started_at, 1),
                "width": self.width,
                "height": self.height,
                "flags": self.flags.to_dict(),
                "raw_unlocked": time.time() < self._raw_wanted_until,
                "meta": dict(self.meta),
            }

    def _redacted_source(self) -> str:
        """Never show credentials embedded in an RTSP URL on the dashboard."""
        s = self.source_str
        if "@" in s and "//" in s:
            scheme, rest = s.split("//", 1)
            return f"{scheme}//***@{rest.split('@', 1)[1]}"
        return s

    # -- frame access -------------------------------------------------------- #

    def display_jpeg(self) -> bytes | None:
        with self._lock:
            return self._display_jpeg

    def display_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._display_frame is None else self._display_frame.copy()

    def raw_jpeg(self) -> bytes | None:
        """Only ever non-None while a raw view is unlocked."""
        with self._lock:
            if time.time() >= self._raw_wanted_until:
                return None
            return self._raw_jpeg

    def request_raw(self, seconds: float) -> None:
        """Called after a successful PIN check: start encoding raw pixels."""
        with self._lock:
            self._raw_wanted_until = max(self._raw_wanted_until,
                                         time.time() + seconds)

    def revoke_raw(self) -> None:
        with self._lock:
            self._raw_wanted_until = 0.0
            self._raw_jpeg = None

    def raw_unlocked(self) -> bool:
        with self._lock:
            return time.time() < self._raw_wanted_until

    # -- flags --------------------------------------------------------------- #

    def set_flag(self, key: str, value: bool) -> bool:
        if not hasattr(self.flags, key):
            return False
        setattr(self.flags, key, bool(value))
        if key == "detectors_off" and self.processor is not None:
            self.processor.reset(self)
        return True

    def stop(self) -> None:
        self._stop.set()

    # -- the loop ------------------------------------------------------------ #

    def run(self) -> None:
        backoff = config.RECONNECT_DELAY_S
        fail_streak = 0
        # A simulated scene is a demonstration aid. When it shares a machine
        # with a camera that is actually watching a corridor, the corridor
        # wins: measured here, letting the scenes run flat out cost the real
        # camera half its frame rate.
        target = config.SIM_TARGET_FPS if self.simulated else config.TARGET_FPS
        interval = 1.0 / max(target, 1.0)
        next_tick = time.monotonic()
        fps_ema = 0.0
        last_ts = 0.0

        while not self._stop.is_set():
            try:
                # ---- make sure we have a live source --------------------- #
                if self._src is None:
                    self._set_state("reconnecting" if self.reconnects else "starting")
                    src = build_source(self.source_str)
                    if src.open():
                        self._src = src
                        fail_streak = 0
                        backoff = config.RECONNECT_DELAY_S
                        self._set_state("live")
                        if self.processor is not None:
                            self.processor.reset(self)
                    else:
                        self.last_error = src.last_error or "could not open source"
                        src.release()
                        self.reconnects += 1
                        self._set_state("reconnecting")
                        # Back off gently: a camera that is unplugged should not
                        # be hammered once a second for the next three hours.
                        if self._stop.wait(backoff):
                            break
                        backoff = min(backoff * 1.6, 15.0)
                        continue

                if self.flags.paused:
                    self._set_state("paused")
                    if self._stop.wait(0.2):
                        break
                    continue

                # ---- pace ------------------------------------------------- #
                now = time.monotonic()
                if now < next_tick:
                    if self._stop.wait(min(next_tick - now, 0.25)):
                        break
                    continue
                next_tick = max(now, next_tick + interval)

                # ---- read ------------------------------------------------- #
                ok, frame = self._src.read()
                if not ok or frame is None:
                    fail_streak += 1
                    self.last_error = self._src.last_error or "read failed"
                    if fail_streak >= config.READ_FAIL_LIMIT:
                        self._src.release()
                        self._src = None
                        self.reconnects += 1
                        self._set_state("reconnecting")
                        fail_streak = 0
                    else:
                        if self._stop.wait(0.05):
                            break
                    continue

                fail_streak = 0
                if self.state != "live":
                    self._set_state("live")

                frame = self._fit(frame)
                ts = time.time()
                self.frame_idx += 1
                self.height, self.width = frame.shape[:2]

                # ---- the rolling raw buffer: memory only ------------------ #
                if self.recorder is not None:
                    self.recorder.offer(frame, ts)

                # ---- detect, track, render -------------------------------- #
                if self.processor is not None and not self.flags.detectors_off:
                    result = self.processor.process(self, frame, self.frame_idx, ts)
                else:
                    result = self._passthrough(frame)

                self.people = result.people
                self.meta = result.meta

                # ---- encode ----------------------------------------------- #
                disp_jpeg = _encode(result.display, config.JPEG_QUALITY)
                raw_jpeg = None
                if time.time() < self._raw_wanted_until:
                    raw_jpeg = _encode(frame, config.JPEG_QUALITY)

                with self._lock:
                    self._display_frame = result.display
                    self._display_jpeg = disp_jpeg
                    self._raw_jpeg = raw_jpeg
                    self.last_frame_at = ts
                    if last_ts:
                        inst = 1.0 / max(ts - last_ts, 1e-3)
                        fps_ema = inst if not fps_ema else fps_ema * 0.85 + inst * 0.15
                        self.fps = fps_ema
                    last_ts = ts

                if self.on_frame is not None:
                    self.on_frame(self, result)

                # ---- drop whatever queued up while we were thinking ------- #
                drain = getattr(self._src, "drain", None)
                if drain is not None:
                    drain()

            except Exception as exc:                            # noqa: BLE001
                # One camera must never take the process with it.
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._set_state("error")
                print(f"[{self.cam_name}] worker error: {self.last_error}")
                traceback.print_exc()
                if self._src is not None:
                    self._src.release()
                    self._src = None
                self.reconnects += 1
                if self._stop.wait(2.0):
                    break

        if self._src is not None:
            self._src.release()
        self._set_state("stopped")

    # -- helpers -------------------------------------------------------------- #

    def _set_state(self, state: str) -> None:
        with self._lock:
            self.state = state

    @staticmethod
    def _fit(frame: np.ndarray) -> np.ndarray:
        """Resize to the processing width, keeping the aspect ratio."""
        h, w = frame.shape[:2]
        target = config.PROC_WIDTH
        if w == target:
            return frame
        scale = target / float(w)
        # INTER_AREA for shrinking is both faster and cleaner than the default.
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(frame, (target, max(2, int(round(h * scale)))),
                          interpolation=interp)

    @staticmethod
    def _passthrough(frame: np.ndarray) -> ProcessedFrame:
        """No processor attached: show a dimmed frame so it is obviously inert."""
        dim = (frame * 0.35).astype(np.uint8)
        cv2.putText(dim, "detectors off", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 190), 1, cv2.LINE_AA)
        return ProcessedFrame(display=dim, people=0)


def _encode(frame: np.ndarray, quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------- #
# The manager
# --------------------------------------------------------------------------- #

class CameraManager:
    """Owns the workers and is the only thing the API talks to."""

    def __init__(self) -> None:
        self.workers: dict[str, CameraWorker] = {}
        self._order: list[str] = []

    def add(self, name: str, source: str, **kw: Any) -> CameraWorker:
        worker = CameraWorker(name, source, **kw)
        self.workers[name] = worker
        self._order.append(name)
        return worker

    def start_all(self) -> None:
        for name in self._order:
            self.workers[name].start()

    def stop_all(self, timeout: float = 3.0) -> None:
        for w in self.workers.values():
            w.stop()
        deadline = time.time() + timeout
        for w in self.workers.values():
            w.join(max(0.05, deadline - time.time()))

    def get(self, name: str) -> CameraWorker | None:
        return self.workers.get(name)

    def names(self) -> list[str]:
        return list(self._order)

    def status(self) -> list[dict[str, Any]]:
        return [self.workers[n].status() for n in self._order]

    def totals(self) -> dict[str, Any]:
        live = [self.workers[n] for n in self._order
                if self.workers[n].state == "live"]
        return {
            "cameras": len(self._order),
            "online": len(live),
            "people": sum(w.people for w in live),
            "fps": round(sum(w.fps for w in live), 1),
        }
