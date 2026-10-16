"""
Where frames come from.

Four kinds of source, behind one small interface:

``webcam``   an integer index, e.g. ``0`` — a laptop camera or a USB camera.
``stream``   an http/https/rtsp URL — DroidCam, IP Webcam, or real CCTV.
``file``     a path to a video file, looped forever (handy for rehearsing).
``sim``      ``sim:atrium`` / ``sim:exam`` — a scripted scene, no camera at all.

The simulated sources are not a shortcut past the real work. They exist for two
honest reasons: a judge's laptop may have no webcam, and nobody should have to
throw themselves on the floor of an auditorium to prove that fall detection
fires. A simulated source hands the pipeline a ready-made world (skeletons,
bags, their motion over time) which then runs through exactly the same
detectors, cooldowns, clip recorder and alert path as a real camera. The UI
labels these feeds "Simulated" in every place they appear.

Every source is resilient by contract: ``open`` returns a bool rather than
raising, ``read`` returns ``(False, None)`` instead of throwing, and the owning
worker thread decides when to reconnect.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config


def classify(source: str) -> str:
    """Work out which kind of source a configuration string describes."""
    s = source.strip()
    if s.lower().startswith("sim:"):
        return "sim"
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return "webcam"
    low = s.lower()
    if low.startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://",
                       "tcp://")):
        return "stream"
    return "file"


class FrameSource:
    """Base class. Subclasses must not raise from ``open``/``read``/``release``."""

    kind = "unknown"
    simulated = False

    def __init__(self, source: str) -> None:
        self.source = source
        self.last_error: str = ""

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    # -- optional extras ---------------------------------------------------- #
    def synthetic_world(self) -> dict[str, Any] | None:
        """Ground truth for simulated sources; ``None`` for real cameras.

        When this returns a world, the pipeline skips the neural networks and
        uses it directly. Real cameras always return ``None``, so there is no
        path by which a real frame can be faked.
        """
        return None

    def describe(self) -> str:
        return f"{self.kind}:{self.source}"


class CaptureSource(FrameSource):
    """Anything OpenCV's :class:`VideoCapture` can open."""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.kind = classify(source)
        self.cap: cv2.VideoCapture | None = None
        self._loop = self.kind == "file"
        self._file_delay = 0.0
        self._last_read = 0.0

    def _backend(self) -> int:
        # On Windows, DirectShow opens a webcam in a fraction of the time MSMF
        # takes and is far less fussy about being reopened after a disconnect.
        if self.kind == "webcam" and os.name == "nt":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def open(self) -> bool:
        self.release()
        try:
            target: int | str = (int(self.source) if self.kind == "webcam"
                                 else self.source)
            cap = cv2.VideoCapture(target, self._backend())
            if not cap.isOpened():
                self.last_error = "device or stream would not open"
                cap.release()
                return False

            # Keep the driver's queue at one frame so we always read something
            # current. Without this, a slow pipeline makes a webcam feel like a
            # delayed replay of the past few seconds.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
            if self.kind == "webcam":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if self.kind == "file":
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                self._file_delay = 1.0 / fps if 1.0 < fps < 121.0 else 1 / 25.0

            ok, probe = cap.read()
            if not ok or probe is None:
                self.last_error = "opened but returned no frames"
                cap.release()
                return False

            self.cap = cap
            self.last_error = ""
            return True
        except (cv2.error, ValueError, OSError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        cap = self.cap
        if cap is None:
            return False, None
        try:
            # A video file read as fast as the CPU allows would play at 400 fps;
            # pace it to its own frame rate so detection timings mean something.
            if self._file_delay:
                wait = self._file_delay - (time.monotonic() - self._last_read)
                if wait > 0:
                    time.sleep(wait)
                self._last_read = time.monotonic()

            ok, frame = cap.read()
            if (not ok or frame is None) and self._loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                self.last_error = "read failed"
                return False, None
            return True, frame
        except cv2.error as exc:
            self.last_error = f"cv2: {exc}"
            return False, None

    def drain(self, max_frames: int = 4) -> None:
        """Throw away frames the driver queued while we were busy.

        Network streams in particular will happily hand back a backlog, and a
        safety system showing three-second-old footage is worse than useless.
        """
        cap = self.cap
        if cap is None or self.kind == "file":
            return
        try:
            for _ in range(max_frames):
                if not cap.grab():
                    break
        except cv2.error:
            pass

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error:
                pass
            self.cap = None


class SimSource(FrameSource):
    """A scripted scene. See :mod:`backend.vision.simscene`."""

    kind = "sim"
    simulated = True

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.scene_name = source.split(":", 1)[1] if ":" in source else "atrium"
        self.scene: Any = None
        self._world: dict[str, Any] | None = None

    def open(self) -> bool:
        from backend.vision import simscene
        try:
            self.scene = simscene.build(self.scene_name)
            self.last_error = ""
            return True
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"scene '{self.scene_name}': {exc}"
            return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.scene is None:
            return False, None
        # The scene advances in real time, so its timers line up with the
        # detector timers that are watching it.
        frame, world = self.scene.step()
        self._world = world
        return True, frame

    def synthetic_world(self) -> dict[str, Any] | None:
        return self._world

    def describe(self) -> str:
        return f"sim:{self.scene_name}"


def build_source(source: str) -> FrameSource:
    kind = classify(source)
    if kind == "sim":
        return SimSource(source)
    if kind == "file":
        p = Path(source)
        if not p.is_absolute():
            p = config.ROOT / source
        return CaptureSource(str(p))
    return CaptureSource(source)
