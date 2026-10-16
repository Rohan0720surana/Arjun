"""
The privacy core. Everything else in ARJUN depends on the guarantees here.

There are three pieces, and the order they run in is the whole argument of this
project:

1.  :class:`PrivacyRenderer` throws the pixels away. It is handed a frame and a
    :class:`~backend.world.FrameWorld`, and it draws *only* the world —
    skeletons, boxes, zones — onto a fresh dark canvas. The camera frame is
    never a layer, never a background, never faded underneath. The dashboard
    cannot show a face because the dashboard is never sent one.

2.  :class:`RawRingBuffer` holds the last fifteen seconds of real frames in
    memory, as JPEG bytes in a deque with a fixed length. Every new frame
    pushes an old one out and the old one is garbage. There is no file handle,
    no temp directory, no cache. Pull the power out and the footage is gone.

3.  :class:`IncidentRecorder` is the only thing in the system allowed to write
    pixels to disk, and only when a detector has said something happened. It
    takes ten seconds from the ring buffer, waits five more, and writes one MP4
    into ``clips/``. That file is the sole raw artefact this system produces.

A useful way to state the result: a conventional camera network records
everybody and keeps the recording in case somebody turns out to matter. ARJUN
records nobody and keeps fifteen seconds in volatile memory in case somebody
turns out to need help.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

import config
from backend.vision import skeleton as sk
from backend.world import Detection, FrameWorld
from backend.zone_store import Zone, ZoneSet

FONT = cv2.FONT_HERSHEY_DUPLEX
MONO = cv2.FONT_HERSHEY_SIMPLEX


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    try:
        r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except ValueError:
        return (120, 160, 210)
    return (b, g, r)


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #

class Heatmap:
    """Where people walk, accumulated — and who they were, discarded.

    Each frame adds a soft blob at every person's foot point and multiplies the
    whole accumulator by a decay factor just under 1. After a few minutes the
    result is the shape of how the space is used. It is built from foot points
    alone: there is no way to pull one person's path back out of it, which is
    the difference between a heatmap and a tracking log.
    """

    def __init__(self, width: int, height: int, scale: int = 4) -> None:
        self.scale = max(2, scale)
        self.w = max(8, width // self.scale)
        self.h = max(8, height // self.scale)
        self.acc = np.zeros((self.h, self.w), np.float32)
        self._stamp = self._make_stamp(max(2, config.HEATMAP_RADIUS // self.scale))
        self._colour_cache: tuple[int, int, np.ndarray] | None = None
        self.samples = 0

    @staticmethod
    def _make_stamp(radius: int) -> np.ndarray:
        size = radius * 2 + 1
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        d2 = (xx ** 2 + yy ** 2) / float(max(radius, 1) ** 2)
        return np.clip(1.0 - d2, 0.0, 1.0).astype(np.float32) ** 1.6

    def add(self, points: Iterable[tuple[float, float]]) -> None:
        self.acc *= config.HEATMAP_DECAY
        r = self._stamp.shape[0] // 2
        for px, py in points:
            x, y = int(px) // self.scale, int(py) // self.scale
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(self.w, x + r + 1), min(self.h, y + r + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0, sy0 = x0 - (x - r), y0 - (y - r)
            self.acc[y0:y1, x0:x1] += self._stamp[sy0:sy0 + (y1 - y0),
                                                  sx0:sx0 + (x1 - x0)]
            self.samples += 1
        np.clip(self.acc, 0.0, 60.0, out=self.acc)

    def overlay(self, canvas: np.ndarray) -> None:
        """Composite the accumulator over the canvas.

        All the arithmetic happens at accumulator resolution (a sixteenth of
        the pixels) and the compositing itself is two OpenCV uint8 calls. The
        straightforward float version of this was costing eleven milliseconds
        a frame, which is most of a frame-rate.
        """
        peak = float(self.acc.max())
        if peak < 0.15:
            return
        h, w = canvas.shape[:2]
        norm = np.clip(self.acc / peak, 0.0, 1.0)
        coloured = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)

        a = norm * config.HEATMAP_GAIN
        src_small = (coloured.astype(np.float32) * a[:, :, None]).astype(np.uint8)
        inv_small = ((1.0 - a) * 255.0).astype(np.uint8)
        inv_small = cv2.cvtColor(inv_small, cv2.COLOR_GRAY2BGR)

        src = cv2.resize(src_small, (w, h), interpolation=cv2.INTER_LINEAR)
        inv = cv2.resize(inv_small, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.multiply(canvas, inv, canvas, scale=1.0 / 255.0)
        cv2.add(canvas, src, canvas)

    def reset(self) -> None:
        self.acc[:] = 0.0
        self.samples = 0


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #

class PrivacyRenderer:
    """Draws the anonymous view. Never receives a reason to draw a pixel of input."""

    def __init__(self) -> None:
        self._bg_cache: dict[tuple[int, int], np.ndarray] = {}

    # -- backdrop ------------------------------------------------------------ #

    def _backdrop(self, w: int, h: int) -> np.ndarray:
        key = (w, h)
        cached = self._bg_cache.get(key)
        if cached is not None:
            return cached

        base = np.zeros((h, w, 3), np.uint8)
        base[:] = config.PALETTE["canvas"]

        if config.RENDER_GRID:
            step = max(28, w // 22)
            grid = config.PALETTE["grid"]
            for x in range(step, w, step):
                cv2.line(base, (x, 0), (x, h), grid, 1, cv2.LINE_AA)
            for y in range(step, h, step):
                cv2.line(base, (0, y), (w, y), grid, 1, cv2.LINE_AA)

        if config.RENDER_VIGNETTE:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = w / 2.0, h / 2.0
            d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            v = np.clip(1.0 - 0.42 * np.clip(d - 0.45, 0, None) ** 1.5, 0.45, 1.0)
            base = (base.astype(np.float32) * v[:, :, None]).astype(np.uint8)

        self._bg_cache[key] = base
        return base

    # -- the main entry point -------------------------------------------------- #

    def render(
        self,
        world: FrameWorld,
        zones: ZoneSet | None = None,
        heatmap: Heatmap | None = None,
        highlights: dict[int, str] | None = None,
        zone_counts: dict[int, int] | None = None,
        hud: dict[str, Any] | None = None,
    ) -> np.ndarray:
        w, h = world.width, world.height
        canvas = self._backdrop(w, h).copy()

        if heatmap is not None:
            heatmap.overlay(canvas)

        if zones is not None:
            self._draw_zones(canvas, zones, zone_counts or {})

        highlights = highlights or {}

        # Bloom pass: every bone goes onto its own layer, which is blurred and
        # added back. This is what makes the skeletons glow instead of looking
        # like a wireframe debug view - and it costs about a millisecond.
        glow = np.zeros_like(canvas) if config.RENDER_BLOOM else None

        for obj in world.objects:
            self._draw_object(canvas, obj, highlights.get(obj.track_id))

        for person in world.persons:
            self._draw_person(canvas, glow, person, highlights.get(person.track_id))

        if glow is not None:
            # Blurring at half resolution is visually identical for a soft
            # bloom and about four times cheaper.
            gh, gw = glow.shape[0] // 2, glow.shape[1] // 2
            small = cv2.resize(glow, (gw, gh), interpolation=cv2.INTER_LINEAR)
            small = cv2.GaussianBlur(small, (0, 0), 3.0)
            glow = cv2.resize(small, (glow.shape[1], glow.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(canvas, 1.0, glow, 0.62, 0.0, canvas)

        self._draw_hud(canvas, world, hud or {})
        return canvas

    # -- zones ------------------------------------------------------------------ #

    def _draw_zones(self, canvas: np.ndarray, zones: ZoneSet,
                    counts: dict[int, int]) -> None:
        h, w = canvas.shape[:2]
        polygons = [z for z in zones if not z.is_line and len(z.points) >= 3]
        if polygons:
            fill = canvas.copy()
            for zone in polygons:
                cv2.fillPoly(fill, [zone.polygon(w, h)], _hex_to_bgr(zone.colour))
            cv2.addWeighted(fill, 0.10, canvas, 0.90, 0, canvas)

        for zone in zones:
            colour = _hex_to_bgr(zone.colour)
            if zone.is_line:
                self._draw_line_zone(canvas, zone, colour)
                continue
            poly = zone.polygon(w, h)
            if len(poly) < 3:
                continue
            _dashed_poly(canvas, poly, colour, 1)
            label = zone.name
            n = counts.get(zone.id)
            if n is not None:
                label = f"{label}  {n}"
            cx, cy = zone.centroid(w, h)
            top = int(poly[:, 1].min())
            _chip(canvas, label, (int(cx), max(14, top - 6)), colour,
                  centred=True)

    @staticmethod
    def _draw_line_zone(canvas: np.ndarray, zone: Zone,
                        colour: tuple[int, int, int]) -> None:
        h, w = canvas.shape[:2]
        if len(zone.points) < 2:
            return
        p1 = (int(zone.points[0][0] * w), int(zone.points[0][1] * h))
        p2 = (int(zone.points[1][0] * w), int(zone.points[1][1] * h))
        cv2.line(canvas, p1, p2, colour, 2, cv2.LINE_AA)
        # Little ticks showing which way is "in".
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.arrowedLine(canvas, (mx, my), (int(mx + nx * 16), int(my + ny * 16)),
                        colour, 1, cv2.LINE_AA, tipLength=0.4)
        _chip(canvas, zone.name, (mx, my - 10), colour, centred=True)

    # -- people -------------------------------------------------------------------- #

    def _draw_person(self, canvas: np.ndarray, glow: np.ndarray | None,
                     det: Detection, highlight: str | None) -> None:
        kps = det.keypoints
        colour = config.PALETTE["skeleton"]
        bloom = config.PALETTE["glow"]
        if highlight == "critical":
            colour = bloom = config.PALETTE["critical"]
        elif highlight == "warning":
            colour = bloom = config.PALETTE["warning"]
        elif highlight:
            colour = bloom = config.PALETTE["gold"]

        scale = det.scale()
        thick = max(1, int(round(scale * 0.022)))

        if kps is not None and sk.visible_fraction(kps) > 0.15:
            for a, b, weight in sk.EDGES:
                if not sk.seen(kps, a, b):
                    continue
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                t = max(1, int(round(thick * weight)))
                cv2.line(canvas, pa, pb, colour, t, cv2.LINE_AA)
                if glow is not None:
                    cv2.line(glow, pa, pb, bloom, t + 3, cv2.LINE_AA)

            joint = config.PALETTE["joint"] if not highlight else colour
            for i in sk.DRAW_JOINTS:
                if not sk.seen(kps, i):
                    continue
                cv2.circle(canvas, (int(kps[i][0]), int(kps[i][1])),
                           max(1, thick), joint, -1, cv2.LINE_AA)

            # The head is one empty circle. There is no face to draw: the eye,
            # ear and nose keypoints inform this centre and are never rendered.
            head = det.head()
            if head is not None:
                r = max(3, int(scale * 0.075))
                cv2.circle(canvas, (int(head[0]), int(head[1])), r, colour,
                           max(1, thick), cv2.LINE_AA)
                if glow is not None:
                    cv2.circle(glow, (int(head[0]), int(head[1])), r, bloom,
                               max(1, thick) + 2, cv2.LINE_AA)
        else:
            # No skeleton: show the footprint and extent, nothing more.
            x1, y1, x2, y2 = (int(v) for v in det.box)
            _corner_box(canvas, (x1, y1, x2, y2), colour, 1)

        if highlight:
            fx, fy = det.foot()
            cv2.ellipse(canvas, (int(fx), int(fy)),
                        (max(8, int(scale * 0.22)), max(3, int(scale * 0.07))),
                        0, 0, 360, colour, 1, cv2.LINE_AA)

        label_at = det.head() or sk.center(det.box)
        _tag(canvas, f"#{_short_id(det.track_id)}",
             (int(label_at[0]), int(label_at[1] - max(14, scale * 0.14))),
             colour if highlight else config.PALETTE["gold"])

    @staticmethod
    def _draw_object(canvas: np.ndarray, det: Detection,
                     highlight: str | None) -> None:
        colour = config.PALETTE["object"]
        if highlight == "critical":
            colour = config.PALETTE["critical"]
        elif highlight:
            colour = config.PALETTE["warning"]
        x1, y1, x2, y2 = (int(v) for v in det.box)
        _corner_box(canvas, (x1, y1, x2, y2), colour, 1)
        _tag(canvas, det.label, (x1 + 2, max(10, y1 - 5)), colour, left=True)

    # -- hud ------------------------------------------------------------------------ #

    @staticmethod
    def _draw_hud(canvas: np.ndarray, world: FrameWorld,
                  hud: dict[str, Any]) -> None:
        h, w = canvas.shape[:2]
        gold = config.PALETTE["gold"]
        text = config.PALETTE["text"]

        cv2.rectangle(canvas, (0, 0), (w, 22), (18, 17, 16), -1)
        cv2.line(canvas, (0, 22), (w, 22), (40, 38, 35), 1)
        cv2.putText(canvas, world.camera[:28], (8, 15), FONT, 0.42, text, 1,
                    cv2.LINE_AA)

        right = w - 8
        bits: list[tuple[str, tuple[int, int, int]]] = []
        if world.simulated:
            bits.append(("SIMULATED", (120, 150, 190)))
        if hud.get("exam_mode"):
            bits.append(("EXAM", gold))
        if hud.get("heatmap"):
            bits.append(("HEATMAP", (150, 120, 90)))
        if hud.get("demo"):
            bits.append(("DEMO", gold))
        if hud.get("after_hours"):
            bits.append(("AFTER HOURS", (150, 140, 120)))
        for label, colour in reversed(bits):
            (tw, _), _ = cv2.getTextSize(label, MONO, 0.34, 1)
            right -= tw + 10
            cv2.putText(canvas, label, (right, 15), MONO, 0.34, colour, 1,
                        cv2.LINE_AA)

        fps = hud.get("fps")
        people = len(world.persons)
        foot = f"{people} present"
        if fps:
            foot += f"   {float(fps):.0f} fps"
        cv2.putText(canvas, foot, (8, h - 9), MONO, 0.36,
                    (150, 148, 144), 1, cv2.LINE_AA)

        stamp = time.strftime("%H:%M:%S", time.localtime(world.ts))
        (tw, _), _ = cv2.getTextSize(stamp, MONO, 0.36, 1)
        cv2.putText(canvas, stamp, (w - tw - 8, h - 9), MONO, 0.36,
                    (150, 148, 144), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Small drawing helpers
# --------------------------------------------------------------------------- #

def _short_id(track_id: int) -> int:
    """Simulated ids are offset by 900000; show the human-sized number."""
    return track_id - 900_000 if track_id >= 900_000 else track_id


def _corner_box(img: np.ndarray, box: tuple[int, int, int, int],
                colour: tuple[int, int, int], thickness: int = 1) -> None:
    """An L-cornered box — lighter on the eye than a full rectangle."""
    x1, y1, x2, y2 = box
    k = max(6, min(16, (x2 - x1) // 4, (y2 - y1) // 4))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * k, py), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * k), colour, thickness, cv2.LINE_AA)


#: Dashed outlines are recomputed only when a zone's shape changes. Zones are
#: static almost all the time, and the segment maths was the single most
#: expensive thing in the renderer — a polygon with a 1000px perimeter is a
#: hundred individual antialiased line calls, per zone, per frame.
_DASH_CACHE: dict[tuple, list[np.ndarray]] = {}


def _dash_segments(poly: np.ndarray, dash: int) -> list[np.ndarray]:
    key = (poly.tobytes(), poly.shape, dash)
    cached = _DASH_CACHE.get(key)
    if cached is not None:
        return cached

    segments: list[np.ndarray] = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i].astype(np.float64)
        p2 = poly[(i + 1) % n].astype(np.float64)
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist < 1:
            continue
        steps = max(1, int(dist // dash))
        for s in range(0, steps, 2):
            a = s / steps
            b = min(1.0, (s + 1) / steps)
            segments.append(np.array([p1 + (p2 - p1) * a,
                                      p1 + (p2 - p1) * b], dtype=np.int32))
    if len(_DASH_CACHE) > 256:          # a zone editor session, bounded
        _DASH_CACHE.clear()
    _DASH_CACHE[key] = segments
    return segments


def _dashed_poly(img: np.ndarray, poly: np.ndarray,
                 colour: tuple[int, int, int], thickness: int = 1,
                 dash: int = 9) -> None:
    segments = _dash_segments(poly, dash)
    if segments:
        # One call into OpenCV for the whole outline rather than one per dash.
        cv2.polylines(img, segments, False, colour, thickness, cv2.LINE_AA)


def _tag(img: np.ndarray, text: str, at: tuple[int, int],
         colour: tuple[int, int, int], left: bool = False) -> None:
    (tw, th), _ = cv2.getTextSize(text, MONO, 0.36, 1)
    x = at[0] if left else at[0] - tw // 2
    y = max(th + 2, at[1])
    cv2.putText(img, text, (x, y), MONO, 0.36, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), MONO, 0.36, colour, 1, cv2.LINE_AA)


def _chip(img: np.ndarray, text: str, at: tuple[int, int],
          colour: tuple[int, int, int], centred: bool = False) -> None:
    (tw, th), _ = cv2.getTextSize(text, MONO, 0.34, 1)
    x = at[0] - tw // 2 if centred else at[0]
    y = at[1]
    pad = 4
    x = max(2, min(x, img.shape[1] - tw - pad * 2 - 2))
    # Blend only the chip's own rectangle. Copying the whole frame per label
    # was costing more than every skeleton in the scene put together.
    x0, y0 = max(0, x - pad), max(0, y - th - pad)
    x1, y1 = min(img.shape[1], x + tw + pad), min(img.shape[0], y + pad)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * 0.35 + 24.0).astype(np.uint8)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                  colour, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y - 1), MONO, 0.34, colour, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# The rolling raw buffer
# --------------------------------------------------------------------------- #

class RawRingBuffer:
    """The last N seconds of real frames, in memory, as JPEG bytes.

    JPEG rather than raw arrays for a practical reason: fifteen seconds of
    640x360 BGR is 155 MB per camera, which is not a buffer, it is a problem.
    Encoded it is around 10 MB, and the encode costs a couple of milliseconds
    on a frame we were about to encode anyway.
    """

    def __init__(self, seconds: float, fps: float) -> None:
        self.seconds = float(seconds)
        self.capacity = max(8, int(seconds * fps) + 8)
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self.bytes_held = 0

    def offer(self, frame: np.ndarray, ts: float) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),
                            int(config.BUFFER_JPEG_QUALITY)])
        if not ok:
            return
        with self._lock:
            self._frames.append((ts, buf.tobytes()))
            # Trim by age as well as by count. The deque's maxlen is sized for
            # the target frame rate, so a camera running slower than that (a
            # busy CPU, a stuttering stream) would hold the same number of
            # frames spread over a longer span — and the system would be
            # keeping more seconds of footage than it tells people it keeps.
            # The claim on the privacy panel has to be the true one.
            horizon = ts - self.seconds
            while len(self._frames) > 2 and self._frames[0][0] < horizon:
                self._frames.popleft()
            self.bytes_held = sum(len(b) for _, b in self._frames)

    def window(self, start_ts: float, end_ts: float) -> list[tuple[float, bytes]]:
        with self._lock:
            return [(t, b) for t, b in self._frames if start_ts <= t <= end_ts]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._frames)
            span = (self._frames[-1][0] - self._frames[0][0]) if n > 1 else 0.0
            return {"frames": n, "seconds": round(span, 1),
                    "megabytes": round(self.bytes_held / 1e6, 1),
                    "on_disk": False}

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self.bytes_held = 0


# --------------------------------------------------------------------------- #
# Incident clips
# --------------------------------------------------------------------------- #

@dataclass
class PendingClip:
    key: str
    event_ref: str
    started: float
    end_at: float
    pre: list[tuple[float, bytes]]
    post: list[tuple[float, bytes]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class IncidentRecorder:
    """Owns one camera's ring buffer and writes clips when something happens.

    This is the only writer of raw pixels in the project. The path it writes to
    is fixed at ``config.CLIPS_DIR`` and the filename is derived from the
    camera and the event, so there is exactly one place to look to audit what
    exists.
    """

    def __init__(self, camera: str,
                 on_ready: Callable[[str, Path, dict[str, Any]], None] | None = None
                 ) -> None:
        self.camera = camera
        self.buffer = RawRingBuffer(config.RAW_BUFFER_SECONDS, config.TARGET_FPS)
        self.on_ready = on_ready
        self._pending: list[PendingClip] = []
        self._lock = threading.Lock()
        self.written = 0
        self.last_error = ""

    # -- feeding ---------------------------------------------------------------- #

    def offer(self, frame: np.ndarray, ts: float) -> None:
        self.buffer.offer(frame, ts)
        ready: list[PendingClip] = []
        with self._lock:
            if self._pending:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),
                                    int(config.BUFFER_JPEG_QUALITY)])
                blob = buf.tobytes() if ok else None
                for clip in list(self._pending):
                    if blob is not None and ts <= clip.end_at:
                        clip.post.append((ts, blob))
                    if ts >= clip.end_at:
                        self._pending.remove(clip)
                        ready.append(clip)
        for clip in ready:
            threading.Thread(target=self._write, args=(clip,),
                             name=f"clip::{self.camera}", daemon=True).start()

    # -- triggering --------------------------------------------------------------- #

    def trigger(self, key: str, event_ref: str,
                meta: dict[str, Any] | None = None) -> bool:
        """Start a clip. Returns False if one is already running for this key."""
        now = time.time()
        with self._lock:
            if any(c.key == key for c in self._pending):
                return False
            pre = self.buffer.window(now - config.CLIP_PRE_SECONDS, now)
            self._pending.append(PendingClip(
                key=key, event_ref=event_ref, started=now,
                end_at=now + config.CLIP_POST_SECONDS, pre=pre,
                meta=dict(meta or {}),
            ))
        return True

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- writing ------------------------------------------------------------------- #

    def _write(self, clip: PendingClip) -> None:
        frames = clip.pre + clip.post
        if len(frames) < 4:
            self.last_error = "not enough buffered frames to write a clip"
            return
        frames.sort(key=lambda f: f[0])

        safe_cam = "".join(ch if ch.isalnum() else "-"
                           for ch in self.camera).strip("-").lower()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clip.started))
        # The event id is part of the name, not decoration: two people falling
        # on the same camera in the same second are two incidents, and without
        # it they raced for one filename — which also made the second writer's
        # H.264 encoder fail to open and silently drop the clip to mp4v.
        name = f"{stamp}_{safe_cam}_{clip.key.split(':')[0]}_e{clip.event_ref}.mp4"
        path = config.CLIPS_DIR / name

        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8),
                             cv2.IMREAD_COLOR)
        if first is None:
            self.last_error = "could not decode the buffered frames"
            return
        h, w = first.shape[:2]

        span = max(0.4, frames[-1][0] - frames[0][0])
        fps = max(4.0, min(30.0, len(frames) / span))

        writer, codec = None, ""
        for fourcc in config.CLIP_CODECS:
            candidate = cv2.VideoWriter(str(path),
                                        cv2.VideoWriter_fourcc(*fourcc),
                                        fps, (w, h))
            if candidate.isOpened():
                writer, codec = candidate, fourcc
                break
            candidate.release()
        if writer is None:
            self.last_error = "no usable video codec on this machine"
            return

        trigger_ts = clip.started
        try:
            for ts, blob in frames:
                img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))
                _clip_overlay(img, ts, trigger_ts, clip)
                writer.write(img)
        finally:
            writer.release()

        self.written += 1
        if self.on_ready is not None:
            meta = dict(clip.meta)
            meta.update({"codec": codec, "fps": round(fps, 1),
                         "frames": len(frames),
                         "seconds": round(span, 1),
                         "size_bytes": path.stat().st_size if path.exists() else 0})
            try:
                self.on_ready(clip.event_ref, path, meta)
            except Exception as exc:                        # noqa: BLE001
                self.last_error = f"clip callback: {exc}"


def _clip_overlay(img: np.ndarray, ts: float, trigger_ts: float,
                  clip: PendingClip) -> None:
    """Burn provenance into the clip itself.

    An incident clip is evidence, and evidence that cannot say where it came
    from is worth very little. Each frame carries the camera, the wall-clock
    time, the offset from the trigger and the incident type — and, for a
    simulated feed, an unmissable marker.
    """
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 22), (w, h), (12, 12, 12), -1)
    offset = ts - trigger_ts
    sign = "+" if offset >= 0 else "-"
    left = (f"{clip.meta.get('camera', '')}   "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}   "
            f"T{sign}{abs(offset):04.1f}s")
    cv2.putText(img, left, (8, h - 7), MONO, 0.4, (190, 190, 195), 1, cv2.LINE_AA)

    right = str(clip.meta.get("title", clip.key.split(":")[0]))
    (tw, _), _ = cv2.getTextSize(right, MONO, 0.4, 1)
    colour = (80, 80, 230) if clip.meta.get("severity") == "critical" else (
        90, 170, 220)
    cv2.putText(img, right, (w - tw - 8, h - 7), MONO, 0.4, colour, 1, cv2.LINE_AA)

    if clip.meta.get("simulated"):
        cv2.putText(img, "SIMULATED", (8, 18), FONT, 0.5, (20, 20, 20), 3,
                    cv2.LINE_AA)
        cv2.putText(img, "SIMULATED", (8, 18), FONT, 0.5, (150, 160, 180), 1,
                    cv2.LINE_AA)

    # A red marker on the frames at and just after the trigger instant.
    if -0.15 <= offset <= 0.6:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 60, 220), 2)


def prune_clips(retention_days: int | None = None,
                max_files: int | None = None) -> dict[str, Any]:
    """Delete clips past their retention window. Runs at startup.

    Data you no longer need is data you should no longer have. The defaults
    keep a week, or sixty clips, whichever bites first.
    """
    days = config.CLIP_RETENTION_DAYS if retention_days is None else retention_days
    cap = config.MAX_CLIPS_ON_DISK if max_files is None else max_files
    removed, freed = 0, 0
    try:
        clips = sorted(config.CLIPS_DIR.glob("*.mp4"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return {"removed": 0, "freed_bytes": 0}

    cutoff = time.time() - days * 86400
    for i, path in enumerate(clips):
        try:
            too_old = path.stat().st_mtime < cutoff
            too_many = i >= cap
            if too_old or too_many:
                freed += path.stat().st_size
                path.unlink()
                removed += 1
        except OSError:
            continue
    return {"removed": removed, "freed_bytes": freed}


def storage_report() -> dict[str, Any]:
    """What actually exists on disk, for the privacy panel.

    The headline number on the dashboard — hours of continuous footage stored —
    is always zero, and it is zero because nothing in the codebase can write
    continuous footage, not because a cleanup job ran.
    """
    clips = list(config.CLIPS_DIR.glob("*.mp4"))
    total = sum(p.stat().st_size for p in clips if p.exists())
    snaps = list(config.SNAPSHOT_DIR.glob("*.jpg"))
    return {
        "continuous_hours": 0.0,
        "incident_clips": len(clips),
        "clip_bytes": total,
        "clip_megabytes": round(total / 1e6, 1),
        "skeleton_snapshots": len(snaps),
        "retention_days": config.CLIP_RETENTION_DAYS,
        "clips_dir": str(config.CLIPS_DIR),
    }
