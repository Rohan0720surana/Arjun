"""
Memory: what each track was doing a moment ago, and the context a rule sees.

A detection on its own says almost nothing. "A person is at (320, 240), lying
horizontally" could be someone who has collapsed or someone doing push-ups in
the gym; the difference is entirely in what happened in the two seconds before.
So every rule in this project is written against :class:`TrackState`, a short
rolling history per track, rather than against a single frame.

Three deliberate properties:

**The history is short and it expires.** ``config.ACTIVE.history_seconds`` of
samples, and a track that has not been seen for ``track_memory_s`` is deleted
outright. There is no long-term store, nothing accumulates across a day, and
nothing survives a restart. A track id is a handle for the next few seconds of
a person's movement and then it is gone.

**Nothing identifying is kept.** A sample is six numbers about geometry. There
is no appearance descriptor, no crop, no embedding, so two visits by the same
person are two unrelated tracks — which is a limitation we accept gladly,
because the alternative is re-identification.

**Speeds are scale-free.** :meth:`TrackState.speed` returns body heights per
second, not pixels per second, so one threshold is right for a person near the
lens and a person at the far end of a corridor.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import config
from backend.world import Detection, FrameWorld
from backend.zone_store import Zone, ZoneSet


@dataclass(frozen=True)
class Sample:
    """One frame's worth of geometry for one track."""

    ts: float
    foot: tuple[float, float]
    center: tuple[float, float]
    hip_y: float | None
    scale: float
    aspect: float
    torso: float | None


class TrackState:
    """The rolling history of one tracked person or object."""

    __slots__ = ("track_id", "label", "first_seen", "last_seen", "samples",
                 "data", "zones", "zone_since", "zone_left_at", "frames",
                 "simulated")

    def __init__(self, track_id: int, label: str, ts: float,
                 simulated: bool = False) -> None:
        self.track_id = track_id
        self.label = label
        self.first_seen = ts
        self.last_seen = ts
        self.simulated = simulated
        self.samples: deque[Sample] = deque(maxlen=256)
        #: Scratch space for detectors, namespaced by detector name. Keeping
        #: rule state here (rather than in the detector) means it disappears
        #: with the track, so a rule can never leak state between people.
        self.data: dict[str, Any] = {}
        self.zones: set[int] = set()
        self.zone_since: dict[int, float] = {}
        self.zone_left_at: dict[int, float] = {}
        self.frames = 0

    # -- updating ------------------------------------------------------------ #

    def observe(self, det: Detection, ts: float) -> None:
        hip = det.hip()
        self.samples.append(Sample(
            ts=ts, foot=det.foot(), center=det.center(),
            hip_y=None if hip is None else hip[1],
            scale=det.scale(), aspect=det.aspect(), torso=det.torso_angle(),
        ))
        self.last_seen = ts
        self.frames += 1
        horizon = ts - config.thresholds().history_seconds
        while len(self.samples) > 2 and self.samples[0].ts < horizon:
            self.samples.popleft()

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def latest(self) -> Sample | None:
        return self.samples[-1] if self.samples else None

    @property
    def scale(self) -> float:
        s = self.latest
        return s.scale if s else 1.0

    # -- queries the rules are written in -------------------------------------- #

    def since(self, seconds: float) -> list[Sample]:
        if not self.samples:
            return []
        cutoff = self.samples[-1].ts - seconds
        return [s for s in self.samples if s.ts >= cutoff]

    def displacement(self, window_s: float) -> float:
        """Straight-line distance moved in the window, in body heights."""
        window = self.since(window_s)
        if len(window) < 2:
            return 0.0
        a, b = window[0], window[-1]
        scale = max(1.0, (a.scale + b.scale) / 2.0)
        return float(math.hypot(b.foot[0] - a.foot[0],
                                b.foot[1] - a.foot[1]) / scale)

    def speed(self, window_s: float | None = None) -> float:
        """Average speed over the window, in body heights per second."""
        w = window_s if window_s is not None else config.thresholds().run_window_s
        window = self.since(w)
        if len(window) < 2:
            return 0.0
        dt = window[-1].ts - window[0].ts
        if dt <= 1e-3:
            return 0.0
        return self.displacement(w) / dt

    def direction(self, window_s: float | None = None) -> float | None:
        """Heading in degrees (0 = right, 90 = down), or None if barely moving."""
        w = window_s if window_s is not None else config.thresholds().run_window_s
        window = self.since(w)
        if len(window) < 2:
            return None
        a, b = window[0], window[-1]
        dx, dy = b.foot[0] - a.foot[0], b.foot[1] - a.foot[1]
        if math.hypot(dx, dy) < max(2.0, a.scale * 0.05):
            return None
        return float(math.degrees(math.atan2(dy, dx)))

    def vertical_drop(self, window_s: float) -> float:
        """How far the hip fell inside the window, in body heights.

        Measured from the highest hip position in the window to the current
        one, so a stumble-then-collapse is caught even if the fall is not
        monotonic. Falls back to the box centre when the hips are not visible,
        because a hidden hip is common exactly when someone is folding up.
        """
        window = self.since(window_s)
        if len(window) < 2:
            return 0.0
        def value(s: Sample) -> float:
            return s.hip_y if s.hip_y is not None else s.center[1]
        peak = min(value(s) for s in window)          # smallest y = highest up
        now = value(window[-1])
        scale = max(1.0, window[-1].scale)
        return float((now - peak) / scale)

    def held(self, key: str, condition: bool, ts: float) -> float:
        """How long ``condition`` has been continuously true, in seconds.

        The workhorse of the time-based rules: ``if t.held("down", lying, now)
        > 2.0``. Resets to zero the moment the condition lapses.
        """
        store = self.data.setdefault("_held", {})
        if not condition:
            store.pop(key, None)
            return 0.0
        start = store.setdefault(key, ts)
        return max(0.0, ts - start)

    def held_reset(self, key: str) -> None:
        self.data.setdefault("_held", {}).pop(key, None)

    def scratch(self, name: str) -> dict[str, Any]:
        return self.data.setdefault(name, {})


class TrackStore:
    """All live tracks on one camera."""

    def __init__(self) -> None:
        self.tracks: dict[int, TrackState] = {}
        self.vanished: dict[int, float] = {}

    def update(self, world: FrameWorld) -> None:
        ts = world.ts
        seen: set[int] = set()
        for det in world.everything:
            if det.track_id < 0:
                continue                      # untracked detection, no history
            seen.add(det.track_id)
            st = self.tracks.get(det.track_id)
            if st is None:
                st = TrackState(det.track_id, det.label, ts, world.simulated)
                self.tracks[det.track_id] = st
            st.label = det.label
            st.observe(det, ts)

        memory = config.thresholds().track_memory_s
        for tid, st in list(self.tracks.items()):
            if tid in seen:
                self.vanished.pop(tid, None)
                continue
            if ts - st.last_seen > memory:
                del self.tracks[tid]
                self.vanished[tid] = ts
        # Keep the tombstones briefly so a rule can tell "gone" from "new".
        for tid, when in list(self.vanished.items()):
            if ts - when > 60.0:
                del self.vanished[tid]

    def get(self, track_id: int) -> TrackState | None:
        return self.tracks.get(track_id)

    def of_label(self, label: str) -> list[TrackState]:
        return [t for t in self.tracks.values() if t.label == label]

    def reset(self) -> None:
        self.tracks.clear()
        self.vanished.clear()

    def __len__(self) -> int:
        return len(self.tracks)


@dataclass
class CameraContext:
    """Everything a detector is allowed to look at.

    Deliberately does not carry the frame. A rule that cannot reach the pixels
    cannot quietly start depending on them, which is what keeps the privacy
    guarantee a property of the architecture rather than a promise.
    """

    camera: str
    world: FrameWorld
    tracks: TrackStore
    zones: ZoneSet
    flags: Any                                   # camera_manager.CameraFlags
    now: float
    simulated: bool = False
    memberships: dict[int, list[Zone]] = field(default_factory=dict)
    zone_counts: dict[int, int] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)

    # -- geometry helpers ------------------------------------------------------ #

    @property
    def width(self) -> int:
        return self.world.width

    @property
    def height(self) -> int:
        return self.world.height

    def zones_for(self, det: Detection) -> list[Zone]:
        return self.memberships.get(det.track_id, [])

    def in_zone(self, det: Detection, zone: Zone) -> bool:
        return zone in self.memberships.get(det.track_id, [])

    def track_of(self, det: Detection) -> TrackState | None:
        return self.tracks.get(det.track_id)

    def compute_memberships(self) -> None:
        """Which polygon zones each detection's ground point falls inside.

        Done once per frame and cached, because point-in-polygon for every
        detection against every zone on every rule adds up quickly.
        """
        self.memberships.clear()
        self.zone_counts.clear()
        polygons = self.zones.polygons()
        if not polygons:
            return
        w, h = self.world.width, self.world.height
        for zone in polygons:
            self.zone_counts[zone.id] = 0
        for det in self.world.everything:
            point = det.foot() if det.is_person else det.center()
            inside = [z for z in polygons if z.contains(point, w, h)]
            if inside:
                self.memberships[det.track_id] = inside
                if det.is_person:
                    for z in inside:
                        self.zone_counts[z.id] += 1

    def update_zone_timers(self) -> None:
        """Maintain per-track "how long inside this zone" clocks.

        A short exit does not reset the clock: someone pacing in and out of a
        doorway is still loitering, and a tracker that drops a box for two
        frames should not grant an amnesty.
        """
        grace = config.thresholds().zone_exit_grace_s
        for det in self.world.persons:
            st = self.tracks.get(det.track_id)
            if st is None:
                continue
            current = {z.id for z in self.memberships.get(det.track_id, [])}
            for zid in current:
                if zid not in st.zone_since:
                    left = st.zone_left_at.pop(zid, None)
                    if left is not None and self.now - left <= grace:
                        # Resume the old clock rather than starting a new one.
                        st.zone_since[zid] = st.data.get("_zone_resume", {}).get(
                            zid, self.now)
                    else:
                        st.zone_since[zid] = self.now
                st.data.setdefault("_zone_resume", {})[zid] = st.zone_since[zid]
            for zid in list(st.zone_since):
                if zid in current:
                    st.zone_left_at.pop(zid, None)
                    continue
                # Stamp the moment of leaving once, then let the grace run.
                left = st.zone_left_at.setdefault(zid, self.now)
                if self.now - left > grace:
                    st.zone_since.pop(zid, None)
                    st.data.get("_zone_resume", {}).pop(zid, None)
            st.zones = current

    def dwell(self, det: Detection, zone: Zone) -> float:
        st = self.tracks.get(det.track_id)
        if st is None:
            return 0.0
        started = st.zone_since.get(zone.id)
        return 0.0 if started is None else max(0.0, self.now - started)


def now() -> float:
    return time.time()
