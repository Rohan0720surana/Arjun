"""
After-hours occupancy, and the empty-room hint.

Two readings of the same signal.

**Somebody is here when nobody should be.** Each camera has opening hours.
Outside them, a person who stays in view for a few seconds raises a warning.
The grace period matters: a security guard walking their round should not set
off an alert every time they pass a lens, so the rule waits before speaking and
then stays quiet about that same person.

**Nobody is here, and it is the middle of the night.** The inverse is worth
saying too. A room that has been empty for a few minutes after closing is a
room whose lights and air conditioning are probably still running. This is an
*occupancy-derived hint*, not a power meter — ARJUN cannot see a light switch,
and the message says so. On a campus with a hundred rooms it is still the
cheapest energy audit anyone will ever install, and it comes free from hardware
that was bought for safety.

Opening hours wrap around midnight correctly: a hall open 07:00–21:00 and a
lab open 20:00–06:00 are both expressed the same way and both work.
"""

from __future__ import annotations

import time

import config
from backend.detectors.base import Candidate, Detector
from backend.detectors.fall import short_id
from backend.state import CameraContext


def parse_clock(value: str, fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        hh, mm = str(value).strip().split(":")
        return max(0, min(23, int(hh))), max(0, min(59, int(mm)))
    except (ValueError, AttributeError):
        return fallback


def minutes_now(ts: float) -> int:
    lt = time.localtime(ts)
    return lt.tm_hour * 60 + lt.tm_min


def is_after_hours(ts: float, open_at: str, close_at: str) -> bool:
    """True when ``ts`` falls outside the open window, midnight-safe."""
    oh, om = parse_clock(open_at, (7, 0))
    ch, cm = parse_clock(close_at, (21, 0))
    start, end = oh * 60 + om, ch * 60 + cm
    now = minutes_now(ts)
    if start == end:
        return False                       # open around the clock
    if start < end:
        return not (start <= now < end)    # ordinary daytime window
    return end <= now < start              # window that spans midnight


class AfterHoursDetector(Detector):

    name = "after_hours"
    title = "After-hours occupancy"
    explanation = (
        "Anyone present outside this camera's opening hours, and the inverse: "
        "a room left empty and lit after closing."
    )

    def __init__(self) -> None:
        super().__init__()
        self._empty_since: float | None = None
        self._empty_reported = 0.0

    def reset(self) -> None:
        self._empty_since = None
        self._empty_reported = 0.0

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        # The override is for demonstrations: it lets a two o'clock audience
        # see the night-time behaviour without changing the clock.
        forced = bool(getattr(ctx.flags, "after_hours_override", False))
        after = forced or is_after_hours(ctx.now, t.after_hours_open,
                                         t.after_hours_close)
        ctx.shared["after_hours"] = after

        if not after:
            self._empty_since = None
            return []

        out: list[Candidate] = []

        # -- somebody is here ------------------------------------------- #
        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None:
                continue
            present = track.held("after_hours", True, ctx.now)
            if present < t.after_hours_grace_s:
                continue
            state = track.scratch("after_hours")
            if state.get("fired"):
                continue
            state["fired"] = True
            zones = ctx.zones_for(det)
            where = f" in {zones[0].name}" if zones else ""
            out.append(Candidate(
                type="after_hours",
                message=(f"Person #{short_id(det.track_id)} present{where} at "
                         f"{time.strftime('%H:%M', time.localtime(ctx.now))}, "
                         f"outside opening hours "
                         f"({t.after_hours_open}–{t.after_hours_close})."),
                track_id=det.track_id, zone=zones[0] if zones else None,
                highlight="warning",
                meta={"present_seconds": round(present, 1),
                      "open_at": t.after_hours_open,
                      "close_at": t.after_hours_close,
                      "forced": forced},
            ))

        # -- nobody is here ---------------------------------------------- #
        if ctx.world.persons:
            self._empty_since = None
            return out

        if self._empty_since is None:
            self._empty_since = ctx.now
        empty_for = ctx.now - self._empty_since
        if (empty_for >= t.after_hours_empty_room_s
                and ctx.now - self._empty_reported >= t.after_hours_empty_room_s):
            self._empty_reported = ctx.now
            out.append(Candidate(
                type="lights_on",
                message=(f"{ctx.camera} has been empty for "
                         f"{_mmss(empty_for)} after closing — lights and air "
                         f"conditioning are probably still on."),
                severity="info",
                meta={"empty_seconds": round(empty_for, 1),
                      "basis": "occupancy only; ARJUN cannot see a light switch"},
            ))
        return out


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}:{seconds % 60:02d}"
