"""
Panic: several people running the same way at once.

One person sprinting down a corridor is late for a lecture. Five people
sprinting down a corridor *in the same direction* is a reason to look at that
camera right now — a fight, a fire, something falling. The signal is not speed
on its own; it is **coherence**.

So the rule asks three questions:

1.  Who is moving faster than ``run_speed_bh_s`` body heights per second?
    Around 2.2 is a jog; normal walking sits near 1.0. Expressing it in body
    heights is what makes one number work across the whole frame.
2.  Are there at least ``run_min_people`` of them?
3.  Do their headings agree? The spread is measured with circular statistics,
    because the naive version — averaging 350° and 10° to get 180° — would
    conclude that two people running side by side are running at each other.

And then it insists the whole picture holds for half a second, so that one
noisy frame of tracking cannot raise a panic alarm in a building.

Deliberately uses straight-line displacement rather than path length: a crowd
surging one way is exactly the case where "where you started versus where you
are" is the right question.
"""

from __future__ import annotations

import math
from typing import Any

import config
from backend.detectors.base import Candidate, Detector
from backend.detectors.fall import short_id
from backend.state import CameraContext


class PanicMovementDetector(Detector):

    name = "panic"
    title = "Running / panic movement"
    explanation = (
        "Several people moving much faster than walking pace, in the same "
        "direction, at the same moment."
    )

    def __init__(self) -> None:
        super().__init__()
        self._since: float | None = None
        self._fired_at: float = 0.0

    def reset(self) -> None:
        self._since = None
        self._fired_at = 0.0

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()

        runners: list[dict[str, Any]] = []
        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None or track.age < t.run_window_s * 0.8:
                continue
            speed = track.speed(t.run_window_s)
            if speed < t.run_speed_bh_s:
                continue
            heading = track.direction(t.run_window_s)
            if heading is None:
                continue
            runners.append({"id": det.track_id, "speed": speed,
                            "heading": heading})

        if len(runners) < t.run_min_people:
            self._since = None
            return []

        mean, spread = _circular_stats([r["heading"] for r in runners])
        if spread > t.run_direction_spread_deg:
            self._since = None
            return []

        if self._since is None:
            self._since = ctx.now
        held = ctx.now - self._since
        if held < t.run_sustain_s:
            return []
        # One alert per episode; a long chase should not be a hundred of them.
        if ctx.now - self._fired_at < max(10.0, t.cooldown_s):
            return []
        self._fired_at = ctx.now

        fastest = max(runners, key=lambda r: r["speed"])
        return [Candidate(
            type="panic",
            message=(f"{len(runners)} people moving "
                     f"{fastest['speed']:.1f} body heights per second "
                     f"{_compass(mean)} together — possible panic."),
            track_id=fastest["id"], highlight="critical", highlight_for=8.0,
            meta={
                "runners": len(runners),
                "mean_heading_deg": round(mean, 1),
                "heading_spread_deg": round(spread, 1),
                "fastest_body_heights_s": round(fastest["speed"], 2),
                "threshold_speed": t.run_speed_bh_s,
                "track_ids": [short_id(r["id"]) for r in runners],
                "held_seconds": round(held, 2),
            },
        )]


def _circular_stats(headings: list[float]) -> tuple[float, float]:
    """Mean heading and the widest deviation from it, all in degrees.

    Angles have to be averaged as unit vectors. Treating them as plain numbers
    puts the average of 350 and 10 degrees at 180 — pointing backwards.
    """
    if not headings:
        return 0.0, 0.0
    xs = sum(math.cos(math.radians(h)) for h in headings)
    ys = sum(math.sin(math.radians(h)) for h in headings)
    mean = math.degrees(math.atan2(ys, xs))
    spread = max(abs(_wrap(h - mean)) for h in headings)
    return mean, spread


def _wrap(degrees: float) -> float:
    """Fold an angle into -180..180."""
    return (degrees + 180.0) % 360.0 - 180.0


def _compass(heading: float) -> str:
    """A readable direction. Screen axes: y grows downward, so 90 is 'down'."""
    names = ((-22.5, "to the right"), (22.5, "to the right"),
             (67.5, "down and right"), (112.5, "down the frame"),
             (157.5, "down and left"), (180.1, "to the left"))
    h = _wrap(heading)
    if h < -157.5 or h > 157.5:
        return "to the left"
    if h < -112.5:
        return "up and left"
    if h < -67.5:
        return "up the frame"
    if h < -22.5:
        return "up and right"
    for limit, label in names:
        if h <= limit:
            return label
    return "across the frame"
