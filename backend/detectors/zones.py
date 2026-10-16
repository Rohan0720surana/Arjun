"""
Zone rules: restricted entry, loitering, and crowd density.

All three share one piece of geometry — the person's **foot point**, from
:func:`backend.vision.skeleton.foot_point`. Using the feet rather than the
centre of the bounding box is not fussiness. A tall person leaning over a
restricted line would trip a centre-point test while both feet are safely
outside it, and someone standing in a doorway would be counted as inside a room
they have not entered. Zones are drawn on the floor, so they should be tested
against the part of the person that is on the floor.

The three rules differ in what they do with time:

* **Restricted** is about an instant — crossing the boundary — with a short
  grace period so a foot clipping the edge of a polygon is not an incident.
* **Loitering** is about duration, and forgives brief exits: someone pacing in
  and out of a doorway is still waiting there, and a tracker that drops a box
  for two frames should not grant an amnesty.
* **Crowd** is about a population rather than a person, and insists the number
  holds for a couple of seconds so that a group walking through a hall is not
  mistaken for a crowd gathering in it.
"""

from __future__ import annotations

from typing import Any

import config
from backend.detectors.base import Candidate, Detector
from backend.detectors.fall import short_id
from backend.state import CameraContext
from backend.zone_store import Zone


class RestrictedZoneDetector(Detector):

    name = "restricted"
    title = "Restricted zones"
    requires_zones = ("restricted",)
    explanation = (
        "A person's feet enter a zone marked restricted and stay inside it "
        "for longer than a brief clip of the boundary."
    )

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []

        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None:
                continue
            state = track.scratch("restricted")
            inside_now = {z.id for z in ctx.zones_for(det)
                          if z.ztype == "restricted"}

            for zone in ctx.zones.of("restricted"):
                if zone.id not in inside_now:
                    # Leaving re-arms the zone for this person, so a second
                    # deliberate entry is reported as a second incident.
                    state.pop(zone.id, None)
                    continue
                dwell = ctx.dwell(det, zone)
                if dwell < zone.param("restricted_grace_s", t.restricted_grace_s):
                    continue
                if state.get(zone.id):
                    continue
                state[zone.id] = True
                out.append(Candidate(
                    type="restricted",
                    message=(f"Person #{short_id(det.track_id)} entered "
                             f"{zone.name}, a restricted area."),
                    track_id=det.track_id, zone=zone, highlight="warning",
                    meta={"dwell_seconds": round(dwell, 1),
                          "foot_point": [round(v) for v in det.foot()]},
                ))
        return out


class LoiterDetector(Detector):

    name = "loiter"
    title = "Loitering"
    requires_zones = ("loiter",)
    explanation = (
        "A person stays inside a loiter-watch zone for longer than the zone's "
        "patience, brief exits forgiven."
    )

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []

        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None:
                continue
            state = track.scratch("loiter")

            for zone in ctx.zones.of("loiter"):
                if not ctx.in_zone(det, zone):
                    continue
                limit = float(zone.param("loiter_seconds", t.loiter_seconds))
                dwell = ctx.dwell(det, zone)
                if dwell < limit:
                    continue
                # Alert once per stay, then again only if they are still there
                # a full period later — a person who settles in for an hour
                # should produce a handful of reminders, not thousands.
                last = state.get(zone.id, 0.0)
                if last and ctx.now - last < limit:
                    continue
                state[zone.id] = ctx.now
                moved = track.displacement(min(dwell, 30.0))
                out.append(Candidate(
                    type="loiter",
                    message=(f"Person #{short_id(det.track_id)} has been in "
                             f"{zone.name} for {_mmss(dwell)}, moving "
                             f"{moved:.1f} body-widths in that time."),
                    track_id=det.track_id, zone=zone, highlight="warning",
                    dedupe=f"n{int(dwell // max(limit, 1))}",
                    meta={"dwell_seconds": round(dwell, 1),
                          "threshold_seconds": limit,
                          "displacement_body_heights": round(moved, 2)},
                ))
        return out


class CrowdDetector(Detector):

    name = "crowd"
    title = "Crowd density"
    requires_zones = ("crowd",)
    explanation = (
        "More people inside a crowd zone than its limit allows, held for a "
        "couple of seconds so a group passing through is not a gathering."
    )

    def __init__(self) -> None:
        super().__init__()
        #: Per-zone, not per-track: this rule is about a population.
        self._state: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []

        for zone in ctx.zones.of("crowd"):
            count = ctx.zone_counts.get(zone.id, 0)
            warn = int(zone.param("crowd_warn", t.crowd_warn))
            crit = int(zone.param("crowd_critical", t.crowd_critical))
            state = self._state.setdefault(zone.id, {})
            state["count"] = count

            level = "critical" if count >= crit else (
                "warning" if count >= warn else None)

            if level is None:
                # Below the warning line, the zone calms down and re-arms.
                state.pop("since", None)
                state.pop("fired", None)
                continue

            if state.get("level") != level:
                state["level"] = level
                state["since"] = ctx.now
                state.pop("fired", None)
            state.setdefault("since", ctx.now)

            held = ctx.now - state["since"]
            if held < t.crowd_sustain_s or state.get("fired"):
                continue

            state["fired"] = True
            etype = "crowd_critical" if level == "critical" else "crowd_warning"
            limit = crit if level == "critical" else warn
            out.append(Candidate(
                type=etype,
                message=(f"{count} people in {zone.name} — the "
                         f"{level} limit is {limit}. Held for {held:.1f}s."),
                zone=zone, highlight=level,
                meta={"count": count, "warn_limit": warn,
                      "critical_limit": crit, "held_seconds": round(held, 1)},
            ))
        return out

    def counts(self) -> dict[int, int]:
        return {zid: s.get("count", 0) for zid, s in self._state.items()}


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
