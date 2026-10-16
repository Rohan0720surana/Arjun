"""
Entry and exit counting across a drawn line.

Draw a line across a doorway and this counts people through it, each way, and
keeps a running "inside" figure. It is the most useful boring feature in the
system: how full is the library, did the hall actually clear during the drill,
is anybody still in the lab.

**How a crossing is decided.** Each person's foot point is placed on one side
of the line or the other using the sign of a cross product, normalised by the
line's own length so the result is a distance in "line lengths" rather than
pixels. A crossing is recorded when that sign flips *and* both the old and new
positions are further from the line than a small dead-band. Without the band,
somebody standing exactly on the threshold would be counted dozens of times a
second as the tracker jitters by one pixel.

A track also has to be at least ``line_min_track_age_s`` old before it can
count. A brand-new track has no "previous side" worth trusting, and tracking
ids are sometimes handed out to a person already halfway across.

Direction is decided by which way the sign flipped, which is why the privacy
view draws a little arrow on the line: that arrow is the "in" direction.
"""

from __future__ import annotations

from typing import Any

import config
from backend.detectors.base import Candidate, Detector
from backend.detectors.fall import short_id
from backend.state import CameraContext


class LineCounter(Detector):

    name = "line_counter"
    title = "Entry / exit counting"
    requires_zones = ("line",)
    explanation = (
        "Counts foot points crossing a drawn line in each direction, with a "
        "dead-band so somebody standing on the threshold is not counted twice "
        "a second."
    )

    def __init__(self) -> None:
        super().__init__()
        #: zone id -> {"in": n, "out": n}
        self._counts: dict[int, dict[str, int]] = {}

    def reset(self) -> None:
        # The counts themselves survive a reconnect; they belong to the room,
        # not to the camera session. Only the per-track sides are forgotten,
        # and those live on the tracks, which the store has already cleared.
        pass

    def counts(self) -> dict[int, dict[str, int]]:
        return {zid: dict(v) for zid, v in self._counts.items()}

    def inside(self, zone_id: int) -> int:
        c = self._counts.get(zone_id, {})
        return max(0, c.get("in", 0) - c.get("out", 0))

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []
        lines = ctx.zones.lines()
        if not lines:
            return out

        w, h = ctx.width, ctx.height

        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None:
                continue
            sides = track.scratch("line_sides")
            foot = det.foot()

            for zone in lines:
                side = zone.side(foot, w, h)
                margin = float(zone.param("line_cross_margin",
                                          t.line_cross_margin))

                # Only remember *confident* sides. Storing every reading,
                # including those inside the dead-band, meant that a sample
                # landing on the line became the "previous side" and the
                # comparison on the far side was skipped — so a person whose
                # stride happened to put a foot exactly on the threshold was
                # never counted at all. The band is hysteresis, not a hole.
                if abs(side) < margin:
                    continue
                previous = sides.get(zone.id)
                sides[zone.id] = side

                if track.age < t.line_min_track_age_s or previous is None:
                    continue
                if (side > 0) == (previous > 0):
                    continue

                direction = "in" if side > 0 else "out"
                counts = self._counts.setdefault(zone.id, {"in": 0, "out": 0})
                counts[direction] += 1
                inside = self.inside(zone.id)

                out.append(Candidate(
                    type="line_cross",
                    message=(f"Person #{short_id(det.track_id)} crossed "
                             f"{zone.name} ({direction}). "
                             f"{inside} now inside."),
                    track_id=det.track_id, zone=zone,
                    dedupe=direction, highlight=None,
                    meta={"direction": direction, "inside": inside,
                          "in_total": counts["in"], "out_total": counts["out"]},
                ))

        # Publish for the dashboard tile, whether or not anything crossed.
        counters: dict[str, Any] = ctx.shared.setdefault("counters", {})
        for zone in lines:
            c = self._counts.get(zone.id, {"in": 0, "out": 0})
            counters[str(zone.id)] = {
                "zone": zone.name, "in": c["in"], "out": c["out"],
                "inside": self.inside(zone.id),
            }
        return out

    def seed(self, zone_id: int, in_count: int, out_count: int) -> None:
        """Restore counts from the database after a restart."""
        self._counts[zone_id] = {"in": int(in_count), "out": int(out_count)}
