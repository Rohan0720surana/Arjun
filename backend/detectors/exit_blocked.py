"""
Fire exits that are not clear.

Blocked fire exits are the quietest dangerous thing on a campus. Nobody
notices the stack of chairs by the stairwell door until the day it matters,
and nobody is standing there watching for it — which is exactly the kind of
problem a system that is already looking at the room can solve for free.

Two things count as blocking:

* **An object** — a bag, a chair, a bicycle, a box — whose footprint is inside
  an exit zone and which has not moved for N seconds.
* **A person** who is standing, rather than passing through. Walking through a
  fire exit is what a fire exit is for; standing in one and chatting for
  fifteen seconds is not. The test is the person's own speed in body heights
  per second, so it does not care how far from the camera they are.

The severity is critical on purpose. It is not an emergency in the moment, but
it is the single cheapest life to save on a campus, and burying it as a
"warning" among loitering alerts is how it would get ignored.
"""

from __future__ import annotations

import config
from backend.detectors.base import Candidate, Detector
from backend.detectors.fall import short_id
from backend.state import CameraContext


class ExitBlockedDetector(Detector):

    name = "exit_blocked"
    title = "Fire exit blocked"
    requires_zones = ("exit",)
    explanation = (
        "Something is left in a fire exit, or somebody stands in one rather "
        "than walking through it, for longer than the configured time."
    )

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []
        exits = ctx.zones.of("exit")
        if not exits:
            return out

        for zone in exits:
            limit = float(zone.param("exit_blocked_seconds",
                                     t.exit_blocked_seconds))

            # -- objects in the doorway -------------------------------- #
            for obj in ctx.world.objects:
                if obj.label not in config.BLOCKER_CLASSES:
                    continue
                if not ctx.in_zone(obj, zone):
                    continue
                track = ctx.track_of(obj)
                if track is None:
                    continue
                if track.displacement(2.0) > t.abandoned_move_tolerance:
                    track.scratch("exit_blocked").clear()
                    continue
                held = ctx.dwell(obj, zone) or (ctx.now - track.first_seen)
                cand = self._maybe(ctx, track, zone, held, limit,
                                   subject=f"A {obj.label}",
                                   track_id=obj.track_id,
                                   kind="object", label=obj.label)
                if cand is not None:
                    out.append(cand)

            # -- people standing in the doorway -------------------------- #
            for person in ctx.world.persons:
                if not ctx.in_zone(person, zone):
                    continue
                track = ctx.track_of(person)
                if track is None:
                    continue
                # Path length, not displacement: someone pacing back and
                # forth across a doorway is not standing still.
                speed = track.path_speed(1.5)
                if speed > t.exit_person_still_bh_s:
                    # Moving through: that is the exit doing its job.
                    track.scratch("exit_blocked").clear()
                    continue
                held = track.held("exit_still", True, ctx.now)
                cand = self._maybe(ctx, track, zone, held, limit,
                                   subject=f"Person #{short_id(person.track_id)}",
                                   track_id=person.track_id,
                                   kind="person", label="person", speed=speed)
                if cand is not None:
                    out.append(cand)

        return out

    @staticmethod
    def _maybe(ctx: CameraContext, track, zone, held: float, limit: float,
               subject: str, track_id: int, kind: str, label: str,
               speed: float | None = None) -> Candidate | None:
        state = track.scratch("exit_blocked")
        if held < limit:
            return None
        last = state.get(zone.id, 0.0)
        # Re-remind once per period rather than once per frame.
        if last and ctx.now - last < max(limit, 20.0):
            return None
        state[zone.id] = ctx.now
        verb = "has been left in" if kind == "object" else "has been standing in"
        meta = {
            "blocker": label,
            "kind": kind,
            "blocked_seconds": round(held, 1),
            "threshold_seconds": limit,
        }
        if speed is not None:
            meta["speed_body_heights_s"] = round(speed, 2)
        return Candidate(
            type="exit_blocked",
            message=f"{subject} {verb} {zone.name} for {_mmss(held)}.",
            track_id=track_id, zone=zone, highlight="critical",
            dedupe=kind, meta=meta,
        )


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
