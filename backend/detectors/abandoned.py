"""
Unattended bags.

The rule is "a bag that has stopped moving, with nobody standing near it, for
longer than N seconds". Each of those three clauses is doing real work:

* **Stopped moving** — a bag on someone's shoulder is tracked as a bag, and it
  travels. Requiring stillness removes every bag that is being carried without
  having to decide whose it is.
* **Nobody near it** — proximity is measured in the *person's* body heights,
  not pixels, so "about two metres away" means the same thing for someone
  beside the lens and someone at the end of the corridor.
* **Ownership has to be earned** — someone walking past a bag does not reclaim
  it. A person must stay within range for a full second before the clock is
  reset. Without that, a busy corridor would keep the timer at zero forever
  and the rule would never fire at all.

What this rule deliberately does not do is work out *whose* bag it is. There is
no appearance matching, no association of a bag with an identity. It reports a
place and a duration, which is all a human needs to go and look.
"""

from __future__ import annotations

import math

import config
from backend.detectors.base import Candidate, Detector
from backend.state import CameraContext
from backend.world import Detection


class AbandonedObjectDetector(Detector):

    name = "abandoned"
    title = "Unattended bags"
    explanation = (
        "A backpack, handbag or suitcase stays still with nobody within about "
        "two metres of it for longer than the configured time."
    )

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []

        for obj in ctx.world.objects:
            if obj.label not in config.BAG_CLASSES:
                continue
            track = ctx.track_of(obj)
            if track is None or track.age < 1.0:
                continue

            state = track.scratch("abandoned")

            # -- is it still? -------------------------------------------- #
            moved = track.displacement(2.0)
            if moved > t.abandoned_move_tolerance:
                state.clear()
                continue

            # -- is anybody with it? -------------------------------------- #
            # Custody is "near it and stopped", not merely "near it". In a
            # busy corridor somebody is within two metres of a bag most of the
            # time, and treating that as ownership would hold the clock at
            # zero forever - the rule would quietly never fire at all.
            owner, distance, owner_speed = self._nearest_person(ctx, obj)
            attended = (owner is not None
                        and distance <= t.abandoned_owner_radius
                        and owner_speed <= t.abandoned_claim_speed_bh_s)

            if attended:
                # Being nearby for an instant is not custody. A passer-by must
                # linger before the bag counts as reclaimed.
                near_since = state.get("near_since")
                if near_since is None:
                    state["near_since"] = ctx.now
                elif ctx.now - near_since >= t.abandoned_reclaim_s:
                    state.clear()
                    continue
            else:
                state.pop("near_since", None)
                state.setdefault("alone_since", ctx.now)

            alone_since = state.get("alone_since")
            if alone_since is None:
                continue

            alone_for = ctx.now - alone_since
            limit = self._limit(ctx, obj, t)
            if alone_for < limit or state.get("fired"):
                continue

            state["fired"] = True
            zone = self._zone_of(ctx, obj)
            where = f" in {zone.name}" if zone else ""
            out.append(Candidate(
                type="abandoned",
                message=(f"A {obj.label} has been unattended{where} for "
                         f"{_mmss(alone_for)} — nearest person is "
                         f"{distance:.1f} body heights away." if owner else
                         f"A {obj.label} has been unattended{where} for "
                         f"{_mmss(alone_for)}, with nobody else in view."),
                track_id=obj.track_id, zone=zone, highlight="warning",
                highlight_for=10.0,
                meta={
                    "object": obj.label,
                    "unattended_seconds": round(alone_for, 1),
                    "threshold_seconds": limit,
                    "nearest_person_body_heights": (
                        None if owner is None else round(distance, 2)),
                    "nearest_person_speed_bh_s": (
                        None if owner is None else round(owner_speed, 2)),
                    "drift_body_heights": round(moved, 3),
                    "position": [round(v) for v in obj.center()],
                },
            ))
        return out

    # -- helpers ---------------------------------------------------------------- #

    @staticmethod
    def _nearest_person(
        ctx: CameraContext, obj: Detection
    ) -> tuple[Detection | None, float, float]:
        """Closest person: who, how far in their own body heights, how fast."""
        ox, oy = obj.center()
        best: Detection | None = None
        best_d = math.inf
        best_speed = 0.0
        for p in ctx.world.persons:
            px, py = p.center()
            scale = max(1.0, p.scale())
            d = math.hypot(px - ox, py - oy) / scale
            if d < best_d:
                track = ctx.track_of(p)
                best, best_d = p, d
                best_speed = track.path_speed(1.2) if track else 0.0
        return best, best_d, best_speed

    @staticmethod
    def _zone_of(ctx: CameraContext, obj: Detection):
        zones = ctx.zones_for(obj)
        return zones[0] if zones else None

    @staticmethod
    def _limit(ctx: CameraContext, obj: Detection, t) -> float:
        zone = AbandonedObjectDetector._zone_of(ctx, obj)
        if zone is not None:
            return float(zone.param("abandoned_seconds", t.abandoned_seconds))
        return float(t.abandoned_seconds)


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
