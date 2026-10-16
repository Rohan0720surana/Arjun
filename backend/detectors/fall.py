"""
Fall detection — the rule this whole system exists for.

A fall is not "a person is horizontal". People lie on the grass, stretch in a
corridor, sit on the floor against a wall and do push-ups in the gym. What
separates a fall from all of those is the *sequence*, so the rule insists on
three things in order:

1.  **A sudden drop.** The hip centre falls by at least 16% of the person's own
    body height within about 0.6 seconds. Measured against the *highest* hip
    position in that window rather than the first, so a stumble that ends in a
    collapse still counts.
2.  **Ending near horizontal.** Either the torso line (shoulders to hips) tilts
    more than 55 degrees from vertical, or the bounding box becomes wider than
    it is tall. Two independent signals, because the torso angle is the better
    one and the box ratio is the one that survives a bad pose estimate.
3.  **Staying down.** Both conditions must hold for more than two seconds. This
    single clause removes almost every false positive: sitting down quickly,
    bending to pick something up, a child ducking, a tracker glitch. Someone
    who trips and gets straight back up is not someone who needs an ambulance.

The two-second wait is a deliberate trade. It costs two seconds of response
time and it buys an alert that a human will still trust at the end of a week —
and a rule nobody trusts is a rule nobody acts on.

Everything is measured in body heights, so the same numbers work for a person
close to the lens and one at the end of a corridor.
"""

from __future__ import annotations

import config
from backend.detectors.base import Candidate, Detector
from backend.state import CameraContext


class FallDetector(Detector):

    name = "fall"
    title = "Fall detection"
    explanation = (
        "Hip drops sharply, the body ends up near horizontal, and stays down "
        "for more than two seconds."
    )

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        out: list[Candidate] = []

        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None or len(track.samples) < 3:
                continue

            state = track.scratch("fall")
            horizontal = self._is_horizontal(det, t)

            # -- stage 1: did they drop? ---------------------------------- #
            drop = track.vertical_drop(t.fall_drop_window_s)
            if drop >= t.fall_drop_fraction and horizontal:
                state.setdefault("dropped_at", ctx.now)
                state.setdefault("drop", round(drop, 3))

            if "dropped_at" not in state:
                continue

            # -- stage 2: are they still down? ----------------------------- #
            if horizontal:
                track.held_reset("upright")
                down_for = ctx.now - state["dropped_at"]
                state["down_for"] = down_for

                if down_for >= t.fall_down_seconds and not state.get("fired"):
                    state["fired"] = True
                    out.append(self._alert(ctx, det, state, down_for, t))
                continue

            # -- stage 3: back on their feet ------------------------------- #
            # A short wobble upright does not clear the incident; a sustained
            # recovery does, and only then can the same track alert again.
            upright_for = track.held("upright", True, ctx.now)
            if upright_for >= t.fall_recover_seconds:
                if state.get("fired"):
                    state["recovered_at"] = ctx.now
                track.scratch("fall").clear()

        return out

    # -- helpers ---------------------------------------------------------------- #

    @staticmethod
    def _is_horizontal(det, t) -> bool:
        """Near-horizontal by torso angle, or by box shape as a backstop."""
        torso = det.torso_angle()
        if torso is not None and torso >= t.fall_torso_angle_deg:
            return True
        # The box test is only trusted when the pose is poor: a crouching
        # person can have a wide box while their torso is plainly upright.
        if torso is None and det.aspect() >= t.fall_horizontal_ratio:
            return True
        return det.aspect() >= t.fall_horizontal_ratio * 1.25

    @staticmethod
    def _alert(ctx: CameraContext, det, state, down_for: float,
               t) -> Candidate:
        drop = state.get("drop", 0.0)
        torso = det.torso_angle()
        angle_text = (f"torso {torso:.0f} deg from vertical" if torso is not None
                      else f"box {det.aspect():.2f}x wider than tall")
        # Name the zone if they fell in one — "near the west stairwell" is far
        # more use to a responder than a pair of coordinates.
        zones = ctx.zones_for(det)
        where = f" in {zones[0].name}" if zones else ""
        return Candidate(
            type="fall",
            message=(f"Person #{short_id(det.track_id)} dropped "
                     f"{drop:.0%} of their height in under "
                     f"{t.fall_drop_window_s:.1f}s, {angle_text}, and has been "
                     f"down {down_for:.1f}s{where}."),
            track_id=det.track_id,
            zone=zones[0] if zones else None,
            highlight="critical",
            highlight_for=12.0,
            meta={
                "drop_body_heights": round(float(drop), 3),
                "torso_angle_deg": None if torso is None else round(torso, 1),
                "box_aspect": round(det.aspect(), 2),
                "down_seconds": round(down_for, 1),
                "threshold_drop": t.fall_drop_fraction,
                "threshold_down_seconds": t.fall_down_seconds,
                "scale_px": round(det.scale(), 1),
            },
        )


def short_id(track_id: int) -> int:
    """Simulated tracks are offset by 900000; show the human-sized number."""
    return track_id - 900_000 if track_id >= 900_000 else track_id
