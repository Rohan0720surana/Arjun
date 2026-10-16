"""
The Signal for Help.

A hand gesture designed to be made silently, on a video call or across a room,
by somebody who cannot safely say anything: palm to the camera, thumb folded
into the palm, then the four fingers closed down over the thumb. It was
published by the Canadian Women's Foundation in 2020 and has since been
recognised worldwide.

Why it belongs in this project specifically: ARJUN deliberately cannot hear,
cannot identify anyone and cannot read a face. A student in trouble has no way
to tell it anything — unless there is a gesture it is watching for. This is the
one channel in the whole system that a person can deliberately open, and it is
the clearest answer to "what good is a camera that refuses to look at people".

**How it is recognised.** Two stages, in order, within about three seconds:

* *Stage one* — fingers extended, thumb tucked across the palm. Measured as
  the thumb tip being close to the ring-finger knuckle relative to the size of
  the hand.
* *Stage two* — fingers folded down, thumb still hidden underneath.

Each stage must hold briefly so a hand waving through the two shapes by
accident does not trigger it. Requiring the *sequence* is what separates the
signal from an ordinary wave or a fist.

**How it is kept cheap.** Hand landmarking is far too expensive to run on
everybody. A pose-level gate runs first — is this person's wrist raised to
around shoulder height — and only then is a small square around that one wrist
handed to MediaPipe. In a room of thirty seated students the model usually runs
zero times a second.

The alert is deliberately **silent** on the dashboard: no chime, no flashing
tile. If somebody signals for help because of the person standing next to
them, the last thing that should happen is a noise that announces it.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

import config
from backend.detectors.base import Candidate, Detector
from backend.detectors.fall import short_id
from backend.state import CameraContext
from backend.vision import mp_tasks
from backend.vision import skeleton as sk

OPEN_THUMB_IN = "palm_thumb_tucked"
CLOSED_OVER_THUMB = "fingers_over_thumb"
OTHER = "other"


# --------------------------------------------------------------------------- #
# The classifier — a pure function, so it can be tested without a camera
# --------------------------------------------------------------------------- #

def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]),
                            float(a[1]) - float(b[1])))


def hand_size(lm: np.ndarray) -> float:
    """A scale for this hand: wrist to the middle-finger knuckle."""
    return max(1e-3, _dist(lm[mp_tasks.WRIST], lm[mp_tasks.MIDDLE_MCP]))


def finger_states(lm: np.ndarray) -> list[bool]:
    """Extended (True) or folded (False) for index, middle, ring and pinky.

    Compares how far the fingertip is from the wrist against how far the
    middle joint is. Using the wrist as the origin makes this independent of
    how the hand is rotated in the frame, which matters because a hand held up
    beside a face is rarely upright.
    """
    wrist = lm[mp_tasks.WRIST]
    out: list[bool] = []
    for tip, pip in zip(mp_tasks.FINGER_TIPS, mp_tasks.FINGER_PIPS):
        out.append(_dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.08)
    return out


def thumb_tucked(lm: np.ndarray) -> bool:
    """Is the thumb folded across the palm rather than sticking out?"""
    size = hand_size(lm)
    # Across the palm, the thumb tip ends up near the ring knuckle. Held out,
    # it is roughly a hand's length away from it.
    to_ring = _dist(lm[mp_tasks.THUMB_TIP], lm[mp_tasks.RING_MCP]) / size
    to_index = _dist(lm[mp_tasks.THUMB_TIP], lm[mp_tasks.INDEX_MCP]) / size
    return to_ring < 0.75 and to_index < 0.85


def classify_hand(lm: np.ndarray) -> str:
    """Which stage of the Signal for Help, if any, this hand shape is."""
    if lm is None or len(lm) < 21:
        return OTHER
    tucked = thumb_tucked(lm)
    if not tucked:
        return OTHER
    extended = finger_states(lm)
    n_extended = sum(extended)
    if n_extended >= 3:
        return OPEN_THUMB_IN
    if n_extended <= 1:
        return CLOSED_OVER_THUMB
    return OTHER


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #

class SosGestureDetector(Detector):

    name = "sos"
    title = "Signal for Help"
    explanation = (
        "Open palm with the thumb tucked in, followed by the fingers folding "
        "over it, within about three seconds. Raises a silent priority alert."
    )

    def __init__(self) -> None:
        super().__init__()
        self._hands: mp_tasks.HandLandmarks | None = None
        self._frame_no = 0
        self.model_error = ""

    def reset(self) -> None:
        self._frame_no = 0

    # -- model ---------------------------------------------------------------- #

    def _model(self) -> mp_tasks.HandLandmarks | None:
        if self._hands is None:
            self._hands = mp_tasks.HandLandmarks(max_hands=1)
        if not self._hands.available:
            self.model_error = self._hands.error
            return None
        return self._hands

    # -- the rule ------------------------------------------------------------- #

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        self._frame_no += 1
        # The gesture is held for seconds, so there is nothing to gain from
        # looking at every frame, and a good deal of CPU to lose.
        if self._frame_no % max(1, t.sos_hands_every_n):
            return []

        frame = ctx.shared.get("frame")
        if frame is None:
            return []

        out: list[Candidate] = []
        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None:
                continue
            for side in ("left", "right"):
                if not sk.hand_is_raised(det.keypoints, side):
                    continue
                stage = self._read_hand(frame, det, side, t)
                if stage is None:
                    continue
                cand = self._advance(ctx, track, det, side, stage, t)
                if cand is not None:
                    out.append(cand)
        return out

    # -- reading one hand ------------------------------------------------------- #

    def _read_hand(self, frame: np.ndarray, det, side: str, t) -> str | None:
        box = sk.hand_box(det.keypoints, side)
        if box is None:
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w, box[2]), min(h, box[3])
        if (x2 - x1) < t.sos_min_hand_crop_px or (y2 - y1) < t.sos_min_hand_crop_px:
            # Too few pixels for a believable answer; say nothing rather than
            # guess, because a false SOS is an expensive mistake.
            return None
        model = self._model()
        if model is None:
            return None
        hands = model.detect(frame[y1:y2, x1:x2])
        if not hands:
            return None
        return classify_hand(hands[0])

    # -- the two-stage state machine ---------------------------------------------- #

    def _advance(self, ctx: CameraContext, track, det, side: str,
                 stage: str, t) -> Candidate | None:
        state: dict[str, Any] = track.scratch("sos").setdefault(side, {})

        if stage == OPEN_THUMB_IN:
            state.setdefault("open_since", ctx.now)
            if ctx.now - state["open_since"] >= t.sos_hold_s:
                state["stage_one_at"] = ctx.now
            state.pop("closed_since", None)
            return None

        if stage == CLOSED_OVER_THUMB:
            one_at = state.get("stage_one_at")
            if one_at is None:
                return None
            if ctx.now - one_at > t.sos_window_s:
                state.clear()           # too slow to be the deliberate signal
                return None
            state.setdefault("closed_since", ctx.now)
            if ctx.now - state["closed_since"] < t.sos_hold_s:
                return None
            if state.get("fired"):
                return None
            state["fired"] = True
            elapsed = ctx.now - one_at
            zones = ctx.zones_for(det)
            return Candidate(
                type="sos",
                message=(f"Person #{short_id(det.track_id)} made the Signal "
                         f"for Help with their {side} hand "
                         f"({elapsed:.1f}s between the two stages). "
                         f"Respond discreetly."),
                track_id=det.track_id,
                zone=zones[0] if zones else None,
                highlight="critical", highlight_for=20.0,
                meta={"hand": side, "stage_gap_seconds": round(elapsed, 2),
                      "gesture": "Signal for Help (Canadian Women's Foundation)",
                      "silent": True},
            )

        # Anything else relaxes the first stage, but only after a moment, so a
        # single mis-read frame mid-gesture does not throw the sequence away.
        if state.get("stage_one_at") and ctx.now - state["stage_one_at"] > t.sos_window_s:
            state.clear()
        else:
            state.pop("open_since", None)
        return None

    def status(self) -> dict[str, Any]:
        base = super().status()
        if self.model_error:
            base["last_error"] = base["last_error"] or self.model_error
        return base
