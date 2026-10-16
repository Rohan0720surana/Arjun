"""
Exam mode — the feature that needed the most care about what it is *not*.

Switched on per camera, and only meaningful on a camera that has desk zones
drawn on it. It raises two kinds of flag:

* **A phone-shaped object is visible at a desk** for more than a couple of
  seconds.
* **A head stays turned away from the desk** beyond a generous angle for more
  than three seconds.

Both are deliberately framed as *observations for an invigilator*, never as
accusations. The wording in every message says what was seen, where, and for
how long, and stops. ARJUN does not know who is sitting at desk three, has no
way to find out, and does not record anything that would let anyone else find
out either: it sees an anonymous skeleton at a numbered desk.

That restraint is not decoration, it is the design. A system that could say
"candidate X cheated" would need identity, and identity is precisely what this
project refuses to collect. What a human gets is "Desk 3: phone visible for 4
seconds at 10:42" — enough to walk over and look, which is what invigilators do
anyway, and nothing more.

Two more guard rails. Head turn is the weaker signal of the two, so it is
logged at *info* severity: students look up, look at the clock, and stretch.
And the yaw estimate itself prefers MediaPipe's 3D face transform; when that is
unavailable it falls back to a blunt estimate from ear and nose positions and
says so in the event metadata, so nobody reads more precision into a flag than
is there.
"""

from __future__ import annotations

from typing import Any

import config
from backend.detectors.base import Candidate, Detector
from backend.state import CameraContext
from backend.vision import mp_tasks
from backend.vision import skeleton as sk
from backend.zone_store import Zone


class ExamModeDetector(Detector):

    name = "exam"
    title = "Exam mode"
    requires_flag = "exam_mode"
    requires_zones = ("desk",)
    explanation = (
        "At desks: a phone-shaped object visible for more than two seconds, "
        "or a head turned away for more than three. Flags for invigilator "
        "review, never accusations, never identities."
    )

    def __init__(self) -> None:
        super().__init__()
        self._face: mp_tasks.FaceGeometry | None = None
        self._frame_no = 0
        self.model_error = ""

    def reset(self) -> None:
        self._frame_no = 0

    # -- the rule ------------------------------------------------------------- #

    def update(self, ctx: CameraContext) -> list[Candidate]:
        t = config.thresholds()
        self._frame_no += 1
        desks = ctx.zones.of("desk")
        if not desks:
            return []

        out: list[Candidate] = []
        out.extend(self._phones(ctx, desks, t))
        out.extend(self._gaze(ctx, desks, t))
        return out

    # -- phones ---------------------------------------------------------------- #

    def _phones(self, ctx: CameraContext, desks: list[Zone], t) -> list[Candidate]:
        out: list[Candidate] = []
        for obj in ctx.world.objects:
            if obj.label not in config.PHONE_CLASSES:
                continue
            track = ctx.track_of(obj)
            if track is None:
                continue
            desk = next((z for z in desks if ctx.in_zone(obj, z)), None)
            if desk is None:
                track.held_reset("phone")
                continue

            seen_for = track.held("phone", True, ctx.now)
            limit = float(desk.param("exam_phone_seconds", t.exam_phone_seconds))
            if seen_for < limit:
                continue

            state = track.scratch("exam_phone")
            last = state.get(desk.id, 0.0)
            if last and ctx.now - last < max(limit * 4, 15.0):
                continue
            state[desk.id] = ctx.now

            out.append(Candidate(
                type="exam_phone",
                message=(f"{desk.name}: phone visible ({_mmss(seen_for)}) — "
                         f"for invigilator review."),
                track_id=obj.track_id, zone=desk, highlight="warning",
                meta={"desk": desk.name, "seconds_visible": round(seen_for, 1),
                      "threshold_seconds": limit,
                      "detector_confidence": round(obj.conf, 2),
                      "note": "observation only; ARJUN does not identify people"},
            ))
        return out

    # -- head turn --------------------------------------------------------------- #

    def _gaze(self, ctx: CameraContext, desks: list[Zone], t) -> list[Candidate]:
        out: list[Candidate] = []
        use_mesh = self._frame_no % max(1, t.exam_face_every_n) == 0

        for det in ctx.world.persons:
            track = ctx.track_of(det)
            if track is None:
                continue
            desk = next((z for z in desks if ctx.in_zone(det, z)), None)
            if desk is None:
                track.held_reset("yaw")
                continue

            yaw, source = self._yaw(ctx, det, use_mesh, t)
            if yaw is None:
                # Unknown is not "facing forward": let the timer lapse rather
                # than accumulate against a reading we never took.
                track.held_reset("yaw")
                continue

            limit_deg = float(desk.param("exam_yaw_deg", t.exam_yaw_deg))
            limit_s = float(desk.param("exam_yaw_seconds", t.exam_yaw_seconds))
            turned = abs(yaw) >= limit_deg
            held = track.held("yaw", turned, ctx.now)
            if not turned or held < limit_s:
                continue

            state = track.scratch("exam_gaze")
            last = state.get(desk.id, 0.0)
            if last and ctx.now - last < max(limit_s * 5, 20.0):
                continue
            state[desk.id] = ctx.now

            out.append(Candidate(
                type="exam_gaze",
                message=(f"{desk.name}: head turned away "
                         f"({abs(yaw):.0f} deg) for {_mmss(held)} — "
                         f"for invigilator review."),
                track_id=det.track_id, zone=desk, severity="info",
                highlight=None,
                meta={"desk": desk.name, "yaw_deg": round(yaw, 1),
                      "held_seconds": round(held, 1),
                      "threshold_deg": limit_deg,
                      "yaw_source": source,
                      "note": "observation only; ARJUN does not identify people"},
            ))
        return out

    # -- yaw ---------------------------------------------------------------------- #

    def _yaw(self, ctx: CameraContext, det, use_mesh: bool,
             t) -> tuple[float | None, str]:
        """Head yaw in degrees, from the face mesh if we can afford it."""
        cached = ctx.track_of(det).scratch("yaw_cache") if ctx.track_of(det) else {}

        if use_mesh:
            frame = ctx.shared.get("frame")
            box = sk.head_box(det.box, det.keypoints)
            if frame is not None and box is not None:
                h, w = frame.shape[:2]
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(w, box[2]), min(h, box[3])
                if (x2 - x1) >= t.exam_min_face_crop_px and (y2 - y1) >= t.exam_min_face_crop_px:
                    face = self._face_model()
                    if face is not None:
                        angles = face.head_angles(frame[y1:y2, x1:x2])
                        if angles is not None:
                            cached["yaw"] = angles[0]
                            cached["source"] = "mediapipe_face_mesh"
                            return angles[0], "mediapipe_face_mesh"

        # Between mesh frames, reuse the last mesh reading if it is fresh.
        if cached.get("source") == "mediapipe_face_mesh" and "yaw" in cached:
            return cached["yaw"], "mediapipe_face_mesh"

        yaw = sk.head_yaw_deg(det.keypoints)
        return (yaw, "pose_keypoints") if yaw is not None else (None, "none")

    def _face_model(self) -> mp_tasks.FaceGeometry | None:
        if self._face is None:
            self._face = mp_tasks.FaceGeometry(max_faces=1)
        if not self._face.available:
            self.model_error = self._face.error
            return None
        return self._face

    def status(self) -> dict[str, Any]:
        base = super().status()
        if self.model_error:
            base["last_error"] = base["last_error"] or self.model_error
        return base


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
