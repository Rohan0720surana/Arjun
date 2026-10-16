"""
The Signal for Help classifier, and exam mode.

The hand classifier is tested as a pure function over synthetic 21-point hands,
which is the only honest way to check a gesture rule: no camera, no model
weights, and no volunteer required to make a distress signal on demand.

Exam mode is tested through the pose-keypoint yaw fallback rather than the face
mesh, because that path is deterministic and because it is the path that runs
when the face model is missing — which is exactly when a quiet failure would
matter most.
"""

from __future__ import annotations

import config
from backend.detectors.exam import ExamModeDetector
from backend.detectors.sos import (CLOSED_OVER_THUMB, OPEN_THUMB_IN, OTHER,
                                   classify_hand, finger_states, thumb_tucked)
from backend.vision import skeleton as sk
from backend.vision import synth
from conftest import Harness, make_zone, person, thing

# --------------------------------------------------------------------------- #
# The gesture classifier
# --------------------------------------------------------------------------- #

def test_the_two_stages_are_recognised():
    assert classify_hand(synth.hand("thumb_tucked")) == OPEN_THUMB_IN
    assert classify_hand(synth.hand("signal_closed")) == CLOSED_OVER_THUMB


def test_an_ordinary_open_palm_is_not_the_signal():
    """A wave has the fingers out but the thumb beside them, not inside."""
    assert classify_hand(synth.hand("open")) == OTHER
    assert not thumb_tucked(synth.hand("open"))


def test_a_plain_fist_is_not_the_signal():
    """Closed fingers only count when the thumb is underneath them."""
    assert classify_hand(synth.hand("fist")) == OTHER


def test_pointing_is_not_the_signal():
    assert classify_hand(synth.hand("point")) == OTHER


def test_the_classifier_is_rotation_invariant():
    """A hand held up beside a face is rarely upright."""
    for rotation in (-120, -45, 0, 30, 95, 175):
        assert classify_hand(
            synth.hand("thumb_tucked", rotation=rotation)) == OPEN_THUMB_IN
        assert classify_hand(
            synth.hand("signal_closed", rotation=rotation)) == CLOSED_OVER_THUMB
        assert classify_hand(synth.hand("open", rotation=rotation)) == OTHER


def test_the_classifier_is_scale_invariant():
    for size in (24.0, 60.0, 180.0):
        assert classify_hand(
            synth.hand("thumb_tucked", size=size)) == OPEN_THUMB_IN
        assert classify_hand(synth.hand("fist", size=size)) == OTHER


def test_finger_states_read_correctly():
    assert finger_states(synth.hand("open")) == [True, True, True, True]
    assert finger_states(synth.hand("signal_closed")) == [False] * 4
    assert finger_states(synth.hand("point"))[0] is True


def test_a_raised_hand_is_the_cheap_gate_before_the_model():
    """The expensive model only runs on arms that are actually up."""
    down = synth.standing(320, 330, 200, arms="down")
    up = synth.standing(320, 330, 200, arms="up_left")
    assert not sk.hand_is_raised(down, "left")
    assert sk.hand_is_raised(up, "left")
    assert sk.hand_box(up, "left") is not None


# --------------------------------------------------------------------------- #
# Exam mode — head yaw from pose keypoints
# --------------------------------------------------------------------------- #

def test_head_yaw_reads_zero_when_facing_the_camera():
    kps = synth.sitting(320, 300, 180, yaw=0.0)
    yaw = sk.head_yaw_deg(kps)
    assert yaw is not None and abs(yaw) < 8.0


def test_head_yaw_grows_with_the_turn():
    small = sk.head_yaw_deg(synth.sitting(320, 300, 180, yaw=12.0))
    large = sk.head_yaw_deg(synth.sitting(320, 300, 180, yaw=50.0))
    assert small is not None and large is not None
    assert abs(large) > abs(small)
    assert abs(large) >= config.thresholds().exam_yaw_deg


def test_head_yaw_is_unknown_without_a_head():
    assert sk.head_yaw_deg(sk.empty()) is None


# --------------------------------------------------------------------------- #
# Exam mode — the rule
# --------------------------------------------------------------------------- #

DESK = [[0.35, 0.45], [0.65, 0.45], [0.65, 0.95], [0.35, 0.95]]
SEAT_Y, BODY = 300.0, 180.0


def _candidate(yaw: float = 0.0):
    return person(1, synth.sitting(320, SEAT_Y, BODY, yaw=yaw))


def test_exam_mode_is_off_unless_the_camera_says_so(flags):
    zone = make_zone("desk", DESK, name="Desk 3")
    flags.exam_mode = False
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(8.0, scene=lambda t: ([_candidate(60.0)],
                                [thing(60, "cell phone", 320, 300, 24)]))
    h.assert_quiet("exam mode was never switched on")


def test_phone_at_a_desk_is_flagged(flags):
    zone = make_zone("desk", DESK, name="Desk 3")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(5.0, scene=lambda t: ([_candidate()],
                                [thing(60, "cell phone", 320, 300, 24)]))

    hits = h.of_type("exam_phone")
    assert len(hits) == 1
    assert hits[0].message.startswith("Desk 3: phone visible")
    assert "invigilator review" in hits[0].message
    # The flag must describe, never accuse, and never identify.
    assert "cheat" not in hits[0].message.lower()
    assert hits[0].meta["desk"] == "Desk 3"


def test_a_phone_glimpsed_briefly_is_not_flagged(flags):
    zone = make_zone("desk", DESK, name="Desk 3")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)

    def scene(t: float):
        objs = ([thing(60, "cell phone", 320, 300, 24)]
                if 1.0 <= t < 1.8 else [])
        return [_candidate()], objs

    h.run(6.0, scene=scene)
    h.assert_quiet("under the two second threshold")


def test_a_phone_off_the_desks_is_ignored(flags):
    zone = make_zone("desk", DESK, name="Desk 3")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(6.0, scene=lambda t: ([_candidate()],
                                [thing(60, "cell phone", 60, 300, 24)]))
    h.assert_quiet("the invigilator's own phone, across the room")


def test_sustained_head_turn_is_flagged_at_info_severity(flags):
    zone = make_zone("desk", DESK, name="Desk 5")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(6.0, scene=lambda t: ([_candidate(yaw=55.0)], []))

    hits = h.of_type("exam_gaze")
    assert len(hits) == 1
    assert hits[0].resolved_severity() == "info", (
        "head turn is the weaker signal and must not shout")
    assert hits[0].meta["yaw_source"] == "pose_keypoints"
    assert "invigilator review" in hits[0].message


def test_a_glance_up_is_not_flagged(flags):
    """Students look at the clock. Three seconds is the point of the timer."""
    zone = make_zone("desk", DESK, name="Desk 5")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)

    def scene(t: float):
        return [_candidate(yaw=55.0 if 1.0 <= t < 2.4 else 0.0)], []

    h.run(8.0, scene=scene)
    h.assert_quiet("a glance, not a sustained turn")


def test_facing_the_paper_is_never_flagged(flags):
    zone = make_zone("desk", DESK, name="Desk 5")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(20.0, scene=lambda t: ([_candidate(yaw=4.0)], []))
    h.assert_quiet("sitting still and working")


def test_desk_zone_can_set_its_own_angle(flags):
    zone = make_zone("desk", DESK, name="Front row", exam_yaw_deg=80.0)
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(8.0, scene=lambda t: ([_candidate(yaw=50.0)], []))
    h.assert_quiet("below this desk's own, stricter angle")


def test_flags_do_not_repeat_every_frame(flags):
    zone = make_zone("desk", DESK, name="Desk 3")
    flags.exam_mode = True
    h = Harness(ExamModeDetector(), zones=[zone], flags=flags)
    h.run(30.0, scene=lambda t: ([_candidate()],
                                 [thing(60, "cell phone", 320, 300, 24)]))
    n = len(h.of_type("exam_phone"))
    assert 1 <= n <= 3, f"expected a couple of reminders over 30s, got {n}"


# --------------------------------------------------------------------------- #
# The SOS two-stage sequence
# --------------------------------------------------------------------------- #
#
# The classifier above is tested on its own. What is left is the state machine:
# the order of the stages, the hold on each, and the three second window. Those
# are driven here with the hand model stubbed out, so the test exercises the
# sequencing rather than MediaPipe.

import numpy as np                                             # noqa: E402

from backend.detectors.sos import SosGestureDetector            # noqa: E402


class _ScriptedSos(SosGestureDetector):
    """An SOS detector whose eyes are a script instead of a model."""

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.t0 = None

    def _read_hand(self, frame, det, side, t):
        if side != "left":
            return None
        if self.t0 is None:
            self.t0 = self._now
        return self.script(self._now - self.t0)

    def update(self, ctx):
        self._now = ctx.now
        return super().update(ctx)


def _signaller():
    return person(1, synth.standing(320, 330, 200, arms="up_left"))


def _run_sos(script, seconds=6.0):
    det = _ScriptedSos(script)
    h = Harness(det)
    # The rule needs a frame present to decide a crop is worth taking; the
    # stub never looks at it.
    def scene(t):
        return [_signaller()], []
    started = h.now
    while h.now - started < seconds:
        h.now += h.dt
        h.frame += 1
        from backend.world import FrameWorld
        from backend.state import CameraContext
        world = FrameWorld(camera="Test", ts=h.now, frame_idx=h.frame,
                           width=640, height=360, persons=[_signaller()])
        h.tracks.update(world)
        ctx = CameraContext(camera="Test", world=world, tracks=h.tracks,
                            zones=h.zones, flags=h.flags, now=h.now)
        ctx.shared["frame"] = np.zeros((360, 640, 3), np.uint8)
        ctx.compute_memberships()
        h.fired.extend(det.safe_update(ctx))
        h.last_ctx = ctx
    return h


def test_the_full_sequence_raises_a_silent_critical_alert():
    def script(t):
        if t < 1.0:
            return OPEN_THUMB_IN
        return CLOSED_OVER_THUMB

    h = _run_sos(script)
    hits = h.of_type("sos")
    assert len(hits) == 1
    assert hits[0].resolved_severity() == "critical"
    assert hits[0].spec.silent, (
        "an SOS must not announce itself to the person being escaped from")
    assert hits[0].meta["hand"] == "left"


def test_holding_only_the_first_stage_is_not_a_signal():
    h = _run_sos(lambda t: OPEN_THUMB_IN, seconds=8.0)
    assert not h.of_type("sos"), "the palm alone is not the signal"


def test_a_fist_with_no_preceding_palm_is_not_a_signal():
    h = _run_sos(lambda t: CLOSED_OVER_THUMB, seconds=8.0)
    assert not h.of_type("sos"), "stage two without stage one is just a fist"


def test_the_stages_must_be_close_together():
    """Palm now and a fist half a minute later is not one gesture."""
    def script(t):
        if t < 1.0:
            return OPEN_THUMB_IN
        if t < 8.0:
            return OTHER
        return CLOSED_OVER_THUMB

    h = _run_sos(script, seconds=12.0)
    assert not h.of_type("sos")


def test_the_signal_is_reported_once():
    def script(t):
        return OPEN_THUMB_IN if t < 1.0 else CLOSED_OVER_THUMB

    h = _run_sos(script, seconds=12.0)
    assert len(h.of_type("sos")) == 1
