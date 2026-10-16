"""
Fall detection, driven by synthetic poses on a virtual clock.

The point of these tests is not that the code runs. It is that the rule says
"no" to the four things people actually do that look like falls — sitting,
bending, walking, standing still — and "yes", once, at the right moment, to
someone who goes down and stays down.
"""

from __future__ import annotations

import config
from backend.detectors.fall import FallDetector
from backend.vision import synth
from conftest import Harness, person, standing_person

FEET = 330.0
HEIGHT = 200.0
X = 320.0


def _collapse_scene(fall_at: float, stay: bool = True):
    """Walk for ``fall_at`` seconds, drop over 0.4s, then lie there."""
    def scene(t: float):
        if t < fall_at:
            kps = synth.standing(X, FEET, HEIGHT, phase=t * 6.0)
        elif t < fall_at + 0.4:
            u = (t - fall_at) / 0.4
            kps = synth.lying(X, FEET, HEIGHT, angle_deg=4 + 82 * u)
        elif stay:
            kps = synth.lying(X, FEET, HEIGHT, angle_deg=86)
        else:
            kps = synth.standing(X, FEET, HEIGHT, phase=t * 6.0)
        return [person(1, kps)], []
    return scene


def test_fall_fires_after_the_person_stays_down():
    h = Harness(FallDetector())
    h.run(6.0, scene=_collapse_scene(fall_at=2.0))

    falls = h.of_type("fall")
    assert len(falls) == 1, f"expected exactly one fall, got {h.types()}"

    meta = falls[0].meta
    assert meta["drop_body_heights"] >= config.thresholds().fall_drop_fraction
    assert meta["down_seconds"] >= config.thresholds().fall_down_seconds
    assert falls[0].resolved_severity() == "critical"
    assert "down" in falls[0].message


def test_fall_does_not_fire_before_the_two_second_wait():
    """At 1.5s on the floor the rule must still be holding its tongue."""
    h = Harness(FallDetector())
    # Drop at t=1.0, finish falling at 1.4, then only 1.2s of lying down.
    h.run(2.6, scene=_collapse_scene(fall_at=1.0))
    assert not h.of_type("fall"), "fired before the confirmation window elapsed"

    # Carry on in the same run and it should now fire.
    h.run(2.0, scene=lambda t: ([person(1, synth.lying(X, FEET, HEIGHT, 86))], []))
    assert len(h.of_type("fall")) == 1


def test_trip_and_recover_is_not_a_fall():
    h = Harness(FallDetector())
    h.run(4.0, scene=_collapse_scene(fall_at=1.0, stay=False))
    h.assert_quiet("a stumble that ends standing up is not a fall")


def test_sitting_down_is_not_a_fall():
    """Hips drop fast, but the torso stays vertical, so the rule says no."""
    def scene(t: float):
        if t < 1.0:
            kps = synth.standing(X, FEET, HEIGHT)
        else:
            # Sit in 0.4s: hips fall to seat height, shoulders stay above them.
            u = min(1.0, (t - 1.0) / 0.4)
            kps = synth.sitting(X, FEET - HEIGHT * 0.52 * (1 - u) - 10, HEIGHT)
        return [person(1, kps)], []

    h = Harness(FallDetector())
    h.run(6.0, scene=scene)
    h.assert_quiet("sitting is not falling")


def test_walking_and_standing_are_quiet():
    h = Harness(FallDetector())
    h.run(8.0, scene=lambda t: (
        [person(1, synth.standing(200 + t * 30, FEET, HEIGHT, phase=t * 6)),
         person(2, synth.standing(480, FEET, HEIGHT, phase=0.2))], []))
    h.assert_quiet("ordinary movement")


def test_already_lying_down_does_not_fire():
    """Someone who was on the floor before the camera started is not an event."""
    h = Harness(FallDetector())
    h.run(8.0, scene=lambda t: ([person(1, synth.lying(X, FEET, HEIGHT, 86))], []))
    h.assert_quiet("no drop was ever observed")


def test_one_incident_produces_one_alert():
    h = Harness(FallDetector())
    h.run(12.0, scene=_collapse_scene(fall_at=1.0))
    assert len(h.of_type("fall")) == 1, "the rule re-armed while still down"


def test_second_fall_after_recovery_fires_again():
    def scene(t: float):
        if t < 1.0 or 5.0 <= t < 9.0:
            kps = synth.standing(X, FEET, HEIGHT, phase=t * 6.0)
        else:
            kps = synth.lying(X, FEET, HEIGHT, angle_deg=86)
        return [person(1, kps)], []

    h = Harness(FallDetector())
    h.run(14.0, scene=scene)
    assert len(h.of_type("fall")) == 2, (
        "a genuine second fall after a full recovery must alert again")


def test_scale_free_a_distant_person_is_treated_the_same():
    """Same fall, a third the size, far up the frame: identical verdict."""
    small_feet, small_height = 150.0, 62.0

    def scene(t: float):
        if t < 1.0:
            kps = synth.standing(X, small_feet, small_height, phase=t * 6)
        else:
            kps = synth.lying(X, small_feet, small_height, angle_deg=86)
        return [person(1, kps)], []

    h = Harness(FallDetector())
    h.run(6.0, scene=scene)
    assert len(h.of_type("fall")) == 1, (
        "thresholds in body heights should not care about distance")


def test_two_people_falling_are_two_incidents():
    def scene(t: float):
        a = (synth.standing(200, FEET, HEIGHT) if t < 1.0
             else synth.lying(200, FEET, HEIGHT, 86))
        b = (synth.standing(460, FEET, HEIGHT) if t < 2.0
             else synth.lying(460, FEET, HEIGHT, 86))
        return [person(1, a), person(2, b)], []

    h = Harness(FallDetector())
    h.run(7.0, scene=scene)
    falls = h.of_type("fall")
    assert len(falls) == 2
    assert {c.track_id for c in falls} == {1, 2}


def test_pose_free_fallback_uses_the_box_shape():
    """With no keypoints at all, a box that turns wide and stays wide counts."""
    def scene(t: float):
        if t < 1.0:
            box = (300.0, 130.0, 360.0, 330.0)      # tall
        else:
            box = (250.0, 280.0, 450.0, 330.0)      # wide and low
        return [person(1, None, box=box)], []

    h = Harness(FallDetector())
    h.run(6.0, scene=scene)
    assert len(h.of_type("fall")) == 1
    assert h.of_type("fall")[0].meta["torso_angle_deg"] is None
