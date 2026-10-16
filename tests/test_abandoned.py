"""Unattended bags, and fire exits that are not clear."""

from __future__ import annotations

import config
from backend.detectors.abandoned import AbandonedObjectDetector
from backend.detectors.exit_blocked import ExitBlockedDetector
from conftest import Harness, full_frame_zone, make_zone, standing_person, thing

FEET, HEIGHT = 330.0, 160.0
BAG_X, BAG_Y = 200.0, 300.0


def _bag(x: float = BAG_X, y: float = BAG_Y):
    return thing(50, "backpack", x, y, size=36)


# --------------------------------------------------------------------------- #
# Abandoned bags
# --------------------------------------------------------------------------- #

def test_bag_left_alone_is_reported():
    limit = config.thresholds().abandoned_seconds
    h = Harness(AbandonedObjectDetector())
    h.run(limit + 3.0, scene=lambda t: ([], [_bag()]))

    hits = h.of_type("abandoned")
    assert len(hits) == 1
    assert hits[0].meta["object"] == "backpack"
    assert hits[0].meta["unattended_seconds"] >= limit


def test_a_carried_bag_is_never_abandoned():
    """It is moving, so the rule never starts its clock."""
    limit = config.thresholds().abandoned_seconds
    h = Harness(AbandonedObjectDetector())
    h.run(limit + 5.0, scene=lambda t: (
        [standing_person(1, 100 + t * 14, FEET, HEIGHT)],
        [thing(50, "backpack", 100 + t * 14, 290, size=36)]))
    h.assert_quiet("a bag being carried")


def test_bag_with_its_owner_beside_it_is_fine():
    limit = config.thresholds().abandoned_seconds
    h = Harness(AbandonedObjectDetector())
    # Person stands right next to the bag the whole time.
    h.run(limit + 5.0, scene=lambda t: (
        [standing_person(1, BAG_X + 30, FEET, HEIGHT)], [_bag()]))
    h.assert_quiet("somebody is with the bag")


def test_someone_walking_past_does_not_reclaim_the_bag():
    """The crucial case: a busy corridor must not reset the clock forever."""
    limit = config.thresholds().abandoned_seconds

    def scene(t: float):
        # A pedestrian sweeps past the bag roughly every six seconds.
        x = 40 + ((t * 110) % 560)
        return [standing_person(1, x, FEET, HEIGHT)], [_bag()]

    h = Harness(AbandonedObjectDetector())
    h.run(limit + 6.0, scene=scene)
    assert h.of_type("abandoned"), (
        "passers-by should not count as taking custody of the bag")


def test_the_owner_coming_back_cancels_it():
    limit = config.thresholds().abandoned_seconds

    def scene(t: float):
        # Away for most of the window, back beside the bag well before time.
        near = t > limit * 0.5
        return [standing_person(1, BAG_X + 25 if near else 600, FEET, HEIGHT)], \
               [_bag()]

    h = Harness(AbandonedObjectDetector())
    h.run(limit + 6.0, scene=scene)
    h.assert_quiet("the owner returned and stayed")


def test_bag_alerts_only_once():
    limit = config.thresholds().abandoned_seconds
    h = Harness(AbandonedObjectDetector())
    h.run(limit * 2 + 10.0, scene=lambda t: ([], [_bag()]))
    assert len(h.of_type("abandoned")) == 1


def test_only_bag_classes_count():
    limit = config.thresholds().abandoned_seconds
    h = Harness(AbandonedObjectDetector())
    h.run(limit + 4.0, scene=lambda t: ([], [thing(51, "chair", 300, 300)]))
    h.assert_quiet("a chair standing still is furniture")


# --------------------------------------------------------------------------- #
# Fire exits
# --------------------------------------------------------------------------- #

EXIT = [[0.6, 0.6], [0.95, 0.6], [0.95, 1.0], [0.6, 1.0]]


def test_object_left_in_a_fire_exit():
    zone = make_zone("exit", EXIT, name="North stairwell")
    limit = config.thresholds().exit_blocked_seconds
    h = Harness(ExitBlockedDetector(), zones=[zone])
    h.run(limit + 3.0, scene=lambda t: ([], [thing(60, "suitcase", 500, 300)]))

    hits = h.of_type("exit_blocked")
    assert len(hits) == 1
    assert hits[0].meta["kind"] == "object"
    assert hits[0].resolved_severity() == "critical"
    assert "North stairwell" in hits[0].message


def test_person_standing_in_a_fire_exit():
    zone = make_zone("exit", EXIT, name="North stairwell")
    limit = config.thresholds().exit_blocked_seconds
    h = Harness(ExitBlockedDetector(), zones=[zone])
    h.run(limit + 3.0, scene=lambda t: ([standing_person(1, 500, FEET, HEIGHT)], []))

    hits = h.of_type("exit_blocked")
    assert len(hits) == 1
    assert hits[0].meta["kind"] == "person"


def test_walking_through_a_fire_exit_is_fine():
    """Which is, after all, the entire purpose of a fire exit."""
    zone = make_zone("exit", EXIT, name="North stairwell")
    limit = config.thresholds().exit_blocked_seconds

    def scene(t: float):
        # Cross the zone steadily, back and forth, never standing still.
        x = 400 + 180 * abs(((t / 3.0) % 2.0) - 1.0)
        return [standing_person(1, x, FEET, HEIGHT)], []

    h = Harness(ExitBlockedDetector(), zones=[zone])
    h.run(limit + 6.0, scene=scene)
    h.assert_quiet("someone passing through")


def test_exit_blocked_ignores_things_outside_the_zone():
    zone = make_zone("exit", EXIT)
    limit = config.thresholds().exit_blocked_seconds
    h = Harness(ExitBlockedDetector(), zones=[zone])
    h.run(limit + 3.0, scene=lambda t: ([standing_person(1, 100, FEET, HEIGHT)],
                                        [thing(60, "suitcase", 120, 300)]))
    h.assert_quiet("nothing was in the exit")


def test_exit_zone_can_tighten_its_own_limit():
    zone = make_zone("exit", EXIT, name="Strict", exit_blocked_seconds=4.0)
    h = Harness(ExitBlockedDetector(), zones=[zone])
    h.run(6.0, scene=lambda t: ([], [thing(60, "suitcase", 500, 300)]))
    hits = h.of_type("exit_blocked")
    assert hits and hits[0].meta["threshold_seconds"] == 4.0
