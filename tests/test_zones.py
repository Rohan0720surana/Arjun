"""Restricted entry, loitering and crowd density."""

from __future__ import annotations

import config
from backend.detectors.zones import (CrowdDetector, LoiterDetector,
                                     RestrictedZoneDetector)
from backend.vision import synth
from conftest import Harness, make_zone, person, standing_person

# A zone covering the right half of the frame, on the floor.
RIGHT_HALF = [[0.5, 0.4], [1.0, 0.4], [1.0, 1.0], [0.5, 1.0]]
FEET, HEIGHT = 330.0, 160.0


# --------------------------------------------------------------------------- #
# Restricted
# --------------------------------------------------------------------------- #

def test_restricted_fires_once_on_entry():
    zone = make_zone("restricted", RIGHT_HALF, name="Server room")
    h = Harness(RestrictedZoneDetector(), zones=[zone])
    # Walk from the left half into the right half and stay.
    h.run(6.0, scene=lambda t: ([standing_person(1, 120 + t * 90, FEET, HEIGHT)], []))

    hits = h.of_type("restricted")
    assert len(hits) == 1, f"expected one entry alert, got {len(hits)}"
    assert hits[0].zone is zone
    assert "Server room" in hits[0].message


def test_restricted_ignores_a_foot_clipping_the_edge():
    """In and straight out again inside the grace period is not an entry."""
    zone = make_zone("restricted", RIGHT_HALF)

    def scene(t: float):
        # 0.3s just inside the boundary, then back out.
        x = 330.0 if 1.0 <= t < 1.3 else 290.0
        return [standing_person(1, x, FEET, HEIGHT)], []

    h = Harness(RestrictedZoneDetector(), zones=[zone])
    h.run(4.0, scene=scene)
    h.assert_quiet("a momentary clip of the boundary")


def test_restricted_stays_outside_is_quiet():
    zone = make_zone("restricted", RIGHT_HALF)
    h = Harness(RestrictedZoneDetector(), zones=[zone])
    h.run(6.0, scene=lambda t: ([standing_person(1, 200, FEET, HEIGHT)], []))
    h.assert_quiet("person never entered")


def test_restricted_re_arms_after_leaving():
    zone = make_zone("restricted", RIGHT_HALF)

    def scene(t: float):
        inside = t < 3.0 or t > 8.0
        return [standing_person(1, 420 if inside else 150, FEET, HEIGHT)], []

    h = Harness(RestrictedZoneDetector(), zones=[zone])
    h.run(12.0, scene=scene)
    assert len(h.of_type("restricted")) == 2, "a second entry is a second event"


def test_restricted_uses_feet_not_box_centre():
    """Someone leaning over a line is not standing behind it.

    The zone is a band across the far side of the room — the space beyond a
    counter, say. A person standing in front of it with an arm raised has most
    of their body, and therefore their box centre, inside the band, while both
    feet remain on the public side of the line.
    """
    zone = make_zone("restricted", [[0.0, 0.0], [1.0, 0.0], [1.0, 0.72],
                                    [0.0, 0.72]], name="Behind the counter")

    def scene(t: float):
        kps = synth.standing(330.0, FEET, HEIGHT, arms="up_left")
        return [person(1, kps)], []

    h = Harness(RestrictedZoneDetector(), zones=[zone])
    h.run(4.0, scene=scene)
    ctx = h.last_ctx
    det = ctx.world.persons[0]
    cx, cy = det.center()
    assert zone.contains((cx, cy), ctx.width, ctx.height), (
        "test is only meaningful if the box centre really is inside")
    h.assert_quiet("feet were outside the zone")


# --------------------------------------------------------------------------- #
# Loitering
# --------------------------------------------------------------------------- #

def test_loiter_fires_after_the_configured_patience():
    zone = make_zone("loiter", RIGHT_HALF, name="West alcove")
    h = Harness(LoiterDetector(), zones=[zone])
    limit = config.thresholds().loiter_seconds

    h.run(limit - 2.0, scene=lambda t: ([standing_person(1, 420, FEET, HEIGHT)], []))
    assert not h.fired, "fired early"

    h.run(4.0, scene=lambda t: ([standing_person(1, 420, FEET, HEIGHT)], []))
    hits = h.of_type("loiter")
    assert len(hits) == 1
    assert hits[0].meta["dwell_seconds"] >= limit


def test_loiter_forgives_a_brief_step_out():
    """Pacing in and out of a doorway is still waiting there."""
    zone = make_zone("loiter", RIGHT_HALF)
    limit = config.thresholds().loiter_seconds

    def scene(t: float):
        # Step outside for 0.8s every five seconds - inside the exit grace.
        outside = (t % 5.0) < 0.8
        return [standing_person(1, 200 if outside else 420, FEET, HEIGHT)], []

    h = Harness(LoiterDetector(), zones=[zone])
    h.run(limit + 4.0, scene=scene)
    assert h.of_type("loiter"), "brief exits should not reset the clock"


def test_loiter_resets_after_a_real_departure():
    zone = make_zone("loiter", RIGHT_HALF)
    limit = config.thresholds().loiter_seconds

    def scene(t: float):
        # Leave for four seconds, well past the grace, then come back.
        away = (limit * 0.6) <= t < (limit * 0.6 + 4.0)
        return [standing_person(1, 180 if away else 420, FEET, HEIGHT)], []

    h = Harness(LoiterDetector(), zones=[zone])
    h.run(limit + 3.0, scene=scene)
    h.assert_quiet("the clock should have restarted on return")


def test_loiter_zone_can_override_the_global_patience():
    zone = make_zone("loiter", RIGHT_HALF, name="Impatient", loiter_seconds=5.0)
    h = Harness(LoiterDetector(), zones=[zone])
    h.run(7.0, scene=lambda t: ([standing_person(1, 420, FEET, HEIGHT)], []))
    assert h.of_type("loiter"), "per-zone override was ignored"
    assert h.of_type("loiter")[0].meta["threshold_seconds"] == 5.0


def test_loiter_repeats_at_most_once_per_period():
    zone = make_zone("loiter", RIGHT_HALF, loiter_seconds=4.0)
    h = Harness(LoiterDetector(), zones=[zone])
    h.run(21.0, scene=lambda t: ([standing_person(1, 420, FEET, HEIGHT)], []))
    n = len(h.of_type("loiter"))
    assert 3 <= n <= 5, f"expected a handful of reminders over 21s, got {n}"


# --------------------------------------------------------------------------- #
# Crowd
# --------------------------------------------------------------------------- #

def _crowd(n: int, inside: bool = True):
    xs = [360 + i * 24 for i in range(n)] if inside else [40 + i * 18
                                                          for i in range(n)]
    return [standing_person(i + 1, x, FEET, HEIGHT) for i, x in enumerate(xs)]


def test_crowd_warning_then_critical():
    zone = make_zone("crowd", RIGHT_HALF, name="Atrium", crowd_warn=3,
                     crowd_critical=5)
    h = Harness(CrowdDetector(), zones=[zone])

    h.run(4.0, scene=lambda t: (_crowd(3), []))
    assert [c.type for c in h.fired] == ["crowd_warning"]

    h.run(4.0, scene=lambda t: (_crowd(6), []))
    assert "crowd_critical" in h.types()
    crit = h.of_type("crowd_critical")[0]
    assert crit.meta["count"] == 6
    assert crit.resolved_severity() == "critical"


def test_crowd_must_hold_to_count():
    """A group walking through is not a crowd gathering."""
    zone = make_zone("crowd", RIGHT_HALF, crowd_warn=3, crowd_critical=5)

    def scene(t: float):
        # Six people present for only one second, less than the sustain time.
        return (_crowd(6) if 1.0 <= t < 2.0 else _crowd(1)), []

    h = Harness(CrowdDetector(), zones=[zone])
    h.run(6.0, scene=scene)
    h.assert_quiet("the number never held")


def test_crowd_counts_only_people_inside_the_zone():
    zone = make_zone("crowd", RIGHT_HALF, crowd_warn=3, crowd_critical=9)
    h = Harness(CrowdDetector(), zones=[zone])
    h.run(6.0, scene=lambda t: (_crowd(6, inside=False), []))
    h.assert_quiet("everyone was outside the zone")


def test_crowd_re_arms_after_dispersing():
    zone = make_zone("crowd", RIGHT_HALF, crowd_warn=3, crowd_critical=9)

    def scene(t: float):
        busy = t < 4.0 or t > 9.0
        return (_crowd(4) if busy else _crowd(1)), []

    h = Harness(CrowdDetector(), zones=[zone])
    h.run(14.0, scene=scene)
    assert len(h.of_type("crowd_warning")) == 2, (
        "a zone that empties and fills again is a second gathering")
