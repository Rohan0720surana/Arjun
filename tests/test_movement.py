"""Panic movement, line counting and after-hours occupancy."""

from __future__ import annotations

import time

import config
from backend.detectors.after_hours import AfterHoursDetector, is_after_hours
from backend.detectors.line_counter import LineCounter
from backend.detectors.running import PanicMovementDetector, _circular_stats
from conftest import Harness, make_zone, standing_person

FEET, HEIGHT = 330.0, 150.0


# --------------------------------------------------------------------------- #
# Panic
# --------------------------------------------------------------------------- #

def _runners(n: int, t: float, speed_px: float, direction: int = 1,
             spread: bool = False):
    people = []
    for i in range(n):
        heading = direction if not spread or i % 2 == 0 else -direction
        x = 320 + heading * speed_px * t + i * 20
        people.append(standing_person(i + 1, x, FEET, HEIGHT))
    return people


def test_several_people_running_together_is_panic():
    # 150px body height, so 2.2 body heights per second is 330 px/s.
    h = Harness(PanicMovementDetector())
    h.run(3.0, scene=lambda t: (_runners(3, t, 400.0), []))

    hits = h.of_type("panic")
    assert len(hits) == 1
    assert hits[0].meta["runners"] >= 3
    assert hits[0].resolved_severity() == "critical"
    assert hits[0].meta["heading_spread_deg"] < 70


def test_walking_pace_is_not_panic():
    h = Harness(PanicMovementDetector())
    h.run(4.0, scene=lambda t: (_runners(4, t, 120.0), []))
    h.assert_quiet("ordinary walking speed")


def test_one_person_running_is_not_panic():
    """Somebody late for a lecture is not an emergency."""
    h = Harness(PanicMovementDetector())
    h.run(4.0, scene=lambda t: (_runners(1, t, 420.0), []))
    h.assert_quiet("a single runner")


def test_people_running_in_different_directions_is_not_panic():
    h = Harness(PanicMovementDetector())
    h.run(4.0, scene=lambda t: (_runners(4, t, 400.0, spread=True), []))
    h.assert_quiet("headings disagree, so there is no common cause")


def test_panic_reports_once_per_episode():
    h = Harness(PanicMovementDetector())
    h.run(8.0, scene=lambda t: (_runners(3, t, 400.0), []))
    assert len(h.of_type("panic")) == 1


def test_circular_mean_handles_the_wraparound():
    """350 degrees and 10 degrees are 20 degrees apart, not 340."""
    mean, spread = _circular_stats([350.0, 10.0])
    assert abs(mean) < 1e-6 or abs(abs(mean) - 360) < 1e-6
    assert spread < 11.0


# --------------------------------------------------------------------------- #
# Line counting
# --------------------------------------------------------------------------- #

VERTICAL_LINE = [[0.5, 0.3], [0.5, 1.0]]


def test_counts_a_crossing_in_each_direction():
    zone = make_zone("line", VERTICAL_LINE, name="Main gate")
    det = LineCounter()
    h = Harness(det, zones=[zone])

    # Left to right, then right to left.
    h.run(4.0, scene=lambda t: ([standing_person(1, 180 + t * 120, FEET, HEIGHT)], []))
    h.run(4.0, scene=lambda t: ([standing_person(1, 660 - t * 120, FEET, HEIGHT)], []))

    crossings = h.of_type("line_cross")
    assert len(crossings) == 2, f"expected two crossings, got {len(crossings)}"
    assert {c.meta["direction"] for c in crossings} == {"in", "out"}
    assert det.inside(zone.id) == 0, "one in and one out leaves nobody inside"


def test_inside_count_tracks_net_crossings():
    zone = make_zone("line", VERTICAL_LINE)
    det = LineCounter()
    h = Harness(det, zones=[zone])

    for i in range(3):
        h.run(3.0, scene=lambda t, i=i: (
            [standing_person(i + 1, 200 + t * 120, FEET, HEIGHT)], []))
    assert det.inside(zone.id) == 3


def test_standing_on_the_line_is_not_a_stream_of_crossings():
    """The dead-band earning its keep against one-pixel tracker jitter."""
    zone = make_zone("line", VERTICAL_LINE)
    det = LineCounter()
    h = Harness(det, zones=[zone])

    def scene(t: float):
        # Hover within a pixel of the line for ten seconds.
        x = 320 + (1.0 if int(t * 15) % 2 else -1.0)
        return [standing_person(1, x, FEET, HEIGHT)], []

    h.run(10.0, scene=scene)
    assert not h.of_type("line_cross"), "jitter on the threshold was counted"


def test_a_brand_new_track_cannot_count_immediately():
    zone = make_zone("line", VERTICAL_LINE)
    h = Harness(LineCounter(), zones=[zone])
    # A track that appears already past the line, as happens when ByteTrack
    # hands out a fresh id to somebody mid-stride.
    h.step([standing_person(9, 200, FEET, HEIGHT)])
    h.step([standing_person(9, 500, FEET, HEIGHT)])
    assert not h.of_type("line_cross")


def test_counters_are_published_for_the_dashboard():
    zone = make_zone("line", VERTICAL_LINE, name="Main gate")
    h = Harness(LineCounter(), zones=[zone])
    h.run(3.0, scene=lambda t: ([standing_person(1, 200 + t * 120, FEET, HEIGHT)], []))
    counters = h.last_ctx.shared["counters"]
    assert counters[str(zone.id)]["zone"] == "Main gate"
    assert counters[str(zone.id)]["inside"] == 1


# --------------------------------------------------------------------------- #
# After hours
# --------------------------------------------------------------------------- #

def test_opening_hours_window():
    def at(hour: int, minute: int = 0) -> float:
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0,
                            lt.tm_wday, lt.tm_yday, lt.tm_isdst))

    assert not is_after_hours(at(12), "07:00", "21:00")
    assert is_after_hours(at(23), "07:00", "21:00")
    assert is_after_hours(at(3), "07:00", "21:00")
    # A window that spans midnight: open 20:00 to 06:00.
    assert not is_after_hours(at(23), "20:00", "06:00")
    assert not is_after_hours(at(2), "20:00", "06:00")
    assert is_after_hours(at(12), "20:00", "06:00")
    # Equal times mean open around the clock.
    assert not is_after_hours(at(3), "00:00", "00:00")


def test_person_after_hours_raises_a_warning(flags):
    flags.after_hours_override = True
    h = Harness(AfterHoursDetector(), flags=flags)
    grace = config.thresholds().after_hours_grace_s
    h.run(grace + 3.0, scene=lambda t: ([standing_person(1, 300, FEET, HEIGHT)], []))

    hits = h.of_type("after_hours")
    assert len(hits) == 1
    assert hits[0].resolved_severity() == "warning"
    assert hits[0].meta["forced"] is True


def test_a_quick_pass_within_the_grace_is_ignored(flags):
    flags.after_hours_override = True
    h = Harness(AfterHoursDetector(), flags=flags)
    grace = config.thresholds().after_hours_grace_s

    def scene(t: float):
        return ([standing_person(1, 300, FEET, HEIGHT)] if t < grace * 0.4
                else []), []

    h.run(grace * 0.4 + 0.5, scene=scene)
    assert not h.of_type("after_hours")


def test_empty_room_after_hours_suggests_the_lights(flags):
    flags.after_hours_override = True
    h = Harness(AfterHoursDetector(), flags=flags)
    empty = config.thresholds().after_hours_empty_room_s
    h.run(empty + 4.0, scene=lambda t: ([], []))

    hints = h.of_type("lights_on")
    assert len(hints) == 1
    assert hints[0].resolved_severity() == "info"
    assert "empty" in hints[0].message


def test_nothing_happens_during_opening_hours(flags):
    """With the override off and hours wide open, the rule is silent."""
    flags.after_hours_override = False
    h = Harness(AfterHoursDetector(), flags=flags)
    config.set_demo_mode(False)
    t = config.thresholds()
    object.__setattr__(t, "after_hours_open", "00:00")
    object.__setattr__(t, "after_hours_close", "00:00")
    try:
        h.run(30.0, scene=lambda _t: ([standing_person(1, 300, FEET, HEIGHT)], []))
        h.assert_quiet("the building is open")
    finally:
        object.__setattr__(t, "after_hours_open", "07:00")
        object.__setattr__(t, "after_hours_close", "21:00")
