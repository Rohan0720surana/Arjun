"""
Test harness for the detection rules.

Every rule in ARJUN is a pure function of track history, which is what makes
this possible: a detector can be driven by a scripted sequence of synthetic
poses, on a virtual clock, with no camera, no models and no waiting. A thirty
second loitering timer is tested in about two milliseconds.

The clock is the important part. :class:`Harness` advances time by a fixed
step per frame rather than sleeping, so the tests exercise the real thresholds
at their real values — not shortened "test-friendly" ones that would prove
nothing about what runs on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from backend.camera_manager import CameraFlags             # noqa: E402
from backend.detectors.base import Candidate, Detector     # noqa: E402
from backend.state import CameraContext, TrackStore        # noqa: E402
from backend.vision import synth                           # noqa: E402
from backend.world import Detection, FrameWorld            # noqa: E402
from backend.zone_store import Zone, ZoneSet               # noqa: E402

WIDTH, HEIGHT = 640, 360
START = 1_700_000_000.0          # a fixed, arbitrary epoch for repeatability


def make_zone(ztype: str, points: Sequence[Sequence[float]], zid: int = 1,
              name: str | None = None, **params: Any) -> Zone:
    return Zone(id=zid, camera="Test", name=name or f"{ztype} {zid}",
                ztype=ztype, points=[[float(x), float(y)] for x, y in points],
                params=params)


def full_frame_zone(ztype: str, zid: int = 1, **params: Any) -> Zone:
    return make_zone(ztype, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                     zid, **params)


def person(track_id: int, keypoints: np.ndarray | None = None,
           box: tuple[float, float, float, float] | None = None,
           conf: float = 0.9) -> Detection:
    if box is None:
        box = synth.box_for(keypoints) if keypoints is not None else (0, 0, 10, 10)
    return Detection(track_id=track_id, label="person", conf=conf,
                     box=box, keypoints=keypoints)


def standing_person(track_id: int, x: float, feet_y: float = 330.0,
                    height: float = 200.0, **kw: Any) -> Detection:
    return person(track_id, synth.standing(x, feet_y, height, **kw))


def thing(track_id: int, label: str, cx: float, cy: float,
          size: float = 40.0, conf: float = 0.7) -> Detection:
    half = size / 2.0
    return Detection(track_id=track_id, label=label, conf=conf,
                     box=(cx - half, cy - half, cx + half, cy + half))


class Harness:
    """Drives one detector over a scripted sequence of frames."""

    def __init__(
        self,
        detector: Detector,
        zones: Iterable[Zone] = (),
        flags: CameraFlags | None = None,
        width: int = WIDTH,
        height: int = HEIGHT,
        simulated: bool = False,
        fps: float = 15.0,
        camera: str = "Test",
    ) -> None:
        self.detector = detector
        self.zones = ZoneSet(list(zones))
        self.flags = flags or CameraFlags()
        self.width, self.height = width, height
        self.simulated = simulated
        self.dt = 1.0 / fps
        self.camera = camera
        self.now = START
        self.tracks = TrackStore()
        self.frame = 0
        self.fired: list[Candidate] = []
        self.last_ctx: CameraContext | None = None

    # -- one frame ----------------------------------------------------------- #

    def step(self, persons: Iterable[Detection] = (),
             objects: Iterable[Detection] = (),
             dt: float | None = None) -> list[Candidate]:
        self.now += self.dt if dt is None else dt
        self.frame += 1
        world = FrameWorld(
            camera=self.camera, ts=self.now, frame_idx=self.frame,
            width=self.width, height=self.height,
            persons=list(persons), objects=list(objects),
            simulated=self.simulated,
        )
        self.tracks.update(world)
        ctx = CameraContext(
            camera=self.camera, world=world, tracks=self.tracks,
            zones=self.zones, flags=self.flags, now=self.now,
            simulated=self.simulated,
        )
        ctx.compute_memberships()
        ctx.update_zone_timers()
        self.last_ctx = ctx
        out = self.detector.safe_update(ctx)
        self.fired.extend(out)
        return out

    # -- many frames ----------------------------------------------------------- #

    def run(self, seconds: float,
            scene: Callable[[float], tuple[Iterable[Detection],
                                           Iterable[Detection]]] | None = None,
            persons: Iterable[Detection] = (),
            objects: Iterable[Detection] = ()) -> list[Candidate]:
        """Run for a span of virtual seconds.

        ``scene(t)`` is called with seconds elapsed since this call began and
        returns ``(persons, objects)`` for that moment; without it the same
        static detections are replayed.
        """
        fired: list[Candidate] = []
        started = self.now
        while self.now - started < seconds:
            if scene is not None:
                p, o = scene(self.now - started + self.dt)
            else:
                p, o = persons, objects
            fired.extend(self.step(p, o))
        return fired

    # -- assertions ------------------------------------------------------------ #

    def types(self) -> list[str]:
        return [c.type for c in self.fired]

    def of_type(self, etype: str) -> list[Candidate]:
        return [c for c in self.fired if c.type == etype]

    def assert_quiet(self, message: str = "") -> None:
        assert not self.fired, (
            f"expected no alerts{': ' + message if message else ''}, got "
            f"{[(c.type, c.message) for c in self.fired]}")

    def elapsed(self) -> float:
        return self.now - START


@pytest.fixture(autouse=True)
def live_thresholds():
    """Tests run against the real thresholds, never the demo ones."""
    was = config.demo_mode()
    config.set_demo_mode(False)
    yield config.thresholds()
    config.set_demo_mode(was)


@pytest.fixture
def flags() -> CameraFlags:
    return CameraFlags()
