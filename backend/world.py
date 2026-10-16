"""
What one frame looks like after the models have had their say.

:class:`Detection` and :class:`FrameWorld` are the only vocabulary shared
between the vision layer, the renderer and the detection rules. Keeping that
vocabulary small and boring is deliberate: a rule that can only see tracks,
boxes and keypoints cannot accidentally start depending on pixels, and the
privacy claim at the centre of this project stays true by construction rather
than by discipline.

Note what is *not* in here: no crop, no image, no embedding, no identity.
A :class:`Detection` carries a track id that was invented a moment ago, will be
forgotten within seconds of the person leaving, and means nothing in any other
camera or on any other day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.vision import skeleton as sk

Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass
class Detection:
    """One tracked thing in one frame."""

    track_id: int
    label: str
    conf: float
    box: Box
    keypoints: np.ndarray | None = None

    # -- geometry (cheap, called a lot, so memoised per frame) ------------- #
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def _memo(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def is_person(self) -> bool:
        return self.label == "person"

    @property
    def has_pose(self) -> bool:
        return (self.keypoints is not None
                and sk.visible_fraction(self.keypoints) > 0.25)

    def foot(self) -> Point:
        """Where it touches the ground — the point every zone test uses."""
        return self._memo("foot", lambda: sk.foot_point(self.box, self.keypoints))

    def center(self) -> Point:
        return self._memo("center", lambda: sk.center(self.box))

    def hip(self) -> Point | None:
        return self._memo("hip", lambda: sk.hip_center(self.keypoints))

    def head(self) -> Point | None:
        return self._memo("head", lambda: sk.head_point(self.keypoints))

    def scale(self) -> float:
        """This subject's size in pixels; the denominator of every threshold."""
        return self._memo("scale", lambda: sk.body_height(self.box, self.keypoints))

    def torso_angle(self) -> float | None:
        return self._memo("torso", lambda: sk.torso_angle_deg(self.keypoints))

    def aspect(self) -> float:
        return self._memo("aspect", lambda: sk.aspect_ratio(self.box))

    def diagonal(self) -> float:
        return self._memo("diag", lambda: sk.diagonal(self.box))

    def area(self) -> float:
        return sk.box_width(self.box) * sk.box_height(self.box)

    def distance_to(self, p: Point) -> float:
        c = self.center()
        return float(math.hypot(c[0] - p[0], c[1] - p[1]))

    def to_dict(self, include_keypoints: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.track_id,
            "label": self.label,
            "conf": round(float(self.conf), 3),
            "box": [round(float(v), 1) for v in self.box],
        }
        if include_keypoints and self.keypoints is not None:
            d["keypoints"] = [[round(float(x), 1), round(float(y), 1),
                               round(float(c), 2)] for x, y, c in self.keypoints]
        return d


@dataclass
class FrameWorld:
    """Everything known about one frame of one camera."""

    camera: str
    ts: float
    frame_idx: int
    width: int
    height: int
    persons: list[Detection] = field(default_factory=list)
    objects: list[Detection] = field(default_factory=list)
    simulated: bool = False
    note: str = ""

    @property
    def everything(self) -> list[Detection]:
        return self.persons + self.objects

    def by_label(self, labels: tuple[str, ...]) -> list[Detection]:
        return [d for d in self.objects if d.label in labels]

    def person(self, track_id: int) -> Detection | None:
        for p in self.persons:
            if p.track_id == track_id:
                return p
        return None

    def to_dict(self, include_keypoints: bool = False) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "ts": self.ts,
            "frame": self.frame_idx,
            "size": [self.width, self.height],
            "simulated": self.simulated,
            "persons": [p.to_dict(include_keypoints) for p in self.persons],
            "objects": [o.to_dict() for o in self.objects],
        }
