"""
Zones: the polygons and lines staff draw on a camera.

Two design decisions worth stating.

**Normalised coordinates.** A zone is stored as fractions of the frame
(0..1), not pixels. Draw a restricted area on a 640-wide preview, then switch
the camera to 1280 or point a phone at the same corridor in portrait, and the
zone is still in the right place. It also means the zone editor in the browser
can work in whatever size the canvas happens to be.

**Per-zone overrides.** A corridor and a stairwell do not deserve the same
loitering patience, and the canteen can hold far more people than a lab. Any
threshold from :mod:`config` can be overridden on a single zone through its
``params``, and :meth:`Zone.param` resolves zone value -> active global value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

import config

Point = tuple[float, float]

POLYGON_TYPES = ("restricted", "loiter", "desk", "crowd", "exit")
LINE_TYPES = ("line",)


@dataclass
class Zone:
    """A named region on one camera."""

    id: int
    camera: str
    name: str
    ztype: str
    points: list[list[float]]              # normalised [[x, y], ...]
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    _px_cache: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict, repr=False, compare=False)

    # -- geometry ----------------------------------------------------------- #

    @property
    def is_line(self) -> bool:
        return self.ztype in LINE_TYPES

    @property
    def colour(self) -> str:
        meta = config.ZONE_TYPES.get(self.ztype, {})
        return str(meta.get("colour", "#d9a13b"))

    def polygon(self, width: int, height: int) -> np.ndarray:
        """The zone in pixels, as an int32 array suitable for OpenCV."""
        key = (int(width), int(height))
        cached = self._px_cache.get(key)
        if cached is None:
            pts = [(float(x) * width, float(y) * height) for x, y in self.points]
            cached = np.array(pts, dtype=np.int32)
            self._px_cache[key] = cached
        return cached

    def contains(self, p: Point, width: int, height: int) -> bool:
        """Point-in-polygon, in pixels. Lines contain nothing."""
        if self.is_line or len(self.points) < 3:
            return False
        poly = self.polygon(width, height)
        # measureDist=False returns +1 inside, 0 on the edge, -1 outside. We
        # count the boundary as inside so a foot exactly on the line is caught.
        import cv2
        return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0

    def centroid(self, width: int, height: int) -> Point:
        poly = self.polygon(width, height)
        if len(poly) == 0:
            return (0.0, 0.0)
        return (float(poly[:, 0].mean()), float(poly[:, 1].mean()))

    def side(self, p: Point, width: int, height: int) -> float:
        """Signed side of a counting line, normalised by the line's own length.

        Positive on one side, negative on the other. The magnitude is a
        distance in "line lengths", which is what lets the crossing test use a
        small dead-band instead of reacting to a single jittery pixel.
        """
        if len(self.points) < 2:
            return 0.0
        (x1, y1), (x2, y2) = (
            (self.points[0][0] * width, self.points[0][1] * height),
            (self.points[1][0] * width, self.points[1][1] * height),
        )
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy)) or 1.0
        cross = (p[0] - x1) * dy - (p[1] - y1) * dx
        return float(cross / (length * length))

    def midpoint(self, width: int, height: int) -> Point:
        if len(self.points) < 2:
            return (0.0, 0.0)
        return ((self.points[0][0] + self.points[1][0]) / 2.0 * width,
                (self.points[0][1] + self.points[1][1]) / 2.0 * height)

    # -- thresholds ---------------------------------------------------------- #

    def param(self, key: str, default: Any = None) -> Any:
        """Zone override, else the live global threshold, else ``default``."""
        if key in self.params and self.params[key] not in (None, ""):
            return self.params[key]
        t = config.thresholds()
        if hasattr(t, key):
            return getattr(t, key)
        return default

    # -- serialisation ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "camera": self.camera,
            "name": self.name,
            "type": self.ztype,
            "points": self.points,
            "params": self.params,
            "enabled": self.enabled,
            "colour": self.colour,
        }

    @classmethod
    def from_row(cls, row: Any) -> "Zone":
        """Build from a sqlite row (or any mapping with the same keys)."""
        def get(k: str, default: Any = None) -> Any:
            try:
                return row[k]
            except (KeyError, IndexError, TypeError):
                return default

        pts = get("points_json") or "[]"
        prm = get("params_json") or "{}"
        return cls(
            id=int(get("id", 0) or 0),
            camera=str(get("camera", "")),
            name=str(get("name", "zone")),
            ztype=str(get("type", "restricted")),
            points=json.loads(pts) if isinstance(pts, str) else list(pts),
            params=json.loads(prm) if isinstance(prm, str) else dict(prm),
            enabled=bool(get("enabled", 1)),
        )


class ZoneSet:
    """The zones of one camera, grouped for quick lookup by type."""

    def __init__(self, zones: Iterable[Zone] = ()) -> None:
        self.zones: list[Zone] = [z for z in zones if z.enabled]

    def __iter__(self):
        return iter(self.zones)

    def __len__(self) -> int:
        return len(self.zones)

    def of(self, *types: str) -> list[Zone]:
        return [z for z in self.zones if z.ztype in types]

    def polygons(self) -> list[Zone]:
        return self.of(*POLYGON_TYPES)

    def lines(self) -> list[Zone]:
        return self.of(*LINE_TYPES)

    def containing(self, p: Point, width: int, height: int,
                   *types: str) -> list[Zone]:
        pool = self.of(*types) if types else self.polygons()
        return [z for z in pool if z.contains(p, width, height)]

    def by_id(self, zone_id: int) -> Zone | None:
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    def to_list(self) -> list[dict[str, Any]]:
        return [z.to_dict() for z in self.zones]


def validate_points(ztype: str, points: Any) -> list[list[float]]:
    """Reject a malformed zone at the door, with a message a human can act on."""
    if not isinstance(points, (list, tuple)):
        raise ValueError("points must be a list of [x, y] pairs")
    cleaned: list[list[float]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("each point must be a pair of numbers")
        x, y = float(p[0]), float(p[1])
        if not (-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
            raise ValueError("points must be normalised to the 0..1 frame")
        cleaned.append([min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)])

    need = 2 if ztype in LINE_TYPES else 3
    if len(cleaned) < need:
        word = "a line needs 2 points" if need == 2 else "a polygon needs at least 3"
        raise ValueError(f"not enough points: {word}")
    if ztype in LINE_TYPES:
        cleaned = cleaned[:2]
    return cleaned
