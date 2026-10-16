"""
A parametric human, in keypoints.

This module can draw a plausible COCO-17 skeleton for a person standing,
walking, sitting at a desk, raising a hand or lying on the floor, from nothing
but a position and a height in pixels. It has two users:

* **the test suite**, which feeds detectors a scripted sequence of poses and
  asserts that the rule fires at the right moment and not before. Writing a
  fall test as "stand for two seconds, then lie down, then wait" is only
  readable because the poses themselves are one function call;
* **the simulated camera scenes**, which need people to move around.

The proportions below are the standard artist's canon — a figure about seven
and a half heads tall — expressed as fractions of total standing height, so a
figure 90 px tall at the far end of a corridor is the same shape as one 400 px
tall in the foreground.
"""

from __future__ import annotations

import math

import numpy as np

from backend.vision import skeleton as sk

# Vertical position of each joint, as a fraction of standing height measured up
# from the soles of the feet.
_UP = {
    "head": 0.935, "ear": 0.930, "eye": 0.945, "nose": 0.920,
    "shoulder": 0.820, "elbow": 0.655, "wrist": 0.500,
    "hip": 0.520, "knee": 0.265, "ankle": 0.025,
}
# Horizontal half-offsets, also as fractions of standing height.
_OUT = {
    "ear": 0.038, "eye": 0.022, "shoulder": 0.105, "elbow": 0.128,
    "wrist": 0.140, "hip": 0.072, "knee": 0.062, "ankle": 0.055,
}


def standing(
    cx: float,
    feet_y: float,
    height: float,
    phase: float = 0.0,
    facing: int = 1,
    arms: str = "down",
    conf: float = 0.92,
) -> np.ndarray:
    """A standing or walking figure.

    ``phase`` advances the walk cycle in radians; 0 is mid-stride. ``arms`` is
    one of ``down``, ``up_left``, ``up_right``, ``both_up`` or ``folded``.
    """
    h = float(height)
    swing = math.sin(phase)
    bob = abs(math.sin(phase * 2.0)) * 0.012 * h
    base = feet_y - bob

    def y(key: str) -> float:
        return base - _UP[key] * h

    def x(key: str, side: int) -> float:
        return cx + side * _OUT[key] * h

    leg = swing * 0.085 * h * facing
    arm = -swing * 0.070 * h * facing

    pts: dict[str, tuple[float, float]] = {
        "nose": (cx + facing * 0.018 * h, y("nose")),
        "left_eye": (x("eye", 1) + facing * 0.012 * h, y("eye")),
        "right_eye": (x("eye", -1) + facing * 0.012 * h, y("eye")),
        "left_ear": (x("ear", 1), y("ear")),
        "right_ear": (x("ear", -1), y("ear")),
        "left_shoulder": (x("shoulder", 1), y("shoulder")),
        "right_shoulder": (x("shoulder", -1), y("shoulder")),
        "left_hip": (x("hip", 1), y("hip")),
        "right_hip": (x("hip", -1), y("hip")),
        "left_knee": (x("knee", 1) + leg * 0.55, y("knee")),
        "right_knee": (x("knee", -1) - leg * 0.55, y("knee")),
        "left_ankle": (x("ankle", 1) + leg, y("ankle")),
        "right_ankle": (x("ankle", -1) - leg, y("ankle")),
    }

    if arms == "both_up":
        sides = (1, -1)
    elif arms == "up_left":
        sides = (1,)
    elif arms == "up_right":
        sides = (-1,)
    else:
        sides = ()

    for side, tag in ((1, "left"), (-1, "right")):
        if side in sides:
            # Hand up beside the head, elbow out — the Signal for Help posture.
            pts[f"{tag}_elbow"] = (x("shoulder", side) + side * 0.075 * h,
                                   y("shoulder") - 0.055 * h)
            pts[f"{tag}_wrist"] = (x("shoulder", side) + side * 0.055 * h,
                                   y("shoulder") - 0.185 * h)
        elif arms == "folded":
            pts[f"{tag}_elbow"] = (x("elbow", side), y("elbow"))
            pts[f"{tag}_wrist"] = (cx + side * 0.03 * h, y("elbow") - 0.02 * h)
        else:
            pts[f"{tag}_elbow"] = (x("elbow", side) + arm * 0.5, y("elbow"))
            pts[f"{tag}_wrist"] = (x("wrist", side) + arm, y("wrist"))

    return sk.from_xy(pts, conf)


def sitting(
    cx: float,
    seat_y: float,
    height: float,
    yaw: float = 0.0,
    lean: float = 0.0,
    conf: float = 0.9,
) -> np.ndarray:
    """A figure seated at a desk, seen from the front.

    ``yaw`` in degrees swings the head sideways, which is what exam mode's
    head-turn rule measures. ``lean`` tilts the torso.
    """
    h = float(height)
    hip_y = seat_y
    sho_y = hip_y - 0.30 * h
    lean_dx = math.sin(math.radians(lean)) * 0.12 * h
    yaw_r = math.radians(yaw)
    head_dx = math.sin(yaw_r) * 0.10 * h
    # As the head turns away, the far ear disappears behind it — mirroring what
    # a real face mesh reports, so the yaw estimate is exercised honestly.
    ear_vis_l = conf * max(0.1, math.cos(yaw_r - 0.5))
    ear_vis_r = conf * max(0.1, math.cos(yaw_r + 0.5))

    pts = {
        "nose": (cx + lean_dx + head_dx * 1.3, sho_y - 0.14 * h),
        "left_eye": (cx + lean_dx + head_dx * 1.2 + 0.022 * h, sho_y - 0.16 * h),
        "right_eye": (cx + lean_dx + head_dx * 1.2 - 0.022 * h, sho_y - 0.16 * h),
        "left_shoulder": (cx + lean_dx + 0.105 * h, sho_y),
        "right_shoulder": (cx + lean_dx - 0.105 * h, sho_y),
        "left_elbow": (cx + lean_dx + 0.130 * h, sho_y + 0.14 * h),
        "right_elbow": (cx + lean_dx - 0.130 * h, sho_y + 0.14 * h),
        "left_wrist": (cx + lean_dx + 0.095 * h, sho_y + 0.26 * h),
        "right_wrist": (cx + lean_dx - 0.095 * h, sho_y + 0.26 * h),
        "left_hip": (cx + 0.072 * h, hip_y),
        "right_hip": (cx - 0.072 * h, hip_y),
        "left_knee": (cx + 0.075 * h, hip_y + 0.05 * h),
        "right_knee": (cx - 0.075 * h, hip_y + 0.05 * h),
    }
    kps = sk.from_xy(pts, conf)
    kps[sk.L_EAR] = (cx + lean_dx + head_dx * 1.1 + 0.038 * h,
                     sho_y - 0.145 * h, ear_vis_l)
    kps[sk.R_EAR] = (cx + lean_dx + head_dx * 1.1 - 0.038 * h,
                     sho_y - 0.145 * h, ear_vis_r)
    return kps


def rotate(kps: np.ndarray, about: tuple[float, float], degrees: float) -> np.ndarray:
    """Rotate every visible joint about a point, leaving confidences alone."""
    out = kps.copy()
    a = math.radians(degrees)
    ca, sa = math.cos(a), math.sin(a)
    ox, oy = about
    for i in range(sk.N_KP):
        if out[i][2] <= 0:
            continue
        dx, dy = out[i][0] - ox, out[i][1] - oy
        out[i][0] = ox + dx * ca - dy * sa
        out[i][1] = oy + dx * sa + dy * ca
    return out


def translate(kps: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = kps.copy()
    mask = out[:, 2] > 0
    out[mask, 0] += dx
    out[mask, 1] += dy
    return out


def lying(
    cx: float,
    ground_y: float,
    height: float,
    angle_deg: float = 86.0,
    facing: int = 1,
    conf: float = 0.86,
) -> np.ndarray:
    """A figure on the floor: a standing pose tipped over about its hips.

    Generated by rotation rather than by hand so that a fall is geometrically
    the same person it was a moment ago — which is exactly what the fall rule
    claims to notice.
    """
    h = float(height)
    base = standing(cx, ground_y, h, phase=0.35, facing=facing, conf=conf)
    hip = sk.hip_center(base) or (cx, ground_y - 0.52 * h)
    tipped = rotate(base, hip, angle_deg * facing)
    # Drop the whole body so the lowest joint rests on the floor line.
    lowest = float(tipped[tipped[:, 2] > 0][:, 1].max())
    return translate(tipped, 0.0, ground_y - lowest)


def jitter(kps: np.ndarray, px: float = 1.5, seed: int | None = None) -> np.ndarray:
    """Add a little noise, so tests do not depend on arithmetically perfect poses."""
    rng = np.random.default_rng(seed)
    out = kps.copy()
    mask = out[:, 2] > 0
    out[mask, 0] += rng.normal(0.0, px, mask.sum())
    out[mask, 1] += rng.normal(0.0, px, mask.sum())
    return out


def box_for(kps: np.ndarray, pad: float = 0.1) -> tuple[float, float, float, float]:
    """A bounding box around a synthetic pose, as a detector would receive it."""
    b = sk.bbox_from_keypoints(kps, pad=pad)
    if b is None:                                # pragma: no cover - defensive
        return (0.0, 0.0, 1.0, 1.0)
    return b
