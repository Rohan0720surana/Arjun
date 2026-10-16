"""
The skeleton: one canonical keypoint layout, and the geometry built on it.

ARJUN standardises on the 17-point COCO layout, because that is what the
pretrained pose weights emit and because every detection rule in the project
can be written against it. If the MediaPipe pose backend is selected instead,
its 33 landmarks are remapped onto these same 17 slots on the way in
(:func:`from_mediapipe`), so no detector ever needs to know which backend
produced the numbers it is reading.

Three things live here and nowhere else:

* the index <-> name mapping, so no detector contains a bare ``kps[11]``;
* the bone list used by the renderer — note that it contains **no facial
  edges**. The eyes, ears and nose are used for geometry (head position, yaw)
  and are never drawn. The head is a plain circle;
* the small geometric predicates the rules are phrased in: torso angle, hip
  height, foot point, body height. These are pure functions of an array, which
  is what makes the detectors testable from synthetic keypoints.

Coordinates are pixels in the processed frame. Each keypoint is ``(x, y, c)``
where ``c`` is the model's confidence for that joint; anything below
``config.KEYPOINT_CONF`` is treated as not seen rather than as being at (0, 0),
which is the single most common source of nonsense in pose-based rules.
"""

from __future__ import annotations

import math

import numpy as np

import config

NAMES: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

IDX: dict[str, int] = {n: i for i, n in enumerate(NAMES)}
N_KP = len(NAMES)

NOSE = IDX["nose"]
L_EAR, R_EAR = IDX["left_ear"], IDX["right_ear"]
L_SHO, R_SHO = IDX["left_shoulder"], IDX["right_shoulder"]
L_ELB, R_ELB = IDX["left_elbow"], IDX["right_elbow"]
L_WRI, R_WRI = IDX["left_wrist"], IDX["right_wrist"]
L_HIP, R_HIP = IDX["left_hip"], IDX["right_hip"]
L_KNE, R_KNE = IDX["left_knee"], IDX["right_knee"]
L_ANK, R_ANK = IDX["left_ankle"], IDX["right_ankle"]

#: Bones, as (a, b, weight). Weight drives line thickness so the torso reads
#: as the trunk of the figure rather than every limb being the same wire.
EDGES: tuple[tuple[int, int, float], ...] = (
    (L_SHO, R_SHO, 1.25),
    (L_SHO, L_HIP, 1.15), (R_SHO, R_HIP, 1.15), (L_HIP, R_HIP, 1.25),
    (L_SHO, L_ELB, 1.0), (L_ELB, L_WRI, 0.85),
    (R_SHO, R_ELB, 1.0), (R_ELB, R_WRI, 0.85),
    (L_HIP, L_KNE, 1.1), (L_KNE, L_ANK, 0.95),
    (R_HIP, R_KNE, 1.1), (R_KNE, R_ANK, 0.95),
)

#: Joints the renderer marks with a dot — deliberately excludes 0..4 (the face).
DRAW_JOINTS: tuple[int, ...] = (L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI,
                                L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK)

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def empty() -> np.ndarray:
    """An all-unseen keypoint array."""
    return np.zeros((N_KP, 3), dtype=np.float32)


def from_xy(points: dict[str, Point], conf: float = 1.0) -> np.ndarray:
    """Build keypoints from a ``{"left_hip": (x, y), ...}`` mapping.

    This is how the test suite and the simulated scenes express a pose: by
    name, so the tests read like the rule they are checking.
    """
    kps = empty()
    for name, (x, y) in points.items():
        i = IDX[name]
        kps[i] = (float(x), float(y), float(conf))
    return kps


#: MediaPipe's 33-landmark model -> our 17 slots. MediaPipe has no ear-canal
#: equivalent problem here: indices 7/8 are the ears, 0 the nose, 2/5 the eyes.
_MP_TO_COCO: tuple[tuple[int, int], ...] = (
    (0, 0), (2, 1), (5, 2), (7, 3), (8, 4),
    (11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10),
    (23, 11), (24, 12), (25, 13), (26, 14), (27, 15), (28, 16),
)


def from_mediapipe(landmarks, width: float, height: float) -> np.ndarray:
    """Remap a MediaPipe pose result (normalised) into COCO-17 pixels."""
    kps = empty()
    for mp_i, coco_i in _MP_TO_COCO:
        if mp_i >= len(landmarks):
            continue
        lm = landmarks[mp_i]
        vis = float(getattr(lm, "visibility", 1.0) or 0.0)
        pres = float(getattr(lm, "presence", 1.0) or 0.0)
        kps[coco_i] = (float(lm.x) * width, float(lm.y) * height,
                       min(1.0, max(vis, pres)))
    return kps


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #

def seen(kps: np.ndarray | None, *idx: int, thr: float | None = None) -> bool:
    """True only if *every* listed joint is confidently visible."""
    if kps is None:
        return False
    t = config.KEYPOINT_CONF if thr is None else thr
    return all(float(kps[i][2]) >= t for i in idx)


def pt(kps: np.ndarray, i: int) -> Point:
    return float(kps[i][0]), float(kps[i][1])


def mid(kps: np.ndarray, a: int, b: int) -> Point | None:
    """Midpoint of two joints, or whichever single one is visible."""
    ok_a, ok_b = seen(kps, a), seen(kps, b)
    if ok_a and ok_b:
        return ((kps[a][0] + kps[b][0]) / 2.0, (kps[a][1] + kps[b][1]) / 2.0)
    if ok_a:
        return pt(kps, a)
    if ok_b:
        return pt(kps, b)
    return None


def hip_center(kps: np.ndarray | None) -> Point | None:
    return None if kps is None else mid(kps, L_HIP, R_HIP)


def shoulder_center(kps: np.ndarray | None) -> Point | None:
    return None if kps is None else mid(kps, L_SHO, R_SHO)


def ankle_center(kps: np.ndarray | None) -> Point | None:
    return None if kps is None else mid(kps, L_ANK, R_ANK)


def head_point(kps: np.ndarray | None) -> Point | None:
    """A single point standing in for the head — never a face landmark set."""
    if kps is None:
        return None
    ears = mid(kps, L_EAR, R_EAR)
    if ears is not None:
        return ears
    if seen(kps, NOSE):
        return pt(kps, NOSE)
    sc = shoulder_center(kps)
    if sc is None:
        return None
    # No head joints at all: park the circle a shoulder-width above the line.
    w = shoulder_width(kps) or 30.0
    return sc[0], sc[1] - w * 0.55


def shoulder_width(kps: np.ndarray | None) -> float | None:
    if not seen(kps, L_SHO, R_SHO):
        return None
    assert kps is not None
    return float(math.hypot(kps[L_SHO][0] - kps[R_SHO][0],
                            kps[L_SHO][1] - kps[R_SHO][1]))


def box_height(box: tuple[float, float, float, float]) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def box_width(box: tuple[float, float, float, float]) -> float:
    return max(1.0, float(box[2]) - float(box[0]))


def aspect_ratio(box: tuple[float, float, float, float]) -> float:
    """Width over height. Above ~1.15 the person is lying rather than standing."""
    return box_width(box) / box_height(box)


def body_height(box: tuple[float, float, float, float],
                kps: np.ndarray | None = None) -> float:
    """A scale for this person, in pixels.

    Every distance and speed threshold in the project is divided by this, which
    is what makes one set of numbers work for someone close to the lens and
    someone at the end of a corridor.

    The bounding box height is the honest answer while a person is upright, but
    it collapses the moment they lie down — exactly when fall detection needs a
    stable scale. So when we can see both a head and a hip we use the vertical
    span of the skeleton's longest reliable chain, and fall back to the box
    diagonal (which is rotation invariant) rather than its height.
    """
    hp = head_point(kps)
    ank = ankle_center(kps)
    if hp is not None and ank is not None:
        span = math.hypot(ank[0] - hp[0], ank[1] - hp[1])
        if span > 8.0:
            return float(span)
    hip = hip_center(kps)
    if hp is not None and hip is not None:
        span = math.hypot(hip[0] - hp[0], hip[1] - hp[1])
        if span > 5.0:
            # head-to-hip is roughly half a standing body
            return float(span * 2.0)
    w, h = box_width(box), box_height(box)
    if w > h:                      # lying down: the long side is the body
        return float(math.hypot(w, h) * 0.92)
    return float(h)


def torso_angle_deg(kps: np.ndarray | None) -> float | None:
    """Angle of the shoulders-to-hips line away from vertical, in degrees.

    0 is standing straight up, 90 is lying flat. Returns ``None`` when either
    end of the torso is not visible, and callers must treat that as "unknown"
    rather than as zero — a missing torso is not an upright one.
    """
    sc = shoulder_center(kps)
    hc = hip_center(kps)
    if sc is None or hc is None:
        return None
    dx, dy = hc[0] - sc[0], hc[1] - sc[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    # atan2(|dx|, |dy|): 0 when the torso is vertical, 90 when horizontal.
    return float(math.degrees(math.atan2(abs(dx), abs(dy))))


def foot_point(box: tuple[float, float, float, float],
               kps: np.ndarray | None = None) -> Point:
    """Where this person is standing — the point used for every zone test.

    Ankles when we can see them, otherwise the bottom centre of the box. Using
    the feet rather than the box centre matters: a tall person leaning over a
    restricted line should not trip it, and someone standing just outside a
    doorway should not be counted as inside the room.
    """
    if kps is not None:
        a = ankle_center(kps)
        if a is not None:
            return a
        # One ankle hidden behind the other leg is common; a knee is still a
        # better guide to standing position than the centre of the box.
        k = mid(kps, L_KNE, R_KNE)
        if k is not None:
            return k[0], k[1] + box_height(box) * 0.18
    return (float(box[0]) + float(box[2])) / 2.0, float(box[3])


def center(box: tuple[float, float, float, float]) -> Point:
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def diagonal(box: tuple[float, float, float, float]) -> float:
    return float(math.hypot(box_width(box), box_height(box)))


def visible_fraction(kps: np.ndarray | None) -> float:
    if kps is None:
        return 0.0
    return float((kps[:, 2] >= config.KEYPOINT_CONF).mean())


def head_yaw_deg(kps: np.ndarray | None) -> float | None:
    """A rough head yaw from body keypoints alone, in degrees.

    This is the fallback for exam mode when the MediaPipe face mesh is not
    available. It is a far blunter instrument than a 3D transform matrix and it
    is used as one: only the magnitude matters, and only against a generous
    threshold.

    Two cases. With both ears visible, the nose sits between them when a person
    faces the camera and slides towards one as they turn; the offset, scaled by
    the ear separation, gives a signed angle. When one ear disappears behind
    the head the turn is already large, and we say so with a fixed value rather
    than pretending to a precision we do not have.

    Returns ``None`` when there is not enough of a head to judge, which callers
    must treat as "unknown" rather than as "facing forward".
    """
    if kps is None:
        return None
    left, right = seen(kps, L_EAR), seen(kps, R_EAR)
    if not seen(kps, NOSE):
        return None

    nose_x = float(kps[NOSE][0])
    if left and right:
        lx, rx = float(kps[L_EAR][0]), float(kps[R_EAR][0])
        span = abs(lx - rx)
        if span < 2.0:
            # Ears on top of each other: the head is side-on to the camera.
            return 75.0 if nose_x > (lx + rx) / 2.0 else -75.0
        ratio = (nose_x - (lx + rx) / 2.0) / (span / 2.0)
        return float(max(-1.0, min(1.0, ratio)) * 70.0)

    if left or right:
        # One ear hidden behind the head: a large turn, direction known.
        visible_x = float(kps[L_EAR if left else R_EAR][0])
        return 55.0 if nose_x < visible_x else -55.0
    return None


def head_box(box: tuple[float, float, float, float],
             kps: np.ndarray | None,
             scale: float = 1.9) -> tuple[int, int, int, int] | None:
    """A square crop around the head, for the face mesh to look at.

    Derived from the head point and the shoulder width so it tracks the
    subject's size. Returns integer pixel bounds, or ``None`` when there is no
    reliable head position — we would rather skip a face than crop a torso.
    """
    head = head_point(kps)
    if head is None:
        return None
    width = shoulder_width(kps)
    if width is None or width < 6.0:
        width = box_width(box) * 0.55
    half = max(12.0, width * scale * 0.5)
    return (int(head[0] - half), int(head[1] - half * 1.15),
            int(head[0] + half), int(head[1] + half * 0.95))


def hand_box(kps: np.ndarray | None, side: str,
             scale: float = 2.2) -> tuple[int, int, int, int] | None:
    """A square crop around one wrist, for the hand model to look at."""
    if kps is None:
        return None
    idx = L_WRI if side == "left" else R_WRI
    if not seen(kps, idx):
        return None
    width = shoulder_width(kps) or 30.0
    half = max(14.0, width * scale * 0.5)
    x, y = float(kps[idx][0]), float(kps[idx][1])
    return (int(x - half), int(y - half), int(x + half), int(y + half))


def hand_is_raised(kps: np.ndarray | None, side: str) -> bool:
    """Is this wrist up near or above the shoulder line?

    A cheap gate in front of an expensive model: the Signal for Help is made
    with the hand held up, so there is no reason to run hand landmarking on
    every arm hanging by every side in the room.
    """
    if kps is None:
        return False
    wrist = L_WRI if side == "left" else R_WRI
    shoulder = L_SHO if side == "left" else R_SHO
    if not seen(kps, wrist):
        return False
    sc = shoulder_center(kps)
    if sc is None:
        if not seen(kps, shoulder):
            return False
        sy = float(kps[shoulder][1])
    else:
        sy = sc[1]
    width = shoulder_width(kps) or 30.0
    return float(kps[wrist][1]) < sy + width * 0.35


def bbox_from_keypoints(kps: np.ndarray,
                        pad: float = 0.12) -> tuple[float, float, float, float] | None:
    """A box around whatever joints are visible — used by the simulated scenes."""
    vis = kps[kps[:, 2] >= config.KEYPOINT_CONF]
    if len(vis) < 2:
        return None
    x1, y1 = float(vis[:, 0].min()), float(vis[:, 1].min())
    x2, y2 = float(vis[:, 0].max()), float(vis[:, 1].max())
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return (x1 - w * pad, y1 - h * pad * 1.4, x2 + w * pad, y2 + h * pad * 0.4)
