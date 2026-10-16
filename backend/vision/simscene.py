"""
Scripted camera scenes, for when there is no camera — or no volunteer.

A ``sim:`` source hands the pipeline a finished world: people with skeletons
and track ids, bags with boxes, moving in real time. That world then goes
through precisely the same detectors, the same scale-free thresholds, the same
cooldowns, the same clip recorder and the same alert path as a real camera.
Nothing downstream knows the difference, and nothing downstream is softened —
if the fall rule fires here, it is the fall rule firing.

Why this is in the repository rather than hacked up for the demo:

* A judge's laptop may have no webcam, and a venue's wifi may not carry a
  phone stream. The dashboard should still be alive.
* Fall detection is the headline feature and the hardest to show. Asking a
  teammate to throw themselves onto an auditorium floor, repeatedly, on cue,
  is a bad plan.
* An unattended-bag rule with a thirty second timer needs thirty quiet
  seconds. A scene can stage that reliably while someone is still talking.

Every simulated feed is labelled "Simulated" on its tile, in its status
payload, in the event record and on the incident clip itself. There is no code
path that lets a simulated world reach the dashboard pretending to be a real
one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

import config
from backend.vision import skeleton as sk
from backend.vision import synth

WIDTH = config.PROC_WIDTH
HEIGHT = int(round(WIDTH * 9 / 16))


# --------------------------------------------------------------------------- #
# Motion helpers
# --------------------------------------------------------------------------- #

def _ease(t: float) -> float:
    """Smoothstep, so actors start and stop like people rather than lifts."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _seg(t: float, t0: float, t1: float) -> float:
    """Progress through a time window, clamped to 0..1."""
    if t1 <= t0:
        return 1.0
    return min(max((t - t0) / (t1 - t0), 0.0), 1.0)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


@dataclass
class Actor:
    """One simulated person or object over the whole scene timeline."""

    tid: int
    kind: str                       # "person" | label of a COCO object
    appear: float
    vanish: float
    fn: Callable[[float], dict[str, Any] | None]
    note: str = ""

    def at(self, t: float) -> dict[str, Any] | None:
        if not (self.appear <= t < self.vanish):
            return None
        return self.fn(t)


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #

class Scene:
    """Base class: owns the clock, the backdrop and the handover to the pipeline."""

    name = "scene"
    period = 60.0
    caption = ""

    def __init__(self) -> None:
        self.width, self.height = WIDTH, HEIGHT
        self.started = time.monotonic()
        self.loop = 0
        self._backdrop = self._paint_backdrop()
        self.actors: list[Actor] = []
        self.build()

    # -- to be provided by subclasses --------------------------------------- #
    def build(self) -> None:
        raise NotImplementedError

    def _paint_backdrop(self) -> np.ndarray:
        raise NotImplementedError

    # -- clock ---------------------------------------------------------------- #
    def _now(self) -> float:
        elapsed = time.monotonic() - self.started
        loop = int(elapsed // self.period)
        if loop != self.loop:
            # A new cycle is a new cast: fresh track ids mean the detectors see
            # new people, cooldowns do not suppress anything, and the demo can
            # be run again two minutes later with the same result.
            self.loop = loop
        return elapsed % self.period

    def track_id(self, base: int) -> int:
        return self.loop * 1000 + base

    # -- the one method the source calls ------------------------------------- #
    def step(self) -> tuple[np.ndarray, dict[str, Any]]:
        t = self._now()
        persons: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        notes: list[str] = []

        for actor in self.actors:
            state = actor.at(t)
            if state is None:
                continue
            state = dict(state)
            state["track_id"] = self.track_id(actor.tid)
            state["label"] = "person" if actor.kind == "person" else actor.kind
            state.setdefault("conf", 0.88)
            if "box" not in state:
                kps = state.get("keypoints")
                state["box"] = synth.box_for(kps) if kps is not None else (0, 0, 1, 1)
            (persons if actor.kind == "person" else objects).append(state)
            if actor.note:
                notes.append(actor.note)

        frame = self._render(persons, objects, t)
        world = {
            "persons": persons,
            "objects": objects,
            "simulated": True,
            "note": " · ".join(dict.fromkeys(notes)),
            "scene": self.name,
            "scene_t": round(t, 1),
            "scene_period": self.period,
        }
        return frame, world

    # -- drawing -------------------------------------------------------------- #

    def _render(self, persons, objects, t: float) -> np.ndarray:
        """Paint a plausible 'raw' frame.

        This is what the rolling buffer stores and what an incident clip
        contains, so it has to look like a camera rather than like a debug
        overlay — but it is stamped SIMULATED so that no clip from here could
        ever be mistaken for footage of a real person.
        """
        img = self._backdrop.copy()
        for o in objects:
            _draw_bag(img, o["box"])
        # Painter's algorithm: people lower in the frame are nearer the camera.
        for p in sorted(persons, key=lambda d: d["box"][3]):
            kps = p.get("keypoints")
            if kps is not None:
                _draw_figure(img, kps, p.get("tone", 0))
        _stamp(img, t)
        return img


# --------------------------------------------------------------------------- #
# Backdrops
# --------------------------------------------------------------------------- #

def _corridor_backdrop() -> np.ndarray:
    """A plain interior: wall, floor, pillars, a lit doorway at the far end."""
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    horizon = int(HEIGHT * 0.42)

    wall = np.linspace(46, 28, horizon, dtype=np.float32)
    img[:horizon] = wall[:, None, None] * np.array([1.02, 1.0, 0.96])

    floor = np.linspace(30, 64, HEIGHT - horizon, dtype=np.float32)
    img[horizon:] = floor[:, None, None] * np.array([1.0, 0.99, 0.97])

    # Perspective floor seams towards a vanishing point.
    vp = (int(WIDTH * 0.52), horizon)
    for i in range(-6, 7):
        x = int(WIDTH * 0.5 + i * WIDTH * 0.19)
        cv2.line(img, vp, (x, HEIGHT), (58, 60, 62), 1, cv2.LINE_AA)
    for k in range(1, 7):
        y = horizon + int((HEIGHT - horizon) * (k / 6.0) ** 1.8)
        cv2.line(img, (0, y), (WIDTH, y), (54, 56, 58), 1, cv2.LINE_AA)

    # A doorway, so there is somewhere for people to come from.
    dw, dh = int(WIDTH * 0.09), int(HEIGHT * 0.26)
    dx = int(WIDTH * 0.47)
    cv2.rectangle(img, (dx, horizon - dh), (dx + dw, horizon),
                  (96, 100, 104), -1)
    cv2.rectangle(img, (dx, horizon - dh), (dx + dw, horizon), (120, 126, 130), 1)

    for px in (int(WIDTH * 0.13), int(WIDTH * 0.86)):
        cv2.rectangle(img, (px - 9, int(horizon * 0.25)), (px + 9, HEIGHT),
                      (52, 53, 55), -1)
        cv2.line(img, (px - 9, int(horizon * 0.25)), (px - 9, HEIGHT),
                 (70, 72, 74), 1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.default_rng(7).normal(0, 2.4, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _hall_backdrop() -> np.ndarray:
    """An exam hall: rows of desks on a pale floor."""
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    horizon = int(HEIGHT * 0.3)
    img[:horizon] = np.array([52, 54, 56])
    floor = np.linspace(38, 78, HEIGHT - horizon, dtype=np.float32)
    img[horizon:] = floor[:, None, None] * np.array([1.0, 0.99, 0.95])
    cv2.rectangle(img, (0, 0), (WIDTH, horizon), (50, 52, 54), -1)
    cv2.line(img, (0, horizon), (WIDTH, horizon), (72, 74, 76), 1)
    # A board at the front
    cv2.rectangle(img, (int(WIDTH * 0.3), int(horizon * 0.25)),
                  (int(WIDTH * 0.7), int(horizon * 0.8)), (64, 68, 66), -1)

    for row, y in enumerate((0.62, 0.86)):
        yy = int(HEIGHT * y)
        th = int(HEIGHT * (0.05 + row * 0.02))
        for col in range(3):
            x0 = int(WIDTH * (0.07 + col * 0.31))
            x1 = x0 + int(WIDTH * (0.22 + row * 0.03))
            cv2.rectangle(img, (x0, yy), (x1, yy + th), (104, 112, 120), -1)
            cv2.rectangle(img, (x0, yy), (x1, yy + th), (130, 138, 146), 1)
    noise = np.random.default_rng(11).normal(0, 2.0, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Figure drawing
# --------------------------------------------------------------------------- #

_TONES = ((132, 128, 150), (118, 134, 122), (140, 124, 112), (112, 120, 140),
          (128, 118, 128), (120, 128, 134))


def _draw_figure(img: np.ndarray, kps: np.ndarray, tone: int = 0) -> None:
    """Paint a soft, clothed-looking body, composited through its own alpha.

    Drawing into a scratch layer and blending by that layer's coverage (rather
    than by a hard mask) is what stops every figure from wearing a black halo
    where the blur fades out.
    """
    colour = _TONES[tone % len(_TONES)]
    scale = max(5.0, (sk.shoulder_width(kps) or 18.0))
    thick = max(2, int(scale * 0.30))

    vis = kps[kps[:, 2] > 0]
    if not len(vis):
        return
    H, W = img.shape[:2]

    # The shadow and the body get their own tight windows. Sharing one padded
    # window meant blurring an area several times the figure, with a kernel
    # sized for the shadow, once per person per frame.
    _contact_shadow(img, float(vis[:, 0].mean()), float(vis[:, 1].max()), scale)

    pad = int(thick * 0.9) + 4
    x0 = max(0, int(vis[:, 0].min()) - pad)
    y0 = max(0, int(vis[:, 1].min()) - pad - int(scale * 0.5))
    x1 = min(W, int(vis[:, 0].max()) + pad)
    y1 = min(H, int(vis[:, 1].max()) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    roi = img[y0:y1, x0:x1]

    alpha = np.zeros(roi.shape[:2], np.uint8)
    for a, b, w in sk.EDGES:
        if not sk.seen(kps, a, b):
            continue
        pa = (int(kps[a][0]) - x0, int(kps[a][1]) - y0)
        pb = (int(kps[b][0]) - x0, int(kps[b][1]) - y0)
        cv2.line(alpha, pa, pb, 255, max(2, int(thick * w)), cv2.LINE_AA)
    head = sk.head_point(kps)
    if head is not None:
        cv2.circle(alpha, (int(head[0]) - x0, int(head[1]) - y0),
                   max(3, int(scale * 0.38)), 255, -1, cv2.LINE_AA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.8, thick * 0.22))

    # Composite through the alpha: roi = roi*(1-a) + colour*a, in uint8.
    inv = cv2.cvtColor(255 - alpha, cv2.COLOR_GRAY2BGR)
    tint = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.multiply(tint, np.array(colour, np.uint8), tint, scale=1.0 / 255.0)
    cv2.multiply(roi, inv, roi, scale=1.0 / 255.0)
    cv2.add(roi, tint, roi)


def _contact_shadow(img: np.ndarray, fx: float, fy: float, scale: float) -> None:
    """A soft pool of shade under the feet, drawn in its own small window."""
    H, W = img.shape[:2]
    rx, ry = max(4, int(scale * 1.1)), max(2, int(scale * 0.32))
    sigma = max(1.2, scale * 0.22)
    pad = int(sigma * 3) + 2
    x0, y0 = max(0, int(fx) - rx - pad), max(0, int(fy) - ry - pad)
    x1, y1 = min(W, int(fx) + rx + pad), min(H, int(fy) + ry + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    roi = img[y0:y1, x0:x1]
    shadow = np.zeros(roi.shape[:2], np.uint8)
    cv2.ellipse(shadow, (int(fx) - x0, int(fy) - y0), (rx, ry), 0, 0, 360,
                140, -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigma)
    inv = cv2.cvtColor(255 - (shadow // 2), cv2.COLOR_GRAY2BGR)
    cv2.multiply(roi, inv, roi, scale=1.0 / 255.0)


def _draw_bag(img: np.ndarray, box) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (58, 72, 96), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (84, 104, 132), 1, cv2.LINE_AA)
    cv2.ellipse(img, ((x1 + x2) // 2, y1), ((x2 - x1) // 3, (y2 - y1) // 5),
                0, 180, 360, (84, 104, 132), 2, cv2.LINE_AA)


def _stamp(img: np.ndarray, t: float) -> None:
    """Burn an unmistakable marker into the pixels themselves."""
    h, w = img.shape[:2]
    txt = "SIMULATED FEED"
    cv2.putText(img, txt, (12, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.42,
                (26, 26, 26), 2, cv2.LINE_AA)
    cv2.putText(img, txt, (12, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.42,
                (168, 176, 186), 1, cv2.LINE_AA)
    clock = time.strftime("%H:%M:%S")
    cv2.putText(img, f"{clock}  t+{t:05.1f}s", (w - 168, h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (26, 26, 26), 2, cv2.LINE_AA)
    cv2.putText(img, f"{clock}  t+{t:05.1f}s", (w - 168, h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 158, 168), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# The atrium: a timeline that exercises most of the system
# --------------------------------------------------------------------------- #

class AtriumScene(Scene):
    """A 78 second loop that stages, in order:

    ====  ===================================================================
     t    what happens
    ====  ===================================================================
    0+    two people walk through, continuously, as background traffic
    10    someone carries a bag in, puts it down at 16s and leaves at 20s
    22    a person settles into the loiter zone and stays
    42    a person walks in, drops to the floor at 46s and does not get up
    58    three people run across together
    66    five people converge into the crowd zone
    ====  ===================================================================

    The timings are chosen so that the unattended-bag and loitering rules fire
    inside the loop in demo mode, and so a presenter talking through the
    privacy panel will be interrupted by the fall at a useful moment.
    """

    name = "atrium"
    period = 78.0
    caption = "Atrium — scripted scene"

    def _paint_backdrop(self) -> np.ndarray:
        return _corridor_backdrop()

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def _ground(x_frac: float) -> tuple[float, float]:
        """Ground line and figure height at a horizontal position.

        People further from the camera stand higher in the frame and are
        smaller — without this the scale-free thresholds would never be tested.
        """
        y = HEIGHT * _lerp(0.62, 0.97, abs(x_frac - 0.5) * 2.0)
        h = HEIGHT * _lerp(0.33, 0.62, abs(x_frac - 0.5) * 2.0)
        return y, h

    def _walker(self, tid: int, t0: float, t1: float, x0: float, x1: float,
                tone: int, speed: float = 1.0, note: str = "") -> Actor:
        def fn(t: float) -> dict[str, Any]:
            u = _seg(t, t0, t1)
            xf = _lerp(x0, x1, u)
            gy, h = self._ground(xf)
            phase = (t - t0) * 6.0 * speed
            facing = 1 if x1 >= x0 else -1
            kps = synth.standing(xf * WIDTH, gy, h, phase=phase, facing=facing)
            return {"keypoints": kps, "tone": tone}
        return Actor(tid, "person", t0, t1, fn, note)

    def build(self) -> None:
        A = self.actors

        # --- background traffic, always present --------------------------- #
        def pacer(tid: int, offset: float, tone: int, span: float) -> Actor:
            def fn(t: float) -> dict[str, Any]:
                # A triangle wave: walk right, walk back, forever.
                u = ((t + offset) / span) % 2.0
                xf = _lerp(0.08, 0.92, u if u <= 1.0 else 2.0 - u)
                gy, h = self._ground(xf)
                facing = 1 if u <= 1.0 else -1
                kps = synth.standing(xf * WIDTH, gy, h,
                                     phase=(t + offset) * 5.4, facing=facing)
                return {"keypoints": kps, "tone": tone}
            return Actor(tid, "person", 0.0, self.period, fn)

        A.append(pacer(1, 0.0, 0, 26.0))
        A.append(pacer(2, 13.0, 1, 34.0))

        # --- the bag ------------------------------------------------------- #
        bag_x = 0.26

        def owner(t: float) -> dict[str, Any]:
            if t < 16.0:                      # carry it in
                xf = _lerp(0.02, bag_x, _ease(_seg(t, 10.0, 16.0)))
            elif t < 20.0:                    # crouch, set it down
                xf = bag_x
            else:                             # leave without it
                xf = _lerp(bag_x, 0.98, _ease(_seg(t, 20.0, 27.0)))
            gy, h = self._ground(xf)
            stooped = 16.0 <= t < 19.0
            kps = synth.standing(xf * WIDTH, gy, h * (0.82 if stooped else 1.0),
                                 phase=t * 5.0, facing=1 if t >= 20 else -1)
            return {"keypoints": kps, "tone": 2}

        A.append(Actor(3, "person", 10.0, 27.0, owner, "bag being left"))

        def bag(t: float) -> dict[str, Any]:
            gy, h = self._ground(bag_x)
            s = h * 0.20
            x = bag_x * WIDTH
            return {"box": (x - s * 0.6, gy - s, x + s * 0.6, gy), "conf": 0.74}

        A.append(Actor(50, "backpack", 16.0, self.period, bag,
                       "unattended bag staged"))

        # --- something left in the fire exit --------------------------------- #
        def crate(t: float) -> dict[str, Any]:
            xf = 0.68
            gy, h = self._ground(xf)
            s = h * 0.26
            x = xf * WIDTH
            return {"box": (x - s * 0.7, gy - s, x + s * 0.7, gy), "conf": 0.7}

        A.append(Actor(51, "suitcase", 30.0, 50.0, crate,
                       "blocked fire exit staged"))

        # --- the loiterer --------------------------------------------------- #
        def loiterer(t: float) -> dict[str, Any]:
            xf = _lerp(0.03, 0.22, _ease(_seg(t, 22.0, 28.0)))
            gy, h = self._ground(xf)
            # Shifting weight on the spot, as people actually do when waiting.
            sway = math.sin(t * 0.9) * 0.004 * WIDTH
            kps = synth.standing(xf * WIDTH + sway, gy, h,
                                 phase=math.sin(t * 0.7) * 0.4, facing=1,
                                 arms="folded")
            return {"keypoints": kps, "tone": 3}

        A.append(Actor(4, "person", 22.0, 56.0, loiterer, "loitering staged"))

        # --- the fall -------------------------------------------------------- #
        fall_x = 0.56
        t_fall = 46.0

        def faller(t: float) -> dict[str, Any]:
            gy, h = self._ground(fall_x)
            if t < t_fall:
                xf = _lerp(0.88, fall_x, _ease(_seg(t, 42.0, t_fall)))
                gy, h = self._ground(xf)
                kps = synth.standing(xf * WIDTH, gy, h, phase=t * 6.0)
                return {"keypoints": kps, "tone": 4}
            # 0.45s of collapse, then still on the floor.
            u = _ease(_seg(t, t_fall, t_fall + 0.45))
            kps = synth.lying(fall_x * WIDTH, gy, h,
                              angle_deg=_lerp(4.0, 86.0, u))
            return {"keypoints": kps, "tone": 4}

        A.append(Actor(5, "person", 42.0, 57.0, faller, "fall staged"))

        # --- the panic run ---------------------------------------------------- #
        for i in range(3):
            t0 = 58.0 + i * 0.4
            A.append(self._walker(6 + i, t0, t0 + 3.4,
                                  0.05 + i * 0.04, 0.95, tone=i,
                                  speed=2.6, note="panic run staged"))

        # --- the crowd --------------------------------------------------------- #
        for i in range(5):
            target = 0.40 + (i - 2) * 0.045

            def cluster(t: float, target=target, i=i) -> dict[str, Any]:
                xf = _lerp(0.02 + i * 0.22, target, _ease(_seg(t, 66.0, 71.0)))
                gy, h = self._ground(xf)
                kps = synth.standing(xf * WIDTH, gy, h, phase=t * 4.0,
                                     facing=1 if i % 2 else -1)
                return {"keypoints": kps, "tone": i}

            A.append(Actor(10 + i, "person", 66.0, self.period, cluster,
                           "crowd staged"))


class ExamScene(Scene):
    """An invigilation scene: six seated candidates, one phone, one head turn."""

    name = "exam"
    period = 54.0
    caption = "Exam hall — scripted scene"

    def _paint_backdrop(self) -> np.ndarray:
        return _hall_backdrop()

    #: Desk centres as fractions of the frame, back row first.
    DESKS = ((0.18, 0.62), (0.49, 0.62), (0.80, 0.62),
             (0.18, 0.86), (0.49, 0.86), (0.80, 0.86))

    def build(self) -> None:
        for i, (dx, dy) in enumerate(self.DESKS):
            seat_y = HEIGHT * dy
            h = HEIGHT * (0.40 + 0.06 * (i // 3))

            def sitter(t: float, dx=dx, seat_y=seat_y, h=h, i=i) -> dict[str, Any]:
                # Desk 5 turns its head away for eight seconds, twice a loop.
                yaw = 0.0
                if i == 4:
                    for t0 in (24.0, 44.0):
                        if t0 <= t < t0 + 8.0:
                            yaw = 44.0 * _ease(_seg(t, t0, t0 + 1.2))
                breathe = math.sin(t * 1.3 + i) * 1.2
                kps = synth.sitting(dx * WIDTH, seat_y + breathe, h,
                                    yaw=yaw, lean=math.sin(t * 0.6 + i) * 3.0)
                return {"keypoints": kps, "tone": i,
                        "desk_hint": i + 1, "yaw_hint": yaw}

            self.actors.append(Actor(1 + i, "person", 0.0, self.period, sitter))

        # A phone appears at desk 2 between t=14 and t=26.
        def phone(t: float) -> dict[str, Any]:
            dx, dy = self.DESKS[1]
            x, y = dx * WIDTH + 18, HEIGHT * dy - 6
            s = HEIGHT * 0.035
            return {"box": (x - s * 0.4, y - s, x + s * 0.4, y), "conf": 0.66}

        self.actors.append(Actor(60, "cell phone", 14.0, 26.0, phone,
                                 "phone at desk staged"))


class GateScene(Scene):
    """A doorway, for the entry/exit counter: people in, people out."""

    name = "gate"
    period = 40.0
    caption = "Gate — scripted scene"

    def _paint_backdrop(self) -> np.ndarray:
        return _corridor_backdrop()

    def build(self) -> None:
        # Six crossings, alternating direction, each taking four seconds.
        for i in range(6):
            t0 = 2.0 + i * 6.0
            inbound = i % 2 == 0

            def cross(t: float, t0=t0, inbound=inbound, i=i) -> dict[str, Any]:
                u = _ease(_seg(t, t0, t0 + 4.0))
                yf = _lerp(0.52, 0.95, u) if inbound else _lerp(0.95, 0.52, u)
                xf = 0.5 + math.sin(u * 3.0 + i) * 0.05
                h = HEIGHT * _lerp(0.30, 0.60, (yf - 0.52) / 0.43)
                kps = synth.standing(xf * WIDTH, yf * HEIGHT, h, phase=t * 6.0)
                return {"keypoints": kps, "tone": i}

            self.actors.append(Actor(1 + i, "person", t0, t0 + 4.2, cross))


_SCENES: dict[str, type[Scene]] = {
    "atrium": AtriumScene,
    "exam": ExamScene,
    "gate": GateScene,
}


def build(name: str) -> Scene:
    key = (name or "atrium").strip().lower()
    cls = _SCENES.get(key)
    if cls is None:
        raise ValueError(
            f"unknown scene '{name}'. Available: {', '.join(sorted(_SCENES))}")
    return cls()


def available() -> list[str]:
    return sorted(_SCENES)


#: Zones that make each scene meaningful, seeded the first time a simulated
#: camera is seen so that the demo works with nothing drawn by hand.
DEFAULT_ZONES: dict[str, list[dict[str, Any]]] = {
    # Positioned against AtriumScene._ground(), so that the actors genuinely
    # walk into them. Draw your own over a real camera in the zone editor.
    "atrium": [
        {"name": "Service corridor", "type": "restricted",
         "points": [[0.84, 0.78], [1.00, 0.78], [1.00, 1.00], [0.84, 1.00]]},
        {"name": "West alcove", "type": "loiter",
         "points": [[0.02, 0.62], [0.21, 0.62], [0.21, 1.00], [0.02, 1.00]]},
        {"name": "Atrium floor", "type": "crowd",
         "points": [[0.28, 0.58], [0.74, 0.58], [0.78, 1.00], [0.24, 1.00]]},
        {"name": "Fire exit", "type": "exit",
         "points": [[0.58, 0.70], [0.80, 0.70], [0.80, 0.90], [0.58, 0.90]]},
        {"name": "Atrium doorway", "type": "line",
         "points": [[0.50, 0.55], [0.50, 1.00]]},
    ],
    "exam": [
        {"name": f"Desk {i + 1}", "type": "desk",
         "points": [[dx - 0.13, dy - 0.10], [dx + 0.13, dy - 0.10],
                    [dx + 0.13, dy + 0.09], [dx - 0.13, dy + 0.09]]}
        for i, (dx, dy) in enumerate(ExamScene.DESKS)
    ],
    "gate": [
        {"name": "Main gate", "type": "line",
         "points": [[0.18, 0.74], [0.82, 0.74]]},
        {"name": "Gate crowd", "type": "crowd",
         "points": [[0.10, 0.60], [0.90, 0.60], [0.95, 0.99], [0.05, 0.99]]},
    ],
}
