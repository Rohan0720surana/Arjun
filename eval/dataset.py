"""
Synthetic benchmark dataset generator for offline fall detection evaluation.

This module provides a reproducible set of synthetic sequence benchmarks
covering both true falls (forward, backward, distant, and box-only) and
challenging Normal Activities of Daily Living (ADL: walking, sitting, bending,
tripping with quick recovery, crouching, and preexisting lying down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence as Seq

import numpy as np

from backend.vision import synth
from backend.world import Detection

WIDTH, HEIGHT = 640, 360
GROUND_Y = 330.0
HEIGHT_PX = 200.0
CENTER_X = 320.0


@dataclass(frozen=True)
class FrameObservation:
    """One discrete frame observation in a sequence."""
    frame_idx: int
    timestamp: float
    detections: list[Detection]


@dataclass(frozen=True)
class SequenceAnnotation:
    """Ground truth annotation for an evaluation sequence."""
    sequence_id: str
    name: str
    is_fall: bool | None = None
    duration_s: float = 0.0
    description: str = ""
    fall_start_s: float | None = None
    fall_impact_s: float | None = None
    fall_recovery_s: float | None = None
    target_track_id: int = 1
    fall_transition_start_s: float | None = None
    fallen_state_start_s: float | None = None

    def confirmation_anchor_s(self) -> float | None:
        """
        Return the event timestamp anchor for post-fall observation sufficiency.

        In ARJUN's production FallDetector (backend/detectors/fall.py), the 2.0s
        confirmation timer begins only after the subject has completed the drop
        and entered a horizontal posture on the ground (Stage 1: drop + horizontal).

        Priority order:
        1. fallen_state_start_s: Start of the fallen state (URFD state 1: down on floor).
        2. fall_impact_s: Physically verified impact timestamp.

        Note: fall_transition_start_s / fall_start_s (URFD state 0) is deliberately
        NOT used as the confirmation anchor because the subject is still actively
        falling/collapsing, not yet resting in horizontal posture on the floor.
        """
        if self.fallen_state_start_s is not None:
            return self.fallen_state_start_s
        if self.fall_impact_s is not None:
            return self.fall_impact_s
        return None

    def matching_window(
        self,
        window_after_impact_s: float = 5.0,
        window_before_impact_s: float = 0.0,
    ) -> tuple[float, float] | None:
        """
        Algorithm-neutral evaluation matching window.

        By evaluation convention, an alert is eligible to match a fall if it fires
        between ground-truth impact/fallen onset (minus optional pre-impact tolerance)
        and window_after_impact_s (default: 5.0s) following impact.
        """
        if not self.is_fall:
            return None
        anchor = self.confirmation_anchor_s()
        if anchor is None:
            anchor = self.fall_start_s or self.fall_transition_start_s
        if anchor is None:
            return None
        start = max(0.0, anchor - window_before_impact_s)
        end = anchor + window_after_impact_s
        return (start, end)

    @property
    def expected_alert_window(self) -> tuple[float, float] | None:
        """Return the default algorithm-neutral matching window: [anchor, anchor + 5.0s]."""
        return self.matching_window(window_after_impact_s=5.0, window_before_impact_s=0.0)

    def has_sufficient_observation(
        self,
        required_confirmation_s: float = 2.0,
        tolerance_s: float = 0.0,
    ) -> bool:
        """
        Check if the sequence contains adequate post-fall footage to permit confirmation.

        Requires that the video duration extends at least (required_confirmation_s - tolerance_s)
        beyond the post-fall confirmation anchor (fallen_state_start_s or fall_impact_s).
        Returns False if ground-truth confirmation anchor is unknown.
        """
        if not self.is_fall:
            return True
        anchor = self.confirmation_anchor_s()
        if anchor is None:
            return False
        return (self.duration_s - anchor) >= (required_confirmation_s - tolerance_s)



@dataclass
class Sequence:
    """A full synthetic sequence ready for evaluation."""
    annotation: SequenceAnnotation
    fps: float
    frames: list[FrameObservation] = field(default_factory=list)


def _person(track_id: int, keypoints: np.ndarray | None,
            box: tuple[float, float, float, float] | None = None,
            conf: float = 0.92) -> Detection:
    if box is None:
        box = synth.box_for(keypoints) if keypoints is not None else (0.0, 0.0, 10.0, 10.0)
    return Detection(track_id=track_id, label="person", conf=conf,
                     box=box, keypoints=keypoints)


# --------------------------------------------------------------------------- #
# Sequence Builders
# --------------------------------------------------------------------------- #

def build_fall_forward_stay(fps: float = 15.0) -> Sequence:
    """A person stands/walks, falls forward over 0.4s, and stays down."""
    duration = 6.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))
    fall_start = 2.0
    fall_impact = 2.4

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < fall_start:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX, phase=t * 6.0)
        elif t < fall_impact:
            u = (t - fall_start) / (fall_impact - fall_start)
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=4.0 + 82.0 * u)
        else:
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=86.0)

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-FALL-01",
        name="Fall Forward (Sustained)",
        is_fall=True,
        description="Walks for 2.0s, collapses forward over 0.4s, stays down on the floor.",
        duration_s=duration,
        fall_start_s=fall_start,
        fall_impact_s=fall_impact,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_fall_backward_stay(fps: float = 15.0) -> Sequence:
    """A person stands, loses footing, falls backward over 0.4s, and stays down."""
    duration = 6.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))
    fall_start = 1.5
    fall_impact = 1.9

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < fall_start:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX, phase=0.2)
        elif t < fall_impact:
            u = (t - fall_start) / (fall_impact - fall_start)
            # Tipped backward (-86 deg)
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=-(4.0 + 82.0 * u), facing=-1)
        else:
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=-86.0, facing=-1)

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-FALL-02",
        name="Fall Backward (Sustained)",
        is_fall=True,
        description="Stands for 1.5s, falls backward over 0.4s, stays down.",
        duration_s=duration,
        fall_start_s=fall_start,
        fall_impact_s=fall_impact,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_fall_distant(fps: float = 15.0) -> Sequence:
    """A distant person (height ~65px) collapses; verifies scale invariance."""
    duration = 5.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))
    fall_start = 1.0
    fall_impact = 1.4
    dist_height = 65.0
    dist_feet = 160.0
    dist_cx = 240.0

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < fall_start:
            kps = synth.standing(dist_cx, dist_feet, dist_height, phase=t * 5.0)
        elif t < fall_impact:
            u = (t - fall_start) / (fall_impact - fall_start)
            kps = synth.lying(dist_cx, dist_feet, dist_height, angle_deg=4.0 + 82.0 * u)
        else:
            kps = synth.lying(dist_cx, dist_feet, dist_height, angle_deg=86.0)

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-FALL-03",
        name="Distant Fall (Scale Invariant)",
        is_fall=True,
        description="Distant small subject (65px) collapses and stays down.",
        duration_s=duration,
        fall_start_s=fall_start,
        fall_impact_s=fall_impact,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_fall_box_only(fps: float = 15.0) -> Sequence:
    """Pose estimation missing; verifies bounding-box aspect ratio fallback."""
    duration = 5.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))
    fall_start = 1.0
    fall_impact = 1.4

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < fall_start:
            box = (290.0, 130.0, 350.0, 330.0)      # tall upright box
        elif t < fall_impact:
            u = (t - fall_start) / (fall_impact - fall_start)
            # Interpolate from tall box to wide horizontal box
            box = (
                290.0 - 50.0 * u,
                130.0 + 145.0 * u,
                350.0 + 90.0 * u,
                330.0,
            )
        else:
            box = (240.0, 275.0, 440.0, 330.0)      # wide horizontal box (w/h = 200/55 = 3.63)

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, None, box=box)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-FALL-04",
        name="Fall Without Keypoints (Box Fallback)",
        is_fall=True,
        description="Keypoint-free collapse where bounding box aspect ratio flips from vertical to horizontal.",
        duration_s=duration,
        fall_start_s=fall_start,
        fall_impact_s=fall_impact,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_adl_walking(fps: float = 15.0) -> Sequence:
    """Normal continuous walking across the scene."""
    duration = 6.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        cx = 120.0 + 45.0 * t
        kps = synth.standing(cx, GROUND_Y, HEIGHT_PX, phase=t * 6.0)
        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-ADL-01",
        name="ADL: Normal Walking",
        is_fall=False,
        description="Subject walks steadily across the room for 6 seconds.",
        duration_s=duration,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_adl_sitting(fps: float = 15.0) -> Sequence:
    """Sitting down quickly into a chair; hips drop but torso stays vertical."""
    duration = 6.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))
    sit_start = 1.5
    sit_end = 1.9

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < sit_start:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX)
        else:
            u = min(1.0, (t - sit_start) / (sit_end - sit_start))
            # Hips drop to seat height, shoulders stay directly above
            seat_y = GROUND_Y - HEIGHT_PX * 0.52 * (1.0 - u) - 10.0
            kps = synth.sitting(CENTER_X, seat_y, HEIGHT_PX)

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-ADL-02",
        name="ADL: Fast Sitting",
        is_fall=False,
        description="Subject sits down in 0.4s; rapid hip drop but torso remains upright.",
        duration_s=duration,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_adl_bending(fps: float = 15.0) -> Sequence:
    """Bending over to pick up an item or tie a shoe, then standing back up."""
    duration = 6.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < 1.5 or t >= 4.5:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX)
        elif 1.5 <= t < 2.2:
            u = (t - 1.5) / 0.7
            # Torso bends forward, hips push back slightly but barely drop
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX, phase=0.0)
            hip = (CENTER_X, GROUND_Y - 0.52 * HEIGHT_PX)
            # Tilt torso forward up to 60 deg
            kps = synth.rotate(kps, hip, 50.0 * u)
        elif 2.2 <= t < 3.8:
            hip = (CENTER_X, GROUND_Y - 0.52 * HEIGHT_PX)
            kps = synth.rotate(synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX), hip, 50.0)
        else:
            u = (t - 3.8) / 0.7
            hip = (CENTER_X, GROUND_Y - 0.52 * HEIGHT_PX)
            kps = synth.rotate(synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX), hip, 50.0 * (1.0 - u))

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-ADL-03",
        name="ADL: Bending and Reaching",
        is_fall=False,
        description="Subject bends over for 2s and stands back up without a sudden vertical drop.",
        duration_s=duration,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_adl_trip_and_recover(fps: float = 15.0) -> Sequence:
    """A stumble/trip to the floor, but the person immediately gets up (< 2s)."""
    duration = 5.5
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))
    trip_start = 1.0
    trip_down = 1.4
    trip_rise = 2.2
    trip_up = 2.8

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < trip_start:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX, phase=t * 6.0)
        elif t < trip_down:
            u = (t - trip_start) / (trip_down - trip_start)
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=4.0 + 82.0 * u)
        elif t < trip_rise:
            # Stays down only 0.8s
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=86.0)
        elif t < trip_up:
            u = (t - trip_rise) / (trip_up - trip_rise)
            kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=86.0 * (1.0 - u))
        else:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX, phase=t * 6.0)

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-ADL-04",
        name="ADL: Stumble and Quick Recovery",
        is_fall=False,
        description="Subject trips to the floor but stands back up in under 1 second.",
        duration_s=duration,
        fall_start_s=trip_start,
        fall_impact_s=trip_down,
        fall_recovery_s=trip_up,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_adl_preexisting_lying(fps: float = 15.0) -> Sequence:
    """Person already lying on the floor at t=0 (e.g. stretching / resting)."""
    duration = 6.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        kps = synth.lying(CENTER_X, GROUND_Y, HEIGHT_PX, angle_deg=86.0)
        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-ADL-05",
        name="ADL: Preexisting Lying Down",
        is_fall=False,
        description="Subject already horizontal at stream start; no drop event occurred.",
        duration_s=duration,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


def build_adl_squat_crouch(fps: float = 15.0) -> Sequence:
    """Squatting down low; hips drop but torso remains upright (< 30 deg)."""
    duration = 5.0
    dt = 1.0 / fps
    n_frames = int(round(duration * fps))

    frames: list[FrameObservation] = []
    for i in range(n_frames):
        t = i * dt
        if t < 1.0 or t >= 4.0:
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX)
        elif 1.0 <= t < 1.5:
            u = (t - 1.0) / 0.5
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX * (1.0 - 0.45 * u))
        elif 1.5 <= t < 3.5:
            # Low crouch: body height is compressed, but torso is vertical
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX * 0.55)
        else:
            u = (t - 3.5) / 0.5
            kps = synth.standing(CENTER_X, GROUND_Y, HEIGHT_PX * (0.55 + 0.45 * u))

        frames.append(FrameObservation(frame_idx=i, timestamp=round(t, 4),
                                       detections=[_person(1, kps)]))

    annotation = SequenceAnnotation(
        sequence_id="SYNTH-ADL-06",
        name="ADL: Squat and Hold",
        is_fall=False,
        description="Subject squats down low for 2 seconds with an upright torso, then stands.",
        duration_s=duration,
    )
    return Sequence(annotation=annotation, fps=fps, frames=frames)


# --------------------------------------------------------------------------- #
# Benchmark Collection
# --------------------------------------------------------------------------- #

def get_synthetic_benchmark(fps: float = 15.0) -> list[Sequence]:
    """Return the canonical 10-sequence synthetic evaluation benchmark."""
    return [
        build_fall_forward_stay(fps=fps),
        build_fall_backward_stay(fps=fps),
        build_fall_distant(fps=fps),
        build_fall_box_only(fps=fps),
        build_adl_walking(fps=fps),
        build_adl_sitting(fps=fps),
        build_adl_bending(fps=fps),
        build_adl_trip_and_recover(fps=fps),
        build_adl_preexisting_lying(fps=fps),
        build_adl_squat_crouch(fps=fps),
    ]
