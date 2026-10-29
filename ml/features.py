"""
ML Feature Engineering Module for Human Pose & Fall Detection (Phase 4A).

Computes scale-normalized geometric posture features and strictly causal
temporal movement kinematics over configurable rolling windows from
tracked person detections (COCO-17 keypoints + ByteTrack boxes).

Design Principles:
1. Purely Causal: Computes features for timestamp t using strictly past and
   present observations (t_obs <= t). Never accesses future frames.
2. Label & Detector Agnostic: Does not inspect ground truth labels, detector
   alert states, or clip boundaries.
3. Scale-Free Normalization: All spatial displacements and speeds are expressed
   in units of body heights (scale) rather than raw pixels.
4. Transparent Missingness: Low-confidence keypoints (< config.KEYPOINT_CONF)
   are never silently fabricated; missing values are explicitly NaN with
   corresponding binary missingness indicator columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from backend.vision import skeleton as sk
from backend.world import Detection

Point = tuple[float, float]

# Default temporal window durations (in seconds)
DEFAULT_WINDOWS: tuple[float, ...] = (0.3, 0.6, 1.0)


def compute_joint_angle(
    p1: Point | None,
    vertex: Point | None,
    p3: Point | None,
) -> float | None:
    """
    Compute 2D interior angle at `vertex` between segments (p1 - vertex) and (p3 - vertex).

    Returns:
        Angle in degrees in [0.0, 180.0], or None if any point is missing or collocated.
        180 deg corresponds to a completely straight limb; 90 deg to a right angle bend.
    """
    if p1 is None or vertex is None or p3 is None:
        return None

    v1_x, v1_y = p1[0] - vertex[0], p1[1] - vertex[1]
    v2_x, v2_y = p3[0] - vertex[0], p3[1] - vertex[1]

    len1 = math.hypot(v1_x, v1_y)
    len2 = math.hypot(v2_x, v2_y)

    if len1 < 1e-4 or len2 < 1e-4:
        return None

    dot = v1_x * v2_x + v1_y * v2_y
    cos_theta = dot / (len1 * len2)
    # Numerical safety clamp
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return float(math.degrees(math.acos(cos_theta)))


@dataclass
class InstantFeatures:
    """Instantaneous posture, shape, and keypoint confidence features for one frame."""

    # Scale & Bounding Box
    body_scale: float
    box_aspect_ratio: float
    box_width_norm: float
    box_height_norm: float

    # Torso Orientation
    torso_angle_deg: float | None
    torso_angle_cos: float | None
    torso_angle_sin: float | None

    # Shoulder-to-Hip Geometry
    shoulder_to_hip_dx_norm: float | None
    shoulder_to_hip_dy_norm: float | None
    shoulder_to_hip_dist_norm: float | None
    shoulder_width_norm: float | None
    hip_width_norm: float | None

    # Joint Angles (degrees)
    left_knee_angle_deg: float | None
    right_knee_angle_deg: float | None
    left_hip_angle_deg: float | None
    right_hip_angle_deg: float | None

    # Keypoint Confidence & Missingness Indicators
    keypoint_visible_fraction: float
    keypoint_mean_conf: float
    has_torso: int
    has_hips: int
    has_knees: int
    has_ankles: int
    has_full_legs: int

    # Missing Angle Indicators (1 if missing, 0 if present)
    missing_torso_angle: int
    missing_left_knee: int
    missing_right_knee: int
    missing_left_hip: int
    missing_right_hip: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert instant features to flat dictionary, replacing None with np.nan."""
        return {
            "body_scale": float(self.body_scale),
            "box_aspect_ratio": float(self.box_aspect_ratio),
            "box_width_norm": float(self.box_width_norm),
            "box_height_norm": float(self.box_height_norm),
            "torso_angle_deg": float(self.torso_angle_deg) if self.torso_angle_deg is not None else np.nan,
            "torso_angle_cos": float(self.torso_angle_cos) if self.torso_angle_cos is not None else np.nan,
            "torso_angle_sin": float(self.torso_angle_sin) if self.torso_angle_sin is not None else np.nan,
            "shoulder_to_hip_dx_norm": float(self.shoulder_to_hip_dx_norm) if self.shoulder_to_hip_dx_norm is not None else np.nan,
            "shoulder_to_hip_dy_norm": float(self.shoulder_to_hip_dy_norm) if self.shoulder_to_hip_dy_norm is not None else np.nan,
            "shoulder_to_hip_dist_norm": float(self.shoulder_to_hip_dist_norm) if self.shoulder_to_hip_dist_norm is not None else np.nan,
            "shoulder_width_norm": float(self.shoulder_width_norm) if self.shoulder_width_norm is not None else np.nan,
            "hip_width_norm": float(self.hip_width_norm) if self.hip_width_norm is not None else np.nan,
            "left_knee_angle_deg": float(self.left_knee_angle_deg) if self.left_knee_angle_deg is not None else np.nan,
            "right_knee_angle_deg": float(self.right_knee_angle_deg) if self.right_knee_angle_deg is not None else np.nan,
            "left_hip_angle_deg": float(self.left_hip_angle_deg) if self.left_hip_angle_deg is not None else np.nan,
            "right_hip_angle_deg": float(self.right_hip_angle_deg) if self.right_hip_angle_deg is not None else np.nan,
            "keypoint_visible_fraction": float(self.keypoint_visible_fraction),
            "keypoint_mean_conf": float(self.keypoint_mean_conf),
            "has_torso": int(self.has_torso),
            "has_hips": int(self.has_hips),
            "has_knees": int(self.has_knees),
            "has_ankles": int(self.has_ankles),
            "has_full_legs": int(self.has_full_legs),
            "missing_torso_angle": int(self.missing_torso_angle),
            "missing_left_knee": int(self.missing_left_knee),
            "missing_right_knee": int(self.missing_right_knee),
            "missing_left_hip": int(self.missing_left_hip),
            "missing_right_hip": int(self.missing_right_hip),
        }


def extract_instant_features(det: Detection) -> InstantFeatures:
    """Extract instantaneous geometric and keypoint features from one Detection."""
    kps = det.keypoints
    scale = max(1.0, det.scale())

    # Bounding box ratios
    aspect = det.aspect()
    bw_norm = sk.box_width(det.box) / scale
    bh_norm = sk.box_height(det.box) / scale

    # Torso orientation
    torso_deg = det.torso_angle()
    if torso_deg is not None:
        rad = math.radians(torso_deg)
        torso_cos = float(math.cos(rad))
        torso_sin = float(math.sin(rad))
        missing_torso = 0
    else:
        torso_cos = None
        torso_sin = None
        missing_torso = 1

    # Shoulder-to-hip vector
    sc = sk.shoulder_center(kps)
    hc = sk.hip_center(kps)
    if sc is not None and hc is not None:
        dx = hc[0] - sc[0]
        dy = hc[1] - sc[1]
        dist = math.hypot(dx, dy)
        s_hip_dx = dx / scale
        s_hip_dy = dy / scale
        s_hip_dist = dist / scale
    else:
        s_hip_dx = None
        s_hip_dy = None
        s_hip_dist = None

    # Widths
    sw = sk.shoulder_width(kps)
    sw_norm = (sw / scale) if sw is not None else None

    if sk.seen(kps, sk.L_HIP, sk.R_HIP):
        assert kps is not None
        hw = math.hypot(kps[sk.L_HIP][0] - kps[sk.R_HIP][0], kps[sk.L_HIP][1] - kps[sk.R_HIP][1])
        hw_norm = hw / scale
    else:
        hw_norm = None

    # Knee and Hip angles
    l_knee = None
    if sk.seen(kps, sk.L_HIP, sk.L_KNE, sk.L_ANK):
        assert kps is not None
        l_knee = compute_joint_angle(sk.pt(kps, sk.L_HIP), sk.pt(kps, sk.L_KNE), sk.pt(kps, sk.L_ANK))

    r_knee = None
    if sk.seen(kps, sk.R_HIP, sk.R_KNE, sk.R_ANK):
        assert kps is not None
        r_knee = compute_joint_angle(sk.pt(kps, sk.R_HIP), sk.pt(kps, sk.R_KNE), sk.pt(kps, sk.R_ANK))

    l_hip_ang = None
    if sk.seen(kps, sk.L_SHO, sk.L_HIP, sk.L_KNE):
        assert kps is not None
        l_hip_ang = compute_joint_angle(sk.pt(kps, sk.L_SHO), sk.pt(kps, sk.L_HIP), sk.pt(kps, sk.L_KNE))

    r_hip_ang = None
    if sk.seen(kps, sk.R_SHO, sk.R_HIP, sk.R_KNE):
        assert kps is not None
        r_hip_ang = compute_joint_angle(sk.pt(kps, sk.R_SHO), sk.pt(kps, sk.R_HIP), sk.pt(kps, sk.R_KNE))

    # Visibility masks
    vis_frac = sk.visible_fraction(kps)
    mean_conf = float(kps[:, 2].mean()) if kps is not None else 0.0

    has_torso = int(sk.seen(kps, sk.L_SHO, sk.R_SHO, sk.L_HIP, sk.R_HIP))
    has_hips = int(sk.seen(kps, sk.L_HIP) or sk.seen(kps, sk.R_HIP))
    has_knees = int(sk.seen(kps, sk.L_KNE) or sk.seen(kps, sk.R_KNE))
    has_ankles = int(sk.seen(kps, sk.L_ANK) or sk.seen(kps, sk.R_ANK))
    has_full_legs = int(
        sk.seen(kps, sk.L_HIP, sk.L_KNE, sk.L_ANK) or
        sk.seen(kps, sk.R_HIP, sk.R_KNE, sk.R_ANK)
    )

    return InstantFeatures(
        body_scale=scale,
        box_aspect_ratio=aspect,
        box_width_norm=bw_norm,
        box_height_norm=bh_norm,
        torso_angle_deg=torso_deg,
        torso_angle_cos=torso_cos,
        torso_angle_sin=torso_sin,
        shoulder_to_hip_dx_norm=s_hip_dx,
        shoulder_to_hip_dy_norm=s_hip_dy,
        shoulder_to_hip_dist_norm=s_hip_dist,
        shoulder_width_norm=sw_norm,
        hip_width_norm=hw_norm,
        left_knee_angle_deg=l_knee,
        right_knee_angle_deg=r_knee,
        left_hip_angle_deg=l_hip_ang,
        right_hip_angle_deg=r_hip_ang,
        keypoint_visible_fraction=vis_frac,
        keypoint_mean_conf=mean_conf,
        has_torso=has_torso,
        has_hips=has_hips,
        has_knees=has_knees,
        has_ankles=has_ankles,
        has_full_legs=has_full_legs,
        missing_torso_angle=missing_torso,
        missing_left_knee=int(l_knee is None),
        missing_right_knee=int(r_knee is None),
        missing_left_hip=int(l_hip_ang is None),
        missing_right_hip=int(r_hip_ang is None),
    )


@dataclass
class WindowTemporalFeatures:
    """Rolling temporal movement & kinematic features over one window duration."""

    window_duration_s: float
    sample_count: int
    actual_dt_s: float
    hip_vertical_drop: float           # Max downward drop from highest hip in window (scale-normalized)
    hip_vertical_velocity: float       # Net downward velocity: (y_now - y_start) / (dt * scale)
    torso_angle_delta: float | None    # torso_now - torso_start
    torso_angle_velocity: float | None # delta / dt (deg/s)
    aspect_ratio_delta: float          # aspect_now - aspect_start
    aspect_ratio_velocity: float       # delta / dt (1/s)
    foot_displacement: float           # Distance moved in body heights
    foot_speed: float                  # Foot displacement / dt (body heights/s)
    horizontal_posture_fraction: float # Fraction of frames in window where posture is near horizontal


@dataclass
class TemporalFeatures:
    """Multi-window temporal kinematics dictionary."""

    windows: dict[float, WindowTemporalFeatures] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        """Convert multi-window features to flat dictionary with window-prefixed keys."""
        out: dict[str, float | int] = {}
        for w_s, wf in sorted(self.windows.items()):
            prefix = f"w{int(round(w_s * 1000))}ms_"
            out[f"{prefix}sample_count"] = int(wf.sample_count)
            out[f"{prefix}actual_dt_s"] = float(wf.actual_dt_s)
            out[f"{prefix}hip_vertical_drop"] = float(wf.hip_vertical_drop)
            out[f"{prefix}hip_vertical_velocity"] = float(wf.hip_vertical_velocity)
            out[f"{prefix}torso_angle_delta"] = float(wf.torso_angle_delta) if wf.torso_angle_delta is not None else np.nan
            out[f"{prefix}torso_angle_velocity"] = float(wf.torso_angle_velocity) if wf.torso_angle_velocity is not None else np.nan
            out[f"{prefix}aspect_ratio_delta"] = float(wf.aspect_ratio_delta)
            out[f"{prefix}aspect_ratio_velocity"] = float(wf.aspect_ratio_velocity)
            out[f"{prefix}foot_displacement"] = float(wf.foot_displacement)
            out[f"{prefix}foot_speed"] = float(wf.foot_speed)
            out[f"{prefix}horizontal_posture_fraction"] = float(wf.horizontal_posture_fraction)
        return out


@dataclass
class ObservationRecord:
    """Internal historical record for one observation of a track."""

    timestamp: float
    frame_idx: int
    scale: float
    center: Point
    foot: Point
    hip_y: float                       # Hip y if visible, else center y
    torso_angle_deg: float | None
    aspect_ratio: float
    is_horizontal: bool


def extract_window_temporal_features(
    history: list[ObservationRecord],
    current_idx: int,
    window_duration_s: float,
) -> WindowTemporalFeatures:
    """
    Compute strictly causal rolling temporal features over [t_now - window_duration_s, t_now].

    Only considers observations in history[:current_idx + 1].
    """
    assert current_idx < len(history)
    cur = history[current_idx]
    now_ts = cur.timestamp
    scale = max(1.0, cur.scale)
    cutoff = now_ts - window_duration_s

    # Subsequence of past observations within window
    window = [obs for obs in history[:current_idx + 1] if obs.timestamp >= cutoff]

    if not window:
        # Fallback for empty window
        return WindowTemporalFeatures(
            window_duration_s=window_duration_s,
            sample_count=0,
            actual_dt_s=0.0,
            hip_vertical_drop=0.0,
            hip_vertical_velocity=0.0,
            torso_angle_delta=None,
            torso_angle_velocity=None,
            aspect_ratio_delta=0.0,
            aspect_ratio_velocity=0.0,
            foot_displacement=0.0,
            foot_speed=0.0,
            horizontal_posture_fraction=1.0 if cur.is_horizontal else 0.0,
        )

    first = window[0]
    actual_dt = max(1e-4, now_ts - first.timestamp)

    # 1. Hip vertical drop: (y_now - y_peak) / scale
    # Note: image coordinates increase downwards; smallest y is highest position
    peak_y = min(obs.hip_y for obs in window)
    hip_drop = float(max(0.0, cur.hip_y - peak_y) / scale)

    # 2. Hip vertical velocity: (y_now - y_start) / (dt * scale)
    hip_vel = float((cur.hip_y - first.hip_y) / (actual_dt * scale))

    # 3. Torso angle change
    if cur.torso_angle_deg is not None and first.torso_angle_deg is not None:
        torso_delta = float(cur.torso_angle_deg - first.torso_angle_deg)
        torso_vel = float(torso_delta / actual_dt)
    else:
        torso_delta = None
        torso_vel = None

    # 4. Aspect ratio change
    aspect_delta = float(cur.aspect_ratio - first.aspect_ratio)
    aspect_vel = float(aspect_delta / actual_dt)

    # 5. Foot movement
    foot_disp = float(math.hypot(cur.foot[0] - first.foot[0], cur.foot[1] - first.foot[1]) / scale)
    foot_spd = float(foot_disp / actual_dt)

    # 6. Horizontal posture fraction
    horiz_count = sum(1 for obs in window if obs.is_horizontal)
    horiz_frac = float(horiz_count / len(window))

    return WindowTemporalFeatures(
        window_duration_s=window_duration_s,
        sample_count=len(window),
        actual_dt_s=round(actual_dt, 4),
        hip_vertical_drop=round(hip_drop, 4),
        hip_vertical_velocity=round(hip_vel, 4),
        torso_angle_delta=round(torso_delta, 2) if torso_delta is not None else None,
        torso_angle_velocity=round(torso_vel, 2) if torso_vel is not None else None,
        aspect_ratio_delta=round(aspect_delta, 4),
        aspect_ratio_velocity=round(aspect_vel, 4),
        foot_displacement=round(foot_disp, 4),
        foot_speed=round(foot_spd, 4),
        horizontal_posture_fraction=round(horiz_frac, 4),
    )


class FeatureExtractor:
    """
    Stateful feature extractor that processes frame-by-frame detections for a track.

    Maintains rolling history per track_id to compute causal instantaneous and
    temporal features.
    """

    def __init__(self, windows: tuple[float, ...] = DEFAULT_WINDOWS) -> None:
        self.windows = tuple(sorted(windows))
        # track_id -> list of ObservationRecord
        self._history: dict[int, list[ObservationRecord]] = {}

    def reset(self) -> None:
        """Clear all historical track buffers."""
        self._history.clear()

    def process_detection(
        self,
        det: Detection,
        timestamp: float,
        frame_idx: int,
    ) -> dict[str, float | int]:
        """
        Extract complete feature dictionary for one Detection at timestamp.

        Features extracted are strictly causal (using only timestamps <= timestamp).
        """
        # 1. Instantaneous features
        inst = extract_instant_features(det)
        inst_dict = inst.to_dict()

        # Determine horizontal state
        torso_horiz = (inst.torso_angle_deg is not None and inst.torso_angle_deg >= 55.0)
        box_horiz = (inst.box_aspect_ratio >= 1.15)
        is_horiz = torso_horiz or box_horiz

        # 2. Record historical observation
        kps = det.keypoints
        hip = sk.hip_center(kps)
        hip_y = hip[1] if hip is not None else det.center()[1]

        rec = ObservationRecord(
            timestamp=timestamp,
            frame_idx=frame_idx,
            scale=inst.body_scale,
            center=det.center(),
            foot=det.foot(),
            hip_y=hip_y,
            torso_angle_deg=inst.torso_angle_deg,
            aspect_ratio=inst.box_aspect_ratio,
            is_horizontal=is_horiz,
        )

        track_hist = self._history.setdefault(det.track_id, [])
        track_hist.append(rec)
        current_idx = len(track_hist) - 1

        # 3. Temporal features over configured windows
        tf_dict: dict[str, float | int] = {}
        for w_s in self.windows:
            wf = extract_window_temporal_features(track_hist, current_idx, w_s)
            prefix = f"w{int(round(w_s * 1000))}ms_"
            tf_dict[f"{prefix}sample_count"] = int(wf.sample_count)
            tf_dict[f"{prefix}actual_dt_s"] = float(wf.actual_dt_s)
            tf_dict[f"{prefix}hip_vertical_drop"] = float(wf.hip_vertical_drop)
            tf_dict[f"{prefix}hip_vertical_velocity"] = float(wf.hip_vertical_velocity)
            tf_dict[f"{prefix}torso_angle_delta"] = float(wf.torso_angle_delta) if wf.torso_angle_delta is not None else np.nan
            tf_dict[f"{prefix}torso_angle_velocity"] = float(wf.torso_angle_velocity) if wf.torso_angle_velocity is not None else np.nan
            tf_dict[f"{prefix}aspect_ratio_delta"] = float(wf.aspect_ratio_delta)
            tf_dict[f"{prefix}aspect_ratio_velocity"] = float(wf.aspect_ratio_velocity)
            tf_dict[f"{prefix}foot_displacement"] = float(wf.foot_displacement)
            tf_dict[f"{prefix}foot_speed"] = float(wf.foot_speed)
            tf_dict[f"{prefix}horizontal_posture_fraction"] = float(wf.horizontal_posture_fraction)

        # Merge and return
        out = {**inst_dict, **tf_dict}
        return out


def extract_features_for_sequence(
    sequence,
    windows: tuple[float, ...] = DEFAULT_WINDOWS,
) -> list[dict[str, Any]]:
    """
    Extract feature dictionaries for all detections in a Sequence in chronological order.

    Returns:
        List of dictionaries containing metadata, ground truth, and features.
    """
    extractor = FeatureExtractor(windows=windows)
    rows: list[dict[str, Any]] = []

    ann = sequence.annotation
    seq_id = ann.sequence_id

    for obs in sequence.frames:
        t = obs.timestamp
        fidx = obs.frame_idx

        for det in obs.detections:
            feat = extractor.process_detection(det, timestamp=t, frame_idx=fidx)

            row: dict[str, Any] = {
                "sequence_id": seq_id,
                "track_id": det.track_id,
                "frame_idx": fidx,
                "timestamp": round(float(t), 4),
                **feat,
            }
            rows.append(row)

    return rows
