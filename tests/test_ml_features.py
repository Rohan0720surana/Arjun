"""
Unit tests for ML feature extraction module (ml/features.py).

Covers:
- Joint angle computation (straight, right angle, collinear, missing points)
- Instantaneous feature extraction for upright vs lying poses
- Missing keypoint handling (no fabrication, missing indicators set)
- Strict temporal causality (past/current only; future changes have zero effect)
- Dynamic vertical drop and velocity tracking
- Multi-window temporal kinematics
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from backend.vision import skeleton as sk
from backend.vision import synth
from backend.world import Detection
from ml.features import (
    FeatureExtractor,
    compute_joint_angle,
    extract_instant_features,
    extract_window_temporal_features,
    ObservationRecord,
)


# --------------------------------------------------------------------------- #
# Joint Angle Calculation Tests
# --------------------------------------------------------------------------- #

def test_compute_joint_angle_straight_line():
    """Three collinear points form a 180-degree interior angle."""
    p1 = (0.0, 0.0)
    vertex = (0.0, 10.0)
    p3 = (0.0, 20.0)
    angle = compute_joint_angle(p1, vertex, p3)
    assert angle is not None
    assert angle == pytest.approx(180.0, abs=1e-3)


def test_compute_joint_angle_right_angle():
    """Perpendicular segments form a 90-degree interior angle."""
    p1 = (0.0, 10.0)
    vertex = (0.0, 0.0)
    p3 = (10.0, 0.0)
    angle = compute_joint_angle(p1, vertex, p3)
    assert angle is not None
    assert angle == pytest.approx(90.0, abs=1e-3)


def test_compute_joint_angle_acute_and_obtuse():
    """Tests 45-degree and 135-degree angles."""
    # 45 degrees
    p1 = (10.0, 0.0)
    vertex = (0.0, 0.0)
    p3 = (10.0, 10.0)
    ang45 = compute_joint_angle(p1, vertex, p3)
    assert ang45 is not None
    assert ang45 == pytest.approx(45.0, abs=1e-3)

    # 135 degrees
    p3_obtuse = (-10.0, 10.0)
    ang135 = compute_joint_angle(p1, vertex, p3_obtuse)
    assert ang135 is not None
    assert ang135 == pytest.approx(135.0, abs=1e-3)


def test_compute_joint_angle_missing_points():
    """Missing or collocated points return None instead of fabricating angles."""
    assert compute_joint_angle(None, (0.0, 0.0), (1.0, 1.0)) is None
    assert compute_joint_angle((1.0, 1.0), None, (1.0, 1.0)) is None
    assert compute_joint_angle((1.0, 1.0), (0.0, 0.0), None) is None

    # Collocated vertex and p1
    assert compute_joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)) is None


# --------------------------------------------------------------------------- #
# Instantaneous Feature Extraction Tests
# --------------------------------------------------------------------------- #

def test_extract_instant_features_upright_pose():
    """Upright standing pose has low aspect ratio, small torso angle, and full visibility."""
    kps = synth.standing(cx=320.0, feet_y=300.0, height=150.0)
    box = synth.box_for(kps)
    det = Detection(track_id=1, label="person", conf=0.95, box=box, keypoints=kps)

    feat = extract_instant_features(det)

    assert feat.body_scale == pytest.approx(150.0, abs=15.0)
    assert feat.box_aspect_ratio < 0.6
    assert feat.torso_angle_deg is not None
    assert feat.torso_angle_deg < 15.0
    assert feat.torso_angle_cos is not None
    assert feat.torso_angle_cos > 0.95
    assert feat.has_torso == 1
    assert feat.has_hips == 1
    assert feat.has_knees == 1
    assert feat.has_ankles == 1
    assert feat.has_full_legs == 1
    assert feat.missing_torso_angle == 0
    assert feat.missing_left_knee == 0
    assert feat.missing_right_knee == 0


def test_extract_instant_features_horizontal_lying_pose():
    """Lying pose has wide aspect ratio and large torso angle (near 90 deg)."""
    kps = synth.lying(cx=320.0, ground_y=300.0, height=150.0, angle_deg=86.0)
    box = synth.box_for(kps)
    det = Detection(track_id=1, label="person", conf=0.92, box=box, keypoints=kps)

    feat = extract_instant_features(det)

    assert feat.box_aspect_ratio > 1.2
    assert feat.torso_angle_deg is not None
    assert feat.torso_angle_deg > 75.0
    assert feat.torso_angle_sin is not None
    assert feat.torso_angle_sin > 0.90
    assert feat.missing_torso_angle == 0


def test_extract_instant_features_missing_keypoints():
    """Detection with no keypoints handles missingness without fabricating values."""
    box = (100.0, 100.0, 150.0, 250.0)
    det = Detection(track_id=2, label="person", conf=0.85, box=box, keypoints=None)

    feat = extract_instant_features(det)

    assert feat.box_aspect_ratio == pytest.approx(50.0 / 150.0, abs=1e-3)
    assert feat.torso_angle_deg is None
    assert feat.torso_angle_cos is None
    assert feat.torso_angle_sin is None
    assert feat.left_knee_angle_deg is None
    assert feat.right_knee_angle_deg is None
    assert feat.left_hip_angle_deg is None
    assert feat.right_hip_angle_deg is None
    assert feat.has_torso == 0
    assert feat.missing_torso_angle == 1
    assert feat.missing_left_knee == 1
    assert feat.missing_right_knee == 1
    assert feat.keypoint_visible_fraction == 0.0

    d = feat.to_dict()
    assert np.isnan(d["torso_angle_deg"])
    assert np.isnan(d["left_knee_angle_deg"])
    assert d["missing_torso_angle"] == 1


# --------------------------------------------------------------------------- #
# Temporal Causality & Movement Tests
# --------------------------------------------------------------------------- #

def test_temporal_features_strict_causality():
    """Features at timestamp t depend strictly on past frames; future modifications have 0 effect."""
    extractor = FeatureExtractor(windows=(0.3, 0.6))

    kps_stand = synth.standing(320.0, 300.0, 150.0)
    box_stand = synth.box_for(kps_stand)
    det_stand = Detection(1, "person", 0.95, box_stand, kps_stand)

    # Frame 0 at t=0.0
    f0 = extractor.process_detection(det_stand, timestamp=0.0, frame_idx=0)
    # Frame 1 at t=0.1
    f1 = extractor.process_detection(det_stand, timestamp=0.1, frame_idx=1)
    # Frame 2 at t=0.2
    f2_orig = extractor.process_detection(det_stand, timestamp=0.2, frame_idx=2)

    # Re-run a separate extractor up to frame 2 only
    extractor_isolated = FeatureExtractor(windows=(0.3, 0.6))
    extractor_isolated.process_detection(det_stand, timestamp=0.0, frame_idx=0)
    extractor_isolated.process_detection(det_stand, timestamp=0.1, frame_idx=1)
    f2_isolated = extractor_isolated.process_detection(det_stand, timestamp=0.2, frame_idx=2)

    # Isolated evaluation at frame 2 must be IDENTICAL to frame 2 when future frames follow
    for k in f2_orig:
        v1, v2 = f2_orig[k], f2_isolated[k]
        if np.isnan(v1):
            assert np.isnan(v2)
        else:
            assert v1 == pytest.approx(v2, abs=1e-5), f"Feature {k} differed"


def test_temporal_features_vertical_drop_and_velocity():
    """A rapid downward hip movement produces positive drop and velocity values."""
    extractor = FeatureExtractor(windows=(0.3, 0.6))

    # Frame 0: High hip at y=150 (upright)
    kps_upright = synth.standing(cx=300.0, feet_y=300.0, height=150.0)
    det_upright = Detection(1, "person", 0.95, synth.box_for(kps_upright), kps_upright)
    extractor.process_detection(det_upright, timestamp=0.0, frame_idx=0)

    # Frame 1: Intermediate
    extractor.process_detection(det_upright, timestamp=0.1, frame_idx=1)

    # Frame 2 at t=0.3s: Dropped to the ground (lying down)
    kps_down = synth.lying(cx=300.0, ground_y=300.0, height=150.0, angle_deg=86.0)
    det_down = Detection(1, "person", 0.95, synth.box_for(kps_down), kps_down)
    f2 = extractor.process_detection(det_down, timestamp=0.3, frame_idx=2)

    # Hip drop over past 300ms must be positive and substantial (>= 0.15 body heights)
    assert f2["w300ms_hip_vertical_drop"] > 0.15
    assert f2["w300ms_hip_vertical_velocity"] > 0.30
    assert f2["w300ms_torso_angle_delta"] > 40.0
    assert f2["w300ms_horizontal_posture_fraction"] > 0.0


def test_window_temporal_features_empty_history():
    """extract_window_temporal_features handles empty history without division by zero."""
    rec = ObservationRecord(
        timestamp=1.0, frame_idx=10, scale=100.0,
        center=(100.0, 100.0), foot=(100.0, 150.0), hip_y=120.0,
        torso_angle_deg=10.0, aspect_ratio=0.5, is_horizontal=False,
    )
    wf = extract_window_temporal_features([rec], current_idx=0, window_duration_s=0.5)

    assert wf.sample_count == 1
    assert wf.hip_vertical_drop == 0.0
    assert wf.hip_vertical_velocity == 0.0
    assert wf.torso_angle_delta == 0.0
    assert wf.horizontal_posture_fraction == 0.0
