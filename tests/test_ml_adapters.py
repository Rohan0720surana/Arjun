"""
Unit tests for multi-dataset adapter architecture (Phase 4E).

Covers:
- StandardizedAnnotation behavior and dwell calculations
- Prediction-independent eligibility verification
- URFDAdapter discovery, annotation parsing, and sequence loading
- Blueprint adapters for Le2i and UP-Fall datasets
- Subject identity preservation for Leave-One-Subject-Out validation
- Separation of ground-truth labels from inference feature matrices
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from ml.adapters import (
    DatasetAdapter,
    Le2iDatasetAdapter,
    StandardizedAnnotation,
    UPFallDatasetAdapter,
    URFDAdapter,
)


def test_standardized_annotation_post_fallen_dwell_calculation():
    """Calculates post-fallen dwell accurately and identifies observation sufficiency."""
    # Fall with insufficient observation (1.4s post-fall < 2.0s requirement)
    ann_short = StandardizedAnnotation(
        sequence_id="fall_test_short",
        dataset_name="TestDataset",
        is_fall=True,
        duration_s=4.4,
        fps=30.0,
        total_frames=132,
        fall_transition_start_s=2.0,
        fallen_state_start_s=3.0,
    )
    assert ann_short.post_fallen_dwell_s == pytest.approx(1.4, abs=1e-3)
    assert not ann_short.is_eligible_for_confirmation(required_confirmation_s=2.0)
    assert ann_short.is_eligible_for_confirmation(required_confirmation_s=1.0)

    # Fall with sufficient observation (5.0s post-fall >= 2.0s requirement)
    ann_long = StandardizedAnnotation(
        sequence_id="fall_test_long",
        dataset_name="TestDataset",
        is_fall=True,
        duration_s=8.0,
        fps=30.0,
        total_frames=240,
        fall_transition_start_s=1.5,
        fallen_state_start_s=3.0,
    )
    assert ann_long.post_fallen_dwell_s == pytest.approx(5.0, abs=1e-3)
    assert ann_long.is_eligible_for_confirmation(required_confirmation_s=2.0)
    assert ann_long.is_eligible_for_confirmation(required_confirmation_s=5.0)

    # ADL sequence (non-fall is always eligible for false-alert evaluation)
    ann_adl = StandardizedAnnotation(
        sequence_id="adl_test",
        dataset_name="TestDataset",
        is_fall=False,
        duration_s=6.0,
        fps=30.0,
        total_frames=180,
    )
    assert ann_adl.post_fallen_dwell_s is None
    assert ann_adl.is_eligible_for_confirmation(required_confirmation_s=2.0)


def test_urfd_adapter_discovers_and_loads_10_sequences():
    """URFDAdapter discovers all 10 canonical sequences and parses annotations correctly."""
    adapter = URFDAdapter()
    recordings = adapter.discover_recordings()

    assert len(recordings) == 10
    assert "fall-01-cam0-rgb" in recordings
    assert "adl-01-cam0-rgb" in recordings

    # Check fall annotation
    ann_fall = adapter.load_annotation("fall-01-cam0-rgb")
    assert ann_fall.is_fall is True
    assert ann_fall.dataset_name == "URFD"
    assert ann_fall.camera_id == "cam0"
    assert ann_fall.duration_s > 0.0
    assert ann_fall.fallen_state_start_s is not None
    # All 5 URFD falls fail the 2.0s dwell eligibility requirement
    assert not ann_fall.is_eligible_for_confirmation(required_confirmation_s=2.0)

    # Check ADL annotation
    ann_adl = adapter.load_annotation("adl-01-cam0-rgb")
    assert ann_adl.is_fall is False
    assert ann_adl.is_eligible_for_confirmation(required_confirmation_s=2.0)

    # Load sequence object
    seq = adapter.load_sequence("adl-01-cam0-rgb")
    assert len(seq.frames) == 150
    assert seq.fps == 30.0


def test_upfall_adapter_preserves_subject_identity_for_loso():
    """UPFallDatasetAdapter preserves subject IDs to enable Leave-One-Subject-Out cross validation."""
    adapter = UPFallDatasetAdapter(root_dir="data/up_fall", cache_dir="eval/cache/up_fall")

    ann = adapter.load_annotation("Subject5_Activity2_Trial1")
    assert ann.dataset_name == "UP-Fall"
    assert ann.subject_id == "Subject5"
    assert ann.activity_type == "fall"  # Activity 2 is a fall

    ann_adl = adapter.load_annotation("Subject12_Activity7_Trial2")
    assert ann_adl.subject_id == "Subject12"
    assert ann_adl.activity_type == "adl"   # Activity 7 is an ADL


def test_le2i_adapter_blueprint_metadata():
    """Le2iDatasetAdapter assigns correct dataset metadata and camera properties."""
    adapter = Le2iDatasetAdapter(root_dir="data/le2i", cache_dir="eval/cache/le2i")
    ann = adapter.load_annotation("Home_fall_video (12)")

    assert ann.dataset_name == "Le2i"
    assert ann.fps == 25.0
    assert ann.camera_id == "fixed_cam"
    assert ann.is_fall is True


def test_upfall_activity_inventory_and_unmeasured_dwell_safeguard():
    """Ensure UP-Fall activities are defined and unmeasured fall dwell does not assume confirmation eligibility."""
    from ml.adapters import UPFALL_ACTIVITIES

    # Check 11 activities defined
    assert len(UPFALL_ACTIVITIES) == 11
    assert UPFALL_ACTIVITIES[1]["is_fall"] is True
    assert UPFALL_ACTIVITIES[9]["name"] == "picking_up_object"
    assert UPFALL_ACTIVITIES[9]["is_fall"] is False
    assert UPFALL_ACTIVITIES[11]["name"] == "lying"
    assert UPFALL_ACTIVITIES[11]["is_fall"] is False

    # When CSV is missing (e.g. Subject99), adapter does NOT assume dwell
    adapter = UPFallDatasetAdapter(root_dir="data/up_fall", cache_dir="eval/cache/up_fall")
    ann_missing = adapter.load_annotation("Subject99_Activity1_Trial1")
    assert ann_missing.is_fall is True
    assert ann_missing.fallen_state_start_s is None
    assert ann_missing.metadata["dwell_measured"] is False
    assert ann_missing.is_eligible_for_confirmation(required_confirmation_s=2.0) is False

    # When real CSV is present (Subject1), adapter measures empirical dwell
    ann_real = adapter.load_annotation("Subject1_Activity1_Trial1")
    assert ann_real.is_fall is True
    assert ann_real.metadata["dwell_measured"] is True
    assert ann_real.fallen_state_start_s == pytest.approx(5.800, abs=0.01)
    assert ann_real.post_fallen_dwell_s == pytest.approx(4.080, abs=0.01)
    assert ann_real.is_eligible_for_confirmation(required_confirmation_s=2.0) is True


def test_upfall_csv_empirical_dwell_measurement(tmp_path):
    """Verify that UP-Fall CSV timestamps measure post-fall dwell empirically."""
    from ml.adapters import parse_upfall_timestamps_csv

    # Construct a synthetic UP-Fall trial CSV with microsecond timestamps
    csv_file = tmp_path / "Subject1Activity1Trial1.csv"
    timestamps = [
        "2018-07-04T12:00:00.000000",  # t=0.0s, state 7 (standing)
        "2018-07-04T12:00:01.000000",  # t=1.0s, state 7
        "2018-07-04T12:00:02.000000",  # t=2.0s, state 1 (fall transition)
        "2018-07-04T12:00:03.000000",  # t=3.0s, state 1
        "2018-07-04T12:00:04.000000",  # t=4.0s, state 11 (lying)
        "2018-07-04T12:00:07.000000",  # t=7.0s, state 11 (lying)
        "2018-07-04T12:00:10.000000",  # t=10.0s, state 11 (lying)
    ]
    activities = [7, 7, 1, 1, 11, 11, 11]
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Sensor1": [0.1] * 7,
        "Subject": [1] * 7,
        "Activity": activities,
        "Trial": [1] * 7,
    })
    df.to_csv(csv_file, index=False)

    measured = parse_upfall_timestamps_csv(csv_file)
    assert measured["duration_s"] == 10.0
    assert measured["fall_transition_start_s"] == 2.0
    assert measured["fallen_state_start_s"] == 4.0
    assert measured["post_fallen_dwell_s"] == 6.0  # 10.0 - 4.0 = 6.0s

    # Point adapter to directory with this CSV
    subj_dir = tmp_path / "Subject1" / "Activity1" / "Trial1"
    subj_dir.mkdir(parents=True)
    df.to_csv(subj_dir / "Subject1Activity1Trial1.csv", index=False)

    adapter = UPFallDatasetAdapter(root_dir=tmp_path, cache_dir=tmp_path / "cache")
    ann = adapter.load_annotation("Subject1_Activity1_Trial1")

    assert ann.metadata["dwell_measured"] is True
    assert ann.duration_s == 10.0
    assert ann.fall_transition_start_s == 2.0
    assert ann.fallen_state_start_s == 4.0
    assert ann.post_fallen_dwell_s == 6.0
    assert ann.is_eligible_for_confirmation(required_confirmation_s=2.0) is True
    assert ann.is_eligible_for_confirmation(required_confirmation_s=5.0) is True


def test_upfall_activity11_intentional_lying_not_interpreted_as_fall(tmp_path):
    """
    Regression test: Activity 11 in UP-Fall is an intentional lying ADL, NOT a fall.
    Activity code 11 in this context must not be interpreted as post-fall lying.
    """
    from ml.adapters import parse_upfall_timestamps_csv, UPFallDatasetAdapter

    # Construct synthetic Activity 11 CSV with tag 11 throughout
    csv_file = tmp_path / "Subject1Activity11Trial1.csv"
    timestamps = [
        "2018-07-04T12:00:00.000000",
        "2018-07-04T12:00:01.000000",
        "2018-07-04T12:00:02.000000",
        "2018-07-04T12:00:05.000000",
    ]
    activities = [11, 11, 11, 11]  # Intentional lying ADL
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Sensor1": [0.0] * 4,
        "Subject": [1] * 4,
        "Activity": activities,
        "Trial": [1] * 4,
    })
    df.to_csv(csv_file, index=False)

    measured = parse_upfall_timestamps_csv(csv_file)
    assert measured["duration_s"] == 5.0
    assert measured["fall_transition_start_s"] is None
    assert measured["fallen_state_start_s"] is None
    assert measured["post_fallen_dwell_s"] is None

    # Load via adapter
    subj_dir = tmp_path / "Subject1" / "Activity11" / "Trial1"
    subj_dir.mkdir(parents=True)
    df.to_csv(subj_dir / "Subject1Activity11Trial1.csv", index=False)

    adapter = UPFallDatasetAdapter(root_dir=tmp_path, cache_dir=tmp_path / "cache")
    ann = adapter.load_annotation("Subject1_Activity11_Trial1")

    assert ann.is_fall is False
    assert ann.activity_type == "adl"
    assert ann.fall_transition_start_s is None
    assert ann.fallen_state_start_s is None
    assert ann.post_fallen_dwell_s is None
    assert ann.metadata["dwell_measured"] is False
    assert ann.is_eligible_for_confirmation() is True


def test_upfall_activity_code_11_dual_use_recording_context(tmp_path):
    """
    Regression test: Code 11 in Activities 1-5 represents post-fall lying,
    whereas in Activity 11 it represents intentional lying down (ADL).
    """
    from ml.adapters import parse_upfall_timestamps_csv

    # Same timestamp and activity tags [7, 7, 1, 1, 11, 11]
    timestamps = [
        "2018-07-04T12:00:00.000000",  # t=0.0s, standing (7)
        "2018-07-04T12:00:01.000000",  # t=1.0s, standing (7)
        "2018-07-04T12:00:02.000000",  # t=2.0s, fall transition (1)
        "2018-07-04T12:00:03.000000",  # t=3.0s, fall transition (1)
        "2018-07-04T12:00:04.000000",  # t=4.0s, code 11
        "2018-07-04T12:00:06.000000",  # t=6.0s, code 11
    ]
    activities = [7, 7, 1, 1, 11, 11]
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Sensor1": [0.0] * 6,
        "Subject": [1] * 6,
        "Activity": activities,
        "Trial": [1] * 6,
    })

    # When evaluated in fall context (Activity 1)
    fall_csv = tmp_path / "Subject1Activity1Trial1.csv"
    df.to_csv(fall_csv, index=False)
    measured_fall = parse_upfall_timestamps_csv(fall_csv)
    assert measured_fall["fall_transition_start_s"] == 2.0
    assert measured_fall["fallen_state_start_s"] == 4.0
    assert measured_fall["post_fallen_dwell_s"] == 2.0

    # When evaluated in ADL context (explicitly passed activity_num=11)
    measured_adl = parse_upfall_timestamps_csv(fall_csv, activity_num=11)
    assert measured_adl["fallen_state_start_s"] is None
    assert measured_adl["post_fallen_dwell_s"] is None
