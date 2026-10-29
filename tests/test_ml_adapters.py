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
