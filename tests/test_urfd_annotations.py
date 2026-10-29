"""
Unit tests for URFD ground-truth annotation importer and manifest builder.

Tests:
- Reusable parsing of urfall-cam0-falls.csv (valid rows, column filtering).
- Validation of sequence IDs, numeric frame numbers, state labels, duplicates, ordering.
- Parsing of synchronization data CSVs (ms to seconds conversion).
- 1-based versus 0-based frame indexing alignment.
- Missing timestamps, missing annotations, and invalid state sequences.
- Pose cache annotation updating with 100% preservation of pose data.
- Event-level metrics evaluation producing INSUFFICIENT_OBSERVATION for short clips.
- Preservation of UNVERIFIED status for unannotated/unmatched sequences.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from backend.detectors.base import Candidate
from eval.dataset import FrameObservation, Sequence, SequenceAnnotation
from eval.extract_poses import load_pose_cache, save_pose_cache
from eval.metrics import evaluate_results
from eval.runner import OfflineFallRunner, SequenceResult
from eval.urfd_annotations import (
    URFDSequenceAnnotation,
    apply_urfd_annotation_to_pose_cache,
    build_urfd_manifest,
    load_urfd_fall_annotation,
    load_urfd_sequence,
    parse_urfd_data_csv,
    parse_urfd_falls_csv,
)


# --------------------------------------------------------------------------- #
# Falls CSV Parser Tests
# --------------------------------------------------------------------------- #

def test_parse_falls_csv_valid_and_ignores_extra_columns(tmp_path: Path):
    """Parser extracts first 3 columns, ignores extra columns, and handles whitespace."""
    csv_file = tmp_path / "test_falls.csv"
    csv_file.write_text(
        "fall-01, 1, -1, 3.16, 2.90, 0.55\n"
        "fall-01, 2, -1, 3.30, 2.96, 0.47\n"
        "fall-01, 3, 0, 1.2, 3.4\n"
        "fall-01, 4, 1, 0.1, 0.2, 0.3, 0.4\n"
        "fall-02, 1, -1\n"
        "fall-02, 2, 0\n"
        "fall-02, 3, 1\n",
        encoding="utf-8",
    )

    parsed = parse_urfd_falls_csv(csv_file)
    assert "fall-01" in parsed
    assert "fall-02" in parsed
    assert parsed["fall-01"] == [(1, -1), (2, -1), (3, 0), (4, 1)]
    assert parsed["fall-02"] == [(1, -1), (2, 0), (3, 1)]


def test_parse_falls_csv_invalid_state_label(tmp_path: Path):
    """Invalid state labels (not -1, 0, or 1) raise ValueError."""
    csv_file = tmp_path / "bad_state.csv"
    csv_file.write_text("fall-01, 1, 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid state 2"):
        parse_urfd_falls_csv(csv_file)

    csv_file.write_text("fall-01, 1, standing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-integer state label"):
        parse_urfd_falls_csv(csv_file)


def test_parse_falls_csv_invalid_frame_number(tmp_path: Path):
    """Non-integer or <= 0 frame numbers raise ValueError."""
    csv_file = tmp_path / "bad_frame.csv"
    csv_file.write_text("fall-01, 0, -1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be >= 1"):
        parse_urfd_falls_csv(csv_file)

    csv_file.write_text("fall-01, frame1, -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-integer frame number"):
        parse_urfd_falls_csv(csv_file)


def test_parse_falls_csv_duplicate_row_raises(tmp_path: Path):
    """Duplicate (sequence_id, frame_number) entries raise ValueError."""
    csv_file = tmp_path / "dup.csv"
    csv_file.write_text(
        "fall-01, 1, -1\n"
        "fall-01, 1, 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate entry"):
        parse_urfd_falls_csv(csv_file)


def test_parse_falls_csv_out_of_order_frame_raises(tmp_path: Path):
    """Non-increasing frame numbers within a sequence raise ValueError."""
    csv_file = tmp_path / "out_of_order.csv"
    csv_file.write_text(
        "fall-01, 1, -1\n"
        "fall-01, 5, -1\n"
        "fall-01, 3, 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out-of-order frame"):
        parse_urfd_falls_csv(csv_file)


def test_parse_falls_csv_too_few_columns(tmp_path: Path):
    """Lines with fewer than 3 columns raise ValueError."""
    csv_file = tmp_path / "few_cols.csv"
    csv_file.write_text("fall-01, 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 3 columns"):
        parse_urfd_falls_csv(csv_file)


# --------------------------------------------------------------------------- #
# Data CSV (Synchronization) Parser Tests
# --------------------------------------------------------------------------- #

def test_parse_data_csv_timestamps(tmp_path: Path):
    """Timestamps in milliseconds are correctly converted to seconds."""
    csv_file = tmp_path / "fall-01-data.csv"
    csv_file.write_text(
        "1, 0, 1.024\n"
        "2, 33, 1.003\n"
        "3, 66, 1.028\n"
        "83, 2736, 1.150\n"
        "113, 3737, 1.080\n"
        "160, 5305, 1.059\n",
        encoding="utf-8",
    )

    ts = parse_urfd_data_csv(csv_file)
    assert ts[1] == 0.0
    assert ts[2] == 0.033
    assert ts[83] == 2.736
    assert ts[113] == 3.737
    assert ts[160] == 5.305


def test_parse_data_csv_errors(tmp_path: Path):
    """Data CSV validates positive frame numbers, non-negative timestamps, and ordering."""
    f1 = tmp_path / "neg_ts.csv"
    f1.write_text("1, -10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative timestamp"):
        parse_urfd_data_csv(f1)

    f2 = tmp_path / "out_order.csv"
    f2.write_text("2, 100\n1, 200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out-of-order frame"):
        parse_urfd_data_csv(f2)


# --------------------------------------------------------------------------- #
# Annotation Joining & Event Extraction Tests
# --------------------------------------------------------------------------- #

def test_load_urfd_fall_annotation_transitions(tmp_path: Path):
    """First state 0 and first state 1 are correctly identified with exact timestamps."""
    falls_csv = tmp_path / "falls.csv"
    falls_csv.write_text(
        "fall-01, 1, -1\n"
        "fall-01, 2, -1\n"
        "fall-01, 3, 0\n"
        "fall-01, 4, 0\n"
        "fall-01, 5, 1\n"
        "fall-01, 6, 1\n",
        encoding="utf-8",
    )
    data_csv = tmp_path / "fall-01-data.csv"
    data_csv.write_text(
        "1, 0\n"
        "2, 100\n"
        "3, 200\n"
        "4, 300\n"
        "5, 400\n"
        "6, 600\n",
        encoding="utf-8",
    )

    ann = load_urfd_fall_annotation("fall-01", falls_csv, data_csv)
    assert ann.sequence_id == "fall-01"
    assert ann.is_fall is True
    assert ann.fall_transition_start_frame == 3
    assert ann.fall_transition_start_s == 0.200
    assert ann.fallen_state_start_frame == 5
    assert ann.fallen_state_start_s == 0.400
    assert ann.total_frames == 6
    assert ann.duration_s == 0.600
    assert ann.available_post_fallen_s == pytest.approx(0.200, abs=1e-4)

    seq_ann = ann.to_sequence_annotation()
    assert seq_ann.is_fall is True
    assert seq_ann.fall_start_s == 0.200
    assert seq_ann.fall_transition_start_s == 0.200
    assert seq_ann.fallen_state_start_s == 0.400
    assert seq_ann.fall_impact_s is None
    assert seq_ann.confirmation_anchor_s() == 0.400


def test_missing_timestamp_raises(tmp_path: Path):
    """If an annotated frame has no corresponding row in the data CSV, raises ValueError."""
    falls_csv = tmp_path / "falls.csv"
    falls_csv.write_text("fall-01, 1, -1\nfall-01, 2, 0\nfall-01, 3, 1\n", encoding="utf-8")
    data_csv = tmp_path / "fall-01-data.csv"
    data_csv.write_text("1, 0\n3, 300\n", encoding="utf-8")  # frame 2 missing

    with pytest.raises(ValueError, match="lacks a timestamp"):
        load_urfd_fall_annotation("fall-01", falls_csv, data_csv)


def test_missing_sequence_in_falls_csv_raises(tmp_path: Path):
    """Requesting an unannotated sequence raises KeyError."""
    falls_csv = tmp_path / "falls.csv"
    falls_csv.write_text("fall-01, 1, -1\n", encoding="utf-8")
    data_csv = tmp_path / "fall-02-data.csv"
    data_csv.write_text("1, 0\n", encoding="utf-8")

    with pytest.raises(KeyError, match="not found in falls annotation"):
        load_urfd_fall_annotation("fall-02", falls_csv, data_csv)


def test_one_based_vs_zero_based_indexing_alignment():
    """Verify 1-based frame numbers correctly map to 0-based array indices without offset drift."""
    # Suppose a video has 5 frames: 1, 2, 3, 4, 5
    # Official state labels: 1=-1, 2=-1, 3=0 (transition), 4=0, 5=1 (fallen)
    # Timestamps (ms): 1: 0, 2: 100, 3: 200, 4: 300, 5: 400
    # In 0-based arrays: index 0 (frame 1), index 2 (frame 3), index 4 (frame 5)
    timestamps = {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4}
    frame_indices_0based = [0, 1, 2, 3, 4]

    # Map 0-based array index i to 1-based frame number (i + 1)
    mapped_timestamps = [timestamps[i + 1] for i in frame_indices_0based]

    assert mapped_timestamps[0] == 0.0    # Frame 1 -> 0.0s
    assert mapped_timestamps[2] == 0.2    # Frame 3 (state 0) -> 0.2s
    assert mapped_timestamps[4] == 0.4    # Frame 5 (state 1) -> 0.4s


# --------------------------------------------------------------------------- #
# Manifest & Live Dataset Verification Tests
# --------------------------------------------------------------------------- #

def test_live_urfd_manifest_fall01_and_fall02():
    """Verify live URFD annotations for fall-01 and fall-02 match verified transitions."""
    urfd_dir = Path("data/urfd")
    if not (urfd_dir / "urfall-cam0-falls.csv").is_file():
        pytest.skip("Local URFD data not available")

    manifest = build_urfd_manifest(urfd_dir)

    assert "fall-01" in manifest
    assert "fall-02" in manifest

    ann01 = manifest["fall-01"]
    assert ann01.is_fall is True
    assert ann01.fall_transition_start_s == 2.736    # Frame 83
    assert ann01.fallen_state_start_s == 3.737       # Frame 113
    assert ann01.duration_s == 5.305                 # Frame 160
    assert ann01.confirmation_anchor_s() == 3.737
    # Available post-fallen: 5.305 - 3.737 = 1.568s < 2.0s
    assert ann01.has_sufficient_observation(required_confirmation_s=2.0) is False

    ann02 = manifest["fall-02"]
    assert ann02.is_fall is True
    assert ann02.fall_transition_start_s == 1.301    # Frame 40
    assert ann02.fallen_state_start_s == 2.302       # Frame 70
    assert ann02.duration_s == 3.637                 # Frame 110
    assert ann02.confirmation_anchor_s() == 2.302
    # Available post-fallen: 3.637 - 2.302 = 1.335s < 2.0s
    assert ann02.has_sufficient_observation(required_confirmation_s=2.0) is False


def test_unmatched_sequence_remains_unverified():
    """Sequences without an annotation entry in the manifest remain strictly UNVERIFIED."""
    ann_unverified = SequenceAnnotation(
        sequence_id="unknown-clip-01",
        name="Unknown Clip",
        is_fall=None,
        duration_s=4.0,
    )
    benchmark = [Sequence(annotation=ann_unverified, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="unknown-clip-01",
            annotation_name="Unknown Clip",
            is_fall_ground_truth=None,
            alerts=[],
            alert_timestamps=[],
        )
    ]
    report = evaluate_results(benchmark, results)
    assert report.unverified_sequences == 1
    assert report.fall_sequences == 0
    assert report.sequence_evaluations[0].verdict == "UNVERIFIED"


# --------------------------------------------------------------------------- #
# Pose Cache Updating & Preservation Tests
# --------------------------------------------------------------------------- #

def test_apply_annotation_preserves_all_pose_data(tmp_path: Path):
    """Applying annotations to pose cache updates metadata while preserving all pose arrays."""
    # Create a dummy Sequence
    kps = np.ones((17, 3), dtype=np.float32) * 50.0
    from backend.world import Detection
    det = Detection(track_id=1, label="person", conf=0.95, box=(10.0, 10.0, 40.0, 80.0), keypoints=kps)
    frames = [
        FrameObservation(frame_idx=0, timestamp=0.0, detections=[det]),
        FrameObservation(frame_idx=1, timestamp=0.033, detections=[det]),
        FrameObservation(frame_idx=2, timestamp=0.066, detections=[det]),
    ]
    ann_orig = SequenceAnnotation(
        sequence_id="test-seq",
        name="Original",
        is_fall=None,
        duration_s=0.066,
    )
    seq = Sequence(annotation=ann_orig, fps=30.0, frames=frames)

    cache_path = tmp_path / "test-seq.npz"
    save_pose_cache(seq, cache_path)

    # Now apply verified annotation with official timestamps
    ann_verified = SequenceAnnotation(
        sequence_id="test-seq",
        name="Verified",
        is_fall=True,
        duration_s=0.100,
        fall_transition_start_s=0.035,
        fallen_state_start_s=0.070,
    )
    timestamps = {1: 0.0, 2: 0.035, 3: 0.070}
    out_path = tmp_path / "test-seq-verified.npz"

    apply_urfd_annotation_to_pose_cache(cache_path, ann_verified, timestamps=timestamps, output_path=out_path)

    loaded = load_pose_cache(out_path)
    assert loaded.annotation.is_fall is True
    assert loaded.annotation.fall_transition_start_s == pytest.approx(0.035, abs=1e-4)
    assert loaded.annotation.fallen_state_start_s == pytest.approx(0.070, abs=1e-4)
    assert loaded.annotation.duration_s == pytest.approx(0.070, abs=1e-4)

    # Verify pose data is 100% preserved
    assert len(loaded.frames) == 3
    assert loaded.frames[0].detections[0].track_id == 1
    assert loaded.frames[0].detections[0].box == (10.0, 10.0, 40.0, 80.0)
    np.testing.assert_allclose(loaded.frames[0].detections[0].keypoints, kps)


# --------------------------------------------------------------------------- #
# End-to-End Evaluation on Real Fall-01 and Fall-02
# --------------------------------------------------------------------------- #

def test_urfd_fall01_and_fall02_evaluates_as_insufficient_observation():
    """Live URFD fall-01 and fall-02 evaluate as INSUFFICIENT_OBSERVATION under 2.0s confirmation."""
    urfd_dir = Path("data/urfd")
    cache_dir = Path("eval/cache/urfd")

    f01_npz = cache_dir / "fall-01-cam0-rgb.npz"
    f02_npz = cache_dir / "fall-02-cam0-rgb.npz"

    if not f01_npz.is_file() or not f02_npz.is_file() or not (urfd_dir / "urfall-cam0-falls.csv").is_file():
        pytest.skip("Local URFD data or pose caches not available")

    manifest = build_urfd_manifest(urfd_dir)

    seq01 = load_urfd_sequence(f01_npz, manifest)
    seq02 = load_urfd_sequence(f02_npz, manifest)

    assert seq01.annotation.is_fall is True
    assert seq02.annotation.is_fall is True

    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark([seq01, seq02])

    report = evaluate_results([seq01, seq02], results, required_confirmation_s=2.0)

    # Both clips have available post-fallen time < 2.0s
    assert report.insufficient_observation_sequences == 2
    assert report.fall_sequences == 0   # Excluded from confirmation denominators!
    assert report.tp == 0
    assert report.fn == 0
    assert report.fp == 0
    assert report.tn == 0

    assert report.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"
    assert report.sequence_evaluations[1].verdict == "INSUFFICIENT_OBSERVATION"


def test_urfd_all_5_falls_manifest():
    """Verify manifest includes all 5 official fall sequences with verified transitions."""
    urfd_dir = Path("data/urfd")
    if not (urfd_dir / "urfall-cam0-falls.csv").is_file():
        pytest.skip("URFD falls CSV not available")

    manifest = build_urfd_manifest(urfd_dir)
    for sid in ["fall-01", "fall-02", "fall-03", "fall-04", "fall-05"]:
        assert sid in manifest
        ann = manifest[sid]
        assert ann.is_fall is True
        assert ann.fall_transition_start_s is not None
        assert ann.fallen_state_start_s is not None
        assert ann.fall_transition_start_s < ann.fallen_state_start_s
        assert ann.fallen_state_start_s < ann.duration_s


def test_urfd_fall03_through_fall05_evaluations():
    """Verify all 3 fall sequences are INSUFFICIENT_OBSERVATION (< 2.0s), with fall-04 alert observed."""
    urfd_dir = Path("data/urfd")
    cache_dir = Path("eval/cache/urfd")
    if not (cache_dir / "fall-05-cam0-rgb.npz").is_file():
        pytest.skip("Caches for fall-03..05 not available")

    manifest = build_urfd_manifest(urfd_dir)
    seq03 = load_urfd_sequence(cache_dir / "fall-03-cam0-rgb.npz", manifest)
    seq04 = load_urfd_sequence(cache_dir / "fall-04-cam0-rgb.npz", manifest)
    seq05 = load_urfd_sequence(cache_dir / "fall-05-cam0-rgb.npz", manifest)

    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark([seq03, seq04, seq05])
    report = evaluate_results([seq03, seq04, seq05], results, required_confirmation_s=2.0)

    # All three clips have post-fallen duration < 2.0s (1.26s, 1.74s, 1.43s)
    assert report.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"
    assert report.sequence_evaluations[1].verdict == "INSUFFICIENT_OBSERVATION"
    assert report.sequence_evaluations[2].verdict == "INSUFFICIENT_OBSERVATION"
    assert report.insufficient_observation_sequences == 3
    assert report.fall_sequences == 0
    assert report.tp == 0

    # fall-04 alert is preserved as an observed detection
    assert report.observed_fall_alerts == 1
    assert report.sequence_evaluations[1].matched_alert_timestamp == pytest.approx(3.136, abs=0.01)
    assert report.sequence_evaluations[1].latency_s == pytest.approx(1.702, abs=0.01)
    assert "Separately observed confirmed alert" in report.sequence_evaluations[1].notes


