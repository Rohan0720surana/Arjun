"""
Tests for URFD ADL false-alert baseline evaluation pipeline (Phase 3A).

Covers:
- make_adl_annotation: correct is_fall, duration, provenance note.
- resolve_frames_dir: nested and direct directory resolution.
- extract_or_load_adl_sequence: cache-hit path (no extraction).
- sequence_summary: frame counts, coverage, track IDs, alert counts.
- evaluate_urfd_adl_baseline: integration with mocked sequences.
- Metric correctness: TN for zero alerts, FP for false alarms.
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
from eval.urfd_adl_evaluation import (
    EXPLORATORY_SAMPLE_WARNING,
    PROVENANCE_NOTE,
    TIMESTAMP_DISCLAIMER,
    URFD_ADL_SEQUENCES,
    compute_tracking_continuity,
    extract_or_load_adl_sequence,
    make_adl_annotation,
    resolve_frames_dir,
    sequence_summary,
)
from backend.world import Detection


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _dummy_detection(track_id: int = 1) -> Detection:
    """Minimal Detection for use in test frames."""
    return Detection(
        track_id=track_id,
        label="person",
        conf=0.90,
        box=(100.0, 50.0, 200.0, 300.0),
        keypoints=None,
    )


def _adl_sequence(
    seq_id: str = "adl-01-cam0-rgb",
    num_frames: int = 10,
    fps: float = 30.0,
    with_person: bool = True,
    is_fall: bool | None = False,
) -> Sequence:
    """Build a minimal Sequence for testing."""
    duration_s = round((num_frames - 1) / fps, 4)
    ann = SequenceAnnotation(
        sequence_id=seq_id,
        name=f"URFD {seq_id}",
        is_fall=is_fall,
        duration_s=duration_s,
    )
    frames = [
        FrameObservation(
            frame_idx=i,
            timestamp=round(i / fps, 4),
            detections=[_dummy_detection()] if with_person else [],
        )
        for i in range(num_frames)
    ]
    return Sequence(annotation=ann, fps=fps, frames=frames)


def _mock_result(
    seq_id: str,
    alert_timestamps: list[float] | None = None,
    is_fall_gt: bool | None = False,
) -> SequenceResult:
    """Build a mock SequenceResult."""
    timestamps = alert_timestamps or []
    alerts = [
        Candidate(type="fall", message="fall", track_id=1, highlight="critical")
        for _ in timestamps
    ]
    return SequenceResult(
        sequence_id=seq_id,
        annotation_name=f"URFD {seq_id}",
        is_fall_ground_truth=is_fall_gt,
        alerts=alerts,
        alert_timestamps=timestamps,
    )


# --------------------------------------------------------------------------- #
# make_adl_annotation
# --------------------------------------------------------------------------- #

def test_make_adl_annotation_is_fall_false():
    """ADL annotation always has is_fall=False."""
    ann = make_adl_annotation("adl-01-cam0-rgb", num_frames=150)
    assert ann.is_fall is False


def test_make_adl_annotation_duration():
    """Duration is (num_frames - 1) / assumed_fps."""
    ann = make_adl_annotation("adl-01-cam0-rgb", num_frames=150, assumed_fps=30.0)
    expected = round(149 / 30.0, 4)
    assert ann.duration_s == pytest.approx(expected, abs=1e-4)


def test_make_adl_annotation_duration_single_frame():
    """Single-frame sequence has duration 0.0."""
    ann = make_adl_annotation("adl-01-cam0-rgb", num_frames=1)
    assert ann.duration_s == 0.0


def test_make_adl_annotation_provenance_in_description():
    """Annotation description embeds both provenance note and timestamp disclaimer."""
    ann = make_adl_annotation("adl-01-cam0-rgb", num_frames=150)
    assert PROVENANCE_NOTE in ann.description
    assert TIMESTAMP_DISCLAIMER in ann.description


def test_make_adl_annotation_sequence_id_preserved():
    """Sequence ID is preserved verbatim."""
    seq_id = "adl-02-cam0-rgb"
    ann = make_adl_annotation(seq_id, num_frames=180)
    assert ann.sequence_id == seq_id


def test_make_adl_annotation_custom_fps():
    """Custom FPS affects duration computation."""
    ann_30 = make_adl_annotation("adl-01-cam0-rgb", num_frames=150, assumed_fps=30.0)
    ann_25 = make_adl_annotation("adl-01-cam0-rgb", num_frames=150, assumed_fps=25.0)
    assert ann_30.duration_s != ann_25.duration_s
    assert ann_25.duration_s == pytest.approx(149 / 25.0, abs=1e-4)


def test_make_adl_annotation_confirmation_anchor_is_none():
    """ADL annotations have no fall anchor (confirmation_anchor_s returns None)."""
    ann = make_adl_annotation("adl-01-cam0-rgb", num_frames=150)
    assert ann.confirmation_anchor_s() is None


# --------------------------------------------------------------------------- #
# resolve_frames_dir
# --------------------------------------------------------------------------- #

def test_resolve_frames_dir_nested(tmp_path: Path):
    """Nested layout <urfd>/<seq_id>/<seq_id>/ is resolved correctly."""
    seq_id = "adl-01-cam0-rgb"
    nested = tmp_path / seq_id / seq_id
    nested.mkdir(parents=True)
    result = resolve_frames_dir(tmp_path, seq_id)
    assert result == nested


def test_resolve_frames_dir_direct_fallback(tmp_path: Path):
    """Direct layout <urfd>/<seq_id>/ is used as fallback when nested is absent."""
    seq_id = "adl-02-cam0-rgb"
    direct = tmp_path / seq_id
    direct.mkdir(parents=True)
    result = resolve_frames_dir(tmp_path, seq_id)
    assert result == direct


def test_resolve_frames_dir_missing_raises(tmp_path: Path):
    """Missing directory raises NotADirectoryError."""
    with pytest.raises(NotADirectoryError, match="Cannot find frame directory"):
        resolve_frames_dir(tmp_path, "adl-99-cam0-rgb")


# --------------------------------------------------------------------------- #
# sequence_summary
# --------------------------------------------------------------------------- #

def test_sequence_summary_full_coverage():
    """All frames have a person => 100% coverage."""
    seq = _adl_sequence(num_frames=10, with_person=True)
    result = _mock_result("adl-01-cam0-rgb", alert_timestamps=[])
    s = sequence_summary(seq, result)
    assert s["frames"] == 10
    assert s["frames_with_person"] == 10
    assert s["detection_coverage_pct"] == 100.0
    assert s["confirmed_alerts"] == 0


def test_sequence_summary_zero_coverage():
    """No person in any frame => 0% coverage."""
    seq = _adl_sequence(num_frames=10, with_person=False)
    result = _mock_result("adl-01-cam0-rgb", alert_timestamps=[])
    s = sequence_summary(seq, result)
    assert s["frames_with_person"] == 0
    assert s["detection_coverage_pct"] == 0.0


def test_sequence_summary_alert_count():
    """Alert timestamps are reflected in confirmed_alerts."""
    seq = _adl_sequence()
    result = _mock_result("adl-01-cam0-rgb", alert_timestamps=[3.0, 4.5])
    s = sequence_summary(seq, result)
    assert s["confirmed_alerts"] == 2
    assert s["alert_timestamps"] == [3.0, 4.5]


def test_sequence_summary_unique_tracks():
    """Multiple detections with the same track ID are deduplicated."""
    seq_id = "adl-01-cam0-rgb"
    ann = SequenceAnnotation(seq_id, "URFD adl-01-cam0-rgb", is_fall=False, duration_s=1.0)
    frames = [
        FrameObservation(
            frame_idx=i,
            timestamp=i / 30.0,
            detections=[_dummy_detection(track_id=1), _dummy_detection(track_id=2)],
        )
        for i in range(5)
    ]
    seq = Sequence(annotation=ann, fps=30.0, frames=frames)
    result = _mock_result(seq_id, alert_timestamps=[])
    s = sequence_summary(seq, result)
    assert s["num_tracks"] == 2
    assert set(s["unique_track_ids"]) == {1, 2}


# --------------------------------------------------------------------------- #
# extract_or_load_adl_sequence -- cache-hit path
# --------------------------------------------------------------------------- #

def test_extract_or_load_uses_cache(tmp_path: Path):
    """When NPZ cache exists, no extraction is run and annotation is refreshed."""
    from eval.urfd_adl_evaluation import extract_or_load_adl_sequence

    seq_id = "adl-01-cam0-rgb"
    # Build a minimal Sequence and save as cache
    kps = np.ones((17, 3), dtype=np.float32) * 50.0
    det = Detection(track_id=1, label="person", conf=0.9, box=(10.0, 10.0, 40.0, 80.0), keypoints=kps)
    frames_list = [
        FrameObservation(frame_idx=i, timestamp=i / 30.0, detections=[det])
        for i in range(5)
    ]
    orig_ann = SequenceAnnotation(
        sequence_id=seq_id, name="Old Name", is_fall=None, duration_s=4.0 / 30.0
    )
    orig_seq = Sequence(annotation=orig_ann, fps=30.0, frames=frames_list)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    save_pose_cache(orig_seq, cache_dir / f"{seq_id}.npz")

    # Load via our helper -- should get is_fall=False and fresh annotation
    loaded = extract_or_load_adl_sequence(
        seq_id,
        urfd_dir=tmp_path / "urfd",  # doesn't need to exist; cache is used
        cache_dir=cache_dir,
        assumed_fps=30.0,
    )

    assert loaded.annotation.is_fall is False
    assert loaded.annotation.sequence_id == seq_id
    assert len(loaded.frames) == 5
    # Provenance note must be in description
    assert PROVENANCE_NOTE in loaded.annotation.description


# --------------------------------------------------------------------------- #
# Metric Integration: ADL with zero alerts -> TN
# --------------------------------------------------------------------------- #

def test_adl_zero_alerts_is_tn():
    """ADL sequence with no alerts is counted as TN in evaluate_results."""
    ann = SequenceAnnotation(
        sequence_id="adl-01-cam0-rgb",
        name="URFD adl-01-cam0-rgb",
        is_fall=False,
        duration_s=5.0,
    )
    seq = Sequence(annotation=ann, fps=30.0, frames=[])
    result = SequenceResult(
        sequence_id="adl-01-cam0-rgb",
        annotation_name="URFD adl-01-cam0-rgb",
        is_fall_ground_truth=False,
        alerts=[],
        alert_timestamps=[],
    )
    report = evaluate_results([seq], [result])
    assert report.tn == 1
    assert report.fp == 0
    assert report.adl_sequences == 1


def test_adl_with_alert_is_fp():
    """ADL sequence with a confirmed alert is counted as FP in evaluate_results."""
    ann = SequenceAnnotation(
        sequence_id="adl-02-cam0-rgb",
        name="URFD adl-02-cam0-rgb",
        is_fall=False,
        duration_s=6.0,
    )
    seq = Sequence(annotation=ann, fps=30.0, frames=[])
    alert = Candidate(type="fall", message="false alarm", track_id=1, highlight="critical")
    result = SequenceResult(
        sequence_id="adl-02-cam0-rgb",
        annotation_name="URFD adl-02-cam0-rgb",
        is_fall_ground_truth=False,
        alerts=[alert],
        alert_timestamps=[2.5],
    )
    report = evaluate_results([seq], [result])
    assert report.fp == 1
    assert report.tn == 0


def test_mixed_adl_specificity():
    """2 TN + 1 FP gives specificity = 2/3 = 66.7%."""
    adls = [
        SequenceAnnotation(f"adl-0{i}-cam0-rgb", f"ADL {i}", is_fall=False, duration_s=5.0)
        for i in range(1, 4)
    ]
    seqs = [Sequence(annotation=a, fps=30.0, frames=[]) for a in adls]
    alert = Candidate(type="fall", message="fp", track_id=1, highlight="critical")
    results = [
        SequenceResult("adl-01-cam0-rgb", "ADL 1", False, [], []),
        SequenceResult("adl-02-cam0-rgb", "ADL 2", False, [alert], [3.0]),
        SequenceResult("adl-03-cam0-rgb", "ADL 3", False, [], []),
    ]
    report = evaluate_results(seqs, results)
    assert report.tn == 2
    assert report.fp == 1
    assert report.specificity == pytest.approx(2 / 3, abs=1e-6)


# --------------------------------------------------------------------------- #
# URFD_ADL_SEQUENCES inventory
# --------------------------------------------------------------------------- #

def test_urfd_adl_sequences_inventory():
    """The ADL sequence inventory contains exactly the five expected entries (adl-01..05)."""
    assert len(URFD_ADL_SEQUENCES) == 5
    assert "adl-01-cam0-rgb" in URFD_ADL_SEQUENCES
    assert "adl-02-cam0-rgb" in URFD_ADL_SEQUENCES
    assert "adl-03-cam0-rgb" in URFD_ADL_SEQUENCES
    assert "adl-04-cam0-rgb" in URFD_ADL_SEQUENCES
    assert "adl-05-cam0-rgb" in URFD_ADL_SEQUENCES
    assert URFD_ADL_SEQUENCES["adl-01-cam0-rgb"]["expected_frames"] == 150
    assert URFD_ADL_SEQUENCES["adl-02-cam0-rgb"]["expected_frames"] == 180
    assert URFD_ADL_SEQUENCES["adl-03-cam0-rgb"]["expected_frames"] == 180
    assert URFD_ADL_SEQUENCES["adl-04-cam0-rgb"]["expected_frames"] == 150
    assert URFD_ADL_SEQUENCES["adl-05-cam0-rgb"]["expected_frames"] == 180


# --------------------------------------------------------------------------- #
# Live integration (skipped when data unavailable)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not Path("data/urfd/adl-01-cam0-rgb").is_dir(),
    reason="Live URFD ADL data not available",
)
def test_live_adl_extraction_integration():
    """Integration: extract adl-01, verify frame count, duration, and no fall alert."""
    import tempfile
    from eval.urfd_adl_evaluation import extract_or_load_adl_sequence

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir)
        seq = extract_or_load_adl_sequence(
            "adl-01-cam0-rgb",
            urfd_dir=Path("data/urfd"),
            cache_dir=cache_path,
            assumed_fps=30.0,
        )

    assert seq.annotation.is_fall is False
    # Exact expected frame count for adl-01
    assert len(seq.frames) == 150
    expected_duration = round(149 / 30.0, 4)
    assert seq.annotation.duration_s == pytest.approx(expected_duration, abs=1e-3)


# --------------------------------------------------------------------------- #
# Cache FPS Validation & Provenance Tests
# --------------------------------------------------------------------------- #

def test_extract_or_load_fps_mismatch_raises_when_no_source(tmp_path: Path):
    """Loading a cache extracted at a different FPS raises ValueError when raw frames are missing."""
    seq_id = "adl-01-cam0-rgb"
    kps = np.ones((17, 3), dtype=np.float32) * 50.0
    det = Detection(track_id=1, label="person", conf=0.9, box=(10.0, 10.0, 40.0, 80.0), keypoints=kps)
    frames_list = [
        FrameObservation(frame_idx=i, timestamp=round(i / 15.0, 4), detections=[det])
        for i in range(5)
    ]
    orig_ann = SequenceAnnotation(sequence_id=seq_id, name="Old", is_fall=False, duration_s=4.0 / 15.0)
    orig_seq = Sequence(annotation=orig_ann, fps=15.0, frames=frames_list)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    save_pose_cache(orig_seq, cache_dir / f"{seq_id}.npz")

    # Request 30.0 FPS with no raw frames available -> must raise ValueError
    with pytest.raises(ValueError, match="does not match requested FPS"):
        extract_or_load_adl_sequence(
            seq_id,
            urfd_dir=tmp_path / "urfd",  # frames don't exist here
            cache_dir=cache_dir,
            assumed_fps=30.0,
            allow_retime=False,
        )


def test_extract_or_load_fps_mismatch_safe_retime(tmp_path: Path):
    """When allow_retime=True, mismatched cache is safely retimed with documented provenance."""
    seq_id = "adl-01-cam0-rgb"
    kps = np.ones((17, 3), dtype=np.float32) * 50.0
    det = Detection(track_id=1, label="person", conf=0.9, box=(10.0, 10.0, 40.0, 80.0), keypoints=kps)
    frames_list = [
        FrameObservation(frame_idx=i, timestamp=round(i / 15.0, 4), detections=[det])
        for i in range(5)
    ]
    orig_ann = SequenceAnnotation(sequence_id=seq_id, name="Old", is_fall=False, duration_s=4.0 / 15.0)
    orig_seq = Sequence(annotation=orig_ann, fps=15.0, frames=frames_list)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    save_pose_cache(orig_seq, cache_dir / f"{seq_id}.npz")

    retimed_seq = extract_or_load_adl_sequence(
        seq_id,
        urfd_dir=tmp_path / "urfd",
        cache_dir=cache_dir,
        assumed_fps=30.0,
        allow_retime=True,
    )

    assert retimed_seq.fps == 30.0
    assert len(retimed_seq.frames) == 5
    assert retimed_seq.annotation.duration_s == pytest.approx(4.0 / 30.0, abs=1e-4)
    # Check that frame timestamps were updated to 30 FPS
    assert retimed_seq.frames[1].timestamp == pytest.approx(1.0 / 30.0, abs=1e-4)
    assert retimed_seq.frames[4].timestamp == pytest.approx(4.0 / 30.0, abs=1e-4)
    # Explicit retiming documentation present
    assert "Retimed from cached 15.0 FPS" in retimed_seq.annotation.description


def test_extract_or_load_timestamp_monotonicity_validation(tmp_path: Path):
    """Non-monotonic timestamps in cache raise ValueError."""
    seq_id = "adl-01-cam0-rgb"
    kps = np.ones((17, 3), dtype=np.float32) * 50.0
    det = Detection(track_id=1, label="person", conf=0.9, box=(10.0, 10.0, 40.0, 80.0), keypoints=kps)
    # Intentionally corrupt timestamps: 0.0, 0.2, 0.1 (non-monotonic)
    frames_list = [
        FrameObservation(frame_idx=0, timestamp=0.0, detections=[det]),
        FrameObservation(frame_idx=1, timestamp=0.2, detections=[det]),
        FrameObservation(frame_idx=2, timestamp=0.1, detections=[det]),
    ]
    orig_ann = SequenceAnnotation(sequence_id=seq_id, name="Old", is_fall=False, duration_s=0.2)
    orig_seq = Sequence(annotation=orig_ann, fps=30.0, frames=frames_list)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    save_pose_cache(orig_seq, cache_dir / f"{seq_id}.npz")

    with pytest.raises(ValueError, match="not monotonically increasing"):
        extract_or_load_adl_sequence(
            seq_id,
            urfd_dir=tmp_path / "urfd",
            cache_dir=cache_dir,
            assumed_fps=30.0,
        )


# --------------------------------------------------------------------------- #
# Tracking Continuity & Missing Detection Tests
# --------------------------------------------------------------------------- #

def test_tracking_continuity_perfect_tracking():
    """Unbroken sequence with 1 track ID reports zero gap and full continuous segment."""
    frames = [
        FrameObservation(
            frame_idx=i,
            timestamp=round(i / 30.0, 4),
            detections=[_dummy_detection(track_id=1)],
        )
        for i in range(15)
    ]
    cont = compute_tracking_continuity(frames, fps=30.0)
    assert cont["frames_with_person"] == 15
    assert cont["detection_coverage_pct"] == 100.0
    assert cont["longest_consecutive_detection_gap"] == 0
    assert cont["longest_detection_gap_s"] == 0.0
    assert cont["longest_continuous_track_segment"] == 15
    assert cont["longest_track_segment_s"] == round(15 / 30.0, 2)
    assert cont["track_id_switches"] == 0
    assert cont["unique_track_ids"] == [1]
    assert cont["num_tracks"] == 1


def test_tracking_continuity_detection_gaps():
    """Sequence with missing detections correctly reports longest gap and broken segments."""
    # Frames:
    # 0, 1: track 1 (len 2)
    # 2, 3, 4, 5: empty (gap 4)
    # 6, 7, 8: track 1 (len 3)
    # 9: empty (gap 1)
    frames = [
        FrameObservation(frame_idx=0, timestamp=0.00, detections=[_dummy_detection(track_id=1)]),
        FrameObservation(frame_idx=1, timestamp=0.03, detections=[_dummy_detection(track_id=1)]),
        FrameObservation(frame_idx=2, timestamp=0.07, detections=[]),
        FrameObservation(frame_idx=3, timestamp=0.10, detections=[]),
        FrameObservation(frame_idx=4, timestamp=0.13, detections=[]),
        FrameObservation(frame_idx=5, timestamp=0.17, detections=[]),
        FrameObservation(frame_idx=6, timestamp=0.20, detections=[_dummy_detection(track_id=1)]),
        FrameObservation(frame_idx=7, timestamp=0.23, detections=[_dummy_detection(track_id=1)]),
        FrameObservation(frame_idx=8, timestamp=0.27, detections=[_dummy_detection(track_id=1)]),
        FrameObservation(frame_idx=9, timestamp=0.30, detections=[]),
    ]
    cont = compute_tracking_continuity(frames, fps=30.0)
    assert cont["frames_with_person"] == 5
    assert cont["detection_coverage_pct"] == 50.0
    assert cont["longest_consecutive_detection_gap"] == 4
    assert cont["longest_detection_gap_s"] == round(4 / 30.0, 2)
    # Longest continuous segment is 3 (frames 6, 7, 8)
    assert cont["longest_continuous_track_segment"] == 3
    assert cont["longest_track_segment_s"] == round(3 / 30.0, 2)
    # No ID switches (track_id remains 1 across detections)
    assert cont["track_id_switches"] == 0
    assert cont["unique_track_ids"] == [1]


def test_tracking_continuity_track_id_switches():
    """Track ID switches across consecutive frames are measured accurately."""
    # Frames 0..2: track 1
    # Frames 3..5: track 2 (switch 1)
    # Frames 6..8: track 3 (switch 2)
    frames = (
        [FrameObservation(i, i / 30.0, [_dummy_detection(track_id=1)]) for i in range(3)]
        + [FrameObservation(i, i / 30.0, [_dummy_detection(track_id=2)]) for i in range(3, 6)]
        + [FrameObservation(i, i / 30.0, [_dummy_detection(track_id=3)]) for i in range(6, 9)]
    )
    cont = compute_tracking_continuity(frames, fps=30.0)
    assert cont["frames_with_person"] == 9
    assert cont["longest_consecutive_detection_gap"] == 0
    assert cont["longest_continuous_track_segment"] == 3
    assert cont["track_id_switches"] == 2
    assert cont["unique_track_ids"] == [1, 2, 3]
    assert cont["num_tracks"] == 3


def test_tracking_continuity_empty_frames():
    """Empty frames list returns zeroed continuity metrics safely."""
    cont = compute_tracking_continuity([])
    assert cont["frames_with_person"] == 0
    assert cont["detection_coverage_pct"] == 0.0
    assert cont["longest_consecutive_detection_gap"] == 0
    assert cont["longest_continuous_track_segment"] == 0
    assert cont["track_id_switches"] == 0
    assert cont["unique_track_ids"] == []
    assert cont["num_tracks"] == 0


def test_sequence_summary_includes_tracking_continuity_fields():
    """sequence_summary includes all tracking continuity metric keys."""
    seq = _adl_sequence(num_frames=10, with_person=True)
    res = _mock_result("adl-01-cam0-rgb", alert_timestamps=[])
    s = sequence_summary(seq, res)
    assert "longest_consecutive_detection_gap" in s
    assert "longest_detection_gap_s" in s
    assert "longest_continuous_track_segment" in s
    assert "longest_track_segment_s" in s
    assert "track_id_switches" in s
    assert s["longest_consecutive_detection_gap"] == 0
    assert s["longest_continuous_track_segment"] == 10
    assert s["track_id_switches"] == 0


# --------------------------------------------------------------------------- #
# Denominator Protection & Alert Rates Tests
# --------------------------------------------------------------------------- #

def test_unverified_recordings_excluded_from_confusion_denominators():
    """Unverified sequences (is_fall=None) must not enter TN/FP denominators."""
    # 2 verified ADL sequences (TN), 1 unverified sequence (None)
    adls = [
        SequenceAnnotation("adl-01-cam0-rgb", "ADL 1", is_fall=False, duration_s=5.0),
        SequenceAnnotation("adl-02-cam0-rgb", "ADL 2", is_fall=False, duration_s=5.0),
        SequenceAnnotation("unverified-01", "Unverified", is_fall=None, duration_s=5.0),
    ]
    seqs = [Sequence(annotation=a, fps=30.0, frames=[]) for a in adls]
    results = [
        SequenceResult("adl-01-cam0-rgb", "ADL 1", False, [], []),
        SequenceResult("adl-02-cam0-rgb", "ADL 2", False, [], []),
        # Even if unverified sequence produced an alert, it must not become an FP in metrics
        SequenceResult("unverified-01", "Unverified", None, [
            Candidate(type="fall", message="unv alert", track_id=1, highlight="critical")
        ], [2.0]),
    ]
    report = evaluate_results(seqs, results)
    assert report.tn == 2
    assert report.fp == 0
    assert report.adl_sequences == 2
    assert report.unverified_sequences == 1
    # Specificity denominator is strictly tn + fp == 2, giving 100%
    assert report.specificity == 1.0


def test_exploratory_sample_warning_text():
    """Exploratory sample warning contains the expected statistical caveats."""
    assert "small exploratory sample" in EXPLORATORY_SAMPLE_WARNING
    assert "not general real-world specificity" in EXPLORATORY_SAMPLE_WARNING

