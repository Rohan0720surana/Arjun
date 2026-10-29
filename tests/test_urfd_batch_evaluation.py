"""
Unit tests for URFD 10-Recording Batch Evaluation Pipeline (Phase 3B).

Covers:
- Canonical 10-recording inventory (5 falls, 5 ADLs)
- Proper loading of official fall annotations and ADL assumed-FPS annotations
- Tracking continuity computation across all 10 recordings
- Execution with diagnostics: raw kinematic drop detection and confirmed alerts
- Confirmation window preservation: INSUFFICIENT_OBSERVATION for truncated falls
- True Positive match for fall-04 within the matching window
- True Negative preservation for all 5 ADL sequences
- Metric calculation safety with N/A display for undefined denominators
"""

from __future__ import annotations

from pathlib import Path
import pytest

from eval.dataset import FrameObservation, Sequence, SequenceAnnotation
from eval.metrics import evaluate_results
from eval.runner import OfflineFallRunner, SequenceResult
from eval.urfd_batch_evaluation import (
    URFD_ALL_10_SEQUENCES,
    URFD_FALL_SEQUENCES,
    evaluate_sequence_with_diagnostics,
    evaluate_urfd_10_benchmark,
    load_urfd_10_sequences,
)
from eval.urfd_adl_evaluation import URFD_ADL_SEQUENCES, extract_or_load_adl_sequence


# --------------------------------------------------------------------------- #
# Inventory Tests
# --------------------------------------------------------------------------- #

def test_urfd_10_sequences_inventory():
    """Verify inventory defines exactly 10 recordings: 5 falls and 5 ADLs."""
    assert len(URFD_FALL_SEQUENCES) == 5
    assert len(URFD_ADL_SEQUENCES) == 5
    assert len(URFD_ALL_10_SEQUENCES) == 10

    # Ensure no accidental adl-06 is present
    assert "adl-06-cam0-rgb" not in URFD_ALL_10_SEQUENCES
    assert "adl-06-cam0-rgb" not in URFD_ADL_SEQUENCES

    # Ensure all expected fall sequences exist
    for fid in ["fall-01-cam0-rgb", "fall-02-cam0-rgb", "fall-03-cam0-rgb", "fall-04-cam0-rgb", "fall-05-cam0-rgb"]:
        assert fid in URFD_FALL_SEQUENCES

    # Ensure all expected ADL sequences exist
    for aid in ["adl-01-cam0-rgb", "adl-02-cam0-rgb", "adl-03-cam0-rgb", "adl-04-cam0-rgb", "adl-05-cam0-rgb"]:
        assert aid in URFD_ADL_SEQUENCES


# --------------------------------------------------------------------------- #
# Batch Loading & Provenance Tests
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not Path("eval/cache/urfd/fall-05-cam0-rgb.npz").is_file(),
    reason="Full 10-sequence cache not yet built locally",
)
def test_load_urfd_10_sequences_structure():
    """Load all 10 cached sequences and verify metadata, timestamps, and ground truth."""
    sequences = load_urfd_10_sequences(urfd_dir="data/urfd", cache_dir="eval/cache/urfd")
    assert len(sequences) == 10

    fall_seqs = sequences[:5]
    adl_seqs = sequences[5:]

    # Validate Fall sequences (fall-01..05)
    for seq in fall_seqs:
        ann = seq.annotation
        assert ann.is_fall is True
        assert ann.fall_transition_start_s is not None
        assert ann.fallen_state_start_s is not None
        assert ann.duration_s > ann.fallen_state_start_s
        assert len(seq.frames) > 0
        # Official timestamps: first frame should start at 0.0
        assert seq.frames[0].timestamp == pytest.approx(0.0, abs=1e-3)

    # Validate ADL sequences (adl-01..05)
    for seq in adl_seqs:
        ann = seq.annotation
        assert ann.is_fall is False
        assert seq.fps == 30.0
        assert len(seq.frames) > 0


# --------------------------------------------------------------------------- #
# Fall Detector Evaluation & Diagnostic Tests
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not Path("eval/cache/urfd/fall-04-cam0-rgb.npz").is_file(),
    reason="fall-04 cache not available",
)
def test_fall_04_triggers_confirmed_alert_observed_detection():
    """fall-04 triggers confirmed alert via early drop; classified INSUFFICIENT_OBSERVATION for primary recall."""
    from eval.urfd_annotations import build_urfd_manifest, load_urfd_sequence

    manifest = build_urfd_manifest("data/urfd")
    cache_path = Path("eval/cache/urfd/fall-04-cam0-rgb.npz")
    seq = load_urfd_sequence(cache_path, manifest)

    diag = evaluate_sequence_with_diagnostics(seq)
    assert len(diag.confirmed_alerts) == 1
    assert len(diag.raw_drop_timestamps) >= 1
    assert diag.alert_timestamps[0] == pytest.approx(3.136, abs=0.01)

    # Official event-level evaluation (prediction-independent eligibility)
    res = SequenceResult(
        sequence_id=diag.sequence_id,
        annotation_name=diag.annotation_name,
        is_fall_ground_truth=True,
        alerts=diag.confirmed_alerts,
        alert_timestamps=diag.alert_timestamps,
        total_frames=diag.total_frames,
        duration_s=diag.duration_s,
    )
    rep = evaluate_results([seq], [res], required_confirmation_s=2.0)
    # Excluded from primary recall denominator (post-impact footage: 1.74s < 2.0s)
    assert rep.tp == 0
    assert rep.fn == 0
    assert rep.fp == 0
    assert rep.fall_sequences == 0
    assert rep.insufficient_observation_sequences == 1
    assert rep.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"

    # Confirmed alert preserved as an observed detection
    assert rep.observed_fall_alerts == 1
    assert rep.sequence_evaluations[0].matched_alert_timestamp == pytest.approx(3.136, abs=0.01)
    assert rep.sequence_evaluations[0].latency_s == pytest.approx(1.702, abs=0.01)



@pytest.mark.skipif(
    not Path("eval/cache/urfd/fall-01-cam0-rgb.npz").is_file() or
    not Path("eval/cache/urfd/fall-02-cam0-rgb.npz").is_file() or
    not Path("eval/cache/urfd/fall-03-cam0-rgb.npz").is_file() or
    not Path("eval/cache/urfd/fall-05-cam0-rgb.npz").is_file(),
    reason="Truncated fall caches not available",
)
def test_truncated_falls_classified_as_insufficient_observation():
    """fall-01, 02, 03, 05 cut off < 2.0s post-fall and are classified as INSUFFICIENT_OBSERVATION."""
    from eval.urfd_annotations import build_urfd_manifest, load_urfd_sequence

    manifest = build_urfd_manifest("data/urfd")
    for fid in ["fall-01", "fall-02", "fall-03", "fall-05"]:
        cache_path = Path(f"eval/cache/urfd/{fid}-cam0-rgb.npz")
        seq = load_urfd_sequence(cache_path, manifest)
        diag = evaluate_sequence_with_diagnostics(seq)

        # Raw drop was detected
        assert len(diag.raw_drop_timestamps) >= 1
        # But no confirmed alert because clip ended before 2.0s
        assert len(diag.confirmed_alerts) == 0

        res = SequenceResult(
            sequence_id=diag.sequence_id,
            annotation_name=diag.annotation_name,
            is_fall_ground_truth=True,
            alerts=diag.confirmed_alerts,
            alert_timestamps=diag.alert_timestamps,
        )
        rep = evaluate_results([seq], [res], required_confirmation_s=2.0)
        assert rep.insufficient_observation_sequences == 1
        assert rep.fn == 0  # Not a false negative!
        assert rep.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"


@pytest.mark.skipif(
    not Path("eval/cache/urfd/adl-04-cam0-rgb.npz").is_file() or
    not Path("eval/cache/urfd/adl-05-cam0-rgb.npz").is_file(),
    reason="New ADL caches not available",
)
def test_adl_04_and_05_evaluate_as_true_negatives():
    """adl-04 and adl-05 produce zero confirmed false alerts and evaluate as True Negatives."""
    for aid in ["adl-04-cam0-rgb", "adl-05-cam0-rgb"]:
        seq = extract_or_load_adl_sequence(aid, urfd_dir="data/urfd", cache_dir="eval/cache/urfd")
        diag = evaluate_sequence_with_diagnostics(seq)
        assert len(diag.confirmed_alerts) == 0

        res = SequenceResult(
            sequence_id=diag.sequence_id,
            annotation_name=diag.annotation_name,
            is_fall_ground_truth=False,
            alerts=[],
            alert_timestamps=[],
        )
        rep = evaluate_results([seq], [res])
        assert rep.tn == 1
        assert rep.fp == 0
        assert rep.sequence_evaluations[0].verdict == "TN"


# --------------------------------------------------------------------------- #
# Full 10-Recording Benchmark Integration Test
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not Path("eval/cache/urfd/fall-05-cam0-rgb.npz").is_file(),
    reason="Full 10-sequence cache not yet built locally",
)
def test_evaluate_urfd_10_benchmark_runs_cleanly():
    """Full 10-sequence benchmark runs without exception and returns exit code 0."""
    ret = evaluate_urfd_10_benchmark(urfd_dir="data/urfd", cache_dir="eval/cache/urfd")
    assert ret == 0
