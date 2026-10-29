"""
Unit tests for the offline fall-detection evaluation framework.

Tests metric calculations, zero-division cases, event matching windows,
duplicate counting, and benchmark reproducibility.
"""

from __future__ import annotations

import pytest

from backend.detectors.base import Candidate
from eval.dataset import (
    FrameObservation,
    Sequence,
    SequenceAnnotation,
    get_synthetic_benchmark,
)
from eval.metrics import evaluate_results
from eval.runner import OfflineFallRunner, SequenceResult


def _mock_candidate(alert_type: str = "fall", track_id: int = 1) -> Candidate:
    return Candidate(
        type=alert_type,
        message="Fall detected",
        track_id=track_id,
        highlight="critical",
    )


# --------------------------------------------------------------------------- #
# Benchmark Reproducibility
# --------------------------------------------------------------------------- #

def test_benchmark_reproducibility():
    """Generating the benchmark twice produces identical sequences and timestamps."""
    b1 = get_synthetic_benchmark(fps=15.0)
    b2 = get_synthetic_benchmark(fps=15.0)

    assert len(b1) == len(b2) == 10
    for s1, s2 in zip(b1, b2):
        assert s1.annotation.sequence_id == s2.annotation.sequence_id
        assert s1.annotation.is_fall == s2.annotation.is_fall
        assert len(s1.frames) == len(s2.frames)
        for f1, f2 in zip(s1.frames, s2.frames):
            assert f1.frame_idx == f2.frame_idx
            assert f1.timestamp == f2.timestamp
            assert len(f1.detections) == len(f2.detections)
            assert f1.detections[0].box == f2.detections[0].box


def test_runner_determinism():
    """Running the benchmark twice through OfflineFallRunner yields identical results."""
    benchmark = get_synthetic_benchmark(fps=15.0)
    runner1 = OfflineFallRunner()
    runner2 = OfflineFallRunner()

    res1 = runner1.evaluate_benchmark(benchmark)
    res2 = runner2.evaluate_benchmark(benchmark)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.sequence_id == r2.sequence_id
        assert r1.alert_timestamps == r2.alert_timestamps
        assert len(r1.alerts) == len(r2.alerts)


# --------------------------------------------------------------------------- #
# Metric Calculations & Confusion Matrix
# --------------------------------------------------------------------------- #

def test_metrics_perfect_classification():
    """Evaluating the canonical synthetic benchmark yields 100% precision and recall."""
    benchmark = get_synthetic_benchmark(fps=15.0)
    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(benchmark)
    report = evaluate_results(benchmark, results)

    assert report.total_sequences == 10
    assert report.fall_sequences == 4
    assert report.adl_sequences == 6
    assert report.tp == 4
    assert report.tn == 6
    assert report.fp == 0
    assert report.fn == 0
    assert report.duplicate_alerts == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.specificity == 1.0
    assert report.mean_latency_s is not None
    assert 1.8 < report.mean_latency_s < 2.05


def test_metrics_false_positive_and_false_negative():
    """Synthesized results with FP and FN calculate correct precision, recall, and F1."""
    ann_fall = SequenceAnnotation(
        sequence_id="S-FALL", name="Fall", is_fall=True, duration_s=6.0,
        fall_start_s=1.0, fall_impact_s=1.5,
    )
    ann_adl = SequenceAnnotation(
        sequence_id="S-ADL", name="ADL", is_fall=False, duration_s=6.0,
    )
    benchmark = [
        Sequence(annotation=ann_fall, fps=15.0),
        Sequence(annotation=ann_adl, fps=15.0),
    ]

    # Case: Fall missed (FN), and ADL falsely triggered (FP)
    results = [
        SequenceResult(sequence_id="S-FALL", annotation_name="Fall",
                       is_fall_ground_truth=True, alerts=[], alert_timestamps=[]),
        SequenceResult(sequence_id="S-ADL", annotation_name="ADL",
                       is_fall_ground_truth=False, alerts=[_mock_candidate()],
                       alert_timestamps=[2.5]),
    ]

    report = evaluate_results(benchmark, results)
    assert report.tp == 0
    assert report.fn == 1
    assert report.fp == 1
    assert report.tn == 0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0


# --------------------------------------------------------------------------- #
# Event Matching Window & Duplicate Counting
# --------------------------------------------------------------------------- #

def test_event_matching_window_and_duplicates():
    """Alert inside window is TP; additional alerts are counted as duplicates."""
    ann = SequenceAnnotation(
        sequence_id="S-WINDOW", name="Window Test", is_fall=True, duration_s=8.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Default matching window: [impact, impact + 5.0s] = [2.0s, 7.0s]
    benchmark = [Sequence(annotation=ann, fps=15.0)]

    # 2 alerts inside window at t=3.5s and t=4.5s
    results = [
        SequenceResult(
            sequence_id="S-WINDOW", annotation_name="Window Test",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate(), _mock_candidate()],
            alert_timestamps=[3.5, 4.5],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.tp == 1
    assert report.fn == 0
    assert report.fp == 0
    assert report.duplicate_alerts == 1
    assert report.sequence_evaluations[0].latency_s == pytest.approx(1.5, abs=0.01)


def test_alert_outside_window_counts_as_fn_and_fp():
    """An alert firing outside the [impact, impact + 5.0s] window counts as FN and FP."""
    ann = SequenceAnnotation(
        sequence_id="S-LATE", name="Late Alert", is_fall=True, duration_s=10.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Default window: [2.0s, 7.0s]. Alert fires at t=7.5s (after window closes)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-LATE", annotation_name="Late Alert",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[7.5],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.tp == 0
    assert report.fn == 1
    assert report.fp == 1


def test_premature_alert_before_impact_counts_as_fp():
    """An alert firing before impact (t < impact) counts as an unmatched false alarm (FP)."""
    ann = SequenceAnnotation(
        sequence_id="S-EARLY", name="Early Alert", is_fall=True, duration_s=6.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Default window: [2.0s, 7.0s]. Alert fires at t=1.2s before impact.
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-EARLY", annotation_name="Early Alert",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[1.2],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.tp == 0
    assert report.fn == 1
    assert report.fp == 1
    assert "premature alert" in report.sequence_evaluations[0].notes


def test_premature_alert_with_subsequent_in_window_alert():
    """Premature alert is FP, while subsequent in-window alert matches as TP."""
    ann = SequenceAnnotation(
        sequence_id="S-BOTH", name="Early + In-Window", is_fall=True, duration_s=8.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Impact at 2.0s, window [2.0s, 7.0s]
    # Alert at 1.0s (premature -> FP) and 3.0s (in-window -> TP, latency=1.0s)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-BOTH", annotation_name="Early + In-Window",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate(), _mock_candidate()],
            alert_timestamps=[1.0, 3.0],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.tp == 1
    assert report.fp == 1
    assert report.fn == 0
    assert report.sequence_evaluations[0].latency_s == pytest.approx(1.0, abs=0.01)


def test_configurable_window_parameters():
    """Custom matching window parameters (window_after and window_before) are respected."""
    ann = SequenceAnnotation(
        sequence_id="S-CUSTOM", name="Custom Window", is_fall=True, duration_s=6.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Custom window: [impact - 0.5s, impact + 2.0s] = [1.5s, 4.0s]
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-CUSTOM", annotation_name="Custom Window",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[1.8],
        )
    ]

    report = evaluate_results(
        benchmark, results,
        window_after_impact_s=2.0,
        window_before_impact_s=0.5,
    )
    assert report.tp == 1
    assert report.fn == 0
    assert report.fp == 0
    assert "[impact - 0.5s, impact + 2.0s]" in report.matching_window_desc


def test_missing_impact_annotation_fallback():
    """Missing fall_impact_s falls back gracefully to fall_start_s with a documented note."""
    ann = SequenceAnnotation(
        sequence_id="S-NO-IMPACT", name="No Impact", is_fall=True, duration_s=6.0,
        fall_start_s=1.5, fall_impact_s=None,
    )
    # Anchors to fall_start_s=1.5s; window is [1.5s, 6.5s] (clamped to duration 6.0s)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-NO-IMPACT", annotation_name="No Impact",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[2.5],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.tp == 1
    assert "fall_impact_s missing" in report.sequence_evaluations[0].notes


def test_truncated_video_note_on_fn():
    """When video ends before the 5.0s matching window closes, FN note documents truncation."""
    ann = SequenceAnnotation(
        sequence_id="S-SHORT", name="Short Video", is_fall=True, duration_s=4.5,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Nominal window: [2.0s, 7.0s], but video ends at 4.5s (post-impact=2.5s >= 2.0s confirmation)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-SHORT", annotation_name="Short Video",
            is_fall_ground_truth=True,
            alerts=[],
            alert_timestamps=[],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.fn == 1
    assert "Video ended at 4.5s before 5.0s window closed" in report.sequence_evaluations[0].notes


def test_insufficient_observation_when_clip_cuts_off_before_confirmation():
    """Fall clip cutting off before required 2.0s confirmation is INSUFFICIENT_OBSERVATION, not FN."""
    ann = SequenceAnnotation(
        sequence_id="S-CUTOFF", name="Cutoff Clip", is_fall=True, duration_s=3.6,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Post-impact footage: 3.6s - 2.0s = 1.6s (< 2.0s confirmation required)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-CUTOFF", annotation_name="Cutoff Clip",
            is_fall_ground_truth=True,
            alerts=[],
            alert_timestamps=[],
        )
    ]

    report = evaluate_results(benchmark, results, required_confirmation_s=2.0)
    assert report.fn == 0
    assert report.tp == 0
    assert report.insufficient_observation_sequences == 1
    assert report.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"
    assert "shorter than required confirmation time" in report.sequence_evaluations[0].notes


def test_insufficient_observation_excluded_from_recall_denominator():
    """INSUFFICIENT_OBSERVATION sequences are excluded from TP/FN denominators, preserving true recall."""
    ann_adequate = SequenceAnnotation(
        sequence_id="S-ADEQUATE", name="Adequate Fall", is_fall=True, duration_s=6.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    ann_insufficient = SequenceAnnotation(
        sequence_id="S-INSUF", name="Short Fall", is_fall=True, duration_s=3.4,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    benchmark = [
        Sequence(annotation=ann_adequate, fps=15.0),
        Sequence(annotation=ann_insufficient, fps=15.0),
    ]

    # S-ADEQUATE detects an alert in window; S-INSUF emits 0 alerts
    results = [
        SequenceResult(
            sequence_id="S-ADEQUATE", annotation_name="Adequate Fall",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[3.5],
        ),
        SequenceResult(
            sequence_id="S-INSUF", annotation_name="Short Fall",
            is_fall_ground_truth=True,
            alerts=[],
            alert_timestamps=[],
        ),
    ]

    report = evaluate_results(benchmark, results, required_confirmation_s=2.0)
    assert report.tp == 1
    assert report.fn == 0
    assert report.insufficient_observation_sequences == 1
    assert report.recall == 1.0  # S-INSUF is not penalized as an FN


def test_adequate_observation_without_detector_trigger_is_genuine_fn():
    """When adequate post-fall footage (>=2.0s) exists and detector fails to alert, it is a genuine FN."""
    ann = SequenceAnnotation(
        sequence_id="S-MISSED", name="Missed Fall", is_fall=True, duration_s=6.0,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Available post-impact is 4.0s (plenty of time for 2.0s confirmation)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-MISSED", annotation_name="Missed Fall",
            is_fall_ground_truth=True,
            alerts=[],
            alert_timestamps=[],
        )
    ]

    report = evaluate_results(benchmark, results, required_confirmation_s=2.0)
    assert report.fn == 1
    assert report.tp == 0
    assert report.insufficient_observation_sequences == 0
    assert report.sequence_evaluations[0].verdict == "FN"


def test_insufficient_observation_with_confirmed_alert_counts_as_insufficient():
    """An alert firing on short footage does NOT grant eligibility for primary recall.

    Predefined observation requirement (< 2.0s post-fall) classifies the clip as
    INSUFFICIENT_OBSERVATION, avoiding outcome-dependent eligibility bias. The confirmed
    alert is preserved as an observed detection.
    """
    ann = SequenceAnnotation(
        sequence_id="S-FAST-ALERT", name="Fast Alert", is_fall=True, duration_s=3.5,
        fall_start_s=1.0, fall_impact_s=2.0,
    )
    # Available post-impact: 1.5s (< 2.0s required). Alert fires at t=2.8s
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-FAST-ALERT", annotation_name="Fast Alert",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[2.8],
        )
    ]

    report = evaluate_results(benchmark, results, required_confirmation_s=2.0)
    assert report.tp == 0
    assert report.fn == 0
    assert report.insufficient_observation_sequences == 1
    assert report.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"
    assert report.observed_fall_alerts == 1
    assert report.observed_mean_latency_s == pytest.approx(0.8, abs=0.01)


def test_exact_boundary_cases_for_observation_window():
    """Exact boundary post-impact == 2.0s is adequate (FN), whereas 1.99s is insufficient."""
    ann_exact = SequenceAnnotation(
        sequence_id="S-EXACT", name="Exact 2.0s Post-Impact", is_fall=True, duration_s=4.00,
        fall_start_s=1.0, fall_impact_s=2.00,
    )
    ann_below = SequenceAnnotation(
        sequence_id="S-BELOW", name="Just Below 2.0s", is_fall=True, duration_s=3.99,
        fall_start_s=1.0, fall_impact_s=2.00,
    )
    benchmark = [
        Sequence(annotation=ann_exact, fps=15.0),
        Sequence(annotation=ann_below, fps=15.0),
    ]
    results = [
        SequenceResult(sequence_id="S-EXACT", annotation_name="Exact", is_fall_ground_truth=True, alerts=[], alert_timestamps=[]),
        SequenceResult(sequence_id="S-BELOW", annotation_name="Below", is_fall_ground_truth=True, alerts=[], alert_timestamps=[]),
    ]

    report = evaluate_results(benchmark, results, required_confirmation_s=2.0)
    # S-EXACT had exactly 2.00s post-impact -> adequate observation -> FN
    eval_exact = next(e for e in report.sequence_evaluations if e.sequence_id == "S-EXACT")
    assert eval_exact.verdict == "FN"

    # S-BELOW had 1.99s post-impact -> insufficient observation -> INSUFFICIENT_OBSERVATION
    eval_below = next(e for e in report.sequence_evaluations if e.sequence_id == "S-BELOW")
    assert eval_below.verdict == "INSUFFICIENT_OBSERVATION"


def test_unverified_ground_truth_status():
    """Sequences without ground-truth fall status (is_fall=None) receive UNVERIFIED status."""
    ann_unverified = SequenceAnnotation(
        sequence_id="S-UNVERIFIED-REAL", name="Real Unannotated", is_fall=None, duration_s=5.0,
    )
    benchmark = [Sequence(annotation=ann_unverified, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-UNVERIFIED-REAL", annotation_name="Real Unannotated",
            is_fall_ground_truth=None,
            alerts=[_mock_candidate()],
            alert_timestamps=[2.5],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.unverified_sequences == 1
    assert report.tp == 0
    assert report.fp == 0
    assert report.sequence_evaluations[0].verdict == "UNVERIFIED"


def test_adl_multiple_alerts_counted_as_one_fp_and_duplicates():
    """In an ADL sequence, the first alert is an FP and remaining alerts are duplicate alerts."""
    ann = SequenceAnnotation(
        sequence_id="S-ADL-MULTI", name="Multi False Alarm", is_fall=False, duration_s=6.0,
    )
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-ADL-MULTI", annotation_name="Multi False Alarm",
            is_fall_ground_truth=False,
            alerts=[_mock_candidate(), _mock_candidate(), _mock_candidate()],
            alert_timestamps=[1.0, 2.0, 3.0],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.fp == 1
    assert report.duplicate_alerts == 2
    assert report.tn == 0


# --------------------------------------------------------------------------- #
# Zero-Division Edge Cases
# --------------------------------------------------------------------------- #

def test_zero_division_empty_benchmark():
    """An empty benchmark evaluates safely with 0/0 edge cases returning defined values."""
    report = evaluate_results([], [])
    assert report.total_sequences == 0
    assert report.tp == 0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.specificity == 0.0
    assert report.mean_latency_s is None


def test_zero_division_no_positives_all_tn():
    """All ADL sequences correctly producing 0 alerts returns 0.0 precision/recall gracefully."""
    ann_adl = SequenceAnnotation(
        sequence_id="S-ONLY-ADL", name="ADL Only", is_fall=False, duration_s=4.0,
    )
    benchmark = [Sequence(annotation=ann_adl, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-ONLY-ADL", annotation_name="ADL Only",
            is_fall_ground_truth=False, alerts=[], alert_timestamps=[],
        )
    ]
    report = evaluate_results(benchmark, results)
    assert report.tn == 1
    assert report.tp == 0
    assert report.fp == 0
    assert report.fn == 0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.specificity == 1.0


# --------------------------------------------------------------------------- #
# Report Text Formatting — Labels & Dataset Type
# --------------------------------------------------------------------------- #

def test_report_text_contains_disclaimer_and_metrics():
    """Synthetic report text contains correct disclaimer, heading and metric labels."""
    benchmark = get_synthetic_benchmark(fps=15.0)
    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(benchmark)
    report = evaluate_results(benchmark, results)
    text = report.to_text()

    assert "SYNTHETIC BASELINE EVALUATION REPORT" in text
    assert "DOES NOT REFLECT REAL-WORLD FIELD ACCURACY" in text
    assert "Precision" in text
    assert "Recall" in text
    assert "F1-Score" in text
    assert "True Positives" in text
    assert "True Negatives" in text
    # Synthetic note must appear in the protocol footer
    assert "Synthetic Poses" in text
    # Real-dataset label must NOT appear in a synthetic report
    assert "REAL DATASET" not in text


def test_report_text_urfd_pilot_label():
    """A report with dataset_type='URFD_PILOT' uses the URFD title, not the synthetic banner."""
    benchmark = get_synthetic_benchmark(fps=15.0)
    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(benchmark)
    report = evaluate_results(
        benchmark,
        results,
        dataset_type="URFD_PILOT",
        report_title="ARJUN FALL DETECTION -- URFD PILOT EVALUATION REPORT",
    )
    text = report.to_text()

    # Real-dataset heading must be present
    assert "URFD PILOT EVALUATION REPORT" in text
    assert "REAL DATASET: URFD_PILOT" in text
    # Synthetic-only disclaimer must NOT appear
    assert "SYNTHETIC BENCHMARK ONLY" not in text
    assert "DOES NOT REFLECT REAL-WORLD FIELD ACCURACY" not in text
    # Real-video footer note must appear instead of the synthetic note
    assert "Real Video" in text
    assert "Synthetic Poses" not in text


# --------------------------------------------------------------------------- #
# Report Text Formatting — N/A for Zero Denominators
# --------------------------------------------------------------------------- #

def test_report_text_na_for_zero_precision_and_recall_denominators():
    """When both TP+FP==0 and TP+FN==0, precision, recall, and F1 display as N/A."""
    # A benchmark with only an UNVERIFIED sequence produces no TP, FP, FN, TN.
    ann_unverified = SequenceAnnotation(
        sequence_id="S-UNV", name="Unverified", is_fall=None, duration_s=4.0,
    )
    benchmark = [Sequence(annotation=ann_unverified, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-UNV", annotation_name="Unverified",
            is_fall_ground_truth=None, alerts=[], alert_timestamps=[],
        )
    ]
    report = evaluate_results(benchmark, results)
    text = report.to_text()

    # All four metrics are undefined: denominators are all zero
    assert "Precision              : N/A" in text
    assert "Recall (Sensitivity)   : N/A" in text
    assert "F1-Score               : N/A" in text
    assert "Specificity (TNR)      : N/A" in text

    # Underlying float values are still 0.0 (the safe sentinel), not changed
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.specificity == 0.0


def test_report_text_na_specificity_only_fall_sequences():
    """When there are no ADL sequences (TN+FP==0), specificity displays as N/A.

    Precision and recall are still defined if TP and FN are non-zero.
    """
    ann_fall = SequenceAnnotation(
        sequence_id="S-FALL", name="Fall", is_fall=True, duration_s=6.0,
        fall_start_s=1.0, fall_impact_s=1.5,
    )
    benchmark = [Sequence(annotation=ann_fall, fps=15.0)]
    # Alert fires inside the window -> TP=1, FN=0, FP=0, TN=0
    results = [
        SequenceResult(
            sequence_id="S-FALL", annotation_name="Fall",
            is_fall_ground_truth=True, alerts=[_mock_candidate()],
            alert_timestamps=[2.5],
        )
    ]
    report = evaluate_results(benchmark, results)
    text = report.to_text()

    # Precision and recall are defined (TP=1)
    assert "Precision              : N/A" not in text
    assert "Recall (Sensitivity)   : N/A" not in text
    # Specificity is undefined because there are no ADL sequences (TN+FP==0)
    assert "Specificity (TNR)      : N/A" in text


def test_report_text_valid_denominators_show_percentages():
    """When denominators are non-zero, metrics render as percentages (not N/A)."""
    benchmark = get_synthetic_benchmark(fps=15.0)
    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(benchmark)
    # This benchmark has both falls and ADL, so all denominators are non-zero
    report = evaluate_results(benchmark, results)
    text = report.to_text()

    # None of the four metrics should show N/A
    for metric_label in ["Precision", "Recall (Sensitivity)", "F1-Score", "Specificity (TNR)"]:
        assert f"{metric_label}      : N/A" not in text, (
            f"Expected a percentage for {metric_label!r}, found N/A"
        )
    # All should show 100.0% (perfect synthetic benchmark)
    assert "Precision              : 100.0%" in text
    assert "Recall (Sensitivity)   : 100.0%" in text
    assert "F1-Score               : 100.0%" in text
    assert "Specificity (TNR)      : 100.0%" in text


# --------------------------------------------------------------------------- #
# Prediction-Independent Eligibility Tests
# --------------------------------------------------------------------------- #

def test_prediction_independent_eligibility_insufficient_clip_with_alert():
    """An alert firing on a clip with < 2.0s post-fall observation does not force TP eligibility.

    The sequence must remain INSUFFICIENT_OBSERVATION for the primary recall denominator,
    while preserving the confirmed alert as a separately observed detection.
    """
    ann = SequenceAnnotation(
        sequence_id="S-SHORT-ALERT", name="Short Clip With Alert",
        is_fall=True, duration_s=3.2,
        fall_start_s=0.5, fall_impact_s=1.5,
    )
    # Available post-fall footage: 3.2s - 1.5s = 1.7s (< 2.0s confirmation required)
    benchmark = [Sequence(annotation=ann, fps=15.0)]
    results = [
        SequenceResult(
            sequence_id="S-SHORT-ALERT", annotation_name="Short Clip With Alert",
            is_fall_ground_truth=True,
            alerts=[_mock_candidate()],
            alert_timestamps=[3.1],  # Alert inside [1.5s, 6.5s] window
        )
    ]

    report = evaluate_results(benchmark, results, required_confirmation_s=2.0)

    # Primary recall denominator excludes the sequence
    assert report.fall_sequences == 0
    assert report.insufficient_observation_sequences == 1
    assert report.tp == 0
    assert report.fn == 0
    assert report.fp == 0
    assert report.sequence_evaluations[0].verdict == "INSUFFICIENT_OBSERVATION"

    # Confirmed alert and latency are preserved as diagnostic observed detections
    assert report.observed_fall_alerts == 1
    assert report.observed_mean_latency_s == pytest.approx(1.6, abs=0.01)
    assert report.sequence_evaluations[0].matched_alert_timestamp == pytest.approx(3.1, abs=0.01)
    assert report.sequence_evaluations[0].latency_s == pytest.approx(1.6, abs=0.01)
    assert "Separately observed confirmed alert" in report.sequence_evaluations[0].notes


def test_report_metrics_na_when_all_falls_insufficient_observation():
    """When all fall clips have insufficient observation, recall and F1 display as N/A."""
    ann_fall = SequenceAnnotation(
        sequence_id="S-INSUF", name="Short Fall",
        is_fall=True, duration_s=3.0,
        fall_start_s=0.5, fall_impact_s=1.5,
    )
    ann_adl = SequenceAnnotation(
        sequence_id="S-ADL", name="Normal ADL",
        is_fall=False, duration_s=5.0,
    )
    benchmark = [
        Sequence(annotation=ann_fall, fps=15.0),
        Sequence(annotation=ann_adl, fps=15.0),
    ]
    results = [
        SequenceResult(
            sequence_id="S-INSUF", annotation_name="Short Fall",
            is_fall_ground_truth=True, alerts=[], alert_timestamps=[],
        ),
        SequenceResult(
            sequence_id="S-ADL", annotation_name="Normal ADL",
            is_fall_ground_truth=False, alerts=[], alert_timestamps=[],
        ),
    ]

    report = evaluate_results(
        benchmark, results,
        required_confirmation_s=2.0,
        dataset_type="PILOT_BENCHMARK",
        report_title="ARJUN FALL DETECTION -- PILOT BENCHMARK",
    )
    text = report.to_text()

    assert report.insufficient_observation_sequences == 1
    assert report.fall_sequences == 0
    assert report.adl_sequences == 1
    assert report.tp == 0
    assert report.fn == 0
    assert report.tn == 1

    # Primary recall and precision are N/A; specificity is 100.0%
    assert "Precision              : N/A" in text
    assert "Recall (Sensitivity)   : N/A" in text
    assert "F1-Score               : N/A" in text
    assert "Specificity (TNR)      : 100.0%" in text
    assert "Latency                : N/A" in text
    assert "confirmed-alert recall is undefined (N/A)" in text

