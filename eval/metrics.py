"""
Evaluation metrics for event-level fall detection.

Evaluates sequences as whole events rather than independent frame classifications.
Implements an algorithm-neutral matching window convention [impact, impact + 5.0s]
so that rule-based systems and future ML models can be compared on an equal footing.

Event Matching Rules:
1. Matching Window: Begins at ground-truth impact (minus window_before_impact_s)
   and ends window_after_impact_s (default: 5.0s) afterward.
2. Single-Alert Match: At most one alert in the window matches the fall event (TP).
   Any additional alerts in the sequence are tracked separately as duplicate alerts.
3. Unmatched Alerts: Any alerts firing outside the matching window (e.g. before impact
   or after the window closes when no fall was matched) count as False Positives (FP).
   All alerts in normal-activity (ADL) sequences count as False Positives (FP).
4. Premature Alerts: Alerts firing before impact (t < impact - window_before_s) are
   unmatched false alarms (FP). If an alert also fires in the window, the fall is a TP
   and the premature alert is an FP.
5. Missing Impact Annotations: If fall_impact_s is missing on a fall sequence, the
   matching window defaults to fall_start_s or the full sequence, with a documented note.
6. Truncated Videos: If the video ends before the matching window closes, the window is
   clamped to sequence duration, and truncation is explicitly noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from eval.dataset import Sequence
from eval.runner import SequenceResult


@dataclass
class SequenceEvaluation:
    """Detailed evaluation verdict for one sequence."""
    sequence_id: str
    name: str
    is_fall_ground_truth: bool
    verdict: str                        # "TP", "FP", "FN", "TN"
    matched_alert_timestamp: float | None
    latency_s: float | None
    total_alerts: int
    duplicate_alerts: int
    notes: str


@dataclass
class EvaluationReport:
    """Full aggregate evaluation report across a benchmark."""
    total_sequences: int
    fall_sequences: int
    adl_sequences: int
    tp: int
    fp: int
    fn: int
    tn: int
    duplicate_alerts: int
    precision: float
    recall: float
    f1: float
    specificity: float
    mean_latency_s: float | None
    min_latency_s: float | None
    max_latency_s: float | None
    insufficient_observation_sequences: int = 0
    unverified_sequences: int = 0
    matching_window_desc: str = "[impact, impact + 5.0s]"
    sequence_evaluations: list[SequenceEvaluation] = field(default_factory=list)

    def to_text(self) -> str:
        """Render a readable summary table and disclaimer."""
        lines: list[str] = [
            "=" * 78,
            " ARJUN FALL DETECTION -- SYNTHETIC BASELINE EVALUATION REPORT",
            " [SYNTHETIC BENCHMARK ONLY -- DOES NOT REFLECT REAL-WORLD FIELD ACCURACY]",
            "=" * 78,
            "",
            f" Evaluation Protocol       : Algorithm-neutral event matching",
            f" Matching Window           : {self.matching_window_desc}",
            f" Total Sequences Evaluated : {self.total_sequences}",
            f"   - Fall Scenarios         : {self.fall_sequences}",
            f"   - Normal ADL Scenarios   : {self.adl_sequences}",
            "",
            " Event-Level Confusion Matrix:",
            f"   - True Positives  (TP)   : {self.tp:<4} (Actual falls correctly detected in window)",
            f"   - True Negatives  (TN)   : {self.tn:<4} (Normal ADL correctly ignored)",
            f"   - False Positives (FP)   : {self.fp:<4} (Unmatched alerts & ADL false alarms)",
            f"   - False Negatives (FN)   : {self.fn:<4} (Missed falls in window)",
            f"   - Duplicate Alerts (DUP) : {self.duplicate_alerts:<4} (Redundant re-triggers for same incident)",
        ]

        if self.insufficient_observation_sequences > 0 or self.unverified_sequences > 0:
            lines.extend([
                "",
                " Excluded from Classification Denominators:",
                f"   - Insufficient Obs (ISO) : {self.insufficient_observation_sequences:<4} (Cut off before confirmation window)",
                f"   - Unverified Sequences   : {self.unverified_sequences:<4} (Lacks verified ground truth)",
            ])

        lines.extend([
            "",
            " Performance Metrics:",
            f"   - Precision              : {self.precision:.1%}",
            f"   - Recall (Sensitivity)   : {self.recall:.1%}",
            f"   - F1-Score               : {self.f1:.1%}",
            f"   - Specificity (TNR)      : {self.specificity:.1%}",
            "",
            " Detection Latency (Time from Impact to Matched Alert):",
        ])

        if self.mean_latency_s is not None:
            lines.extend([
                f"   - Mean Latency           : {self.mean_latency_s:.2f}s",
                f"   - Min Latency            : {self.min_latency_s:.2f}s",
                f"   - Max Latency            : {self.max_latency_s:.2f}s",
            ])
        else:
            lines.append("   - Latency                : N/A (No True Positives detected)")

        lines.extend([
            "",
            "-" * 78,
            f"{'Sequence ID':<16} {'Type':<6} {'Verdict':<25} {'Alerts':<8} {'Latency':<10} {'Name'}",
            "-" * 78,
        ])

        for seq in self.sequence_evaluations:
            stype = "FALL" if seq.is_fall_ground_truth else ("ADL" if seq.verdict != "UNVERIFIED" else "UNV")
            lat_str = f"{seq.latency_s:.2f}s" if seq.latency_s is not None else "--"
            lines.append(
                f"{seq.sequence_id:<16} {stype:<6} {seq.verdict:<25} {seq.total_alerts:<8} {lat_str:<10} {seq.name}"
            )

        lines.extend([
            "-" * 78,
            "",
            " PROTOCOL & MATCHING CONVENTIONS:",
            " 1. Algorithm-Neutral Matching: A fall is credited if an alert fires between",
            f"    ground-truth impact and the configured window ({self.matching_window_desc}).",
            " 2. Exact Latency: Measured as (alert_timestamp - impact_timestamp) separately.",
            " 3. At Most One Match: The first in-window alert counts as TP; subsequent alerts",
            "    are counted separately in 'Duplicate Alerts'.",
            " 4. False Positive Policy: Alerts before impact or outside the matching window",
            "    count as False Positives (FP). All alerts in ADL sequences count as FP.",
            " 5. Insufficient Observation: Fall sequences where the clip ends before the",
            "    required confirmation period are marked INSUFFICIENT_OBSERVATION and excluded",
            "    from TP/FN denominators.",
            " 6. Synthetic Poses: Benchmark poses are geometrically synthesized; real video",
            "    evaluation requires subsequent empirical testing on recorded video datasets.",
            "=" * 78,
        ])

        return "\n".join(lines)


def evaluate_results(
    benchmark: list[Sequence],
    results: list[SequenceResult],
    window_after_impact_s: float = 5.0,
    window_before_impact_s: float = 0.0,
    required_confirmation_s: float = 2.0,
    confirmation_tolerance_s: float = 0.0,
) -> EvaluationReport:
    """
    Compute event-level metrics comparing Sequence annotations with runner results.

    Parameters:
        benchmark: List of benchmark sequences containing ground truth annotations.
        results: List of SequenceResult instances produced by OfflineFallRunner.
        window_after_impact_s: Seconds after impact within which an alert matches (default: 5.0).
        window_before_impact_s: Seconds before impact allowed for pre-impact alerts (default: 0.0).
        required_confirmation_s: Minimum post-impact observation window required for a
            detector to confirm a fall (default: 2.0s, matching production FallDetector).
        confirmation_tolerance_s: Allowable tolerance on confirmation window (default: 0.0s).
    """
    res_map = {r.sequence_id: r for r in results}

    tp = fp = fn = tn = duplicates = 0
    insufficient_obs = 0
    unverified = 0
    latencies: list[float] = []
    evaluations: list[SequenceEvaluation] = []

    fall_count = 0
    adl_count = 0

    window_desc = (
        f"[impact - {window_before_impact_s:.1f}s, impact + {window_after_impact_s:.1f}s]"
        if window_before_impact_s > 0
        else f"[impact, impact + {window_after_impact_s:.1f}s]"
    )

    for seq in benchmark:
        ann = seq.annotation
        res = res_map.get(ann.sequence_id)
        alerts = res.alerts if res is not None else []
        timestamps = sorted(res.alert_timestamps) if res is not None else []

        if ann.is_fall is None:
            # Unverified sequence: do not count as fall (TP/FN) or ADL (TN/FP).
            # Record verdict as UNVERIFIED and exclude from accuracy metrics.
            unverified += 1
            evaluations.append(SequenceEvaluation(
                sequence_id=ann.sequence_id,
                name=ann.name,
                is_fall_ground_truth=False,
                verdict="UNVERIFIED",
                matched_alert_timestamp=timestamps[0] if timestamps else None,
                latency_s=None,
                total_alerts=len(alerts),
                duplicate_alerts=max(0, len(alerts) - 1) if alerts else 0,
                notes="Ground truth unverified: excluded from accuracy metrics until official labels are imported.",
            ))
            continue

        if ann.is_fall is True:
            # Determine anchor point for impact / confirmation
            anchor = ann.confirmation_anchor_s()
            missing_impact = False
            if anchor is not None:
                impact_point = anchor
            elif ann.fall_impact_s is not None:
                impact_point = ann.fall_impact_s
            elif ann.fall_start_s is not None:
                impact_point = ann.fall_start_s
                missing_impact = True
            elif ann.fall_transition_start_s is not None:
                impact_point = ann.fall_transition_start_s
                missing_impact = True
            else:
                impact_point = 0.0
                missing_impact = True

            w_start = max(0.0, impact_point - window_before_impact_s)
            nominal_w_end = impact_point + window_after_impact_s

            # Handle edge case: video ends before matching window closes
            truncated_video = False
            if ann.duration_s < nominal_w_end:
                w_end = ann.duration_s
                truncated_video = True
            else:
                w_end = nominal_w_end

            # Partition alerts: premature (before window), in-window, late (after window)
            premature_alerts = [t for t in timestamps if t < w_start]
            in_window_alerts = [t for t in timestamps if w_start <= t <= w_end]
            late_alerts = [t for t in timestamps if t > w_end]

            # 1. Unmatched premature alerts before impact count as FP
            if premature_alerts:
                fp += len(premature_alerts)

            matched_ts: float | None = None
            seq_latency: float | None = None
            seq_duplicates = 0

            # 2. In-window alerts: exactly one matches as TP; subsequent are duplicates
            if in_window_alerts:
                fall_count += 1
                tp += 1
                matched_ts = in_window_alerts[0]
                seq_latency = matched_ts - impact_point
                latencies.append(seq_latency)
                seq_duplicates += len(in_window_alerts) - 1
                # Any late alerts after a matched TP are extra duplicates
                seq_duplicates += len(late_alerts)
                duplicates += seq_duplicates
                verdict = "TP"
                notes = f"Matched alert at t={matched_ts:.2f}s (latency={seq_latency:.2f}s)."
            else:
                # No alert fired inside window. Check observation window sufficiency!
                obs_anchor = ann.confirmation_anchor_s()
                if obs_anchor is None:
                    # Lacks verified confirmation anchor; cannot verify observation sufficiency
                    fall_count += 1
                    fn += 1
                    verdict = "FN"
                    notes = "No alert fired; missing confirmed fall anchor to assess observation sufficiency."
                    if late_alerts:
                        fp += len(late_alerts)
                        notes += f" Late alert(s) at {late_alerts} counted as FP."
                else:
                    available_post_fall = ann.duration_s - obs_anchor
                    min_post_fall = required_confirmation_s - confirmation_tolerance_s

                    if available_post_fall < min_post_fall:
                        insufficient_obs += 1
                        verdict = "INSUFFICIENT_OBSERVATION"
                        notes = (
                            f"Post-fall observation window ({available_post_fall:.2f}s) is shorter "
                            f"than required confirmation time ({required_confirmation_s:.2f}s). "
                            f"Excluded from confirmation-based TP/FN denominators."
                        )
                        if late_alerts:
                            fp += len(late_alerts)
                            notes += f" Late alert(s) at {late_alerts} counted as FP."
                    else:
                        fall_count += 1
                        fn += 1
                        verdict = "FN"
                        notes = "No alert fired within matching window."
                        if late_alerts:
                            fp += len(late_alerts)
                            notes += f" Late alert(s) at {late_alerts} counted as FP."

            if premature_alerts:
                notes += f" {len(premature_alerts)} premature alert(s) before impact counted as FP."
            if missing_impact:
                notes += " Note: fall_impact_s missing; anchored window to fallback timestamp."
            if truncated_video and verdict == "FN":
                notes += f" Note: Video ended at {ann.duration_s:.1f}s before 5.0s window closed at {nominal_w_end:.1f}s."

            evaluations.append(SequenceEvaluation(
                sequence_id=ann.sequence_id,
                name=ann.name,
                is_fall_ground_truth=True,
                verdict=verdict,
                matched_alert_timestamp=matched_ts,
                latency_s=seq_latency,
                total_alerts=len(alerts),
                duplicate_alerts=seq_duplicates,
                notes=notes.strip(),
            ))

        else:
            adl_count += 1
            if len(alerts) == 0:
                tn += 1
                verdict = "TN"
                notes = "Correctly remained quiet during normal activity."
                seq_dups = 0
            else:
                # All alerts in an ADL sequence are unmatched false positives
                verdict = "FP"
                # First alert is the sequence-level FP; additional alerts are tracked as duplicates
                fp += 1
                seq_dups = max(0, len(alerts) - 1)
                duplicates += seq_dups
                notes = f"False alarm triggered on ADL at t={timestamps} ({seq_dups} duplicate(s))."

            evaluations.append(SequenceEvaluation(
                sequence_id=ann.sequence_id,
                name=ann.name,
                is_fall_ground_truth=False,
                verdict=verdict,
                matched_alert_timestamp=timestamps[0] if timestamps else None,
                latency_s=None,
                total_alerts=len(alerts),
                duplicate_alerts=seq_dups,
                notes=notes,
            ))

    # Metric calculations with standard zero-division safety (scikit-learn convention)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    mean_lat = float(np.mean(latencies)) if latencies else None
    min_lat = float(np.min(latencies)) if latencies else None
    max_lat = float(np.max(latencies)) if latencies else None

    return EvaluationReport(
        total_sequences=len(benchmark),
        fall_sequences=fall_count,
        adl_sequences=adl_count,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        duplicate_alerts=duplicates,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        mean_latency_s=mean_lat,
        min_latency_s=min_lat,
        max_latency_s=max_lat,
        insufficient_observation_sequences=insufficient_obs,
        unverified_sequences=unverified,
        matching_window_desc=window_desc,
        sequence_evaluations=evaluations,
    )
