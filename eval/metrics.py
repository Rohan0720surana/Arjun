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
    dataset_type: str = "SYNTHETIC"
    report_title: str = "ARJUN FALL DETECTION -- SYNTHETIC BASELINE EVALUATION REPORT"
    observed_fall_alerts: int = 0
    observed_mean_latency_s: float | None = None

    # ---------------------------------------------------------------------- #
    # Zero-denominator metric display helpers
    # ---------------------------------------------------------------------- #

    def _precision_str(self) -> str:
        """Display precision as percentage, or N/A when TP+FP == 0 (undefined)."""
        return "N/A" if (self.tp + self.fp) == 0 else f"{self.precision:.1%}"

    def _recall_str(self) -> str:
        """Display recall as percentage, or N/A when TP+FN == 0 (undefined)."""
        return "N/A" if (self.tp + self.fn) == 0 else f"{self.recall:.1%}"

    def _f1_str(self) -> str:
        """Display F1 as percentage, or N/A when both precision and recall are undefined."""
        # F1 is undefined when both its inputs are undefined (zero denominator).
        # When precision + recall > 0 it is well-defined even if one input is zero.
        return "N/A" if (self.tp + self.fp) == 0 and (self.tp + self.fn) == 0 else f"{self.f1:.1%}"

    def _specificity_str(self) -> str:
        """Display specificity as percentage, or N/A when TN+FP == 0 (no ADL sequences)."""
        return "N/A" if (self.tn + self.fp) == 0 else f"{self.specificity:.1%}"

    def to_text(self) -> str:
        """Render a readable summary table and disclaimer.

        The heading and disclaimer lines adapt to ``dataset_type``:
        - ``"SYNTHETIC"`` (default): prints the synthetic-only disclaimer.
        - Any other value (e.g. ``"URFD_PILOT"``): prints a real-dataset heading
          without the synthetic disclaimer, so that real URFD results are not
          misleadingly labelled as synthetic.

        Metrics whose denominator is zero are displayed as ``N/A`` (undefined)
        rather than ``0.0%``.  The underlying float attributes are unchanged so
        downstream callers can still compare them numerically.
        """
        is_synthetic = self.dataset_type == "SYNTHETIC"

        lines: list[str] = [
            "=" * 78,
            f" {self.report_title}",
        ]
        if is_synthetic:
            lines.append(" [SYNTHETIC BENCHMARK ONLY -- DOES NOT REFLECT REAL-WORLD FIELD ACCURACY]")
        else:
            lines.append(f" [REAL DATASET: {self.dataset_type}]")  # actual dataset label
        lines.append("=" * 78)

        lines.extend([
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
        ])

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
            # N/A is displayed instead of 0.0% when the denominator is zero
            # (undefined metrics), so that reports are not misleadingly precise.
            f"   - Precision              : {self._precision_str()}",
            f"   - Recall (Sensitivity)   : {self._recall_str()}",
            f"   - F1-Score               : {self._f1_str()}",
            f"   - Specificity (TNR)      : {self._specificity_str()}",
        ])
        if not is_synthetic and (self.fall_sequences <= 2):
            if self.fall_sequences == 0:
                lines.append(
                    f"   [!] Exploratory notice   : Metrics derived from small pilot sample (0 eligible fall(s), "
                    f"{self.adl_sequences} ADL(s)); confirmed-alert recall is undefined (N/A) and specificity reflects 5/5 negatives."
                )
            else:
                lines.append(
                    f"   [!] Exploratory notice   : Metrics derived from small sample ({self.fall_sequences} evaluable fall(s), "
                    f"{self.adl_sequences} ADL(s)); 100% scores do not represent general real-world accuracy."
                )
        lines.extend([
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
            lines.append("   - Latency                : N/A (No eligible True Positives in primary denominator)")

        if self.observed_fall_alerts > 0:
            lines.extend([
                "",
                " Separately Reported Observed Detections (Truncated Clips):",
                f"   - Confirmed Alerts       : {self.observed_fall_alerts} alert(s) observed on clip(s) with < 2.0s post-impact observation",
            ])
            if self.observed_mean_latency_s is not None:
                lines.append(f"   - Observed Alert Latency : {self.observed_mean_latency_s:.2f}s (e.g. fall-04 alert at t=3.14s)")

        lines.extend([
            "",
            "-" * 78,
            f"{'Sequence ID':<16} {'Type':<6} {'Verdict':<25} {'Alerts':<8} {'Latency':<10} {'Name'}",
            "-" * 78,
        ])

        for seq in self.sequence_evaluations:
            stype = "FALL" if seq.is_fall_ground_truth else ("ADL" if seq.verdict != "UNVERIFIED" else "UNV")
            if seq.latency_s is not None:
                lat_str = f"{seq.latency_s:.2f}s"
                if seq.verdict == "INSUFFICIENT_OBSERVATION":
                    lat_str += "*"
            else:
                lat_str = "--"
            lines.append(
                f"{seq.sequence_id:<16} {stype:<6} {seq.verdict:<25} {seq.total_alerts:<8} {lat_str:<10} {seq.name}"
            )

        lines.append("-" * 78)
        if any(seq.verdict == "INSUFFICIENT_OBSERVATION" and seq.latency_s is not None for seq in self.sequence_evaluations):
            lines.append(" * Alert confirmed on truncated clip (< 2.0s post-impact). Excluded from primary recall.")
            lines.append("-" * 78)

        # Protocol footer adapts its last note to indicate whether poses are synthetic.
        synth_note = (
            " 6. Synthetic Poses: Benchmark poses are geometrically synthesized; real video"
            "\n    evaluation requires subsequent empirical testing on recorded video datasets."
            if is_synthetic else
            " 6. Real Video: Results are derived from actual video recordings and official"
            "\n    ground-truth annotations. Findings reflect real-world detector behaviour."
        )

        lines.extend([
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
            synth_note,
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
    dataset_type: str = "SYNTHETIC",
    report_title: str = "ARJUN FALL DETECTION -- SYNTHETIC BASELINE EVALUATION REPORT",
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
        dataset_type: Short identifier for the dataset used (e.g. ``"SYNTHETIC"``,
            ``"URFD_PILOT"``).  Controls the report heading and disclaimer text.
            Default ``"SYNTHETIC"`` retains the existing synthetic-only banner.
        report_title: Full title line printed at the top of the report.  Callers
            using real datasets should pass an appropriate descriptive title.
    """
    res_map = {r.sequence_id: r for r in results}

    tp = fp = fn = tn = duplicates = 0
    insufficient_obs = 0
    unverified = 0
    observed_fall_alerts = 0
    latencies: list[float] = []
    observed_latencies: list[float] = []
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
            # 1. Determine anchor point for impact and observation window sufficiency.
            # Observation sufficiency MUST be evaluated from ground-truth timing BEFORE
            # inspecting detector predictions to prevent outcome-dependent eligibility bias.
            obs_anchor = ann.confirmation_anchor_s()
            missing_impact = False
            if obs_anchor is not None:
                impact_point = obs_anchor
                available_post_fall = ann.duration_s - obs_anchor
                has_sufficient_obs = available_post_fall >= (required_confirmation_s - confirmation_tolerance_s)
            elif ann.fall_impact_s is not None:
                impact_point = ann.fall_impact_s
                available_post_fall = ann.duration_s - impact_point
                has_sufficient_obs = available_post_fall >= (required_confirmation_s - confirmation_tolerance_s)
            elif ann.fall_start_s is not None:
                impact_point = ann.fall_start_s
                missing_impact = True
                available_post_fall = ann.duration_s - impact_point
                has_sufficient_obs = available_post_fall >= (required_confirmation_s - confirmation_tolerance_s)
            elif ann.fall_transition_start_s is not None:
                impact_point = ann.fall_transition_start_s
                missing_impact = True
                available_post_fall = ann.duration_s - impact_point
                has_sufficient_obs = available_post_fall >= (required_confirmation_s - confirmation_tolerance_s)
            else:
                impact_point = 0.0
                missing_impact = True
                available_post_fall = ann.duration_s
                has_sufficient_obs = True  # fallback if no timing annotations exist

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

            # Premature alerts before impact count as FP
            if premature_alerts:
                fp += len(premature_alerts)

            matched_ts: float | None = None
            seq_latency: float | None = None
            seq_duplicates = 0

            # 2. Evaluate according to observation window sufficiency
            if not has_sufficient_obs:
                # Ineligible for primary confirmed-alert denominator
                insufficient_obs += 1
                verdict = "INSUFFICIENT_OBSERVATION"
                if in_window_alerts:
                    matched_ts = in_window_alerts[0]
                    seq_latency = matched_ts - impact_point
                    observed_latencies.append(seq_latency)
                    observed_fall_alerts += 1
                    seq_duplicates = len(in_window_alerts) - 1 + len(late_alerts)
                    notes = (
                        f"Post-fall observation window ({available_post_fall:.2f}s) is shorter than required "
                        f"confirmation time ({required_confirmation_s:.2f}s); classified as INSUFFICIENT_OBSERVATION "
                        f"for primary recall denominator. Separately observed confirmed alert at t={matched_ts:.2f}s "
                        f"(latency={seq_latency:.2f}s) enabled by transition-phase kinematic drop trigger."
                    )
                else:
                    notes = (
                        f"Post-fall observation window ({available_post_fall:.2f}s) is shorter than required "
                        f"confirmation time ({required_confirmation_s:.2f}s). Excluded from primary recall denominator."
                    )
                if late_alerts and not in_window_alerts:
                    notes += f" Late alert(s) at {late_alerts} noted."
            else:
                # Eligible fall sequence for primary confirmed-alert evaluation
                fall_count += 1
                if in_window_alerts:
                    tp += 1
                    matched_ts = in_window_alerts[0]
                    seq_latency = matched_ts - impact_point
                    latencies.append(seq_latency)
                    seq_duplicates = len(in_window_alerts) - 1 + len(late_alerts)
                    duplicates += seq_duplicates
                    verdict = "TP"
                    notes = f"Matched alert at t={matched_ts:.2f}s (latency={seq_latency:.2f}s)."
                else:
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
    obs_mean_lat = float(np.mean(observed_latencies)) if observed_latencies else None

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
        dataset_type=dataset_type,
        report_title=report_title,
        observed_fall_alerts=observed_fall_alerts,
        observed_mean_latency_s=obs_mean_lat,
    )
