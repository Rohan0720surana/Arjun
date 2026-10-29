"""
Unit tests for offline temporal event-level fall evaluation (Phase 4D).

Covers:
- Temporal state machine transitions: normal -> transition -> fallen -> confirmed alert
- Transient posture rejection (recovery back to normal suppresses alerts)
- Tracking gap handling (gap > max_gap_s resets state)
- Alert deduplication (single alert per fall event)
- Prediction-independent 2.0s observation eligibility rules and N/A metric handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.event_evaluation import (
    FallAlertEvent,
    TemporalFallStateMachine,
    evaluate_temporal_state_machine_on_dataset,
)


def test_state_machine_normal_posture_emits_zero_alerts():
    """Continuous normal predictions never emit a fall alert."""
    fsm = TemporalFallStateMachine(required_confirmation_s=2.0)
    alerts: list[FallAlertEvent] = []

    for i in range(100):
        ts = i * 0.033
        evt = fsm.update(timestamp=ts, predicted_label="normal", sequence_id="test_norm")
        if evt:
            alerts.append(evt)

    assert len(alerts) == 0
    assert fsm.state == "NORMAL"


def test_state_machine_confirmed_fall_triggers_at_confirmation_duration():
    """A fall transition followed by sustained fallen posture triggers an alert exactly at confirmation_s."""
    fsm = TemporalFallStateMachine(required_confirmation_s=2.0)
    alerts: list[FallAlertEvent] = []

    # t=0.0s to 1.0s: normal
    for i in range(30):
        ts = i * 0.033
        fsm.update(ts, "normal", "test_fall")

    # t=1.0s to 1.5s: fall_transition
    for i in range(30, 45):
        ts = i * 0.033
        fsm.update(ts, "fall_transition", "test_fall")

    # t=1.5s: first fallen frame
    fallen_start_ts = 45 * 0.033

    # t=1.5s to 4.0s: continuous fallen
    for i in range(45, 120):
        ts = i * 0.033
        evt = fsm.update(ts, "fallen", "test_fall")
        if evt:
            alerts.append(evt)

    assert len(alerts) == 1
    alt = alerts[0]
    assert alt.first_fallen_timestamp == pytest.approx(fallen_start_ts, abs=1e-3)
    assert alt.alert_timestamp >= fallen_start_ts + 2.0
    assert alt.confirmation_duration_s >= 2.0
    assert alt.sequence_id == "test_fall"


def test_state_machine_transient_bend_recovers_without_alert():
    """A brief fallen/transition prediction (e.g. tying shoes) that recovers to normal emits no alert."""
    fsm = TemporalFallStateMachine(required_confirmation_s=2.0, recovery_window_s=0.3)
    alerts: list[FallAlertEvent] = []

    # Normal upright
    for i in range(20):
        fsm.update(i * 0.033, "normal")

    # Deep bend: 6 frames of fall_transition and fallen (~200ms)
    t_bend = 20 * 0.033
    fsm.update(t_bend, "fall_transition")
    fsm.update(t_bend + 0.033, "fallen")
    fsm.update(t_bend + 0.066, "fallen")
    fsm.update(t_bend + 0.100, "fallen")

    # Recover back to normal for 20 frames
    for i in range(25, 45):
        evt = fsm.update(i * 0.033, "normal")
        if evt:
            alerts.append(evt)

    assert len(alerts) == 0
    assert fsm.state == "NORMAL"


def test_state_machine_tracking_gap_resets_fallen_timer():
    """A tracking gap exceeding max_tracking_gap_s resets fallen timer, preventing false alert."""
    fsm = TemporalFallStateMachine(required_confirmation_s=2.0, max_tracking_gap_s=0.8)
    alerts: list[FallAlertEvent] = []

    # t=0.0s: falls down
    fsm.update(0.0, "fallen")
    fsm.update(0.5, "fallen")

    # Tracking gap: next frame is at t=2.0s (> 0.8s gap)
    evt1 = fsm.update(2.0, "fallen")
    if evt1:
        alerts.append(evt1)

    # Even though 2.0s have elapsed from t=0, the gap reset the timer at t=2.0s
    assert len(alerts) == 0
    assert fsm.first_fallen_ts == 2.0

    # Dwell with normal continuous frames until t=4.1s (2.1s after gap) -> alert triggers
    evt2 = None
    for t in np.linspace(2.05, 4.15, 20):
        evt = fsm.update(float(t), "fallen")
        if evt:
            evt2 = evt
    assert evt2 is not None


def test_insufficient_observation_preserves_na_primary_recall():
    """Clips with < 2.0s of post-fallen observation are marked INSUFFICIENT_OBSERVATION and yield N/A recall."""
    rows = []
    # Clip 1: Fall clip ending only 1.2s post-fallen
    for i in range(60):
        ts = i * 0.033
        label = "normal" if ts < 1.0 else ("fall_transition" if ts < 1.5 else "fallen")
        rows.append({
            "sequence_id": "short_fall_01",
            "track_id": 1,
            "timestamp": ts,
            "rf_pred": label,
            "rf_prob_fallen": 0.95 if label == "fallen" else 0.01,
        })
    # Clip 2: Pure ADL clip (5.0s, normal)
    for i in range(150):
        ts = i * 0.033
        rows.append({
            "sequence_id": "adl_01",
            "track_id": 1,
            "timestamp": ts,
            "rf_pred": "normal",
            "rf_prob_fallen": 0.01,
        })

    df = pd.DataFrame(rows)
    meta = {
        "short_fall_01": {"is_fall": True, "fallen_s": 1.5, "dur_s": 2.7},  # 1.2s post-fallen (< 2.0s)
        "adl_01": {"is_fall": False, "fallen_s": None, "dur_s": 5.0},
    }

    summary = evaluate_temporal_state_machine_on_dataset(
        oof_df=df,
        pred_col="rf_pred",
        prob_col="rf_prob_fallen",
        detector_name="Test Detector",
        required_confirmation_s=2.0,
        metadata_map=meta,
    )

    assert summary.total_sequences == 2
    assert summary.insufficient_obs_falls == 1
    assert summary.eligible_falls_primary == 0
    assert summary.primary_recall is None  # N/A undefined
    assert summary.primary_f1 is None      # N/A undefined
    assert summary.specificity == 1.0      # 1/1 TN on ADL
    assert summary.adl_false_alerts == 0
