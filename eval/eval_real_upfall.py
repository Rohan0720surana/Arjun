"""
Evaluation of Rule-Based, Random Forest, and XGBoost on UP-Fall Recording
Subject1Activity1Trial1Camera1 (Phase 5A).
"""

import sys
from pathlib import Path
from collections import Counter
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.adapters import UPFallDatasetAdapter
from eval.urfd_batch_evaluation import evaluate_sequence_with_diagnostics
from ml.features import extract_features_for_sequence
from ml.train_xgboost import INT_TO_CLASS
from ml.event_evaluation import TemporalFallStateMachine

def main():
    adapter = UPFallDatasetAdapter(root_dir="data/up_fall", cache_dir="eval/cache/up_fall")
    seq = adapter.load_sequence("Subject1Activity1Trial1Camera1")
    ann = seq.annotation

    print("=====================================================================")
    print("ARJUN Campus Guardian — UP-Fall Real Evaluation (Phase 5A)")
    print("=====================================================================")
    print(f"Recording ID: {ann.sequence_id}")
    print(f"Total Frames: {len(seq.frames)}")
    print(f"Nominal FPS: {seq.fps:.2f}")
    print(f"Total Duration: {ann.duration_s:.3f}s")
    print(f"Ground Truth:")
    print(f"  Fall Transition Start: {ann.fall_transition_start_s:.3f}s")
    print(f"  Fallen State Start:     {ann.fallen_state_start_s:.3f}s")
    print(f"  Post-Fallen Dwell:      {ann.duration_s - ann.fallen_state_start_s:.3f}s")
    print(f"  Confirmation Window:    2.000s")

    # 1. Detection Audit & Tracking Continuity
    print("\n---------------------------------------------------------------------")
    print("1. Detection Audit & Track Continuity")
    print("---------------------------------------------------------------------")
    det_counts = [len(f.detections) for f in seq.frames]
    print(f"Frames with 0 detections: {det_counts.count(0)}")
    print(f"Frames with 1 detection:  {det_counts.count(1)}")
    print(f"Frames with >1 detection: {sum(1 for c in det_counts if c > 1)}")

    all_dets = [(f.frame_idx, f.timestamp, d.track_id, d.conf, d.box) for f in seq.frames for d in f.detections]
    track_counts = Counter(d[2] for d in all_dets)
    print("\nTracks present in sequence:")
    for tid, count in track_counts.most_common():
        sub_frames = [d[0] for d in all_dets if d[2] == tid]
        sub_ts = [d[1] for d in all_dets if d[2] == tid]
        gaps = [sub_frames[i+1] - sub_frames[i] for i in range(len(sub_frames)-1)]
        max_gap = max(gaps) if gaps else 0
        print(f"  Track {tid}: {count} frames ({sub_frames[0]} to {sub_frames[-1]}), "
              f"time: {sub_ts[0]:.3f}s to {sub_ts[-1]:.3f}s, max frame gap: {max_gap}")

    # Primary subject track selection (largest frame coverage without using labels)
    primary_track_id = track_counts.most_common(1)[0][0]
    print(f"\nPrimary subject track (unsupervised, max frame coverage): Track {primary_track_id} "
          f"({track_counts[primary_track_id]} frames, {track_counts[primary_track_id]/len(seq.frames)*100:.1f}%)")

    # 2. Rule-Based Fall Detector Evaluation
    print("\n---------------------------------------------------------------------")
    print("2. Production Rule-Based FallDetector Evaluation")
    print("---------------------------------------------------------------------")
    diag = evaluate_sequence_with_diagnostics(seq)
    print(f"Raw drop triggers detected: {len(diag.raw_drop_timestamps)}")
    for i, ts in enumerate(diag.raw_drop_timestamps):
        print(f"  Drop trigger {i+1}: at t={ts:.3f}s (latency relative to transition start: {ts - ann.fall_transition_start_s:+.3f}s)")
    print(f"Confirmed alerts emitted:   {len(diag.confirmed_alerts)}")
    for i, ts in enumerate(diag.alert_timestamps):
        print(f"  Confirmed alert {i+1}:   at t={ts:.3f}s (latency relative to fallen start: {ts - ann.fallen_state_start_s:+.3f}s)")

    # 3. ML Feature Extraction
    print("\n---------------------------------------------------------------------")
    print("3. ML Pipeline Evaluation (Random Forest & XGBoost)")
    print("---------------------------------------------------------------------")
    raw_rows = extract_features_for_sequence(seq)
    df_feat = pd.DataFrame(raw_rows)
    print(f"Extracted feature rows across all tracks: {len(df_feat)}")

    rf_path = Path("ml/models/random_forest_baseline.joblib")
    xgb_path = Path("ml/models/xgboost_baseline.joblib")

    if not rf_path.is_file():
        print("[!] Random Forest model artifact NOT found: ml/models/random_forest_baseline.joblib")
        rf = None
    else:
        rf = joblib.load(rf_path)
        print(f"Loaded Random Forest model artifact ({rf_path.stat().st_size} bytes)")

    if not xgb_path.is_file():
        print("[!] XGBoost model artifact NOT found: ml/models/xgboost_baseline.joblib")
        xgb = None
    else:
        xgb = joblib.load(xgb_path)
        print(f"Loaded XGBoost model artifact ({xgb_path.stat().st_size} bytes)")

    if rf is not None and xgb is not None:
        feature_cols = list(rf.feature_names_in_)
        X = df_feat[feature_cols]

        # RF Inferences
        rf_preds = rf.predict(X)
        rf_probs = rf.predict_proba(X)
        df_feat["rf_pred"] = rf_preds
        fallen_idx_rf = list(rf.classes_).index("fallen")
        df_feat["rf_prob_fallen"] = rf_probs[:, fallen_idx_rf]

        # XGB Inferences
        xgb_pred_ints = xgb.predict(X)
        xgb_probs = xgb.predict_proba(X)
        df_feat["xgb_pred"] = [INT_TO_CLASS[p] for p in xgb_pred_ints]
        df_feat["xgb_prob_fallen"] = xgb_probs[:, 2]

        print(f"\nOverall Frame Predictions (all detections):")
        print(f"  Random Forest : {dict(df_feat['rf_pred'].value_counts())}")
        print(f"  XGBoost       : {dict(df_feat['xgb_pred'].value_counts())}")

        # Evaluate state machine on each individual track and across combined subject tracks
        print("\n--- Event State Machine (2.0s Sustained-Down Confirmation) ---")
        for tid in [1, 2]:
            sub = df_feat[df_feat["track_id"] == tid].sort_values("timestamp")
            if sub.empty:
                continue

            print(f"\nEvaluation on Track {tid} ({len(sub)} frames, t={sub['timestamp'].iloc[0]:.3f}s to {sub['timestamp'].iloc[-1]:.3f}s):")
            print(f"  RF Class breakdown : {dict(sub['rf_pred'].value_counts())}")
            print(f"  XGB Class breakdown: {dict(sub['xgb_pred'].value_counts())}")

            # RF Temporal Confirmation
            sm_rf = TemporalFallStateMachine(required_confirmation_s=2.0)
            rf_event = None
            for _, r in sub.iterrows():
                ev = sm_rf.update(r["timestamp"], r["rf_pred"], track_id=tid, prob_fallen=r["rf_prob_fallen"])
                if ev and not rf_event:
                    rf_event = ev

            # XGB Temporal Confirmation
            sm_xgb = TemporalFallStateMachine(required_confirmation_s=2.0)
            xgb_event = None
            for _, r in sub.iterrows():
                ev = sm_xgb.update(r["timestamp"], r["xgb_pred"], track_id=tid, prob_fallen=r["xgb_prob_fallen"])
                if ev and not xgb_event:
                    xgb_event = ev

            if rf_event:
                lat = rf_event.alert_timestamp - ann.fallen_state_start_s
                print(f"  [ALERT] Random Forest Event Alert emitted at t={rf_event.alert_timestamp:.3f}s "
                      f"(first fallen: {rf_event.first_fallen_timestamp:.3f}s, latency relative to fallen start: {lat:+.3f}s)")
            else:
                print(f"  [-] Random Forest: No confirmed event alert emitted on Track {tid}")

            if xgb_event:
                lat = xgb_event.alert_timestamp - ann.fallen_state_start_s
                print(f"  [ALERT] XGBoost Event Alert emitted at t={xgb_event.alert_timestamp:.3f}s "
                      f"(first fallen: {xgb_event.first_fallen_timestamp:.3f}s, latency relative to fallen start: {lat:+.3f}s)")
            else:
                print(f"  [-] XGBoost: No confirmed event alert emitted on Track {tid}")

if __name__ == "__main__":
    main()
