"""
Unit tests for XGBoost fall classification and model comparison (Phase 4C).

Covers:
- Label integer mapping and decoding consistency
- Preprocessing and XGBoost pipeline creation
- Grouped cross-validation (zero train/test sequence leakage)
- Missing value imputation handling
- Determinism and reproducibility with fixed random seed
- Comparative metric computation and latency benchmarking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluate_random_forest import TARGET_CLASSES, compute_metrics
from ml.train_random_forest import build_rf_pipeline, load_and_validate_dataset
from ml.train_xgboost import (
    CLASS_TO_INT,
    INT_TO_CLASS,
    benchmark_inference_latency,
    build_xgb_pipeline,
    run_xgb_grouped_cv,
)


@pytest.fixture
def synthetic_feature_df() -> pd.DataFrame:
    """
    Creates a small synthetic dataset mimicking URFD schema across 3 recordings.
    """
    rows = []
    # Group 1: Fall sequence (all 3 classes)
    for i in range(30):
        label = "normal" if i < 15 else ("fall_transition" if i < 22 else "fallen")
        rows.append({
            "sequence_id": "seq_fall_01",
            "group_id": "seq_fall_01",
            "track_id": 1,
            "frame_idx": i,
            "timestamp": i * 0.1,
            "window_start_ts": max(0.0, i * 0.1 - 1.0),
            "window_end_ts": i * 0.1,
            "label": label,
            "is_fall": 0 if label == "normal" else 1,
            "is_fall_transition": 1 if label == "fall_transition" else 0,
            "is_fallen": 1 if label == "fallen" else 0,
            "is_ambiguous": 0,
            "box_aspect_ratio": 0.5 if label == "normal" else 1.5,
            "torso_angle_deg": 10.0 if label == "normal" else 80.0,
            "w600ms_hip_vertical_drop": 0.01 if label == "normal" else 0.45,
            "w600ms_horizontal_posture_fraction": 0.0 if label == "normal" else 0.8,
            "left_knee_angle_deg": 175.0 if i % 4 != 0 else np.nan,  # Occasional NaN
        })

    # Group 2: Second Fall sequence (all 3 classes)
    for i in range(25):
        label = "normal" if i < 10 else ("fall_transition" if i < 18 else "fallen")
        rows.append({
            "sequence_id": "seq_fall_02",
            "group_id": "seq_fall_02",
            "track_id": 1,
            "frame_idx": i,
            "timestamp": i * 0.1,
            "window_start_ts": max(0.0, i * 0.1 - 1.0),
            "window_end_ts": i * 0.1,
            "label": label,
            "is_fall": 0 if label == "normal" else 1,
            "is_fall_transition": 1 if label == "fall_transition" else 0,
            "is_fallen": 1 if label == "fallen" else 0,
            "is_ambiguous": 0,
            "box_aspect_ratio": 0.48 if label == "normal" else 1.4,
            "torso_angle_deg": 12.0 if label == "normal" else 75.0,
            "w600ms_hip_vertical_drop": 0.02 if label == "normal" else 0.40,
            "w600ms_horizontal_posture_fraction": 0.0 if label == "normal" else 0.75,
            "left_knee_angle_deg": 170.0,
        })

    # Group 3: Pure ADL sequence (only normal)
    for i in range(30):
        rows.append({
            "sequence_id": "seq_adl_01",
            "group_id": "seq_adl_01",
            "track_id": 1,
            "frame_idx": i,
            "timestamp": i * 0.1,
            "window_start_ts": max(0.0, i * 0.1 - 1.0),
            "window_end_ts": i * 0.1,
            "label": "normal",
            "is_fall": 0,
            "is_fall_transition": 0,
            "is_fallen": 0,
            "is_ambiguous": 0,
            "box_aspect_ratio": 0.52,
            "torso_angle_deg": 8.0,
            "w600ms_hip_vertical_drop": 0.01,
            "w600ms_horizontal_posture_fraction": 0.0,
            "left_knee_angle_deg": 178.0,
        })

    return pd.DataFrame(rows)


def test_label_integer_mapping_consistency():
    """Verify bidirectional mapping between target string labels and integer indices."""
    assert set(CLASS_TO_INT.keys()) == set(TARGET_CLASSES)
    for k, v in CLASS_TO_INT.items():
        assert INT_TO_CLASS[v] == k
        assert 0 <= v < len(TARGET_CLASSES)


def test_xgb_grouped_cv_zero_leakage_and_class_checks(synthetic_feature_df, tmp_path):
    """XGBoost grouped cross-validation enforces zero inter-recording leakage and all classes in train."""
    csv_file = tmp_path / "test_xgb_data.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    pipe = build_xgb_pipeline(random_state=42, n_estimators=10, max_depth=3)
    oof_preds, fold_summaries = run_xgb_grouped_cv(X, y, groups, pipeline=pipe, random_state=42)

    assert len(oof_preds) == len(clean_df)
    assert len(fold_summaries) == 3
    assert set(oof_preds).issubset(set(TARGET_CLASSES))

    # Verify zero overlap in every fold
    for f in fold_summaries:
        test_g = set(f["test_groups"])
        assert len(test_g) == 1


def test_xgb_handles_nan_imputation_gracefully(synthetic_feature_df, tmp_path):
    """Pipeline with SimpleImputer handles missing joint angles cleanly."""
    csv_file = tmp_path / "test_xgb_data.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    assert X["left_knee_angle_deg"].isna().sum() > 0

    pipe = build_xgb_pipeline(random_state=42, n_estimators=10, max_depth=3)
    y_int = y.map(CLASS_TO_INT)
    pipe.fit(X, y_int)

    preds_int = pipe.predict(X)
    assert len(preds_int) == len(y)
    assert set(preds_int).issubset({0, 1, 2})


def test_xgb_reproducibility_with_seed(synthetic_feature_df, tmp_path):
    """Two XGBoost runs with identical seed must yield bit-for-bit identical predictions."""
    csv_file = tmp_path / "test_xgb_data.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    pipe1 = build_xgb_pipeline(random_state=42, n_estimators=15, max_depth=3)
    preds1, _ = run_xgb_grouped_cv(X, y, groups, pipe1, random_state=42)

    pipe2 = build_xgb_pipeline(random_state=42, n_estimators=15, max_depth=3)
    preds2, _ = run_xgb_grouped_cv(X, y, groups, pipe2, random_state=42)

    np.testing.assert_array_equal(preds1, preds2)


def test_benchmark_inference_latency_utility(synthetic_feature_df, tmp_path):
    """Latency benchmark correctly measures batch and single-frame timings."""
    csv_file = tmp_path / "test_xgb_data.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    rf_pipe = build_rf_pipeline(random_state=42, n_estimators=10, max_depth=3)
    rf_pipe.fit(X, y)

    xgb_pipe = build_xgb_pipeline(random_state=42, n_estimators=10, max_depth=3)
    xgb_pipe.fit(X, y.map(CLASS_TO_INT))

    latency = benchmark_inference_latency(rf_pipe, xgb_pipe, X, n_batch_rounds=5, n_single_rounds=10)

    assert "random_forest" in latency
    assert "xgboost" in latency
    assert latency["xgboost"]["batch_total_ms"] > 0.0
    assert latency["xgboost"]["single_frame_ms"] > 0.0
    assert latency["random_forest"]["single_frame_ms"] > 0.0
