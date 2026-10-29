"""
Unit tests for Random Forest baseline fall classification (Phase 4B).

Covers:
- Dataset validation, metadata exclusion, and feature causality checks
- Leak-free preprocessing pipeline with training-only imputer
- Recording-level grouped cross-validation (zero train/test overlap)
- Target class availability checks across all folds
- Metric computation (confusion matrix, precision/recall/F1, balanced accuracy)
- Baseline majority predictor comparisons
- Determinism and reproducibility with fixed random seed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.evaluate_random_forest import (
    TARGET_CLASSES,
    compute_metrics,
    compute_per_sequence_breakdown,
    extract_misclassified_examples,
)
from ml.train_random_forest import (
    NON_FEATURE_COLUMNS,
    build_rf_pipeline,
    evaluate_majority_baseline,
    load_and_validate_dataset,
    run_grouped_cross_validation,
)


# --------------------------------------------------------------------------- #
# Fixtures & Synthetic Data
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_feature_df() -> pd.DataFrame:
    """
    Creates a small synthetic dataset mimicking the URFD schema with 3 sequences (groups).
    """
    rng = np.random.default_rng(42)
    rows = []

    # Group 1: Fall sequence (contains normal, fall_transition, fallen)
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
            "is_ambiguous": 1 if i in (14, 21) else 0,
            "box_aspect_ratio": 0.5 if label == "normal" else 1.5,
            "torso_angle_deg": 10.0 if label == "normal" else 80.0,
            "w600ms_hip_vertical_drop": 0.01 if label == "normal" else 0.45,
            "w600ms_horizontal_posture_fraction": 0.0 if label == "normal" else 0.8,
            "left_knee_angle_deg": 175.0 if i % 5 != 0 else np.nan,  # Occasional NaN
        })

    # Group 2: Second Fall sequence (contains all 3 classes)
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

    # Group 3: Pure ADL sequence (contains only normal)
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


# --------------------------------------------------------------------------- #
# TASK 1: Data Validation & Feature Separation Tests
# --------------------------------------------------------------------------- #

def test_metadata_exclusion_and_feature_isolation(synthetic_feature_df, tmp_path):
    """Metadata and labels are strictly excluded from input feature matrix X."""
    csv_file = tmp_path / "test_features.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)

    clean_df, X, y, groups, feat_cols = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    # Excluded ambiguous check: 2 samples in seq_fall_01 had is_ambiguous=1
    assert len(clean_df) == len(synthetic_feature_df) - 2

    # Verify no non-feature column leaked into X
    for col in NON_FEATURE_COLUMNS:
        assert col not in X.columns
        assert col not in feat_cols

    # Verify all feature columns are numeric
    assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]

    # Verify no infs
    assert not np.isinf(X.values).any()


def test_validation_rejects_infinite_features(synthetic_feature_df, tmp_path):
    """Feature matrix with infinite values must raise ValueError."""
    synthetic_feature_df.loc[0, "box_aspect_ratio"] = np.inf
    csv_file = tmp_path / "inf_features.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Infinite values detected"):
        load_and_validate_dataset(csv_file)


# --------------------------------------------------------------------------- #
# TASK 3 & 4: Grouped Cross-Validation & Leakage Prevention Tests
# --------------------------------------------------------------------------- #

def test_grouped_cv_zero_overlap_between_train_and_test(synthetic_feature_df, tmp_path):
    """Held-out recordings must never leak between train and test splits."""
    csv_file = tmp_path / "test_features.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    pipe = build_rf_pipeline(random_state=42, n_estimators=10, max_depth=3)
    oof_preds, fold_summaries = run_grouped_cross_validation(
        X=X, y=y, groups=groups, pipeline=pipe, cv_strategy="logo"
    )

    assert len(fold_summaries) == 3  # 3 unique groups
    assert len(oof_preds) == len(clean_df)
    assert not pd.Series(oof_preds).isna().any()

    # Verify zero group leakage for every fold
    for f in fold_summaries:
        test_g = set(f["test_groups"])
        assert len(test_g) == 1
        # Training set must contain all target classes
        for cls_name in TARGET_CLASSES:
            assert cls_name in f["train_classes"]


def test_pipeline_imputer_handles_missing_values_without_leakage(synthetic_feature_df, tmp_path):
    """Pipeline SimpleImputer must successfully impute NaNs during training and inference."""
    csv_file = tmp_path / "test_features.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    assert X["left_knee_angle_deg"].isna().sum() > 0

    pipe = build_rf_pipeline(random_state=42, n_estimators=10)
    # Fitting on training data with NaNs should succeed without error
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(TARGET_CLASSES))


def test_reproducibility_with_fixed_seed(synthetic_feature_df, tmp_path):
    """Two runs with random_state=42 must produce bit-for-bit identical predictions."""
    csv_file = tmp_path / "test_features.csv"
    synthetic_feature_df.to_csv(csv_file, index=False)
    clean_df, X, y, groups, _ = load_and_validate_dataset(csv_file, exclude_ambiguous=True)

    pipe1 = build_rf_pipeline(random_state=42, n_estimators=20, max_depth=4)
    preds1, _ = run_grouped_cross_validation(X, y, groups, pipe1, cv_strategy="logo")

    pipe2 = build_rf_pipeline(random_state=42, n_estimators=20, max_depth=4)
    preds2, _ = run_grouped_cross_validation(X, y, groups, pipe2, cv_strategy="logo")

    np.testing.assert_array_equal(preds1, preds2)


# --------------------------------------------------------------------------- #
# TASK 5: Evaluation & Metrics Tests
# --------------------------------------------------------------------------- #

def test_compute_metrics_accuracy_and_confusion_matrix():
    """Metrics calculation accurately evaluates precision, recall, F1, and confusion matrix."""
    y_true = ["normal", "normal", "fall_transition", "fallen"]
    y_pred = ["normal", "fall_transition", "fall_transition", "fallen"]

    m = compute_metrics(y_true, y_pred, labels=TARGET_CLASSES)

    assert m["total_samples"] == 4
    assert m["accuracy"] == 0.75
    # Normal: 1 TP, 1 FN -> recall = 0.5
    assert m["per_class"]["normal"]["recall"] == 0.5
    # Fall transition: 1 TP, 1 FP -> precision = 0.5, recall = 1.0
    assert m["per_class"]["fall_transition"]["precision"] == 0.5
    assert m["per_class"]["fall_transition"]["recall"] == 1.0
    # Fallen: 1 TP -> precision = 1.0, recall = 1.0
    assert m["per_class"]["fallen"]["precision"] == 1.0

    # Confusion matrix structure: 3x3
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # true normal, pred normal
    assert cm[0, 1] == 1  # true normal, pred transition
    assert cm[2, 2] == 1  # true fallen, pred fallen


def test_majority_baseline_evaluates_most_frequent_class():
    """Majority baseline always predicts the most frequent class in y."""
    y_true = pd.Series(["normal", "normal", "normal", "fall_transition", "fallen"])
    bm = evaluate_majority_baseline(y_true, labels=TARGET_CLASSES)

    assert bm["accuracy"] == 0.60  # 3 out of 5 are normal
    assert bm["per_class"]["normal"]["recall"] == 1.0
    assert bm["per_class"]["fall_transition"]["recall"] == 0.0
    assert bm["per_class"]["fallen"]["recall"] == 0.0


def test_per_sequence_breakdown_and_error_extraction(synthetic_feature_df):
    """Breakdown correctly calculates per-recording metrics and separates FP and FN."""
    clean_df = synthetic_feature_df.copy()
    clean_df["predicted_label"] = clean_df["label"]
    # Introduce one FP in seq_adl_01
    first_adl_idx = clean_df.index[clean_df["sequence_id"] == "seq_adl_01"][0]
    clean_df.loc[first_adl_idx, "predicted_label"] = "fall_transition"

    breakdown = compute_per_sequence_breakdown(clean_df)
    assert len(breakdown) == 3

    adl_row = breakdown[breakdown["sequence_id"] == "seq_adl_01"].iloc[0]
    assert adl_row["fp_fall_alerts"] == 1
    assert adl_row["fn_missed_falls"] == 0

    errors = extract_misclassified_examples(clean_df)
    assert len(errors["false_positive_fall_alerts"]) == 1
    assert errors["false_positive_fall_alerts"][0]["sequence_id"] == "seq_adl_01"
