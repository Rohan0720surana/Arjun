"""
Random Forest Baseline Fall Classifier Training and Evaluation (Phase 4B).

Builds, evaluates, and documents a 3-class Random Forest baseline on causal pose features
using recording-level grouped cross-validation (Leave-One-Group-Out and GroupKFold).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.dataset import build_urfd_ml_dataset
from ml.evaluate_random_forest import (
    TARGET_CLASSES,
    compute_metrics,
    compute_per_sequence_breakdown,
    extract_misclassified_examples,
    generate_markdown_report,
)

NON_FEATURE_COLUMNS = {
    "sample_id",
    "sequence_id",
    "group_id",
    "track_id",
    "frame_idx",
    "timestamp",
    "window_start_ts",
    "window_end_ts",
    "label",
    "is_fall",
    "is_fall_transition",
    "is_fallen",
    "is_ambiguous",
    "is_fall_sequence",
}


def load_and_validate_dataset(
    csv_path: str | Path = "ml/data/urfd_features.csv",
    exclude_ambiguous: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Load feature dataset, perform validation, and separate X, y, groups.

    Returns:
        (clean_df, X, y, groups, feature_names)
    """
    path = Path(csv_path)
    if not path.is_file():
        # Build if not already existing
        df = build_urfd_ml_dataset()
    else:
        df = pd.read_csv(path)

    # 1. Validation & Schema Verification
    if "is_ambiguous" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset missing required columns ('is_ambiguous', 'label')")

    if exclude_ambiguous:
        clean_df = df[df["is_ambiguous"] == 0].copy().reset_index(drop=True)
    else:
        clean_df = df.copy().reset_index(drop=True)

    # 2. Extract strictly causal features
    feature_cols = [c for c in clean_df.columns if c not in NON_FEATURE_COLUMNS]

    # Verify no non-numeric columns in feature set
    non_numeric = clean_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found in X: {non_numeric}")

    # Verify no infinite values
    inf_count = int(np.isinf(clean_df[feature_cols].values).sum())
    if inf_count > 0:
        raise ValueError(f"Infinite values detected in feature set ({inf_count} occurrences)")

    X = clean_df[feature_cols].copy()
    y = clean_df["label"].copy()
    groups = clean_df["group_id"].copy()

    return clean_df, X, y, groups, feature_cols


def build_rf_pipeline(
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    min_samples_leaf: int = 2,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """
    Construct leak-free preprocessing and Random Forest pipeline.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def run_grouped_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    pipeline: Pipeline | None = None,
    cv_strategy: str = "logo",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Execute grouped cross validation (Leave-One-Group-Out or GroupKFold).

    Ensures zero recording-level leakage between train and test folds,
    and validates class availability in each training fold.

    Returns:
        (oof_predictions, fold_summaries)
    """
    if pipeline is None:
        pipeline = build_rf_pipeline(random_state=random_state)

    if cv_strategy == "logo":
        cv = LeaveOneGroupOut()
        splits = cv.split(X, y, groups)
    elif cv_strategy == "group_kfold":
        cv = GroupKFold(n_splits=n_splits)
        splits = cv.split(X, y, groups)
    else:
        raise ValueError(f"Unknown cv_strategy '{cv_strategy}', expected 'logo' or 'group_kfold'")

    oof_preds = np.empty(len(X), dtype=object)
    fold_summaries: list[dict[str, Any]] = []

    unique_classes = set(TARGET_CLASSES)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(groups.iloc[train_idx].unique())
        test_groups = set(groups.iloc[test_idx].unique())

        # Leakage assertion: no group can appear in both train and test
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(f"Data leakage detected! Overlapping groups in fold {fold_idx}: {overlap}")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Check class availability in training fold
        train_classes = set(y_train.unique())
        missing_classes = unique_classes - train_classes
        if missing_classes:
            raise RuntimeError(
                f"Fold {fold_idx} training set is missing required target classes: {missing_classes}"
            )

        # Fit model on training fold only
        pipeline.fit(X_train, y_train)

        # Predict on held-out test fold
        fold_preds = pipeline.predict(X_test)
        oof_preds[test_idx] = fold_preds

        fold_summaries.append({
            "fold_idx": fold_idx,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_groups": list(test_groups),
            "train_classes": list(train_classes),
            "test_classes": list(set(y_test.unique())),
        })

    return oof_preds, fold_summaries


def evaluate_majority_baseline(
    y_true: pd.Series,
    labels: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    """
    Evaluate a naive majority-class dummy predictor.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_true), 1)), y_true)
    preds = dummy.predict(np.zeros((len(y_true), 1)))
    return compute_metrics(y_true, preds, labels=labels)


def train_and_evaluate(
    csv_path: str | Path = "ml/data/urfd_features.csv",
    output_model_path: str | Path | None = "ml/models/random_forest_baseline.joblib",
    output_report_path: str | Path | None = "ml/results/RANDOM_FOREST_BASELINE.md",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    End-to-end execution of Phase 4B baseline training, validation, and documentation.
    """
    # 1. Load & validate data
    raw_df = pd.read_csv(csv_path) if Path(csv_path).is_file() else build_urfd_ml_dataset()
    clean_df, X, y, groups, feature_cols = load_and_validate_dataset(csv_path=csv_path, exclude_ambiguous=True)

    n_raw = len(raw_df)
    n_clean = len(clean_df)
    n_ambiguous = n_raw - n_clean

    print("=" * 86)
    print(" ARJUN ML RANDOM FOREST BASELINE TRAINING & EVALUATION (Phase 4B)")
    print(" [10 URFD Sequences | Grouped Cross-Validation | Causal Features Only]")
    print("=" * 86)
    print(f"- Total raw samples       : {n_raw}")
    print(f"- Clean evaluation samples: {n_clean} (excluded {n_ambiguous} ambiguous boundary samples)")
    print(f"- Total features in X     : {len(feature_cols)}")
    print(f"- Unique groups           : {len(groups.unique())}")
    print(f"- Random State            : {random_state}")

    # 2. Run Leave-One-Group-Out CV (10 folds)
    pipeline = build_rf_pipeline(random_state=random_state)
    oof_preds, fold_summaries = run_grouped_cross_validation(
        X=X,
        y=y,
        groups=groups,
        pipeline=pipeline,
        cv_strategy="logo",
        random_state=random_state,
    )

    clean_df["predicted_label"] = oof_preds

    # 3. Compute Metrics
    rf_metrics = compute_metrics(y, oof_preds, labels=TARGET_CLASSES)
    majority_metrics = evaluate_majority_baseline(y, labels=TARGET_CLASSES)
    seq_breakdown = compute_per_sequence_breakdown(clean_df, true_col="label", pred_col="predicted_label")
    misclassified = extract_misclassified_examples(clean_df, true_col="label", pred_col="predicted_label")

    # 4. Train final exploratory model on all clean data to extract feature importances
    pipeline.fit(X, y)
    rf_model: RandomForestClassifier = pipeline.named_steps["clf"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # 5. Save Model Artifact if requested
    if output_model_path:
        out_p = Path(output_model_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out_p)
        print(f"\n[Saved Model Pipeline]: {out_p}")

    # 6. Generate Markdown Report
    report_md = generate_markdown_report(
        metrics=rf_metrics,
        majority_metrics=majority_metrics,
        seq_breakdown=seq_breakdown,
        misclassified=misclassified,
        feature_importances=importances,
        n_features=len(feature_cols),
        clean_sample_count=n_clean,
        ambiguous_excluded_count=n_ambiguous,
    )

    if output_report_path:
        rep_p = Path(output_report_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        rep_p.write_text(report_md, encoding="utf-8")
        print(f"[Saved Report]: {rep_p}")

    # 7. Print Terminal Summary
    print("\n" + "=" * 45 + " RESULTS OVERVIEW " + "=" * 45)
    print(f"Overall Accuracy    : {rf_metrics['accuracy'] * 100:.2f}%  (Majority Baseline: {majority_metrics['accuracy'] * 100:.2f}%)")
    print(f"Balanced Accuracy   : {rf_metrics['balanced_accuracy'] * 100:.2f}%  (Majority Baseline: {majority_metrics['balanced_accuracy'] * 100:.2f}%)")
    print(f"Macro F1 Score      : {rf_metrics['macro_f1'] * 100:.2f}%  (Majority Baseline: {majority_metrics['macro_f1'] * 100:.2f}%)")

    print("\nPer-Class Breakdown:")
    print(f"{'Class':<18s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s} {'Support'}")
    print("-" * 62)
    for cname in TARGET_CLASSES:
        cs = rf_metrics["per_class"][cname]
        print(f"{cname:<18s} {cs['precision']*100:>8.2f}%    {cs['recall']*100:>8.2f}%    {cs['f1']*100:>8.2f}%    {cs['support']:>6d}")

    cm = rf_metrics["confusion_matrix"]
    print("\nConfusion Matrix (Held-out Out-Of-Fold Predictions):")
    print(f"{'':<18s} {'Pred Normal':<14s} {'Pred Transition':<18s} {'Pred Fallen'}")
    print(f"{'True Normal':<18s} {cm[0][0]:<14d} {cm[0][1]:<18d} {cm[0][2]:<10d}")
    print(f"{'True Transition':<18s} {cm[1][0]:<14d} {cm[1][1]:<18d} {cm[1][2]:<10d}")
    print(f"{'True Fallen':<18s} {cm[2][0]:<14d} {cm[2][1]:<18d} {cm[2][2]:<10d}")
    print("=" * 108 + "\n")

    return {
        "rf_metrics": rf_metrics,
        "majority_metrics": majority_metrics,
        "seq_breakdown": seq_breakdown,
        "importances": importances,
        "pipeline": pipeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest baseline on URFD features.")
    parser.add_argument("--csv", default="ml/data/urfd_features.csv", help="Path to feature dataset CSV")
    parser.add_argument("--model-out", default="ml/models/random_forest_baseline.joblib", help="Output path for trained model pipeline")
    parser.add_argument("--report-out", default="ml/results/RANDOM_FOREST_BASELINE.md", help="Output markdown report path")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save model to disk")
    args = parser.parse_args()

    model_path = None if args.no_save_model else args.model_out
    train_and_evaluate(
        csv_path=args.csv,
        output_model_path=model_path,
        output_report_path=args.report_out,
    )


if __name__ == "__main__":
    main()
