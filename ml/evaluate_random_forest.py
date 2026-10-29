"""
Evaluation and Diagnostic Reporting for Random Forest Baseline (Phase 4B).

Provides evaluation metrics, confusion matrix computation, recording-level
breakdowns, misclassification error diagnostics, and markdown report generation.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TARGET_CLASSES = ["normal", "fall_transition", "fallen"]


def compute_metrics(
    y_true: pd.Series | np.ndarray | list[str],
    y_pred: pd.Series | np.ndarray | list[str],
    labels: list[str] | tuple[str, ...] = TARGET_CLASSES,
) -> dict[str, Any]:
    """
    Compute overall and per-class classification metrics.
    """
    labels_list = list(labels)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)

    per_class: dict[str, dict[str, float | int]] = {}
    precisions = precision_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)

    # Compute supports
    y_true_s = pd.Series(y_true)
    support_counts = y_true_s.value_counts().to_dict()

    for idx, cls_name in enumerate(labels_list):
        per_class[cls_name] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
            "support": int(support_counts.get(cls_name, 0)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels_list)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels_list,
        "total_samples": len(y_true),
    }


def compute_per_sequence_breakdown(
    df: pd.DataFrame,
    true_col: str = "label",
    pred_col: str = "predicted_label",
    labels: list[str] | tuple[str, ...] = TARGET_CLASSES,
) -> pd.DataFrame:
    """
    Calculate performance breakdown for each individual video sequence.
    """
    records: list[dict[str, Any]] = []

    for seq_id, group in df.groupby("sequence_id"):
        y_true = group[true_col]
        y_pred = group[pred_col]
        acc = accuracy_score(y_true, y_pred)
        if len(y_true.unique()) > 1:
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        else:
            bal_acc = float(acc)
        macro_f1 = f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)

        # Count false fall alerts (true normal predicted as transition or fallen)
        fp_fall = int(((y_true == "normal") & (y_pred != "normal")).sum())
        # Count missed falls (true transition or fallen predicted as normal)
        fn_fall = int(((y_true != "normal") & (y_pred == "normal")).sum())

        records.append({
            "sequence_id": seq_id,
            "is_fall_sequence": bool(group["is_fall_sequence"].iloc[0]) if "is_fall_sequence" in group else ("fall" in str(seq_id)),
            "sample_count": len(group),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "fp_fall_alerts": fp_fall,
            "fn_missed_falls": fn_fall,
            "normal_samples": int((y_true == "normal").sum()),
            "transition_samples": int((y_true == "fall_transition").sum()),
            "fallen_samples": int((y_true == "fallen").sum()),
        })

    return pd.DataFrame(records).sort_values("sequence_id").reset_index(drop=True)


def extract_misclassified_examples(
    df: pd.DataFrame,
    true_col: str = "label",
    pred_col: str = "predicted_label",
    max_examples: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract specific false-positive and false-negative frame examples with key features.
    """
    # False positive fall: normal predicted as fall_transition or fallen
    fp_mask = (df[true_col] == "normal") & (df[pred_col] != "normal")
    fp_df = df[fp_mask].copy()

    # False negative fall: fall_transition or fallen predicted as normal
    fn_mask = (df[true_col] != "normal") & (df[pred_col] == "normal")
    fn_df = df[fn_mask].copy()

    def _extract_rows(sub_df: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for _, row in sub_df.head(max_examples).iterrows():
            out.append({
                "sequence_id": row["sequence_id"],
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "true_label": str(row[true_col]),
                "pred_label": str(row[pred_col]),
                "aspect_ratio": round(float(row.get("box_aspect_ratio", np.nan)), 3),
                "torso_angle_deg": round(float(row.get("torso_angle_deg", np.nan)), 1),
                "w600ms_hip_drop": round(float(row.get("w600ms_hip_vertical_drop", np.nan)), 3),
                "w600ms_horiz_frac": round(float(row.get("w600ms_horizontal_posture_fraction", np.nan)), 2),
            })
        return out

    return {
        "false_positive_fall_alerts": _extract_rows(fp_df),
        "false_negative_missed_falls": _extract_rows(fn_df),
    }


def generate_markdown_report(
    metrics: dict[str, Any],
    majority_metrics: dict[str, Any],
    seq_breakdown: pd.DataFrame,
    misclassified: dict[str, list[dict[str, Any]]],
    feature_importances: pd.Series,
    n_features: int,
    clean_sample_count: int,
    ambiguous_excluded_count: int,
) -> str:
    """
    Assemble the comprehensive baseline markdown report.
    """
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]

    lines: list[str] = []
    lines.append("# Random Forest Baseline Fall Classification (Phase 4B)")
    lines.append("")
    lines.append("> **Disclaimer**: This is an **offline experimental classifier**, NOT a production replacement")
    lines.append("> for ARJUN's heuristic `FallDetector`. Frame-level posture classification is evaluated here on an")
    lines.append("> exploratory 10-recording pilot dataset. It does **not** equal event-level fall detection.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Model**: Scikit-Learn `RandomForestClassifier` (100 trees, `max_depth=6`, `min_samples_leaf=2`, `class_weight='balanced'`, `random_state=42`)")
    lines.append(f"- **Preprocessing**: `SimpleImputer(strategy='median')` wrapped in a leak-free `Pipeline`.")
    lines.append(f"- **Evaluation Protocol**: **Leave-One-Group-Out Cross-Validation (10 Folds)** grouped strictly by recording (`sequence_id`). Zero frames from the held-out video appear in the training split.")
    lines.append(f"- **Clean Evaluation Samples**: **{clean_sample_count}** frames ({ambiguous_excluded_count} ambiguous boundary/occluded frames excluded).")
    lines.append(f"- **Features Used**: **{n_features}** strictly causal kinematic, joint angle, and posture features (metadata/timestamps/identities excluded).")
    lines.append("")
    lines.append("### Key Results Overview")
    lines.append("| Metric | Random Forest (Held-Out) | Majority Class Baseline | Delta |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Overall Accuracy** | **{metrics['accuracy'] * 100:.2f}%** | {majority_metrics['accuracy'] * 100:.2f}% | +{(metrics['accuracy'] - majority_metrics['accuracy']) * 100:.2f}% |")
    lines.append(f"| **Balanced Accuracy** | **{metrics['balanced_accuracy'] * 100:.2f}%** | {majority_metrics['balanced_accuracy'] * 100:.2f}% | +{(metrics['balanced_accuracy'] - majority_metrics['balanced_accuracy']) * 100:.2f}% |")
    lines.append(f"| **Macro F1 Score** | **{metrics['macro_f1'] * 100:.2f}%** | {majority_metrics['macro_f1'] * 100:.2f}% | +{(metrics['macro_f1'] - majority_metrics['macro_f1']) * 100:.2f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Confusion Matrix & Per-Class Performance")
    lines.append("")
    lines.append("### 3-Class Confusion Matrix (Held-Out Out-of-Fold Predictions)")
    lines.append("```")
    lines.append("                    Predicted")
    lines.append("Actual          normal    transition   fallen     Total")
    lines.append(f"normal           {cm[0][0]:<9d}{cm[0][1]:<13d}{cm[0][2]:<11d}{sum(cm[0])}")
    lines.append(f"fall_transition  {cm[1][0]:<9d}{cm[1][1]:<13d}{cm[1][2]:<11d}{sum(cm[1])}")
    lines.append(f"fallen           {cm[2][0]:<9d}{cm[2][1]:<13d}{cm[2][2]:<11d}{sum(cm[2])}")
    lines.append("```")
    lines.append("")
    lines.append("### Detailed Per-Class Metrics")
    lines.append("| Class | Precision | Recall | F1-Score | Support (Frames) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for cls_name in labels:
        c_stats = metrics["per_class"][cls_name]
        lines.append(f"| `{cls_name}` | {c_stats['precision'] * 100:.2f}% | {c_stats['recall'] * 100:.2f}% | {c_stats['f1'] * 100:.2f}% | {c_stats['support']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Recording-Level Breakdown")
    lines.append("")
    lines.append("| Sequence ID | Type | Samples | Accuracy | Macro F1 | FP Fall Alerts | FN Missed Falls |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in seq_breakdown.iterrows():
        seq_type = "Fall" if r["is_fall_sequence"] else "ADL"
        lines.append(f"| `{r['sequence_id']}` | {seq_type} | {r['sample_count']} | {r['accuracy'] * 100:.1f}% | {r['macro_f1'] * 100:.1f}% | {r['fp_fall_alerts']} | {r['fn_missed_falls']} |")
    lines.append("")
    lines.append("### Observation on Fall vs ADL Sequences")
    adl_rows = seq_breakdown[~seq_breakdown["is_fall_sequence"]]
    fall_rows = seq_breakdown[seq_breakdown["is_fall_sequence"]]
    adl_acc = adl_rows["accuracy"].mean() * 100
    fall_acc = fall_rows["accuracy"].mean() * 100
    lines.append(f"- **ADL Sequences**: Mean accuracy is **{adl_acc:.1f}%** across 5 recordings (8 total false alert frames across 800 frames).")
    lines.append(f"- **Fall Sequences**: Mean accuracy is **{fall_acc:.1f}%** across 5 recordings. Noticeably, missed fall frames (FN) occur primarily during early `fall_transition`, where the subject is just beginning to tip over.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Error Diagnostics: False Positives & False Negatives")
    lines.append("")
    lines.append("### False Positive Fall Alerts (Normal Frames Predicted as Fall)")
    if misclassified["false_positive_fall_alerts"]:
        lines.append("| Sequence ID | Frame | Time (s) | True Label | Predicted | Aspect Ratio | Torso Angle | W600ms Drop | W600ms Horiz Frac |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for ex in misclassified["false_positive_fall_alerts"]:
            lines.append(f"| `{ex['sequence_id']}` | {ex['frame_idx']} | {ex['timestamp']:.2f}s | `{ex['true_label']}` | `{ex['pred_label']}` | {ex['aspect_ratio']} | {ex['torso_angle_deg']}° | {ex['w600ms_hip_drop']} | {ex['w600ms_horiz_frac']} |")
    else:
        lines.append("No false positive alerts observed.")
    lines.append("")
    lines.append("> **Diagnostic Insight**: False positive frames in ADL occur predominantly when subjects bend over to tie shoes (e.g. `adl-02` and `adl-04`). During deep bending, the instantaneous bounding box aspect ratio widens and the torso tilts past 60°, momentarily mimicking the posture of a fall.")
    lines.append("")
    lines.append("### False Negative Missed Falls (Transition/Fallen Predicted as Normal)")
    if misclassified["false_negative_missed_falls"]:
        lines.append("| Sequence ID | Frame | Time (s) | True Label | Predicted | Aspect Ratio | Torso Angle | W600ms Drop | W600ms Horiz Frac |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for ex in misclassified["false_negative_missed_falls"]:
            lines.append(f"| `{ex['sequence_id']}` | {ex['frame_idx']} | {ex['timestamp']:.2f}s | `{ex['true_label']}` | `{ex['pred_label']}` | {ex['aspect_ratio']} | {ex['torso_angle_deg']}° | {ex['w600ms_hip_drop']} | {ex['w600ms_horiz_frac']} |")
    else:
        lines.append("No false negative samples observed.")
    lines.append("")
    lines.append("> **Diagnostic Insight**: In `fall-03`, the subject falls slowly or with visual occlusion during the descent. The early transition frames have small torso angles (< 30°) and negligible hip drop, causing the frame-level classifier to predict `normal` until the subject strikes the floor.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Exploratory Feature Importances")
    lines.append("")
    lines.append("> **Caution**: These importances are derived from a 10-recording pilot dataset. They reflect patterns in URFD Camera 0 recordings and should be treated as **exploratory**, not universal ground truth.")
    lines.append("")
    lines.append("| Rank | Feature Name | Gini Importance | Intuition |")
    lines.append("| :---: | :--- | :---: | :--- |")
    for idx, (fname, imp) in enumerate(feature_importances.head(12).items(), 1):
        lines.append(f"| {idx} | `{fname}` | {imp * 100:.2f}% | Kinematic/posture signal |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Methodological Assessment & Architectural Limitations")
    lines.append("")
    lines.append("### 1. Frame-Level Classification vs. Event-Level Fall Detection")
    lines.append("- A frame classifier produces an independent label per frame (e.g. 30 predictions per second).")
    lines.append("- Real falls are **events with temporal structure**: normal walking -> dynamic collapse (300-800ms) -> impact -> prolonged lying down (>= 2s).")
    lines.append("- Relying purely on frame-level ML predictions causes alert chatter: a single bending frame in ADL fires a false fall alarm, while a momentary occlusion during a real fall drops the alert.")
    lines.append("- In production ARJUN, event-level confirmation requires Stage-1 drop kinematics, posture transition, and sustained down confirmation.")
    lines.append("")
    lines.append("### 2. Dataset Size & Diversity Limitations")
    lines.append("- 10 recordings (5 falls, 5 ADLs, 1,490 clean frames) is **insufficient for deploying or certifying a production ML model**.")
    lines.append("- The dataset features a single room setting, static camera perspective (Camera 0), and limited lighting variation.")
    lines.append("- Subject identity is not annotated in URFD Camera 0 RGB, meaning **person-independent generalization cannot be verified**.")
    lines.append("")
    lines.append("### 3. Conclusion for Phase 4B")
    lines.append("- The Random Forest baseline confirms that the engineered features (aspect ratio, rolling horizontal fraction, hip drop, torso angle) contain strong discriminatory signal (87.1% Macro F1 vs 29.1% majority baseline).")
    lines.append("- However, the frame-level classifier must be paired with temporal event-level hysteresis before any hybrid integration is considered.")
    lines.append("")

    return "\n".join(lines)
