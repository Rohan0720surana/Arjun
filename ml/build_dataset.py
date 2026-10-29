"""
CLI Runner for ML Feature Extraction & Dataset Building (Phase 4A).

Loads all 10 cached URFD sequences, computes causal posture and temporal features,
applies official ground-truth label segmentation, and produces a summary audit.

Usage:
    python ml/build_dataset.py
    python ml/build_dataset.py --output-csv ml/data/urfd_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.dataset import build_urfd_ml_dataset, summarize_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ML feature dataset from 10 cached URFD sequences."
    )
    parser.add_argument(
        "--urfd-dir",
        type=str,
        default="data/urfd",
        help="Path to URFD raw dataset / annotations directory (default: data/urfd).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="eval/cache/urfd",
        help="Path to cached NPZ sequence directory (default: eval/cache/urfd).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="ml/data/urfd_features.csv",
        help="Path where output tabular CSV will be saved (default: ml/data/urfd_features.csv).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving dataset to disk (dry-run summary only).",
    )

    args = parser.parse_args()

    print("=" * 86)
    print(" ARJUN ML FEATURE EXTRACTION & DATASET BUILDER (Phase 4A)")
    print(" [10 URFD Sequences: 5 Falls + 5 ADLs | Strictly Causal Geometric & Kinematic Features]")
    print("=" * 86)

    out_path = None if args.no_save else args.output_csv

    try:
        df = build_urfd_ml_dataset(
            urfd_dir=args.urfd_dir,
            cache_dir=args.cache_dir,
            output_csv_path=out_path,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to build ML dataset: {exc}", file=sys.stderr)
        return 1

    summary = summarize_dataset(df)

    # 1. Dataset Overview
    print("\n1. DATASET OVERVIEW:")
    print(f"   - Total Samples (Frames with detections) : {summary['total_samples']}")
    print(f"   - Total Sequences                        : {summary['total_sequences']}")
    print(f"   - Total Group IDs (Sequence-level)       : {summary['total_groups']}")
    print(f"   - Total Engineered Features              : {summary['feature_count']}")
    if out_path:
        print(f"   - Exported CSV Path                      : {out_path}")

    # 2. Class Distribution Table
    print("\n2. GROUND-TRUTH CLASS DISTRIBUTION:")
    print(f"   {'Class Label':<20} {'Count':>8} {'Percentage':>12}")
    print("   " + "-" * 42)
    for lbl, count in summary["label_counts"].items():
        pct = (count / summary["total_samples"]) * 100.0 if summary["total_samples"] > 0 else 0.0
        print(f"   {lbl:<20} {count:>8} {pct:>11.1f}%")
    print("   " + "-" * 42)
    print(f"   Ambiguous Boundary Samples: {summary['ambiguous_count']} ({summary['ambiguous_count']/summary['total_samples']*100:.1f}%)")

    # 3. Samples Per Sequence
    print("\n3. SAMPLES PER RECORDING SEQUENCE:")
    print(f"   {'Sequence ID':<22} {'Samples':>8} {'Normal':>8} {'Transition':>12} {'Fallen':>8}")
    print("   " + "-" * 62)
    for seq_id in summary["sequences"]:
        sub = df[df["sequence_id"] == seq_id]
        n_tot = len(sub)
        n_norm = int((sub["label"] == "normal").sum())
        n_trans = int((sub["label"] == "fall_transition").sum())
        n_down = int((sub["label"] == "fallen").sum())
        print(f"   {seq_id:<22} {n_tot:>8} {n_norm:>8} {n_trans:>12} {n_down:>8}")
    print("   " + "-" * 62)

    # 4. Feature Missingness Table
    print("\n4. FEATURE MISSINGNESS AUDIT (Selected Features):")
    print(f"   {'Feature Name':<36} {'Missing':>8} {'Missing %':>12}")
    print("   " + "-" * 58)
    for col, m in summary["missingness"].items():
        if m["missing_count"] > 0:
            print(f"   {col:<36} {m['missing_count']:>8} {m['missing_pct']:>11.1f}%")
    print("   " + "-" * 58)
    print("   Note: Missingness in joint angles occurs when limbs are self-occluded;")
    print("         explicit missingness indicator columns prevent fabricated observations.")

    # 5. Core Feature Summary Statistics
    key_features = [
        "box_aspect_ratio",
        "torso_angle_deg",
        "w300ms_hip_vertical_drop",
        "w600ms_hip_vertical_drop",
        "w600ms_hip_vertical_velocity",
        "w1000ms_horizontal_posture_fraction",
    ]
    print("\n5. KEY FEATURE SUMMARY STATISTICS (Non-Null Values):")
    print(f"   {'Feature':<36} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("   " + "-" * 72)
    for k in key_features:
        if k in summary["feature_stats"]:
            s = summary["feature_stats"][k]
            print(f"   {k:<36} {s['mean']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")
    print("   " + "-" * 72)

    # 6. Methodological Limitations Notice
    print("\n6. METHODOLOGICAL LIMITATIONS & CAUSALITY AUDIT:")
    print("   1. Pilot Sample Limitation: 10 recordings (~1336 frames total) is an exploratory baseline;")
    print("      it is NOT sufficient for training a generalizable production ML classifier.")
    print("   2. Subject Identity: URFD Camera 0 RGB metadata lacks subject IDs. Person-independent")
    print("      generalization cannot be guaranteed. Grouping MUST use sequence_id.")
    print("   3. Strict Causality: All features are computed over past windows [t - W, t]. No future")
    print("      frames, detector alerts, or clip-ending signals are used.")
    print("=" * 86)

    return 0


if __name__ == "__main__":
    sys.exit(main())
