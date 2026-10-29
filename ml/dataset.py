"""
ML Dataset Builder for URFD Pose Sequences (Phase 4A).

Constructs tabular training/evaluation datasets from extracted pose sequences.
Assigns causal ground-truth labels based on official URFD annotations, documents
label uncertainty, and enforces sequence-level grouping to prevent temporal leakage.

Label Taxonomy:
- "normal": Upright, walking, bending, sitting, or normal daily activities.
  (All ADL frames, and fall frames prior to fall_transition_start_s).
- "fall_transition": Active dynamic collapse phase (between State 0 and State 1).
- "fallen": Resting on the ground after the fall (at or after State 1).
- "is_ambiguous": Boundary frames within tolerance of transition thresholds or
  frames with severely occluded keypoints (< 20% visible).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.dataset import Sequence
from eval.urfd_batch_evaluation import load_urfd_10_sequences
from ml.features import DEFAULT_WINDOWS, extract_features_for_sequence


@dataclass
class DatasetSample:
    """A single tabular ML observation sample."""

    sequence_id: str
    group_id: str
    track_id: int
    frame_idx: int
    timestamp: float
    window_start_ts: float
    window_end_ts: float
    label: str                         # "normal", "fall_transition", "fallen"
    is_fall: int                       # 1 if transition or fallen, 0 if normal
    is_fall_transition: int            # 1 if fall_transition, 0 otherwise
    is_fallen: int                     # 1 if fallen, 0 otherwise
    is_ambiguous: int                  # 1 if near boundary or low-visibility, 0 otherwise
    features: dict[str, float | int]


class MLDatasetBuilder:
    """
    Builds tabular datasets from collections of evaluated Sequence objects.
    """

    def __init__(
        self,
        windows: tuple[float, ...] = DEFAULT_WINDOWS,
        boundary_margin_s: float = 0.05,
        min_keypoint_vis: float = 0.20,
    ) -> None:
        self.windows = tuple(sorted(windows))
        self.boundary_margin_s = boundary_margin_s
        self.min_keypoint_vis = min_keypoint_vis
        self.max_window_s = max(self.windows) if self.windows else 0.0

    def extract_sequence_samples(self, sequence: Sequence) -> list[dict[str, Any]]:
        """
        Extract all feature rows with aligned ground truth for a single Sequence.
        """
        ann = sequence.annotation
        seq_id = ann.sequence_id
        is_fall_seq = ann.is_fall is True

        trans_start = ann.fall_transition_start_s
        fallen_start = ann.fallen_state_start_s

        feature_rows = extract_features_for_sequence(sequence, windows=self.windows)
        dataset_rows: list[dict[str, Any]] = []

        for row in feature_rows:
            ts = float(row["timestamp"])
            vis_frac = float(row.get("keypoint_visible_fraction", 1.0))

            # 1. Label Assignment based on official ground truth
            if not is_fall_seq:
                label = "normal"
                is_fall = 0
                is_trans = 0
                is_down = 0
                is_ambig = 1 if vis_frac < self.min_keypoint_vis else 0
            else:
                # Sequence is a verified fall recording
                if trans_start is not None and ts < trans_start:
                    label = "normal"
                    is_fall = 0
                    is_trans = 0
                    is_down = 0
                    # Ambiguity check near transition start boundary
                    is_ambig = 1 if (trans_start - ts) <= self.boundary_margin_s or vis_frac < self.min_keypoint_vis else 0

                elif trans_start is not None and fallen_start is not None and (trans_start <= ts < fallen_start):
                    label = "fall_transition"
                    is_fall = 1
                    is_trans = 1
                    is_down = 0
                    # Ambiguity check near boundaries
                    near_trans = abs(ts - trans_start) <= self.boundary_margin_s
                    near_fallen = abs(ts - fallen_start) <= self.boundary_margin_s
                    is_ambig = 1 if near_trans or near_fallen or vis_frac < self.min_keypoint_vis else 0

                elif fallen_start is not None and ts >= fallen_start:
                    label = "fallen"
                    is_fall = 1
                    is_trans = 0
                    is_down = 1
                    near_fallen = abs(ts - fallen_start) <= self.boundary_margin_s
                    is_ambig = 1 if near_fallen or vis_frac < self.min_keypoint_vis else 0

                else:
                    # Fall sequence without granular transition timestamps
                    label = "fall_transition"
                    is_fall = 1
                    is_trans = 1
                    is_down = 0
                    is_ambig = 1

            # Window boundaries
            w_start_ts = round(max(0.0, ts - self.max_window_s), 4)
            w_end_ts = round(ts, 4)

            entry: dict[str, Any] = {
                "sequence_id": seq_id,
                "group_id": seq_id,            # Group ID for Leave-One-Sequence-Out CV
                "track_id": int(row["track_id"]),
                "frame_idx": int(row["frame_idx"]),
                "timestamp": round(ts, 4),
                "window_start_ts": w_start_ts,
                "window_end_ts": w_end_ts,
                "label": label,
                "is_fall": is_fall,
                "is_fall_transition": is_trans,
                "is_fallen": is_down,
                "is_ambiguous": is_ambig,
            }

            # Append all feature columns (excluding metadata already captured)
            for k, v in row.items():
                if k not in ("sequence_id", "track_id", "frame_idx", "timestamp"):
                    entry[k] = v

            dataset_rows.append(entry)

        return dataset_rows

    def build_dataset(self, sequences: list[Sequence]) -> pd.DataFrame:
        """
        Build complete tabular DataFrame across multiple Sequence recordings.
        """
        all_rows: list[dict[str, Any]] = []
        for seq in sequences:
            seq_rows = self.extract_sequence_samples(seq)
            all_rows.extend(seq_rows)

        df = pd.DataFrame(all_rows)
        return df


def build_urfd_ml_dataset(
    urfd_dir: str | Path = "data/urfd",
    cache_dir: str | Path = "eval/cache/urfd",
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load all 10 URFD sequences (5 falls + 5 ADLs) and build the full ML dataset.
    """
    sequences = load_urfd_10_sequences(urfd_dir=urfd_dir, cache_dir=cache_dir)
    builder = MLDatasetBuilder()
    df = builder.build_dataset(sequences)

    if output_csv_path is not None:
        p = Path(output_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute distribution, missingness, and grouping statistics for a dataset DataFrame.
    """
    total_samples = len(df)
    unique_seqs = sorted(df["sequence_id"].unique().tolist())
    unique_groups = sorted(df["group_id"].unique().tolist())

    # Class distributions
    label_counts = df["label"].value_counts().to_dict()
    is_fall_counts = df["is_fall"].value_counts().to_dict()
    ambiguous_count = int(df["is_ambiguous"].sum())

    # Samples per sequence
    seq_counts = df["sequence_id"].value_counts().to_dict()

    # Feature columns (exclude metadata & label columns)
    meta_cols = {
        "sequence_id", "group_id", "track_id", "frame_idx",
        "timestamp", "window_start_ts", "window_end_ts",
        "label", "is_fall", "is_fall_transition", "is_fallen", "is_ambiguous",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Missingness across features
    missingness: dict[str, dict[str, Any]] = {}
    for col in feature_cols:
        n_missing = int(df[col].isna().sum())
        pct_missing = round(n_missing / total_samples * 100.0, 2) if total_samples > 0 else 0.0
        missingness[col] = {
            "missing_count": n_missing,
            "missing_pct": pct_missing,
        }

    # Summary statistics for numeric features
    feature_stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        series = df[col].dropna()
        if len(series) > 0 and pd.api.types.is_numeric_dtype(series):
            feature_stats[col] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "median": round(float(series.median()), 4),
                "max": round(float(series.max()), 4),
            }

    return {
        "total_samples": total_samples,
        "total_sequences": len(unique_seqs),
        "sequences": unique_seqs,
        "total_groups": len(unique_groups),
        "label_counts": label_counts,
        "is_fall_counts": is_fall_counts,
        "ambiguous_count": ambiguous_count,
        "samples_per_sequence": seq_counts,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "missingness": missingness,
        "feature_stats": feature_stats,
    }
