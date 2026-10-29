"""
Unit tests for ML dataset building and validation (ml/dataset.py).

Covers:
- Label assignment across normal, fall_transition, and fallen states
- Boundary ambiguity detection
- Sequence-level grouping and zero leakage between recordings
- Tabular dataset building and summary statistics
- Integration test on cached 10 URFD sequences
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from eval.dataset import FrameObservation, Sequence, SequenceAnnotation
from backend.vision import synth
from backend.world import Detection
from ml.dataset import MLDatasetBuilder, build_urfd_ml_dataset, summarize_dataset


def _make_dummy_detection(track_id: int, y_offset: float = 0.0) -> Detection:
    kps = synth.standing(cx=300.0, feet_y=300.0 + y_offset, height=150.0)
    box = synth.box_for(kps)
    return Detection(track_id=track_id, label="person", conf=0.95, box=box, keypoints=kps)


def test_dataset_builder_label_assignment_fall_sequence():
    """Fall sequence assigns normal, fall_transition, and fallen labels accurately."""
    ann = SequenceAnnotation(
        sequence_id="synth-fall-test",
        name="Synthetic Fall",
        is_fall=True,
        duration_s=4.0,
        fall_transition_start_s=1.0,
        fallen_state_start_s=2.0,
    )

    frames = [
        # Normal upright: t=0.0, 0.5
        FrameObservation(frame_idx=0, timestamp=0.0, detections=[_make_dummy_detection(1)]),
        FrameObservation(frame_idx=1, timestamp=0.5, detections=[_make_dummy_detection(1)]),
        # Fall transition: t=1.2, 1.5
        FrameObservation(frame_idx=2, timestamp=1.2, detections=[_make_dummy_detection(1)]),
        FrameObservation(frame_idx=3, timestamp=1.5, detections=[_make_dummy_detection(1)]),
        # Fallen: t=2.5, 3.0
        FrameObservation(frame_idx=4, timestamp=2.5, detections=[_make_dummy_detection(1)]),
        FrameObservation(frame_idx=5, timestamp=3.0, detections=[_make_dummy_detection(1)]),
    ]
    seq = Sequence(annotation=ann, fps=10.0, frames=frames)

    builder = MLDatasetBuilder(boundary_margin_s=0.05)
    df = builder.build_dataset([seq])

    assert len(df) == 6

    # Normal phase
    assert df.loc[0, "label"] == "normal"
    assert df.loc[0, "is_fall"] == 0
    assert df.loc[1, "label"] == "normal"
    assert df.loc[1, "is_fall"] == 0

    # Fall transition phase
    assert df.loc[2, "label"] == "fall_transition"
    assert df.loc[2, "is_fall"] == 1
    assert df.loc[2, "is_fall_transition"] == 1
    assert df.loc[2, "is_fallen"] == 0
    assert df.loc[3, "label"] == "fall_transition"
    assert df.loc[3, "is_fall"] == 1

    # Fallen phase
    assert df.loc[4, "label"] == "fallen"
    assert df.loc[4, "is_fall"] == 1
    assert df.loc[4, "is_fall_transition"] == 0
    assert df.loc[4, "is_fallen"] == 1
    assert df.loc[5, "label"] == "fallen"
    assert df.loc[5, "is_fall"] == 1


def test_dataset_builder_label_assignment_adl_sequence():
    """ADL sequence assigns all observations to 'normal' with is_fall=0."""
    ann = SequenceAnnotation(
        sequence_id="synth-adl-test",
        name="Synthetic ADL",
        is_fall=False,
        duration_s=2.0,
    )
    frames = [
        FrameObservation(frame_idx=0, timestamp=0.0, detections=[_make_dummy_detection(1)]),
        FrameObservation(frame_idx=1, timestamp=1.0, detections=[_make_dummy_detection(1)]),
    ]
    seq = Sequence(annotation=ann, fps=10.0, frames=frames)

    builder = MLDatasetBuilder()
    df = builder.build_dataset([seq])

    assert len(df) == 2
    assert (df["label"] == "normal").all()
    assert (df["is_fall"] == 0).all()
    assert (df["is_fall_transition"] == 0).all()
    assert (df["is_fallen"] == 0).all()


def test_dataset_builder_grouping_no_leakage():
    """Group IDs strictly match sequence_id, ensuring no cross-sequence data leakage."""
    ann1 = SequenceAnnotation(sequence_id="seq-alpha", name="Alpha", is_fall=False, duration_s=1.0)
    ann2 = SequenceAnnotation(sequence_id="seq-beta", name="Beta", is_fall=False, duration_s=1.0)

    seq1 = Sequence(ann1, fps=10.0, frames=[FrameObservation(0, 0.0, [_make_dummy_detection(1)])])
    seq2 = Sequence(ann2, fps=10.0, frames=[FrameObservation(0, 0.0, [_make_dummy_detection(1)])])

    builder = MLDatasetBuilder()
    df = builder.build_dataset([seq1, seq2])

    assert len(df) == 2
    assert df.loc[0, "sequence_id"] == "seq-alpha"
    assert df.loc[0, "group_id"] == "seq-alpha"
    assert df.loc[1, "sequence_id"] == "seq-beta"
    assert df.loc[1, "group_id"] == "seq-beta"

    # Groups are disjoint
    alpha_groups = set(df[df["sequence_id"] == "seq-alpha"]["group_id"])
    beta_groups = set(df[df["sequence_id"] == "seq-beta"]["group_id"])
    assert alpha_groups.isdisjoint(beta_groups)


def test_dataset_boundary_ambiguity_flag():
    """Samples very close to transition boundaries are flagged as ambiguous."""
    ann = SequenceAnnotation(
        sequence_id="synth-boundary",
        name="Boundary Test",
        is_fall=True,
        duration_s=3.0,
        fall_transition_start_s=1.0,
        fallen_state_start_s=2.0,
    )
    frames = [
        # Right at transition boundary (1.0s) -> ambiguous
        FrameObservation(frame_idx=0, timestamp=1.00, detections=[_make_dummy_detection(1)]),
        # Far from boundary (1.50s) -> unambiguous
        FrameObservation(frame_idx=1, timestamp=1.50, detections=[_make_dummy_detection(1)]),
    ]
    seq = Sequence(annotation=ann, fps=10.0, frames=frames)

    builder = MLDatasetBuilder(boundary_margin_s=0.05)
    df = builder.build_dataset([seq])

    assert df.loc[0, "is_ambiguous"] == 1
    assert df.loc[1, "is_ambiguous"] == 0


def test_summarize_dataset_structure():
    """summarize_dataset generates accurate counts, class distributions, and feature statistics."""
    ann = SequenceAnnotation(sequence_id="seq-test", name="Test", is_fall=False, duration_s=1.0)
    seq = Sequence(ann, fps=10.0, frames=[
        FrameObservation(0, 0.0, [_make_dummy_detection(1)]),
        FrameObservation(1, 0.5, [_make_dummy_detection(1)]),
    ])

    builder = MLDatasetBuilder()
    df = builder.build_dataset([seq])
    summary = summarize_dataset(df)

    assert summary["total_samples"] == 2
    assert summary["total_sequences"] == 1
    assert summary["sequences"] == ["seq-test"]
    assert "normal" in summary["label_counts"]
    assert summary["feature_count"] > 30
    assert "box_aspect_ratio" in summary["feature_stats"]


@pytest.mark.skipif(
    not Path("eval/cache/urfd/fall-05-cam0-rgb.npz").is_file(),
    reason="Full 10-sequence cache not yet built locally",
)
def test_build_urfd_ml_dataset_live_10_sequences():
    """Build dataset from 10 live URFD sequences and verify schema and sample counts."""
    df = build_urfd_ml_dataset(
        urfd_dir="data/urfd",
        cache_dir="eval/cache/urfd",
        output_csv_path=None,
    )

    assert len(df) > 1000
    assert df["sequence_id"].nunique() == 10
    assert df["group_id"].nunique() == 10

    # Ensure all 3 labels are represented in the dataset
    classes = set(df["label"].unique())
    assert classes == {"normal", "fall_transition", "fallen"}

    # Ensure binary indicators match
    assert ((df["label"] == "normal") == (df["is_fall"] == 0)).all()
    assert ((df["label"] == "fall_transition") == (df["is_fall_transition"] == 1)).all()
    assert ((df["label"] == "fallen") == (df["is_fallen"] == 1)).all()

    # Check that required columns exist
    for req in ["sequence_id", "group_id", "track_id", "frame_idx", "timestamp",
                "window_start_ts", "window_end_ts", "label", "is_fall"]:
        assert req in df.columns
