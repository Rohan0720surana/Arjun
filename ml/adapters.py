"""
Reusable Dataset Adapter Architecture for Multi-Dataset Fall Evaluation (Phase 4E).

Defines an abstract, standardized adapter interface allowing heterogeneous public
fall datasets (URFD, Le2i, UP-Fall, CAUCAFall) to feed directly into:
1. YOLOv8 Pose extraction
2. ByteTrack tracking
3. Causal ML feature engineering (ml/features.py)
4. Offline classifier inference (Random Forest, XGBoost)
5. Temporal event-level confirmation (ml/event_evaluation.py)

Strict Separation of Concerns:
- Ground-truth annotations, subject identities, and camera IDs are preserved in
  metadata for evaluation and splitting, but are NEVER exposed as model features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from eval.dataset import FrameObservation, Sequence, SequenceAnnotation
from eval.urfd_batch_evaluation import load_urfd_10_sequences


@dataclass
class StandardizedAnnotation:
    """Standardized ground-truth metadata across diverse fall datasets."""
    sequence_id: str
    dataset_name: str
    is_fall: bool
    duration_s: float
    fps: float
    total_frames: int
    subject_id: str | None = None
    camera_id: str | None = None
    activity_type: str = "unknown"
    fall_transition_start_s: float | None = None
    fallen_state_start_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def post_fallen_dwell_s(self) -> float | None:
        """Available ground-truth observation duration after subject reaches the floor."""
        if self.is_fall and self.fallen_state_start_s is not None:
            return max(0.0, self.duration_s - self.fallen_state_start_s)
        return None

    def is_eligible_for_confirmation(self, required_confirmation_s: float = 2.0) -> bool:
        """Prediction-independent eligibility: must have sufficient post-impact dwell."""
        if not self.is_fall:
            return True
        dwell = self.post_fallen_dwell_s
        return dwell is not None and dwell >= required_confirmation_s


class DatasetAdapter(ABC):
    """
    Abstract adapter interface for multi-dataset ingest.
    """

    def __init__(self, dataset_name: str, root_dir: str | Path, cache_dir: str | Path) -> None:
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self.cache_dir = Path(cache_dir)

    @abstractmethod
    def discover_recordings(self) -> list[str]:
        """Return list of canonical recording identifiers available in dataset."""
        pass

    @abstractmethod
    def load_annotation(self, recording_id: str) -> StandardizedAnnotation:
        """Parse official dataset annotations into StandardizedAnnotation."""
        pass

    @abstractmethod
    def load_sequence(self, recording_id: str) -> Sequence:
        """Load Sequence object (detections, bounding boxes, keypoints, timestamps)."""
        pass

    def load_all_sequences(self) -> list[Sequence]:
        """Convenience method to load all available sequences."""
        return [self.load_sequence(rid) for rid in self.discover_recordings()]


# --------------------------------------------------------------------------- #
# URFD Adapter Implementation
# --------------------------------------------------------------------------- #

class URFDAdapter(DatasetAdapter):
    """
    Adapter for the 10-sequence URFD Camera 0 benchmark (5 Falls + 5 ADLs).
    """

    def __init__(
        self,
        root_dir: str | Path = "data/urfd",
        cache_dir: str | Path = "eval/cache/urfd",
    ) -> None:
        super().__init__(dataset_name="URFD", root_dir=root_dir, cache_dir=cache_dir)
        self._cached_sequences: dict[str, Sequence] = {}

    def discover_recordings(self) -> list[str]:
        return [
            "fall-01-cam0-rgb",
            "fall-02-cam0-rgb",
            "fall-03-cam0-rgb",
            "fall-04-cam0-rgb",
            "fall-05-cam0-rgb",
            "adl-01-cam0-rgb",
            "adl-02-cam0-rgb",
            "adl-03-cam0-rgb",
            "adl-04-cam0-rgb",
            "adl-05-cam0-rgb",
        ]

    def _ensure_loaded(self) -> None:
        if not self._cached_sequences:
            seqs = load_urfd_10_sequences(urfd_dir=self.root_dir, cache_dir=self.cache_dir)
            for s in seqs:
                self._cached_sequences[s.annotation.sequence_id] = s

    def load_annotation(self, recording_id: str) -> StandardizedAnnotation:
        self._ensure_loaded()
        clean_id = recording_id if recording_id in self._cached_sequences else recording_id.replace("-cam0-rgb", "")
        seq = self._cached_sequences[clean_id]
        ann = seq.annotation

        return StandardizedAnnotation(
            sequence_id=ann.sequence_id,
            dataset_name="URFD",
            is_fall=bool(ann.is_fall),
            duration_s=ann.duration_s,
            fps=seq.fps,
            total_frames=len(seq.frames),
            subject_id=None,  # URFD Camera 0 RGB does not provide official per-clip subject IDs
            camera_id="cam0",
            activity_type="fall" if ann.is_fall else "adl",
            fall_transition_start_s=ann.fall_transition_start_s,
            fallen_state_start_s=ann.fallen_state_start_s,
            metadata={"fps_provenance": "Official CSV Sync" if ann.is_fall else "Assumed 30.0 FPS"},
        )

    def load_sequence(self, recording_id: str) -> Sequence:
        self._ensure_loaded()
        clean_id = recording_id if recording_id in self._cached_sequences else recording_id.replace("-cam0-rgb", "")
        return self._cached_sequences[clean_id]


# --------------------------------------------------------------------------- #
# Le2i & UP-Fall Dataset Adapter Blueprints
# --------------------------------------------------------------------------- #

class Le2iDatasetAdapter(DatasetAdapter):
    """
    Blueprint adapter for the Le2i Fall Detection Dataset (Université de Bourgogne).

    Structure:
    - 191 video recordings across 4 rooms: Home (Coffee room), Office, Lecture room.
    - Video layout: <room>/Videos/video (<index>).avi
    - Annotations: <room>/Annotation_files/video (<index>).txt
      Format: [fall_start_frame, fall_end_frame (impact), fallen_dwell_end_frame]
    """

    def __init__(
        self,
        root_dir: str | Path = "data/le2i",
        cache_dir: str | Path = "eval/cache/le2i",
    ) -> None:
        super().__init__(dataset_name="Le2i", root_dir=root_dir, cache_dir=cache_dir)

    def discover_recordings(self) -> list[str]:
        if not self.root_dir.is_dir():
            return []
        recordings = []
        for avi_file in self.root_dir.glob("*/*/video (*).avi"):
            rel_name = f"{avi_file.parent.parent.name}_{avi_file.stem}"
            recordings.append(rel_name)
        return sorted(recordings)

    def load_annotation(self, recording_id: str) -> StandardizedAnnotation:
        # Blueprint parser for Le2i text annotations
        return StandardizedAnnotation(
            sequence_id=recording_id,
            dataset_name="Le2i",
            is_fall="fall" in recording_id.lower(),
            duration_s=0.0,
            fps=25.0,
            total_frames=0,
            subject_id=None,
            camera_id="fixed_cam",
            activity_type="fall" if "fall" in recording_id.lower() else "adl",
            metadata={"specification": "Le2i 25 FPS AVI"},
        )

    def load_sequence(self, recording_id: str) -> Sequence:
        npz_file = self.cache_dir / f"{recording_id}.npz"
        if not npz_file.is_file():
            raise FileNotFoundError(f"Le2i pose cache not found: {npz_file}")
        ann = SequenceAnnotation(sequence_id=recording_id, name=f"Le2i {recording_id}", is_fall=None, duration_s=0.0)
        return Sequence(annotation=ann, fps=25.0, frames=[])


class UPFallDatasetAdapter(DatasetAdapter):
    """
    Blueprint adapter for the UP-Fall Detection Dataset (Universidad Panamericana).

    Key Attributes:
    - 17 healthy young subjects (Subject1..Subject17).
    - 5 Fall types (forward hands, forward knees, backward, chair, lateral).
    - 6 ADLs (walking, standing, sitting down, picking up object, jumping, lying down on bed/mat).
    - Two RGB camera perspectives: Camera 1 (front), Camera 2 (side).
    - Timestamped CSV format with subject IDs and activity labels.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/up_fall",
        cache_dir: str | Path = "eval/cache/up_fall",
    ) -> None:
        super().__init__(dataset_name="UP-Fall", root_dir=root_dir, cache_dir=cache_dir)

    def discover_recordings(self) -> list[str]:
        if not self.root_dir.is_dir():
            return []
        recordings = []
        for subj_dir in self.root_dir.glob("Subject*"):
            for act_dir in subj_dir.glob("Activity*"):
                for trial_dir in act_dir.glob("Trial*"):
                    recordings.append(f"{subj_dir.name}_{act_dir.name}_{trial_dir.name}")
        return sorted(recordings)

    def load_annotation(self, recording_id: str) -> StandardizedAnnotation:
        parts = recording_id.split("_")
        subj_id = parts[0] if parts else "Subject1"
        is_fall = "fall" in recording_id.lower() or any(f"Activity{i}" in recording_id for i in range(1, 6))

        return StandardizedAnnotation(
            sequence_id=recording_id,
            dataset_name="UP-Fall",
            is_fall=is_fall,
            duration_s=0.0,
            fps=18.0,
            total_frames=0,
            subject_id=subj_id,
            camera_id="cam1",
            activity_type="fall" if is_fall else "adl",
            metadata={"specification": "UP-Fall Multi-Camera & IMU"},
        )

    def load_sequence(self, recording_id: str) -> Sequence:
        npz_file = self.cache_dir / f"{recording_id}.npz"
        if not npz_file.is_file():
            raise FileNotFoundError(f"UP-Fall pose cache not found: {npz_file}")
        ann = SequenceAnnotation(sequence_id=recording_id, name=f"UP-Fall {recording_id}", is_fall=None, duration_s=0.0)
        return Sequence(annotation=ann, fps=18.0, frames=[])
