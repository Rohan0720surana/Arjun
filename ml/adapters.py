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
import re
from typing import Any

import numpy as np
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


# --------------------------------------------------------------------------- #
# UP-Fall Activity Specifications & Metadata
# --------------------------------------------------------------------------- #

UPFALL_ACTIVITIES: dict[int, dict[str, Any]] = {
    1: {"name": "falling_forward_hands", "is_fall": True, "category": "fall", "nominal_duration_s": 10.0},
    2: {"name": "falling_forward_knees", "is_fall": True, "category": "fall", "nominal_duration_s": 10.0},
    3: {"name": "falling_backward", "is_fall": True, "category": "fall", "nominal_duration_s": 10.0},
    4: {"name": "falling_sideward", "is_fall": True, "category": "fall", "nominal_duration_s": 10.0},
    5: {"name": "falling_chair", "is_fall": True, "category": "fall", "nominal_duration_s": 10.0},
    6: {"name": "walking", "is_fall": False, "category": "adl", "nominal_duration_s": 60.0},
    7: {"name": "standing", "is_fall": False, "category": "adl", "nominal_duration_s": 60.0},
    8: {"name": "sitting", "is_fall": False, "category": "adl", "nominal_duration_s": 60.0},
    9: {"name": "picking_up_object", "is_fall": False, "category": "adl", "nominal_duration_s": 10.0},
    10: {"name": "jumping", "is_fall": False, "category": "adl", "nominal_duration_s": 30.0},
    11: {"name": "lying", "is_fall": False, "category": "adl", "nominal_duration_s": 60.0},
}


def parse_upfall_timestamps_csv(
    csv_path: str | Path,
    activity_num: int | None = None,
) -> dict[str, Any]:
    """
    Parse an official UP-Fall trial CSV to extract empirical timings:
    - total duration in seconds
    - total rows / frame count
    - empirical FPS from timestamp delta
    - transition start timestamp
    - fallen (ground / knees) state timestamp
    - measured post-fall dwell time

    In UP-Fall CSVs:
    - Activity code transitions: 7 (standing) -> 1..5 (fall transition) -> 11 (lying) / 20 (on knees).
    - Code 11 is interpreted using recording context:
      * Activity 11 ADL: intentional lying down (ADL), NOT a fall or fallen state.
      * Activities 1–5 (Falls): code 11 represents post-fall lying on the ground.
    - Robust to multi-line header/subheaders or single-line headers.
    """
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"UP-Fall trial CSV not found: {csv_file}")

    import csv
    from datetime import datetime

    with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        raw_rows = [r for r in reader if r]

    header = None
    data_rows = []
    for r in raw_rows:
        try:
            ts_str = r[0].strip()
            datetime.fromisoformat(ts_str)
            data_rows.append(r)
        except (ValueError, TypeError, IndexError):
            if header is None and any("activity" in col.lower() or "timestamp" in col.lower() for col in r):
                header = [c.strip() for c in r]

    if not data_rows:
        raise ValueError(f"No valid timestamped data rows found in {csv_file}")

    timestamps = [datetime.fromisoformat(r[0].strip()) for r in data_rows]
    t0 = timestamps[0]
    elapsed_s = np.array([(t - t0).total_seconds() for t in timestamps], dtype=np.float64)
    total_duration_s = float(elapsed_s[-1]) if len(elapsed_s) > 0 else 0.0

    fps = float(len(elapsed_s) - 1) / total_duration_s if total_duration_s > 0 else 18.0

    # Determine tag column:
    # 1. Real UP-Fall CSVs have an extra trailing column beyond line 1 header
    # 2. Synthetic or standardized CSVs have a Tag or Activity column
    row_len = len(data_rows[0])
    hdr_len = len(header) if header else 0
    if hdr_len > 0 and row_len > hdr_len:
        tag_col = -1
    elif header:
        low_hdr = [c.lower() for c in header]
        if "tag" in low_hdr:
            tag_col = low_hdr.index("tag")
        elif "activity" in low_hdr:
            tag_col = low_hdr.index("activity")
        else:
            tag_col = -1
    else:
        tag_col = -1

    activities = []
    for r in data_rows:
        try:
            tag_val = int(float(r[tag_col].strip()))
            activities.append(tag_val)
        except (ValueError, TypeError):
            activities.append(0)
    activities = np.array(activities, dtype=np.int32)

    # Determine recording context: infer activity_num if not explicitly provided
    if activity_num is None:
        act_match = re.search(r"Activity(\d+)", csv_file.name, re.IGNORECASE)
        if not act_match:
            act_match = re.search(r"Activity(\d+)", str(csv_file), re.IGNORECASE)
        if act_match:
            activity_num = int(act_match.group(1))

    # Identify fall transition start (activities 1..5)
    fall_transition_mask = np.isin(activities, [1, 2, 3, 4, 5])
    fall_transition_start_s = None
    if np.any(fall_transition_mask):
        first_trans_idx = int(np.argmax(fall_transition_mask))
        fall_transition_start_s = float(elapsed_s[first_trans_idx])

    # Interpret activity code 11 using recording context:
    # - Activities 1-5 (Falls): code 11 (lying) or 20 (on knees) represents post-fall lying.
    # - Activity 11 (ADL): code 11 is intentional lying, not a fall.
    # - Activities 6-10 (ADL): no fall state.
    # - When activity_num is unknown, fall codes (1..5) presence determines fall context.
    is_fall_context = (activity_num in range(1, 6)) if activity_num is not None else (fall_transition_start_s is not None)

    fallen_state_start_s = None
    if is_fall_context:
        fallen_mask = np.isin(activities, [11, 20])
        if np.any(fallen_mask):
            if fall_transition_start_s is not None:
                after_trans_mask = fallen_mask & (elapsed_s >= fall_transition_start_s)
                if np.any(after_trans_mask):
                    first_fallen_idx = int(np.argmax(after_trans_mask))
                    fallen_state_start_s = float(elapsed_s[first_fallen_idx])
                else:
                    first_fallen_idx = int(np.argmax(fallen_mask))
                    fallen_state_start_s = float(elapsed_s[first_fallen_idx])
            else:
                first_fallen_idx = int(np.argmax(fallen_mask))
                fallen_state_start_s = float(elapsed_s[first_fallen_idx])

    post_fallen_dwell_s = None
    if is_fall_context and fallen_state_start_s is not None:
        post_fallen_dwell_s = max(0.0, total_duration_s - fallen_state_start_s)

    return {
        "duration_s": total_duration_s,
        "fps": fps,
        "total_frames": len(elapsed_s),
        "fall_transition_start_s": fall_transition_start_s,
        "fallen_state_start_s": fallen_state_start_s,
        "post_fallen_dwell_s": post_fallen_dwell_s,
        "timestamps": elapsed_s,
        "tags": activities,
    }


class UPFallDatasetAdapter(DatasetAdapter):
    """
    Adapter for the UP-Fall Detection Dataset (Universidad Panamericana).

    Key Attributes:
    - 17 healthy young subjects (Subject1..Subject17).
    - 5 Fall types (forward hands, forward knees, backward, lateral, chair).
    - 6 ADLs (walking, standing, sitting down, picking up object, jumping, lying down).
    - Two RGB camera perspectives: Camera 1 (lateral), Camera 2 (frontal).
    - Microsecond-accurate timestamped PNG image sequences and companion CSVs.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/up_fall",
        cache_dir: str | Path = "eval/cache/up_fall",
    ) -> None:
        super().__init__(dataset_name="UP-Fall", root_dir=root_dir, cache_dir=cache_dir)

    def discover_recordings(self) -> list[str]:
        recordings = set()
        if self.root_dir.is_dir():
            for subj_dir in self.root_dir.glob("Subject*"):
                if subj_dir.is_dir():
                    for act_dir in subj_dir.glob("Activity*"):
                        for trial_dir in act_dir.glob("Trial*"):
                            recordings.add(f"{subj_dir.name}_{act_dir.name}_{trial_dir.name}")
                    for zip_file in subj_dir.glob("*.zip"):
                        recordings.add(zip_file.stem)
                    for frame_dir in subj_dir.glob("*Camera*"):
                        if frame_dir.is_dir():
                            recordings.add(frame_dir.name)
            for zip_file in self.root_dir.glob("Subject*.zip"):
                recordings.add(zip_file.stem)
        if self.cache_dir.is_dir():
            for npz_file in self.cache_dir.glob("*.npz"):
                recordings.add(npz_file.stem)
        return sorted(recordings)

    def load_annotation(self, recording_id: str) -> StandardizedAnnotation:
        # Extract Subject
        subj_match = re.search(r"Subject(\d+)", recording_id, re.IGNORECASE)
        subj_id = f"Subject{subj_match.group(1)}" if subj_match else "Subject1"

        # Extract Activity
        act_match = re.search(r"Activity(\d+)", recording_id, re.IGNORECASE)
        act_num = int(act_match.group(1)) if act_match else 1

        # Extract Trial
        trial_match = re.search(r"Trial(\d+)", recording_id, re.IGNORECASE)
        trial_id = f"Trial{trial_match.group(1)}" if trial_match else "Trial1"

        # Extract Camera
        cam_match = re.search(r"Cam(?:era)?(\d+)", recording_id, re.IGNORECASE)
        camera_id = f"cam{cam_match.group(1)}" if cam_match else "cam1"

        act_info = UPFALL_ACTIVITIES.get(
            act_num,
            {
                "name": f"activity_{act_num}",
                "is_fall": act_num in range(1, 6),
                "category": "fall" if act_num in range(1, 6) else "adl",
                "nominal_duration_s": 10.0 if act_num in range(1, 6) or act_num == 9 else 60.0,
            },
        )
        is_fall = act_info["is_fall"]

        # Search for official trial CSV
        csv_candidates = [
            self.root_dir / subj_id / f"{subj_id}Activity{act_num}{trial_id}.csv",
            self.root_dir / subj_id / f"Activity{act_num}" / trial_id / f"{subj_id}Activity{act_num}{trial_id}.csv",
            self.root_dir / subj_id / f"{recording_id}.csv",
            self.root_dir / f"{recording_id}.csv",
        ]
        csv_found = None
        for cand in csv_candidates:
            if cand.is_file():
                csv_found = cand
                break

        if csv_found is not None:
            measured = parse_upfall_timestamps_csv(csv_found, activity_num=act_num)
            return StandardizedAnnotation(
                sequence_id=recording_id,
                dataset_name="UP-Fall",
                is_fall=is_fall,
                duration_s=measured["duration_s"],
                fps=measured["fps"],
                total_frames=measured["total_frames"],
                subject_id=subj_id,
                camera_id=camera_id,
                activity_type="fall" if is_fall else "adl",
                fall_transition_start_s=measured["fall_transition_start_s"] if is_fall else None,
                fallen_state_start_s=measured["fallen_state_start_s"] if is_fall else None,
                metadata={
                    "specification": "UP-Fall Multi-Camera & IMU",
                    "activity_name": act_info["name"],
                    "dwell_measured": bool(is_fall and measured.get("post_fallen_dwell_s") is not None),
                    "measured_post_fallen_dwell_s": measured["post_fallen_dwell_s"] if is_fall else None,
                    "csv_path": str(csv_found),
                },
            )

        # When CSV is absent, do not assume post-fall dwell time
        return StandardizedAnnotation(
            sequence_id=recording_id,
            dataset_name="UP-Fall",
            is_fall=is_fall,
            duration_s=act_info["nominal_duration_s"],
            fps=18.0,
            total_frames=int(act_info["nominal_duration_s"] * 18.0),
            subject_id=subj_id,
            camera_id=camera_id,
            activity_type="fall" if is_fall else "adl",
            fall_transition_start_s=None,
            fallen_state_start_s=None,
            metadata={
                "specification": "UP-Fall Multi-Camera & IMU",
                "activity_name": act_info["name"],
                "dwell_measured": False,
                "nominal_only": True,
            },
        )

    def load_sequence(self, recording_id: str) -> Sequence:
        npz_file = self.cache_dir / f"{recording_id}.npz"
        if not npz_file.is_file():
            raise FileNotFoundError(f"UP-Fall pose cache not found: {npz_file}")

        from eval.extract_poses import load_pose_cache

        ann = self.load_annotation(recording_id)
        seq_ann = SequenceAnnotation(
            sequence_id=ann.sequence_id,
            name=f"UP-Fall {recording_id}",
            is_fall=ann.is_fall,
            duration_s=ann.duration_s,
            fall_transition_start_s=ann.fall_transition_start_s,
            fallen_state_start_s=ann.fallen_state_start_s,
        )
        return load_pose_cache(npz_file, annotation_override=seq_ann)
