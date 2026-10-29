"""
Official URFD ground-truth annotation importer and manifest builder.

Parses official annotations from the UR Fall Detection Dataset (URFD):
- `urfall-cam0-falls.csv`: Frame-level posture state annotations (-1, 0, 1).
- `fall-XX-data.csv`: Millisecond-precision timestamps and accelerometer data.

State Labels (official URFD convention):
- -1: Normal / walking / upright posture
-  0: Falling transition (subject is dynamically falling/collapsing)
-  1: Fallen state (subject is resting down on the floor)

Indexing Conventions:
- URFD CSVs use 1-based frame numbers (frame 1, 2, ...).
- Pose cache arrays and Python sequences use 0-based indexing (index 0, 1, ...).
- Frame N corresponds to pose cache array index N - 1.

Confirmation Window Semantics:
- ARJUN's production FallDetector requires 2.0s of sustained horizontal posture
  on the floor (fall_down_seconds = 2.0s).
- The 2.0s countdown starts only after the subject has completed the drop and
  reached the ground.
- Therefore, the observation-sufficiency check must anchor at `fallen_state_start_s`
  (first frame with state label 1).
- `fall_transition_start_s` (state 0) represents the dynamic falling phase and
  MUST NOT be used as the floor confirmation anchor.
- The fallen-state boundary is a visual posture classification, not a physically
  verified accelerometer impact timestamp.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from eval.dataset import Sequence, SequenceAnnotation
from eval.extract_poses import SCHEMA_VERSION, load_pose_cache


@dataclass(frozen=True)
class URFDSequenceAnnotation:
    """Official URFD ground truth event metadata for one fall sequence."""
    sequence_id: str
    is_fall: bool
    fall_transition_start_frame: int       # 1-based frame number of first state 0
    fall_transition_start_s: float         # Verified timestamp of first state 0
    fallen_state_start_frame: int          # 1-based frame number of first state 1
    fallen_state_start_s: float            # Verified timestamp of first state 1
    total_frames: int                      # Number of annotated frames
    duration_s: float                      # Last observed frame timestamp
    last_frame_number: int                 # 1-based frame number of last frame
    available_post_fallen_s: float         # duration_s - fallen_state_start_s

    def to_sequence_annotation(self, name: str | None = None) -> SequenceAnnotation:
        """Convert to ARJUN's standardized SequenceAnnotation model."""
        return SequenceAnnotation(
            sequence_id=self.sequence_id,
            name=name or f"URFD {self.sequence_id}",
            is_fall=self.is_fall,
            duration_s=self.duration_s,
            fall_start_s=self.fall_transition_start_s,
            fall_transition_start_s=self.fall_transition_start_s,
            fallen_state_start_s=self.fallen_state_start_s,
            fall_impact_s=None,  # State 1 is posture onset, not physical impact
            description=(
                f"URFD official annotation: transition start at frame {self.fall_transition_start_frame} "
                f"({self.fall_transition_start_s:.3f}s), fallen state at frame {self.fallen_state_start_frame} "
                f"({self.fallen_state_start_s:.3f}s), ends at {self.duration_s:.3f}s."
            ),
        )


def parse_urfd_falls_csv(
    csv_path: str | Path,
) -> dict[str, list[tuple[int, int]]]:
    """
    Parse official urfall-cam0-falls.csv using only the first three columns.

    Format:
        sequence_id, frame_number, state_label, [extra feature columns...]

    Validations performed:
    - sequence_id: non-empty string.
    - frame_number: 1-based integer (>= 1).
    - state_label: integer in {-1, 0, 1}.
    - duplicates: raises ValueError if duplicate (sequence_id, frame_number) exists.
    - ordering: raises ValueError if frame numbers within a sequence are non-increasing.
    - extra columns: safely ignored.

    Returns:
        Dictionary mapping sequence_id -> list of (frame_number_1based, state_label).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"URFD falls annotation file not found: {path}")

    sequences: dict[str, list[tuple[int, int]]] = {}
    seen_entries: set[tuple[str, int]] = set()

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_idx, row in enumerate(reader, 1):
            if not row or not any(field.strip() for field in row):
                continue

            if len(row) < 3:
                raise ValueError(
                    f"Line {line_idx} in {path.name} has fewer than 3 columns: {row}"
                )

            seq_id = row[0].strip()
            if not seq_id:
                raise ValueError(f"Line {line_idx} in {path.name} has an empty sequence ID.")

            try:
                frame_num = int(row[1].strip())
            except ValueError as e:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: non-integer frame number '{row[1]}'."
                ) from e

            if frame_num < 1:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: frame number {frame_num} must be >= 1 (1-based)."
                )

            try:
                state_val = int(row[2].strip())
            except ValueError as e:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: non-integer state label '{row[2]}'."
                ) from e

            if state_val not in (-1, 0, 1):
                raise ValueError(
                    f"Line {line_idx} in {path.name}: invalid state {state_val}; must be -1, 0, or 1."
                )

            entry_key = (seq_id, frame_num)
            if entry_key in seen_entries:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: duplicate entry for sequence '{seq_id}' frame {frame_num}."
                )
            seen_entries.add(entry_key)

            if seq_id not in sequences:
                sequences[seq_id] = []
            else:
                last_frame = sequences[seq_id][-1][0]
                if frame_num <= last_frame:
                    raise ValueError(
                        f"Line {line_idx} in {path.name}: out-of-order frame {frame_num} after {last_frame} "
                        f"in sequence '{seq_id}'."
                    )

            sequences[seq_id].append((frame_num, state_val))

    return sequences


def parse_urfd_data_csv(
    data_csv_path: str | Path,
) -> dict[int, float]:
    """
    Parse a sequence synchronization CSV (e.g. fall-01-data.csv).

    Format:
        frame_number, timestamp_ms, [accel_mag, ...]

    Returns:
        Dictionary mapping 1-based frame_number -> timestamp_in_seconds (float).
    """
    path = Path(data_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"URFD sequence data CSV not found: {path}")

    timestamps: dict[int, float] = {}
    last_frame = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_idx, row in enumerate(reader, 1):
            if not row or not any(field.strip() for field in row):
                continue

            if len(row) < 2:
                raise ValueError(
                    f"Line {line_idx} in {path.name} has fewer than 2 columns: {row}"
                )

            try:
                frame_num = int(row[0].strip())
            except ValueError as e:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: non-integer frame number '{row[0]}'."
                ) from e

            if frame_num < 1:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: frame number {frame_num} must be >= 1."
                )

            if frame_num in timestamps:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: duplicate frame number {frame_num}."
                )

            if frame_num <= last_frame:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: out-of-order frame {frame_num} after {last_frame}."
                )
            last_frame = frame_num

            try:
                time_ms = float(row[1].strip())
            except ValueError as e:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: non-numeric timestamp '{row[1]}'."
                ) from e

            if time_ms < 0:
                raise ValueError(
                    f"Line {line_idx} in {path.name}: negative timestamp {time_ms}ms."
                )

            timestamps[frame_num] = round(time_ms / 1000.0, 4)

    return timestamps


def load_urfd_fall_annotation(
    sequence_id: str,
    falls_csv_path: str | Path,
    data_csv_path: str | Path,
) -> URFDSequenceAnnotation:
    """
    Join official urfall-cam0-falls.csv annotations for `sequence_id` with
    official timestamps from `data_csv_path`.

    Extracts:
    - fall_transition_start_frame: First frame with state 0.
    - fall_transition_start_s: Timestamp at first frame with state 0.
    - fallen_state_start_frame: First frame with state 1.
    - fallen_state_start_s: Timestamp at first frame with state 1.
    - total_frames: Number of annotated frames.
    - duration_s: Timestamp of the final annotated frame.
    - available_post_fallen_s: duration_s - fallen_state_start_s.

    Raises:
        KeyError: If sequence_id is not in the falls CSV.
        ValueError: If state 0 or state 1 is missing, or an annotated frame lacks a timestamp.
    """
    all_seqs = parse_urfd_falls_csv(falls_csv_path)
    if sequence_id not in all_seqs:
        raise KeyError(
            f"Sequence '{sequence_id}' not found in falls annotation file {Path(falls_csv_path).name}."
        )

    frame_states = all_seqs[sequence_id]
    timestamps = parse_urfd_data_csv(data_csv_path)

    # Validate that every annotated frame has an official timestamp
    for frame_num, _ in frame_states:
        if frame_num not in timestamps:
            raise ValueError(
                f"Frame {frame_num} in sequence '{sequence_id}' lacks a timestamp in {Path(data_csv_path).name}."
            )

    # Locate transitions
    first_state_0: int | None = None
    first_state_1: int | None = None

    for frame_num, state_val in frame_states:
        if state_val == 0 and first_state_0 is None:
            first_state_0 = frame_num
        elif state_val == 1 and first_state_1 is None:
            first_state_1 = frame_num

    if first_state_0 is None:
        raise ValueError(f"Sequence '{sequence_id}' has no falling transition state (state 0).")
    if first_state_1 is None:
        raise ValueError(f"Sequence '{sequence_id}' has no fallen state (state 1).")
    if first_state_1 <= first_state_0:
        raise ValueError(
            f"Sequence '{sequence_id}': fallen state (frame {first_state_1}) occurs at or before "
            f"falling transition (frame {first_state_0})."
        )

    t_transition = timestamps[first_state_0]
    t_fallen = timestamps[first_state_1]
    last_frame_num = frame_states[-1][0]
    duration_s = timestamps[last_frame_num]
    post_fallen_s = round(duration_s - t_fallen, 4)

    return URFDSequenceAnnotation(
        sequence_id=sequence_id,
        is_fall=True,
        fall_transition_start_frame=first_state_0,
        fall_transition_start_s=t_transition,
        fallen_state_start_frame=first_state_1,
        fallen_state_start_s=t_fallen,
        total_frames=len(frame_states),
        duration_s=duration_s,
        last_frame_number=last_frame_num,
        available_post_fallen_s=post_fallen_s,
    )


def build_urfd_manifest(
    urfd_dir: str | Path,
    falls_csv_name: str = "urfall-cam0-falls.csv",
) -> dict[str, SequenceAnnotation]:
    """
    Build a verified manifest mapping sequence IDs to SequenceAnnotation objects.

    Scans `urfd_dir` for `falls_csv_name` and individual `fall-XX-data.csv` files.
    Sequences without corresponding timestamp files are omitted so they remain
    strictly UNVERIFIED in the evaluation pipeline.
    """
    urfd_path = Path(urfd_dir)
    falls_csv = urfd_path / falls_csv_name
    if not falls_csv.is_file():
        raise FileNotFoundError(f"Falls CSV not found: {falls_csv}")

    all_seqs = parse_urfd_falls_csv(falls_csv)
    manifest: dict[str, SequenceAnnotation] = {}

    for seq_id in sorted(all_seqs.keys()):
        data_csv = urfd_path / f"{seq_id}-data.csv"
        if not data_csv.is_file():
            # Timestamps not yet available; do not invent or guess timestamps
            continue

        try:
            urfd_ann = load_urfd_fall_annotation(seq_id, falls_csv, data_csv)
            ann = urfd_ann.to_sequence_annotation()
            # Store under primary sequence ID and common camera naming variants
            manifest[seq_id] = ann
            manifest[f"{seq_id}-cam0"] = ann
            manifest[f"{seq_id}-cam0-rgb"] = ann
        except Exception:
            # If parsing fails for any sequence, omit from verified manifest
            continue

    return manifest


def apply_urfd_annotation_to_pose_cache(
    npz_path: str | Path,
    annotation: SequenceAnnotation,
    timestamps: dict[int, float] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Apply verified URFD annotations to an existing NPZ pose cache without rerunning YOLO.

    Preserves 100% of all pose observations (boxes, keypoints, track IDs, confidences).
    If `timestamps` is provided (mapping 1-based frame numbers to timestamps in seconds),
    `frame_timestamps` are aligned to the official timestamps (0-based index i maps to frame i+1).
    """
    in_path = Path(npz_path)
    out_path = Path(output_path) if output_path is not None else in_path

    if not in_path.is_file():
        raise FileNotFoundError(f"Input pose cache not found: {in_path}")

    with np.load(in_path) as data:
        schema = str(data["schema_version"])
        sequence_id = str(annotation.sequence_id)
        name = str(annotation.name)
        fps = float(data["fps"])
        num_frames = int(data["num_frames"])
        frame_indices = data["frame_indices"]
        det_frame_indices = data["det_frame_indices"]
        det_track_ids = data["det_track_ids"]
        det_confs = data["det_confs"]
        det_boxes = data["det_boxes"]
        det_keypoints = data["det_keypoints"]
        det_has_keypoints = data["det_has_keypoints"]

        if timestamps is not None:
            new_timestamps = np.empty(num_frames, dtype=np.float32)
            for i in range(num_frames):
                frame_num_1based = i + 1
                if frame_num_1based in timestamps:
                    new_timestamps[i] = timestamps[frame_num_1based]
                else:
                    new_timestamps[i] = data["frame_timestamps"][i]
            frame_timestamps = new_timestamps
            duration_s = float(frame_timestamps[-1]) if num_frames > 0 else annotation.duration_s
        else:
            frame_timestamps = data["frame_timestamps"]
            duration_s = annotation.duration_s

    is_fall_val = 1 if annotation.is_fall is True else (0 if annotation.is_fall is False else -1)
    fall_start_val = annotation.fall_start_s if annotation.fall_start_s is not None else np.nan
    fall_impact_val = annotation.fall_impact_s if annotation.fall_impact_s is not None else np.nan
    fall_transition_val = (
        annotation.fall_transition_start_s
        if annotation.fall_transition_start_s is not None
        else np.nan
    )
    fallen_state_val = (
        annotation.fallen_state_start_s
        if annotation.fallen_state_start_s is not None
        else np.nan
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.array(schema),
        sequence_id=np.array(sequence_id),
        name=np.array(name),
        fps=np.array(fps, dtype=np.float32),
        duration_s=np.array(duration_s, dtype=np.float32),
        num_frames=np.array(num_frames, dtype=np.int32),
        is_fall=np.array(is_fall_val, dtype=np.int32),
        fall_start_s=np.array(fall_start_val, dtype=np.float32),
        fall_impact_s=np.array(fall_impact_val, dtype=np.float32),
        fall_transition_start_s=np.array(fall_transition_val, dtype=np.float32),
        fallen_state_start_s=np.array(fallen_state_val, dtype=np.float32),
        frame_indices=frame_indices,
        frame_timestamps=frame_timestamps,
        det_frame_indices=det_frame_indices,
        det_track_ids=det_track_ids,
        det_confs=det_confs,
        det_boxes=det_boxes,
        det_keypoints=det_keypoints,
        det_has_keypoints=det_has_keypoints,
    )

    return out_path


def load_urfd_sequence(
    npz_path: str | Path,
    manifest: Mapping[str, SequenceAnnotation] | None = None,
) -> Sequence:
    """
    Load a pose cache Sequence, applying verified manifest annotations if available.

    If the sequence is not in the manifest, the sequence remains UNVERIFIED (is_fall=None).
    """
    path = Path(npz_path)
    seq = load_pose_cache(path)

    if manifest is not None:
        seq_id = seq.annotation.sequence_id
        if seq_id in manifest:
            override_ann = manifest[seq_id]
            seq = load_pose_cache(path, annotation_override=override_ann)

    return seq


URFD_FALL_SEQUENCES: list[str] = [
    "fall-01",
    "fall-02",
    "fall-03",
    "fall-04",
    "fall-05",
]


def evaluate_urfd_pilot(
    urfd_dir: str | Path = "data/urfd",
    cache_dir: str | Path = "eval/cache/urfd",
    required_confirmation_s: float = 2.0,
    seq_ids: list[str] | None = None,
) -> int:
    """
    Evaluate the real URFD fall recordings using verified ground truth.
    """
    from eval.metrics import evaluate_results
    from eval.runner import OfflineFallRunner

    manifest = build_urfd_manifest(urfd_dir)

    print("=" * 78)
    print(" URFD GROUND-TRUTH ANNOTATION INSPECTION")
    print("=" * 78)

    pilot_ids = seq_ids if seq_ids is not None else URFD_FALL_SEQUENCES
    for sid in pilot_ids:
        ann = manifest.get(sid)
        if ann is None:
            print(f"[-] {sid}: Missing verified ground truth manifest entry.")
            continue

        post_fallen = ann.duration_s - (ann.fallen_state_start_s or 0.0)
        sufficient = ann.has_sufficient_observation(required_confirmation_s)
        status_str = "SUFFICIENT" if sufficient else "INSUFFICIENT"

        print(f"[*] Sequence: {sid}")
        print(f"    - Transition Start  (State 0) : {ann.fall_transition_start_s:.3f}s")
        print(f"    - Fallen State Start (State 1) : {ann.fallen_state_start_s:.3f}s")
        print(f"    - Last Observed Timestamp      : {ann.duration_s:.3f}s")
        print(f"    - Post-Fallen Observation Time : {post_fallen:.3f}s (required: {required_confirmation_s:.1f}s)")
        print(f"    - Observation Sufficiency      : {status_str} (confirmation window: {required_confirmation_s:.1f}s)")

    print("-" * 78)

    benchmark_sequences: list[Sequence] = []
    cache_path = Path(cache_dir)

    for sid in pilot_ids:
        npz_file = cache_path / f"{sid}-cam0-rgb.npz"
        if not npz_file.is_file():
            print(f"[!] Warning: Pose cache not found at {npz_file}")
            continue

        seq = load_urfd_sequence(npz_file, manifest)
        benchmark_sequences.append(seq)

    if not benchmark_sequences:
        print("[!] No cached pose sequences found to evaluate.")
        return 1

    print(f"\nRunning {len(benchmark_sequences)} sequence(s) through production FallDetector...")
    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(benchmark_sequences)

    print("Computing evaluation metrics...")
    report = evaluate_results(
        benchmark_sequences,
        results,
        required_confirmation_s=required_confirmation_s,
        dataset_type="URFD_PILOT",
        report_title="ARJUN FALL DETECTION -- URFD PILOT EVALUATION REPORT",
    )

    print("\n" + report.to_text() + "\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(evaluate_urfd_pilot())
