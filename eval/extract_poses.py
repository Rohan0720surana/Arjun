"""
Offline video-to-pose extraction pipeline for ARJUN fall-detection evaluation.

Extracts person bounding boxes, ByteTrack tracking IDs, and 17 COCO keypoints
from local video files or folders containing numbered RGB frames (e.g. URFD PNG
sequences) using YOLOv8-Pose without running the live dashboard.
Saves observations in a compact, versioned NPZ schema that can be loaded
directly into OfflineFallRunner.

NPZ Cache Schema (v1.0):
- schema_version       : ASCII string, currently "1.0"
- sequence_id          : Sequence identifier (e.g. "fall-01-cam0")
- name                 : Descriptive name of the sequence
- fps                  : Nominal frame rate of the video (float32)
- duration_s           : Total duration in seconds (float32)
- num_frames           : Total number of discrete video frames (int32)
- is_fall              : Ground-truth fall flag (-1=unknown/unverified, 0=false/ADL, 1=true/fall)
- fall_start_s         : Ground-truth fall start timestamp in seconds (NaN if unknown)
- fall_impact_s        : Ground-truth impact timestamp in seconds (NaN if unknown)
- frame_indices        : 1D array of frame indices [0 .. num_frames-1] (int32)
- frame_timestamps     : 1D array of video-derived timestamps in seconds (float32)
- det_frame_indices    : 1D array mapping each detection to its frame index (int32)
- det_track_ids        : 1D array of ByteTrack assigned track IDs (int32)
- det_confs            : 1D array of detection confidences (float32)
- det_boxes            : 2D array of bounding boxes [x1, y1, x2, y2] (float32, N x 4)
- det_keypoints        : 3D array of 17 COCO keypoints [x, y, conf] (float32, N x 17 x 3)
- det_has_keypoints    : 1D array indicating whether keypoints were detected (bool, N)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import config
from backend.vision import model_store, skeleton as sk
from backend.world import Detection
from eval.dataset import FrameObservation, Sequence, SequenceAnnotation

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Natural (Numerical) Sorting
# --------------------------------------------------------------------------- #

def natural_sort_key(path: Path | str) -> list[int | str]:
    """
    Sort key that orders numbers numerically rather than lexicographically.

    For example: ['frame-001.png', 'frame-002.png', 'frame-010.png'] instead of
    lexicographic pitfalls where 'frame-10' sorts before 'frame-2'.
    """
    name = path.name if isinstance(path, Path) else Path(path).name
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", name)
    ]


# --------------------------------------------------------------------------- #
# Tracker Reset API
# --------------------------------------------------------------------------- #

def reset_tracker_state(model: Any) -> None:
    """
    Safely reset YOLO / ByteTrack state between independent video recordings.

    Ultralytics stores tracker state inside `model.predictor.trackers`. Because
    internal library APIs may evolve, this helper:
    1. Resets individual tracker instances if `.reset()` / `.reset_id()` exist.
    2. Sets `model.predictor = None` to enforce clean tracker reinitialization
       on the next `track()` call without reloading model weights.
    """
    if model is None:
        return

    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tr in trackers:
                if hasattr(tr, "reset"):
                    try:
                        tr.reset()
                    except Exception:
                        pass
                if hasattr(tr, "reset_id"):
                    try:
                        tr.reset_id()
                    except Exception:
                        pass

        # Clearing predictor forces Ultralytics to re-instantiate clean trackers
        # upon the subsequent track(persist=True) call.
        try:
            model.predictor = None
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# NPZ Serialization & Deserialization
# --------------------------------------------------------------------------- #

def save_pose_cache(sequence: Sequence, output_path: str | Path) -> Path:
    """
    Save a Sequence of FrameObservations into a versioned NPZ cache file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ann = sequence.annotation
    num_frames = len(sequence.frames)

    frame_indices = np.empty(num_frames, dtype=np.int32)
    frame_timestamps = np.empty(num_frames, dtype=np.float32)

    det_frame_indices: list[int] = []
    det_track_ids: list[int] = []
    det_confs: list[float] = []
    det_boxes: list[list[float]] = []
    det_keypoints: list[np.ndarray] = []
    det_has_keypoints: list[bool] = []

    for i, obs in enumerate(sequence.frames):
        frame_indices[i] = obs.frame_idx
        frame_timestamps[i] = obs.timestamp

        for det in obs.detections:
            det_frame_indices.append(obs.frame_idx)
            det_track_ids.append(det.track_id)
            det_confs.append(float(det.conf))
            det_boxes.append(list(det.box))

            if det.keypoints is not None and det.keypoints.shape == (sk.N_KP, 3):
                det_keypoints.append(det.keypoints.astype(np.float32))
                det_has_keypoints.append(True)
            elif det.keypoints is not None and det.keypoints.shape == (sk.N_KP, 2):
                kp_padded = np.concatenate(
                    [det.keypoints, np.ones((sk.N_KP, 1), dtype=np.float32)], axis=1
                )
                det_keypoints.append(kp_padded.astype(np.float32))
                det_has_keypoints.append(True)
            else:
                det_keypoints.append(np.zeros((sk.N_KP, 3), dtype=np.float32))
                det_has_keypoints.append(False)

    num_det = len(det_frame_indices)
    det_boxes_arr = (
        np.array(det_boxes, dtype=np.float32) if num_det > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    det_keypoints_arr = (
        np.array(det_keypoints, dtype=np.float32) if num_det > 0
        else np.empty((0, sk.N_KP, 3), dtype=np.float32)
    )

    is_fall_val = -1
    if ann.is_fall is True:
        is_fall_val = 1
    elif ann.is_fall is False:
        is_fall_val = 0

    fall_start_val = ann.fall_start_s if ann.fall_start_s is not None else np.nan
    fall_impact_val = ann.fall_impact_s if ann.fall_impact_s is not None else np.nan
    fall_transition_val = (
        ann.fall_transition_start_s if ann.fall_transition_start_s is not None else np.nan
    )
    fallen_state_val = (
        ann.fallen_state_start_s if ann.fallen_state_start_s is not None else np.nan
    )

    np.savez_compressed(
        path,
        schema_version=np.array(SCHEMA_VERSION),
        sequence_id=np.array(ann.sequence_id),
        name=np.array(ann.name),
        fps=np.array(sequence.fps, dtype=np.float32),
        duration_s=np.array(ann.duration_s, dtype=np.float32),
        num_frames=np.array(num_frames, dtype=np.int32),
        is_fall=np.array(is_fall_val, dtype=np.int32),
        fall_start_s=np.array(fall_start_val, dtype=np.float32),
        fall_impact_s=np.array(fall_impact_val, dtype=np.float32),
        fall_transition_start_s=np.array(fall_transition_val, dtype=np.float32),
        fallen_state_start_s=np.array(fallen_state_val, dtype=np.float32),
        frame_indices=frame_indices,
        frame_timestamps=frame_timestamps,
        det_frame_indices=np.array(det_frame_indices, dtype=np.int32),
        det_track_ids=np.array(det_track_ids, dtype=np.int32),
        det_confs=np.array(det_confs, dtype=np.float32),
        det_boxes=det_boxes_arr,
        det_keypoints=det_keypoints_arr,
        det_has_keypoints=np.array(det_has_keypoints, dtype=bool),
    )

    return path


def load_pose_cache(
    npz_path: str | Path,
    annotation_override: SequenceAnnotation | None = None,
) -> Sequence:
    """
    Load a Sequence of FrameObservations from a versioned NPZ cache file.

    Parameters:
        npz_path: Path to the .npz file.
        annotation_override: Optional SequenceAnnotation to use instead of cached metadata.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pose cache not found: {path}")

    with np.load(path) as data:
        schema = str(data["schema_version"])
        if not schema.startswith("1."):
            raise ValueError(f"Unsupported cache schema version '{schema}' in {path}")

        sequence_id = str(data["sequence_id"])
        name = str(data["name"])
        fps = float(data["fps"])
        duration_s = float(data["duration_s"])

        is_fall_val = int(data["is_fall"])
        if is_fall_val == 1:
            is_fall: bool | None = True
        elif is_fall_val == 0:
            is_fall = False
        else:
            is_fall = None

        fall_start = float(data["fall_start_s"])
        fall_start_s = None if math.isnan(fall_start) else round(fall_start, 4)

        fall_impact = float(data["fall_impact_s"])
        fall_impact_s = None if math.isnan(fall_impact) else round(fall_impact, 4)

        fall_trans = float(data["fall_transition_start_s"]) if "fall_transition_start_s" in data else np.nan
        fall_transition_start_s = None if math.isnan(fall_trans) else round(fall_trans, 4)

        fallen_st = float(data["fallen_state_start_s"]) if "fallen_state_start_s" in data else np.nan
        fallen_state_start_s = None if math.isnan(fallen_st) else round(fallen_st, 4)

        frame_indices = data["frame_indices"]
        frame_timestamps = data["frame_timestamps"]

        det_frame_indices = data["det_frame_indices"]
        det_track_ids = data["det_track_ids"]
        det_confs = data["det_confs"]
        det_boxes = data["det_boxes"]
        det_keypoints = data["det_keypoints"]
        det_has_keypoints = data["det_has_keypoints"]

    # Index detections by frame_idx
    num_frames = len(frame_indices)
    frame_dets: dict[int, list[Detection]] = {int(idx): [] for idx in frame_indices}

    num_det = len(det_frame_indices)
    for i in range(num_det):
        f_idx = int(det_frame_indices[i])
        kps = det_keypoints[i] if det_has_keypoints[i] else None
        box = tuple(float(x) for x in det_boxes[i])

        det = Detection(
            track_id=int(det_track_ids[i]),
            label=config.CLASS_PERSON,
            conf=float(det_confs[i]),
            box=box,  # type: ignore[arg-type]
            keypoints=kps,
        )
        if f_idx in frame_dets:
            frame_dets[f_idx].append(det)

    frames: list[FrameObservation] = []
    for i in range(num_frames):
        f_idx = int(frame_indices[i])
        ts = float(frame_timestamps[i])
        frames.append(
            FrameObservation(
                frame_idx=f_idx,
                timestamp=ts,
                detections=frame_dets.get(f_idx, []),
            )
        )

    if annotation_override is not None:
        ann = annotation_override
    else:
        ann = SequenceAnnotation(
            sequence_id=sequence_id,
            name=name,
            is_fall=is_fall,
            duration_s=duration_s,
            fall_start_s=fall_start_s,
            fall_impact_s=fall_impact_s,
            fall_transition_start_s=fall_transition_start_s,
            fallen_state_start_s=fallen_state_start_s,
        )

    return Sequence(annotation=ann, fps=fps, frames=frames)


# --------------------------------------------------------------------------- #
# Video & Image Sequence Pose Extractor
# --------------------------------------------------------------------------- #

class VideoPoseExtractor:
    """
    Extracts person detections, ByteTrack IDs, and 17 COCO keypoints from video
    files or numbered image directories.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf: float | None = None,
        infer_size: tuple[int, int] | int | None = None,
        tracker_cfg: str | None = None,
        fallback_fps: float = 30.0,
    ) -> None:
        self.conf = conf if conf is not None else config.POSE_CONF
        self.infer_size = infer_size if infer_size is not None else config.INFER_SIZE
        self.tracker_cfg = tracker_cfg if tracker_cfg is not None else config.TRACKER_CFG
        self.fallback_fps = fallback_fps

        if weights_path is not None:
            self.weights_path = Path(weights_path)
        else:
            self.weights_path = model_store.ensure_yolo(config.YOLO_POSE_WEIGHTS)

        self._model = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
        return self._model

    def extract_from_video(
        self,
        video_path: str | Path,
        sequence_id: str | None = None,
        name: str | None = None,
        annotation: SequenceAnnotation | None = None,
    ) -> Sequence:
        """
        Extract poses frame-by-frame from a local video file.

        Stops at EOF and computes video-derived timestamps (t = frame_idx / fps).
        Resets ByteTrack state before and after extraction.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"cv2.VideoCapture failed to open: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0 or math.isnan(fps):
            fps = self.fallback_fps

        model = self._ensure_model()
        reset_tracker_state(model)

        seq_id = sequence_id or path.stem
        seq_name = name or seq_id

        frames: list[FrameObservation] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                ts = frame_idx / fps
                detections = self._infer_frame(model, frame)
                frames.append(
                    FrameObservation(
                        frame_idx=frame_idx,
                        timestamp=ts,
                        detections=detections,
                    )
                )
                frame_idx += 1
        finally:
            cap.release()
            reset_tracker_state(model)

        duration_s = (frame_idx / fps) if frame_idx > 0 else 0.0

        if annotation is not None:
            seq_ann = annotation
        else:
            seq_ann = SequenceAnnotation(
                sequence_id=seq_id,
                name=seq_name,
                is_fall=None,  # Unverified until ground truth is officially imported
                duration_s=duration_s,
                fall_start_s=None,
                fall_impact_s=None,
            )

        return Sequence(annotation=seq_ann, fps=fps, frames=frames)

    def extract_from_image_dir(
        self,
        image_dir: str | Path,
        fps: float = 30.0,
        sequence_id: str | None = None,
        name: str | None = None,
        annotation: SequenceAnnotation | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        timestamps: list[float] | np.ndarray | None = None,
    ) -> Sequence:
        """
        Extract poses from a directory of numbered image frames (e.g. URFD or UP-Fall PNG sequence).

        Frames are strictly sorted numerically rather than lexicographically.
        Timestamps are taken from the `timestamps` array if provided, or generated as t = frame_idx / fps.
        Corrupt or unreadable images are skipped with a warning.
        """
        dir_path = Path(image_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Image directory does not exist: {dir_path}")

        img_paths = [
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
        img_paths.sort(key=natural_sort_key)

        if not img_paths:
            raise ValueError(f"No image files with extensions {extensions} found in {dir_path}")

        model = self._ensure_model()
        reset_tracker_state(model)

        seq_id = sequence_id or dir_path.name
        seq_name = name or seq_id

        frames: list[FrameObservation] = []
        frame_idx = 0

        try:
            for img_path in img_paths:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"[WARN] Failed to read or parse image: {img_path}", file=sys.stderr)
                    continue

                if timestamps is not None and frame_idx < len(timestamps):
                    ts = float(timestamps[frame_idx])
                else:
                    ts = frame_idx / fps

                detections = self._infer_frame(model, frame)
                frames.append(
                    FrameObservation(
                        frame_idx=frame_idx,
                        timestamp=ts,
                        detections=detections,
                    )
                )
                frame_idx += 1
        finally:
            reset_tracker_state(model)

        if not frames:
            raise ValueError(f"Directory {dir_path} contains no readable image frames.")

        if timestamps is not None and len(timestamps) > 0:
            duration_s = float(timestamps[-1])
        else:
            duration_s = (frame_idx / fps) if frame_idx > 0 else 0.0

        if annotation is not None:
            seq_ann = annotation
        else:
            seq_ann = SequenceAnnotation(
                sequence_id=seq_id,
                name=seq_name,
                is_fall=None,  # Unverified until ground truth is officially imported
                duration_s=duration_s,
                fall_start_s=None,
                fall_impact_s=None,
            )

        return Sequence(annotation=seq_ann, fps=fps, frames=frames)

    def _infer_frame(self, model: Any, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8-Pose tracking on a single frame and convert to Detections."""
        results = model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            imgsz=self.infer_size,
            conf=self.conf,
            verbose=False,
        )
        if not results:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = (
            boxes.conf.cpu().numpy() if boxes.conf is not None
            else np.ones(len(xyxy), np.float32)
        )
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array([-(i + 1) for i in range(len(xyxy))], dtype=int)

        kps_all = None
        kp = getattr(res, "keypoints", None)
        if kp is not None and getattr(kp, "data", None) is not None:
            raw_data = kp.data
            if hasattr(raw_data, "cpu"):
                raw_data = raw_data.cpu()
            if hasattr(raw_data, "numpy"):
                raw_data = raw_data.numpy()
            if len(raw_data) > 0:
                kps_all = raw_data

        detections: list[Detection] = []
        for i, box in enumerate(xyxy):
            kps = None
            if kps_all is not None and i < len(kps_all):
                arr = np.asarray(kps_all[i], dtype=np.float32)
                if arr.shape == (sk.N_KP, 3):
                    kps = arr
                elif arr.shape == (sk.N_KP, 2):
                    kps = np.concatenate(
                        [arr, np.ones((sk.N_KP, 1), np.float32)], axis=1
                    )

            detections.append(
                Detection(
                    track_id=int(ids[i]),
                    label=config.CLASS_PERSON,
                    conf=float(confs[i]),
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    keypoints=kps,
                )
            )

        return detections


# --------------------------------------------------------------------------- #
# Command-Line Interface
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARJUN Offline Video/Image-to-Pose Extractor for Fall Detection Evaluation."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="One or more paths to input video files (.mp4, .avi) or frame folders.",
    )
    parser.add_argument(
        "--frames-dir",
        "--frames-dirs",
        nargs="+",
        dest="frames_dirs",
        help="One or more directories containing numbered RGB image frames (e.g. URFD PNG sequence).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing video files to extract.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate assumption for image sequences (default: 30.0, assumed unverified FPS).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/cache/urfd",
        help="Directory where extracted NPZ pose caches will be stored (default: eval/cache/urfd).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the extracted sequence through OfflineFallRunner immediately after extraction.",
    )

    args = parser.parse_args()

    video_inputs: list[Path] = []
    frame_dirs: list[Path] = []

    if args.videos:
        for v in args.videos:
            p = Path(v)
            if p.is_file():
                video_inputs.append(p)
            elif p.is_dir():
                frame_dirs.append(p)
            else:
                print(f"[WARN] Input path not found: {p}", file=sys.stderr)

    if args.frames_dirs:
        for fd in args.frames_dirs:
            p = Path(fd)
            if p.is_dir():
                frame_dirs.append(p)
            else:
                print(f"[WARN] Frame directory not found: {p}", file=sys.stderr)

    if args.video_dir:
        vdir = Path(args.video_dir)
        if vdir.is_dir():
            for ext in ("*.mp4", "*.avi"):
                video_inputs.extend(sorted(vdir.glob(ext)))
        else:
            print(f"[WARN] Video directory not found: {vdir}", file=sys.stderr)

    if not video_inputs and not frame_dirs:
        print(
            "No valid video or frame directory inputs specified.\n"
            "Example usage for a folder of URFD PNG frames:\n"
            "  python eval/extract_poses.py \\\n"
            "    --frames-dir data/urfd/fall-01-cam0-rgb \\\n"
            "    --fps 30.0 \\\n"
            "    --output-dir eval/cache/urfd \\\n"
            "    --eval\n"
        )
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing YOLOv8-Pose Extractor (weights: {config.YOLO_POSE_WEIGHTS})...")
    extractor = VideoPoseExtractor()

    from eval.runner import OfflineFallRunner
    runner = OfflineFallRunner() if args.eval else None

    # 1. Process Video Files
    for vpath in video_inputs:
        seq_id = vpath.stem
        npz_target = out_dir / f"{seq_id}.npz"

        print(f"\n--> Extracting video: {vpath.name} -> {npz_target}")
        seq = extractor.extract_from_video(vpath, sequence_id=seq_id)
        save_pose_cache(seq, npz_target)

        num_det = sum(len(f.detections) for f in seq.frames)
        print(
            f"    Saved: {len(seq.frames)} frames, {seq.annotation.duration_s:.2f}s, "
            f"{num_det} person detection(s)."
        )

        if runner is not None:
            res = runner.evaluate_sequence(seq)
            alert_count = len(res.alerts)
            ts_str = f" at {res.alert_timestamps}" if alert_count > 0 else ""
            print(f"    FallDetector Output: {alert_count} alert(s){ts_str}")
            if seq.annotation.is_fall is None:
                print("    [Note: Ground truth unverified; accuracy metrics skipped]")

    # 2. Process Image Directories
    for fdir in frame_dirs:
        seq_id = fdir.name
        npz_target = out_dir / f"{seq_id}.npz"

        print(f"\n--> Extracting frame sequence: {fdir.name} (assumed FPS={args.fps}) -> {npz_target}")
        seq = extractor.extract_from_image_dir(fdir, fps=args.fps, sequence_id=seq_id)
        save_pose_cache(seq, npz_target)

        num_det = sum(len(f.detections) for f in seq.frames)
        print(
            f"    Saved: {len(seq.frames)} frames, {seq.annotation.duration_s:.2f}s, "
            f"{num_det} person detection(s)."
        )

        if runner is not None:
            res = runner.evaluate_sequence(seq)
            alert_count = len(res.alerts)
            ts_str = f" at {res.alert_timestamps}" if alert_count > 0 else ""
            print(f"    FallDetector Output: {alert_count} alert(s){ts_str}")
            if seq.annotation.is_fall is None:
                print("    [Note: Ground truth unverified; accuracy metrics skipped]")

    print("\nExtraction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
