"""
Unit tests for the offline video-to-pose extraction pipeline.

Tests video frame reading, image directory sequence ingestion, numerical frame ordering,
video-derived timestamps, tracker state resetting, NPZ cache schema
serialization/deserialization, and runner integration without requiring external
dataset downloads.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from backend.world import Detection
from eval.dataset import FrameObservation, Sequence, SequenceAnnotation
from eval.extract_poses import (
    SCHEMA_VERSION,
    VideoPoseExtractor,
    load_pose_cache,
    natural_sort_key,
    reset_tracker_state,
    save_pose_cache,
)
from eval.metrics import evaluate_results
from eval.runner import OfflineFallRunner, SequenceResult


# --------------------------------------------------------------------------- #
# Helpers & Fixtures
# --------------------------------------------------------------------------- #

def _create_dummy_video(path: Path, num_frames: int = 15, fps: float = 30.0) -> Path:
    """Create a minimal playable video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    for i in range(num_frames):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _create_dummy_image(path: Path, val: int = 128) -> Path:
    """Create a minimal PNG image."""
    img = np.full((48, 64, 3), val, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _mock_yolo_result(with_person: bool = True) -> MagicMock:
    """Create a mocked Ultralytics Results object with bounding boxes and keypoints."""
    res = MagicMock()
    if not with_person:
        res.boxes = None
        res.keypoints = None
        return res

    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [[10.0, 10.0, 50.0, 40.0]], dtype=np.float32
    )
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95], dtype=np.float32)
    boxes.id.cpu.return_value.numpy.return_value.astype.return_value = np.array([1], dtype=int)
    res.boxes = boxes

    kps = np.zeros((1, 17, 3), dtype=np.float32)
    # Give nose, shoulders, hips non-zero coordinates and confidences
    kps[0, 0] = [30.0, 15.0, 0.9]
    kps[0, 5] = [20.0, 20.0, 0.85]
    kps[0, 6] = [40.0, 20.0, 0.85]

    keypoints = MagicMock()
    keypoints.data = kps
    res.keypoints = keypoints

    return res


# --------------------------------------------------------------------------- #
# Natural (Numerical) Sorting
# --------------------------------------------------------------------------- #

def test_natural_sort_key_orders_numerically():
    """natural_sort_key orders numbered files by numeric value instead of character code."""
    raw_names = [
        Path("frame_10.png"),
        Path("frame_2.png"),
        Path("frame_1.png"),
        Path("frame_20.png"),
    ]
    sorted_names = sorted(raw_names, key=natural_sort_key)
    assert [p.name for p in sorted_names] == [
        "frame_1.png",
        "frame_2.png",
        "frame_10.png",
        "frame_20.png",
    ]

    urfd_names = [
        Path("fall-01-cam0-rgb-010.png"),
        Path("fall-01-cam0-rgb-002.png"),
        Path("fall-01-cam0-rgb-001.png"),
        Path("fall-01-cam0-rgb-160.png"),
    ]
    sorted_urfd = sorted(urfd_names, key=natural_sort_key)
    assert [p.name for p in sorted_urfd] == [
        "fall-01-cam0-rgb-001.png",
        "fall-01-cam0-rgb-002.png",
        "fall-01-cam0-rgb-010.png",
        "fall-01-cam0-rgb-160.png",
    ]


# --------------------------------------------------------------------------- #
# Image Directory Ingestion & Timestamps
# --------------------------------------------------------------------------- #

def test_extract_from_image_dir_orders_numerically_and_generates_timestamps(tmp_path: Path):
    """extract_from_image_dir ingests PNGs in numeric order and calculates timestamps via fps."""
    img_dir = tmp_path / "frames"
    img_dir.mkdir()

    # Create frames out of order
    _create_dummy_image(img_dir / "frame_10.png", val=100)
    _create_dummy_image(img_dir / "frame_2.png", val=20)
    _create_dummy_image(img_dir / "frame_1.png", val=10)

    extractor = VideoPoseExtractor()
    mock_model = MagicMock()
    mock_model.track.return_value = [_mock_yolo_result(with_person=True)]
    extractor._model = mock_model

    seq = extractor.extract_from_image_dir(
        img_dir,
        fps=20.0,
        sequence_id="test-frame-seq",
        name="Test Frames",
    )

    assert seq.annotation.sequence_id == "test-frame-seq"
    assert seq.annotation.is_fall is None  # Unverified
    assert seq.fps == 20.0
    assert len(seq.frames) == 3
    assert seq.annotation.duration_s == pytest.approx(3 / 20.0, abs=0.001)

    # Frame 0 is frame_1.png, Frame 1 is frame_2.png, Frame 2 is frame_10.png
    for i, obs in enumerate(seq.frames):
        assert obs.frame_idx == i
        assert obs.timestamp == pytest.approx(i / 20.0, abs=0.001)
        assert len(obs.detections) == 1
        assert obs.detections[0].track_id == 1


def test_extract_from_image_dir_skips_corrupt_images(tmp_path: Path):
    """Corrupt or non-image files are skipped gracefully without crashing extraction."""
    img_dir = tmp_path / "corrupt_frames"
    img_dir.mkdir()

    _create_dummy_image(img_dir / "valid_01.png", val=10)
    # Corrupt image (text file with .png extension)
    (img_dir / "valid_02.png").write_text("not an image")
    _create_dummy_image(img_dir / "valid_03.png", val=30)

    extractor = VideoPoseExtractor()
    mock_model = MagicMock()
    mock_model.track.return_value = [_mock_yolo_result(with_person=True)]
    extractor._model = mock_model

    seq = extractor.extract_from_image_dir(img_dir, fps=10.0)
    # Only the 2 valid images should be processed
    assert len(seq.frames) == 2
    assert seq.frames[0].frame_idx == 0
    assert seq.frames[1].frame_idx == 1


def test_extract_from_image_dir_errors_on_empty_or_missing_directory(tmp_path: Path):
    """Empty or non-existent directories raise clear errors."""
    extractor = VideoPoseExtractor()

    with pytest.raises(NotADirectoryError):
        extractor.extract_from_image_dir(tmp_path / "non_existent_folder")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No image files"):
        extractor.extract_from_image_dir(empty_dir)


# --------------------------------------------------------------------------- #
# Video Ingestion & Video-Derived Timestamps
# --------------------------------------------------------------------------- #

def test_extract_from_video_reads_frames_and_timestamps(tmp_path: Path):
    """VideoPoseExtractor reads frames to EOF and derives timestamps from frame index and FPS."""
    video_path = _create_dummy_video(tmp_path / "test.avi", num_frames=10, fps=20.0)

    extractor = VideoPoseExtractor(fallback_fps=20.0)
    mock_model = MagicMock()
    mock_model.track.return_value = [_mock_yolo_result(with_person=True)]
    extractor._model = mock_model

    seq = extractor.extract_from_video(video_path, sequence_id="test-seq", name="Test Video")

    assert seq.annotation.sequence_id == "test-seq"
    assert seq.annotation.name == "Test Video"
    assert seq.annotation.is_fall is None  # Unverified
    assert len(seq.frames) == 10
    assert seq.fps == pytest.approx(20.0, abs=0.1)
    assert seq.annotation.duration_s == pytest.approx(0.5, abs=0.01)

    for i, obs in enumerate(seq.frames):
        assert obs.frame_idx == i
        assert obs.timestamp == pytest.approx(i / 20.0, abs=0.001)
        assert len(obs.detections) == 1
        assert obs.detections[0].track_id == 1
        assert obs.detections[0].keypoints is not None
        assert obs.detections[0].keypoints.shape == (17, 3)


def test_extract_from_video_handles_missing_detections(tmp_path: Path):
    """Frames with no person detections produce valid observations with empty detection lists."""
    video_path = _create_dummy_video(tmp_path / "empty.avi", num_frames=5, fps=10.0)

    extractor = VideoPoseExtractor(fallback_fps=10.0)
    mock_model = MagicMock()
    mock_model.track.return_value = [_mock_yolo_result(with_person=False)]
    extractor._model = mock_model

    seq = extractor.extract_from_video(video_path)
    assert len(seq.frames) == 5
    for obs in seq.frames:
        assert len(obs.detections) == 0


def test_extract_from_video_file_not_found():
    """Attempting to extract a non-existent video file raises FileNotFoundError."""
    extractor = VideoPoseExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_video("non_existent_file.mp4")


# --------------------------------------------------------------------------- #
# Tracker Reset State
# --------------------------------------------------------------------------- #

def test_reset_tracker_state_resets_trackers_and_clears_predictor():
    """reset_tracker_state safely invokes tracker reset methods and clears predictor."""
    mock_tracker = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.trackers = [mock_tracker]

    mock_model = MagicMock()
    mock_model.predictor = mock_predictor

    reset_tracker_state(mock_model)

    assert mock_tracker.reset.called
    assert mock_tracker.reset_id.called
    assert mock_model.predictor is None


def test_reset_tracker_state_handles_none_or_missing_attributes():
    """reset_tracker_state handles models with None or missing attributes gracefully."""
    reset_tracker_state(None)

    empty_model = object()
    reset_tracker_state(empty_model)


# --------------------------------------------------------------------------- #
# NPZ Serialization & Deserialization
# --------------------------------------------------------------------------- #

def test_save_and_load_pose_cache_roundtrip(tmp_path: Path):
    """Saving a Sequence to NPZ and reloading it recovers identical annotations and detections."""
    kps1 = np.ones((17, 3), dtype=np.float32)
    kps1[:, :2] = np.random.uniform(10.0, 100.0, (17, 2)).astype(np.float32)

    det1 = Detection(
        track_id=1, label="person", conf=0.92,
        box=(10.0, 15.0, 50.0, 90.0), keypoints=kps1,
    )
    det2 = Detection(
        track_id=2, label="person", conf=0.88,
        box=(60.0, 20.0, 100.0, 95.0), keypoints=None,  # Box-only detection
    )

    frames = [
        FrameObservation(frame_idx=0, timestamp=0.0, detections=[det1, det2]),
        FrameObservation(frame_idx=1, timestamp=0.033, detections=[]),  # Empty frame
        FrameObservation(frame_idx=2, timestamp=0.066, detections=[det1]),
    ]

    ann = SequenceAnnotation(
        sequence_id="test-cache-01",
        name="Test Cache Roundtrip",
        is_fall=True,
        duration_s=0.1,
        fall_start_s=0.03,
        fall_impact_s=0.06,
    )

    original_seq = Sequence(annotation=ann, fps=30.0, frames=frames)

    cache_file = tmp_path / "test-cache.npz"
    save_pose_cache(original_seq, cache_file)
    assert cache_file.is_file()

    loaded_seq = load_pose_cache(cache_file)

    assert loaded_seq.annotation.sequence_id == "test-cache-01"
    assert loaded_seq.annotation.name == "Test Cache Roundtrip"
    assert loaded_seq.annotation.is_fall is True
    assert loaded_seq.annotation.fall_start_s == pytest.approx(0.03, abs=0.001)
    assert loaded_seq.annotation.fall_impact_s == pytest.approx(0.06, abs=0.001)
    assert loaded_seq.fps == pytest.approx(30.0, abs=0.001)
    assert len(loaded_seq.frames) == 3

    # Frame 0: 2 detections
    f0 = loaded_seq.frames[0]
    assert len(f0.detections) == 2
    assert f0.detections[0].track_id == 1
    assert f0.detections[0].box == (10.0, 15.0, 50.0, 90.0)
    np.testing.assert_allclose(f0.detections[0].keypoints, kps1, atol=1e-5)
    assert f0.detections[1].track_id == 2
    assert f0.detections[1].keypoints is None  # Preserved as None

    # Frame 1: empty detections
    f1 = loaded_seq.frames[1]
    assert len(f1.detections) == 0

    # Frame 2: 1 detection
    f2 = loaded_seq.frames[2]
    assert len(f2.detections) == 1
    assert f2.detections[0].track_id == 1


def test_unverified_annotation_excluded_from_metrics():
    """Unverified sequences (is_fall=None) are excluded from fall/ADL counts and accuracy metrics."""
    ann = SequenceAnnotation(
        sequence_id="S-UNVERIFIED", name="Unverified Sequence",
        is_fall=None, duration_s=5.0,
    )
    benchmark = [Sequence(annotation=ann, fps=30.0)]
    results = [
        SequenceResult(
            sequence_id="S-UNVERIFIED",
            annotation_name="Unverified Sequence",
            is_fall_ground_truth=None,
            alerts=[],
            alert_timestamps=[],
        )
    ]

    report = evaluate_results(benchmark, results)
    assert report.fall_sequences == 0
    assert report.adl_sequences == 0
    assert report.tp == 0
    assert report.tn == 0
    assert report.fp == 0
    assert report.fn == 0
    assert report.sequence_evaluations[0].verdict == "UNVERIFIED"


def test_load_pose_cache_file_not_found():
    """Attempting to load a non-existent NPZ file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pose_cache("does_not_exist.npz")


def test_load_pose_cache_invalid_schema(tmp_path: Path):
    """Loading an NPZ with an incompatible schema version raises ValueError."""
    bad_file = tmp_path / "bad_schema.npz"
    np.savez(bad_file, schema_version=np.array("99.0"))

    with pytest.raises(ValueError, match="Unsupported cache schema version"):
        load_pose_cache(bad_file)


# --------------------------------------------------------------------------- #
# Runner Integration
# --------------------------------------------------------------------------- #

def test_runner_evaluates_loaded_cached_sequence(tmp_path: Path):
    """An NPZ-loaded Sequence runs cleanly through OfflineFallRunner without errors."""
    frames = [
        FrameObservation(frame_idx=0, timestamp=0.0, detections=[]),
        FrameObservation(frame_idx=1, timestamp=0.1, detections=[]),
    ]
    ann = SequenceAnnotation(
        sequence_id="test-runner-seq",
        name="Runner Integration",
        is_fall=False,
        duration_s=0.2,
    )
    seq = Sequence(annotation=ann, fps=10.0, frames=frames)

    cache_file = tmp_path / "runner_test.npz"
    save_pose_cache(seq, cache_file)

    loaded_seq = load_pose_cache(cache_file)

    runner = OfflineFallRunner()
    result = runner.evaluate_sequence(loaded_seq)

    assert result.sequence_id == "test-runner-seq"
    assert result.total_frames == 2
    assert result.duration_s == pytest.approx(0.2, abs=0.01)
    assert len(result.alerts) == 0
