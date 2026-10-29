"""
Offline evaluation runner for ARJUN fall detection.

Drives the real FallDetector over standardized sequences using an isolated
virtual clock and track store, with strict state isolation between sequences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence as Seq

import config
from backend.camera_manager import CameraFlags
from backend.detectors.base import Candidate
from backend.detectors.fall import FallDetector
from backend.state import CameraContext, TrackStore
from backend.world import FrameWorld
from backend.zone_store import ZoneSet
from eval.dataset import Sequence


@dataclass
class SequenceResult:
    """Evaluation output for a single sequence."""
    sequence_id: str
    annotation_name: str
    is_fall_ground_truth: bool | None = None
    alerts: list[Candidate] = field(default_factory=list)
    alert_timestamps: list[float] = field(default_factory=list)
    total_frames: int = 0
    duration_s: float = 0.0
    execution_time_ms: float = 0.0


class OfflineFallRunner:
    """
    Offline execution engine for evaluating the production FallDetector.

    Avoids importing test fixtures or starting FastAPI/Uvicorn servers.
    Provides deterministic stepping and clean isolation between runs.
    """

    def __init__(self, detector: FallDetector | None = None) -> None:
        self.detector = detector if detector is not None else FallDetector()

    def evaluate_sequence(self, sequence: Sequence) -> SequenceResult:
        """
        Run a single sequence through the fall detector with clean state.

        The detector scratchpad and track history are guaranteed fresh at the
        start of each sequence to prevent cross-sequence contamination.
        """
        # Ensure we evaluate against production thresholds, not shortened demo ones
        was_demo = config.demo_mode()
        config.set_demo_mode(False)

        tracks = TrackStore()
        zones = ZoneSet([])
        flags = CameraFlags()
        camera = f"eval::{sequence.annotation.sequence_id}"

        alerts: list[Candidate] = []
        alert_timestamps: list[float] = []

        t0 = time.perf_counter()

        try:
            for obs in sequence.frames:
                world = FrameWorld(
                    camera=camera,
                    ts=obs.timestamp,
                    frame_idx=obs.frame_idx,
                    width=640,
                    height=360,
                    persons=obs.detections,
                    objects=[],
                    simulated=True,
                )
                tracks.update(world)

                ctx = CameraContext(
                    camera=camera,
                    world=world,
                    tracks=tracks,
                    zones=zones,
                    flags=flags,
                    now=obs.timestamp,
                    simulated=True,
                )
                ctx.compute_memberships()
                ctx.update_zone_timers()

                candidates = self.detector.safe_update(ctx)
                for cand in candidates:
                    if cand.type == "fall":
                        alerts.append(cand)
                        alert_timestamps.append(obs.timestamp)

        finally:
            config.set_demo_mode(was_demo)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return SequenceResult(
            sequence_id=sequence.annotation.sequence_id,
            annotation_name=sequence.annotation.name,
            is_fall_ground_truth=sequence.annotation.is_fall,
            alerts=alerts,
            alert_timestamps=alert_timestamps,
            total_frames=len(sequence.frames),
            duration_s=sequence.annotation.duration_s,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def evaluate_benchmark(self, benchmark: list[Sequence]) -> list[SequenceResult]:
        """Evaluate a full collection of sequences sequentially."""
        return [self.evaluate_sequence(seq) for seq in benchmark]
