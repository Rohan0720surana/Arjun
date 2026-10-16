"""
The per-frame loop: pixels in, anonymous view and incidents out.

One :class:`Pipeline` belongs to one camera worker and runs entirely on that
worker's thread. The order of operations is the privacy argument made
executable:

1.  **See.** The models turn the frame into a :class:`~backend.world.FrameWorld`
    — boxes, track ids, skeletons. From this line onwards the frame itself is
    needed for exactly two things: the memory-only ring buffer, and the small
    crops that the hand and face models look at.
2.  **Remember.** Track histories are updated, zone timers advanced.
3.  **Reason.** The detectors run against that history. None of them can see
    the frame.
4.  **Decide.** Candidates go to the alert manager, which applies cooldowns,
    writes to the database and starts an incident clip.
5.  **Draw.** The privacy view is rendered from the world onto a fresh canvas.

The frame is never a layer in step 5, which is why there is no setting, bug or
race that could put a face on the dashboard.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

import config
from backend.camera_manager import CameraWorker, ProcessedFrame
from backend.detectors import base as detector_base
from backend.privacy import Heatmap, PrivacyRenderer
from backend.state import CameraContext, TrackStore
from backend.vision.engine import VisionEngine, world_from_sim
from backend.world import FrameWorld
from backend.zone_store import ZoneSet

ZoneProvider = Callable[[str], ZoneSet]
AlertSink = Callable[[list, CameraContext], list]
HighlightSource = Callable[[str, float], dict[int, str]]
SnapshotSink = Callable[[list, np.ndarray], None]


class Pipeline:
    """Per-camera processing. Lives on, and is only touched by, one thread."""

    def __init__(
        self,
        camera: str,
        zone_provider: ZoneProvider | None = None,
        alert_sink: AlertSink | None = None,
        highlight_source: HighlightSource | None = None,
        snapshot_sink: SnapshotSink | None = None,
    ) -> None:
        self.camera = camera
        self.engine = VisionEngine(camera)
        self.renderer = PrivacyRenderer()
        self.tracks = TrackStore()
        self.detectors = detector_base.build_all()
        self.zone_provider = zone_provider
        self.alert_sink = alert_sink
        self.highlight_source = highlight_source
        self.snapshot_sink = snapshot_sink

        self.heatmap: Heatmap | None = None
        self._loaded = False
        self._load_status: dict[str, Any] = {}
        self._zones: ZoneSet = ZoneSet()
        self._zones_fetched = 0.0
        self.timings: dict[str, float] = {"infer": 0.0, "rules": 0.0,
                                          "render": 0.0}
        self.last_world: FrameWorld | None = None
        self.last_note = ""

    # -- lifecycle ------------------------------------------------------------- #

    def ensure_loaded(self) -> dict[str, Any]:
        if not self._loaded:
            self._loaded = True
            self._load_status = self.engine.load()
        return self._load_status

    def reset(self, worker: CameraWorker | None = None) -> None:
        """Called when a camera reconnects: old identities are meaningless."""
        self.tracks.reset()
        for det in self.detectors:
            det.reset()
        if self.heatmap is not None:
            self.heatmap.reset()

    # -- zones ------------------------------------------------------------------ #

    def zones(self, force: bool = False) -> ZoneSet:
        """Zones are re-read every couple of seconds so the editor feels live."""
        if self.zone_provider is None:
            return self._zones
        if force or time.time() - self._zones_fetched > 2.0:
            try:
                self._zones = self.zone_provider(self.camera)
            except Exception:                               # noqa: BLE001
                pass                                        # keep the old set
            self._zones_fetched = time.time()
        return self._zones

    def invalidate_zones(self) -> None:
        self._zones_fetched = 0.0

    # -- the frame ----------------------------------------------------------------- #

    def process(self, worker: CameraWorker, frame: np.ndarray,
                frame_idx: int, ts: float) -> ProcessedFrame:
        self.ensure_loaded()
        h, w = frame.shape[:2]

        # 1. see ----------------------------------------------------------- #
        t0 = time.perf_counter()
        sim_payload = None
        src = getattr(worker, "_src", None)
        if src is not None:
            sim_payload = src.synthetic_world()
        if sim_payload is not None:
            world = world_from_sim(self.camera, sim_payload, frame_idx, ts, w, h)
        else:
            world = self.engine.infer(frame, frame_idx, ts)
        t1 = time.perf_counter()

        # 2. remember -------------------------------------------------------- #
        self.tracks.update(world)
        ctx = CameraContext(
            camera=self.camera, world=world, tracks=self.tracks,
            zones=self.zones(), flags=worker.flags, now=ts,
            simulated=world.simulated,
        )
        ctx.shared["frame"] = frame          # only the crop-based rules use this
        ctx.shared["worker"] = worker
        ctx.shared["engine"] = self.engine
        ctx.compute_memberships()
        ctx.update_zone_timers()

        # 3. reason ----------------------------------------------------------- #
        candidates: list = []
        for detector in self.detectors:
            candidates.extend(detector.safe_update(ctx))
        t2 = time.perf_counter()

        # 4. decide ------------------------------------------------------------ #
        fired: list = []
        if self.alert_sink is not None and candidates:
            try:
                fired = self.alert_sink(candidates, ctx) or []
            except Exception as exc:                        # noqa: BLE001
                print(f"[{self.camera}] alert sink failed: {exc}")

        # 5. draw ---------------------------------------------------------------- #
        if worker.flags.heatmap:
            if self.heatmap is None or self.heatmap.w * self.heatmap.scale != w:
                self.heatmap = Heatmap(w, h)
            self.heatmap.add([p.foot() for p in world.persons])

        highlights: dict[int, str] = {}
        if self.highlight_source is not None:
            try:
                highlights = self.highlight_source(self.camera, ts)
            except Exception:                               # noqa: BLE001
                highlights = {}
        for cand in candidates:
            if cand.highlight and cand.track_id is not None:
                highlights.setdefault(cand.track_id, cand.highlight)

        display = self.renderer.render(
            world,
            zones=ctx.zones,
            heatmap=self.heatmap if worker.flags.heatmap else None,
            highlights=highlights,
            zone_counts=ctx.zone_counts,
            hud={
                "fps": worker.fps,
                "exam_mode": worker.flags.exam_mode,
                "heatmap": worker.flags.heatmap,
                "demo": config.demo_mode(),
                "after_hours": ctx.shared.get("after_hours", False),
            },
        )
        t3 = time.perf_counter()

        if fired and self.snapshot_sink is not None:
            try:
                self.snapshot_sink(fired, display)
            except Exception as exc:                        # noqa: BLE001
                print(f"[{self.camera}] snapshot sink failed: {exc}")

        a = 0.85
        self.timings["infer"] = self.timings["infer"] * a + (t1 - t0) * 1000 * (1 - a)
        self.timings["rules"] = self.timings["rules"] * a + (t2 - t1) * 1000 * (1 - a)
        self.timings["render"] = self.timings["render"] * a + (t3 - t2) * 1000 * (1 - a)

        self.last_world = world
        self.last_note = world.note

        return ProcessedFrame(
            display=display,
            people=len(world.persons),
            meta={
                "objects": len(world.objects),
                "tracks": len(self.tracks),
                "zone_counts": ctx.zone_counts,
                "timings": {k: round(v, 1) for k, v in self.timings.items()},
                "models": self._load_status,
                "note": world.note,
                "scene_t": (sim_payload or {}).get("scene_t"),
                "counters": ctx.shared.get("counters", {}),
                "after_hours": ctx.shared.get("after_hours", False),
            },
        )

    # -- introspection -------------------------------------------------------------- #

    def detector_status(self) -> list[dict[str, Any]]:
        rows = [d.status() for d in self.detectors]
        for ref, reason in detector_base.UNAVAILABLE.items():
            rows.append({"name": ref, "title": ref, "explanation": "",
                         "fired": 0, "errors": 1, "last_error": reason,
                         "cost_ms": 0.0, "muted": True})
        return rows
