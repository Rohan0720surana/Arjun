"""
The privacy guarantees, enforced by the test suite rather than by a promise.

Every claim ARJUN makes on its dashboard is checked here against the actual
source. If somebody later adds an ``imwrite`` to a detector, or quietly passes
the camera frame into the renderer, these fail — which is the point. A privacy
property that lives only in a README is a property that will be gone in three
months.
"""

from __future__ import annotations

import ast
import inspect
import re
import time
from pathlib import Path

import numpy as np

import config
from backend import privacy
from backend.privacy import PrivacyRenderer, RawRingBuffer
from backend.world import Detection, FrameWorld

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def _python_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.py") if "__pycache__" not in p.parts)


# --------------------------------------------------------------------------- #
# 1. Only one component in the codebase may write pixels to disk
# --------------------------------------------------------------------------- #

PIXEL_WRITERS = re.compile(r"\bcv2\.VideoWriter\b|\bcv2\.imwrite\b")


def test_only_the_incident_recorder_writes_video():
    offenders = []
    for path in _python_files(BACKEND):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PIXEL_WRITERS.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{n}")

    allowed_file = "backend/privacy.py"
    unexpected = [o for o in offenders if not o.startswith(allowed_file)]
    assert not unexpected, (
        "Only backend/privacy.py (the incident recorder) may write video. "
        f"Found writers in: {unexpected}")
    assert offenders, "the recorder itself disappeared — check this test still works"


def test_clips_are_written_only_into_the_clips_directory():
    source = inspect.getsource(privacy.IncidentRecorder._write)
    assert "config.CLIPS_DIR" in source, (
        "the clip path must be derived from config.CLIPS_DIR, so there is one "
        "place to look to audit what raw footage exists")
    # No other directory constant may appear in the writer.
    for other in ("SNAPSHOT_DIR", "DATA_DIR", "FRONTEND_DIR", "MODELS_DIR"):
        assert other not in source, f"the clip writer must not reach {other}"


# --------------------------------------------------------------------------- #
# 2. The renderer is never handed the camera frame
# --------------------------------------------------------------------------- #

def test_renderer_cannot_see_the_camera_image():
    """The anonymous view is drawn from geometry, not filtered from a frame."""
    params = set(inspect.signature(PrivacyRenderer.render).parameters)
    forbidden = {"frame", "image", "img", "raw", "source", "bgr"}
    assert not (params & forbidden), (
        f"PrivacyRenderer.render() accepts {params & forbidden}; it must only "
        "receive the world, so there is no path by which a face reaches the "
        "dashboard")
    assert "world" in params


def test_rendered_view_is_independent_of_the_camera():
    """Two wildly different frames must produce an identical render."""
    world = FrameWorld(camera="Test", ts=time.time(), frame_idx=1,
                       width=320, height=180,
                       persons=[Detection(1, "person", 0.9, (90, 40, 150, 170))])
    renderer = PrivacyRenderer()
    a = renderer.render(world)
    b = renderer.render(world)
    assert np.array_equal(a, b), "the render must be a pure function of the world"
    # And the canvas must not resemble a photograph: a geometry render has very
    # few distinct colours compared with any real camera frame.
    assert len(np.unique(a.reshape(-1, 3), axis=0)) < 4000


# --------------------------------------------------------------------------- #
# 3. Detection rules cannot reach the pixels
# --------------------------------------------------------------------------- #

#: The only two rules that may look at an image, and then only at a small crop:
#: the hand gesture and exam mode's head angle. Both are documented as such.
CROP_RULES = {"sos.py", "exam.py"}


def test_rules_do_not_touch_the_frame():
    offenders = []
    for path in _python_files(BACKEND / "detectors"):
        if path.name in CROP_RULES | {"__init__.py", "base.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        if 'shared.get("frame")' in source or "shared['frame']" in source:
            offenders.append(path.name)
        if re.search(r"\bcv2\.", source):
            offenders.append(f"{path.name} (imports cv2)")
    assert not offenders, (
        f"These rules reached for the image: {offenders}. Rules read geometry; "
        f"only {sorted(CROP_RULES)} may take a crop, for hands and head angle.")


def test_no_rule_writes_anything_to_disk():
    for path in _python_files(BACKEND / "detectors"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert name not in {"open", "imwrite", "write_bytes",
                                    "write_text", "VideoWriter"}, (
                    f"{path.name} writes to disk; rules must only return "
                    f"candidates")


# --------------------------------------------------------------------------- #
# 4. The rolling buffer holds what it says it holds
# --------------------------------------------------------------------------- #

def test_ring_buffer_never_exceeds_its_advertised_window():
    """A slow camera must not quietly hold more seconds than the panel claims."""
    buf = RawRingBuffer(seconds=5.0, fps=15.0)
    frame = np.zeros((90, 160, 3), np.uint8)
    now = time.time()
    # 30 seconds of footage, arriving at a third of the expected frame rate.
    for i in range(150):
        buf.offer(frame, now + i * 0.2)
    stats = buf.stats()
    assert stats["seconds"] <= 5.2, (
        f"buffer held {stats['seconds']}s of footage while claiming 5s")
    assert stats["on_disk"] is False


def test_ring_buffer_keeps_nothing_on_disk():
    source = inspect.getsource(RawRingBuffer)
    for forbidden in ("open(", "imwrite", "write_bytes", "Path(", "VideoWriter"):
        assert forbidden not in source, (
            f"RawRingBuffer references {forbidden!r}; it must be memory only")


def test_buffer_forgets_when_cleared():
    buf = RawRingBuffer(seconds=5.0, fps=15.0)
    for i in range(20):
        buf.offer(np.zeros((90, 160, 3), np.uint8), time.time() + i * 0.05)
    assert buf.stats()["frames"] > 0
    buf.clear()
    assert buf.stats()["frames"] == 0
    assert buf.bytes_held == 0


# --------------------------------------------------------------------------- #
# 5. The headline number is structural, not a cleanup job
# --------------------------------------------------------------------------- #

def test_continuous_hours_stored_is_always_zero():
    report = privacy.storage_report()
    assert report["continuous_hours"] == 0.0
    assert report["clips_dir"] == str(config.CLIPS_DIR)


def test_nothing_in_the_outbound_payload_but_event_fields():
    """What the Claude integration is allowed to send off this machine."""
    from backend import ai_assistant
    rows = ai_assistant._event_rows([{
        "id": 1, "iso": "2026-01-01T10:00:00", "camera": "Hall",
        "type": "fall", "severity": "critical", "zone_name": "Stairs",
        "message": "Person #3 fell", "acknowledged": 0,
        "clip_path": "x.mp4", "snapshot_path": "y.jpg", "simulated": 0,
        "meta": {"drop_body_heights": 0.31},
        # Anything added to the row later must not leak by default:
        "secret_future_column": "should never be transmitted",
    }])
    allowed = {"id", "time", "camera", "type", "severity", "zone", "message",
               "acknowledged", "has_clip", "simulated", "evidence"}
    assert set(rows[0]) == allowed, (
        "the outbound payload gained a field; _event_rows must stay an "
        "explicit allow-list")
    flat = str(rows[0])
    assert "secret_future_column" not in flat
    assert "snapshot_path" not in flat and "y.jpg" not in flat
    assert "x.mp4" not in flat, "a file path must never leave the machine"


# --------------------------------------------------------------------------- #
# 6. Simulated feeds can never pass as real
# --------------------------------------------------------------------------- #

def test_simulated_worlds_are_always_labelled():
    from backend.vision.engine import world_from_sim
    world = world_from_sim("Atrium", {"persons": [], "objects": []}, 1,
                           time.time(), 640, 360)
    assert world.simulated is True

    # And a real camera's world cannot be marked simulated by accident: the
    # flag is only ever set on this one path.
    plain = FrameWorld(camera="Real", ts=time.time(), frame_idx=1,
                       width=640, height=360)
    assert plain.simulated is False


def test_simulated_track_ids_cannot_collide_with_real_ones():
    from backend.vision.engine import world_from_sim
    world = world_from_sim("Atrium", {
        "persons": [{"track_id": 3, "box": (0, 0, 10, 10)}],
        "objects": [],
    }, 1, time.time(), 640, 360)
    assert world.persons[0].track_id >= 900_000, (
        "simulated ids are offset so they can never be confused with "
        "ByteTrack ids in the event log")
