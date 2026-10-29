"""
Regression tests for simulated camera clip suppression and real-camera preservation.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import config
from backend import events_db
from backend.alerts import AlertManager
from backend.camera_manager import CameraFlags
from backend.detectors.base import Candidate
from backend.state import CameraContext, TrackStore
from backend.world import Detection, FrameWorld
from backend.zone_store import Zone, ZoneSet


@pytest.fixture(autouse=True)
def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_db = tmp_path / "test_arjun.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(events_db, "_CONN", None)
    events_db.init(test_db)
    yield
    events_db.close()
    monkeypatch.setattr(events_db, "_CONN", None)


def _make_context(camera: str, simulated: bool) -> tuple[CameraContext, Zone]:
    zone = Zone(id=1, camera=camera, name="Zone 1", ztype="restricted",
                points=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    world = FrameWorld(
        camera=camera, ts=time.time(), frame_idx=1,
        width=640, height=360,
        persons=[Detection(track_id=1, label="person", conf=0.9, box=(10, 10, 50, 50))],
    )
    ctx = CameraContext(
        camera=camera, world=world, tracks=TrackStore(), zones=ZoneSet([zone]),
        flags=CameraFlags(), now=world.ts, simulated=simulated,
    )
    return ctx, zone


def test_simulated_alert_skips_clip_recording_by_default(monkeypatch: pytest.MonkeyPatch):
    """Simulated camera alerts must not trigger incident clips when RECORD_SIMULATED_CLIPS is False."""
    monkeypatch.setattr(config, "RECORD_SIMULATED_CLIPS", False)
    manager = AlertManager(broadcast=lambda _msg: None)
    camera = "SimCamera"
    rec = manager.recorder_for(camera)
    ctx, zone = _make_context(camera, simulated=True)

    cand = Candidate(type="restricted", message="Restricted zone entry",
                     track_id=1, zone=zone)
    fired = manager.submit([cand], ctx)

    assert len(fired) == 1
    alert = fired[0]
    assert alert.simulated is True
    assert alert.clip_state == "none"

    row = events_db.get_event(alert.event_id)
    assert row is not None
    assert row["clip_state"] == "none"
    assert row["clip_path"] is None
    assert row["simulated"] == 1
    assert rec.pending_count() == 0


def test_real_camera_alert_preserves_clip_recording():
    """Real camera alerts must preserve normal incident clip recording."""
    manager = AlertManager(broadcast=lambda _msg: None)
    camera = "RealCamera"
    rec = manager.recorder_for(camera)
    ctx, zone = _make_context(camera, simulated=False)

    cand = Candidate(type="restricted", message="Restricted zone entry",
                     track_id=1, zone=zone)
    fired = manager.submit([cand], ctx)

    assert len(fired) == 1
    alert = fired[0]
    assert alert.simulated is False
    assert alert.clip_state == "pending"

    row = events_db.get_event(alert.event_id)
    assert row is not None
    assert row["clip_state"] == "pending"
    assert row["simulated"] == 0
    assert rec.pending_count() == 1


def test_simulated_alert_records_clip_when_explicitly_enabled(monkeypatch: pytest.MonkeyPatch):
    """When RECORD_SIMULATED_CLIPS is explicitly True, simulated feeds trigger clips."""
    monkeypatch.setattr(config, "RECORD_SIMULATED_CLIPS", True)
    manager = AlertManager(broadcast=lambda _msg: None)
    camera = "SimCameraEnabled"
    rec = manager.recorder_for(camera)
    ctx, zone = _make_context(camera, simulated=True)

    cand = Candidate(type="restricted", message="Restricted zone entry",
                     track_id=1, zone=zone)
    fired = manager.submit([cand], ctx)

    assert len(fired) == 1
    alert = fired[0]
    assert alert.simulated is True
    assert alert.clip_state == "pending"

    row = events_db.get_event(alert.event_id)
    assert row is not None
    assert row["clip_state"] == "pending"
    assert rec.pending_count() == 1


def test_spec_clip_false_alert_is_none_for_both():
    """Rules with clip=False remain clip_state='none' regardless of simulated flag."""
    manager = AlertManager(broadcast=lambda _msg: None)

    # Real camera with lights_on (clip=False in config.ALERT_SPECS)
    ctx_real, _ = _make_context("RealCamera", simulated=False)
    cand_real = Candidate(type="lights_on", message="Lights left on")
    fired_real = manager.submit([cand_real], ctx_real)
    assert fired_real[0].clip_state == "none"

    # Simulated camera with lights_on
    ctx_sim, _ = _make_context("SimCamera", simulated=True)
    cand_sim = Candidate(type="lights_on", message="Lights left on")
    fired_sim = manager.submit([cand_sim], ctx_sim)
    assert fired_sim[0].clip_state == "none"
