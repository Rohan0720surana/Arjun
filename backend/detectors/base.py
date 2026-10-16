"""
The shape every detection rule takes.

A detector is a small object with one method, :meth:`Detector.update`, which is
handed a :class:`~backend.state.CameraContext` and returns zero or more
:class:`Candidate` objects. It is *not* allowed to raise, send a notification,
touch the database or see a pixel. All it can do is say "I think this happened,
here is why".

Everything downstream of that — cooldowns, severity, persistence, clips,
Telegram, the dashboard — is handled once, in :mod:`backend.alerts`, instead of
eleven times in eleven rules. That separation is why a new rule is about forty
lines, and why every rule automatically gets an incident clip and an audit
trail without asking.

A detector raising an exception is contained by :meth:`Detector.safe_update`:
the rule is disabled for a few seconds, the error surfaces on the camera tile,
and the other ten rules carry on. A bug in loitering should never cost you fall
detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import config
from backend.state import CameraContext
from backend.zone_store import Zone


@dataclass
class Candidate:
    """One rule's claim that something happened."""

    type: str
    message: str
    track_id: int | None = None
    zone: Zone | None = None
    severity: str | None = None          # None -> the catalogue's default
    meta: dict[str, Any] = field(default_factory=dict)
    #: Extra granularity for the cooldown key, when one track can legitimately
    #: produce distinct incidents (a bag in two different zones, say).
    dedupe: str = ""
    #: How to draw the subject of this alert on the privacy view.
    highlight: str | None = None
    #: Seconds the subject stays highlighted after the alert fires.
    highlight_for: float = 6.0

    @property
    def spec(self):
        return config.alert_spec(self.type)

    def resolved_severity(self) -> str:
        return self.severity or self.spec.severity

    def cooldown_key(self) -> str:
        parts = [self.type, str(self.track_id if self.track_id is not None else "-")]
        if self.zone is not None:
            parts.append(f"z{self.zone.id}")
        if self.dedupe:
            parts.append(self.dedupe)
        return ":".join(parts)

    def payload(self) -> dict[str, Any]:
        d = dict(self.meta)
        if self.zone is not None:
            d.setdefault("zone_id", self.zone.id)
            d.setdefault("zone_type", self.zone.ztype)
        return d


class Detector:
    """Base class for every rule."""

    #: Stable key, used in the UI and in the per-camera enable map.
    name: str = "detector"
    #: Human-readable, shown in the detector panel on the dashboard.
    title: str = "Detector"
    #: Camera flag that must be on for this rule to run (``None`` = always).
    requires_flag: str | None = None
    #: Zone types this rule needs; if a camera has none of them, it is skipped.
    requires_zones: tuple[str, ...] = ()
    #: One line explaining the rule, shown in the UI and used in reports.
    explanation: str = ""

    def __init__(self) -> None:
        self.errors = 0
        self.last_error = ""
        self._muted_until = 0.0
        self.fired = 0
        self.cost_ms = 0.0

    # -- lifecycle ------------------------------------------------------------- #

    def reset(self) -> None:
        """Forget everything. Called when a camera reconnects."""

    def update(self, ctx: CameraContext) -> list[Candidate]:
        raise NotImplementedError

    # -- plumbing --------------------------------------------------------------- #

    def applicable(self, ctx: CameraContext) -> bool:
        if self._muted_until > ctx.now:
            return False
        if self.requires_flag and not getattr(ctx.flags, self.requires_flag, False):
            return False
        if self.requires_zones and not ctx.zones.of(*self.requires_zones):
            return False
        return True

    def safe_update(self, ctx: CameraContext) -> list[Candidate]:
        """Run the rule with a seatbelt on."""
        if not self.applicable(ctx):
            return []
        started = time.perf_counter()
        try:
            out = self.update(ctx) or []
        except Exception as exc:                            # noqa: BLE001
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            # Back off rather than spinning on a bug sixty times a second.
            self._muted_until = ctx.now + min(30.0, 2.0 * self.errors)
            return []
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.cost_ms = self.cost_ms * 0.9 + elapsed * 0.1
        self.fired += len(out)
        return out

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "explanation": self.explanation,
            "requires_flag": self.requires_flag,
            "requires_zones": list(self.requires_zones),
            "fired": self.fired,
            "errors": self.errors,
            "last_error": self.last_error,
            "cost_ms": round(self.cost_ms, 2),
            "muted": self._muted_until > time.time(),
        }


#: The rules, in the order they run, as (module, class) pairs. Fall and SOS go
#: first so that the most urgent incidents are raised before anything else has
#: a chance to consume the frame's time budget.
REGISTRY: tuple[tuple[str, str], ...] = (
    ("fall", "FallDetector"),
    ("sos", "SosGestureDetector"),
    ("zones", "RestrictedZoneDetector"),
    ("zones", "LoiterDetector"),
    ("zones", "CrowdDetector"),
    ("abandoned", "AbandonedObjectDetector"),
    ("exit_blocked", "ExitBlockedDetector"),
    ("running", "PanicMovementDetector"),
    ("line_counter", "LineCounter"),
    ("after_hours", "AfterHoursDetector"),
    ("exam", "ExamModeDetector"),
)

#: Rules that could not be imported, as ``{"module.Class": "reason"}``. Shown
#: on the dashboard so a broken rule is visible rather than silently absent.
UNAVAILABLE: dict[str, str] = {}


def build_all() -> list[Detector]:
    """Instantiate every rule that can be imported.

    Each module is imported on its own and failures are recorded rather than
    raised. A rule with a bug in it should cost you that rule — not the
    cameras, not the dashboard, and not the other ten rules.
    """
    import importlib

    UNAVAILABLE.clear()
    out: list[Detector] = []
    for module_name, class_name in REGISTRY:
        ref = f"{module_name}.{class_name}"
        try:
            module = importlib.import_module(f"backend.detectors.{module_name}")
            out.append(getattr(module, class_name)())
        except Exception as exc:                            # noqa: BLE001
            UNAVAILABLE[ref] = f"{type(exc).__name__}: {exc}"
    return out
