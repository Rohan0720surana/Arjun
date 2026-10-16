"""Detection rules. Each one is a small object; see ``base.py`` for the shape."""

from backend.detectors.base import Candidate, Detector, build_all

__all__ = ["Candidate", "Detector", "build_all"]
