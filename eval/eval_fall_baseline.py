#!/usr/bin/env python3
"""
ARJUN Offline Fall Detection Baseline Evaluator.

Executes the synthetic baseline benchmark through the production FallDetector
without requiring video capture, dashboard servers, or ML dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.dataset import get_synthetic_benchmark
from eval.metrics import evaluate_results
from eval.runner import OfflineFallRunner


def run_baseline_evaluation(fps: float = 15.0) -> int:
    """Run the offline benchmark and display the report."""
    print("Loading synthetic fall & ADL benchmark sequences...")
    benchmark = get_synthetic_benchmark(fps=fps)
    print(f"Loaded {len(benchmark)} sequences.")

    print("Running sequences through FallDetector (isolated virtual clock)...")
    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(benchmark)

    print("Computing event-level metrics...")
    report = evaluate_results(benchmark, results)

    print("\n" + report.to_text() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_baseline_evaluation())
