"""
URFD ADL (Normal Activity) False-Alert Baseline Evaluator -- Phase 3A.

Evaluates ARJUN''s production FallDetector on three real URFD ADL sequences
(adl-01-cam0-rgb, adl-02-cam0-rgb, adl-03-cam0-rgb) to establish a baseline
false-positive rate on actual non-fall recordings.

Ground-Truth Provenance
-----------------------
The sequences are named ``adl-XX`` per the official UR Fall Detection Dataset
(URFD) naming convention, which designates them as Activities of Daily Living
(non-fall).  The companion ``urfall-cam0-adls.csv`` per-frame annotation file
is not present locally.  Accordingly, ``is_fall=False`` is recorded with this
provenance note rather than as a fully-verified official import.

Timestamp Assumptions
---------------------
Individual ``adl-XX-data.csv`` synchronization files are unavailable, so
timestamps are derived as ``t = frame_index / assumed_fps`` with
``assumed_fps=30.0``.  These are *not* official URFD timestamps; all timings
in this evaluation are approximate and labelled ``[assumed 30.0 FPS]``.

Cache Behaviour
---------------
If a pose cache NPZ already exists for a sequence the extraction step is
skipped, preserving deterministic reruns without redundant YOLO inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.dataset import FrameObservation, Sequence, SequenceAnnotation
from eval.extract_poses import VideoPoseExtractor, load_pose_cache, save_pose_cache
from eval.metrics import evaluate_results
from eval.runner import OfflineFallRunner, SequenceResult

# --------------------------------------------------------------------------- #
# ADL Sequence Inventory
# --------------------------------------------------------------------------- #

#: Canonical URFD ADL frame-directory names and their expected frame counts.
URFD_ADL_SEQUENCES: dict[str, dict] = {
    "adl-01-cam0-rgb": {
        "expected_frames": 150,
        "description": "URFD ADL-01 Camera-0 RGB -- normal activity, 150 frames",
    },
    "adl-02-cam0-rgb": {
        "expected_frames": 180,
        "description": "URFD ADL-02 Camera-0 RGB -- normal activity, 180 frames",
    },
    "adl-03-cam0-rgb": {
        "expected_frames": 180,
        "description": "URFD ADL-03 Camera-0 RGB -- normal activity, 180 frames",
    },
    "adl-04-cam0-rgb": {
        "expected_frames": 150,
        "description": "URFD ADL-04 Camera-0 RGB -- normal activity, 150 frames",
    },
    "adl-05-cam0-rgb": {
        "expected_frames": 180,
        "description": "URFD ADL-05 Camera-0 RGB -- normal activity, 180 frames",
    },
}

TIMESTAMP_DISCLAIMER = (
    "Timestamps computed as frame_index / 30.0 (assumed FPS). "
    "Official per-sequence synchronization CSV (adl-XX-data.csv) is unavailable."
)

PROVENANCE_NOTE = (
    "Non-fall status established by URFD dataset naming convention (adl-XX). "
    "Official urfall-cam0-adls.csv per-frame annotation CSV is not present locally."
)

EXPLORATORY_SAMPLE_WARNING = (
    "WARNING: ADL true negatives represent a small exploratory sample (short clips, ~27s total), "
    "not general real-world specificity. Real-world deployment involves diverse settings, "
    "lighting variations, furniture occlusions, and long durations that may produce higher false-alert rates."
)


# --------------------------------------------------------------------------- #
# Annotation Builder
# --------------------------------------------------------------------------- #

def make_adl_annotation(
    seq_id: str,
    num_frames: int,
    assumed_fps: float = 30.0,
    extra_note: str = "",
) -> SequenceAnnotation:
    """
    Build a SequenceAnnotation for a URFD ADL sequence.

    ``is_fall=False`` is set based on URFD naming convention (adl-XX).
    The description embeds the provenance and timestamp assumption notes.
    """
    duration_s = round((num_frames - 1) / assumed_fps, 4) if num_frames > 1 else 0.0
    info = URFD_ADL_SEQUENCES.get(seq_id, {})
    description = info.get("description", f"URFD {seq_id}")
    full_desc = f"{description}. {PROVENANCE_NOTE} {TIMESTAMP_DISCLAIMER}"
    if extra_note:
        full_desc += f" {extra_note}"
    return SequenceAnnotation(
        sequence_id=seq_id,
        name=f"URFD {seq_id}",
        is_fall=False,
        duration_s=duration_s,
        description=full_desc,
    )


# --------------------------------------------------------------------------- #
# Directory Resolution
# --------------------------------------------------------------------------- #

def resolve_frames_dir(urfd_dir: Path | str, seq_id: str) -> Path:
    """
    Resolve the actual image-frame directory for a URFD ADL sequence.

    URFD ADL zips unpack into a nested layout:
      ``<urfd_dir>/<seq_id>/<seq_id>/``

    Falls back to ``<urfd_dir>/<seq_id>/`` if the nested path is absent.
    """
    urfd_path = Path(urfd_dir)
    nested = urfd_path / seq_id / seq_id
    if nested.is_dir():
        return nested
    direct = urfd_path / seq_id
    if direct.is_dir():
        return direct
    raise NotADirectoryError(
        f"Cannot find frame directory for {seq_id!r} under {urfd_path}. "
        f"Tried: {nested}  and  {direct}"
    )


# --------------------------------------------------------------------------- #
# Tracking Continuity Computation
# --------------------------------------------------------------------------- #

def compute_tracking_continuity(
    frames: list[FrameObservation],
    fps: float = 30.0,
) -> dict:
    """
    Compute rigorous tracking continuity metrics across frame observations.

    Parameters:
        frames: List of FrameObservation instances.
        fps: Frame rate for computing duration in seconds.

    Returns:
        dict containing:
        - frames_with_person: Number of frames containing at least one detection.
        - detection_coverage_pct: Percentage of total frames with detections.
        - longest_consecutive_detection_gap: Maximum consecutive frames with zero detections.
        - longest_detection_gap_s: Longest gap in seconds.
        - longest_continuous_track_segment: Maximum consecutive frames where any single track ID
          was continuously detected without interruption.
        - longest_track_segment_s: Longest continuous track segment in seconds.
        - track_id_switches: Number of track ID switches between consecutive detection events.
        - unique_track_ids: Sorted list of all unique track IDs.
        - num_tracks: Number of unique track IDs.
    """
    total_frames = len(frames)
    if total_frames == 0:
        return {
            "frames_with_person": 0,
            "detection_coverage_pct": 0.0,
            "longest_consecutive_detection_gap": 0,
            "longest_detection_gap_s": 0.0,
            "longest_continuous_track_segment": 0,
            "longest_track_segment_s": 0.0,
            "track_id_switches": 0,
            "unique_track_ids": [],
            "num_tracks": 0,
        }

    frames_with_person = sum(1 for f in frames if f.detections)
    coverage_pct = round(100.0 * frames_with_person / total_frames, 1)

    # 1. Longest consecutive gap without detections
    longest_gap = 0
    current_gap = 0
    for f in frames:
        if not f.detections:
            current_gap += 1
            if current_gap > longest_gap:
                longest_gap = current_gap
        else:
            current_gap = 0

    gap_s = round(longest_gap / fps, 2) if fps > 0 else 0.0

    # 2. Longest continuous track segment (uninterrupted per-track consecutive frames)
    track_runs: dict[int, int] = {}
    longest_segment_per_track: dict[int, int] = {}
    all_track_ids: set[int] = set()

    # 3. Track ID switches across consecutive detection events
    prev_primary_track: int | None = None
    id_switches = 0

    for f in frames:
        present_tracks = {d.track_id for d in f.detections}
        all_track_ids.update(present_tracks)

        # Update consecutive run lengths for each track
        for tid in present_tracks:
            track_runs[tid] = track_runs.get(tid, 0) + 1
            if track_runs[tid] > longest_segment_per_track.get(tid, 0):
                longest_segment_per_track[tid] = track_runs[tid]

        # Reset tracks that were not detected in this frame
        for tid in list(track_runs.keys()):
            if tid not in present_tracks:
                track_runs[tid] = 0

        # Check for track ID switches across detection events
        if f.detections:
            primary_det = max(f.detections, key=lambda d: d.conf)
            if prev_primary_track is None:
                prev_primary_track = primary_det.track_id
            else:
                if prev_primary_track not in present_tracks:
                    id_switches += 1
                    prev_primary_track = primary_det.track_id

    max_continuous_segment = (
        max(longest_segment_per_track.values()) if longest_segment_per_track else 0
    )
    seg_s = round(max_continuous_segment / fps, 2) if fps > 0 else 0.0

    return {
        "frames_with_person": frames_with_person,
        "detection_coverage_pct": coverage_pct,
        "longest_consecutive_detection_gap": longest_gap,
        "longest_detection_gap_s": gap_s,
        "longest_continuous_track_segment": max_continuous_segment,
        "longest_track_segment_s": seg_s,
        "track_id_switches": id_switches,
        "unique_track_ids": sorted(all_track_ids),
        "num_tracks": len(all_track_ids),
    }


# --------------------------------------------------------------------------- #
# Cache + Extraction
# --------------------------------------------------------------------------- #

def extract_or_load_adl_sequence(
    seq_id: str,
    urfd_dir: Path | str,
    cache_dir: Path | str,
    assumed_fps: float = 30.0,
    extractor: VideoPoseExtractor | None = None,
    allow_retime: bool = False,
) -> Sequence:
    """
    Load a URFD ADL pose cache if present, otherwise extract and save it.

    Validates cache FPS against the requested assumed_fps:
    - If cache FPS matches, timestamp provenance is checked (monotonicity).
    - If cache FPS differs:
      * If allow_retime is True: frames are safely retimed to assumed_fps with
        explicit provenance documentation.
      * Otherwise, if raw frames are available under urfd_dir, the sequence is
        regenerated at assumed_fps.
      * If raw frames are unavailable and allow_retime is False, ValueError is raised.

    The annotation is always rebuilt from ``make_adl_annotation`` so that
    ``is_fall=False`` and the provenance note are current regardless of
    when the cache was originally written.
    """
    urfd_path = Path(urfd_dir)
    cache_path = Path(cache_dir)
    npz_path = cache_path / f"{seq_id}.npz"

    if npz_path.is_file():
        cached_seq = load_pose_cache(npz_path)
        cached_fps = float(cached_seq.fps)
        num_frames = len(cached_seq.frames)

        # Validate cache FPS
        if abs(cached_fps - assumed_fps) > 1e-3:
            if allow_retime:
                # Safely retime derived timestamps when explicitly allowed and documented
                retimed_frames = [
                    FrameObservation(
                        frame_idx=f.frame_idx,
                        timestamp=round(f.frame_idx / assumed_fps, 4),
                        detections=f.detections,
                    )
                    for f in cached_seq.frames
                ]
                annotation = make_adl_annotation(
                    seq_id,
                    num_frames,
                    assumed_fps,
                    extra_note=(
                        f"[Retimed from cached {cached_fps:.1f} FPS to requested {assumed_fps:.1f} FPS "
                        f"via allow_retime=True; derived timestamps]"
                    ),
                )
                print(
                    f"[cache]   {seq_id}: FPS mismatch (cached={cached_fps:.1f}, requested={assumed_fps:.1f}); "
                    f"safely retimed {num_frames} frames under allow_retime=True"
                )
                return Sequence(annotation=annotation, fps=assumed_fps, frames=retimed_frames)

            # Check if source frame directory is available for regeneration
            try:
                frames_dir = resolve_frames_dir(urfd_path, seq_id)
            except NotADirectoryError:
                frames_dir = None

            if frames_dir is not None and frames_dir.is_dir():
                print(
                    f"[cache]   {seq_id}: FPS mismatch (cache={cached_fps:.1f}, requested={assumed_fps:.1f}). "
                    f"Regenerating from source frames at {frames_dir}..."
                )
                # Fall through to extraction below
            else:
                raise ValueError(
                    f"Cache FPS ({cached_fps:.1f}) for '{seq_id}' does not match requested FPS ({assumed_fps:.1f}), "
                    f"and raw frames directory was not found in '{urfd_path}' to regenerate. "
                    f"Pass allow_retime=True if you explicitly intend to retime derived timestamps."
                )
        else:
            # FPS matches. Validate timestamp monotonicity provenance
            if num_frames > 1:
                is_monotonic = all(
                    cached_seq.frames[i].timestamp <= cached_seq.frames[i + 1].timestamp
                    for i in range(num_frames - 1)
                )
                if not is_monotonic:
                    raise ValueError(
                        f"Cache timestamp provenance validation failed for '{seq_id}': "
                        f"timestamps in {npz_path.name} are not monotonically increasing."
                    )

            annotation = make_adl_annotation(seq_id, num_frames, assumed_fps)
            seq = Sequence(annotation=annotation, fps=cached_fps, frames=cached_seq.frames)
            print(f"[cache]   {seq_id}: loaded {num_frames} frames from {npz_path.name} (FPS={cached_fps:.1f})")
            return seq

    frames_dir = resolve_frames_dir(urfd_path, seq_id)
    if extractor is None:
        extractor = VideoPoseExtractor()

    print(f"[extract] {seq_id}: extracting from {frames_dir} (assumed FPS={assumed_fps}) ...")
    seq = extractor.extract_from_image_dir(
        frames_dir,
        fps=assumed_fps,
        sequence_id=seq_id,
    )

    num_frames = len(seq.frames)
    annotation = make_adl_annotation(seq_id, num_frames, assumed_fps)
    seq = Sequence(annotation=annotation, fps=assumed_fps, frames=seq.frames)

    cache_path.mkdir(parents=True, exist_ok=True)
    save_pose_cache(seq, npz_path)
    print(f"[extract] {seq_id}: saved {num_frames} frames -> {npz_path.name}")
    return seq


# --------------------------------------------------------------------------- #
# Per-Sequence Summary
# --------------------------------------------------------------------------- #

def sequence_summary(seq: Sequence, result: SequenceResult) -> dict:
    """Return a rich dict summarising one sequence evaluation including tracking continuity."""
    continuity = compute_tracking_continuity(seq.frames, fps=seq.fps)

    return {
        "seq_id": seq.annotation.sequence_id,
        "frames": len(seq.frames),
        "duration_s": seq.annotation.duration_s,
        "fps_assumed": seq.fps,
        "frames_with_person": continuity["frames_with_person"],
        "detection_coverage_pct": continuity["detection_coverage_pct"],
        "longest_consecutive_detection_gap": continuity["longest_consecutive_detection_gap"],
        "longest_detection_gap_s": continuity["longest_detection_gap_s"],
        "longest_continuous_track_segment": continuity["longest_continuous_track_segment"],
        "longest_track_segment_s": continuity["longest_track_segment_s"],
        "track_id_switches": continuity["track_id_switches"],
        "unique_track_ids": continuity["unique_track_ids"],
        "num_tracks": continuity["num_tracks"],
        "confirmed_alerts": len(result.alerts),
        "alert_timestamps": result.alert_timestamps,
        "is_fall_gt": seq.annotation.is_fall,
    }


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point
# --------------------------------------------------------------------------- #

def evaluate_urfd_adl_baseline(
    urfd_dir: str | Path = "data/urfd",
    cache_dir: str | Path = "eval/cache/urfd",
    assumed_fps: float = 30.0,
    seq_ids: list[str] | None = None,
) -> int:
    """
    Phase 3A: Real-video ADL false-alert baseline evaluation.

    Extracts (or reuses) pose caches, runs the production FallDetector on each
    ADL sequence, then prints per-sequence details and a comparison table.

    Returns 0 on success, 1 on error.
    """
    urfd_path = Path(urfd_dir)
    cache_path = Path(cache_dir)

    if seq_ids is None:
        seq_ids = list(URFD_ADL_SEQUENCES.keys())

    print("=" * 86)
    print(" URFD ADL FALSE-ALERT BASELINE EVALUATION -- Phase 3A")
    print(" [REAL VIDEO: URFD ADL sequences, non-fall]")
    print("=" * 86)
    print(f" FPS assumption : {assumed_fps:.1f}  (assumed; official sync CSVs unavailable)")
    print(f" Provenance     : URFD naming convention (adl-XX); no per-frame CSV")
    print(f" Sequences      : {', '.join(seq_ids)}")
    print()

    # Step 1: Extraction / cache reuse
    print("-" * 86)
    print(" Step 1: Pose Cache Extraction (or reuse)")
    print("-" * 86)

    extractor = VideoPoseExtractor()
    sequences: list[Sequence] = []

    for seq_id in seq_ids:
        try:
            seq = extract_or_load_adl_sequence(
                seq_id, urfd_path, cache_path, assumed_fps, extractor
            )
            sequences.append(seq)
        except Exception as exc:
            print(f"[ERROR] {seq_id}: {exc}", file=sys.stderr)
            return 1

    if not sequences:
        print("[ERROR] No sequences loaded.", file=sys.stderr)
        return 1

    # Step 2: FallDetector evaluation
    print()
    print("-" * 86)
    print(" Step 2: Production FallDetector Evaluation")
    print("-" * 86)

    runner = OfflineFallRunner()
    results = runner.evaluate_benchmark(sequences)

    summaries = [sequence_summary(seq, res) for seq, res in zip(sequences, results)]

    # Step 3: Per-sequence report
    print()
    print("-" * 86)
    print(" Step 3: Per-Sequence Report (including Tracking Continuity)")
    print("-" * 86)

    for s in summaries:
        print(f"\n Sequence : {s['seq_id']}")
        print(
            f"   Frames                : {s['frames']}  "
            f"(duration ~{s['duration_s']:.2f}s @ {s['fps_assumed']:.0f} FPS assumed)"
        )
        print(
            f"   Person coverage       : {s['frames_with_person']}/{s['frames']} frames "
            f"({s['detection_coverage_pct']:.1f}%)"
        )
        print(
            f"   Tracking continuity   : "
            f"max gap={s['longest_consecutive_detection_gap']}f ({s['longest_detection_gap_s']:.2f}s), "
            f"max continuous track={s['longest_continuous_track_segment']}f ({s['longest_track_segment_s']:.2f}s), "
            f"ID switches={s['track_id_switches']}"
        )
        print(f"   Track IDs             : {s['unique_track_ids']}  ({s['num_tracks']} unique track(s))")
        alert_str = f"{s['confirmed_alerts']}"
        if s["alert_timestamps"]:
            ts_str = ", ".join(f"{t:.2f}s" for t in s["alert_timestamps"])
            alert_str += f"  at [{ts_str}]"
        print(f"   Confirmed alerts      : {alert_str}")
        gt_label = (
            "non-fall (ADL)" if s["is_fall_gt"] is False
            else ("UNVERIFIED" if s["is_fall_gt"] is None else "FALL")
        )
        print(f"   Ground truth          : {gt_label}")

    # Step 4: Aggregate metrics (only verified sequences included in denominators)
    print()
    print("-" * 86)
    print(" Step 4: Aggregate Evaluation Metrics")
    print("-" * 86)

    verified_sequences = [s for s in sequences if s.annotation.is_fall is not None]
    verified_results = [r for s, r in zip(sequences, results) if s.annotation.is_fall is not None]

    if verified_sequences:
        report = evaluate_results(
            verified_sequences,
            verified_results,
            dataset_type="URFD_ADL_PILOT",
            report_title="ARJUN FALL DETECTION -- URFD ADL PILOT EVALUATION REPORT",
        )
        print()
        print(report.to_text())
    else:
        print(" [!] All sequences UNVERIFIED. Aggregate metrics skipped.")

    # Comparison table
    print()
    print("=" * 86)
    print(f" Comparison Table -- URFD ADL False-Alert Baseline (assumed {assumed_fps:.0f} FPS)")
    print("=" * 86)
    print(
        f"{'Sequence':<20} {'Frm':>4} {'Dur':>6} {'Cover':>6} "
        f"{'MaxGap':>7} {'MaxSeg':>7} {'Sw':>3} {'Alts':>5} {'Verdict':>8}"
    )
    print("-" * 86)
    for s in summaries:
        verdict = (
            "FP" if (s["is_fall_gt"] is False and s["confirmed_alerts"] > 0)
            else ("TN" if (s["is_fall_gt"] is False and s["confirmed_alerts"] == 0)
                  else ("UNVERIFIED" if s["is_fall_gt"] is None else "TP/FN"))
        )
        print(
            f"{s['seq_id']:<20} {s['frames']:>4} {s['duration_s']:>5.2f}s "
            f"{s['detection_coverage_pct']:>5.1f}% "
            f"{s['longest_consecutive_detection_gap']:>6}f "
            f"{s['longest_continuous_track_segment']:>6}f "
            f"{s['track_id_switches']:>3} "
            f"{s['confirmed_alerts']:>5} {verdict:>8}"
        )
    print("-" * 86)

    # Strictly filter verified ADL sequences to protect denominators
    verified_adls = [s for s in summaries if s["is_fall_gt"] is False]
    unverified_seqs = [s for s in summaries if s["is_fall_gt"] is None]

    tn_count = sum(1 for s in verified_adls if s["confirmed_alerts"] == 0)
    fp_count = sum(1 for s in verified_adls if s["confirmed_alerts"] > 0)
    total_verified_adls = len(verified_adls)

    total_false_alerts = sum(s["confirmed_alerts"] for s in verified_adls)
    total_verified_duration_s = sum(s["duration_s"] for s in verified_adls)

    seq_specificity_str = (
        f"{tn_count / total_verified_adls:.1%}" if total_verified_adls > 0 else "N/A"
    )
    seq_fpr_str = (
        f"{fp_count / total_verified_adls:.1%}" if total_verified_adls > 0 else "N/A"
    )

    if total_verified_duration_s > 0 and total_verified_adls > 0:
        duration_minutes = total_verified_duration_s / 60.0
        far_per_min = total_false_alerts / duration_minutes
        far_str = (
            f"{far_per_min:.2f} alerts/min "
            f"({total_false_alerts} alert(s) over {total_verified_duration_s:.1f}s / {duration_minutes:.2f} min)"
        )
    else:
        far_str = "N/A (insufficient verified duration)"

    print(f"\n Total sequences evaluated       : {len(summaries)}")
    print(f" Verified ADL sequences          : {total_verified_adls}")
    if unverified_seqs:
        print(f" Unverified sequences (excluded) : {len(unverified_seqs)}")
    print(f" True Negatives (no alert)       : {tn_count}")
    print(f" False Positives (alerted)       : {fp_count}")
    print(f" Sequence-level Specificity (TNR): {seq_specificity_str} ({tn_count}/{total_verified_adls})")
    print(f" Sequence-level False-Pos Rate   : {seq_fpr_str} ({fp_count}/{total_verified_adls})")
    print(f" Total confirmed false alerts    : {total_false_alerts}")
    print(f" False Alerts per Minute         : {far_str}")

    print()
    print("=" * 86)
    print(" EXPLORATORY SAMPLE SIZE WARNING:")
    print(" 3/3 ADL true negatives represents a SMALL EXPLORATORY SAMPLE (3 short clips, ~17s total),")
    print(" NOT general real-world specificity. Real-world deployment involves complex camera angles,")
    print(" varied lighting, furniture occlusions, and multi-hour monitoring where false-positive rates")
    print(" may differ significantly.")
    print("=" * 86)

    print()
    print(" LIMITATIONS & PROVENANCE:")
    print(f" 1. Timestamps assumed at {assumed_fps:.0f} FPS; official sync CSVs unavailable.")
    print(" 2. Non-fall label from URFD naming convention; no per-frame annotation CSV.")
    print(" 3. Only 3 of 40 URFD ADL sequences evaluated; small exploratory pilot.")
    print(" 4. Unverified recordings are strictly excluded from confusion matrix denominators.")
    print(" 5. Tracking gaps reflect pose detector dropouts; ByteTrack ID persistence does not")
    print("    guarantee continuous detection across all frames.")
    print(" 6. Results reflect production thresholds; no ML improvements applied.")
    print("=" * 86)

    return 0


if __name__ == "__main__":
    sys.exit(evaluate_urfd_adl_baseline())

