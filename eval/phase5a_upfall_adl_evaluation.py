"""
ARJUN Campus Guardian -- Phase 5A
Real UP-Fall Normal Activity Evaluation

Evaluates false fall alerts on two UP-Fall normal activity recordings:
  Subject1Activity9Trial1Camera1  : Activity 9  -- Picking up an object (ADL)
  Subject1Activity11Trial1Camera1 : Activity 11 -- Intentional lying down (ADL)

Tasks performed per recording:
  1. Verify ZIP integrity and actual frame counts.
  2. Parse companion CSV for timestamps and activity annotations.
  3. Safely extract RGB frames from the ZIP archive.
  4. Run YOLOv8-Pose + ByteTrack using CSV-derived timestamps.
  5. Save pose cache in eval/cache/up_fall/.
  6. Validate UPFallDatasetAdapter.load_sequence().
  7. Run the production rule-based FallDetector.
  8. Run RF and XGBoost models (no retraining).
  9. Apply 2-second event-confirmation FSM.
 10. Count confirmed false alerts per activity and pipeline.
 11. Report tracking ID switches, missed frames, suspicious classifications.
 12. Calculate total evaluated ADL observation time.

Processed sequentially to avoid OOM.
Usage:
    python eval/phase5a_upfall_adl_evaluation.py
"""

from __future__ import annotations

import sys
import textwrap
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from eval.dataset import Sequence, SequenceAnnotation
from eval.extract_poses import VideoPoseExtractor, load_pose_cache, save_pose_cache
from eval.urfd_batch_evaluation import evaluate_sequence_with_diagnostics
from ml.adapters import UPFallDatasetAdapter, parse_upfall_timestamps_csv
from ml.event_evaluation import FallAlertEvent, TemporalFallStateMachine
from ml.features import extract_features_for_sequence
from ml.train_xgboost import INT_TO_CLASS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORDINGS = [
    {
        "recording_id": "Subject1Activity9Trial1Camera1",
        "activity_num": 9,
        "activity_name": "Picking up an object",
        "csv_path": "data/up_fall/Subject1/Subject1Activity9Trial1.csv",
        "zip_path": "data/up_fall/Subject1/Subject1Activity9Trial1Camera1.zip",
        "is_fall": False,
    },
    {
        "recording_id": "Subject1Activity11Trial1Camera1",
        "activity_num": 11,
        "activity_name": "Intentional lying down",
        "csv_path": "data/up_fall/Subject1/Subject1Activity11Trial1.csv",
        "zip_path": "data/up_fall/Subject1/Subject1Activity11Trial1Camera1.zip",
        "is_fall": False,
    },
]

CACHE_DIR = Path("eval/cache/up_fall")
DATA_ROOT = Path("data/up_fall")
RF_MODEL_PATH = Path("ml/models/random_forest_baseline.joblib")
XGB_MODEL_PATH = Path("ml/models/xgboost_baseline.joblib")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RecordingReport:
    recording_id: str
    activity_name: str
    is_fall: bool
    csv_rows: int = 0
    csv_duration_s: float = 0.0
    csv_fps: float = 0.0
    csv_activity_tags: dict = field(default_factory=dict)
    zip_ok: bool = False
    zip_frame_count: int = 0
    zip_size_bytes: int = 0
    annotation_problems: list = field(default_factory=list)
    extracted_frames: int = 0
    extraction_dir: str = ""
    extraction_ok: bool = False
    cache_path: str = ""
    cache_saved: bool = False
    adapter_ok: bool = False
    adapter_frames: int = 0
    adapter_duration_s: float = 0.0
    tracks: dict = field(default_factory=dict)
    frames_with_0_dets: int = 0
    frames_with_1_det: int = 0
    frames_with_gt1_dets: int = 0
    track_id_switches: int = 0
    max_frame_gap: int = 0
    primary_track_id: int = -1
    primary_track_coverage_pct: float = 0.0
    rule_raw_drops: int = 0
    rule_confirmed_alerts: int = 0
    rule_alert_timestamps: list = field(default_factory=list)
    total_feature_rows: int = 0
    rf_pred_counts: dict = field(default_factory=dict)
    xgb_pred_counts: dict = field(default_factory=dict)
    rf_confirmed_alerts: int = 0
    rf_alert_timestamps: list = field(default_factory=list)
    xgb_confirmed_alerts: int = 0
    xgb_alert_timestamps: list = field(default_factory=list)
    suspicious_rf_fallen_frames: int = 0
    suspicious_xgb_fallen_frames: int = 0
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 1: ZIP integrity check
# ---------------------------------------------------------------------------

def verify_zip(zip_path: Path):
    """Test ZIP integrity and count image frames. Returns (ok, frame_count, problems)."""
    problems = []
    if not zip_path.is_file():
        return False, 0, [f"ZIP not found: {zip_path}"]
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            bad = zf.testzip()
            ok = bad is None
            if not ok:
                problems.append(f"Corrupt member: {bad}")
            image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
            frame_count = sum(
                1 for name in zf.namelist()
                if Path(name).suffix.lower() in image_exts
                and not Path(name).name.startswith(".")
                and "__MACOSX" not in name
            )
    except zipfile.BadZipFile as exc:
        return False, 0, [f"BadZipFile: {exc}"]
    except Exception as exc:
        return False, 0, [f"ZIP error: {exc}"]
    return ok, frame_count, problems


# ---------------------------------------------------------------------------
# Task 2: CSV annotation parsing & sanity check
# ---------------------------------------------------------------------------

def parse_and_validate_csv(csv_path: Path, expected_is_fall: bool, activity_num: int):
    """Parse UP-Fall CSV; flag annotation issues. Returns (timing_dict, problems)."""
    problems = []
    if not csv_path.is_file():
        return {}, [f"CSV not found: {csv_path}"]
    try:
        timing = parse_upfall_timestamps_csv(csv_path)
    except Exception as exc:
        return {}, [f"CSV parse error: {exc}"]

    tags = timing.get("tags", np.array([]))
    tag_counts = {}
    for t in tags:
        tag_counts[int(t)] = tag_counts.get(int(t), 0) + 1

    fall_codes = {1, 2, 3, 4, 5}
    found_fall = set(tag_counts.keys()) & fall_codes
    if found_fall:
        problems.append(
            f"Unexpected fall-transition codes {found_fall} in CSV "
            f"for Activity {activity_num} (expected ADL-only)"
        )

    if timing["duration_s"] <= 0:
        problems.append(f"Non-positive duration: {timing['duration_s']:.3f}s")
    if not (0 < timing["fps"] <= 120):
        problems.append(f"Implausible FPS: {timing['fps']:.2f}")

    # Activity 11 uses tag code 11 for the ADL itself (lying), not a post-fall state.
    # parse_upfall_timestamps_csv() will misidentify this as fallen_state_start_s.
    if activity_num == 11 and timing.get("fallen_state_start_s") is not None:
        problems.append(
            "ANNOTATION CAUTION (Activity 11 -- Intentional lying down): "
            "parse_upfall_timestamps_csv() reports "
            f"fallen_state_start_s={timing['fallen_state_start_s']:.3f}s because "
            "tag code 11 (lying) appears in this ADL CSV. "
            "Tag 11 here is the ADL activity code, NOT a post-fall ground state. "
            "This evaluation suppresses fall_transition_start_s and fallen_state_start_s "
            "for both ADL recordings to prevent false fall-state anchoring."
        )

    if activity_num == 9 and timing.get("fall_transition_start_s") is not None:
        problems.append(
            "ANNOTATION CAUTION (Activity 9 -- Picking up object): "
            f"fall_transition_start_s={timing['fall_transition_start_s']:.3f}s detected. "
            "Verify CSV tags manually."
        )

    timing["tag_counts"] = tag_counts
    return timing, problems


# ---------------------------------------------------------------------------
# Task 3: Safe ZIP extraction
# ---------------------------------------------------------------------------

def extract_frames_from_zip(zip_path: Path, extract_dir: Path):
    """Safely extract image frames from ZIP. Returns (ok, count_extracted, problems)."""
    problems = []
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.is_file():
        return False, 0, [f"ZIP not found: {zip_path}"]
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    count = 0
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for info in zf.infolist():
                name = info.filename
                if ".." in name or name.startswith("/"):
                    problems.append(f"Skipping suspicious path: {name}")
                    continue
                if name.endswith("/"):
                    continue
                fname = Path(name).name
                if fname.startswith(".") or "__MACOSX" in name:
                    continue
                if Path(name).suffix.lower() not in image_exts:
                    continue
                dest = extract_dir / fname
                if not dest.exists():
                    dest.write_bytes(zf.read(name))
                count += 1
    except zipfile.BadZipFile as exc:
        return False, 0, [f"BadZipFile: {exc}"]
    except Exception as exc:
        return False, 0, [f"Extraction error: {exc}"]
    return True, count, problems


# ---------------------------------------------------------------------------
# Tasks 4-5: Pose extraction + cache save
# ---------------------------------------------------------------------------

def extract_and_cache(recording_id: str, extract_dir: Path, timing: dict,
                      is_fall: bool, cache_dir: Path):
    """Run YOLOv8-Pose + ByteTrack and save NPZ cache. Returns (seq, npz_path, problems)."""
    problems = []
    # For ADL recordings: always suppress fall timing fields regardless of CSV tags.
    seq_ann = SequenceAnnotation(
        sequence_id=recording_id,
        name=f"UP-Fall {recording_id}",
        is_fall=is_fall,
        duration_s=timing["duration_s"],
        fall_transition_start_s=None,
        fallen_state_start_s=None,
    )
    print(f"    Running YOLOv8-Pose + ByteTrack on {extract_dir} ...")
    try:
        extractor = VideoPoseExtractor()
        seq = extractor.extract_from_image_dir(
            image_dir=extract_dir,
            fps=timing["fps"],
            sequence_id=recording_id,
            name=f"UP-Fall {recording_id}",
            annotation=seq_ann,
            timestamps=timing.get("timestamps"),
        )
    except Exception as exc:
        return None, None, [f"Pose extraction failed: {exc}"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{recording_id}.npz"
    try:
        save_pose_cache(seq, npz_path)
    except Exception as exc:
        problems.append(f"Cache save failed: {exc}")
        return seq, None, problems
    return seq, npz_path, problems


# ---------------------------------------------------------------------------
# Task 6: Adapter validation
# ---------------------------------------------------------------------------

def validate_adapter(recording_id: str, cache_dir: Path, data_root: Path):
    """Validate UPFallDatasetAdapter.load_sequence(). Returns (ok, seq, problems)."""
    try:
        adapter = UPFallDatasetAdapter(root_dir=str(data_root), cache_dir=str(cache_dir))
        seq = adapter.load_sequence(recording_id)
        return True, seq, []
    except Exception as exc:
        return False, None, [f"Adapter load_sequence failed: {exc}"]


# ---------------------------------------------------------------------------
# Task 7: Rule-based detector
# ---------------------------------------------------------------------------

def run_rule_based(seq: Sequence):
    """Run production rule-based FallDetector. Returns (raw_drops, confirmed, alert_ts, errors)."""
    try:
        diag = evaluate_sequence_with_diagnostics(seq)
        return len(diag.raw_drop_timestamps), len(diag.confirmed_alerts), list(diag.alert_timestamps), []
    except Exception as exc:
        return 0, 0, [], [f"Rule-based error: {exc}"]


# ---------------------------------------------------------------------------
# Tasks 8-9: ML pipelines + FSM
# ---------------------------------------------------------------------------

def run_ml_pipelines(seq: Sequence, rf, xgb_model, recording_id: str):
    """Run RF + XGBoost with 2-second FSM on primary track. Returns diagnostics dict."""
    result = {
        "total_feature_rows": 0,
        "rf_pred_counts": {},
        "xgb_pred_counts": {},
        "rf_confirmed_alerts": 0,
        "rf_alert_timestamps": [],
        "xgb_confirmed_alerts": 0,
        "xgb_alert_timestamps": [],
        "suspicious_rf_fallen_frames": 0,
        "suspicious_xgb_fallen_frames": 0,
        "primary_track_id": -1,
        "errors": [],
    }
    try:
        raw_rows = extract_features_for_sequence(seq)
        if not raw_rows:
            result["errors"].append("No feature rows extracted (no detections?)")
            return result
        df_feat = pd.DataFrame(raw_rows)
        result["total_feature_rows"] = len(df_feat)

        if rf is None or xgb_model is None:
            result["errors"].append("One or both ML models not loaded; skipping inference.")
            return result

        feature_cols = list(rf.feature_names_in_)
        missing_cols = [c for c in feature_cols if c not in df_feat.columns]
        if missing_cols:
            result["errors"].append(f"Feature columns missing from data: {missing_cols[:5]}")
            return result

        X = df_feat[feature_cols]

        # RF inference
        rf_preds = rf.predict(X)
        rf_probs = rf.predict_proba(X)
        df_feat["rf_pred"] = rf_preds
        fallen_idx_rf = list(rf.classes_).index("fallen")
        df_feat["rf_prob_fallen"] = rf_probs[:, fallen_idx_rf]

        # XGB inference
        xgb_pred_ints = xgb_model.predict(X)
        xgb_probs = xgb_model.predict_proba(X)
        df_feat["xgb_pred"] = [INT_TO_CLASS[p] for p in xgb_pred_ints]
        df_feat["xgb_prob_fallen"] = xgb_probs[:, 2]

        result["rf_pred_counts"] = dict(df_feat["rf_pred"].value_counts())
        result["xgb_pred_counts"] = dict(df_feat["xgb_pred"].value_counts())
        result["suspicious_rf_fallen_frames"] = int((df_feat["rf_pred"] == "fallen").sum())
        result["suspicious_xgb_fallen_frames"] = int((df_feat["xgb_pred"] == "fallen").sum())

        # Primary track: most frames, label-free selection
        track_counts = Counter(df_feat["track_id"].values)
        if not track_counts:
            result["errors"].append("No tracks in feature data.")
            return result
        primary_tid = track_counts.most_common(1)[0][0]
        result["primary_track_id"] = int(primary_tid)

        sub = df_feat[df_feat["track_id"] == primary_tid].sort_values("timestamp")
        if sub.empty:
            result["errors"].append(f"Primary track {primary_tid} has no feature rows.")
            return result

        # RF FSM (2-second confirmation)
        sm_rf = TemporalFallStateMachine(required_confirmation_s=2.0)
        rf_alerts = []
        for _, row in sub.iterrows():
            ev = sm_rf.update(row["timestamp"], row["rf_pred"],
                              track_id=int(primary_tid), prob_fallen=row["rf_prob_fallen"])
            if ev:
                rf_alerts.append(ev)

        # XGB FSM (2-second confirmation)
        sm_xgb = TemporalFallStateMachine(required_confirmation_s=2.0)
        xgb_alerts = []
        for _, row in sub.iterrows():
            ev = sm_xgb.update(row["timestamp"], row["xgb_pred"],
                                track_id=int(primary_tid), prob_fallen=row["xgb_prob_fallen"])
            if ev:
                xgb_alerts.append(ev)

        result["rf_confirmed_alerts"] = len(rf_alerts)
        result["rf_alert_timestamps"] = [e.alert_timestamp for e in rf_alerts]
        result["xgb_confirmed_alerts"] = len(xgb_alerts)
        result["xgb_alert_timestamps"] = [e.alert_timestamp for e in xgb_alerts]

    except Exception as exc:
        result["errors"].append(f"ML pipeline error: {exc}")
    return result


# ---------------------------------------------------------------------------
# Tracking diagnostics
# ---------------------------------------------------------------------------

def compute_tracking_diagnostics(seq: Sequence):
    """Compute tracking continuity metrics. Returns diagnostics dict."""
    det_counts = [len(f.detections) for f in seq.frames]
    all_dets = [(f.frame_idx, d.track_id) for f in seq.frames for d in f.detections]
    track_counts = Counter(d[1] for d in all_dets)
    primary_tid = track_counts.most_common(1)[0][0] if track_counts else -1
    pct = track_counts[primary_tid] / max(len(seq.frames), 1) * 100 if primary_tid >= 0 else 0.0
    sub_frames = sorted(d[0] for d in all_dets if d[1] == primary_tid)
    gaps = [sub_frames[i + 1] - sub_frames[i] for i in range(len(sub_frames) - 1)]
    max_gap = max(gaps) if gaps else 0
    track_id_switches = 0
    prev_ids = set()
    for f in seq.frames:
        cur_ids = {d.track_id for d in f.detections}
        if prev_ids and cur_ids and not (prev_ids & cur_ids):
            track_id_switches += 1
        prev_ids = cur_ids
    return {
        "tracks": dict(track_counts),
        "frames_with_0_dets": det_counts.count(0),
        "frames_with_1_det": det_counts.count(1),
        "frames_with_gt1_dets": sum(1 for c in det_counts if c > 1),
        "primary_track_id": primary_tid,
        "primary_track_coverage_pct": pct,
        "track_id_switches": track_id_switches,
        "max_frame_gap": max_gap,
    }


# ---------------------------------------------------------------------------
# Single recording pipeline
# ---------------------------------------------------------------------------

def process_recording(rec_spec: dict, rf, xgb_model) -> RecordingReport:
    recording_id = rec_spec["recording_id"]
    activity_name = rec_spec["activity_name"]
    activity_num = rec_spec["activity_num"]
    csv_path = ROOT / rec_spec["csv_path"]
    zip_path = ROOT / rec_spec["zip_path"]
    is_fall = rec_spec["is_fall"]

    report = RecordingReport(recording_id=recording_id, activity_name=activity_name, is_fall=is_fall)
    print(f"\n{'=' * 70}")
    print(f"Recording: {recording_id}")
    print(f"Activity:  {activity_num} -- {activity_name}")
    print(f"{'=' * 70}")

    # --- Task 1: ZIP integrity ---
    print("\n[Task 1] Verifying ZIP integrity...")
    report.zip_size_bytes = zip_path.stat().st_size if zip_path.is_file() else 0
    zip_ok, zip_frames, zip_problems = verify_zip(zip_path)
    report.zip_ok = zip_ok
    report.zip_frame_count = zip_frames
    if zip_problems:
        report.annotation_problems.extend(zip_problems)
        print(f"  [!] ZIP problems: {zip_problems}")
    else:
        print(f"  ZIP OK: {zip_frames} image frames, {report.zip_size_bytes / 1e6:.1f} MB")

    # --- Task 2: CSV parse ---
    print("\n[Task 2] Parsing CSV annotations...")
    timing, csv_problems = parse_and_validate_csv(csv_path, is_fall, activity_num)
    if csv_problems:
        for p in csv_problems:
            print(f"  [CAUTION] {p}")
        report.annotation_problems.extend(csv_problems)
    if not timing:
        report.errors.append("CSV parse returned empty; aborting this recording.")
        return report

    report.csv_rows = timing.get("total_frames", 0)
    report.csv_duration_s = timing.get("duration_s", 0.0)
    report.csv_fps = timing.get("fps", 0.0)
    report.csv_activity_tags = timing.get("tag_counts", {})
    print(f"  CSV rows: {report.csv_rows}, duration: {report.csv_duration_s:.3f}s, "
          f"empirical FPS: {report.csv_fps:.2f}")
    print(f"  Activity tag distribution: {dict(sorted(report.csv_activity_tags.items()))}")
    if timing.get("fall_transition_start_s") is not None:
        print(f"  fall_transition_start_s (CSV): {timing['fall_transition_start_s']:.3f}s")
    if timing.get("fallen_state_start_s") is not None:
        print(f"  fallen_state_start_s (CSV, suppressed for ADL): {timing['fallen_state_start_s']:.3f}s")

    # Frame/CSV consistency
    if zip_frames > 0 and report.csv_rows > 0:
        ratio = zip_frames / report.csv_rows
        if not (0.9 <= ratio <= 1.1):
            msg = (f"Frame count mismatch: ZIP={zip_frames}, CSV={report.csv_rows} "
                   f"(ratio {ratio:.2f})")
            print(f"  [CAUTION] {msg}")
            report.annotation_problems.append(msg)
        else:
            print(f"  Frame/CSV count consistent: ZIP={zip_frames}, CSV={report.csv_rows}")

    # --- Check existing cache ---
    npz_path = CACHE_DIR / f"{recording_id}.npz"
    seq = None

    if npz_path.is_file():
        print(f"\n[Tasks 4-5] Cache found: {npz_path} -- loading (skipping re-extraction).")
        try:
            seq_ann = SequenceAnnotation(
                sequence_id=recording_id,
                name=f"UP-Fall {recording_id}",
                is_fall=is_fall,
                duration_s=timing["duration_s"],
                fall_transition_start_s=None,
                fallen_state_start_s=None,
            )
            seq = load_pose_cache(npz_path, annotation_override=seq_ann)
            report.extracted_frames = len(seq.frames)
            report.cache_path = str(npz_path)
            report.cache_saved = True
            report.extraction_ok = True
            print(f"  Loaded {len(seq.frames)} frames from cache.")
        except Exception as exc:
            print(f"  [!] Cache load failed ({exc}); will re-extract.")
            seq = None

    if seq is None:
        # --- Task 3: Extract from ZIP ---
        extract_dir = DATA_ROOT / "Subject1" / f"{recording_id}_frames"
        print(f"\n[Task 3] Extracting frames from ZIP to {extract_dir} ...")
        ext_ok, ext_count, ext_problems = extract_frames_from_zip(zip_path, extract_dir)
        report.extraction_dir = str(extract_dir)
        report.extraction_ok = ext_ok
        report.extracted_frames = ext_count
        if ext_problems:
            report.errors.extend(ext_problems)
            print(f"  [!] Extraction problems: {ext_problems}")
        else:
            print(f"  Extracted {ext_count} frames.")
        if not ext_ok or ext_count == 0:
            report.errors.append("Frame extraction failed; aborting.")
            return report

        # --- Tasks 4-5: Pose extraction + cache ---
        print(f"\n[Tasks 4-5] Running YOLOv8-Pose + ByteTrack extraction...")
        t_start = time.time()
        seq, npz_saved, pose_problems = extract_and_cache(
            recording_id=recording_id,
            extract_dir=extract_dir,
            timing=timing,
            is_fall=is_fall,
            cache_dir=CACHE_DIR,
        )
        elapsed = time.time() - t_start
        if pose_problems:
            report.errors.extend(pose_problems)
            for p in pose_problems:
                print(f"  [!] {p}")
        if seq is not None:
            report.extracted_frames = len(seq.frames)
            print(f"  Done in {elapsed:.1f}s: {len(seq.frames)} frames")
        if npz_saved is not None:
            report.cache_path = str(npz_saved)
            report.cache_saved = True
            print(f"  Cache saved: {npz_saved}")
        else:
            report.errors.append("Pose cache NOT saved.")

    if seq is None:
        report.errors.append("No sequence available; aborting.")
        return report

    # --- Tracking diagnostics ---
    track_diag = compute_tracking_diagnostics(seq)
    report.tracks = track_diag["tracks"]
    report.frames_with_0_dets = track_diag["frames_with_0_dets"]
    report.frames_with_1_det = track_diag["frames_with_1_det"]
    report.frames_with_gt1_dets = track_diag["frames_with_gt1_dets"]
    report.track_id_switches = track_diag["track_id_switches"]
    report.max_frame_gap = track_diag["max_frame_gap"]
    report.primary_track_id = track_diag["primary_track_id"]
    report.primary_track_coverage_pct = track_diag["primary_track_coverage_pct"]
    print(f"\n[Tracking] 0-det={report.frames_with_0_dets}, "
          f"1-det={report.frames_with_1_det}, >1-det={report.frames_with_gt1_dets}")
    print(f"  Tracks: {dict(sorted(report.tracks.items()))}")
    print(f"  Primary: track {report.primary_track_id} "
          f"({report.primary_track_coverage_pct:.1f}%), max gap={report.max_frame_gap} frames")
    print(f"  Track ID switches: {report.track_id_switches}")

    # --- Task 6: Adapter validation ---
    print(f"\n[Task 6] Validating UPFallDatasetAdapter.load_sequence()...")
    adapter_ok, adapter_seq, adapter_problems = validate_adapter(recording_id, CACHE_DIR, DATA_ROOT)
    report.adapter_ok = adapter_ok
    if adapter_problems:
        report.errors.extend(adapter_problems)
        print(f"  [!] Adapter problems: {adapter_problems}")
    if adapter_seq is not None:
        report.adapter_frames = len(adapter_seq.frames)
        report.adapter_duration_s = adapter_seq.annotation.duration_s
        is_fall_rep = adapter_seq.annotation.is_fall
        print(f"  Adapter OK: {len(adapter_seq.frames)} frames, "
              f"duration={adapter_seq.annotation.duration_s:.3f}s, is_fall={is_fall_rep}")
        if is_fall_rep and not is_fall:
            report.errors.append(
                "ANNOTATION ERROR: Adapter returned is_fall=True for an ADL recording."
            )

    eval_seq = adapter_seq if adapter_seq is not None else seq

    # --- Task 7: Rule-based detector ---
    print(f"\n[Task 7] Running production rule-based FallDetector...")
    rule_drops, rule_alerts, rule_alert_ts, rule_errors = run_rule_based(eval_seq)
    report.rule_raw_drops = rule_drops
    report.rule_confirmed_alerts = rule_alerts
    report.rule_alert_timestamps = rule_alert_ts
    if rule_errors:
        report.errors.extend(rule_errors)
    print(f"  Raw drop triggers: {rule_drops}")
    print(f"  Confirmed alerts (2s FSM): {rule_alerts}")
    for i, ts in enumerate(rule_alert_ts):
        print(f"    [FALSE ALERT] Rule-based alert {i + 1} at t={ts:.3f}s")

    # --- Tasks 8-9: ML pipelines + FSM ---
    print(f"\n[Tasks 8-9] Running RF + XGBoost with 2-second FSM...")
    ml = run_ml_pipelines(eval_seq, rf, xgb_model, recording_id)
    if ml["errors"]:
        report.errors.extend(ml["errors"])
        for e in ml["errors"]:
            print(f"  [!] {e}")

    report.total_feature_rows = ml["total_feature_rows"]
    report.rf_pred_counts = ml["rf_pred_counts"]
    report.xgb_pred_counts = ml["xgb_pred_counts"]
    report.rf_confirmed_alerts = ml["rf_confirmed_alerts"]
    report.rf_alert_timestamps = ml["rf_alert_timestamps"]
    report.xgb_confirmed_alerts = ml["xgb_confirmed_alerts"]
    report.xgb_alert_timestamps = ml["xgb_alert_timestamps"]
    report.suspicious_rf_fallen_frames = ml["suspicious_rf_fallen_frames"]
    report.suspicious_xgb_fallen_frames = ml["suspicious_xgb_fallen_frames"]

    print(f"  Feature rows: {report.total_feature_rows}")
    print(f"  RF class breakdown:  {report.rf_pred_counts}")
    print(f"  XGB class breakdown: {report.xgb_pred_counts}")
    print(f"  RF  'fallen' frame preds (suspicious): {report.suspicious_rf_fallen_frames}")
    print(f"  XGB 'fallen' frame preds (suspicious): {report.suspicious_xgb_fallen_frames}")
    print(f"  RF  confirmed false alerts (2s FSM): {report.rf_confirmed_alerts}")
    for i, ts in enumerate(report.rf_alert_timestamps):
        print(f"    [FALSE ALERT] RF alert {i + 1} at t={ts:.3f}s")
    print(f"  XGB confirmed false alerts (2s FSM): {report.xgb_confirmed_alerts}")
    for i, ts in enumerate(report.xgb_alert_timestamps):
        print(f"    [FALSE ALERT] XGB alert {i + 1} at t={ts:.3f}s")

    return report


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(reports: list):
    total_duration = sum(r.csv_duration_s for r in reports)
    print("\n")
    print("=" * 90)
    print("ARJUN Campus Guardian -- Phase 5A Summary: UP-Fall Normal Activity Evaluation")
    print("=" * 90)
    hdr = (f"{'Activity':<36} {'Duration':>9} {'Rule Alerts':>12} "
           f"{'RF Alerts':>10} {'XGB Alerts':>11}")
    print(hdr)
    print("-" * 90)
    for r in reports:
        print(f"{r.activity_name:<36} {r.csv_duration_s:>8.1f}s "
              f"{r.rule_confirmed_alerts:>12} "
              f"{r.rf_confirmed_alerts:>10} "
              f"{r.xgb_confirmed_alerts:>11}")
    print("-" * 90)
    total_rule = sum(r.rule_confirmed_alerts for r in reports)
    total_rf = sum(r.rf_confirmed_alerts for r in reports)
    total_xgb = sum(r.xgb_confirmed_alerts for r in reports)
    print(f"{'TOTAL (2 ADL recordings)':<36} {total_duration:>8.1f}s "
          f"{total_rule:>12} {total_rf:>10} {total_xgb:>11}")
    print("=" * 90)

    for r in reports:
        print(f"\n--- {r.recording_id} ---")
        print(f"  Activity: {r.activity_name} | Ground truth: ADL (is_fall=False)")
        print(f"  CSV: {r.csv_rows} rows, {r.csv_duration_s:.3f}s, {r.csv_fps:.2f} FPS")
        print(f"  ZIP: {r.zip_frame_count} frames, "
              f"integrity={'OK' if r.zip_ok else 'FAIL'}, "
              f"{r.zip_size_bytes / 1e6:.1f} MB")
        print(f"  Extracted frames: {r.extracted_frames}")
        print(f"  Adapter: {'OK' if r.adapter_ok else 'FAIL'} "
              f"({r.adapter_frames} frames, {r.adapter_duration_s:.3f}s)")
        print(f"  Tracks: {dict(sorted(r.tracks.items()))}")
        print(f"  Primary track {r.primary_track_id}: "
              f"{r.primary_track_coverage_pct:.1f}% coverage, "
              f"max gap {r.max_frame_gap} frames")
        print(f"  Track ID switches: {r.track_id_switches}")
        print(f"  Det-count per frame -- 0:{r.frames_with_0_dets}  "
              f"1:{r.frames_with_1_det}  >1:{r.frames_with_gt1_dets}")
        print(f"  RF  'fallen' frame preds: {r.suspicious_rf_fallen_frames}/{r.total_feature_rows}")
        print(f"  XGB 'fallen' frame preds: {r.suspicious_xgb_fallen_frames}/{r.total_feature_rows}")
        if r.annotation_problems:
            print("  [ANNOTATION PROBLEMS]:")
            for p in r.annotation_problems:
                for line in textwrap.wrap(p, 78):
                    print(f"    {line}")
        if r.errors:
            print("  [ERRORS / WARNINGS]:")
            for e in r.errors:
                for line in textwrap.wrap(e, 78):
                    print(f"    {line}")

    print(f"\nTotal evaluated ADL observation time: {total_duration:.1f}s "
          f"({total_duration / 60:.2f} min) across {len(reports)} recordings.")

    print("\n--- Evaluation Limitations ---")
    print(textwrap.fill(
        "IMPORTANT: Only 2 ADL recordings from 1 subject were evaluated. "
        "No general false-positive rate claims should be drawn from this limited sample.",
        80))
    print(textwrap.fill(
        "Activity 11 (intentional lying down): the subject voluntarily lies "
        "horizontally for the full recording. Tag code 11 in the companion CSV "
        "is the ADL activity identifier, NOT a post-fall ground state. "
        "parse_upfall_timestamps_csv() and UPFallDatasetAdapter now correctly "
        "interpret code 11 using recording context, ensuring no false post-fall "
        "ground state is anchored for this ADL.", 80))
    print(textwrap.fill(
        "Activity 9 (picking up object): involves a brief stooped/bent posture. "
        "False alerts here indicate sensitivity to rapid bounding-box height drops.", 80))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ARJUN Campus Guardian -- Phase 5A")
    print("Real UP-Fall Normal Activity Evaluation")
    print("=" * 70)
    print(f"Cache directory: {CACHE_DIR.resolve()}")
    print(f"Data root:       {DATA_ROOT.resolve()}")

    print("\nLoading ML models...")
    rf = None
    xgb_model = None
    if RF_MODEL_PATH.is_file():
        rf = joblib.load(RF_MODEL_PATH)
        print(f"  Random Forest loaded ({RF_MODEL_PATH.stat().st_size} bytes)")
    else:
        print(f"  [!] Random Forest NOT found: {RF_MODEL_PATH}")
    if XGB_MODEL_PATH.is_file():
        xgb_model = joblib.load(XGB_MODEL_PATH)
        print(f"  XGBoost loaded ({XGB_MODEL_PATH.stat().st_size} bytes)")
    else:
        print(f"  [!] XGBoost NOT found: {XGB_MODEL_PATH}")

    reports = []
    for rec_spec in RECORDINGS:
        report = process_recording(rec_spec, rf, xgb_model)
        reports.append(report)

    print_summary(reports)


if __name__ == "__main__":
    main()
