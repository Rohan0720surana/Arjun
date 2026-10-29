# ARJUN Fall Detection Evaluation Pipeline

ARJUN's fall-detection evaluation suite provides reproducible, offline benchmarking of the production rule-based `FallDetector` across synthetic edge cases and real-world video datasets (specifically the **UR Fall Detection Dataset**, URFD).

The evaluation pipeline is designed to be **isolated, deterministic, and algorithm-neutral**, allowing rule-based heuristics and future machine-learning classifiers to be evaluated under identical protocols.

---

## Architecture Overview

```
Video / Frames Directory (data/urfd/)
     │
     ▼
[extract_poses.py] ──> YOLOv8-Pose + ByteTrack (Tracker Reset per video)
     │
     ▼
Pose Cache NPZ (eval/cache/urfd/)  <── [urfd_annotations.py] Official CSV Annotations & Timestamps
     │
     ▼
[runner.py] OfflineFallRunner (Isolated Virtual Clock & Fresh TrackStore)
     │
     ▼
[metrics.py] Event-Level Evaluator (Algorithm-Neutral Matching Window)
     │
     ├──> eval/urfd_batch_evaluation.py (10-recording URFD benchmark)
     ├──> eval/urfd_adl_evaluation.py   (5 ADL baseline false-alarm benchmark)
     └──> eval/eval_fall_baseline.py    (Synthetic trajectory benchmark)
```

### Core Design Principles
1. **Zero Production Mutation**: Evaluation tools never modify `backend/detectors/fall.py` or production thresholds in `config.py`.
2. **Deterministic Virtual Clock**: Sequences are evaluated frame-by-frame with exact timestamps, eliminating system load jitter and wall-clock dependencies.
3. **Strict State Isolation**: Between sequences, the detector's internal scratchpad, tracker histories, and zone timers are completely re-instantiated.
4. **Algorithm-Neutral Matching**: Falls are evaluated as discrete temporal events within an event window $[t_{\text{impact}}, t_{\text{impact}} + 5.0\text{s}]$ rather than independent per-frame classifications.
5. **Observation Sufficiency Invariant**: Real fall recordings where the clip terminates before the detector's required floor-confirmation period ($2.0\text{s}$) are classified as `INSUFFICIENT_OBSERVATION` and excluded from confusion-matrix denominators, rather than penalizing the detector as false negatives.

---

## Dataset & Annotation Provenance

The evaluation currently benchmarks 10 real URFD Camera 0 (RGB) recordings:

### 1. Fall Recordings (`fall-01` through `fall-05`)
- **Visual Posture Labels**: Imported from official `urfall-cam0-falls.csv`.
  - State `-1`: Walking / upright posture.
  - State `0`: Dynamic falling transition (person collapsing).
  - State `1`: Fallen state (person resting on the floor).
- **Exact Timestamps**: Extracted from companion accelerometer synchronization files (`fall-XX-data.csv`) providing millisecond-accurate timestamps.
- **Timing Provenance**: Official hardware synchronization. Fall start is anchored to State 0 (`fall_transition_start_s`); impact/confirmation anchor is anchored to State 1 (`fallen_state_start_s`).

### 2. Normal Activity (ADL) Recordings (`adl-01` through `adl-05`)
- **Non-Fall Designation**: Established by canonical URFD naming convention (`adl-XX`).
- **Timing Provenance**: Derived as $t = \text{frame\_index} / 30.0$ (`assumed_fps = 30.0`), as official per-sequence `adl-XX-data.csv` synchronization files are unavailable locally.
- **Cache Validation**: Cache loader enforces that cached sequences are extracted at the requested FPS. If an FPS mismatch occurs, silent reuse is refused.

---

## Pose Cache Schema (v1.0)

Extracted pose detections are stored in compressed NumPy archives (`.npz`) under `eval/cache/urfd/`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema_version` | ASCII String | Format version, currently `"1.0"` |
| `sequence_id` | String | Sequence identifier (e.g. `"fall-01-cam0-rgb"`) |
| `name` | String | Human-readable sequence description |
| `fps` | Float32 | Nominal extraction frame rate |
| `duration_s` | Float32 | Total duration in seconds |
| `num_frames` | Int32 | Total frame count |
| `is_fall` | Int32 | Ground truth flag (`1`=Fall, `0`=ADL/non-fall, `-1`=Unverified) |
| `fall_start_s` | Float32 | Ground truth fall transition start (NaN if none) |
| `fall_impact_s` | Float32 | Ground truth impact timestamp (NaN if none) |
| `fall_transition_start_s` | Float32 | Official State 0 onset timestamp (NaN if none) |
| `fallen_state_start_s` | Float32 | Official State 1 floor-rest onset timestamp (NaN if none) |
| `frame_indices` | Int32 (N) | Sequential frame indices `[0 .. N-1]` |
| `frame_timestamps` | Float32 (N) | Synchronized timestamp for each frame |
| `det_frame_indices` | Int32 (M) | Frame index for each person detection |
| `det_track_ids` | Int32 (M) | ByteTrack assigned persistent track ID |
| `det_confs` | Float32 (M) | Detection confidence score |
| `det_boxes` | Float32 (M, 4) | Bounding box coordinates `[x1, y1, x2, y2]` |
| `det_keypoints` | Float32 (M, 17, 3) | 17 COCO pose keypoints `[x, y, conf]` |
| `det_has_keypoints` | Bool (M) | Validity mask for keypoints |

---

## Metric Definitions & Protocols

### 1. Event-Level Confusion Matrix
- **True Positive (TP)**: A confirmed fall alert fires within the event matching window $[t_{\text{anchor}}, t_{\text{anchor}} + 5.0\text{s}]$. At most one alert per fall sequence matches as TP.
- **False Positive (FP)**: 
  - Any alert firing in a verified ADL (normal activity) sequence.
  - Any premature alert firing before the fall begins ($t < t_{\text{anchor}}$).
  - Any alert firing outside the matching window when no fall was matched.
- **False Negative (FN)**: An evaluable fall sequence where the observation window is sufficient ($\ge 2.0\text{s}$ post-fall), but no alert fired in the matching window.
- **True Negative (TN)**: A verified ADL sequence in which zero alerts were emitted.
- **Insufficient Observation (ISO)**: A fall sequence where the video recording cuts off before the required $2.0\text{s}$ floor-confirmation period has elapsed. Excluded from TP/FN denominators.
- **Unverified Sequences**: Recordings with unknown or unverified ground truth. Strictly excluded from all accuracy and rate denominators.

### 2. Zero-Denominator Safety
When a metric denominator is zero, the evaluation pipeline renders `N/A` (undefined) rather than misleading `0.0%` or `100.0%`:
- Precision: `N/A` when $\text{TP} + \text{FP} = 0$.
- Recall: `N/A` when $\text{TP} + \text{FN} = 0$.
- Specificity: `N/A` when $\text{TN} + \text{FP} = 0$.
- Latency: `N/A` when $\text{TP} = 0$.

### 3. Tracking Continuity Metrics
Rather than assuming tracking continuity based solely on single track IDs, the pipeline measures:
- **Detection Coverage %**: Percentage of frames containing at least one person detection.
- **Longest Consecutive Gap**: Longest consecutive frame run with zero person detections.
- **Longest Continuous Track Segment**: Longest uninterrupted consecutive frame run where a single track ID was detected.
- **Track ID Switches**: Number of times the assigned primary track ID changed across detection events.

### 4. Stage-1 Drop Triggers vs Confirmed Alerts
- **Stage 1 (Kinematic Drop Trigger)**: Person's hip drops by $\ge 16\%$ of body height in $\le 0.6\text{s}$ and posture tilts horizontal. Recorded in track scratchpad as `dropped_at`.
- **Stage 2 (Confirmed Alert)**: Posture remains continuously horizontal for $\ge 2.0\text{s}$ on the ground without recovery. Emits confirmed `Candidate(type="fall")`.

---

## Reproducing Evaluations (PowerShell Commands)

Prerequisites: Python 3.10+ installed with dependencies from `requirements.txt`.

### 1. Run the Full 10-Recording Benchmark
Evaluates all 10 recordings (5 falls + 5 ADLs), reports continuity, raw drop triggers, confirmed alerts, and separate comparison tables:
```powershell
python eval/urfd_batch_evaluation.py
```

### 2. Run the ADL False-Alert Baseline
Evaluates the 5 ADL recordings to establish sequence-level specificity and false alerts per minute:
```powershell
python eval/urfd_adl_evaluation.py
```

### 3. Run the Fall Ground-Truth Inspection & Evaluation
Inspects official ground-truth transitions and evaluates the 5 fall recordings:
```powershell
python eval/urfd_annotations.py
```

### 4. Run the Synthetic Trajectory Benchmark
Runs the fast, zero-dependency synthetic baseline:
```powershell
python eval/eval_fall_baseline.py
```

### 5. Run the Automated Test Suite
Executes the full test suite across all evaluation modules, annotations, and detectors:
```powershell
python -m pytest -v
```

### 6. Extracting Fresh Pose Caches
To regenerate pose caches from raw video frames:
```powershell
# Extract ADL sequence
python eval/extract_poses.py --frames-dir data/urfd/adl-01-cam0-rgb --fps 30.0 --output-dir eval/cache/urfd

# Extract Fall sequence
python eval/extract_poses.py --frames-dir data/urfd/fall-01-cam0-rgb --fps 30.0 --output-dir eval/cache/urfd
```

---

## Methodological Nuances & Disclaimers

### Prediction-Independent Eligibility & The `fall-04` Nuance
- **Consistent Predefined Requirement**: Sequence eligibility for the primary confirmed-alert recall denominator requires at least $2.0\text{s}$ of post-impact/post-fallen observation footage ($\ge \text{FALL\_DOWN\_CONFIRM\_SECS}$). This rule is evaluated strictly from ground truth timestamps *before* inspecting detector predictions to prevent outcome-dependent eligibility bias.
- **Why All 5 Falls are Ineligible for Primary Recall**:
  - `fall-01`: 1.57s post-fallen footage ($< 2.0\text{s}$) $\rightarrow$ `INSUFFICIENT_OBSERVATION`
  - `fall-02`: 1.33s post-fallen footage ($< 2.0\text{s}$) $\rightarrow$ `INSUFFICIENT_OBSERVATION`
  - `fall-03`: 1.26s post-fallen footage ($< 2.0\text{s}$) $\rightarrow$ `INSUFFICIENT_OBSERVATION`
  - `fall-04`: 1.74s post-fallen footage ($< 2.0\text{s}$) $\rightarrow$ `INSUFFICIENT_OBSERVATION`
  - `fall-05`: 1.43s post-fallen footage ($< 2.0\text{s}$) $\rightarrow$ `INSUFFICIENT_OBSERVATION`
- **Separately Observed Detection on `fall-04`**:
  - In `fall-04`, the kinematic drop was triggered early during the transition phase (State 0) at $t = 1.134\text{s}$.
  - The subject remained down from $1.134\text{s}$ to the clip end at $3.170\text{s}$ ($2.036\text{s} \ge 2.0\text{s}$), allowing the detector to emit a confirmed alert at $t = 3.136\text{s}$ (latency: $1.702\text{s}$ from State 1 impact).
  - To prevent biased recall, `fall-04` is classified as `INSUFFICIENT_OBSERVATION` for primary recall, and its confirmed alert is preserved and reported separately as an observed diagnostic detection.
- **Resulting Primary Metrics**:
  - Primary Recall & F1: **N/A (undefined)** because zero eligible positive fall recordings exist in the primary denominator.
  - Stage-1 Kinematic Drop Sensitivity: **5/5 (100.0%)** — all 5 falls triggered raw kinematic drops.
  - ADL Specificity: **5/5 (100.0%)** with 0.00 false alerts/minute across 840 normal activity frames (~27.8s).

### Small-Sample Limitations
- 10 recordings (~48 seconds total) represents an **exploratory pilot sample**, not a general measure of multi-day real-world reliability.
- Specificity reflects 5/5 normal activity clips; multi-hour unscripted campus footage is required for deployment certification.

---

## Troubleshooting

- **Pose Cache Not Found**: Ensure caches exist in `eval/cache/urfd/` or raw PNG frames are present under `data/urfd/` to regenerate.
- **Cache FPS Mismatch**: `eval/urfd_adl_evaluation.py` validates cached FPS against requested FPS. If an FPS mismatch error occurs, regenerate from source frames or pass `allow_retime=True` if retiming synthetic timestamps is explicitly intended.
- **OneDrive Sync Freezes**: If running inside an active cloud-synced folder, large temporary files or cache updates can cause performance degradation. Move the repository to a local path (e.g. `C:\dev\Arjun`) or add `eval/cache` and `data` to ignore rules.
