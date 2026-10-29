# ARJUN Fall Detection Evaluation: 10-Recording URFD Real-World Benchmark (Phase 3B)

**Dataset**: UR Fall Detection Dataset (URFD), Camera 0 (RGB)  
**Evaluator**: Offline Event-Level Fall Evaluator (`eval/urfd_batch_evaluation.py`)  
**Production Detector**: `backend/detectors/fall.py` (Rule-Based, 2.0s floor-confirmation requirement)  
**Evaluation Date**: 2026-10-30  
**Test Suite Status**: 190/190 tests passing  

---

## 1. Executive Summary

This report documents the empirical evaluation of ARJUN's production rule-based `FallDetector` across **10 real-world recordings** (5 fall sequences and 5 normal daily living sequences) from the UR Fall Detection Dataset (URFD).

### Headline Results
- **Prediction-Independent Protocol**: The predefined 2.0-second post-impact observation requirement is applied consistently across all fall clips *before* evaluating detector predictions, preventing outcome-dependent eligibility bias.
- **Stage 1 Kinematic Drop Sensitivity**: **5/5 (100.0%)** -- All 5 fall recordings successfully registered a raw kinematic drop event (hip drop >= 16% + horizontal posture).
- **Primary Confirmed-Alert Evaluation**: All 5 fall clips cut off within 1.26s-1.74s after the fallen state, shorter than the required 2.0s floor-confirmation period. All 5 are classified as `INSUFFICIENT_OBSERVATION` and excluded from the primary recall denominator. Primary Recall, Precision, and F1 are therefore **N/A (undefined)**.
- **Separately Reported Observed Detection (`fall-04`)**: Despite only 1.74s of post-impact ground truth, an alert was confirmed at $t = 3.14\text{s}$ with a measured latency of **1.70s** relative to official State 1 impact because the kinematic drop triggered early during the transition phase at $t = 1.13\text{s}$ ($2.04\text{s}$ before clip end).
- **Normal Activity (ADL) Specificity**: **5/5 (100.0%)** -- Zero false alarms across all 5 ADL sequences over 27.8s of normal human activity (**0.00 alerts/min**).

---

## 2. Event-Level Confusion Matrix

```
Evaluation Protocol       : Algorithm-neutral event matching (prediction-independent eligibility)
Matching Window           : [impact, impact + 5.0s]
Required Confirmation     : 2.0s sustained horizontal posture

------------------------------------------------------------------------------
Total Sequences Evaluated : 10
   - Fall Scenarios         : 0 eligible in primary denominator (+ 5 insufficient observation)
   - Normal ADL Scenarios   : 5 eligible
------------------------------------------------------------------------------

Event-Level Confusion Matrix:
   - True Positives  (TP)   : 0    (Actual falls correctly detected in window)
   - True Negatives  (TN)   : 5    (Normal ADL correctly ignored)
   - False Positives (FP)   : 0    (Unmatched alerts & ADL false alarms)
   - False Negatives (FN)   : 0    (Missed falls in window)
   - Duplicate Alerts (DUP) : 0    (Redundant re-triggers for same incident)

Excluded from Primary Denominators:
   - Insufficient Obs (ISO) : 5    (Cut off before 2.0s confirmation window)
   - Unverified Sequences   : 0    (Lacks verified ground truth)

Performance Metrics:
   - Precision              : N/A  (0 / 0)
   - Recall (Sensitivity)   : N/A  (0 / 0 eligible falls)
   - F1-Score               : N/A
   - Specificity (TNR)      : 100.0%  (5 / 5 ADLs)
   - Detection Latency      : N/A  (No eligible TPs in primary denominator)

Separately Reported Diagnostic Detections (Truncated Clips):
   - Confirmed Alerts       : 1 observed alert (fall-04 confirmed at t=3.14s)
   - Observed Alert Latency : 1.70s (time from ground-truth impact to confirmed alert)
```

> **IMPORTANT STATISTICAL WARNING**:  
> The 100% specificity metric is an **exploratory pilot result** from 5 short ADL clips (~27.8s total). Primary confirmed-alert recall is mathematically undefined (N/A) because all 5 available URFD fall recordings end before the 2.0-second post-impact observation window.

---

## 3. Detailed Results by Sequence

### Table 1: Fall Recordings (Official Frame Labels & Synchronized Timestamps)
*Timestamps derived from official `fall-XX-data.csv` accelerometer synchronization.*

| Sequence ID | Total Frames | Total Duration | Transition Start (State 0) | Fallen State (State 1) | Post-Fallen Time | Sufficiency (>= 2.0s) | Detection Coverage % | Max Gap | Stage-1 Drop Triggers | Confirmed Alerts | Primary Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fall-01** | 160 | 5.30s | 2.74s | 3.74s | 1.57s | **NO** | 92.5% | 4f (0.13s) | 2 | 0 | `INSUFFICIENT_OBSERVATION` |
| **fall-02** | 110 | 3.64s | 1.30s | 2.30s | 1.33s | **NO** | 100.0% | 0f (0.00s) | 1 | 0 | `INSUFFICIENT_OBSERVATION` |
| **fall-03** | 215 | 7.21s | 4.93s | 5.95s | 1.26s | **NO** | 100.0% | 0f (0.00s) | 1 | 0 | `INSUFFICIENT_OBSERVATION` |
| **fall-04** | 96 | 3.17s | 0.43s | 1.43s | 1.74s | **NO** | 99.0% | 1f (0.03s) | 1 | 1 | `INSUFFICIENT_OBSERVATION`\* |
| **fall-05** | 151 | 5.03s | 2.60s | 3.60s | 1.43s | **NO** | 88.1% | 10f (0.33s) | 2 | 0 | `INSUFFICIENT_OBSERVATION` |

*\*Note on `fall-04`: Post-fallen duration from official State 1 is 1.74s (< 2.0s), classifying it as `INSUFFICIENT_OBSERVATION` for primary recall. A confirmed alert was separately observed at $t = 3.14\text{s}$ (latency: 1.70s) because kinematic drop detection occurred early during transition at $t = 1.13\text{s}$ ($2.04\text{s}$ before clip end).*

---

### Table 2: ADL (Normal Activity) Recordings (Assumed 30.0 FPS)
*Timestamps computed as $t = \text{frame\_index} / 30.0$ (assumed FPS).*

| Sequence ID | Total Frames | Total Duration | FPS Provenance | Detection Coverage % | Longest Detection Gap | Longest Track Segment | Track ID Switches | Stage-1 Drop Triggers | Confirmed Alerts | Final Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **adl-01-cam0-rgb** | 150 | 4.97s | 30.0 (assumed) | 94.7% | 8f (0.27s) | 142f (4.73s) | 0 | 0 | 0 | `TN` |
| **adl-02-cam0-rgb** | 180 | 5.97s | 30.0 (assumed) | 82.2% | **32f (1.07s)** | 148f (4.93s) | 0 | 0 | 0 | `TN` |
| **adl-03-cam0-rgb** | 180 | 5.97s | 30.0 (assumed) | 100.0% | 0f (0.00s) | 180f (6.00s) | 0 | 0 | 0 | `TN` |
| **adl-04-cam0-rgb** | 150 | 4.97s | 30.0 (assumed) | 100.0% | 0f (0.00s) | 150f (5.00s) | 0 | 0 | 0 | `TN` |
| **adl-05-cam0-rgb** | 180 | 5.97s | 30.0 (assumed) | 100.0% | 0f (0.00s) | 180f (6.00s) | 0 | 1 | 0 | `TN` |

- **Total ADL Exposure**: 27.83 seconds (~0.46 minutes)
- **False Alert Rate**: **0.00 alerts/min** (0 alerts over 27.8s)
- **Sequence-Level Specificity**: **100.0%** (5 / 5)

---

## 4. In-Depth Technical Analysis

### A. True Tracking Continuity vs ByteTrack ID Persistence
A key reliability improvement in Phase 3A/3B was eliminating the false assumption that having a single unique track ID implies unbroken tracking.

In **`adl-02-cam0-rgb`**:
- ByteTrack assigned track ID `1` to all detections in the sequence.
- However, **32 consecutive frames (1.07 seconds)** had zero person detections due to tracker loss behind furniture.
- Despite this 1.07s gap, ByteTrack successfully re-identified the subject when they reappeared, preserving track ID 1.
- By reporting `longest_consecutive_detection_gap = 32f`, the pipeline exposes tracker dropouts that would otherwise remain hidden.

### B. The `fall-04` Methodological Correction
In earlier exploratory runs, `fall-04` was classified as a True Positive while the other 4 falls were excluded as `INSUFFICIENT_OBSERVATION`. This produced outcome-dependent eligibility bias:
1. **The Flaw**: Conditioning sequence eligibility on whether the detector fired allows short clips with alerts to count as TPs while short clips without alerts are discarded, artificially inflating recall.
2. **The Correction**: Applying the predefined 2.0-second post-impact observation rule consistently across all sequences before inspecting predictions reveals that all 5 URFD fall clips truncate before 2.0s post-fall (1.26s-1.74s).
3. **The Result**: All 5 fall clips are classified as `INSUFFICIENT_OBSERVATION` for primary recall, leaving the primary recall denominator empty ($TP+FN = 0$) and rendering primary recall correctly as `N/A`.
4. **Preserved Observation**: The successful confirmation in `fall-04` ($t = 3.136\text{s}$, latency $1.70\text{s}$) is retained and reported separately as an observed diagnostic detection, proving the detector's early kinematic drop capability without compromising protocol integrity.

### C. Stage 1 Drop Sensitivity vs Confirmed Alert Recall
- **Stage 1 Kinematic Sensitivity**: **100.0% (5/5)**. In all 5 fall clips, `track.scratch("fall")["dropped_at"]` was registered. The kinematic drop rule (hip drop fraction >= 0.16 in <= 0.6s + horizontal torso/aspect) had zero misses.
- **Primary Confirmed Alert Recall**: **N/A (undefined)** across available URFD clips due to truncated post-fall recording durations.

### D. 2.0s Confirmation Prevents False Alarms (`adl-05`)
In **`adl-05`**, a subject sat down or bent down quickly, triggering a raw Stage-1 drop event.
- However, the subject did not remain horizontal on the floor for 2.0 seconds.
- The 2.0s confirmation requirement cleared the incident when the person recovered upright.
- Confirmed alerts remained at **0**, demonstrating that the 2.0-second confirmation window works as intended to eliminate transient false alarms.

---

## 5. Limitations & Future Work (Phase 4)

1. **Short Clip Duration**: The average URFD clip is only 3 to 7 seconds long. All 5 fall clips cut off less than 1.75 seconds after the subject hit the floor. Longer continuous recordings (30-60s post-fall) are required to test sustained lying behavior and establish an empirical primary recall figure.
2. **Sample Size**: 10 clips (~48s total) is an exploratory benchmark. Real-world validation requires hundreds of hours of multi-person campus activity.
3. **ADL Synchronization**: ADL timestamps are derived at an assumed 30.0 FPS. While frame counts are exact (150 and 180 frames), hardware-synchronized timestamps are desirable for millisecond latency validation.
4. **Single Camera Perspective**: All sequences were recorded from Camera 0 (wall/tripod mount). Multi-camera fusion and steep overhead angles will be evaluated in Phase 4.

---

## 6. Reproduction Commands

To reproduce this exact benchmark report:
```powershell
python eval/urfd_batch_evaluation.py
```
To run the automated regression tests:
```powershell
python -m pytest tests/test_urfd_batch_evaluation.py -v
```
