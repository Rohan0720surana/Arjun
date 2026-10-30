# UP-Fall Pilot Validation Report (Phase 5A)

**Branch:** `feature/up-fall-validation`
**Date:** 2026-10-30
**Status:** Complete (Zero Retraining, Unmodified Production Logic)

---

## 1. Executive Summary

This report documents the Phase 5A pilot evaluation of the ARJUN Campus Guardian fall detection system against real recordings from the **UP-Fall Detection Dataset** (Universidad Panamericana).

Evaluation encompasses three real recordings from **Subject 1 (Camera 1 — lateral perspective)** across three distinct scenarios:
1. **Activity 1 (Trial 1):** Forward fall with hands (Fall scenario)
2. **Activity 9 (Trial 1):** Picking up an object from the floor (ADL stoop scenario)
3. **Activity 11 (Trial 1):** Intentional lying down on a mat/bed (Challenging horizontal ADL scenario)

The evaluation compared three detection engines operating under identical 2.0-second sustained-down temporal confirmation criteria:
- **Production Rule-Based Detector:** Motion-energy bounding-box velocity with state-machine dwell.
- **Random Forest Baseline:** 65-feature pose geometry classifier + causal temporal finite state machine (FSM).
- **XGBoost Baseline:** Gradient-boosted decision tree classifier + causal temporal FSM.

---

## 2. Dataset & Recording Specifications

All recordings were processed using microsecond-accurate CSV timestamping and YOLOv8-Pose (`yolov8n-pose.pt`) with ByteTrack for multi-object tracking. Pose caches were generated and validated in `eval/cache/up_fall/`.

| Recording ID | Activity Type | Ground Truth | Frames | Duration (s) | Empirical FPS | ZIP Size (MB) | Tracking Continuity (Primary Track) |
|---|---|---|---|---|---|---|---|
| `Subject1Activity1Trial1Camera1` | Act 1: Fall Forward Hands | **Fall** | 195 | 9.881 | 19.63 | 99.7 | 56.4% coverage (switched Track 1 $\rightarrow$ 2 at impact) |
| `Subject1Activity9Trial1Camera1` | Act 9: Picking Up Object | **ADL** | 181 | 9.925 | 18.14 | 92.2 | 100.0% coverage (Track 1, 0 switches) |
| `Subject1Activity11Trial1Camera1` | Act 11: Intentional Lying | **ADL** | 1062 | 59.928 | 17.70 | 540.6 | 85.7% coverage (Track 1, 3 switches, max gap 19 frames) |
| **Total Pilot Pool** | **3 recordings** | **1 Fall, 2 ADL** | **1,438** | **79.734s** | — | **732.5 MB** | — |

---

## 3. Evaluation Results: Rule-Based vs. Random Forest vs. XGBoost

### 3.1 Event-Level Summary Table

| Sequence | Ground Truth | Rule-Based Raw Drops | Rule-Based Confirmed Alerts | RF Confirmed Alert Emissions | RF Unique Incidents | XGB Confirmed Alert Emissions | XGB Unique Incidents |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Activity 1 (Fall Forward)** | **Fall** | 1 (t=4.031s) | **0** (Missed TP) | **1** (t=6.225s) | **1** (TP) | **1** (t=6.225s) | **1** (TP) |
| **Activity 9 (Picking Up Object)** | **ADL** | 1 (t=4.630s) | **0** (Clean TN) | **0** (Clean TN) | **0** (TN) | **0** (Clean TN) | **0** (TN) |
| **Activity 11 (Intentional Lying)** | **ADL** | 1 (t=0.056s) | **0** (Clean TN) | **2** (t=2.001s, 29.908s) | **1** (FP) | **2** (t=2.001s, 29.908s) | **1** (FP) |
| **Total Evaluation** | **1 Fall, 2 ADL** | **3** | **0 Alerts** | **3 Alert Emissions** | **1 TP, 1 FP** | **3 Alert Emissions** | **1 TP, 1 FP** |

*Note on Alert Emissions vs. Unique Incidents:* On Activity 11, the 2 alert emissions at `t=2.001s` and `t=29.908s` were generated on the same person continuously lying down; they represent **1 unique false-alarm incident**, repeated due to an intermittent ByteTrack tracking gap.

---

### 3.2 Detailed Engine Behavior

#### A. Production Rule-Based Detector
- **Activity 1 (Real Fall):** Detected a rapid vertical drop at `t=4.031s` (+1.006s relative to transition start). However, at `t=4.114s` (just 83 ms later), ByteTrack dropped Track 1 during ground impact and re-initialized as Track 2. Because the dwell confirmation window requires 2.0s on the *same track ID*, the rule-based alert was never emitted.
- **Activity 9 (Picking Up Object):** Triggered a raw drop at `t=4.630s` due to rapid downward torso movement, but the subject stood back up before 2.0s elapsed. Dwell timed out safely without an alert (**0 false alerts**).
- **Activity 11 (Intentional Lying):** The subject began the clip already in a lying posture. The rule-based detector requires a dynamic bounding-box descent velocity trigger before entering dwell; lacking this trigger, it emitted **0 false alerts**.

#### B. Random Forest Baseline
- **Activity 1 (Real Fall):** Track 2 (covering `t=4.210s` to `9.881s`) was classified as `fallen` on 110 of 110 frames. The 2-second temporal FSM confirmed the fall at `t=6.225s` (+0.425s relative to ground-truth fallen state start at `5.800s`), registering a **True Positive**.
- **Activity 9 (Picking Up Object):** 174 frames predicted `normal`, 7 frames predicted `fall_transition`, 0 predicted `fallen`. No alert emitted (**0 false alerts**).
- **Activity 11 (Intentional Lying):** Frame classifier predicted `fallen` on 1370 of 1412 detection feature rows (97.0%). Emitted **2 false alert emissions** on the primary track due to horizontal body posture.

#### C. XGBoost Baseline
- Exhibited identical event-level behavior to Random Forest across all three sequences:
  - Activity 1: Confirmed fall at `t=6.225s` (+0.425s latency).
  - Activity 9: 0 false alerts (175 normal, 6 transition, 0 fallen).
  - Activity 11: 2 false alert emissions at `t=2.001s` and `t=29.908s` on Track 1.

---

## 4. Annotation Ambiguity Audit & Resolution

### 4.1 Root Cause in Official UP-Fall Annotations
In UP-Fall companion CSVs, tag code `11` is overloaded with two distinct semantic meanings:
1. **In Fall Recordings (Activities 1–5):** Code `11` represents the post-fall resting state on the floor (`fallen_state_start_s`) following a fall transition (`1..5`).
2. **In ADL Recordings (Activity 11):** Code `11` represents the intentional lying ADL itself. It does **not** signify a post-fall state.

Prior to the fix, `parse_upfall_timestamps_csv()` searched for code `11` without recording context, anchoring `fallen_state_start_s = 0.000s` and `post_fallen_dwell_s = 59.928s` on Activity 11. This erroneously marked the ADL sequence as an empirical fall with dwell.

### 4.2 Adapter & Parser Fix (`ml/adapters.py`)
- Updated `parse_upfall_timestamps_csv(csv_path, activity_num=None)`:
  - Automatically extracts `activity_num` from filename/path or caller arguments.
  - Contextual resolution: Only treats codes `11` or `20` as post-fall states if the recording belongs to Activities 1–5 (or if fall transition codes `1..5` are observed).
  - For Activity 11, returns `fall_transition_start_s = None`, `fallen_state_start_s = None`, and `post_fallen_dwell_s = None`.
- Updated `UPFallDatasetAdapter.load_annotation(recording_id)`:
  - Passes `activity_num=act_num`.
  - For ADL recordings (`is_fall=False`), explicitly sets `dwell_measured = False` and keeps fall transition and fallen state timestamps as `None`.
- Added unit and regression tests in `tests/test_ml_adapters.py`:
  - `test_upfall_activity11_intentional_lying_not_interpreted_as_fall`
  - `test_upfall_activity_code_11_dual_use_recording_context`

---

## 5. Temporal FSM Tracking-Gap Audit: Repeated Alerts vs. Unique Incidents

On `Subject1Activity11Trial1Camera1`, the FSM generated two alerts on Track 1 (`t=2.001s` and `t=29.908s`). A causal timeline audit verified that these do **not** represent separate incidents:

```
Timeline of Track 1 State Machine:
  t =  0.000s:  Posture predicted 'fallen' -> State: NORMAL -> FALLEN_PENDING
  t =  2.001s:  Dwell >= 2.0s reached     -> State: FALLEN_PENDING -> ALERT_EMITTED (Alert 1 emitted)
  t = 26.752s:  Last observation before gap
  t = 27.769s:  Next observation received (tracking gap = 1.017s)
                Gap exceeds max_tracking_gap_s (1.0s) -> FSM calls reset()
                State reset to NORMAL with alert_emitted = False
                Frame at t=27.769s is 'fallen' -> State: NORMAL -> FALLEN_PENDING (first_fallen_ts = 27.769s)
  t = 29.908s:  Dwell >= 2.0s reached     -> State: FALLEN_PENDING -> ALERT_EMITTED (Alert 2 emitted)
```

**Conclusion:** The two alert emissions represent a **single continuous false alarm incident** caused by voluntary horizontal resting. The second emission was an artifact of tracker fragmentation exceeding the 1.0-second tracking-gap reset threshold.

---

## 6. Primary-Track vs. All-Track Evaluation Dynamics

The ML evaluation pipeline operates in an unsupervised manner, selecting the primary subject track via maximum frame count (`most_common(1)[0][0]`) without using ground-truth bounding box labels.

An audit across all tracks in `Subject1Activity11Trial1Camera1` revealed:

| Track ID | Frame Count | Active Time Window | RF Alerts | XGB Alerts | Description |
|---|---|---|:---:|:---:|---|
| **Track 1** (Primary) | 910 | [0.00s – 59.93s] | **2** | **2** | Primary subject track (intermittent gaps) |
| **Track 3** | 91 | [0.44s – 5.60s] | 1 | 1 | Co-occurring fragment during early lying |
| **Track 6** | 363 | [8.30s – 28.92s] | 1 | 1 | Alternate track ID assigned during partial occlusion |
| **Tracks 2, 8, 9, 10, 12, 16, 17** | 1–39 | Transient | 0 | 0 | Transient ID fragments (< 2.0s dwell) |

### Key Architectural Takeaways:
1. **Primary-Track Limitation:** If tracking fragments across impact (as observed in Activity 1), evaluating only the longest track can cause a detector to evaluate only the post-impact or pre-impact phase.
2. **All-Track Limitation:** Evaluating all tracks naively without cross-track spatial stitching produces duplicate alert emissions for a single person when ByteTrack swaps IDs.
3. **Identity-Aware Re-identification:** Future work should incorporate appearance/spatial re-identification to stitch tracks before feeding them to temporal state machines.

---

## 7. Real Fall Pilot Findings (`Subject1Activity1Trial1Camera1`)

The evaluation on real fall recording `Subject1Activity1Trial1Camera1` highlighted key comparative behaviors between rule-based and ML detectors:
1. **Impact-Induced Track Fragmentation:** At `t=4.114s`, high vertical acceleration and body deformation caused ByteTrack to lose Track 1 and initialize Track 2 at `t=4.210s`.
2. **Rule-Based Vulnerability:** The rule-based detector detected the drop on Track 1 at `t=4.031s`, but Track 1 ceased existing 83 ms later. The 2-second dwell confirmation on Track 1 failed to complete, resulting in a false negative.
3. **ML Robustness to Mid-Air Fragmentation:** Track 2 entered directly into the post-fall resting state. Because Random Forest and XGBoost evaluate pose geometry rather than trajectory velocity, they recognized Track 2 as `fallen` immediately, confirmed the dwell after 2.0 seconds (`t=6.225s`), and successfully caught the fall.

---

## 8. Scientific Limitations & Future Work

1. **Sample Size:** This pilot evaluated 3 recordings (1 Fall, 2 ADL) from 1 subject. No generalized false-alarm rate (FAR) or statistical sensitivity can be claimed without executing the full cross-subject evaluation described in `ml/results/DATASET_EXPANSION_PLAN.md`.
2. **Static Pose vs. Dynamic Motion:** Both RF and XGBoost rely purely on instantaneous spatial keypoints. Horizontal body orientation (such as sleeping or voluntary floor exercises) is mathematically indistinguishable from post-fall lying based on instantaneous geometry alone. Addressing Activity 11 false alarms requires integrating temporal velocity features or pre-impact context.
3. **Cross-Track Association:** Tracker fragmentation during high-energy impacts requires tracker-level track stitching to preserve temporal history across ID switches.

---

## 9. Verification & Regression Status

- **Unit & Regression Test Suite:** `python -m pytest` executed cleanly with **232 passed in 13.23s** (zero failures).
- **Git & Data Cleanliness:** No raw datasets, extracted images, model binaries, or pose caches are exposed to git. Production detector thresholds remain unchanged.
