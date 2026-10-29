# ARJUN Campus Guardian — Dataset Expansion & Multi-Dataset Evaluation Plan (Phase 4E)

---

## 1. Executive Summary

Phases 3 and 4 demonstrated that while machine learning (Random Forest and XGBoost) and heuristic detectors extract strong posture and kinematic signals from URFD Camera 0 recordings, **the current 10-recording pilot dataset is structurally incapable of certifying production fall detection**:

1. **Truncated Post-Fall Observation**: All 5 fall recordings truncate between 1.26s and 1.74s post-impact. Because none provide $\ge 2.0\text{s}$ of ground-truth floor dwell, primary confirmed recall is mathematically undefined (**N/A**).
2. **Missing Subject Demographics**: URFD Camera 0 RGB metadata lacks subject IDs; true person-independent generalization (Leave-One-Subject-Out) cannot be guaranteed.
3. **Absence of Challenging ADLs**: The 5 ADL clips lack intentional ground-level activities (lying on rugs, stretching, yoga, resting after exercise).

**Goal of Phase 4E**: Establish a rigorous audit of existing results, evaluate candidate public RGB fall datasets, and deploy a reusable adapter infrastructure (`ml/adapters.py`) allowing new multi-subject, multi-camera datasets to feed directly into ARJUN's pose extraction, feature pipeline, ML classifiers, and temporal event detectors.

---

## 2. Audit of Existing URFD Results & Ground-Truth Reconciliation

### A. Ground-Truth Dwell Reconciliation
In Phase 3B and Phase 4D, slight variances in text reporting of post-fallen durations were audited and reconciled against canonical ground truth in `eval/urfd_annotations.py`:

| Sequence ID | Total Frames | Video Duration | Transition Start | Fallen State Start | Post-Fallen Ground-Truth Dwell | Eligible for 2.0s Confirmation? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fall-01` | 160 | 5.305s | 2.736s | 3.737s | **1.568s (~1.57s)** | **NO** (< 2.0s) |
| `fall-02` | 110 | 3.637s | 1.301s | 2.302s | **1.335s (~1.34s)** | **NO** (< 2.0s) |
| `fall-03` | 215 | 7.207s | 4.929s | 5.949s | **1.258s (~1.26s)** | **NO** (< 2.0s) |
| `fall-04` | 96 | 3.170s | 0.433s | 1.434s | **1.736s (~1.74s)** | **NO** (< 2.0s) |
| `fall-05` | 151 | 5.031s | 2.602s | 3.603s | **1.428s (~1.43s)** | **NO** (< 2.0s) |

**Audit Findings**:
- Post-fallen duration ranges from **1.26s** (`fall-03`) to **1.74s** (`fall-04`).
- Every single fall clip fails the predefined 2.0s post-fallen ground-truth requirement.
- **Prediction-Independent Eligibility Preserved**: All 5 fall recordings remain properly classified as `INSUFFICIENT_OBSERVATION` for primary 2.0s confirmed recall across all detectors (Rule-based, RF, XGBoost).
- **Separately Observed Detections**: `fall-04` triggered an alert across all three detectors because its drop started at $t=1.13\text{s}$, giving > 2.0s of continuous tracking before the clip ended at $t=3.17\text{s}$.

### B. Verification of Causal Event State Machine
We audited `TemporalFallStateMachine` in `ml/event_evaluation.py`:
- The state machine method signature is strictly: `update(timestamp, predicted_label, prob_fallen)`.
- **Zero ground-truth labels are consumed**. Ground-truth annotations are only referenced post-hoc to evaluate detection latency and classify TP vs. FP vs. Insufficient Observation.
- Tracking gaps ($\Delta t > 1.0\text{s}$) strictly reset the dwell timer, preventing phantom alerts across lost tracks.

---

## 3. Comprehensive Research of Candidate Public Datasets

We evaluated four prominent public fall detection benchmarks against ARJUN's engineering requirements:

| Evaluation Criterion | URFD (Current Pilot) | UP-Fall Detection (Recommended #1) | Le2i Fall Dataset (Recommended #2) | CAUCAFall Dataset (Modern HD) |
| :--- | :--- | :--- | :--- | :--- |
| **Official Source** | Univ. of Rzeszow (2014) | Univ. Panamericana (2019) | Univ. de Bourgogne (2013) | Univ. del Cauca (2022) |
| **Access & Licensing** | Open academic | Open (CC BY 4.0) | Open academic form/direct | Open research with citation |
| **RGB Video Available** | Yes (Cam 0 RGB PNGs) | **Yes (Two RGB Cameras)** | **Yes (191 AVI videos)** | Yes (Full HD 1080p MP4) |
| **Total Recordings** | 10 clips (5 Fall, 5 ADL) | **> 500 trials across 11 tasks** | **191 clips across 4 rooms** | 100 clips (50 Fall, 50 ADL) |
| **Subject IDs Available** | **NO** (Unannotated) | **YES (17 Subjects: S1..S17)** | Partial (by scenario) | **YES (10 Subjects)** |
| **Post-Fall Dwell Time** | **1.26s – 1.74s (Too short)** | **> 5.0 seconds** | **5.0 – 15.0 seconds** | **> 5.0 seconds** |
| **Fall Types** | Forward, lateral, backward | 5 types (forward hands, knees, back, chair, side) | 4 types (forward, back, lateral, syncope/faint) | 10 fall variations |
| **Challenging ADLs** | Basic bend/sit/walk | **Lying down on bed/mat, jumping, picking up** | **Lying on sofa/bed, tying shoes, crouching** | Bed transfer, kneeling, floor reach |
| **Multi-Camera Sync** | Single camera evaluated | Yes (Front & Side view) | Fixed camera per room | Single fixed perspective |

---

### Detailed Candidate Profiles

### 1. UP-Fall Detection Dataset (Primary Recommendation)
- **Provenance**: Martínez-Villaseñor et al., *UP-Fall Detection Dataset: A Multimodal Approach*, Sensors 2019.
- **Why It Solves Our Blockers**:
  1. **17 Identified Subjects**: Allows true **Leave-One-Subject-Out (LOSO)** cross-validation. Adjacent video frames and identical humans never leak into training folds.
  2. **Substantial Post-Fall Floor Dwell**: Trials feature full, unconstrained lying on the floor for 5 to 10 seconds post-impact, allowing primary confirmed-recall ($T_{\text{confirm}} = 2.0\text{s}$ and $5.0\text{s}$) to be evaluated on hundreds of legitimate falls without undefined denominators.
  3. **Intentional Lying Down (ADL Activity 11)**: Includes subjects intentionally lying down on beds and floor mats. This provides the ultimate stress-test for posture-based ML classifiers to verify they do not confuse resting with falling.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### 2. Le2i Fall Detection Dataset (Secondary / Cross-Domain Benchmark)
- **Provenance**: Charfi et al., *Definition and Performance Evaluation of a Robust Fall Detection Approach*, VIBOT 2013.
- **Key Characteristics**: 191 videos captured across 4 realistic home rooms (Home, Coffee room, Office, Lecture room) under varying lighting, shadows, and occlusions (chairs, desks).
- **Value to ARJUN**: Serves as an ideal **held-out domain dataset** to evaluate cross-dataset transferability (train on UP-Fall, evaluate without fine-tuning on Le2i).

---

## 4. Reusable Evaluation Architecture (`ml/adapters.py`)

To ensure modularity and prevent coupling to any specific dataset directory structure, we implemented the abstract adapter pattern in [ml/adapters.py](file:///c:/Users/ayman/Downloads/Arjun/ml/adapters.py):

```
┌─────────────────────────────────────────────────────────────┐
│                       DatasetAdapter                        │
│  • discover_recordings()                                    │
│  • load_annotation(recording_id) -> StandardizedAnnotation  │
│  • load_sequence(recording_id)   -> Sequence                │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
     ┌─────────┴─────────┐          ┌─────────┴─────────┐
     │    URFDAdapter    │          │  UPFallAdapter    │  (Also Le2iAdapter)
     │  (10 recordings)  │          │  (17 subjects)    │
     └─────────┬─────────┘          └─────────┬─────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Standardized ARJUN Evaluation Engine                        │
│  1. YOLOv8 Pose Extraction (`eval/extract_poses.py`)        │
│  2. ByteTrack Tracking (`backend/state.py`)                 │
│  3. Causal Feature Engineering (`ml/features.py`)           │
│  4. Out-of-Fold Inference (RF & XGBoost)                    │
│  5. Temporal Event State Machine (`ml/event_evaluation.py`) │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Guarantees:
- **Annotation Isolation**: `StandardizedAnnotation` carries `subject_id`, `camera_id`, `activity_type`, and transition timestamps exclusively in evaluation metadata. They are never ingested into feature matrices.
- **Zero Production Modification**: `backend/detectors/fall.py` remains untouched.

---

## 5. Multi-Phase Validation Plan (Phase 5 Roadmap)

### Phase 5A: UP-Fall Dataset Ingest & Pose Extraction
1. Download a designated subset of UP-Fall RGB recordings (Subject 1 through Subject 17 across Fall Activities 1–5 and ADL Activities 6–11).
2. Extract pose caches using `eval/extract_poses.py` to `eval/cache/up_fall/*.npz`.
3. Ingest annotations using `UPFallDatasetAdapter`.

### Phase 5B: Person-Independent Cross-Validation (LOSO CV)
1. Split data strictly on `subject_id` (17 folds, training on 16 subjects, evaluating on 1 held-out subject).
2. Train Random Forest and XGBoost baselines under LOSO.
3. Quantify whether feature importances (e.g. `box_aspect_ratio`, `w300ms_horizontal_posture_fraction`, `hip_vertical_drop`) remain consistent across different body shapes and subject heights.

### Phase 5C: Event-Level Fall Confirmation with Extended Floor Dwell ($\ge 5.0\text{s}$)
1. Evaluate `TemporalFallStateMachine` under:
   - Primary standard threshold: $T_{\text{confirm}} = 2.0\text{s}$.
   - High-confidence medical alert threshold: $T_{\text{confirm}} = 5.0\text{s}$.
2. Report true event-level recall, precision, latency, and false alerts per exposure hour with zero undefined denominators.

### Phase 5D: Stress-Testing on Ground-Level ADLs
1. Benchmark detector behavior on intentional lying down, floor sitting, and crouching.
2. Evaluate hybrid gating: combining Stage-1 kinematic drop velocity with ML posture state to ensure resting individuals do not trigger emergency alerts.

---

## 6. Verification & Test Suite Status

- **Unit Tests Added**: [tests/test_ml_adapters.py](file:///c:/Users/ayman/tests/test_ml_adapters.py) (4 tests covering adapter discovery, annotation normalization, dwell calculation, and LOSO subject identification).
- **Full Pytest Suite**:
  ```
  ============================= 224 passed in 12.93s =============================
  ```
  - 190 pre-existing tests passing.
  - 34 ML pipeline tests passing (`test_ml_features.py`, `test_ml_dataset.py`, `test_ml_random_forest.py`, `test_ml_xgboost.py`, `test_ml_events.py`, `test_ml_adapters.py`).
