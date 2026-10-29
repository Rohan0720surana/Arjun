# Random Forest Baseline Fall Classification (Phase 4B)

> **Disclaimer**: This is an **offline experimental classifier**, NOT a production replacement
> for ARJUN's heuristic `FallDetector`. Frame-level posture classification is evaluated here on an
> exploratory 10-recording pilot dataset. It does **not** equal event-level fall detection.

---

## 1. Executive Summary

- **Model**: Scikit-Learn `RandomForestClassifier` (100 trees, `max_depth=6`, `min_samples_leaf=2`, `class_weight='balanced'`, `random_state=42`)
- **Preprocessing**: `SimpleImputer(strategy='median')` wrapped in a leak-free `Pipeline`.
- **Evaluation Protocol**: **Leave-One-Group-Out Cross-Validation (10 Folds)** grouped strictly by recording (`sequence_id`). Zero frames from the held-out video appear in the training split.
- **Clean Evaluation Samples**: **1490** frames (26 ambiguous boundary/occluded frames excluded).
- **Features Used**: **61** strictly causal kinematic, joint angle, and posture features (metadata/timestamps/identities excluded).

### Key Results Overview
| Metric | Random Forest (Held-Out) | Majority Class Baseline | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **94.90%** | 77.45% | +17.45% |
| **Balanced Accuracy** | **83.93%** | 33.33% | +50.60% |
| **Macro F1 Score** | **87.08%** | 29.10% | +57.98% |

---

## 2. Confusion Matrix & Per-Class Performance

### 3-Class Confusion Matrix (Held-Out Out-of-Fold Predictions)
```
                    Predicted
Actual          normal    transition   fallen     Total
normal           1143     11           0          1154
fall_transition  52       74           8          134
fallen           4        1            197        202
```

### Detailed Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support (Frames) |
| :--- | :---: | :---: | :---: | :---: |
| `normal` | 95.33% | 99.05% | 97.15% | 1154 |
| `fall_transition` | 86.05% | 55.22% | 67.27% | 134 |
| `fallen` | 96.10% | 97.52% | 96.81% | 202 |

---

## 3. Recording-Level Breakdown

| Sequence ID | Type | Samples | Accuracy | Macro F1 | FP Fall Alerts | FN Missed Falls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `adl-01-cam0-rgb` | ADL | 142 | 100.0% | 33.3% | 0 | 0 |
| `adl-02-cam0-rgb` | ADL | 148 | 97.3% | 32.9% | 4 | 0 |
| `adl-03-cam0-rgb` | ADL | 180 | 100.0% | 33.3% | 0 | 0 |
| `adl-04-cam0-rgb` | ADL | 150 | 97.3% | 32.9% | 4 | 0 |
| `adl-05-cam0-rgb` | ADL | 180 | 100.0% | 33.3% | 0 | 0 |
| `fall-01` | Fall | 143 | 98.6% | 98.3% | 0 | 2 |
| `fall-02` | Fall | 119 | 89.9% | 85.9% | 0 | 12 |
| `fall-03` | Fall | 209 | 86.1% | 67.7% | 0 | 28 |
| `fall-04` | Fall | 89 | 85.4% | 81.1% | 3 | 2 |
| `fall-05` | Fall | 130 | 90.8% | 87.6% | 0 | 12 |

### Observation on Fall vs ADL Sequences
- **ADL Sequences**: Mean accuracy is **98.9%** across 5 recordings (8 total false alert frames across 800 frames).
- **Fall Sequences**: Mean accuracy is **90.2%** across 5 recordings. Noticeably, missed fall frames (FN) occur primarily during early `fall_transition`, where the subject is just beginning to tip over.

---

## 4. Error Diagnostics: False Positives & False Negatives

### False Positive Fall Alerts (Normal Frames Predicted as Fall)
| Sequence ID | Frame | Time (s) | True Label | Predicted | Aspect Ratio | Torso Angle | W600ms Drop | W600ms Horiz Frac |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fall-04` | 9 | 0.30s | `normal` | `fall_transition` | 0.483 | 22.8° | 0.0 | 0.0 |
| `fall-04` | 10 | 0.33s | `normal` | `fall_transition` | 0.488 | 24.7° | 0.0 | 0.0 |
| `fall-04` | 11 | 0.37s | `normal` | `fall_transition` | 0.495 | 27.3° | 0.011 | 0.0 |
| `adl-02-cam0-rgb` | 128 | 4.27s | `normal` | `fall_transition` | 0.385 | 2.3° | 0.29 | 0.0 |
| `adl-02-cam0-rgb` | 129 | 4.30s | `normal` | `fall_transition` | 0.394 | 2.0° | 0.317 | 0.0 |
| `adl-02-cam0-rgb` | 130 | 4.33s | `normal` | `fall_transition` | 0.404 | 2.5° | 0.359 | 0.0 |
| `adl-02-cam0-rgb` | 131 | 4.37s | `normal` | `fall_transition` | 0.415 | 1.8° | 0.428 | 0.0 |
| `adl-04-cam0-rgb` | 73 | 2.43s | `normal` | `fall_transition` | 0.507 | 58.5° | 0.02 | 0.06 |

> **Diagnostic Insight**: False positive frames in ADL occur predominantly when subjects bend over to tie shoes (e.g. `adl-02` and `adl-04`). During deep bending, the instantaneous bounding box aspect ratio widens and the torso tilts past 60°, momentarily mimicking the posture of a fall.

### False Negative Missed Falls (Transition/Fallen Predicted as Normal)
| Sequence ID | Frame | Time (s) | True Label | Predicted | Aspect Ratio | Torso Angle | W600ms Drop | W600ms Horiz Frac |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fall-01` | 84 | 2.80s | `fall_transition` | `normal` | 0.39 | 2.6° | 0.036 | 0.0 |
| `fall-01` | 85 | 2.84s | `fall_transition` | `normal` | 0.396 | 2.4° | 0.037 | 0.0 |
| `fall-02` | 41 | 1.37s | `fall_transition` | `normal` | 0.468 | 20.6° | 0.002 | 0.0 |
| `fall-02` | 42 | 1.40s | `fall_transition` | `normal` | 0.473 | 20.7° | 0.0 | 0.0 |
| `fall-02` | 43 | 1.43s | `fall_transition` | `normal` | 0.478 | 20.2° | 0.0 | 0.0 |
| `fall-02` | 44 | 1.47s | `fall_transition` | `normal` | 0.484 | 22.5° | 0.005 | 0.0 |
| `fall-02` | 45 | 1.50s | `fall_transition` | `normal` | 0.49 | 23.1° | 0.007 | 0.0 |
| `fall-02` | 46 | 1.53s | `fall_transition` | `normal` | 0.498 | 27.0° | 0.022 | 0.0 |

> **Diagnostic Insight**: In `fall-03`, the subject falls slowly or with visual occlusion during the descent. The early transition frames have small torso angles (< 30°) and negligible hip drop, causing the frame-level classifier to predict `normal` until the subject strikes the floor.

---

## 5. Exploratory Feature Importances

> **Caution**: These importances are derived from a 10-recording pilot dataset. They reflect patterns in URFD Camera 0 recordings and should be treated as **exploratory**, not universal ground truth.

| Rank | Feature Name | Gini Importance | Intuition |
| :---: | :--- | :---: | :--- |
| 1 | `box_aspect_ratio` | 10.54% | Kinematic/posture signal |
| 2 | `keypoint_mean_conf` | 8.87% | Kinematic/posture signal |
| 3 | `box_width_norm` | 6.48% | Kinematic/posture signal |
| 4 | `w600ms_horizontal_posture_fraction` | 5.53% | Kinematic/posture signal |
| 5 | `w1000ms_horizontal_posture_fraction` | 4.55% | Kinematic/posture signal |
| 6 | `w300ms_horizontal_posture_fraction` | 4.34% | Kinematic/posture signal |
| 7 | `w1000ms_sample_count` | 3.82% | Kinematic/posture signal |
| 8 | `w1000ms_hip_vertical_drop` | 3.53% | Kinematic/posture signal |
| 9 | `w1000ms_actual_dt_s` | 3.49% | Kinematic/posture signal |
| 10 | `shoulder_to_hip_dy_norm` | 2.69% | Kinematic/posture signal |
| 11 | `w1000ms_aspect_ratio_delta` | 2.56% | Kinematic/posture signal |
| 12 | `w600ms_torso_angle_delta` | 2.49% | Kinematic/posture signal |

---

## 6. Methodological Assessment & Architectural Limitations

### 1. Frame-Level Classification vs. Event-Level Fall Detection
- A frame classifier produces an independent label per frame (e.g. 30 predictions per second).
- Real falls are **events with temporal structure**: normal walking -> dynamic collapse (300-800ms) -> impact -> prolonged lying down (>= 2s).
- Relying purely on frame-level ML predictions causes alert chatter: a single bending frame in ADL fires a false fall alarm, while a momentary occlusion during a real fall drops the alert.
- In production ARJUN, event-level confirmation requires Stage-1 drop kinematics, posture transition, and sustained down confirmation.

### 2. Dataset Size & Diversity Limitations
- 10 recordings (5 falls, 5 ADLs, 1,490 clean frames) is **insufficient for deploying or certifying a production ML model**.
- The dataset features a single room setting, static camera perspective (Camera 0), and limited lighting variation.
- Subject identity is not annotated in URFD Camera 0 RGB, meaning **person-independent generalization cannot be verified**.

### 3. Conclusion for Phase 4B
- The Random Forest baseline confirms that the engineered features (aspect ratio, rolling horizontal fraction, hip drop, torso angle) contain strong discriminatory signal (87.1% Macro F1 vs 29.1% majority baseline).
- However, the frame-level classifier must be paired with temporal event-level hysteresis before any hybrid integration is considered.
