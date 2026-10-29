# Machine Learning Feature Engineering & Dataset Preparation (Phase 4A)

This module provides a reproducible, offline, and leak-free feature extraction and tabular dataset construction pipeline for machine learning fall classification in **ARJUN Campus Guardian**.

> **Note on Scope**: This module is strictly for **dataset preparation and feature engineering**. In accordance with Phase 4A guidelines, no classifier (e.g. Random Forest, XGBoost) is trained in this step, and the production heuristic fall detector in `backend/detectors/fall.py` remains completely untouched.

---

## 1. Architecture Overview

The pipeline transforms raw video/pose sequences into ML-ready tabular datasets (CSV/Parquet) through three primary components:

```
┌─────────────────────────────────┐
│ URFD Sequences (fall / adl)     │
│ 10 Recordings (1,516 frames)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ ml/features.py                  │
│ • Instantaneous geometry        │
│ • Joint angles (knees/hips)     │
│ • Missingness indicator flags   │
│ • Causal temporal rolling winds │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ ml/dataset.py                   │
│ • Label mapping (URFD truth)    │
│ • Ambiguity boundary tagging    │
│ • Sequence-level grouping       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ ml/build_dataset.py (CLI)       │
│ • Tabular output:               │
│   ml/data/urfd_features.csv     │
└─────────────────────────────────┘
```

- **[ml/features.py](file:///c:/Users/ayman/Downloads/Arjun/ml/features.py)**: Contains pure geometric calculations (`compute_joint_angle`, `extract_instant_features`), observation tracking, and strictly causal temporal rolling-window kinematic extractors (`FeatureExtractor`).
- **[ml/dataset.py](file:///c:/Users/ayman/Downloads/Arjun/ml/dataset.py)**: Loads sequences from `eval.dataset` and official annotations from `eval.urfd_annotations`, applies three-class labeling (`normal`, `fall_transition`, `fallen`), identifies boundary ambiguities, assigns grouping identifiers, and compiles tabular DataFrames.
- **[ml/build_dataset.py](file:///c:/Users/ayman/Downloads/Arjun/ml/build_dataset.py)**: Command-line interface to execute the extraction and save the final dataset.

---

## 2. Dataset Schema & Column Definitions

The generated tabular dataset contains **61 columns** per sample (detection):

### A. Metadata & Grouping (8 columns)
| Column | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | str | Unique identifier: `{sequence_id}_{track_id}_f{frame_idx:05d}`. |
| `sequence_id` | str | Recording identifier (e.g., `fall-01-cam0-rgb`, `adl-01-cam0-rgb`). |
| `track_id` | int | ByteTrack tracking ID for the person. |
| `group_id` | str | Grouping ID for cross-validation splits (set to `sequence_id` to prevent intra-sequence leak). |
| `frame_idx` | int | 0-indexed frame index in the sequence. |
| `timestamp` | float | Observation timestamp in seconds (synchronized with URFD timeline). |
| `window_start` | float | Starting timestamp of the longest active feature window ($t - 1.0\text{s}$). |
| `window_end` | float | Current timestamp $t$ of the observation window. |

### B. Instantaneous Bounding Box & Scale Features (4 columns)
| Column | Type | Description |
| :--- | :--- | :--- |
| `box_aspect_ratio` | float | Bounding box width / height ($W / H$). High for fallen/lying; low for standing. |
| `norm_body_height` | float | Bounding box height normalized by `body_scale` ($H / S$). |
| `norm_body_width` | float | Bounding box width normalized by `body_scale` ($W / S$). |
| `body_scale` | float | Characteristic torso scale (torso length or $\max(\text{height}, \text{width})$). |

### C. Torso & Upper Body Orientation (8 columns)
| Column | Type | Description |
| :--- | :--- | :--- |
| `torso_angle_deg` | float | Angle (0–90°) between torso axis and vertical. Upright $\approx 0^\circ$, prone/supine $\approx 90^\circ$. |
| `torso_angle_cos` | float | $\cos(\theta_{\text{torso}})$. Near 1.0 when upright; near 0.0 when horizontal. |
| `torso_angle_sin` | float | $\sin(\theta_{\text{torso}})$. Near 0.0 when upright; near 1.0 when horizontal. |
| `shoulder_hip_dx` | float | Horizontal offset from mid-hip to mid-shoulder normalized by scale. |
| `shoulder_hip_dy` | float | Vertical offset from mid-hip to mid-shoulder normalized by scale (negative = shoulder above hip). |
| `shoulder_hip_dist` | float | Distance between mid-shoulder and mid-hip normalized by scale. |
| `norm_shoulder_width` | float | Distance between left and right shoulders normalized by scale. |
| `norm_hip_width` | float | Distance between left and right hips normalized by scale. |

### D. Joint Angles (4 columns)
| Column | Type | Description |
| :--- | :--- | :--- |
| `left_knee_angle_deg` | float | 2D angle at left knee between left hip and left ankle (degrees). Straight leg $\approx 180^\circ$. |
| `right_knee_angle_deg` | float | 2D angle at right knee between right hip and right ankle (degrees). |
| `left_hip_angle_deg` | float | 2D angle at left hip between left shoulder and left knee (degrees). |
| `right_hip_angle_deg` | float | 2D angle at right hip between right shoulder and right knee (degrees). |

### E. Keypoint Visibility & Confidence (7 columns)
| Column | Type | Description |
| :--- | :--- | :--- |
| `kp_visible_fraction` | float | Fraction of the 17 COCO keypoints with confidence $\ge 0.30$. |
| `kp_mean_conf` | float | Mean detection confidence across all 17 keypoints. |
| `has_torso` | int | 1 if both shoulders and both hips are visible; 0 otherwise. |
| `has_hips` | int | 1 if at least one hip is visible; 0 otherwise. |
| `has_knees` | int | 1 if at least one knee is visible; 0 otherwise. |
| `has_ankles` | int | 1 if at least one ankle is visible; 0 otherwise. |
| `has_full_legs` | int | 1 if both knees and both ankles are visible; 0 otherwise. |

### F. Missing-Data Indicator Flags (7 columns)
To avoid silently fabricating or interpolating data, missing geometric properties are stored as `NaN` and accompanied by explicit binary indicators:
| Column | Type | Description |
| :--- | :--- | :--- |
| `missing_torso_angle` | int | 1 if torso angle could not be computed (missing shoulders or hips). |
| `missing_left_knee` | int | 1 if left knee joint angle is missing (occlusion or low confidence). |
| `missing_right_knee` | int | 1 if right knee joint angle is missing. |
| `missing_left_hip` | int | 1 if left hip joint angle is missing. |
| `missing_right_hip` | int | 1 if right hip joint angle is missing. |
| `missing_shoulder_width` | int | 1 if either shoulder is missing. |
| `missing_hip_width` | int | 1 if either hip is missing. |

### G. Multi-Window Temporal Dynamics (30 columns)
Computed over three causal backward rolling windows: $W \in \{300\text{ms}, 600\text{ms}, 1000\text{ms}\}$. Each window computes 10 features:
- `w{W}ms_hip_vertical_drop`: Downward displacement of mid-hip over window, normalized by body scale ($\Delta y_{\text{hip}} / S$). Positive indicates falling down.
- `w{W}ms_hip_vertical_velocity`: Rate of downward hip displacement ($\Delta y_{\text{hip}} / (S \cdot \Delta t)$).
- `w{W}ms_torso_angle_delta`: Change in torso orientation angle from window start to current frame ($\Delta \theta_{\text{deg}}$).
- `w{W}ms_torso_angle_velocity`: Angular velocity of torso tilt ($\Delta \theta / \Delta t$).
- `w{W}ms_aspect_ratio_delta`: Change in bounding-box aspect ratio over the window ($\Delta (W/H)$).
- `w{W}ms_aspect_ratio_velocity`: Rate of aspect ratio broadening ($\Delta (W/H) / \Delta t$).
- `w{W}ms_foot_displacement`: Lateral/horizontal shift of ground foot contact normalized by scale.
- `w{W}ms_foot_speed`: Speed of foot base movement ($\text{dist} / \Delta t$).
- `w{W}ms_horizontal_posture_fraction`: Fraction of frames in the window where torso angle $> 55^\circ$ or $W/H > 1.2$.
- `w{W}ms_sample_count`: Number of past observations present in the rolling window buffer.

### H. Ground Truth Labels & Validation Masks (3 columns)
| Column | Type | Description |
| :--- | :--- | :--- |
| `label` | str | Class label: `"normal"`, `"fall_transition"`, or `"fallen"`. |
| `is_ambiguous` | int | 1 if sample is within $\pm 0.05\text{s}$ of a state transition or `kp_visible_fraction < 0.20`; 0 otherwise. |
| `is_fall_sequence` | int | 1 if recording contains a fall (`fall-01`..`05`); 0 for ADL (`adl-01`..`05`). |

---

## 3. Ground Truth Labeling Rules & Provenance

Ground truth labels are assigned strictly using official URFD synchronized annotations:
1. **ADL Sequences (`adl-01` to `adl-05`)**:
   - `label = "normal"` for all frames.
   - `is_fall_sequence = 0`.
2. **Fall Sequences (`fall-01` to `fall-05`)**:
   - `timestamp < fall_transition_start_s`: `label = "normal"`.
   - `fall_transition_start_s <= timestamp < fallen_state_start_s`: `label = "fall_transition"`.
   - `timestamp >= fallen_state_start_s`: `label = "fallen"`.
   - `is_fall_sequence = 1`.
3. **Ambiguity Mask (`is_ambiguous = 1`)**:
   - Frames within $\pm 0.05\text{s}$ ($\approx 1.5$ frames at 30 FPS) of `fall_transition_start_s` or `fallen_state_start_s` are flagged as ambiguous.
   - Frames where person keypoint visibility is severely degraded (`kp_visible_fraction < 0.20`) are flagged as ambiguous.
   - **Recommended Training Practice**: Exclude rows with `is_ambiguous == 1` during model training to avoid noisy boundary labels.

---

## 4. Leakage Prevention & Causal Temporal Guarantees

1. **Strict Temporal Causality**:
   - Feature calculation at time $t$ consumes **only** historical frames $t_i \le t$. Future frames, future keypoints, clip-ending flags, and alert signals are never accessed.
2. **Recording-Level Grouping**:
   - `group_id = sequence_id` is baked into every sample.
   - **Mandatory Rule for Phase 4B**: Models must be evaluated using `GroupKFold` or `LeaveOneGroupOut` grouped on `group_id`. Temporal adjacent frames from the same video sequence must never be split across train and validation sets.
3. **Person-Independent Evaluation Limitation**:
   - The URFD Camera 0 RGB distribution does not provide official per-subject demographic or identity tags.
   - Consequently, **true person-independent generalization cannot be mathematically guaranteed** across recordings; grouping is strictly sequence-based.

---

## 5. Dataset Statistics (URFD 10-Recording Benchmark)

Running `python ml/build_dataset.py` on the 10 available URFD recordings generates the following sample breakdown:

- **Total Sequences**: 10 (5 fall clips, 5 ADL clips).
- **Total Samples (Detections)**: **1,516** frames.
- **Total Features**: **61** columns.

### Class Distribution
| Class | Sample Count | Percentage |
| :--- | :--- | :--- |
| `normal` | 1,159 | 76.45% |
| `fallen` | 209 | 13.79% |
| `fall_transition` | 148 | 9.76% |
| **Total** | **1,516** | **100.0%** |

### Ambiguity Breakdown
- Non-ambiguous samples: **1,490** (98.29%)
- Ambiguous boundary/low-visibility samples (`is_ambiguous = 1`): **26** (1.71%)

### Missing Data Distribution
- `missing_left_knee`: 2.64% (due to self-occlusion in side/ground profiles)
- `missing_right_knee`: 2.37%
- `missing_left_hip`: 1.12%
- `missing_right_hip`: 0.99%
- `missing_torso_angle`: 0.00% (at least one shoulder and hip present across all valid track frames)
- `missing_shoulder_width`: 0.99%

---

## 6. Reproducible Execution Commands

All commands can be executed directly in PowerShell from the project root:

### Generate ML Dataset
```powershell
python ml/build_dataset.py
```
This produces the tabular dataset at `ml/data/urfd_features.csv` and prints the full statistical summary to the terminal.

### Dry-Run Summary (No File Writing)
```powershell
python ml/build_dataset.py --no-save
```

### Run Unit Tests
```powershell
# Run all ML-specific unit tests (38 tests passing)
python -m pytest tests/test_ml_features.py tests/test_ml_dataset.py tests/test_ml_random_forest.py tests/test_ml_xgboost.py tests/test_ml_events.py tests/test_ml_adapters.py -v

# Run the complete project test suite (228 tests passing)
python -m pytest
```

---

## 7. Random Forest Baseline Classification (Phase 4B)

Phase 4B trains and evaluates an offline, 3-class baseline classifier using Scikit-Learn's `RandomForestClassifier` on the clean pose features.

### A. Execution Commands
```powershell
# Train and evaluate Random Forest baseline with Leave-One-Group-Out cross-validation
python ml/train_random_forest.py

# Run evaluation without saving model to disk
python ml/train_random_forest.py --no-save-model
```

### B. Baseline Architecture & Leakage Safeguards
- **Target Classes**: `normal`, `fall_transition`, `fallen`
- **Leakage Prevention**: Evaluated strictly using **Leave-One-Group-Out Cross-Validation (10 folds)**. Zero video frames from a held-out recording ever appear in the training partition.
- **Preprocessing**: `SimpleImputer(strategy='median')` wrapped in an `sklearn.pipeline.Pipeline` to ensure imputation statistics are computed strictly on training folds.
- **Feature Matrix $X$**: Consists strictly of the **61 causal features** (metadata, timestamps, identities, and alert annotations are completely excluded).

### C. Performance Summary (Held-Out Out-Of-Fold Predictions)
| Metric | Random Forest (10-Fold Held-Out) | Naive Majority Baseline |
| :--- | :---: | :---: |
| **Overall Accuracy** | **94.90%** | 77.45% |
| **Balanced Accuracy** | **83.93%** | 33.33% |
| **Macro F1 Score** | **87.08%** | 29.10% |

- `normal`: 95.33% Precision, 99.05% Recall, 97.15% F1 (Support: 1,154 frames)
- `fall_transition`: 86.05% Precision, 55.22% Recall, 67.27% F1 (Support: 134 frames)
- `fallen`: 96.10% Precision, 97.52% Recall, 96.81% F1 (Support: 202 frames)

Detailed confusion matrices, per-sequence breakdowns, false alert diagnostics, and feature importance rankings are recorded in [ml/results/RANDOM_FOREST_BASELINE.md](file:///c:/Users/ayman/Downloads/Arjun/ml/results/RANDOM_FOREST_BASELINE.md).

---

## 8. XGBoost Fall-State Classifier & Model Comparison (Phase 4C)

Phase 4C evaluates an `xgboost.XGBClassifier` (version 3.1.2) under identical cross-validation splits, target definitions, and feature exclusions.

### A. Execution Commands
```powershell
# Train XGBoost and execute comparative Leave-One-Group-Out evaluation vs Random Forest
python ml/train_xgboost.py

# Run comparison without saving model to disk
python ml/train_xgboost.py --no-save-model
```

### B. Comparative Performance Summary
| Metric | Random Forest (Phase 4B) | XGBoost (Phase 4C) | Winner / Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 94.90% | **96.11%** | +1.21% (XGBoost) |
| **Balanced Accuracy** | 83.93% | **87.61%** | +3.68% (XGBoost) |
| **Macro F1 Score** | 87.08% | **89.90%** | +2.82% (XGBoost) |
| **Fall-Transition Recall** | 55.22% | **66.42%** | **+11.19%** (XGBoost) |
| **Normal Activity False Alerts (FP)** | 11 frames | **7 frames** | -4 frames (XGBoost) |
| **Single-Frame Inference Latency** | 16.84 ms | **1.00 ms** | **16.8x faster** (XGBoost) |

Full comparative breakdowns, confusion matrices, per-recording stats, and latency benchmarks are available in [ml/results/MODEL_COMPARISON.md](file:///c:/Users/ayman/Downloads/Arjun/ml/results/MODEL_COMPARISON.md).

---

## 9. Offline Event-Level Fall Evaluation (Phase 4D)

Phase 4D tests whether frame-level ML predictions can reliably drive confirmed event alerts when combined with a causal temporal state machine (`TemporalFallStateMachine`), comparing them against ARJUN's production rule-based `FallDetector`.

### A. Execution Commands
```powershell
# Execute complete offline event-level comparison across Rule-Based, RF, and XGBoost
python ml/event_evaluation.py
```

### B. Primary Confirmed-Event Metrics ($T_{\text{confirm}} = 2.0\text{s}$ Sustained Posture)
| Metric | Rule-Based (Production) | Random Forest + Temporal FSM | XGBoost + Temporal FSM |
| :--- | :---: | :---: | :---: |
| **Primary Confirmed Recall** | **N/A** (0/0 eligible) | **N/A** (0/0 eligible) | **N/A** (0/0 eligible) |
| **Insufficient-Observation Falls** | 5 / 5 (100%) | 5 / 5 (100%) | 5 / 5 (100%) |
| **ADL False Alerts Count** | **0 alerts** | **0 alerts** | **0 alerts** |
| **ADL Specificity** | **100.0%** (5/5 TN) | **100.0%** (5/5 TN) | **100.0%** (5/5 TN) |
| **ADL False Alert Rate** | **0.00 / hour** | **0.00 / hour** | **0.00 / hour** |
| **Diagnostic Detections ($T=2.0\text{s}$)** | 1 / 5 (`fall-04` at 3.14s) | 1 / 5 (`fall-04` at 3.14s) | 1 / 5 (`fall-04` at 3.07s) |
| **Diagnostic Detections ($T=1.0\text{s}$)** | N/A | **5 / 5 (100%)** | **4 / 5 (80%)** |

Full event-level comparison tables, latency measurements, and limitation analyses are available in [ml/results/EVENT_LEVEL_COMPARISON.md](file:///c:/Users/ayman/Downloads/Arjun/ml/results/EVENT_LEVEL_COMPARISON.md).

### C. Key Methodological Takeaways
1. **Suppression of Frame-Level False Alerts**: In Phase 4B/4C, isolated bending in ADL produced 4 to 8 transient false positive frames. Coupling predictions with the `TemporalFallStateMachine` successfully rejects 100% of these transient states, yielding **0 confirmed false alarms across all 5 ADL sequences**.
2. **Observation Window Truncation**: All 5 URFD fall recordings truncate within 1.26s–1.74s post-fall. Because none satisfy the predefined 2.0s post-fall ground-truth dwell requirement, primary confirmed recall remains mathematically undefined (`N/A`) across all detectors.
3. **ML Event Detection Feasibility**: ML posture and transition predictions can drive reliable temporal event detectors, but extended video footage ($\ge 5\text{s}$ post-impact dwell) and unconstrained ADL activities (e.g. resting on the floor or yoga) are required before production certification can be considered.
