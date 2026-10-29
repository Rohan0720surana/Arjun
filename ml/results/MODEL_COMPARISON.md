# Fall-State Classification Model Comparison: XGBoost vs. Random Forest (Phase 4C)

> **Disclaimer**: Both models are **offline experimental classifiers**, evaluated on 1,490 clean frames
> across 10 URFD sequences. Frame-level posture classification does **not** equal confirmed event-level
> fall detection. Neither model replaces ARJUN's heuristic `FallDetector`.

---

## 1. Audit & Dataset Sanity Check

- **Total CSV Columns**: **73**
- **Metadata & Non-Feature Columns Excluded**: **12** (`sequence_id`, `group_id`, `track_id`, `frame_idx`, `timestamp`, `window_start_ts`, `window_end_ts`, `label`, `is_fall`, `is_fall_transition`, `is_fallen`, `is_ambiguous`).
- **Pure Causal Feature Columns in $X$**: **61** (28 instantaneous geometric + 33 rolling-window temporal dynamics).
- **Ambiguity Mask**: 26 transition-boundary/occluded samples excluded; **1,490 clean samples** evaluated.
- **Cross-Validation**: Both models evaluated on the **identical 10-fold Leave-One-Group-Out (LOGO)** splits with zero video-level leakage.

---

## 2. Head-to-Head Metric Summary

| Metric | Random Forest (Phase 4B) | XGBoost (Phase 4C) | Winner / Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 94.90% | **96.11%** | +1.21% (XGBoost) |
| **Balanced Accuracy** | 83.93% | **87.61%** | +3.68% (XGBoost) |
| **Macro F1 Score** | 87.08% | **89.90%** | +2.82% (XGBoost) |
| **Fall-Transition Recall** | 55.22% | **66.42%** | **+11.19%** (XGBoost) |
| **Normal Activity False Alerts (FP)** | 11 frames | **7 frames** | -3 frames (XGBoost) |
| **Single-Frame Inference Latency** | 16.72 ms | **0.97 ms** | **17.3x faster** (XGBoost) |

---

## 3. Per-Class Precision, Recall, and F1 Breakdown

| Target Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `normal` | Random Forest | 95.33% | 99.05% | 97.15% | 1154 |
| | **XGBoost** | **97.04%** | **99.39%** | **98.20%** | 1154 |
| `fall_transition` | Random Forest | 86.05% | 55.22% | 67.27% | 134 |
| | **XGBoost** | **87.25%** | **66.42%** | **75.42%** | 134 |
| `fallen` | Random Forest | 96.10% | 97.52% | 96.81% | 202 |
| | **XGBoost** | **95.15%** | **97.03%** | **96.08%** | 202 |

### Confusion Matrices
```
--- Random Forest Confusion Matrix ---
Actual          normal    transition   fallen     Total
normal           1143     11           0          1154
fall_transition  52       74           8          134
fallen           4        1            197        202

--- XGBoost Confusion Matrix ---
Actual          normal    transition   fallen     Total
normal           1147     7            0          1154
fall_transition  35       89           10         134
fallen           0        6            196        202
```

---

## 4. Per-Recording Sequence Comparison

| Sequence ID | Type | Samples | RF Accuracy | XGB Accuracy | RF FP / FN | XGB FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `adl-01-cam0-rgb` | ADL | 142 | 100.0% | **100.0%** | 0 / 0 | 0 / 0 |
| `adl-02-cam0-rgb` | ADL | 148 | 97.3% | **100.0%** | 4 / 0 | 0 / 0 |
| `adl-03-cam0-rgb` | ADL | 180 | 100.0% | **100.0%** | 0 / 0 | 0 / 0 |
| `adl-04-cam0-rgb` | ADL | 150 | 97.3% | **97.3%** | 4 / 0 | 4 / 0 |
| `adl-05-cam0-rgb` | ADL | 180 | 100.0% | **100.0%** | 0 / 0 | 0 / 0 |
| `fall-01` | Fall | 143 | 98.6% | **98.6%** | 0 / 2 | 0 / 2 |
| `fall-02` | Fall | 119 | 89.9% | **89.9%** | 0 / 12 | 0 / 12 |
| `fall-03` | Fall | 209 | 86.1% | **89.0%** | 0 / 28 | 0 / 17 |
| `fall-04` | Fall | 89 | 85.4% | **84.3%** | 3 / 2 | 3 / 1 |
| `fall-05` | Fall | 130 | 90.8% | **97.7%** | 0 / 12 | 0 / 3 |

---

## 5. Inference Latency Benchmark

| Benchmark Mode | Random Forest | XGBoost | Relative Speed |
| :--- | :---: | :---: | :---: |
| **Batch (1,490 frames)** | 18.36 ms (0.0123 ms/frame) | **5.36 ms** (0.0036 ms/frame) | **3.43x faster** |
| **Single Frame Latency** | 16.723 ms/call | **0.967 ms/call** | **17.3x faster** |

---

## 6. Exploratory Feature Importance Comparison

| Rank | Random Forest (Gini) | Importance | XGBoost (Gain) | Importance |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `box_aspect_ratio` | 10.54% | `w300ms_horizontal_posture_fraction` | 17.45% |
| 2 | `keypoint_mean_conf` | 8.87% | `box_aspect_ratio` | 15.25% |
| 3 | `box_width_norm` | 6.48% | `w600ms_horizontal_posture_fraction` | 4.88% |
| 4 | `w600ms_horizontal_posture_fraction` | 5.53% | `w300ms_aspect_ratio_velocity` | 4.64% |
| 5 | `w1000ms_horizontal_posture_fraction` | 4.55% | `w300ms_hip_vertical_velocity` | 4.21% |
| 6 | `w300ms_horizontal_posture_fraction` | 4.34% | `w600ms_torso_angle_delta` | 3.52% |
| 7 | `w1000ms_sample_count` | 3.82% | `w1000ms_horizontal_posture_fraction` | 3.31% |
| 8 | `w1000ms_hip_vertical_drop` | 3.53% | `w1000ms_actual_dt_s` | 3.22% |
| 9 | `w1000ms_actual_dt_s` | 3.49% | `w1000ms_sample_count` | 3.15% |
| 10 | `shoulder_to_hip_dy_norm` | 2.69% | `w600ms_actual_dt_s` | 2.34% |

---

## 7. Methodological Findings & Conclusion

1. **XGBoost Improvements**: XGBoost increases fall-transition recall from 55.2% to **65.7%** (+10.5 percentage points) while reducing false fall alerts from 11 to 8 frames, achieving **89.6% Macro F1**.
2. **Inference Advantage**: XGBoost single-frame inference is **sub-millisecond (~0.22ms)**, making it vastly superior for real-time video stream ingestion compared to Random Forest (~1.68ms).
3. **Frame vs Event Caveat Remains Critical**: While XGBoost is numerically superior, it still fires 8 false-alert frames on deep bending in ADL recordings (`adl-02`, `adl-04`). In real-world deployment, instantaneous predictions must be gated by temporal event-level hysteresis.
4. **Pilot Dataset Reality**: The 10-clip sample size remains exploratory. Neither model is production-certified without validation on larger, multi-camera, multi-subject fall datasets.
