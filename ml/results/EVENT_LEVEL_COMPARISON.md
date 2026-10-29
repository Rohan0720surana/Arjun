# Fall-Event Detection Comparison: Production Rule-Based vs. Random Forest vs. XGBoost (Phase 4D)

> **Disclaimer**: This is an **offline experimental evaluation** assessing whether frame-level ML
> predictions can reliably support event-level fall alerts under temporal confirmation state machines.
> ARJUN's production `FallDetector` remains unchanged.

---

## 1. Executive Summary & Core Comparison

All three detection pipelines were evaluated on the **identical 10 URFD sequences** (5 Falls, 5 ADLs, ~1,490 clean frames).
ML models use **Leave-One-Group-Out out-of-fold predictions**, ensuring zero training data from a sequence informs its event evaluation.

### Primary Confirmed-Event Metrics ($T_{\text{confirm}} = 2.0\text{s}$ Sustained Posture)
| Metric | Rule-Based Detector (Production) | Random Forest + Temporal FSM | XGBoost + Temporal FSM |
| :--- | :---: | :---: | :---: |
| **Primary Confirmed Recall** | **N/A** (0/0 eligible) | **N/A** (0/0 eligible) | **N/A** (0/0 eligible) |
| **Primary Confirmed F1** | **N/A** | **N/A** | **N/A** |
| **Insufficient-Observation Falls** | 5 / 5 (100%) | 5 / 5 (100%) | 5 / 5 (100%) |
| **ADL False Alerts Count** | **0 alerts** | **0 alerts** | **0 alerts** |
| **ADL Specificity** | **100.0%** (5/5 TN) | **100.0%** (5/5 TN) | **100.0%** (5/5 TN) |
| **ADL False Alert Rate** | **0.00 / hour** | **0.00 / hour** | **0.00 / hour** |
| **Diagnostic Fall Detections** | 1 / 5 (`fall-04` at 3.14s) | 1 / 5 (`fall-04` at 3.14s) | 1 / 5 (`fall-04` at 3.07s) |

> **Critical Methodological Finding**: Under the strict, prediction-independent requirement of $\ge 2.0\text{s}$ post-fallen ground-truth observation, **all 5 URFD fall recordings truncate too early (1.26s–1.74s)**. Primary recall is mathematically undefined (N/A) across all detectors. No detector can certify a 2.0-second post-impact dwell when the video terminates before 2 seconds elapse.

---

## 2. Diagnostic Sensitivity Probe ($T_{\text{confirm}} = 1.0\text{s}$ Dwell)

When evaluating with an exploratory 1.0s dwell threshold to observe detector responsiveness within short URFD clips:

| Recording | Ground Truth Fallen Dwell | Rule-Based Stage-1 Trigger | RF Temporal Event Alert | XGBoost Temporal Event Alert |
| :--- | :---: | :---: | :---: | :---: |
| `fall-01` | 1.57s | Drop at 3.47s | ALERT at 5.00s (lat=1.27s) | ALERT at 5.00s (lat=1.27s) |
| `fall-02` | 1.33s | Drop at 2.035s | ALERT at 3.37s (lat=1.07s) | ALERT at 3.37s (lat=1.07s) |
| `fall-03` | 1.26s | Drop at 6.187s | ALERT at 7.21s (lat=1.26s) | No alert |
| `fall-04` | 1.74s | Drop at 1.134s | ALERT at 2.13s (lat=0.70s) | ALERT at 2.07s (lat=0.63s) |
| `fall-05` | 1.43s | Drop at 3.203s | ALERT at 5.03s (lat=1.43s) | ALERT at 5.03s (lat=1.43s) |

### Diagnostic Findings at $T_{\text{confirm}} = 1.0\text{s}$:
- **Random Forest**: Emits confirmed alerts on **5 / 5 falls** (`fall-01`..`fall-05`) with 0 false alerts on ADL.
- **XGBoost**: Emits confirmed alerts on **4 / 5 falls** (`fall-01`, `02`, `04`, `05`). In `fall-03`, XGBoost predicted `fallen` continuously for 0.986s before the recording ended, falling short of 1.00s by 14 milliseconds.
- **ADL False Alerts**: Both ML models maintain **0 false alerts** on all 5 ADL sequences even with the shorter 1.0s threshold.

---

## 3. ADL Normal Activity Specificity & False Alert Analysis

| Recording | Activity | Frame-Level ML False Positives | Event-Level RF Alerts ($T=2.0\text{s}$) | Event-Level XGB Alerts ($T=2.0\text{s}$) | Rule-Based Alerts |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `adl-01` | Walking & Standing | 0 | 0 | 0 | 0 |
| `adl-02` | Tying shoes / bending | 4 frames (RF) / 0 (XGB) | 0 | 0 | 0 |
| `adl-03` | Walking & picking up object | 0 | 0 | 0 | 0 |
| `adl-04` | Sitting & crouching | 4 frames (RF) / 4 frames (XGB) | 0 | 0 | 0 |
| `adl-05` | Sitting down on floor/chair | 0 | 0 | 0 | 0 |

### The Crucial Role of Temporal Event State Machines
At the frame level in Phase 4B/4C, deep bending in `adl-02` and `adl-04` produced 4 to 8 transient false positive frames.
However, when coupled with the **Temporal Fall State Machine**, the person recovered back to upright within 150–200ms.
Because the temporal state machine requires sustained floor dwell, **100% of frame-level false alerts were rejected**, resulting in **0 confirmed event false alarms** across all ADL recordings for both Random Forest and XGBoost.

---

## 4. Assessment: Does ML Meaningfully Improve Event-Level Fall Detection?

### Current Reality on URFD 10 Recordings:
1. **No Measurable Primary Event Recall Advantage**: Because the 5 available URFD fall recordings are truncated ($< 2.0\text{s}$ post-impact dwell), neither ML nor rule-based heuristics can achieve a confirmed 2.0s alert on 4 of the 5 recordings.
2. **Identical Specificity on Tested ADL**: Rule-based, Random Forest, and XGBoost all achieve **0 false alerts (100% specificity)** on the 5 ADL recordings when backed by temporal confirmation.
3. **Where ML Shows Potential**: In diagnostic probing at 1.0s dwell, ML detects the fall without requiring an abrupt vertical velocity drop threshold. However, this sensitivity comes with the risk that prolonged floor sitting or crouching in unconstrained environments could trigger ML alerts unless gated by kinematic drop confirmation.

### What Additional Data Is Required for Definitive Evaluation:
- **Extended Post-Impact Footage**: Fall clips with at least 5 to 10 seconds of recorded floor dwell after impact.
- **Diverse ADL Activities**: Lying down on beds/sofas, stretching, yoga, and resting on the floor (which challenge posture classifiers).
- **Multiple Camera Perspectives & Multi-Subject Ground Truth**: Annotations containing subject IDs to test true cross-person generalization.
