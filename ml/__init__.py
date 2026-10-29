"""
ARJUN Machine Learning Module (Phase 4A: Feature Engineering & Dataset Preparation).

Provides offline feature extraction and tabular dataset construction from
YOLOv8-pose keypoint tracks for fall detection and activity classification.
"""

from ml.features import (
    FeatureExtractor,
    InstantFeatures,
    TemporalFeatures,
    compute_joint_angle,
    extract_features_for_sequence,
)
from ml.dataset import (
    MLDatasetBuilder,
    DatasetSample,
    build_urfd_ml_dataset,
    summarize_dataset,
)

__all__ = [
    "FeatureExtractor",
    "InstantFeatures",
    "TemporalFeatures",
    "compute_joint_angle",
    "extract_features_for_sequence",
    "MLDatasetBuilder",
    "DatasetSample",
    "build_urfd_ml_dataset",
    "summarize_dataset",
]
