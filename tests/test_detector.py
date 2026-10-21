"""Tests for the IsolationForest anomaly detector.

Covers the Strategy interface contract, the healthy/faulty separation, and
the training-validation guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.anomaly_detector import FEATURE_NAMES, IsolationForestDetector
from app.ml.detector_interface import Detector


def test_detector_implements_the_detector_interface(trained_detector) -> None:
    """The IsolationForest detector must satisfy the Detector ABC."""
    assert isinstance(trained_detector, Detector)


def test_feature_names_are_in_canonical_order(trained_detector) -> None:
    """The detector must expose the shared five-feature ordering."""
    assert trained_detector.feature_names == FEATURE_NAMES
    assert trained_detector.feature_names == [
        "temperature",
        "vibration",
        "current",
        "rpm",
        "pressure",
    ]


def test_detect_rejects_untrained_model() -> None:
    """Calling detect() before train() must raise a RuntimeError."""
    detector = IsolationForestDetector()
    with pytest.raises(RuntimeError, match="must be trained"):
        detector.detect(np.array([[60.0, 3.0, 20.0, 1400.0, 5.0]]))


def test_train_rejects_empty_samples() -> None:
    """Training on an empty matrix must raise a ValueError."""
    detector = IsolationForestDetector()
    with pytest.raises(ValueError, match="at least one row"):
        detector.train(np.array([]))


def test_train_rejects_non_2d_samples() -> None:
    """Training on a 1-D array must raise a ValueError."""
    detector = IsolationForestDetector()
    with pytest.raises(ValueError, match="2-D array"):
        detector.train(np.array([1.0, 2.0, 3.0]))


def test_healthy_reading_is_not_flagged(trained_detector, healthy_sample) -> None:
    """A nominal operating point must not be reported as an anomaly."""
    is_anomaly, score, threshold = trained_detector.detect(healthy_sample)
    assert is_anomaly is False
    assert score <= threshold


def test_severe_overheating_is_flagged(trained_detector) -> None:
    """A grossly overheating reading must be flagged as anomalous."""
    overheating = np.array([[160.0, 3.2, 22.0, 1450.0, 5.0]], dtype=float)
    is_anomaly, score, threshold = trained_detector.detect(overheating)
    assert is_anomaly is True
    assert score > threshold


def test_severe_bearing_wear_is_flagged(trained_detector) -> None:
    """Extreme vibration must be flagged as anomalous."""
    bearing_wear = np.array([[62.0, 18.0, 22.0, 1450.0, 5.0]], dtype=float)
    is_anomaly, _, _ = trained_detector.detect(bearing_wear)
    assert is_anomaly is True


def test_anomaly_score_is_higher_for_worse_readings(
    trained_detector, healthy_sample
) -> None:
    """The anomaly score must increase as the reading gets worse."""
    _, healthy_score, _ = trained_detector.detect(healthy_sample)

    moderate = np.array([[110.0, 6.0, 30.0, 1400.0, 5.0]], dtype=float)
    _, moderate_score, _ = trained_detector.detect(moderate)

    severe = np.array([[180.0, 12.0, 45.0, 1400.0, 5.0]], dtype=float)
    _, severe_score, _ = trained_detector.detect(severe)

    assert healthy_score < moderate_score < severe_score


def test_detect_accepts_a_flat_feature_vector(trained_detector) -> None:
    """A 1-D feature vector must be reshaped rather than rejected."""
    vector = np.array([62.0, 3.2, 22.0, 1450.0, 5.0], dtype=float)
    is_anomaly, score, threshold = trained_detector.detect(vector)
    assert isinstance(is_anomaly, bool)
    assert isinstance(score, float)
    assert isinstance(threshold, float)


def test_detector_is_deterministic(trained_detector, healthy_sample) -> None:
    """Repeated calls must return identical results."""
    first = trained_detector.detect(healthy_sample)
    second = trained_detector.detect(healthy_sample)
    assert first == second
