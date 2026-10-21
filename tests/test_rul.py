"""Tests for the linear RUL predictor and the fault classifier."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.fault_classifier import RandomForestFaultClassifier
from app.ml.rul_predictor import (
    FAILURE_THRESHOLD,
    MAX_RUL_HOURS,
    NOMINAL_RUL_HOURS,
    LinearRULPredictor,
)
from app.models.readings import FaultType, MachineType, SensorReading


# ----------------------------------------------------------------------
# RUL predictor
# ----------------------------------------------------------------------
def test_empty_history_returns_nominal_rul(trained_rul) -> None:
    """With no history the predictor reports nominal remaining life."""
    assert trained_rul.predict_rul([]) == NOMINAL_RUL_HOURS


def test_short_history_returns_nominal_rul(trained_rul) -> None:
    """Too few points to fit a trend must not produce a bogus estimate."""
    assert trained_rul.predict_rul([95.0, 94.0, 93.0]) == NOMINAL_RUL_HOURS


def test_flat_history_returns_nominal_rul(trained_rul) -> None:
    """A machine that is not degrading has no imminent failure."""
    flat = [88.0] * 30
    assert trained_rul.predict_rul(flat) == NOMINAL_RUL_HOURS


def test_degrading_history_predicts_a_shorter_rul(trained_rul) -> None:
    """A declining health trend must produce a finite RUL below nominal."""
    declining = [100.0 - 1.5 * i for i in range(30)]
    rul = trained_rul.predict_rul(declining)
    assert 0.0 < rul < NOMINAL_RUL_HOURS


def test_steeper_decline_predicts_a_shorter_rul(trained_rul) -> None:
    """A faster degradation must yield a shorter remaining life."""
    gentle = [100.0 - 0.5 * i for i in range(30)]
    steep = [100.0 - 2.0 * i for i in range(30)]

    assert trained_rul.predict_rul(steep) < trained_rul.predict_rul(gentle)


def test_rul_never_exceeds_the_maximum(trained_rul) -> None:
    """The reported RUL must be clamped to the configured ceiling."""
    almost_flat = [90.0 - 0.001 * i for i in range(30)]
    assert trained_rul.predict_rul(almost_flat) <= MAX_RUL_HOURS


def test_rul_is_never_negative(trained_rul) -> None:
    """A machine already below the failure threshold reports zero RUL."""
    already_failed = [5.0 - 0.5 * i for i in range(30)]
    assert trained_rul.predict_rul(already_failed) >= 0.0


def test_confidence_grows_with_more_history(trained_rul) -> None:
    """Confidence must increase as more data becomes available."""
    short = trained_rul.confidence([90.0 - 1.0 * i for i in range(6)])
    long = trained_rul.confidence([90.0 - 1.0 * i for i in range(40)])
    assert long > short


def test_confidence_is_within_bounds(trained_rul) -> None:
    """Confidence must always be a fraction between 0 and 1."""
    for history in ([], [50.0], [50.0] * 40):
        confidence = trained_rul.confidence(history)
        assert 0.0 <= confidence <= 1.0


def test_train_rejects_mismatched_shapes() -> None:
    """Training with mismatched sample/label counts must raise."""
    predictor = LinearRULPredictor()
    with pytest.raises(ValueError, match="same number of rows"):
        predictor.train(np.zeros((5, 1)), np.zeros(3))


def test_failure_threshold_is_referenced_consistently() -> None:
    """The failure threshold must be a sensible health-index value."""
    assert 0.0 < FAILURE_THRESHOLD < 50.0


# ----------------------------------------------------------------------
# Fault classifier
# ----------------------------------------------------------------------
def test_classifier_predicts_the_expected_faults(trained_classifier) -> None:
    """Each synthetic fault signature must be classified correctly."""
    cases = [
        (FaultType.OVERHEATING, SensorReading(
            machine_id="motor-01", machine_type=MachineType.MOTOR,
            temperature=95.0, vibration=3.5, current=25.0, rpm=1490.0, pressure=5.2,
        )),
        (FaultType.BEARING_WEAR, SensorReading(
            machine_id="motor-01", machine_type=MachineType.MOTOR,
            temperature=70.0, vibration=7.5, current=24.0, rpm=1390.0, pressure=4.8,
        )),
        (FaultType.OVERLOAD, SensorReading(
            machine_id="motor-01", machine_type=MachineType.MOTOR,
            temperature=76.0, vibration=4.3, current=35.0, rpm=1300.0, pressure=6.5,
        )),
        (FaultType.IMBALANCE, SensorReading(
            machine_id="motor-01", machine_type=MachineType.MOTOR,
            temperature=66.0, vibration=5.9, current=26.8, rpm=1595.0, pressure=4.7,
        )),
    ]

    for expected, reading in cases:
        result = trained_classifier.predict(reading)
        assert result.fault_type is expected, (
            f"expected {expected}, got {result.fault_type}"
        )
        assert result.confidence > 0.5


def test_classifier_rejects_untrained_model() -> None:
    """Predicting before training must raise a RuntimeError."""
    classifier = RandomForestFaultClassifier()
    reading = SensorReading(
        machine_id="motor-01", machine_type=MachineType.MOTOR,
        temperature=62.0, vibration=3.2, current=22.0, rpm=1450.0, pressure=5.0,
    )
    with pytest.raises(RuntimeError, match="must be trained"):
        classifier.predict(reading)


def test_classifier_probabilities_sum_to_one(trained_classifier) -> None:
    """The probability distribution must be a valid distribution."""
    reading = SensorReading(
        machine_id="motor-01", machine_type=MachineType.MOTOR,
        temperature=62.0, vibration=3.2, current=22.0, rpm=1450.0, pressure=5.0,
    )
    result = trained_classifier.predict(reading)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    # Four fault modes plus the "Normal" class.
    assert len(result.probabilities) == 5
    assert FaultType.NORMAL.value in result.probabilities


def test_classifier_train_rejects_mismatched_shapes() -> None:
    """Training with mismatched shapes must raise a ValueError."""
    classifier = RandomForestFaultClassifier()
    with pytest.raises(ValueError, match="same number of rows"):
        classifier.train(np.zeros((4, 5)), np.array(["Normal"] * 3))
