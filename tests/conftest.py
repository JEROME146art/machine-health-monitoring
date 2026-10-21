"""Shared pytest fixtures for the test-suite."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.anomaly_detector import IsolationForestDetector
from app.ml.fault_classifier import RandomForestFaultClassifier
from app.ml.rul_predictor import LinearRULPredictor
from app.ml.trainer import ModelTrainer
from app.models.readings import MachineType


@pytest.fixture(scope="session")
def trainer() -> ModelTrainer:
    """A seeded model trainer, shared across the session."""
    return ModelTrainer(seed=42)


@pytest.fixture(scope="session")
def artifacts(trainer: ModelTrainer):
    """The synthetic training datasets."""
    return trainer.build_artifacts()


@pytest.fixture(scope="session")
def trained_detector(trainer: ModelTrainer, artifacts) -> IsolationForestDetector:
    """A detector fitted on synthetic healthy data, one forest per machine type.

    This mirrors how the application trains the detector at start-up.
    """
    detector = IsolationForestDetector()
    detector.train_per_type(artifacts.healthy_by_type)
    return detector


@pytest.fixture(scope="session")
def trained_classifier(trainer: ModelTrainer, artifacts) -> RandomForestFaultClassifier:
    """A fault classifier fitted on synthetic labelled data."""
    classifier = RandomForestFaultClassifier(random_state=42)
    classifier.train(artifacts.fault_samples, artifacts.fault_labels)
    return classifier


@pytest.fixture(scope="session")
def trained_rul() -> LinearRULPredictor:
    """An initialised RUL predictor."""
    predictor = LinearRULPredictor()
    predictor.train(np.zeros((10, 1)), np.zeros(10))
    return predictor


@pytest.fixture
def healthy_sample() -> np.ndarray:
    """A single healthy feature vector for a Motor."""
    return np.array([[62.0, 3.2, 22.0, 1450.0, 5.0]], dtype=float)


@pytest.fixture
def motor_type() -> MachineType:
    """The Motor machine type."""
    return MachineType.MOTOR
