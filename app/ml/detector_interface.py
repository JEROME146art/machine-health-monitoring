"""Strategy pattern interfaces for swappable ML components.

Defining narrow interfaces (rather than a single concrete pipeline) lets the
anomaly detector, fault classifier and RUL predictor be swapped for
alternatives - for example a rule-based detector for testing, or a gradient
boosting classifier - without touching the service layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.models.readings import ClassificationResult, SensorReading


class Detector(ABC):
    """Interface for anomaly detectors.

    A detector answers one question: "is this reading unusual?"
    """

    @abstractmethod
    def detect(self, features: np.ndarray) -> tuple[bool, float, float]:
        """Evaluate a feature vector for anomaly-ness.

        Args:
            features: A 2-D array of shape (1, n_features) matching the
                feature order the detector was trained on.

        Returns:
            A tuple of ``(is_anomaly, score, threshold)``.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, healthy_samples: np.ndarray) -> None:
        """Fit the detector on healthy operating data.

        Args:
            healthy_samples: A 2-D array of shape (n_samples, n_features).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of the features the detector expects, in order."""
        raise NotImplementedError


class Predictor(ABC):
    """Interface for predictive models (fault type, RUL, ...)."""

    @abstractmethod
    def predict(self, reading: SensorReading) -> ClassificationResult:
        """Predict the fault class for a reading.

        Args:
            reading: The sensor reading to classify.

        Returns:
            The predicted fault type with confidence scores.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, samples: np.ndarray, labels: np.ndarray) -> None:
        """Fit the predictor.

        Args:
            samples: Feature matrix of shape (n_samples, n_features).
            labels: Class labels of shape (n_samples,).
        """
        raise NotImplementedError
