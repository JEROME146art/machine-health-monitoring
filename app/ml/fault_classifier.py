"""RandomForestClassifier-based fault type classification.

The classifier maps a five-dimensional sensor vector onto one of five fault
classes.  A RandomForest is used because it needs no feature scaling, handles
the non-linear interaction between sensors (e.g. high current *and* high
vibration) well, and yields calibrated class probabilities that we surface as
the model's confidence in the UI.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.ml.anomaly_detector import FEATURE_NAMES, to_features
from app.ml.detector_interface import Predictor
from app.models.readings import ClassificationResult, FaultType, SensorReading


class RandomForestFaultClassifier(Predictor):
    """Fault type classifier backed by scikit-learn's RandomForest.

    Attributes:
        _model: The underlying RandomForestClassifier estimator.
        _classes: The label order the estimator was fitted with.
        _trained: Whether :meth:`train` has been called.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        """Create the classifier.

        Args:
            n_estimators: Number of trees in the forest.
            random_state: Seed for reproducibility.
        """
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._classes: list[str] = []
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Predictor interface
    # ------------------------------------------------------------------
    def train(self, samples: np.ndarray, labels: np.ndarray) -> None:
        """Fit the classifier on synthetic labelled fault data.

        Args:
            samples: Feature matrix of shape (n_samples, n_features).
            labels: String labels of shape (n_samples,).

        Raises:
            ValueError: If the arrays are empty or have mismatched shapes.
        """
        if samples is None or labels is None or len(samples) == 0:
            raise ValueError("samples and labels must be non-empty")
        if samples.shape[0] != labels.shape[0]:
            raise ValueError(
                f"samples ({samples.shape[0]}) and labels ({labels.shape[0]}) "
                "must contain the same number of rows"
            )

        self._model.fit(samples, labels)
        self._classes = [str(c) for c in self._model.classes_]
        self._trained = True

    def predict(self, reading: SensorReading) -> ClassificationResult:
        """Predict the fault type for a sensor reading.

        Args:
            reading: The reading to classify.

        Returns:
            The winning fault class together with the full probability
            distribution and the confidence of the winning class.

        Raises:
            RuntimeError: If the classifier has not been trained.
        """
        if not self._trained:
            raise RuntimeError("RandomForestFaultClassifier must be trained before use")

        features = to_features(reading)
        proba = self._model.predict_proba(features)[0]
        probabilities = {
            cls: round(float(p), 4) for cls, p in zip(self._classes, proba)
        }
        best_index = int(np.argmax(proba))
        best_class = self._classes[best_index]
        confidence = round(float(proba[best_index]), 4)

        return ClassificationResult(
            fault_type=FaultType(best_class),
            confidence=confidence,
            probabilities=probabilities,
        )

    @property
    def classes(self) -> list[str]:
        """Class labels the classifier was fitted with."""
        return list(self._classes)

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return the relative importance of each feature in the forest."""
        if not self._trained:
            return {name: 0.2 for name in FEATURE_NAMES}
        importances = self._model.feature_importances_
        return {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, importances)
        }

    @property
    def feature_names(self) -> list[str]:
        """Names of the features the classifier expects, in order."""
        return list(FEATURE_NAMES)

    @property
    def is_trained(self) -> bool:
        """Whether the classifier has been fitted."""
        return self._trained
