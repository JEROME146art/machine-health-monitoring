"""Health index computation combining rule-based scoring with ML signals.

Three independent signals contribute to the final 0-100 health index:

1. **Threshold score** - the machine's own ``compute_health()`` method, which
   scores the reading against that machine's operating envelope.
2. **Anomaly penalty** - a deduction when the IsolationForest flags the
   reading as unusual, scaled by how far outside the envelope it sits.
3. **Fault penalty** - a deduction proportional to the classifier's confidence
   in a non-normal fault class.

The rule-based score is deliberately the dominant term: it is stable and
explainable, while the ML terms add sensitivity to subtle degradation.
"""

from __future__ import annotations

from app.ml.anomaly_detector import to_features
from app.ml.detector_interface import Detector
from app.ml.fault_classifier import RandomForestFaultClassifier
from app.ml.rul_predictor import LinearRULPredictor
from app.models.readings import (
    AnomalyResult,
    ClassificationResult,
    FaultType,
    HealthPrediction,
    SensorReading,
)

#: Maximum health deduction applied by the anomaly detector.
MAX_ANOMALY_PENALTY: float = 35.0

#: Maximum health deduction applied by the fault classifier.
MAX_FAULT_PENALTY: float = 25.0

#: Weight of the rule-based threshold score in the final blend.
RULE_WEIGHT: float = 0.70

#: Weight of the ML-derived component in the final blend.
ML_WEIGHT: float = 0.30


class HealthIndexCalculator:
    """Combines the machine rule engine and the ML models into one score.

    Attributes:
        _detector: The anomaly detector (Strategy - swappable).
        _classifier: The fault classifier to use.
        _rul: The RUL predictor to use.
    """

    def __init__(
        self,
        detector: Detector,
        classifier: RandomForestFaultClassifier,
        rul: LinearRULPredictor,
    ) -> None:
        """Create the calculator.

        Args:
            detector: The anomaly detector to use.
            classifier: The fault classifier to use.
            rul: The RUL predictor to use.
        """
        self._detector = detector
        self._classifier = classifier
        self._rul = rul

    def compute(
        self,
        machine,
        reading: SensorReading,
        health_history: list[float] | None = None,
    ) -> HealthPrediction:
        """Compute the full health prediction for a reading.

        Args:
            machine: The :class:`~app.machine.Machine` the reading belongs to.
            reading: The sensor reading to assess.
            health_history: Recent health indices (oldest first) used for RUL.

        Returns:
            A :class:`HealthPrediction` with the blended index, status,
            sub-predictions, RUL and a human readable cause.
        """
        history = list(health_history or [])

        # 1. Rule-based score from the machine's own thresholds.
        rule_score = float(machine.compute_health(reading))

        # 2. Anomaly detection, judged against this machine type's own
        #    learned healthy envelope.
        features = to_features(reading)
        if hasattr(self._detector, "detect_for"):
            is_anomaly, score, threshold = self._detector.detect_for(
                reading, machine.type
            )
        else:
            is_anomaly, score, threshold = self._detector.detect(features)
        anomaly_result = AnomalyResult(
            is_anomaly=is_anomaly, score=score, threshold=threshold
        )

        # 3. Fault classification.
        classification: ClassificationResult = self._classifier.predict(reading)

        # 4. Blend the rule score with the ML signals.
        ml_score = 100.0
        if is_anomaly:
            # Normalise the exceedance past the threshold into 0-1.
            span = max(abs(threshold), 1e-6)
            excess = min(abs(score - threshold) / span, 3.0) / 3.0
            ml_score -= MAX_ANOMALY_PENALTY * excess

        if classification.fault_type is not FaultType.NORMAL:
            ml_score -= MAX_FAULT_PENALTY * classification.confidence

        ml_score = max(0.0, min(100.0, ml_score))
        blended = RULE_WEIGHT * rule_score + ML_WEIGHT * ml_score
        health_index = round(max(0.0, min(100.0, blended)), 1)

        # 5. Remaining useful life and its confidence.
        rul_hours = self._rul.predict_rul(history)
        rul_confidence = self._rul.confidence(history)

        # 6. Status band.
        status = machine.status_for(health_index)

        # 7. Human readable cause of the current condition.
        cause = self._explain(reading, machine, anomaly_result, classification)

        return HealthPrediction(
            health_index=health_index,
            status=status.value,
            anomaly_detection=anomaly_result,
            fault_classification=classification,
            remaining_useful_life=rul_hours,
            confidence=rul_confidence,
            cause=cause,
        )

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------
    def _explain(
        self,
        reading: SensorReading,
        machine,
        anomaly: AnomalyResult,
        classification: ClassificationResult,
    ) -> str:
        """Build a short human readable explanation of the machine's state.

        The wording follows the style requested for the UI, e.g.
        ``"Vibration 3.2σ above normal"``.

        Args:
            reading: The assessed reading.
            machine: The machine the reading belongs to.
            anomaly: The anomaly detection result.
            classification: The fault classification result.

        Returns:
            A one-line explanation.
        """
        thresholds = machine.thresholds()
        baseline = machine.readings

        # Compare against the machine's own recent operating average.
        comparisons = [
            ("Temperature", reading.temperature, "temperature"),
            ("Vibration", reading.vibration, "vibration"),
            ("Current", reading.current, "current"),
            ("Pressure", reading.pressure, "pressure"),
        ]

        best_label = ""
        best_ratio = 0.0
        for label, value, attr in comparisons:
            # The history includes the current reading, so exclude the last
            # entry to obtain a genuine baseline of previous readings.
            history = [getattr(r, attr) for r in baseline[:-1]] if len(baseline) > 1 else []
            if len(history) < 3:
                continue
            mean = sum(history) / len(history)
            if mean <= 0:
                continue
            std = (sum((v - mean) ** 2 for v in history) / len(history)) ** 0.5
            # Fall back to a fraction of the mean when the channel is flat.
            sigma = std if std > 1e-6 else mean * 0.05
            ratio = (value - mean) / sigma
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = label

        if best_ratio >= 2.0:
            return f"{best_label} {best_ratio:.1f}σ above normal"

        if classification.fault_type is not FaultType.NORMAL:
            return (
                f"{classification.fault_type.value} detected "
                f"(confidence {classification.confidence:.0%})"
            )

        if anomaly.is_anomaly:
            return "Anomalous reading outside the learned healthy envelope"

        return "All parameters within normal operating range"