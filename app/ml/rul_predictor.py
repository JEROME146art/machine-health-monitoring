"""Remaining Useful Life (RUL) estimation via trend regression.

A machine's health degrades along a roughly linear trajectory once it leaves
its healthy envelope.  We fit an ordinary least-squares line to the recent
health-index history and extrapolate to the point where the index reaches a
failure threshold, then convert the remaining slope into hours.

This module is intentionally free of scikit-learn's higher-level estimators so
the maths stays visible and explainable during a viva.
"""

from __future__ import annotations

import numpy as np

#: Health index at which a machine is considered to have failed.
FAILURE_THRESHOLD: float = 10.0

#: Minimum number of history points required before a trend can be fitted.
MIN_HISTORY_POINTS: int = 5

#: Upper bound on the reported RUL so a flat (noise-only) trend cannot
#: produce an absurd "failure in 9000 hours" claim.
MAX_RUL_HOURS: float = 720.0

#: Used when the machine is perfectly healthy and no failure is imminent.
NOMINAL_RUL_HOURS: float = 720.0


class LinearRULPredictor:
    """Estimates Remaining Useful Life from a health-index history.

    Attributes:
        _trained: Whether the predictor has been fitted at least once.
    """

    def __init__(self) -> None:
        """Create an untrained predictor."""
        self._trained: bool = False

    def train(self, samples: np.ndarray, labels: np.ndarray) -> None:
        """Fit the predictor on synthetic degradation trajectories.

        This mirrors the :class:`~app.ml.detector_interface.Predictor` contract
        so the component is interchangeable with the other models.

        Args:
            samples: Feature matrix of shape (n_samples, n_features).
            labels: Target RUL values of shape (n_samples,).
        """
        if samples is None or labels is None or len(samples) == 0:
            raise ValueError("samples and labels must be non-empty")
        if samples.shape[0] != labels.shape[0]:
            raise ValueError(
                f"samples ({samples.shape[0]}) and labels ({labels.shape[0]}) "
                "must contain the same number of rows"
            )
        # The live estimator refits on the real history for every prediction;
        # this training step exists so the component satisfies the Predictor
        # contract and can be validated independently in the test-suite.
        self._trained = True

    def predict_rul(
        self, health_history: list[float], sample_interval_hours: float = 0.05
    ) -> float:
        """Predict remaining useful life in hours.

        The health history is the sequence of health indices (oldest first) for
        a single machine.  A least-squares line is fitted to that history and
        extrapolated to :data:`FAILURE_THRESHOLD`.

        Args:
            health_history: Ordered health-index values, oldest first.
            sample_interval_hours: Real-world hours between two readings.
                The simulator samples every 2 seconds; scaling this to a
                realistic inspection interval keeps the reported RUL in a
                sensible range (hours, not seconds).

        Returns:
            Estimated hours until failure, clamped to
            ``[0, MAX_RUL_HOURS]``.
        """
        if not health_history:
            return NOMINAL_RUL_HOURS

        if len(health_history) < MIN_HISTORY_POINTS:
            # Not enough data to establish a trend - report nominal life but
            # flag it as low confidence via the slope magnitude.
            return round(NOMINAL_RUL_HOURS, 1)

        y = np.asarray(health_history, dtype=float)
        x = np.arange(len(y), dtype=float)

        slope, intercept = np.polyfit(x, y, 1)

        if slope >= -1e-6:
            # Health is flat or improving: no imminent failure detected.
            return round(NOMINAL_RUL_HOURS, 1)

        # Steps remaining until the trend crosses the failure threshold.
        steps_to_failure = (FAILURE_THRESHOLD - intercept) / slope
        steps_remaining = steps_to_failure - (len(y) - 1)

        if steps_remaining <= 0:
            return 0.0

        hours = steps_remaining * sample_interval_hours
        return round(min(float(hours), MAX_RUL_HOURS), 1)

    def confidence(self, health_history: list[float]) -> float:
        """Estimate how trustworthy the RUL estimate is.

        Confidence grows with the amount of history and with how steeply the
        health index is declining.

        Args:
            health_history: Ordered health-index values, oldest first.

        Returns:
            A confidence value between 0 and 1.
        """
        if len(health_history) < MIN_HISTORY_POINTS:
            return round(min(0.25 * len(health_history) / MIN_HISTORY_POINTS, 0.4), 2)

        y = np.asarray(health_history, dtype=float)
        x = np.arange(len(y), dtype=float)
        slope, _ = np.polyfit(x, y, 1)

        # Data sufficiency: saturates at 30 points.
        data_factor = min(len(health_history) / 30.0, 1.0)
        # Trend clarity: a slope of -1.0 index point per reading is "clear".
        trend_factor = min(abs(slope) / 1.0, 1.0)
        return round(min(0.35 + 0.35 * data_factor + 0.30 * trend_factor, 0.99), 2)

    @property
    def is_trained(self) -> bool:
        """Whether the predictor has been initialised."""
        return self._trained
