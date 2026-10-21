"""Training pipeline for the ML components.

All models are trained at application start-up on *synthetic* data that is
generated on the fly.  This removes any dependency on external datasets while
still exercising the real scikit-learn estimators, so the ML story in the
project is genuine rather than mocked.

The synthetic generator produces:

* **Healthy samples** - tight clusters around nominal operating points, used
  to fit the IsolationForest.
* **Fault samples** - each of the four fault modes with characteristic
  signatures, used to fit the RandomForest classifier.
* **Degradation trajectories** - linearly declining health curves, used to
  validate the RUL regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.ml.anomaly_detector import FEATURE_NAMES
from app.ml.fault_classifier import RandomForestFaultClassifier
from app.ml.rul_predictor import LinearRULPredictor
from app.models.readings import FaultType, MachineType

#: Nominal operating point per machine type, in FEATURE_NAMES order.
NOMINAL_OPERATING_POINTS: dict[MachineType, dict[str, float]] = {
    MachineType.MOTOR: {
        "temperature": 62.0, "vibration": 3.2, "current": 22.0, "rpm": 1450.0, "pressure": 5.0,
    },
    MachineType.PUMP: {
        "temperature": 58.0, "vibration": 2.8, "current": 19.0, "rpm": 1400.0, "pressure": 8.0,
    },
    MachineType.COMPRESSOR: {
        "temperature": 70.0, "vibration": 3.8, "current": 33.0, "rpm": 2000.0, "pressure": 12.0,
    },
    MachineType.CONVEYOR: {
        "temperature": 48.0, "vibration": 2.2, "current": 14.0, "rpm": 900.0, "pressure": 3.0,
    },
}

#: Relative noise per sensor channel, indexed as FEATURE_NAMES.
SENSOR_NOISE: dict[MachineType, tuple[float, float, float, float, float]] = {
    MachineType.MOTOR: (2.5, 0.35, 1.2, 60.0, 0.4),
    MachineType.PUMP: (2.2, 0.30, 1.0, 55.0, 0.5),
    MachineType.COMPRESSOR: (3.0, 0.45, 1.8, 90.0, 0.7),
    MachineType.CONVEYOR: (1.8, 0.25, 0.8, 40.0, 0.3),
}

#: Multiplicative offsets applied to the nominal point for each fault mode.
FAULT_SIGNATURES: dict[FaultType, dict[str, float]] = {
    FaultType.OVERHEATING: {
        "temperature": 1.32, "vibration": 1.10, "current": 1.18, "rpm": 1.03, "pressure": 1.05,
    },
    FaultType.BEARING_WEAR: {
        "temperature": 1.12, "vibration": 2.05, "current": 1.12, "rpm": 0.96, "pressure": 0.97,
    },
    FaultType.OVERLOAD: {
        "temperature": 1.22, "vibration": 1.35, "current": 1.62, "rpm": 0.90, "pressure": 1.30,
    },
    FaultType.IMBALANCE: {
        "temperature": 1.08, "vibration": 1.85, "current": 1.22, "rpm": 1.10, "pressure": 0.94,
    },
}


@dataclass
class TrainingArtifacts:
    """Everything produced by a training run.

    Attributes:
        healthy_samples: Feature matrix of healthy operating data.
        healthy_by_type: Healthy samples keyed by machine type, used to fit
            one anomaly detector per machine type.
        fault_samples: Feature matrix of faulty operating data.
        fault_labels: String labels aligned with ``fault_samples``.
        degradation_history: A synthetic declining health trajectory.
        degradation_rul: Target RUL values aligned with ``degradation_history``.
    """

    healthy_samples: np.ndarray
    healthy_by_type: dict[MachineType, np.ndarray]
    fault_samples: np.ndarray
    fault_labels: np.ndarray
    degradation_history: np.ndarray
    degradation_rul: np.ndarray


class ModelTrainer:
    """Generates synthetic training data and fits the ML models.

    Attributes:
        _rng: The seeded random generator used for reproducible data.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Create the trainer.

        Args:
            seed: Random seed. Defaults to the global ``RANDOM_SEED``.
        """
        self._rng = np.random.default_rng(
            seed if seed is not None else settings.random_seed
        )

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------
    def generate_healthy(
        self, n_samples: int, machine_type: MachineType
    ) -> np.ndarray:
        """Generate healthy operating samples for one machine type.

        Args:
            n_samples: Number of samples to generate.
            machine_type: The machine type whose nominal point to use.

        Returns:
            A feature matrix of shape (n_samples, 5).
        """
        nominal = NOMINAL_OPERATING_POINTS[machine_type]
        noise = SENSOR_NOISE[machine_type]

        means = np.array([nominal[name] for name in FEATURE_NAMES], dtype=float)
        sigmas = np.array(noise, dtype=float)

        # Gaussian noise around the nominal operating point.
        samples = self._rng.normal(loc=means, scale=sigmas, size=(n_samples, 5))

        # Physical sanity floors so the data stays plausible.
        samples[:, 0] = np.clip(samples[:, 0], 20.0, 95.0)  # temperature
        samples[:, 1] = np.clip(samples[:, 1], 0.1, 12.0)  # vibration
        samples[:, 2] = np.clip(samples[:, 2], 1.0, 60.0)  # current
        samples[:, 3] = np.clip(samples[:, 3], 100.0, 2500.0)  # rpm
        samples[:, 4] = np.clip(samples[:, 4], 0.0, 25.0)  # pressure
        return samples

    def generate_faulty(
        self, n_samples: int, machine_type: MachineType
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate labelled fault samples for one machine type.

        Args:
            n_samples: Number of samples to generate (split evenly across
                the four fault modes).
            machine_type: The machine type whose nominal point to use.

        Returns:
            A tuple of ``(samples, labels)``.
        """
        nominal = NOMINAL_OPERATING_POINTS[machine_type]
        noise = SENSOR_NOISE[machine_type]
        fault_types = [
            FaultType.OVERHEATING,
            FaultType.BEARING_WEAR,
            FaultType.OVERLOAD,
            FaultType.IMBALANCE,
        ]

        per_fault = max(1, n_samples // len(fault_types))
        all_samples: list[np.ndarray] = []
        all_labels: list[str] = []

        for fault in fault_types:
            signature = FAULT_SIGNATURES[fault]
            means = np.array(
                [nominal[name] * signature[name] for name in FEATURE_NAMES], dtype=float
            )
            # Faulty data is noisier than healthy data.
            sigmas = np.array(noise, dtype=float) * 1.25
            block = self._rng.normal(loc=means, scale=sigmas, size=(per_fault, 5))
            block[:, 0] = np.clip(block[:, 0], 25.0, 130.0)
            block[:, 1] = np.clip(block[:, 1], 0.2, 18.0)
            block[:, 2] = np.clip(block[:, 2], 1.0, 90.0)
            block[:, 3] = np.clip(block[:, 3], 100.0, 2600.0)
            block[:, 4] = np.clip(block[:, 4], 0.0, 30.0)
            all_samples.append(block)
            all_labels.extend([fault.value] * per_fault)

        return np.vstack(all_samples), np.array(all_labels)

    def generate_degradation(
        self, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic degradation trajectory for RUL training.

        Args:
            n_samples: Length of the trajectory.

        Returns:
            A tuple of ``(health_history, rul_hours)``.
        """
        steps = np.arange(n_samples, dtype=float)
        # Health declines from ~100 toward the failure threshold.
        slope = -1.0 - self._rng.uniform(0.0, 0.8)
        intercept = 100.0 + self._rng.uniform(-3.0, 3.0)
        health = np.clip(intercept + slope * steps, 0.0, 100.0)
        # RUL is the number of steps left until the health hits 10.
        remaining = np.clip((health - 10.0) / max(abs(slope), 1e-6), 0.0, None)
        return health, remaining

    def build_artifacts(self) -> TrainingArtifacts:
        """Generate every dataset needed for a training run.

        The fault dataset includes healthy samples labelled ``Normal`` so that
        the classifier can express "nothing is wrong" instead of always
        selecting the nearest fault mode.

        Returns:
            A populated :class:`TrainingArtifacts` instance.
        """
        per_type = max(50, settings.train_samples // len(NOMINAL_OPERATING_POINTS))

        healthy_blocks: list[np.ndarray] = []
        healthy_by_type: dict[MachineType, np.ndarray] = {}
        fault_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []

        for machine_type in NOMINAL_OPERATING_POINTS:
            healthy = self.generate_healthy(per_type, machine_type)
            healthy_blocks.append(healthy)
            healthy_by_type[machine_type] = healthy

            samples, labels = self.generate_faulty(per_type, machine_type)
            fault_blocks.append(samples)
            label_blocks.append(labels)

            # Add the same number of healthy samples labelled "Normal" so
            # the classifier learns a genuine normal class.
            normal_block = self.generate_healthy(per_type, machine_type)
            fault_blocks.append(normal_block)
            label_blocks.append(np.array([FaultType.NORMAL.value] * per_type))

        healthy = np.vstack(healthy_blocks)
        faults = np.vstack(fault_blocks)
        fault_labels = np.concatenate(label_blocks)
        degradation, rul = self.generate_degradation(200)

        return TrainingArtifacts(
            healthy_samples=healthy,
            healthy_by_type=healthy_by_type,
            fault_samples=faults,
            fault_labels=fault_labels,
            degradation_history=degradation,
            degradation_rul=rul,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_all(
        self,
        detector,
        classifier: RandomForestFaultClassifier,
        rul: LinearRULPredictor,
    ) -> dict:
        """Fit every model on freshly generated synthetic data.

        Args:
            detector: The anomaly detector to fit.
            classifier: The fault classifier to fit.
            rul: The RUL predictor to fit.

        Returns:
            A summary dictionary describing the training run.
        """
        artifacts = self.build_artifacts()

        # One detector per machine type so each is judged against its own
        # operating envelope.
        if hasattr(detector, "train_per_type"):
            detector.train_per_type(artifacts.healthy_by_type)
        else:
            detector.train(artifacts.healthy_samples)

        classifier.train(artifacts.fault_samples, artifacts.fault_labels)
        rul.train(
            artifacts.degradation_history.reshape(-1, 1),
            artifacts.degradation_rul,
        )

        return {
            "healthy_samples": int(artifacts.healthy_samples.shape[0]),
            "detector_models": len(artifacts.healthy_by_type),
            "fault_samples": int(artifacts.fault_samples.shape[0]),
            "fault_classes": sorted(set(artifacts.fault_labels.tolist())),
            "features": list(FEATURE_NAMES),
        }