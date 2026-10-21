"""Machine service - the orchestration layer of the application.

The service owns the fleet of machines, the ML models and the alert service.
It is the only place where a reading enters the system, which keeps the
business rules (when to raise an alert, how to compute history) in one spot.

Layering rule: ``api -> services -> ml / models / core``.  The API layer never
touches the ML models directly.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque

from app.core.config import settings
from app.core.exceptions import MachineNotFoundError, SimulationStateError
from app.machine import Machine
from app.machine_factory import create_default_fleet
from app.ml.anomaly_detector import IsolationForestDetector
from app.ml.fault_classifier import RandomForestFaultClassifier
from app.ml.health_index import HealthIndexCalculator
from app.ml.rul_predictor import LinearRULPredictor
from app.ml.trainer import ModelTrainer
from app.models.readings import (
    AlertSeverity,
    FaultType,
    HealthPrediction,
    MachineDetail,
    MachineSummary,
    SensorReading,
)
from app.services.alert_service import AlertService
from app.services.sensor_simulator import SensorSimulator

logger = logging.getLogger(__name__)

#: Number of health samples kept per machine for the RUL trend fit.
HEALTH_HISTORY_SIZE: int = 60

#: Consecutive CRITICAL predictions required before a CRITICAL alert fires.
#: A single bad reading should not page an operator.
CRITICAL_STREAK_THRESHOLD: int = 2


class MachineService:
    """Coordinates machines, ML models, alerts and the simulator.

    Attributes:
        _machines: The fleet keyed by machine id.
        _detector: The anomaly detector.
        _classifier: The fault classifier.
        _rul: The RUL predictor.
        _calculator: Combines the models into a health prediction.
        _alerts: The alert service (subject of the observer pattern).
        _simulator: The background sensor simulator.
        _health_history: Per-machine rolling health-index history.
        _critical_streak: Per-machine count of consecutive CRITICAL states.
        _training_summary: Metadata about the last training run.
    """

    def __init__(self) -> None:
        """Create the service and train the ML models."""
        self._machines: dict[str, Machine] = {
            machine.id: machine for machine in create_default_fleet(
                buffer_size=settings.readings_buffer_size
            )
        }
        self._health_history: dict[str, Deque[float]] = {
            machine_id: deque(maxlen=HEALTH_HISTORY_SIZE)
            for machine_id in self._machines
        }
        self._critical_streak: dict[str, int] = {mid: 0 for mid in self._machines}

        # ML components (Strategy pattern - each is swappable).
        self._detector = IsolationForestDetector()
        self._classifier = RandomForestFaultClassifier(
            random_state=settings.random_seed
        )
        self._rul = LinearRULPredictor()
        self._calculator = HealthIndexCalculator(
            detector=self._detector, classifier=self._classifier, rul=self._rul
        )

        self._alerts = AlertService()
        self._simulator: SensorSimulator | None = None
        self._training_summary: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_models(self) -> dict:
        """Train every ML model on freshly generated synthetic data.

        Returns:
            A summary of the training run.
        """
        trainer = ModelTrainer(seed=settings.random_seed)
        self._training_summary = trainer.train_all(
            detector=self._detector,
            classifier=self._classifier,
            rul=self._rul,
        )
        logger.info("ML models trained: %s", self._training_summary)
        return self._training_summary

    @property
    def training_summary(self) -> dict:
        """Metadata about the last training run."""
        return dict(self._training_summary)

    # ------------------------------------------------------------------
    # Simulator lifecycle
    # ------------------------------------------------------------------
    async def start_simulation(self) -> None:
        """Start the background sensor simulator."""
        if self._simulator is None:
            self._simulator = SensorSimulator(
                machines=list(self._machines.values()),
                callback=self.process_reading,
            )
        await self._simulator.start()

    async def stop_simulation(self) -> None:
        """Stop the background sensor simulator."""
        if self._simulator is not None:
            await self._simulator.stop()

    def pause_simulation(self) -> bool:
        """Pause the simulator.

        Returns:
            ``True`` if the simulator was running and is now paused.

        Raises:
            SimulationStateError: If the simulator is not running.
        """
        if self._simulator is None or not self._simulator.running:
            raise SimulationStateError("The simulation is not currently running.")
        self._simulator.pause()
        return True

    def resume_simulation(self) -> bool:
        """Resume a paused simulator.

        Returns:
            ``True`` if the simulator was paused and is now running.

        Raises:
            SimulationStateError: If the simulator is already running.
        """
        if self._simulator is None:
            raise SimulationStateError("The simulator has not been started.")
        if self._simulator.running:
            raise SimulationStateError("The simulation is already running.")
        self._simulator.resume()
        return True

    @property
    def simulation_running(self) -> bool:
        """Whether the simulator is currently emitting readings."""
        return self._simulator is not None and self._simulator.running

    # ------------------------------------------------------------------
    # Reading intake
    # ------------------------------------------------------------------
    def process_reading(self, reading: SensorReading) -> HealthPrediction:
        """Process one sensor reading end to end.

        Stores the reading, computes the health prediction, updates the health
        history and raises alerts when the state warrants it.

        Args:
            reading: The validated reading to process.

        Returns:
            The resulting :class:`HealthPrediction`.

        Raises:
            MachineNotFoundError: If the reading references an unknown machine.
        """
        machine = self.get_machine(reading.machine_id)
        machine.add_reading(reading)

        history = list(self._health_history[machine.id])
        prediction = self._calculator.compute(machine, reading, history)

        self._health_history[machine.id].append(prediction.health_index)
        machine.set_prediction(prediction.model_dump(mode="json"))

        self._maybe_alert(machine, prediction)

        return prediction

    def _maybe_alert(self, machine: Machine, prediction: HealthPrediction) -> None:
        """Raise an alert when a machine's condition warrants attention.

        Args:
            machine: The machine the prediction belongs to.
            prediction: The health prediction to evaluate.
        """
        if prediction.status == "CRITICAL":
            self._critical_streak[machine.id] = self._critical_streak.get(machine.id, 0) + 1
        else:
            self._critical_streak[machine.id] = 0

        if prediction.fault_classification.fault_type is not FaultType.NORMAL:
            severity = (
                AlertSeverity.CRITICAL
                if prediction.status == "CRITICAL"
                else AlertSeverity.WARNING
            )
            self._alerts.publish(
                machine_id=machine.id,
                severity=severity,
                cause=prediction.cause,
            )
        elif (
            prediction.status == "CRITICAL"
            and self._critical_streak[machine.id] >= CRITICAL_STREAK_THRESHOLD
        ):
            self._alerts.publish(
                machine_id=machine.id,
                severity=AlertSeverity.CRITICAL,
                cause=prediction.cause,
            )
        elif prediction.status == "WARNING":
            self._alerts.publish(
                machine_id=machine.id,
                severity=AlertSeverity.WARNING,
                cause=prediction.cause,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_machine(self, machine_id: str) -> Machine:
        """Look up a machine by id.

        Args:
            machine_id: The machine identifier.

        Returns:
            The machine instance.

        Raises:
            MachineNotFoundError: If no such machine exists.
        """
        machine = self._machines.get(machine_id.strip().lower())
        if machine is None:
            raise MachineNotFoundError(
                f"Machine '{machine_id}' does not exist.", machine_id=machine_id
            )
        return machine

    def list_machines(self) -> list[MachineSummary]:
        """Return a summary of every machine in the fleet.

        Returns:
            A list of :class:`MachineSummary` objects.
        """
        summaries: list[MachineSummary] = []
        for machine in self._machines.values():
            prediction = machine.last_prediction
            summaries.append(
                MachineSummary(
                    id=machine.id,
                    name=machine.name,
                    type=machine.type.value,
                    health_index=float(
                        prediction["health_index"] if prediction else 100.0
                    ),
                    status=(prediction["status"] if prediction else "HEALTHY"),
                    last_update=(
                        machine.last_reading.timestamp if machine.last_reading else None
                    ),
                )
            )
        return summaries

    def get_machine_detail(self, machine_id: str) -> MachineDetail:
        """Return the detailed view of one machine.

        Args:
            machine_id: The machine identifier.

        Returns:
            A populated :class:`MachineDetail`.

        Raises:
            MachineNotFoundError: If no such machine exists.
        """
        machine = self.get_machine(machine_id)
        prediction = machine.last_prediction
        last = machine.last_reading

        fault_type = "Normal"
        fault_confidence = 0.0
        rul_hours = 0.0
        confidence = 0.0
        cause = "Awaiting first reading"

        if prediction:
            classification = prediction.get("fault_classification", {})
            fault_type = classification.get("fault_type", "Normal")
            fault_confidence = float(classification.get("confidence", 0.0))
            rul_hours = float(prediction.get("remaining_useful_life", 0.0))
            confidence = float(prediction.get("confidence", 0.0))
            cause = prediction.get("cause", cause)

        return MachineDetail(
            id=machine.id,
            name=machine.name,
            type=machine.type.value,
            health_index=float(prediction["health_index"] if prediction else 100.0),
            status=(prediction["status"] if prediction else "HEALTHY"),
            last_update=(last.timestamp if last else None),
            temperature=float(last.temperature if last else 0.0),
            vibration=float(last.vibration if last else 0.0),
            current=float(last.current if last else 0.0),
            rpm=float(last.rpm if last else 0.0),
            pressure=float(last.pressure if last else 0.0),
            readings_recent=machine.reading_count,
            fault_type=fault_type,
            fault_confidence=round(fault_confidence, 4),
            remaining_useful_life=rul_hours,
            confidence=round(confidence, 4),
            cause=cause,
        )

    def get_readings(self, machine_id: str, limit: int = 100) -> list[SensorReading]:
        """Return the most recent readings for a machine.

        Args:
            machine_id: The machine identifier.
            limit: Maximum number of readings to return (newest last).

        Returns:
            The requested readings.

        Raises:
            MachineNotFoundError: If no such machine exists.
        """
        machine = self.get_machine(machine_id)
        readings = machine.readings
        if limit <= 0:
            return []
        return readings[-limit:]

    def health_history(self, machine_id: str) -> list[float]:
        """Return the health-index history for a machine.

        Args:
            machine_id: The machine identifier.

        Returns:
            The health history, oldest first.
        """
        return list(self._health_history.get(machine_id, ()))

    # ------------------------------------------------------------------
    # Demo controls
    # ------------------------------------------------------------------
    def inject_fault(self, machine_id: str, fault_type: str = "random") -> str:
        """Inject a sudden fault into a machine.

        Args:
            machine_id: The target machine.
            fault_type: The kind of fault to inject.

        Returns:
            The name of the injected fault.

        Raises:
            MachineNotFoundError: If the machine does not exist.
            SimulationStateError: If the simulator is not running.
            ValueError: If the fault type is unknown.
        """
        self.get_machine(machine_id)
        if self._simulator is None:
            raise SimulationStateError("The simulator has not been started.")
        return self._simulator.inject_fault(machine_id, fault_type)

    def clear_fault(self, machine_id: str) -> None:
        """Clear active fault on a machine.

        Args:
            machine_id: The target machine.
        """
        self.get_machine(machine_id)
        if self._simulator is not None:
            self._simulator.clear_fault(machine_id)

    def reset_fleet(self) -> None:
        """Reset all machines, clear faults, clear streaks, and reset health history."""
        if self._simulator is not None:
            self._simulator.reset_all()
        for mid in self._machines:
            self._critical_streak[mid] = 0
            self._health_history[mid].clear()

    def get_ml_metrics(self) -> dict:
        """Return ML performance and architecture metrics."""
        return {
            "training_summary": self.training_summary,
            "feature_importances": self._classifier.feature_importances,
            "fault_classes": self._classifier.classes,
            "features": self._classifier.feature_names,
            "models": {
                "anomaly_detector": {
                    "name": "Isolation Forest",
                    "type": "Unsupervised Outlier Detection",
                    "n_estimators": 100,
                    "contamination": 0.05,
                    "description": "Learns multivariate healthy baseline cluster per machine type to detect zero-day sensor deviations.",
                },
                "fault_classifier": {
                    "name": "Random Forest",
                    "type": "Supervised Multiclass Ensemble",
                    "n_estimators": 200,
                    "max_depth": 12,
                    "description": "Maps multi-axis sensor vectors to precise fault modes with calibrated posterior probabilities.",
                },
                "rul_predictor": {
                    "name": "Linear Health Trend Regression",
                    "type": "Rolling Trajectory Extrapolation",
                    "failure_threshold": 20.0,
                    "description": "Calculates health index degradation velocity to project time-to-catastrophic failure window.",
                },
            },
        }

    # ------------------------------------------------------------------
    # Alert access
    # ------------------------------------------------------------------
    @property
    def alerts(self) -> AlertService:
        """The alert service."""
        return self._alerts

    def export_rows(self) -> list[SensorReading]:
        """Return every buffered reading from every machine.

        Returns:
            A flat list of all readings, ordered by timestamp.
        """
        rows: list[SensorReading] = []
        for machine in self._machines.values():
            rows.extend(machine.readings)
        rows.sort(key=lambda r: r.timestamp)
        return rows
