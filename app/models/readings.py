"""Pydantic models for sensor readings and health calculation results.

All data flowing through the API is validated against these schemas before
entering the domain layer.  The dataclasses defined later extend the models
with private attributes and business logic while keeping the public contract
unchanged.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FaultType(str, Enum):
    """Known fault classifications emitted by the ML classifier."""

    NORMAL = "Normal"
    OVERHEATING = "Overheating"
    BEARING_WEAR = "Bearing Wear"
    OVERLOAD = "Overload"
    IMBALANCE = "Imbalance"


class MachineType(str, Enum):
    """Supported machine categories."""

    MOTOR = "Motor"
    PUMP = "Pump"
    COMPRESSOR = "Compressor"
    CONVEYOR = "Conveyor"


class SensorReading(BaseModel):
    """In-bound sensor reading from an IoT device or the simulator.

    All numeric fields accept negative values because some sensors may
    legitimately report sub-zero measurements (e.g. temperature in cold
    environments).  Validation constraints are intentionally permissive
    to avoid rejecting valid industrial data.
    """

    machine_id: str = Field(..., min_length=1, max_length=50)
    machine_type: MachineType
    temperature: float = Field(..., ge=-50, le=200)  # Celsius
    vibration: float = Field(..., ge=0, le=20)  # mm/s
    current: float = Field(..., ge=0, le=500)  # Amperes
    rpm: float = Field(..., ge=0, le=10000)  # Revolutions per minute
    pressure: float = Field(..., ge=0, le=500)  # Bar
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("machine_id")
    @classmethod
    def normalize_machine_id(cls, v: str) -> str:
        """Store machine IDs in lowercase for consistent lookups."""
        return v.strip().lower()


class AnomalyResult(BaseModel):
    """Result of the IsolationForest anomaly detection.

    Attributes:
        is_anomaly: True if the reading lies outside the healthy envelope.
        score: Raw IsolationForest anomaly score (higher = more anomalous).
        threshold: The boundary used to classify normal vs anomalous.
    """

    is_anomaly: bool
    score: float
    threshold: float


class ClassificationResult(BaseModel):
    """Result of the RandomForestClassifier fault type prediction.

    Attributes:
        fault_type: The predicted fault class.
        confidence: Softmax probability for the winning class.
        probabilities: Per-class probability distribution.
    """

    fault_type: FaultType
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]


class HealthPrediction(BaseModel):
    """Aggregated health assessment for a single reading.

    Combines raw sensor data with ML predictions to produce a single,
    easy-to-consume health metric for UI rendering.
    """

    health_index: float = Field(..., ge=0, le=100)
    status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    anomaly_detection: AnomalyResult
    fault_classification: ClassificationResult
    remaining_useful_life: float = Field(..., ge=0)  # Hours
    confidence: float = Field(..., ge=0, le=1)
    cause: str


class AlertSeverity(str, Enum):
    """Alert priority levels shown in the UI."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert(BaseModel):
    """An alert triggered by an anomaly or degradation event."""

    id: str
    machine_id: str
    severity: AlertSeverity
    timestamp: datetime
    cause: str
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None


class MachineSummary(BaseModel):
    """Lightweight representation returned by the machine list endpoint."""

    id: str
    name: str
    type: str
    health_index: float = Field(..., ge=0, le=100)
    status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    last_update: datetime | None = None


class MachineDetail(MachineSummary):
    """Detailed view including last reading sensor values and ML predictions."""

    temperature: float
    vibration: float
    current: float
    rpm: float
    pressure: float
    readings_recent: int = Field(..., ge=0)
    fault_type: str = "Normal"
    fault_confidence: float = Field(default=0.0, ge=0, le=1)
    remaining_useful_life: float = Field(default=0.0, ge=0)
    confidence: float = Field(default=0.0, ge=0, le=1)
    cause: str = "Awaiting first reading"


class HealthStatus(str, Enum):
    """Three-tier health state used throughout the application."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"