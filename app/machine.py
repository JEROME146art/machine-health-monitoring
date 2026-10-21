"""Abstract Machine base class and concrete machine implementations.

This module is the heart of the domain model and the clearest demonstration of
the four pillars of object-oriented programming:

* **Abstraction**   - :class:`Machine` is an abstract base class (ABC) that
  defines the contract every machine must fulfil.
* **Inheritance**   - :class:`Motor`, :class:`Pump`, :class:`Compressor` and
  :class:`Conveyor` inherit from :class:`Machine`.
* **Polymorphism**  - ``thresholds()`` and ``compute_health()`` are overridden
  per subclass, so the same call site behaves differently per machine type.
* **Encapsulation** - Internal state is stored in private attributes (leading
  underscore) and exposed through read-only :func:`property` accessors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Deque

from app.models.readings import (
    FaultType,
    HealthStatus,
    MachineType,
    SensorReading,
)


class Machine(ABC):
    """Abstract base class representing a monitored industrial machine.

    Subclasses must implement :meth:`thresholds` and :meth:`compute_health`.
    Instances own a rolling buffer of recent readings, exposed read-only.

    Attributes:
        _id: Stable machine identifier.
        _name: Human friendly display name.
        _type: The concrete machine category.
        _readings: Rolling buffer of the most recent sensor readings.
        _thresholds: Per-machine operating limits keyed by sensor name.
    """

    def __init__(
        self,
        machine_id: str,
        name: str,
        machine_type: MachineType,
        thresholds: dict[str, float],
        buffer_size: int = 500,
    ) -> None:
        self._id: str = machine_id
        self._name: str = name
        self._type: MachineType = machine_type
        self._thresholds: dict[str, float] = dict(thresholds)
        self._readings: Deque[SensorReading] = deque(maxlen=buffer_size)
        self._last_prediction: dict | None = None

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------
    @abstractmethod
    def thresholds(self) -> dict[str, float]:
        """Return the critical operating limits for this machine.

        Returns:
            Mapping of sensor name to its critical threshold value.
        """
        raise NotImplementedError

    @abstractmethod
    def nominal(self) -> dict[str, float]:
        """Return the healthy operating point for this machine.

        The nominal point is the value each sensor sits at while the machine
        runs normally.  Together with :meth:`thresholds` it defines the
        span over which the health index degrades: a reading at the nominal
        point scores 100, and a reading at the critical limit scores 0.

        Returns:
            Mapping of sensor name to its healthy operating value.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_health(self, reading: SensorReading) -> float:
        """Compute a 0-100 health index for the given reading.

        Implementations score the reading against the machine's own
        thresholds so that a Pump and a Motor are judged by their own
        operating envelope rather than a global constant.

        Args:
            reading: The sensor reading to score.

        Returns:
            A health index between 0 (failed) and 100 (perfectly healthy).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def id(self) -> str:
        """The stable machine identifier."""
        return self._id

    @property
    def name(self) -> str:
        """The human friendly machine name."""
        return self._name

    @property
    def type(self) -> MachineType:
        """The machine category."""
        return self._type

    @property
    def readings(self) -> list[SensorReading]:
        """A copy of the rolling reading buffer (oldest first)."""
        return list(self._readings)

    @property
    def reading_count(self) -> int:
        """Number of readings currently buffered."""
        return len(self._readings)

    @property
    def last_reading(self) -> SensorReading | None:
        """The most recent reading, or ``None`` if the buffer is empty."""
        return self._readings[-1] if self._readings else None

    @property
    def last_prediction(self) -> dict | None:
        """The most recent ML prediction payload, if any."""
        return self._last_prediction

    def add_reading(self, reading: SensorReading) -> None:
        """Append a reading to the rolling buffer.

        Args:
            reading: The validated reading to store.
        """
        self._readings.append(reading)

    def set_prediction(self, prediction: dict) -> None:
        """Store the latest ML prediction for this machine.

        Args:
            prediction: A serialisable prediction payload.
        """
        self._last_prediction = prediction

    def statistics(self) -> dict[str, dict[str, float]]:
        """Compute min / max / average for every sensor in the buffer.

        Returns:
            Mapping of sensor name to its ``min``/``max``/``avg`` statistics.
            Empty when no readings have been recorded yet.
        """
        if not self._readings:
            return {}

        sensor_names = ("temperature", "vibration", "current", "rpm", "pressure")
        stats: dict[str, dict[str, float]] = {}
        for sensor in sensor_names:
            values = [getattr(r, sensor) for r in self._readings]
            stats[sensor] = {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "avg": round(sum(values) / len(values), 2),
            }
        return stats

    def status_for(self, health_index: float) -> HealthStatus:
        """Map a health index onto a status band.

        Args:
            health_index: The 0-100 health index.

        Returns:
            The matching :class:`HealthStatus` member.
        """
        from app.core.config import settings

        if health_index <= settings.health_critical:
            return HealthStatus.CRITICAL
        if health_index <= settings.health_warning:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} id={self._id!r} readings={len(self._readings)}>"


class Motor(Machine):
    """An electric motor.

    Motors are characterised by strong thermal load and moderate vibration.
    They are sensitive to overload (high current) and bearing wear.
    """

    def __init__(self, machine_id: str = "motor-01", name: str = "Motor 01", **kwargs) -> None:
        super().__init__(
            machine_id=machine_id,
            name=name,
            machine_type=MachineType.MOTOR,
            thresholds=self.thresholds(),
            **kwargs,
        )

    def thresholds(self) -> dict[str, float]:
        """Return the critical operating limits for a motor."""
        return {
            "temperature": 85.0,  # Celsius
            "vibration": 7.1,  # mm/s RMS (ISO 10816 zone D)
            "current": 32.0,  # Amperes
            "rpm": 1800.0,  # Revolutions per minute
            "pressure": 10.0,  # Bar
        }

    def nominal(self) -> dict[str, float]:
        """Return the healthy operating point for a motor."""
        return {
            "temperature": 62.0,
            "vibration": 3.2,
            "current": 22.0,
            "rpm": 1450.0,
            "pressure": 5.0,
        }

    def compute_health(self, reading: SensorReading) -> float:
        """Score a motor reading against its thermal and electrical limits.

        Args:
            reading: The sensor reading to score.

        Returns:
            A health index between 0 and 100.
        """
        limits = self.thresholds()
        base = self.nominal()
        temperature_penalty = _excess_ratio(reading.temperature, base["temperature"], limits["temperature"])
        vibration_penalty = _excess_ratio(reading.vibration, base["vibration"], limits["vibration"])
        current_penalty = _excess_ratio(reading.current, base["current"], limits["current"])

        # Motors degrade mostly through heat and electrical stress.
        penalty = 0.45 * temperature_penalty + 0.30 * vibration_penalty + 0.25 * current_penalty
        return _score(penalty)


class Pump(Machine):
    """A centrifugal pump.

    Pumps run under pressure load, so pressure excursions and cavitation
    induced vibration are the dominant failure signals.
    """

    def __init__(self, machine_id: str = "pump-01", name: str = "Pump 01", **kwargs) -> None:
        super().__init__(
            machine_id=machine_id,
            name=name,
            machine_type=MachineType.PUMP,
            thresholds=self.thresholds(),
            **kwargs,
        )

    def thresholds(self) -> dict[str, float]:
        """Return the critical operating limits for a pump."""
        return {
            "temperature": 80.0,
            "vibration": 6.3,
            "current": 28.0,
            "rpm": 1750.0,
            "pressure": 12.0,
        }

    def nominal(self) -> dict[str, float]:
        """Return the healthy operating point for a pump."""
        return {
            "temperature": 58.0,
            "vibration": 2.8,
            "current": 19.0,
            "rpm": 1400.0,
            "pressure": 8.0,
        }

    def compute_health(self, reading: SensorReading) -> float:
        """Score a pump reading, weighting pressure and vibration heavily.

        Args:
            reading: The sensor reading to score.

        Returns:
            A health index between 0 and 100.
        """
        limits = self.thresholds()
        base = self.nominal()
        temperature_penalty = _excess_ratio(reading.temperature, base["temperature"], limits["temperature"])
        vibration_penalty = _excess_ratio(reading.vibration, base["vibration"], limits["vibration"])
        pressure_penalty = _excess_ratio(reading.pressure, base["pressure"], limits["pressure"])
        current_penalty = _excess_ratio(reading.current, base["current"], limits["current"])

        # Cavitation shows up as vibration plus a pressure drop.
        penalty = (
            0.30 * temperature_penalty
            + 0.30 * vibration_penalty
            + 0.25 * pressure_penalty
            + 0.15 * current_penalty
        )
        return _score(penalty)


class Compressor(Machine):
    """An air compressor.

    Compressors are the hottest and highest-pressure assets in the fleet, and
    are highly sensitive to overload.
    """

    def __init__(
        self, machine_id: str = "compressor-01", name: str = "Compressor 01", **kwargs
    ) -> None:
        super().__init__(
            machine_id=machine_id,
            name=name,
            machine_type=MachineType.COMPRESSOR,
            thresholds=self.thresholds(),
            **kwargs,
        )

    def thresholds(self) -> dict[str, float]:
        """Return the critical operating limits for a compressor."""
        return {
            "temperature": 95.0,
            "vibration": 8.5,
            "current": 45.0,
            "rpm": 2400.0,
            "pressure": 18.0,
        }

    def nominal(self) -> dict[str, float]:
        """Return the healthy operating point for a compressor."""
        return {
            "temperature": 70.0,
            "vibration": 3.8,
            "current": 33.0,
            "rpm": 2000.0,
            "pressure": 12.0,
        }

    def compute_health(self, reading: SensorReading) -> float:
        """Score a compressor reading, weighting current (overload) heavily.

        Args:
            reading: The sensor reading to score.

        Returns:
            A health index between 0 and 100.
        """
        limits = self.thresholds()
        base = self.nominal()
        temperature_penalty = _excess_ratio(reading.temperature, base["temperature"], limits["temperature"])
        current_penalty = _excess_ratio(reading.current, base["current"], limits["current"])
        pressure_penalty = _excess_ratio(reading.pressure, base["pressure"], limits["pressure"])
        vibration_penalty = _excess_ratio(reading.vibration, base["vibration"], limits["vibration"])

        # Compressors fail primarily from sustained overload.
        penalty = (
            0.35 * temperature_penalty
            + 0.30 * current_penalty
            + 0.20 * pressure_penalty
            + 0.15 * vibration_penalty
        )
        return _score(penalty)


class Conveyor(Machine):
    """A belt conveyor.

    Conveyors run at low temperature but suffer from motor imbalance and
    bearing wear, which show up strongly in the vibration channel.
    """

    def __init__(
        self, machine_id: str = "conveyor-01", name: str = "Conveyor 01", **kwargs
    ) -> None:
        super().__init__(
            machine_id=machine_id,
            name=name,
            machine_type=MachineType.CONVEYOR,
            thresholds=self.thresholds(),
            **kwargs,
        )

    def thresholds(self) -> dict[str, float]:
        """Return the critical operating limits for a conveyor."""
        return {
            "temperature": 70.0,
            "vibration": 5.0,
            "current": 22.0,
            "rpm": 1200.0,
            "pressure": 6.0,
        }

    def nominal(self) -> dict[str, float]:
        """Return the healthy operating point for a conveyor."""
        return {
            "temperature": 48.0,
            "vibration": 2.2,
            "current": 14.0,
            "rpm": 900.0,
            "pressure": 3.0,
        }

    def compute_health(self, reading: SensorReading) -> float:
        """Score a conveyor reading, weighting vibration (imbalance) heavily.

        Args:
            reading: The sensor reading to score.

        Returns:
            A health index between 0 and 100.
        """
        limits = self.thresholds()
        base = self.nominal()
        vibration_penalty = _excess_ratio(reading.vibration, base["vibration"], limits["vibration"])
        temperature_penalty = _excess_ratio(reading.temperature, base["temperature"], limits["temperature"])
        current_penalty = _excess_ratio(reading.current, base["current"], limits["current"])
        rpm_penalty = _excess_ratio(reading.rpm, base["rpm"], limits["rpm"])

        # Imbalance and bearing wear dominate conveyor failures.
        penalty = (
            0.40 * vibration_penalty
            + 0.25 * temperature_penalty
            + 0.20 * current_penalty
            + 0.15 * rpm_penalty
        )
        return _score(penalty)


# ----------------------------------------------------------------------
# Shared scoring helpers
# ----------------------------------------------------------------------
def _excess_ratio(value: float, nominal: float, limit: float) -> float:
    """Return how far ``value`` strays from nominal, scaled to the limit.

    The score is measured as a fraction of the span between the healthy
    operating point and the critical limit.  A value at the nominal point
    produces 0.0 and a value at the critical limit produces 1.0, so the
    health index degrades smoothly and predictably across that span.

    Args:
        value: The measured sensor value.
        nominal: The healthy operating value for that sensor.
        limit: The critical threshold for that sensor.

    Returns:
        A non-negative ratio of how far the reading has moved towards the
        critical limit. Values beyond the limit keep growing past 1.0.
    """
    span = limit - nominal
    if abs(span) < 1e-9:
        # Degenerate span: fall back to a plain ratio against the limit.
        if limit <= 0:
            return 0.0
        return max(0.0, value / limit - 1.0)

    ratio = (value - nominal) / span
    return max(0.0, ratio)


def _score(penalty: float) -> float:
    """Convert an aggregated penalty into a 0-100 health index.

    The mapping is a smooth curve anchored on the machine's own operating
    envelope: a penalty of 0 (a perfectly nominal reading) scores 100, a
    penalty of 1 (the reading sits at the critical limit) scores 50, and a
    penalty of 2 (twice the limit) scores near 0.  Small excursions barely
    move the score while severe ones drop it sharply.

    Args:
        penalty: The weighted sum of per-sensor exceedance ratios.

    Returns:
        A health index rounded to one decimal place.
    """
    import math

    # Anchored at (0 -> 100), (1 -> 50) and decaying towards 0 for larger
    # penalties. The exponential term keeps the curve smooth and monotonic.
    health = 100.0 / (1.0 + penalty) * math.exp(-0.35 * max(0.0, penalty - 1.0))
    return round(max(0.0, min(100.0, health)), 1)