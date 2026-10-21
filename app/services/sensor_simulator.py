"""Asynchronous IoT sensor simulator.

The simulator runs as a background asyncio task inside the FastAPI process.
Every ``simulation_interval`` seconds it emits a reading for each machine,
applying two kinds of degradation:

* **Gradual degradation** - a slow, cumulative drift in one or more sensor
  channels that slowly erodes the machine's health index.
* **Sudden faults** - an injected spike that mimics an abrupt mechanical
  failure.

The same simulator can be driven from the ``Inject Fault`` button in the UI
or via ``POST /api/machines/{id}/inject-fault``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from typing import Callable, Optional

from app.core.config import settings
from app.models.readings import MachineType, SensorReading

logger = logging.getLogger(__name__)

#: Nominal operating point and per-channel noise for each machine type.
NOMINAL: dict[MachineType, dict[str, float]] = {
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

#: Standard deviation of the measurement noise for each channel.
NOISE: dict[MachineType, dict[str, float]] = {
    MachineType.MOTOR: {
        "temperature": 2.5, "vibration": 0.35, "current": 1.2, "rpm": 60.0, "pressure": 0.4,
    },
    MachineType.PUMP: {
        "temperature": 2.2, "vibration": 0.30, "current": 1.0, "rpm": 55.0, "pressure": 0.5,
    },
    MachineType.COMPRESSOR: {
        "temperature": 3.0, "vibration": 0.45, "current": 1.8, "rpm": 90.0, "pressure": 0.7,
    },
    MachineType.CONVEYOR: {
        "temperature": 1.8, "vibration": 0.25, "current": 0.8, "rpm": 40.0, "pressure": 0.3,
    },
}

#: Multipliers applied when a sudden fault of the given kind is injected.
FAULT_SPIKES: dict[str, dict[str, float]] = {
    "overheating": {"temperature": 1.45, "vibration": 1.15, "current": 1.20, "rpm": 1.02, "pressure": 1.05},
    "bearing_wear": {"temperature": 1.18, "vibration": 2.30, "current": 1.15, "rpm": 0.95, "pressure": 0.96},
    "overload": {"temperature": 1.28, "vibration": 1.40, "current": 1.75, "rpm": 0.88, "pressure": 1.35},
    "imbalance": {"temperature": 1.12, "vibration": 2.05, "current": 1.28, "rpm": 1.12, "pressure": 0.93},
    "random": {"temperature": 1.35, "vibration": 1.85, "current": 1.45, "rpm": 0.92, "pressure": 1.20},
}


class MachineSimulator:
    """Generates sensor readings for a single machine.

    Attributes:
        _machine_id: The machine this simulator feeds.
        _machine_type: The machine category.
        _drift: Cumulative per-channel degradation multipliers.
        _injected_fault: Name of an active injected fault, if any.
        _fault_decay: Remaining lifetime of the injected fault in ticks.
        _rng: The random generator used for noise.
    """

    def __init__(
        self, machine_id: str, machine_type: MachineType, seed: Optional[int] = None
    ) -> None:
        """Create a simulator for one machine.

        Args:
            machine_id: The machine identifier.
            machine_type: The machine category.
            seed: Optional seed for reproducible noise.
        """
        self._machine_id = machine_id
        self._machine_type = machine_type
        self._drift: dict[str, float] = {k: 1.0 for k in NOISE[machine_type]}
        self._injected_fault: str | None = None
        self._fault_decay: int = 0
        self._rng = random.Random(seed if seed is not None else settings.random_seed)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def machine_id(self) -> str:
        """The machine this simulator feeds."""
        return self._machine_id

    @property
    def drift(self) -> dict[str, float]:
        """The current cumulative degradation multipliers."""
        return dict(self._drift)

    @property
    def active_fault(self) -> str | None:
        """The name of the currently active injected fault, if any."""
        return self._injected_fault

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def inject_fault(self, fault_type: str = "random") -> str:
        """Inject a sudden fault into this machine.

        Args:
            fault_type: One of ``overheating``, ``bearing_wear``, ``overload``,
                ``imbalance`` or ``random``.

        Returns:
            The name of the injected fault.

        Raises:
            ValueError: If the fault name is not recognised.
        """
        key = fault_type.lower().replace(" ", "_").replace("-", "_")
        if key not in FAULT_SPIKES:
            raise ValueError(
                f"Unknown fault type '{fault_type}'. "
                f"Expected one of: {', '.join(sorted(FAULT_SPIKES))}."
            )
        self._injected_fault = key
        self._fault_decay = 12  # the fault persists for roughly 24 seconds
        return key

    def clear_fault(self) -> None:
        """Remove any active injected fault."""
        self._injected_fault = None
        self._fault_decay = 0

    def reset(self) -> None:
        """Reset drift and active fault to factory nominal condition."""
        self._drift = {k: 1.0 for k in NOISE[self._machine_type]}
        self._injected_fault = None
        self._fault_decay = 0

    def _advance_degradation(self) -> None:
        """Apply a small amount of gradual wear to the machine.

        The drift is tiny per tick but accumulates, so a machine left running
        long enough will slowly deteriorate - exactly the behaviour the RUL
        predictor is designed to extrapolate.
        """
        for channel in self._drift:
            # Temperature and vibration degrade slightly faster than the rest.
            rate = 0.0016 if channel in {"temperature", "vibration"} else 0.0009
            self._drift[channel] += rate * self._rng.uniform(0.4, 1.6)
        # Cap the drift so the simulator never produces absurd values.
        for channel, value in self._drift.items():
            self._drift[channel] = min(value, 1.45)

    def generate(self) -> SensorReading:
        """Produce the next reading for this machine.

        Returns:
            A validated :class:`SensorReading`.
        """
        self._advance_degradation()

        nominal = NOMINAL[self._machine_type]
        noise = NOISE[self._machine_type]
        spike = FAULT_SPIKES.get(self._injected_fault or "", {})

        def channel(name: str, lower: float, upper: float) -> float:
            """Generate one channel value with noise, drift and any spike."""
            value = nominal[name] * self._drift[name] * spike.get(name, 1.0)
            value += self._rng.gauss(0.0, noise[name])
            return round(max(lower, min(upper, value)), 2)

        reading = SensorReading(
            machine_id=self._machine_id,
            machine_type=self._machine_type,
            temperature=channel("temperature", 20.0, 145.0),
            vibration=channel("vibration", 0.05, 19.5),
            current=channel("current", 0.5, 95.0),
            rpm=channel("rpm", 50.0, 2600.0),
            pressure=channel("pressure", 0.0, 30.0),
            timestamp=datetime.utcnow(),
        )

        # Count down the injected fault so it eventually clears itself.
        if self._fault_decay > 0:
            self._fault_decay -= 1
            if self._fault_decay == 0:
                self._injected_fault = None

        return reading


class SensorSimulator:
    """Drives every :class:`MachineSimulator` on a fixed interval.

    Attributes:
        _simulators: Per-machine simulators keyed by machine id.
        _callback: Invoked with each freshly generated reading.
        _task: The background asyncio task.
        _running: Whether the simulator is currently emitting.
    """

    def __init__(
        self,
        machines: list,
        callback: Callable[[SensorReading], None],
        interval: float | None = None,
    ) -> None:
        """Create the simulator.

        Args:
            machines: The machines to feed.
            callback: Called with every generated reading.
            interval: Seconds between emission rounds.
        """
        self._interval = (
            interval if interval is not None else settings.simulation_interval
        )
        self._callback = callback
        self._simulators: dict[str, MachineSimulator] = {
            machine.id: MachineSimulator(machine.id, machine.type) for machine in machines
        }
        self._task: asyncio.Task | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background emission loop."""
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="sensor-simulator")
        logger.info("Sensor simulator started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        """Stop the background emission loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Sensor simulator stopped")

    async def _loop(self) -> None:
        """The emission loop.

        Runs until cancelled, emitting one reading per machine per tick.
        """
        try:
            while self._running:
                for simulator in list(self._simulators.values()):
                    reading = simulator.generate()
                    try:
                        self._callback(reading)
                    except Exception:  # noqa: BLE001 - a bad tick must not kill the loop
                        logger.exception(
                            "Failed to process reading for %s", simulator.machine_id
                        )
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("Sensor simulator loop cancelled")
            raise

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Pause emission without cancelling the task."""
        self._running = False

    def resume(self) -> None:
        """Resume emission."""
        self._running = True

    @property
    def running(self) -> bool:
        """Whether the simulator is currently emitting."""
        return self._running

    def inject_fault(self, machine_id: str, fault_type: str = "random") -> str:
        """Inject a fault into one machine.

        Args:
            machine_id: The target machine.
            fault_type: The kind of fault to inject.

        Returns:
            The name of the injected fault.

        Raises:
            KeyError: If the machine is unknown.
            ValueError: If the fault name is not recognised.
        """
        simulator = self._simulators.get(machine_id)
        if simulator is None:
            raise KeyError(machine_id)
        return simulator.inject_fault(fault_type)

    def clear_fault(self, machine_id: str) -> None:
        """Clear active fault on a machine."""
        simulator = self._simulators.get(machine_id)
        if simulator is not None:
            simulator.clear_fault()

    def reset_all(self) -> None:
        """Reset all machines to pristine factory nominal conditions."""
        for simulator in self._simulators.values():
            simulator.reset()

    def machine_ids(self) -> list[str]:
        """The ids of every simulated machine."""
        return list(self._simulators.keys())
