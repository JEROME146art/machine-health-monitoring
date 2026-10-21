"""Factory pattern for constructing machine instances.

The :class:`MachineFactory` centralises machine creation so that the API and
service layers never need to know which concrete subclass backs a given
machine type.  Adding a new machine type requires editing only this file.
"""

from __future__ import annotations

from typing import Callable

from app.core.exceptions import MachineHealthError
from app.machine import Compressor, Conveyor, Machine, Motor, Pump
from app.models.readings import MachineType


class UnknownMachineTypeError(MachineHealthError):
    """Raised when a factory is asked for an unregistered machine type."""

    status_code = 400
    code = "unknown_machine_type"


class MachineFactory:
    """Factory producing :class:`Machine` instances by type.

    The factory maintains a registry of builders, which keeps the creation
    logic in one place and makes new machine types trivial to add.
    """

    # Registry of concrete builders keyed by their machine type.
    _builders: dict[MachineType, Callable[..., Machine]] = {
        MachineType.MOTOR: Motor,
        MachineType.PUMP: Pump,
        MachineType.COMPRESSOR: Compressor,
        MachineType.CONVEYOR: Conveyor,
    }

    @classmethod
    def register(
        cls, machine_type: MachineType, builder: Callable[..., Machine]
    ) -> None:
        """Register a custom builder for a machine type.

        Args:
            machine_type: The type the builder produces.
            builder: A callable returning a :class:`Machine` instance.

        Raises:
            TypeError: If the builder does not return a Machine.
        """
        if not callable(builder):
            raise TypeError("builder must be callable")
        cls._builders[machine_type] = builder

    @classmethod
    def create(
        cls,
        machine_type: MachineType,
        machine_id: str,
        name: str | None = None,
        **kwargs,
    ) -> Machine:
        """Create a machine instance of the requested type.

        Args:
            machine_type: The category of machine to build.
            machine_id: Stable identifier for the new machine.
            name: Optional display name; derived from the type when omitted.
            **kwargs: Extra keyword arguments forwarded to the builder
                (for example ``buffer_size``).

        Returns:
            A fully constructed :class:`Machine` instance.

        Raises:
            UnknownMachineTypeError: If the type has no registered builder.
        """
        builder = cls._builders.get(machine_type)
        if builder is None:
            raise UnknownMachineTypeError(
                f"No builder registered for machine type '{machine_type}'.",
                requested=str(machine_type),
                available=[str(t) for t in cls._builders],
            )
        display_name = name or f"{machine_type.value} {machine_id.split('-')[-1]}"
        return builder(machine_id=machine_id, name=display_name, **kwargs)

    @classmethod
    def available_types(cls) -> list[MachineType]:
        """Return every machine type the factory can build."""
        return list(cls._builders.keys())


def create_default_fleet(buffer_size: int = 500) -> list[Machine]:
    """Build the default four-machine fleet used by the simulator.

    Args:
        buffer_size: Maximum readings retained per machine.

    Returns:
        A list of four machines: Motor, Pump, Compressor and Conveyor.
    """
    specs = [
        (MachineType.MOTOR, "motor-01", "Motor 01"),
        (MachineType.PUMP, "pump-01", "Pump 01"),
        (MachineType.COMPRESSOR, "compressor-01", "Compressor 01"),
        (MachineType.CONVEYOR, "conveyor-01", "Conveyor 01"),
    ]
    return [
        MachineFactory.create(
            machine_type=mtype, machine_id=mid, name=mname, buffer_size=buffer_size
        )
        for mtype, mid, mname in specs
    ]
