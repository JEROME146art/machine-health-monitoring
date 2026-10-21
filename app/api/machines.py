"""Machine and reading endpoints.

Exposes the fleet overview, per-machine detail, raw readings, the public
reading intake used by real IoT devices, and the demo fault-injection control.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_service
from app.models.readings import (
    HealthPrediction,
    MachineDetail,
    MachineSummary,
    SensorReading,
)
from app.services.machine_service import MachineService

router = APIRouter(prefix="/api", tags=["machines"])


@router.get("/machines", response_model=list[MachineSummary])
async def list_machines(
    service: Annotated[MachineService, Depends(get_service)],
) -> list[MachineSummary]:
    """Return a summary of every monitored machine.

    Args:
        service: The injected machine service.

    Returns:
        The fleet summary list.
    """
    return service.list_machines()


@router.get("/machines/{machine_id}", response_model=MachineDetail)
async def get_machine(
    machine_id: str,
    service: Annotated[MachineService, Depends(get_service)],
) -> MachineDetail:
    """Return the detailed view of one machine.

    Args:
        machine_id: The machine identifier.
        service: The injected machine service.

    Returns:
        The machine detail.

    Raises:
        MachineNotFoundError: If the machine does not exist (404).
    """
    return service.get_machine_detail(machine_id)


@router.get("/machines/{machine_id}/readings", response_model=list[SensorReading])
async def get_machine_readings(
    machine_id: str,
    service: Annotated[MachineService, Depends(get_service)],
    limit: Annotated[int, Query(ge=1, le=500, description="Readings to return")] = 100,
) -> list[SensorReading]:
    """Return the most recent readings for a machine.

    Args:
        machine_id: The machine identifier.
        limit: Maximum number of readings to return.
        service: The injected machine service.

    Returns:
        The requested readings, oldest first.

    Raises:
        MachineNotFoundError: If the machine does not exist (404).
    """
    return service.get_readings(machine_id, limit)


@router.post(
    "/readings",
    response_model=HealthPrediction,
    status_code=status.HTTP_201_CREATED,
)
async def post_reading(
    reading: SensorReading,
    service: Annotated[MachineService, Depends(get_service)],
) -> HealthPrediction:
    """Ingest a sensor reading from a real IoT device.

    A physical ESP32 (or any HTTP client) can push readings to this endpoint
    and they will be processed by the same ML pipeline the simulator uses.

    Args:
        reading: The sensor reading payload.
        service: The injected machine service.

    Returns:
        The health prediction computed from the reading.

    Raises:
        MachineNotFoundError: If the machine id is unknown (404).
        ValidationError: If the payload is malformed (422).
    """
    return service.process_reading(reading)


@router.post("/machines/{machine_id}/inject-fault")
async def inject_fault(
    machine_id: str,
    service: Annotated[MachineService, Depends(get_service)],
    fault_type: Annotated[
        str,
        Query(
            pattern="^(random|overheating|bearing_wear|overload|imbalance)$",
            description="Kind of fault to inject",
        ),
    ] = "random",
) -> dict:
    """Inject a sudden fault into a machine (demo control).

    Args:
        machine_id: The machine identifier.
        fault_type: The kind of fault to inject.
        service: The injected machine service.

    Returns:
        A confirmation payload describing the injected fault.

    Raises:
        MachineNotFoundError: If the machine does not exist (404).
        SimulationStateError: If the simulator is not running (409).
    """
    fault = service.inject_fault(machine_id, fault_type)
    return {
        "success": True,
        "machine_id": machine_id,
        "fault_type": fault,
        "message": f"Injected '{fault}' fault into {machine_id}.",
    }


@router.post("/machines/{machine_id}/clear-fault")
async def clear_machine_fault(
    machine_id: str,
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Clear active injected fault from a machine.

    Args:
        machine_id: The machine identifier.
        service: The injected machine service.

    Returns:
        Confirmation payload.
    """
    service.clear_fault(machine_id)
    return {
        "success": True,
        "machine_id": machine_id,
        "message": f"Cleared active fault on {machine_id}.",
    }


@router.post("/fleet/reset")
async def reset_fleet(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Reset all machines and simulators to factory baseline conditions.

    Args:
        service: The injected machine service.

    Returns:
        Confirmation payload.
    """
    service.reset_fleet()
    return {
        "success": True,
        "message": "All machines reset to factory nominal baseline.",
    }


@router.get("/ml/metrics")
async def get_ml_metrics(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Return ML model metadata, architecture details, and feature importances.

    Args:
        service: The injected machine service.

    Returns:
        Dictionary of ML metrics.
    """
    return service.get_ml_metrics()


@router.post("/ml/retrain")
async def retrain_models(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Trigger a full retraining pass on all ML models.

    Args:
        service: The injected machine service.

    Returns:
        Updated training summary and metrics.
    """
    summary = service.train_models()
    metrics = service.get_ml_metrics()
    return {
        "success": True,
        "message": "ML models successfully retrained.",
        "summary": summary,
        "metrics": metrics,
    }

