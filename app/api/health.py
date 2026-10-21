"""Service health and simulation-control endpoints.

``GET /health`` is the endpoint that Render, Railway and Docker use to
determine whether the container is alive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_service
from app.services.machine_service import MachineService

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Report the health of the running service.

    Args:
        service: The injected machine service.

    Returns:
        A status payload with the simulator state and machine count.
    """
    machines = service.list_machines()
    return {
        "status": "ok",
        "service": "machine-health-monitoring",
        "timestamp": datetime.utcnow().isoformat(),
        "simulation_running": service.simulation_running,
        "machines": len(machines),
        "models_trained": bool(service.training_summary),
        "version": "1.0.0",
    }


@router.post("/api/simulation/pause")
async def pause_simulation(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Pause the background sensor simulation.

    Args:
        service: The injected machine service.

    Returns:
        A confirmation payload.

    Raises:
        SimulationStateError: If the simulation is not running (409).
    """
    service.pause_simulation()
    return {"success": True, "simulation_running": False}


@router.post("/api/simulation/resume")
async def resume_simulation(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Resume a paused sensor simulation.

    Args:
        service: The injected machine service.

    Returns:
        A confirmation payload.

    Raises:
        SimulationStateError: If the simulation is already running (409).
    """
    service.resume_simulation()
    return {"success": True, "simulation_running": True}
