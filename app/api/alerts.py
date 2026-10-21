"""Alert endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_service
from app.models.readings import Alert, AlertSeverity
from app.services.machine_service import MachineService

router = APIRouter(prefix="/api", tags=["alerts"])


@router.get("/alerts", response_model=list[Alert])
async def list_alerts(
    service: Annotated[MachineService, Depends(get_service)],
    limit: Annotated[int, Query(ge=1, le=200, description="Alerts to return")] = 50,
    machine_id: Annotated[str | None, Query(description="Filter by machine")] = None,
    severity: Annotated[
        AlertSeverity | None, Query(description="Filter by severity")
    ] = None,
) -> list[Alert]:
    """Return the alert history, newest first.

    Args:
        limit: Maximum number of alerts to return.
        machine_id: Optional machine filter.
        severity: Optional severity filter.
        service: The injected machine service.

    Returns:
        The filtered alert list.
    """
    return service.alerts.list_alerts(
        limit=limit, machine_id=machine_id, severity=severity
    )


@router.post("/alerts/{alert_id}/acknowledge", response_model=Alert)
async def acknowledge_alert(
    alert_id: str,
    service: Annotated[MachineService, Depends(get_service)],
) -> Alert:
    """Acknowledge an alert.

    Args:
        alert_id: The alert identifier.
        service: The injected machine service.

    Returns:
        The updated alert.

    Raises:
        MachineNotFoundError: If the alert id is unknown (404).
    """
    from app.core.exceptions import MachineNotFoundError

    alert = service.alerts.acknowledge(alert_id)
    if alert is None:
        raise MachineNotFoundError(
            f"Alert '{alert_id}' does not exist.", alert_id=alert_id
        )
    return alert


@router.post("/alerts/clear")
async def clear_all_alerts(
    service: Annotated[MachineService, Depends(get_service)],
) -> dict:
    """Clear all stored alerts from memory.

    Args:
        service: The injected machine service.

    Returns:
        Confirmation payload.
    """
    service.alerts.clear()
    return {"success": True, "message": "All alerts cleared."}

