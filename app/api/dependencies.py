"""FastAPI dependency providers.

Dependencies are resolved from ``request.app.state`` so that the service
instance created during application start-up is the single source of truth
for every request handler.
"""

from __future__ import annotations

from fastapi import Request

from app.services.machine_service import MachineService


def get_service(request: Request) -> MachineService:
    """Return the application-wide :class:`MachineService`.

    Args:
        request: The incoming request.

    Returns:
        The service instance stored on the app state.

    Raises:
        RuntimeError: If the service has not been initialised.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise RuntimeError(
            "MachineService has not been initialised; the app lifespan did not run."
        )
    return service
