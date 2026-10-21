"""Custom exception hierarchy and FastAPI exception handlers.

Every error raised by the domain layer is a subclass of :class:`MachineHealthError`.
This gives the API layer a single, predictable place to translate internal
failures into clean JSON error payloads instead of leaking stack traces.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


class MachineHealthError(Exception):
    """Base class for every error raised by this application.

    Attributes:
        status_code: HTTP status code the API should return.
        detail: Human readable message safe to show to end users.
        code: Stable machine readable error identifier.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, detail: str, **context: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.context = context

    def to_payload(self) -> dict[str, Any]:
        """Serialise the error into the standard API error envelope."""
        payload: dict[str, Any] = {
            "success": False,
            "error": {"code": self.code, "message": self.detail},
        }
        if self.context:
            payload["error"]["context"] = self.context
        return payload


class MachineNotFoundError(MachineHealthError):
    """Raised when a requested machine identifier does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "machine_not_found"


class InvalidReadingError(MachineHealthError):
    """Raised when an inbound sensor reading fails validation."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "invalid_reading"


class ModelNotTrainedError(MachineHealthError):
    """Raised when an ML model is used before training has completed."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_trained"


class SimulationStateError(MachineHealthError):
    """Raised when an invalid simulation control action is requested."""

    status_code = status.HTTP_409_CONFLICT
    code = "simulation_state_error"


class DuplicateSubscriberError(MachineHealthError):
    """Raised when the same observer subscribes twice to the alert service."""

    status_code = status.HTTP_409_CONFLICT
    code = "duplicate_subscriber"


async def machine_health_exception_handler(
    request: Request, exc: MachineHealthError
) -> JSONResponse:
    """Translate a domain exception into a structured JSON response.

    Args:
        request: The incoming request (unused, kept for FastAPI signature).
        exc: The raised domain exception.

    Returns:
        A JSON response with the configured status code and error envelope.
    """
    return JSONResponse(
        status_code=exc.status_code, content=exc.to_payload()
    )


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return a consistent envelope for Pydantic validation failures.

    Args:
        request: The incoming request.
        exc: The validation error raised by FastAPI/Pydantic.

    Returns:
        A JSON response with 422 status and a normalised error payload.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "validation_error",
                "message": "Request payload failed validation.",
                "context": {"errors": str(exc)},
            },
        },
    )