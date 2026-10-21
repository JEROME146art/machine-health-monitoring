"""FastAPI application entry point.

Wires together the routers, the static dashboard, the exception handlers and
the application lifespan (which trains the ML models and starts the sensor
simulator).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import alerts, export, health, machines
from app.core.config import settings
from app.core.exceptions import (
    MachineHealthError,
    machine_health_exception_handler,
    validation_exception_handler,
)
from app.services.machine_service import MachineService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("app.main")

#: Directory containing the single-page dashboard assets.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifecycle.

    On start-up the ML models are trained on synthetic data and the background
    sensor simulator is started.  On shutdown the simulator is cancelled
    cleanly.

    Args:
        app: The FastAPI application being started.

    Yields:
        Control back to the server while the application is running.
    """
    service = MachineService()
    logger.info("Training ML models on synthetic data...")
    summary = service.train_models()
    logger.info("ML models ready: %s", summary)

    app.state.service = service

    if settings.enable_simulation:
        await service.start_simulation()
        logger.info(
            "Sensor simulator running for %d machines", len(service.list_machines())
        )

    try:
        yield
    finally:
        await service.stop_simulation()
        logger.info("Application shutting down")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Ingest IoT sensor data from industrial machines, detect anomalies with "
        "IsolationForest, classify fault types with RandomForest, predict "
        "Remaining Useful Life by linear trend regression, and expose a live "
        "health dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ----------------------------------------------------------------------
# Middleware
# ----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Exception handlers
# ----------------------------------------------------------------------
app.add_exception_handler(MachineHealthError, machine_health_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean JSON payload for unexpected server errors.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        A 500 response with a generic error envelope.
    """
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred while processing the request.",
            },
        },
    )


# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------
app.include_router(health.router)
app.include_router(machines.router)
app.include_router(alerts.router)
app.include_router(export.router)


# ----------------------------------------------------------------------
# Static dashboard
# ----------------------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
    )

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        """Serve the single-page dashboard.

        Returns:
            The dashboard HTML file.
        """
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """Serve the favicon.

        Returns:
            The favicon file.
        """
        return FileResponse(
            STATIC_DIR / "favicon.svg", media_type="image/svg+xml"
        )
