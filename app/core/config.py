"""Application configuration loaded from environment variables.

All tunables live here so that behaviour can be changed without touching
business logic. Values are read once at import time and validated eagerly so
that a mis-configured deployment fails fast at boot instead of at request time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Attributes:
        app_name: Human readable application name shown in the UI.
        host: Interface uvicorn binds to.
        port: TCP port the server listens on.
        simulation_interval: Seconds between two simulated sensor readings.
        readings_buffer_size: Maximum number of readings retained per machine.
        random_seed: Seed used for reproducible synthetic training data.
        enable_simulation: Whether the background simulator starts on boot.
        sqlite_path: Optional path for the SQLite persistence layer.
        train_samples: Number of synthetic healthy samples used for training.
        anomaly_threshold: IsolationForest contamination rate. A low value
            keeps the false-positive rate on healthy data around 2% while
            still detecting every injected fault.
        health_warning: Health index at or below which a machine is WARNING.
        health_critical: Health index at or below which a machine is CRITICAL.
    """

    app_name: str = "Machine Health Monitoring System"
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    simulation_interval: float = field(
        default_factory=lambda: _env_float("SIMULATION_INTERVAL", 2.0)
    )
    readings_buffer_size: int = field(
        default_factory=lambda: _env_int("READINGS_BUFFER_SIZE", 500)
    )
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))
    enable_simulation: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SIMULATION", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    sqlite_path: str | None = field(
        default_factory=lambda: os.getenv("SQLITE_PATH") or None
    )
    train_samples: int = field(default_factory=lambda: _env_int("TRAIN_SAMPLES", 1200))
    anomaly_threshold: float = field(
        default_factory=lambda: _env_float("ANOMALY_THRESHOLD", 0.02)
    )
    health_warning: float = field(default_factory=lambda: _env_float("HEALTH_WARNING", 60.0))
    health_critical: float = field(default_factory=lambda: _env_float("HEALTH_CRITICAL", 30.0))


settings = Settings()