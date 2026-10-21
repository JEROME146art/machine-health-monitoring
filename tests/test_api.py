"""Tests for the FastAPI application.

Uses FastAPI's ``TestClient`` so the full request path - routing, dependency
injection, Pydantic validation, the service layer and the exception handlers
- is exercised exactly as it is in production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """A test client with the application lifespan running.

    The lifespan trains the ML models and starts the simulator, so the
    fixture must be a context manager for the lifespan to execute.
    """
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------------------------------------------------
# Health and documentation
# ----------------------------------------------------------------------
def test_health_endpoint_reports_ok(client: TestClient) -> None:
    """The health endpoint must report a healthy service."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["machines"] == 4
    assert body["models_trained"] is True


def test_openapi_docs_are_available(client: TestClient) -> None:
    """The auto-generated documentation must be reachable."""
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_dashboard_is_served(client: TestClient) -> None:
    """The root path must serve the single-page dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Machine Health Monitoring" in response.text


# ----------------------------------------------------------------------
# Machines
# ----------------------------------------------------------------------
def test_list_machines_returns_the_fleet(client: TestClient) -> None:
    """The fleet endpoint must return the four default machines."""
    response = client.get("/api/machines")
    assert response.status_code == 200

    machines = response.json()
    assert len(machines) == 4

    ids = {machine["id"] for machine in machines}
    assert ids == {"motor-01", "pump-01", "compressor-01", "conveyor-01"}

    for machine in machines:
        assert machine["status"] in {"HEALTHY", "WARNING", "CRITICAL"}
        assert 0 <= machine["health_index"] <= 100
        assert machine["name"]


def test_get_machine_detail(client: TestClient) -> None:
    """The detail endpoint must return the full machine payload."""
    response = client.get("/api/machines/motor-01")
    assert response.status_code == 200

    body = response.json()
    assert body["id"] == "motor-01"
    assert body["type"] == "Motor"
    assert "fault_type" in body
    assert "remaining_useful_life" in body
    assert "cause" in body


def test_get_unknown_machine_returns_404(client: TestClient) -> None:
    """An unknown machine id must produce a structured 404."""
    response = client.get("/api/machines/does-not-exist")
    assert response.status_code == 404

    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "machine_not_found"


def test_get_readings_returns_a_list(client: TestClient) -> None:
    """The readings endpoint must return the buffered readings."""
    response = client.get("/api/machines/motor-01/readings?limit=10")
    assert response.status_code == 200

    readings = response.json()
    assert isinstance(readings, list)
    assert len(readings) <= 10

    if readings:
        reading = readings[-1]
        assert set(reading) >= {
            "machine_id", "machine_type", "temperature",
            "vibration", "current", "rpm", "pressure", "timestamp",
        }


def test_readings_limit_is_validated(client: TestClient) -> None:
    """An out-of-range limit must be rejected with a 422."""
    response = client.get("/api/machines/motor-01/readings?limit=99999")
    assert response.status_code == 422


# ----------------------------------------------------------------------
# Reading intake (the ESP32 path)
# ----------------------------------------------------------------------
def test_post_reading_is_accepted(client: TestClient) -> None:
    """A real IoT reading must be ingested and scored."""
    payload = {
        "machine_id": "motor-01",
        "machine_type": "Motor",
        "temperature": 62.5,
        "vibration": 3.4,
        "current": 22.1,
        "rpm": 1452.0,
        "pressure": 5.1,
    }
    response = client.post("/api/readings", json=payload)
    assert response.status_code == 201

    body = response.json()
    assert 0 <= body["health_index"] <= 100
    assert body["status"] in {"HEALTHY", "WARNING", "CRITICAL"}
    assert "anomaly_detection" in body
    assert "fault_classification" in body
    assert "remaining_useful_life" in body
    assert body["cause"]


def test_post_reading_rejects_an_invalid_payload(client: TestClient) -> None:
    """A malformed reading must be rejected with a 422."""
    response = client.post("/api/readings", json={"machine_id": "motor-01"})
    assert response.status_code == 422

    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"


def test_post_reading_rejects_an_unknown_machine(client: TestClient) -> None:
    """A reading for an unknown machine must produce a 404."""
    payload = {
        "machine_id": "ghost-machine",
        "machine_type": "Motor",
        "temperature": 60.0,
        "vibration": 3.0,
        "current": 20.0,
        "rpm": 1400.0,
        "pressure": 5.0,
    }
    response = client.post("/api/readings", json=payload)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "machine_not_found"


# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------
def test_list_alerts_returns_a_list(client: TestClient) -> None:
    """The alerts endpoint must return a list."""
    response = client.get("/api/alerts?limit=10")
    assert response.status_code == 200

    alerts = response.json()
    assert isinstance(alerts, list)
    assert len(alerts) <= 10

    for alert in alerts:
        assert alert["severity"] in {"INFO", "WARNING", "CRITICAL"}
        assert alert["cause"]
        assert alert["timestamp"]


def test_alerts_can_be_filtered_by_machine(client: TestClient) -> None:
    """Filtering by machine must only return that machine's alerts."""
    response = client.get("/api/alerts?machine_id=pump-01")
    assert response.status_code == 200

    for alert in response.json():
        assert alert["machine_id"] == "pump-01"


# ----------------------------------------------------------------------
# Demo controls
# ----------------------------------------------------------------------
def test_inject_fault_returns_a_confirmation(client: TestClient) -> None:
    """The fault-injection control must report what it injected."""
    response = client.post(
        "/api/machines/motor-01/inject-fault?fault_type=overheating"
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["machine_id"] == "motor-01"
    assert body["fault_type"] == "overheating"


def test_inject_fault_rejects_an_unknown_type(client: TestClient) -> None:
    """An unknown fault type must be rejected with a 422."""
    response = client.post(
        "/api/machines/motor-01/inject-fault?fault_type=explosion"
    )
    assert response.status_code == 422


def test_inject_fault_on_unknown_machine_returns_404(client: TestClient) -> None:
    """Fault injection on an unknown machine must produce a 404."""
    response = client.post("/api/machines/ghost/inject-fault")
    assert response.status_code == 404


def test_simulation_can_be_paused_and_resumed(client: TestClient) -> None:
    """The pause/resume controls must toggle the simulator state."""
    paused = client.post("/api/simulation/pause")
    assert paused.status_code == 200
    assert paused.json()["simulation_running"] is False

    resumed = client.post("/api/simulation/resume")
    assert resumed.status_code == 200
    assert resumed.json()["simulation_running"] is True


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
def test_export_csv_returns_a_csv_file(client: TestClient) -> None:
    """The export endpoint must return a downloadable CSV."""
    response = client.get("/api/export/csv?limit=50")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    text = response.text
    assert text.splitlines()[0] == (
        "timestamp,machine_id,machine_type,temperature,vibration,current,rpm,pressure"
    )


def test_export_csv_for_a_single_machine(client: TestClient) -> None:
    """Exporting one machine must only include that machine's rows."""
    response = client.get("/api/export/csv?machine_id=pump-01")
    assert response.status_code == 200

    lines = response.text.strip().splitlines()
    for line in lines[1:]:
        assert ",pump-01," in line
