# Machine Health Monitoring System using IoT and ML

A production-quality web application that ingests IoT sensor data from industrial
machines, detects anomalies with machine learning, predicts failures before they
happen, and presents everything on a live, glassmorphic dashboard.

Four machines (Motor, Pump, Compressor, Conveyor) stream five sensor channels
(temperature, vibration, current, RPM, pressure) into the system. Every reading
passes through a real scikit-learn pipeline that produces a **0-100 health index**,
a **fault classification**, a **remaining useful life estimate**, and a
human-readable explanation such as *"Vibration 3.2σ above normal"*.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Feature tour](#feature-tour)
3. [Architecture](#architecture)
4. [OOP concepts → where they live](#oop-concepts--where-they-live)
5. [How the machine learning works](#how-the-machine-learning-works)
6. [API reference](#api-reference)
7. [Project layout](#project-layout)
8. [Running the tests](#running-the-tests)
9. [Deployment](#deployment)
10. [Connecting a real ESP32](#connecting-a-real-esp32)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --reload
```

Open **<http://localhost:8000>**.

| URL | What it is |
|-----|-----------|
| `http://localhost:8000/` | The live dashboard |
| `http://localhost:8000/docs` | Interactive OpenAPI documentation |
| `http://localhost:8000/redoc` | ReDoc documentation |
| `http://localhost:8000/health` | Health check (used by the cloud platforms) |

The ML models train themselves at start-up on synthetic data (about two
seconds), then the sensor simulator starts emitting a reading per machine every
two seconds.

---

## Feature tour

| # | Feature | How to see it |
|---|---------|---------------|
| 1 | **Live dashboard** | Open the root URL. Machine cards show a health ring, status colour and live sensor values; the whole page refreshes every 2 seconds. |
| 2 | **Live charts** | The *Live Telemetry* panel plots temperature, vibration or current (use the segmented control) for all four machines at once. |
| 3 | **Alerts panel** | Every anomaly raises an alert with a severity, a timestamp and a plain-English cause. |
| 4 | **Failure prediction** | Click a card → *Details* to see *Est. Failure In* (hours) and a confidence percentage. |
| 5 | **Machine detail view** | History chart, min/max/average per sensor, thresholds, and a maintenance recommendation. |
| 6 | **Demo controls** | *Inject Fault* on any card; *Pause* / *Resume* in the header to freeze the simulation. |
| 7 | **CSV export** | The *Export CSV* link in the footer, or `GET /api/export/csv`. |
| 8 | **Dark / light mode** | The moon button in the header. The choice is remembered in `localStorage`. |

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │            Browser (vanilla JS + Chart.js)    │
                          │   Live cards · Alerts · Detail · CSV export │
                          └───────────────────┬──────────────────────────┘
                                              │  REST / JSON  (poll every 2s)
                                              │  POST /api/readings  (ESP32)
  ┌───────────────────────────────────────────▼──────────────────────────────────┐
  │                                   API LAYER                                 │
  │  app/api/machines.py   app/api/alerts.py   app/api/export.py   app/api/health │
  │  Pydantic validation · FastAPI dependency injection · exception handlers    │
  └───────────────────────────────────────────┬──────────────────────────────────┘
                                              │
  ┌───────────────────────────────────────────▼──────────────────────────────────┐
  │                               SERVICE LAYER                                  │
  │                                                                          │
  │   MachineService ──────────┬── AlertService (Subject)                      │
  │   (orchestrator)           │      ├── AlertStore   (Observer)              │
  │        │                   │      └── EventLogger  (Observer)              │
  │        │                   │                                              │
  │        └── SensorSimulator (async IoT generator, 2s tick)                   │
  └──────┬────────────────────┬──────────────────────────┬────────────────────┘
         │                    │                          │
  ┌──────▼──────┐   ┌─────────▼─────────┐   ┌────────────▼───────────────┐
  │ DOMAIN      │   │     ML LAYER      │   │        CORE LAYER           │
  │ app/machine │   │ IsolationForest   │   │ Settings · Custom          │
  │  Machine    │   │ RandomForest      │   │ exceptions · Observer      │
  │  (ABC)      │   │ Linear RUL trend  │   │ primitives                 │
  │  Motor      │   │ HealthIndex       │   │                            │
  │  Pump       │   │  Calculator       │   │                            │
  │  Compressor │   │ ModelTrainer      │   │                            │
  │  Conveyor   │   │ (synthetic data)  │   │                            │
  │ app/machine_│   └───────────────────┘   └────────────────────────────┘
  │  factory.py │
  └─────────────┘
```

**Layering rule:** `api → services → ml / models / core`. The API layer never
touches the ML models directly; it calls the service, and the service calls the
models. This keeps the ML pipeline swappable without rewriting a single route.

### Data flow of a single reading

```
SensorSimulator ──► SensorReading (validated)
                            │
                            ▼
                 MachineService.process_reading()
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  machine.compute_health  IsolationForest   RandomForest
  (per-machine ABC)      .detect_for()      .predict()
        │                   │                    │
        └───────────────────┴────────────────────┘
                            ▼
                   HealthIndexCalculator
                   (blends rule + ML signals)
                            │
                            ▼
                 HealthPrediction{health_index, status,
                                  fault_type, RUL, confidence, cause}
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          AlertService.publish()  stored on the machine
          → Observer subscribers
```

---

## OOP concepts → where they live

| OOP concept | Where it is implemented | File | Concrete example |
|-------------|------------------------|------|------------------|
| **Abstraction** | `Machine` is an abstract base class (`ABC`) with `@abstractmethod` members | `app/machine.py` | `Machine.thresholds()`, `Machine.compute_health()` |
| **Inheritance** | `Motor`, `Pump`, `Compressor`, `Conveyor` extend `Machine` | `app/machine.py` | `class Motor(Machine)` |
| **Polymorphism** | Each subclass overrides `thresholds()` and `compute_health()`; the service calls them through the base-class reference | `app/machine.py` | A Pump weights pressure 25%, a Conveyor weights vibration 40% |
| **Encapsulation** | Private attributes (`_id`, `_readings`, `_thresholds`) exposed through read-only `@property` accessors; validated data held in Pydantic models | `app/machine.py`, `app/models/readings.py` | `machine.id`, `machine.readings` |
| **Custom exceptions** | A hierarchy rooted at `MachineHealthError`, each with an HTTP status and a stable error code | `app/core/exceptions.py` | `MachineNotFoundError` → 404 `machine_not_found` |
| **Exception handlers** | FastAPI handlers translate domain errors into a consistent JSON envelope | `app/core/exceptions.py`, `app/main.py` | `machine_health_exception_handler` |
| **Factory** | `MachineFactory` maps a machine type to its builder; `create_default_fleet()` builds the fleet | `app/machine_factory.py` | `MachineFactory.create(MachineType.PUMP, "pump-01")` |
| **Strategy** | `Detector` and `Predictor` interfaces; the ML components are interchangeable strategies | `app/ml/detector_interface.py` | Swap `IsolationForestDetector` for any other `Detector` |
| **Observer** | `AlertService` is the subject, `AlertStore` and `EventLogger` are the observers | `app/core/observer.py`, `app/services/alert_service.py` | `alert_service.subscribe(MyNotifier())` |
| **Repository-style encapsulation** | `MachineService` owns the fleet and is the only way in | `app/services/machine_service.py` | `service.get_machine("motor-01")` |

### Where each pattern is *demonstrated* in the code

**Abstraction + Inheritance + Polymorphism**

```python
# app/machine.py
class Machine(ABC):
    @abstractmethod
    def thresholds(self) -> dict[str, float]: ...
    @abstractmethod
    def compute_health(self, reading: SensorReading) -> float: ...

class Pump(Machine):
    def compute_health(self, reading: SensorReading) -> float:
        # A Pump is judged by its own pressure envelope.
        ...
```

The service never branches on machine type:

```python
# app/services/machine_service.py
prediction = self._calculator.compute(machine, reading, history)
#      ^^^^^^^ any Machine subclass works here unchanged
```

**Factory**

```python
# app/machine_factory.py
MachineFactory.create(machine_type=MachineType.PUMP, machine_id="pump-01")
```

**Strategy**

```python
# app/ml/detector_interface.py
class Detector(ABC):
    @abstractmethod
    def detect(self, features: np.ndarray) -> tuple[bool, float, float]: ...
    @abstractmethod
    def train(self, healthy_samples: np.ndarray) -> None: ...
```

**Observer**

```python
# app/services/alert_service.py
alerts = AlertService()
alerts.subscribe(AlertStore())    # built in
alerts.subscribe(EventLogger())  # built in
alerts.subscribe(MyEmailNotifier())  # add your own
alerts.publish("motor-01", AlertSeverity.CRITICAL, "Vibration 3.2σ above normal")
```

---

## How the machine learning works

### 1. Anomaly detection — IsolationForest

`app/ml/anomaly_detector.py`

IsolationForest isolates observations by randomly selecting a feature and a
split point, repeating until every point is isolated. Outliers need fewer
splits, so they end up with a shorter average path length. That path length
becomes the anomaly score.

**Design decisions worth explaining:**

- **One forest per machine type.** A Motor and a Compressor have completely
  different operating envelopes. A single pooled forest would judge a
  perfectly healthy Compressor (2000 RPM, 33 A) as anomalous simply because it
  is far from a Motor's nominal point. `train_per_type()` fits one forest per
  machine type, so each machine is compared against *its own* baseline.
- **A widened healthy envelope.** The training set deliberately includes
  mildly degraded-but-still-operating samples. Without them the forest learns
  a very tight boundary and extreme readings can land back *inside* the
  learned region, which breaks monotonicity: a reading at 160 °C would score
  as "less anomalous" than one at 105 °C. Widening the envelope fixes this.
- **A continuous severity score.** IsolationForest's native score saturates —
  every far-outlier gets the same number, so it cannot express *how* far a
  machine has degraded. The reported score is therefore the normalised
  distance from the learned healthy envelope (1.0 = the edge of normal
  operation), while the IsolationForest itself provides the anomaly decision.

**Measured behaviour:** false-positive rate on healthy data ≈ 2%, while every
injected fault (overheating, bearing wear, overload, imbalance) is detected.

### 2. Fault classification — RandomForestClassifier

`app/ml/fault_classifier.py`

Five classes: `Normal`, `Overheating`, `Bearing Wear`, `Overload`, `Imbalance`.

A RandomForest is used because it needs no feature scaling, handles the
non-linear interaction between channels (high current *and* high vibration)
naturally, and returns calibrated class probabilities — which the UI surfaces
as the confidence percentage.

The training set includes healthy samples labelled `Normal`. Without a genuine
normal class the classifier must always pick the closest fault mode, so a
perfectly healthy machine would be reported as "Overheating" with 66%
confidence. Adding the normal class fixes that.

Each fault has a characteristic signature in the synthetic data, for example:

| Fault | Signature |
|-------|-----------|
| Overheating | temperature ×1.32, current ×1.18 |
| Bearing Wear | vibration ×2.05, temperature ×1.12 |
| Overload | current ×1.62, pressure ×1.30, RPM ×0.90 |
| Imbalance | vibration ×1.85, RPM ×1.10 |

### 3. Remaining Useful Life — linear trend regression

`app/ml/rul_predictor.py`

A machine that is degrading follows a roughly linear health trajectory. We fit
an ordinary least-squares line to the recent health-index history:

```
health(t) = slope · t + intercept
```

and extrapolate to the failure threshold (health index 10) to find how many
readings remain before failure. That count is converted into hours.

The maths is deliberately kept explicit and free of scikit-learn's
higher-level estimators, so it can be explained line by line.

The **confidence** blends two factors: how much history has accumulated
(saturating at 30 points) and how steeply the trend is falling.

### 4. The Health Index

`app/ml/health_index.py`

The health index blends the rule-based and ML signals:

```
final = 0.70 × rule_based_score + 0.30 × ml_adjusted_score
```

- **Rule-based score** — `machine.compute_health()`. Each machine measures how
  far each channel has travelled from its *nominal* point towards its
  *critical limit*, then applies a machine-specific weighting (a Pump cares
  about pressure, a Conveyor about vibration). The curve is anchored so that
  nominal → 100, at the limit → ~50, and well past the limit → near 0.
- **ML adjustment** — a deduction of up to 35 points when the IsolationForest
  flags the reading, scaled by how far outside the envelope it sits, plus up to
  25 points when the classifier is confident about a non-normal fault.

The rule-based term is deliberately dominant: it is stable and explainable,
while the ML terms add sensitivity to subtle degradation that has not yet
crossed a threshold.

**Status bands** (configurable in `app/core/config.py`):

| Health index | Status |
|--------------|--------|
| > 60 | `HEALTHY` |
| 30 – 60 | `WARNING` |
| ≤ 30 | `CRITICAL` |

### 5. The training data

`app/ml/trainer.py`

Everything is generated synthetically at start-up — no external dataset, no
download, no licensing question — while still exercising the real
scikit-learn estimators. The generator produces:

- **Healthy samples** — Gaussian noise around each machine type's nominal
  operating point.
- **Fault samples** — the four fault signatures above, plus a `Normal` class.
- **Degradation trajectories** — linearly declining health curves for
  validating the RUL regression.

A fixed random seed makes the training reproducible.

---

## API reference

All responses use a consistent envelope. Errors look like:

```json
{
  "success": false,
  "error": {
    "code": "machine_not_found",
    "message": "Machine 'ghost' does not exist.",
    "context": { "machine_id": "ghost" }
  }
}
```

| Method | Path | Description | Query parameters |
|--------|------|-------------|------------------|
| `GET` | `/health` | Service health, simulator state, machine count | — |
| `GET` | `/api/machines` | Summary of every machine | — |
| `GET` | `/api/machines/{machine_id}` | Full detail for one machine | — |
| `GET` | `/api/machines/{machine_id}/readings` | Recent readings | `limit` (1-500, default 100) |
| `POST` | `/api/readings` | Ingest one reading (the ESP32 endpoint) | — |
| `GET` | `/api/alerts` | Alert history, newest first | `limit` (1-200, default 50), `machine_id`, `severity` |
| `POST` | `/api/alerts/{alert_id}/acknowledge` | Acknowledge an alert | — |
| `POST` | `/api/machines/{machine_id}/inject-fault` | Inject a sudden fault (demo) | `fault_type` |
| `GET` | `/api/export/csv` | Download the readings as CSV | `machine_id`, `limit` |
| `POST` | `/api/simulation/pause` | Pause the simulator | — |
| `POST` | `/api/simulation/resume` | Resume the simulator | — |

### `POST /api/readings`

```bash
curl -X POST http://localhost:8000/api/readings \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "motor-01",
    "machine_type": "Motor",
    "temperature": 62.5,
    "vibration": 3.4,
    "current": 22.1,
    "rpm": 1452.0,
    "pressure": 5.1
  }'
```

Response (201):

```json
{
  "health_index": 98.4,
  "status": "HEALTHY",
  "anomaly_detection": { "is_anomaly": false, "score": 0.31, "threshold": 1.0 },
  "fault_classification": {
    "fault_type": "Normal",
    "confidence": 0.97,
    "probabilities": { "Normal": 0.97, "Overheating": 0.01, "...": 0.0 }
  },
  "remaining_useful_life": 720.0,
  "confidence": 0.62,
  "cause": "All parameters within normal operating range"
}
```

### `POST /api/machines/{machine_id}/inject-fault`

`fault_type` is one of `random`, `overheating`, `bearing_wear`, `overload`,
`imbalance`.

---

## Project layout

```
machine-health-monitoring/
├── app/
│   ├── main.py                     # FastAPI app, lifespan, routers, static mount
│   ├── machine.py                  # Machine ABC + Motor/Pump/Compressor/Conveyor
│   ├── machine_factory.py          # MachineFactory (Factory pattern)
│   ├── core/
│   │   ├── config.py               # Settings from environment variables
│   │   ├── exceptions.py           # Custom exception hierarchy + handlers
│   │   └── observer.py             # Observer / Subject primitives
│   ├── models/
│   │   └── readings.py             # Pydantic schemas
│   ├── ml/
│   │   ├── detector_interface.py   # Detector / Predictor ABCs (Strategy)
│   │   ├── anomaly_detector.py     # IsolationForest
│   │   ├── fault_classifier.py     # RandomForestClassifier
│   │   ├── rul_predictor.py        # Linear RUL regression
│   │   ├── health_index.py         # Blends rule + ML into the health index
│   │   └── trainer.py              # Synthetic data + training pipeline
│   ├── services/
│   │   ├── machine_service.py      # Orchestrator
│   │   ├── sensor_simulator.py     # Async IoT generator
│   │   └── alert_service.py        # Observer subject + observers
│   └── api/
│       ├── dependencies.py         # FastAPI dependency providers
│       ├── machines.py             # Machine + reading routes
│       ├── alerts.py               # Alert routes
│       ├── export.py               # CSV export
│       └── health.py               # /health + simulation controls
├── static/
│   ├── index.html                  # Single-page dashboard
│   ├── style.css                   # Glassmorphism, dark/light, responsive
│   ├── app.js                      # Vanilla JS + Chart.js
│   └── favicon.svg
├── iot/
│   └── esp32_sketch.ino            # ESP32 client for real hardware
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_detector.py            # Anomaly detector tests
│   ├── test_rul.py                 # RUL predictor + classifier tests
│   └── test_api.py                 # API tests
├── docs/
│   └── VIVA_QA.md                 # 20 viva questions with answers
├── requirements.txt
├── Dockerfile
├── render.yaml
├── pytest.ini
└── README.md
```

---

## Running the tests

```bash
pytest                      # the whole suite
pytest tests/test_api.py    # just the API tests
pytest -v                   # verbose
```

45 tests cover the detector, the RUL predictor, the fault classifier and every
API endpoint (including the 404 and 422 error paths).

```
tests/test_detector.py   10 tests   anomaly detection
tests/test_rul.py        15 tests   RUL predictor + fault classifier
tests/test_api.py        20 tests   every endpoint
```

---

## Deployment

### Docker

```bash
docker build -t machine-health-monitoring .
docker run -p 8000:8000 machine-health-monitoring
```

The image is `python:3.11-slim`, runs as a non-root user, has a `HEALTHCHECK`
and reads `PORT` from the environment.

### Render (one click)

`render.yaml` is a Render Blueprint:

1. Push the repository to GitHub.
2. In Render choose **New → Blueprint** and select the repository.
3. Render reads `render.yaml`, builds the Dockerfile and deploys.

The service is then live at `https://<your-service>.onrender.com`.

### Railway

1. **New Project → Deploy from GitHub repo**.
2. Railway detects the `Dockerfile` automatically.
3. Under **Settings → Healthcheck Path**, set `/health`.

### Hugging Face Spaces

1. Create a **Docker Space**.
2. Push the repository (Spaces reads the `Dockerfile` directly).
3. Set the port variable to `7860` in the Space's **Variables** tab, and add
   `PORT=7860` — the Dockerfile already reads `PORT`.

### Any other platform

The only requirement is a Python 3.11 runtime and a start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## Connecting a real ESP32

`iot/esp32_sketch.ino` is a complete, commented Arduino sketch that reads five
sensors and POSTs them to `POST /api/readings` every two seconds.

**Wiring**

| Sensor | ESP32 GPIO | Type | Notes |
|--------|-----------|------|-------|
| DS18B20 (temperature) | 4 | OneWire | 4.7 kΩ pull-up to 3V3 |
| MPU6050 (vibration) | SDA 21 / SCL 22 | I²C | Accel magnitude in g → mm/s |
| ACS712 (current) | 34 | Analog (ADC1) | Zero-offset calibrated at boot |
| Potentiometer (RPM) | 35 | Analog (ADC1) | Stands in for an RPM tachometer |
| Pressure transducer | 32 | Analog (ADC1) | 0-5 V → 0-16 bar |

Only ADC1 pins (GPIO 32-39) are used, because ADC2 is unavailable while the
WiFi driver is active.

**Before uploading, edit three constants:**

```cpp
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL   = "http://YOUR-SERVER-IP:8000";
```

`MACHINE_ID` must match a machine the backend knows about (`motor-01`,
`pump-01`, `compressor-01`, `conveyor-01`).

**Required Arduino libraries** (Library Manager): `OneWire`, `DallasTemperature`,
`Adafruit MPU6050`, `Adafruit Unified Sensor`.

The readings flow through exactly the same ML pipeline as the simulated ones.

---

## Configuration

Every setting is an environment variable, read in `app/core/config.py`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `PORT` | `8000` | The port the server listens on |
| `HOST` | `0.0.0.0` | The interface to bind to |
| `ENABLE_SIMULATION` | `true` | Start the background simulator |
| `SIMULATION_INTERVAL` | `2.0` | Seconds between simulated readings |
| `READINGS_BUFFER_SIZE` | `500` | Readings retained per machine |
| `RANDOM_SEED` | `42` | Seed for reproducible training |
| `TRAIN_SAMPLES` | `1200` | Synthetic samples per machine type |
| `ANOMALY_THRESHOLD` | `0.02` | IsolationForest contamination rate |
| `HEALTH_WARNING` | `60.0` | Health index at or below which a machine is WARNING |
| `HEALTH_CRITICAL` | `30.0` | Health index at or below which a machine is CRITICAL |
| `SQLITE_PATH` | *(unset)* | Optional path for a future persistent store |

Example:

```bash
ENABLE_SIMULATION=false SIMULATION_INTERVAL=1.0 uvicorn app.main:app --reload
```

---

## Troubleshooting

**The dashboard loads but the charts are empty.**
The page polls every two seconds; give it a few seconds to gather points. If
the server was started with `ENABLE_SIMULATION=false` and no ESP32 is posting,
there is no data to plot.

**"Machine 'x' does not exist" (404) when posting a reading.**
The `machine_id` must be one of the four known machines. A real ESP32 should
use `motor-01`, `pump-01`, `compressor-01` or `conveyor-01`.

**`ModuleNotFoundError: No module named 'app'`.**
Run uvicorn from the project root (the directory containing `app/`), or add the
project root to `PYTHONPATH`.

**Everything is CRITICAL immediately.**
The health bands are relative to each machine's own thresholds. Check
`HEALTH_WARNING` and `HEALTH_CRITICAL` in `app/core/config.py` if you changed
the machine set.

**Port 8000 is already in use.**
`uvicorn app.main:app --port 8001`, or set `PORT=8001`.

---

## Licence

MIT — use it freely for coursework, portfolios and demos.
