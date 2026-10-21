# VIVA Q&A — Machine Health Monitoring System using IoT and ML

Twenty likely examiner questions with concise, defensible answers. Each answer
is written to be spoken aloud in 30–60 seconds, and every claim points at real
code in this project.

---

## 1. What is this project, in one sentence?

A web application that ingests IoT sensor data from four industrial machines,
uses machine learning to detect anomalies, classify faults and predict
remaining useful life, and presents the result on a live dashboard.

---

## 2. What is the tech stack and why?

| Layer | Choice | Reason |
|-------|--------|--------|
| Backend | Python 3.11 + FastAPI | Async, typed, automatic OpenAPI docs, Pydantic validation |
| Server | Uvicorn | ASGI server that FastAPI runs on |
| ML | scikit-learn | Battle-tested estimators, no GPU needed, easy to explain |
| Frontend | HTML + CSS + vanilla JS + Chart.js (CDN) | No Node build step — the reviewer can just run `uvicorn` and it works |
| Storage | In-memory `deque` per machine | Zero setup, bounded memory, appropriate for a live demo |

The deliberate choice is **no frontend build step**. A `npm install` failure
cannot stop the demo.

---

## 3. Explain the overall architecture.

Four layers, strictly one-directional: `api → services → ml / models / core`.

- **API layer** (`app/api/`) — HTTP routes, request validation, error translation.
- **Service layer** (`app/services/`) — business logic: `MachineService` orchestrates, `SensorSimulator` generates data, `AlertService` publishes alerts.
- **ML layer** (`app/ml/`) — the models, the trainer and the health-index combiner.
- **Core layer** (`app/core/`) — settings, custom exceptions, the Observer primitives.

The API layer never imports an ML model directly. That is what makes the models
swappable.

---

## 4. Which OOP concepts did you use, and where?

| Concept | File | How |
|---------|------|-----|
| Abstraction | `app/machine.py` | `Machine(ABC)` with two `@abstractmethod`s |
| Inheritance | `app/machine.py` | `Motor`, `Pump`, `Compressor`, `Conveyor` extend `Machine` |
| Polymorphism | `app/machine.py` | Each subclass overrides `thresholds()` and `compute_health()` |
| Encapsulation | `app/machine.py`, `app/models/readings.py` | Private `_id`/`_readings` behind `@property`; Pydantic models validate data |
| Factory | `app/machine_factory.py` | `MachineFactory.create(type, id)` |
| Strategy | `app/ml/detector_interface.py` | `Detector` and `Predictor` ABCs |
| Observer | `app/core/observer.py`, `app/services/alert_service.py` | `AlertService` (subject) → `AlertStore`, `EventLogger` (observers) |

---

## 5. What is the Factory pattern, and where do you use it?

`MachineFactory` keeps a registry mapping a `MachineType` to its builder. The
service and API never write `if machine_type == MOTOR: return Motor(...)`;
they call `MachineFactory.create(machine_type, machine_id)`.

Adding a fifth machine type means editing **one** file — the factory registry
and the `MachineType` enum — and nothing else changes.

`create_default_fleet()` uses it to build the four-machine fleet.

---

## 6. What is the Strategy pattern here, and why does it matter?

The anomaly detector and the fault classifier are behind the `Detector` and
`Predictor` ABCs. The `HealthIndexCalculator` holds a `Detector` reference and
calls `detect()` — it has no idea which implementation it holds.

**Why it matters:** I could swap IsolationForest for a One-Class SVM, or a
simple z-score rule for a fast test, without touching a single route or the
service layer. The same applies to the classifier.

---

## 7. What is the Observer pattern, and how does it help?

`AlertService` extends `Subject`. It keeps a list of observers and calls
`notify(alert)` when an alert is published.

Two observers ship with it:
- `AlertStore` — retains the alerts in memory so the API can serve them.
- `EventLogger` — writes the alert to the application log.

**The key design point:** the `notify()` loop isolates each observer in a
try/except. If a subscriber raises, the remaining subscribers still get the
event — one broken notifier must not take down alerting.

To add e-mail or webhook notification, I subclass `Observer` and call
`alerts.subscribe(MyNotifier())`. No change to the service.

---

## 8. How does the health index actually work?

Two terms, blended 70/30:

```
health = 0.70 × rule_based + 0.30 × ml_adjusted
```

**Rule-based (70%)** — `machine.compute_health()`. For each sensor I measure
how far the reading has moved from the machine's *nominal* value towards its
*critical limit*, expressed as a fraction of that span. The channels are
weighted per machine type (a Pump weights pressure 25%, a Conveyor weights
vibration 40%), and the weighted penalty is mapped through a curve anchored so
that nominal → 100, at the limit → ~50, and far past it → near 0.

**ML-adjusted (30%)** — a deduction of up to 35 points when the IsolationForest
flags the reading (scaled by how far outside the envelope it sits), plus up to
25 points when the classifier is confident about a non-normal fault.

The rule-based term is dominant on purpose: it is stable and explainable,
while the ML terms add sensitivity to degradation that has not yet crossed a
threshold.

---

## 9. What is a health index of 0–100, and what are the bands?

A single number summarising how close a machine is to failure, derived from the
five sensor channels.

| Band | Range | Meaning |
|------|-------|---------|
| HEALTHY | > 60 | Operating normally, no action needed |
| WARNING | 30–60 | Degradation detected, schedule maintenance |
| CRITICAL | ≤ 30 | Failure imminent, act now |

The bands live in `app/core/config.py` as `HEALTH_WARNING` and
`HEALTH_CRITICAL`, so they are configurable without touching logic.

---

## 10. Why IsolationForest for anomaly detection?

Three reasons:

1. **It is unsupervised.** I have no labels for "anomaly" — only healthy
   operating data. IsolationForest learns the shape of normal and flags
   anything outside it.
2. **It works on moderate sample counts.** My synthetic healthy set is ~1200
   samples; a density-based method like KDE would need far more.
3. **It is fast and multi-dimensional.** It handles the correlations between
   temperature, vibration and current natively, and with 300 trees it scores a
   reading in well under a millisecond.

**The mechanism:** the algorithm randomly picks a feature and a split point,
recursing until each point is isolated. Outliers get isolated in fewer splits,
so they have a shorter average path length — that path length *is* the anomaly
score.

---

## 11. You use one IsolationForest. Why per machine type?

Because a Motor and a Compressor have completely different operating
envelopes. A pooled forest would see a perfectly healthy Compressor
(2000 RPM, 33 A) as far from a Motor's nominal point (1450 RPM, 22 A) and
flag it as an anomaly. That is a false positive caused purely by pooling.

So `train_per_type()` fits **one forest per machine type**, and each reading is
scored against its own type's envelope via `detect_for(reading, machine_type)`.

The measured false-positive rate on healthy data is about 2%, and every
injected fault is still detected.

---

## 12. What is the "widened healthy envelope", and why did you do it?

I found a real bug while testing: the anomaly score was **not monotonic**. A
reading at 160 °C scored as *less* anomalous than one at 105 °C.

The cause: the forest had learned a very tight boundary around the healthy
centre, so extreme readings landed back *inside* the learned region. The
forest was answering "is this point in the tight cluster?" rather than "how far
outside normal is this?".

The fix is to train on a deliberately widened envelope — I take the healthy
samples and add copies of them scaled towards the degraded end (temperature
×1.18, vibration ×1.35, and so on). The forest now learns the *edges* of
normal operation, and the score increases monotonically with distance.

I also report a continuous severity score (the normalised distance from the
envelope) rather than IsolationForest's native score, because the native score
saturates — every far-outlier gets the same number, so it cannot express *how
far* a machine has degraded. The IsolationForest still provides the decision.

---

## 13. Why RandomForestClassifier for fault type, and where does the "Normal" class come from?

A RandomForest because it needs no feature scaling, handles the non-linear
interaction between channels (high current *and* high vibration) well, and —
importantly — returns **calibrated class probabilities**, which I surface in the
UI as the confidence percentage.

The `Normal` class is worth calling out because I hit a bug here. I originally
trained on fault samples only, so the classifier had only four classes and
*had* to pick the closest fault mode. A perfectly healthy motor was reported
as **"Overheating, 66% confidence"**. The fix was to add healthy samples
labelled `Normal` to the training set. Now a healthy motor correctly reports
`Normal` at 97% confidence, and the fault modes still classify at 94–100%.

---

## 14. How is Remaining Useful Life calculated?

A degrading machine follows a roughly linear health trajectory, so I fit an
ordinary least-squares line to the recent health-index history:

```
health(t) = slope · t + intercept
```

I then extrapolate to the failure threshold (health index 10) to find how many
readings remain, and convert that into hours.

The maths is deliberately kept explicit — `numpy.polyfit` rather than a
scikit-learn estimator — so it can be explained line by line.

Two safety clamps: if the slope is flat or positive (the machine is not
degrading), RUL is reported as the nominal 720 hours; and RUL is never negative
or above the maximum.

The **confidence** blends two factors: how much history has accumulated
(saturating at 30 points) and how steeply the trend is falling.

---

## 15. Where does the training data come from? Is it real?

Synthetic, generated at start-up by `ModelTrainer`. There is no external
dataset — no download, no licensing question, and it works offline.

The generator produces Gaussian noise around each machine type's nominal
operating point for the healthy class, and characteristic signatures for the
four faults (for example Overheating multiplies temperature by 1.32 and current
by 1.18). A fixed random seed (`RANDOM_SEED=42`) makes it reproducible.

**The important point:** the models themselves are genuine scikit-learn
estimators, fitted with `fit()` on real numpy arrays. Only the *data* is
synthetic. Swapping in a real dataset from a factory would be a one-line change
in `ModelTrainer.build_artifacts()`.

---

## 16. How does the ESP32 send data, and what happens to it?

The ESP32 posts a JSON body to `POST /api/readings`:

```json
{
  "machine_id": "motor-01",
  "machine_type": "Motor",
  "temperature": 62.5,
  "vibration": 3.4,
  "current": 22.1,
  "rpm": 1452.0,
  "pressure": 5.1
}
```

FastAPI validates it against the `SensorReading` Pydantic model, then
`MachineService.process_reading()` runs the **exact same ML pipeline** as the
simulator. The response is the full health prediction, so the device could
react to it directly.

The sketch in `iot/esp32_sketch.ino` reads a DS18B20 (temperature), an MPU6050
(vibration), an ACS712 (current), a potentiometer (RPM) and a pressure
transducer, smooths the values with an exponential filter, and posts every two
seconds. I used ADC1 pins only, because ADC2 is unavailable while the WiFi
driver is active.

---

## 17. How does the dashboard get live updates? Why polling instead of WebSockets?

The frontend calls `GET /api/machines` and `GET /api/alerts` every two seconds
with `setInterval`, matching the simulator's tick rate.

I chose polling deliberately:
- It is simpler to debug and to demonstrate.
- It works identically behind a proxy, a CDN, or on a free-tier platform that
  buffers WebSocket connections.
- At four machines and a two-second cadence the load is trivial.

The data is already in memory, so a poll is just a dict lookup plus a JSON
serialise — microseconds of work. The trade-off is up to two seconds of latency,
which is irrelevant for a monitoring dashboard.

---

## 18. What are the custom exceptions, and why not just let errors bubble up?

I defined a hierarchy rooted at `MachineHealthError` in `app/core/exceptions.py`.
Each subclass carries an HTTP status code and a stable machine-readable code:

| Exception | Status | Code |
|-----------|--------|------|
| `MachineNotFoundError` | 404 | `machine_not_found` |
| `InvalidReadingError` | 422 | `invalid_reading` |
| `ModelNotTrainedError` | 503 | `model_not_trained` |
| `SimulationStateError` | 409 | `simulation_state_error` |
| `DuplicateSubscriberError` | 409 | `duplicate_subscriber` |

A FastAPI exception handler converts them into a consistent envelope:

```json
{ "success": false, "error": { "code": "machine_not_found", "message": "..." } }
```

**Why:** without this, a missing machine would produce FastAPI's default
`{"detail": "..."}` shape, while a validation failure would produce a
completely different shape. The client would need two different parsers. Now
every error has the same structure, and the error `code` is stable enough to
branch on in the frontend.

---

## 19. How is the data stored? What happens on restart?

In memory, in a `collections.deque` with `maxlen=500` per machine. The
`deque` is the right structure because it gives O(1) appends *and* O(1) popleft
from the other end, so trimming the oldest reading is free.

The 500-reading cap is what stops the simulator from leaking memory during a
long demo — at one reading per two seconds, 500 readings is about 17 minutes of
history per machine.

Storage is intentionally in-memory for this version: the demo needs no setup
and no database. The architecture supports adding SQLite later —
`settings.sqlite_path` is already wired up, and the service is the only layer
that touches the buffers, so a repository implementation would slot in without
touching the API.

**Trade-off:** restarting the process loses the history. That is acceptable
here because the ML models retrain automatically on synthetic data at start-up,
so the system is immediately useful again.

---

## 20. How would you scale this to a real factory, and what are the limitations?

**Scaling path:**
1. Replace the in-memory `deque` with TimescaleDB or InfluxDB for time-series
   storage.
2. Run the simulator per-cell as a separate process, or point the ingest
   endpoint at a message broker (MQTT/Kafka) so bursts are absorbed.
3. Move model training into a background job with periodic retraining, and
   cache the model artefacts with `joblib` instead of fitting at boot.
4. Add per-machine authentication (JWT) so an ESP32 can only post to its own
   machine id.

**Honest limitations of this version:**
- Storage is in-memory, so history is lost on restart.
- The models are trained on synthetic data. They will detect the fault modes
  I encoded, but on a real machine the thresholds would need to be
  recalibrated from real failure history.
- The RUL estimate assumes roughly linear degradation. Real machines often
  degrade non-linearly or fail abruptly, so the estimate degrades in accuracy
  the closer the machine gets to failure.
- The IsolationForest assumes the healthy distribution is roughly stationary.
  A machine that is slowly degrading shifts that distribution, so the model
  should be re-fitted periodically.
- There is no authentication on the API.

---

## Bonus: likely follow-up questions

**"Why is the health index 70% rule-based? Isn't that just hard-coded rules
with ML on top?"**
The rule-based term is what makes the score *explainable* — a technician can
see "the pump's pressure hit its limit" rather than an opaque number. The ML
term is 30% because it adds genuine sensitivity to subtle, multi-channel
degradation that no single threshold would catch. I would not trust a pure
black-box health score in a maintenance decision without being able to explain
it.

**"How do you know the anomaly detector actually works?"**
The test suite measures it: `test_healthy_reading_is_not_flagged`,
`test_severe_overheating_is_flagged`, `test_severe_bearing_wear_is_flagged`, and
`test_anomaly_score_is_higher_for_worse_readings` — which asserts the
monotonicity that the widening fix restored. I also measured the false-positive
rate on 300 healthy samples directly.

**"What if two machines are at the same health index — is that a problem?"**
No. The index is a summary, and the underlying channels are always available
per machine in the detail view, which is what a technician actually acts on.

**"Why no database?"**
The demo goal is "run one command and it works". SQLite would work fine but
adds a file and a schema to reason about, and the in-memory `deque` is
genuinely the right structure for a fixed-size rolling window. The
`SQLITE_PATH` setting is already in place for when persistence is actually
needed.

**"How would you test this in production?"**
The `POST /api/readings` endpoint is the integration point. I would replay
recorded historian data through it and compare the system's alerts against the
maintenance log. The current test suite already covers the API contract
end-to-end with FastAPI's `TestClient`, including the 404 and 422 paths.
