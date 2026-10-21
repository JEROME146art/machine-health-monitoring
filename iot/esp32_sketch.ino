/* ==========================================================================
   Machine Health Monitoring · ESP32 IoT Client
   --------------------------------------------------------------------------
   Reads five sensors from an ESP32 and POSTs them to the FastAPI backend
   every 2 seconds over plain HTTP.

   WIRING (see the table below)
   --------------------------------------------------------------------------
   | Sensor          | ESP32 GPIO | Type              | Notes                        |
   |-----------------|------------|-------------------|------------------------------|
   | DS18B20 (temp)  | GPIO 4     | OneWire           | 4.7k pull-up to 3V3          |
   | MPU6050 (vib)   | SDA 21/22  | I2C               | Accel magnitude in g -> mm/s |
   | ACS712 (current)| GPIO 34    | Analog (ADC1)     | Calibrate offset at startup  |
   | Potentiometer   | GPIO 35    | Analog (ADC1)    | Stands in for RPM           |
   | Pressure (bar)  | GPIO 32    | Analog (ADC1)    | 0-5V -> 0-16 bar sensor     |

   NOTE: ADC2 pins are not usable while the WiFi driver is active, so only
   ADC1 pins (GPIO 32-39) are used for the analog channels.

   SETUP
   --------------------------------------------------------------------------
   1. Install "ESP32" board support in the Arduino IDE.
   2. Install these libraries via Library Manager:
        - WiFi                (bundled)
        - HTTPClient          (bundled)
        - OneWire             (by Paul Stoffregen)
        - DallasTemperature   (by Miles Burton)
        - Adafruit MPU6050    (by Adafruit)
        - Adafruit Unified Sensor
   3. Edit the three constants below to match your network and backend.
   4. Upload and open the Serial Monitor at 115200 baud.
   ========================================================================== */

#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <math.h>

/* ----------------------------------------------------------------------
   Configuration
   ---------------------------------------------------------------------- */

// Replace with your own network credentials.
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Replace with the public URL of your deployed backend.
// For a local network use the LAN IP, e.g. "http://192.168.1.50:8000".
const char* SERVER_URL   = "http://YOUR-SERVER-IP:8000";

// The machine this device represents. Must match an id known to the server.
const char* MACHINE_ID   = "motor-01";
const char* MACHINE_TYPE = "Motor";

// How often a reading is sent, in milliseconds.
const unsigned long SEND_INTERVAL_MS = 2000;

/* ----------------------------------------------------------------------
   Pin and sensor configuration
   ---------------------------------------------------------------------- */

const int PIN_ONEWIRE   = 4;    // DS18B20 data pin
const int PIN_CURRENT  = 34;   // ACS712 analog out
const int PIN_RPM      = 35;   // Potentiometer used as an RPM proxy
const int PIN_PRESSURE = 32;   // Pressure transducer analog out

// ADC full-scale for the ESP32's 12-bit ADC.
const float ADC_MAX = 4095.0;

// ACS712 calibration.
float ACS_SENSITIVITY  = 0.185;  // Volts per ampere
float ACS_ZERO_OFFSET  = 2.5;    // Volts at zero current (calibrated at boot)
const float ACS_VCC    = 3.3;

// Potentiometer to RPM mapping.
const float RPM_MIN = 0.0;
const float RPM_MAX = 3000.0;

// Pressure transducer mapping (0-5V sensor, 0-16 bar range).
const float PRESSURE_MAX_BAR = 16.0;

/* ----------------------------------------------------------------------
   Global objects
   ---------------------------------------------------------------------- */

OneWire oneWire(PIN_ONEWIRE);
DallasTemperature sensors(&oneWire);
Adafruit_MPU6050 mpu;

// Smoothed sensor values so a single noisy ADC read does not skew the trend.
float smoothedCurrent   = 0.0;
float smoothedRpm       = 0.0;
float smoothedPressure  = 0.0;
float smoothedVibration = 0.0;
bool  calibrated        = false;
bool  imuReady          = false;

/* ----------------------------------------------------------------------
   Setup
   ---------------------------------------------------------------------- */

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("=== Machine Health Monitoring · ESP32 client ===");

    // Configure the analog inputs.
    analogReadResolution(12);
    pinMode(PIN_CURRENT, INPUT);
    pinMode(PIN_RPM, INPUT);
    pinMode(PIN_PRESSURE, INPUT);

    // Initialise the temperature sensor.
    sensors.begin();
    sensors.setResolution(12);

    // Initialise the accelerometer.
    imuReady = mpu.begin();
    if (!imuReady) {
        Serial.println("[WARN] MPU6050 not found - vibration will read 0.");
    } else {
        Serial.println("[OK]   MPU6050 initialised.");
    }

    // Connect to WiFi.
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);
    WiFi.mode(WIFI_STATION);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
        delay(250);
        Serial.print('.');
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[FAIL] WiFi connection timed out. Restarting...");
        ESP.restart();
    }

    Serial.println();
    Serial.print("[OK]   IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("[OK]   Posting to: ");
    Serial.println(SERVER_URL);
    Serial.println("==============================================");
}

/* ----------------------------------------------------------------------
   Helper: read a calibrated analog value
   ---------------------------------------------------------------------- */

float readAnalogAverage(int pin, int samples) {
    long total = 0;
    for (int i = 0; i < samples; i++) {
        total += analogRead(pin);
        delayMicroseconds(200);
    }
    return (float)total / samples;
}

/* ----------------------------------------------------------------------
   Sensor reading helpers
   ---------------------------------------------------------------------- */

float readCurrentAmps() {
    float raw = readAnalogAverage(PIN_CURRENT, 16);
    float volts = (raw / ADC_MAX) * ACS_VCC;

    // Calibrate the zero-current offset on the first reading.
    if (!calibrated) {
        ACS_ZERO_OFFSET = volts;
        calibrated = true;
        Serial.print("[CAL]  ACS712 zero offset: ");
        Serial.println(ACS_ZERO_OFFSET);
    }

    float amps = (volts - ACS_ZERO_OFFSET) / ACS_SENSITIVITY;
    if (amps < 0.0) amps = 0.0;
    return amps;
}

float readRpm() {
    float raw = readAnalogAverage(PIN_RPM, 12);
    return (raw / ADC_MAX) * (RPM_MAX - RPM_MIN) + RPM_MIN;
}

float readPressureBar() {
    float raw = readAnalogAverage(PIN_PRESSURE, 12);
    // The sensor outputs 0-5V but the ADC only sees 0-3.3V, so scale by 5/3.3.
    float volts = (raw / ADC_MAX) * 5.0;
    return (volts / 5.0) * PRESSURE_MAX_BAR;
}

float readVibration() {
    if (!imuReady) {
        return 0.0;
    }
    sensors_event_t accel, gyro, temp;
    if (!mpu.getEvent(&accel, &gyro, &temp)) {
        return 0.0;
    }
    // Magnitude of the acceleration vector minus 1g (gravity), in g.
    float ax = accel.acceleration.x;
    float ay = accel.acceleration.y;
    float az = accel.acceleration.z;
    float magnitude = sqrt(ax * ax + ay * ay + az * az);
    float vibrationG = fabs(magnitude - 1.0);
    // Convert g to mm/s using the standard 9.81 m/s^2 per g.
    return vibrationG * 9800.0;
}

float readTemperature() {
    sensors.requestTemperatures();
    float celsius = sensors.getTempCByIndex(0);
    if (celsius == DEVICE_DISCONNECTED_C) {
        return 0.0;
    }
    return celsius;
}

/* ----------------------------------------------------------------------
   Smoothing
   ---------------------------------------------------------------------- */

float smooth(float previous, float current, float alpha) {
    return previous + alpha * (current - previous);
}

/* ----------------------------------------------------------------------
   Main loop
   ---------------------------------------------------------------------- */

void loop() {
    static unsigned long lastSend = 0;
    unsigned long now = millis();

    if (now - lastSend < SEND_INTERVAL_MS) {
        delay(10);
        return;
    }
    lastSend = now;

    // Reconnect to WiFi if the link dropped.
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Reconnecting...");
        WiFi.reconnect();
        delay(2000);
        return;
    }

    // Read every sensor.
    float temperature = readTemperature();
    float vibration   = readVibration();
    float current     = readCurrentAmps();
    float rpm         = readRpm();
    float pressure    = readPressureBar();

    // Exponential smoothing keeps the signal clean without lagging.
    smoothedCurrent   = smooth(smoothedCurrent, current, 0.25);
    smoothedRpm       = smooth(smoothedRpm, rpm, 0.25);
    smoothedPressure  = smooth(smoothedPressure, pressure, 0.25);
    smoothedVibration = smooth(smoothedVibration, vibration, 0.30);

    // Build the JSON payload.
    String payload = "{";
    payload += "\"machine_id\":\""   + String(MACHINE_ID) + "\",";
    payload += "\"machine_type\":\"" + String(MACHINE_TYPE) + "\",";
    payload += "\"temperature\":"    + String(temperature, 2) + ",";
    payload += "\"vibration\":"      + String(smoothedVibration, 3) + ",";
    payload += "\"current\":"        + String(smoothedCurrent, 2) + ",";
    payload += "\"rpm\":"            + String(smoothedRpm, 0) + ",";
    payload += "\"pressure\":"       + String(smoothedPressure, 2);
    payload += "}";

    Serial.print("[POST] ");
    Serial.println(payload);

    // Send the reading.
    HTTPClient http;
    String url = String(SERVER_URL) + "/api/readings";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    int status = http.POST(payload);

    if (status > 0) {
        Serial.print("[OK]   HTTP ");
        Serial.print(status);
        Serial.print(" -> ");
        Serial.println(http.getString());
    } else {
        Serial.print("[FAIL] HTTP error: ");
        Serial.println(http.errorToString(status));
    }

    http.end();
}
