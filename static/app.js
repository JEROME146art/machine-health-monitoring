/* ==========================================================================
   APEX MONITOR · Industrial Health AI · Masterpiece Application Engine
   Multi-View SCADA Architecture, Web Audio Synthesizer, Dynamic SVG Twin,
   Machine Learning Feedback & Telemetry Stream.
   Vanilla ES6+ · Zero Build Step · Modern Web Platform APIs.
   ========================================================================== */
'use strict';

/* --------------------------------------------------------------------------
   Configuration & Constants
   -------------------------------------------------------------------------- */
const POLL_INTERVAL_MS = 2000;
const API_BASE = '';

const SENSORS = {
    temperature: { label: 'Temperature', unit: '°C', color: '#f97316', max: 120, warn: 75, crit: 90 },
    vibration: { label: 'Vibration RMS', unit: 'mm/s', color: '#a855f7', max: 15, warn: 4.5, crit: 7.1 },
    current: { label: 'Current Draw', unit: 'A', color: '#22d3ee', max: 60, warn: 30, crit: 45 },
    rpm: { label: 'Rotational Speed', unit: 'RPM', color: '#3b82f6', max: 3000, warn: 2200, crit: 2600 },
    pressure: { label: 'Pressure', unit: 'bar', color: '#ec4899', max: 20, warn: 12, crit: 16 },
    health: { label: 'Composite Health Index', unit: '%', color: '#10b981', max: 100, warn: 60, crit: 30 },
};

const MACHINE_PROFILES = {
    'motor-01': {
        name: 'Motor 01',
        type: 'Three-Phase Induction Motor',
        power: '45 kW · 400V',
        probes: [
            { id: 'p1', name: 'Drive-End Bearing', x: 28, y: 50, sensor: 'vibration', desc: 'Monitors high-frequency raceway shock and flaking.' },
            { id: 'p2', name: 'Stator Winding Core', x: 50, y: 35, sensor: 'temperature', desc: 'Detects thermal insulation degradation and phase imbalance.' },
            { id: 'p3', name: 'Shaft & Rotor Cage', x: 65, y: 50, sensor: 'rpm', desc: 'Measures slip frequency and 1X dynamic rotor unbalance.' },
            { id: 'p4', name: 'Terminal Box & Load', x: 45, y: 15, sensor: 'current', desc: 'Tracks active stator current draw and power factor flux.' },
        ],
        parts: [
            { sku: 'SKF-6208-2RS', name: 'Deep Groove Ball Bearing (Drive End)' },
            { sku: 'MOBIL-POLY-EM', name: 'High-Temperature Synthetic Polyurea Grease' },
            { sku: 'VIB-SHIM-KIT', name: 'Laser Shaft Realignment Shim Set' },
        ]
    },
    'pump-01': {
        name: 'Pump 01',
        type: 'Centrifugal Slurry Pump',
        power: '30 kW · 1450 RPM',
        probes: [
            { id: 'p1', name: 'Thrust Bearing Housing', x: 30, y: 50, sensor: 'vibration', desc: 'Detects axial thrust play and bearing raceway spalling.' },
            { id: 'p2', name: 'Volute Casing Chamber', x: 72, y: 50, sensor: 'pressure', desc: 'Monitors hydraulic discharge pressure and cavitation surge.' },
            { id: 'p3', name: 'Mechanical Seal Flush', x: 50, y: 38, sensor: 'temperature', desc: 'Tracks barrier fluid temperature and face friction.' },
            { id: 'p4', name: 'Drive Coupling Input', x: 15, y: 50, sensor: 'rpm', desc: 'Measures pump shaft rotational velocity and torque slip.' },
        ],
        parts: [
            { sku: 'FLOW-SEAL-210', name: 'Tungsten Carbide Mechanical Seal Assembly' },
            { sku: 'SKF-7309-BEP', name: 'Angular Contact Ball Bearing Pair' },
            { sku: 'EPDM-O-RING-SET', name: 'Casing Volute Elastomer O-Ring Kit' },
        ]
    },
    'compressor-01': {
        name: 'Compressor 01',
        type: 'Two-Stage Reciprocating Compressor',
        power: '55 kW · 12 bar',
        probes: [
            { id: 'p1', name: 'Cylinder Head Stage 2', x: 62, y: 25, sensor: 'temperature', desc: 'High-pressure discharge valve thermal monitoring.' },
            { id: 'p2', name: 'Crankcase Main Journal', x: 35, y: 65, sensor: 'vibration', desc: 'Monitors crosshead wrist pin impact and crank play.' },
            { id: 'p3', name: 'Discharge Manifold', x: 80, y: 40, sensor: 'pressure', desc: 'Measures interstage and discharge delivery pressure.' },
            { id: 'p4', name: 'Crankshaft Drive Pulley', x: 15, y: 55, sensor: 'rpm', desc: 'Monitors V-belt slip and rotational frequency.' },
        ],
        parts: [
            { sku: 'VALVE-PLATE-HP', name: 'Hoerbiger Concentric Reed Valve Plates' },
            { sku: 'OIL-SYNTH-ISO100', name: 'Ester-Based Synthetic Compressor Lubricant' },
            { sku: 'RING-SET-TEFLON', name: 'PTFE Self-Lubricating Piston Ring Kit' },
        ]
    },
    'conveyor-01': {
        name: 'Conveyor 01',
        type: 'Heavy Bulk Belt Conveyor',
        power: '22 kW · 900 RPM',
        probes: [
            { id: 'p1', name: 'Head Drive Pulley Pillow Block', x: 22, y: 48, sensor: 'vibration', desc: 'Tracks drive roller bearing wear and belt tension shock.' },
            { id: 'p2', name: 'Gearbox Reducer Casing', x: 42, y: 45, sensor: 'temperature', desc: 'Monitors helical gear mesh temperature and oil degradation.' },
            { id: 'p3', name: 'Tachometer Idler Roller', x: 75, y: 52, sensor: 'rpm', desc: 'Monitors linear belt surface velocity and slip detection.' },
            { id: 'p4', name: 'Drive Motor Phase Input', x: 12, y: 35, sensor: 'current', desc: 'Measures belt loading resistance and mechanical jam stall.' },
        ],
        parts: [
            { sku: 'TIMKEN-SAF-22215', name: 'Spherical Roller Pillow Block Unit' },
            { sku: 'BELT-SPLICE-HD', name: 'Mechanical Conveyor Belt Fastener Kit' },
            { sku: 'GEAR-OIL-EP320', name: 'Extreme Pressure Mineral Gear Oil (ISO VG 320)' },
        ]
    }
};

/* --------------------------------------------------------------------------
   Global Application State
   -------------------------------------------------------------------------- */
const state = {
    machines: [],
    alerts: [],
    details: {},
    mlMetrics: null,
    activeView: 'fleet',
    activeSensor: 'temperature',
    chartPoints: 30,
    selectedTwinId: 'motor-01',
    selectedProbeId: 'p1',
    simulationRunning: true,
    soundEnabled: true,
    terminalPaused: false,
    pollTimer: null,
    liveSeries: {},
    liveChart: null,
    twinHistoryChart: null,
    featureImportanceChart: null,
    filterSeverity: 'all',
    filterMachine: 'all',
    chaosHistory: [],
    terminalBuffer: [],
};

/* --------------------------------------------------------------------------
   Web Audio API Synthesizer (Zero-Dependency Audio Chime System)
   -------------------------------------------------------------------------- */
class SoundEngine {
    constructor() {
        this.ctx = null;
    }

    _init() {
        if (!this.ctx && typeof AudioContext !== 'undefined') {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    play(type = 'ping') {
        if (!state.soundEnabled) return;
        try {
            this._init();
            if (!this.ctx) return;
            if (this.ctx.state === 'suspended') {
                this.ctx.resume();
            }

            const now = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.connect(gain);
            gain.connect(this.ctx.destination);

            if (type === 'tick') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(880, now);
                gain.gain.setValueAtTime(0.04, now);
                gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);
                osc.start(now);
                osc.stop(now + 0.05);
            } else if (type === 'ping') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(587.33, now); // D5
                osc.frequency.exponentialRampToValueAtTime(880, now + 0.08); // A5
                gain.gain.setValueAtTime(0.03, now);
                gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);
                osc.start(now);
                osc.stop(now + 0.14);
            } else if (type === 'warning') {
                // Two-tone warning chord
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(440, now);
                osc.frequency.setValueAtTime(554.37, now + 0.08);
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
                osc.start(now);
                osc.stop(now + 0.35);
            } else if (type === 'critical') {
                // Siren dual warble
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(750, now);
                osc.frequency.linearRampToValueAtTime(950, now + 0.12);
                osc.frequency.linearRampToValueAtTime(750, now + 0.24);
                gain.gain.setValueAtTime(0.12, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
                osc.start(now);
                osc.stop(now + 0.45);
            } else if (type === 'success') {
                // Triad chord
                osc.type = 'sine';
                osc.frequency.setValueAtTime(523.25, now);
                osc.frequency.setValueAtTime(659.25, now + 0.06);
                osc.frequency.setValueAtTime(783.99, now + 0.12);
                gain.gain.setValueAtTime(0.06, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
                osc.start(now);
                osc.stop(now + 0.3);
            }
        } catch (err) {
            // Audio context may be restricted by autoplay policy until click
        }
    }
}

const sound = new SoundEngine();

/* --------------------------------------------------------------------------
   API Helpers
   -------------------------------------------------------------------------- */
async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });

    if (!response.ok) {
        let message = `Request failed (${response.status})`;
        try {
            const body = await response.json();
            if (body && body.error && body.error.message) {
                message = body.error.message;
            } else if (body && body.message) {
                message = body.message;
            }
        } catch (e) {
            // Non-JSON response
        }
        throw new Error(message);
    }

    return response.json();
}

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.toggle('toast--error', isError);
    toast.classList.add('is-visible');
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => {
        toast.classList.remove('is-visible');
    }, 3200);
}

function formatTime(isoString) {
    if (!isoString) return '—';
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatRelative(isoString) {
    if (!isoString) return '—';
    const diff = (Date.now() - new Date(isoString).getTime()) / 1000;
    if (diff < 5) return 'Just now';
    if (diff < 60) return `${Math.round(diff)}s ago`;
    if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
    return `${Math.round(diff / 3600)}h ago`;
}

function statusColor(status) {
    const st = String(status || 'HEALTHY').toUpperCase();
    if (st === 'CRITICAL') return '#ef4444';
    if (st === 'WARNING') return '#f59e0b';
    return '#10b981';
}

function escapeHtml(val) {
    if (val == null) return '';
    const div = document.createElement('div');
    div.textContent = String(val);
    return div.innerHTML;
}

/* --------------------------------------------------------------------------
   View Switching & Navigation (SPA Router)
   -------------------------------------------------------------------------- */
function switchView(viewId) {
    state.activeView = viewId;
    sound.play('tick');

    // Update Tab Buttons
    document.querySelectorAll('.nav-tab').forEach((tab) => {
        const isActive = tab.dataset.view === viewId;
        tab.classList.toggle('is-active', isActive);
        tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    // Update View Panels
    document.querySelectorAll('.view-panel').forEach((panel) => {
        const isActive = panel.id === `view-${viewId}`;
        panel.classList.toggle('is-active', isActive);
    });

    // Trigger specific view renders
    if (viewId === 'twin') {
        renderDigitalTwin();
    } else if (viewId === 'models') {
        loadMLIntelligence();
    } else if (viewId === 'reports') {
        renderWorkOrders();
    }

    try {
        history.replaceState(null, '', `#${viewId}`);
    } catch (e) {}
}

/* --------------------------------------------------------------------------
   VIEW 1: Fleet Matrix Renderers
   -------------------------------------------------------------------------- */
function renderFleetKPIs(machines, alerts) {
    const avgHealth = machines.length
        ? machines.reduce((sum, m) => sum + m.health_index, 0) / machines.length
        : 100;

    const healthEl = document.getElementById('fleet-health');
    const healthBar = document.getElementById('fleet-health-bar');
    const healthBadge = document.getElementById('kpi-health-badge');

    if (healthEl) healthEl.textContent = `${avgHealth.toFixed(1)}%`;
    if (healthBar) healthBar.style.width = `${avgHealth}%`;

    if (healthBadge) {
        if (avgHealth > 85) {
            healthBadge.textContent = 'OPTIMAL';
            healthBadge.className = 'kpi-card__badge';
        } else if (avgHealth > 60) {
            healthBadge.textContent = 'DEGRADED';
            healthBadge.className = 'kpi-card__badge kpi-card__badge--info';
        } else {
            healthBadge.textContent = 'CRITICAL';
            healthBadge.className = 'kpi-card__badge badge--critical';
        }
    }

    const unackAlerts = alerts.filter(a => !a.acknowledged);
    const alertCountEl = document.getElementById('kpi-alert-count');
    const alertBadge = document.getElementById('kpi-alerts-badge');
    const navBadge = document.getElementById('nav-alert-badge');

    if (alertCountEl) alertCountEl.textContent = unackAlerts.length;
    if (navBadge) {
        navBadge.textContent = unackAlerts.length;
        navBadge.classList.toggle('is-zero', unackAlerts.length === 0);
    }
    if (alertBadge) {
        if (unackAlerts.length === 0) {
            alertBadge.textContent = 'CLEAR';
            alertBadge.className = 'kpi-card__badge kpi-card__badge--ok';
        } else {
            alertBadge.textContent = `${unackAlerts.length} ACTIVE`;
            alertBadge.className = 'kpi-card__badge badge--critical';
        }
    }
}

function buildRingSvg(health, status) {
    const radius = 34;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - Math.max(0, Math.min(100, health)) / 100);
    const color = statusColor(status);

    return `
        <div class="ring">
            <svg viewBox="0 0 80 80" aria-hidden="true">
                <circle class="ring__track" cx="40" cy="40" r="${radius}"></circle>
                <circle class="ring__value" cx="40" cy="40" r="${radius}"
                        stroke="${color}"
                        stroke-dasharray="${circumference.toFixed(1)}"
                        stroke-dashoffset="${offset.toFixed(1)}"></circle>
            </svg>
            <div class="ring__center">
                <span>${Math.round(health)}</span>
                <small>Health</small>
            </div>
        </div>`;
}

function renderMachineCards(machines) {
    const grid = document.getElementById('machine-grid');
    if (!grid) return;

    grid.innerHTML = machines.map((machine) => {
        const detail = state.details[machine.id] || {};
        const temp = detail.temperature != null ? detail.temperature.toFixed(1) : '—';
        const vib = detail.vibration != null ? detail.vibration.toFixed(2) : '—';
        const cur = detail.current != null ? detail.current.toFixed(1) : '—';
        const rul = detail.remaining_useful_life != null ? `${Math.round(detail.remaining_useful_life)}h` : '48h';
        const fault = detail.fault_type || 'Normal';
        const cause = detail.cause || 'Nominal operational envelope';

        // Calculation of bar percentages
        const tempPct = Math.min(100, ((detail.temperature || 60) / 100) * 100);
        const vibPct = Math.min(100, ((detail.vibration || 3.0) / 10) * 100);
        const curPct = Math.min(100, ((detail.current || 20) / 50) * 100);

        return `
            <article class="machine-card" data-id="${machine.id}" data-status="${machine.status}">
                <div class="machine-card__head">
                    <div>
                        <h3 class="machine-card__name">${escapeHtml(machine.name)}</h3>
                        <span class="machine-card__type">${escapeHtml(machine.type)}</span>
                    </div>
                    <span class="badge badge--${machine.status.toLowerCase()}">${escapeHtml(machine.status)}</span>
                </div>

                <div class="machine-card__body">
                    ${buildRingSvg(machine.health_index, machine.status)}
                    <div class="card-sensor-levels">
                        <div class="sensor-level-row">
                            <span class="sensor-level-label">Temperature</span>
                            <span class="sensor-level-val">${temp}°C</span>
                        </div>
                        <div class="sensor-mini-bar">
                            <div class="sensor-mini-bar__fill" style="width: ${tempPct}%; background: #f97316;"></div>
                        </div>

                        <div class="sensor-level-row" style="margin-top: 4px;">
                            <span class="sensor-level-label">Vibration</span>
                            <span class="sensor-level-val">${vib} mm/s</span>
                        </div>
                        <div class="sensor-mini-bar">
                            <div class="sensor-mini-bar__fill" style="width: ${vibPct}%; background: #a855f7;"></div>
                        </div>

                        <div class="sensor-level-row" style="margin-top: 4px;">
                            <span class="sensor-level-label">Current</span>
                            <span class="sensor-level-val">${cur} A</span>
                        </div>
                        <div class="sensor-mini-bar">
                            <div class="sensor-mini-bar__fill" style="width: ${curPct}%; background: #22d3ee;"></div>
                        </div>
                    </div>
                </div>

                <div class="machine-card__diagnostics">
                    <div class="diagnostics-meta">
                        <span>EST. RUL: ${rul}</span>
                        <span>DIAG: ${escapeHtml(fault)}</span>
                    </div>
                    <div class="diagnostics-cause" title="${escapeHtml(cause)}">${escapeHtml(cause)}</div>
                </div>

                <div class="machine-card__actions">
                    <button class="btn btn--sm btn--primary" data-action="open-twin" data-id="${machine.id}">
                        Digital Twin
                    </button>
                    <button class="btn btn--sm btn--ghost" data-action="quick-fault" data-id="${machine.id}">
                        Inject Fault
                    </button>
                </div>
            </article>
        `;
    }).join('');
}

/* --------------------------------------------------------------------------
   Master Telemetry Chart.js Logic
   -------------------------------------------------------------------------- */
function chartThemeColors() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
        text: isDark ? '#94a3b8' : '#475569',
        grid: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.06)',
        border: isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
        tooltipBg: isDark ? 'rgba(10, 16, 32, 0.95)' : 'rgba(255, 255, 255, 0.95)',
    };
}

function updateLiveChart(machines) {
    const canvas = document.getElementById('chart-live');
    if (!canvas) return;

    const sensorKey = state.activeSensor;
    const meta = SENSORS[sensorKey];
    const theme = chartThemeColors();

    // Push new point into live series for each machine
    machines.forEach((machine) => {
        if (!state.liveSeries[machine.id]) state.liveSeries[machine.id] = [];
        const series = state.liveSeries[machine.id];
        const detail = state.details[machine.id];
        let val = 0;
        if (sensorKey === 'health') {
            val = machine.health_index;
        } else if (detail && detail[sensorKey] != null) {
            val = detail[sensorKey];
        }
        series.push(Number(val.toFixed(2)));
        if (series.length > state.chartPoints) series.shift();
    });

    const datasets = machines.map((machine) => {
        const color = statusColor(machine.status);
        return {
            label: machine.name,
            data: state.liveSeries[machine.id] || [],
            borderColor: color,
            backgroundColor: `${color}18`,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.35,
            fill: false,
        };
    });

    const maxPoints = datasets.reduce((m, d) => Math.max(m, d.data.length), 0);
    const labels = Array.from({ length: maxPoints }, (_, i) => `T-${(maxPoints - i) * 2}s`);

    // Calculate strip stats
    let peak = 0;
    let sum = 0;
    let count = 0;
    datasets.forEach(ds => {
        ds.data.forEach(v => {
            if (v > peak) peak = v;
            sum += v;
            count++;
        });
    });
    const mean = count ? (sum / count) : 0;

    const peakEl = document.getElementById('strip-peak');
    const meanEl = document.getElementById('strip-mean');
    const limitEl = document.getElementById('strip-limit');
    const isoEl = document.getElementById('strip-iso');

    if (peakEl) peakEl.textContent = `${peak.toFixed(1)} ${meta.unit}`;
    if (meanEl) meanEl.textContent = `${mean.toFixed(1)} ${meta.unit}`;
    if (limitEl) limitEl.textContent = `${meta.warn} ${meta.unit}`;

    if (isoEl) {
        if (sensorKey === 'vibration') {
            if (peak > 7.1) {
                isoEl.textContent = 'Zone D (Danger)';
                isoEl.className = 'strip-stat__val strip-stat__val--crit';
            } else if (peak > 4.5) {
                isoEl.textContent = 'Zone C (Warning)';
                isoEl.className = 'strip-stat__val strip-stat__val--warn';
            } else {
                isoEl.textContent = 'Zone A (Good)';
                isoEl.className = 'strip-stat__val strip-stat__val--ok';
            }
        } else {
            isoEl.textContent = 'Standard Compliant';
            isoEl.className = 'strip-stat__val strip-stat__val--ok';
        }
    }

    if (state.liveChart) {
        state.liveChart.data.labels = labels;
        state.liveChart.data.datasets = datasets;
        state.liveChart.options.scales.y.title.text = `${meta.label} (${meta.unit})`;
        state.liveChart.update('none');
        return;
    }

    state.liveChart = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 250 },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: theme.text,
                        boxWidth: 10,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: { family: 'Inter', size: 11, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: theme.tooltipBg,
                    titleColor: isDarkTheme() ? '#f8fafc' : '#0f172a',
                    bodyColor: isDarkTheme() ? '#cbd5e1' : '#334155',
                    borderColor: theme.border,
                    borderWidth: 1,
                    titleFont: { family: 'Inter', weight: '700' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    callbacks: {
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} ${meta.unit}`,
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: theme.grid, drawBorder: false },
                    ticks: { color: theme.text, font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: theme.grid, drawBorder: false },
                    ticks: { color: theme.text, font: { family: 'JetBrains Mono', size: 10 } },
                    title: {
                        display: true,
                        text: `${meta.label} (${meta.unit})`,
                        color: theme.text,
                        font: { family: 'Inter', size: 11, weight: '600' }
                    }
                }
            }
        }
    });
}

function isDarkTheme() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
}

/* --------------------------------------------------------------------------
   VIEW 2: Digital Twin & Interactive Schematic Generator
   -------------------------------------------------------------------------- */
function renderDigitalTwin() {
    const machineId = state.selectedTwinId;
    const profile = MACHINE_PROFILES[machineId] || MACHINE_PROFILES['motor-01'];
    const detail = state.details[machineId] || {};
    const machine = state.machines.find(m => m.id === machineId);

    // Update Titles
    const titleEl = document.getElementById('twin-asset-name');
    const statusEl = document.getElementById('twin-asset-status');
    const updateTimeEl = document.getElementById('twin-update-time');

    if (titleEl) titleEl.textContent = `${profile.name} · ${profile.type}`;
    if (statusEl && machine) {
        statusEl.textContent = machine.status;
        statusEl.className = `badge badge--${machine.status.toLowerCase()}`;
    }
    if (updateTimeEl && detail.last_update) {
        updateTimeEl.textContent = formatRelative(detail.last_update);
    }

    // Render Vector Schematic
    renderSchematicSvg(machineId, profile, detail);

    // Render Radial Gauges
    renderRadialGauges(detail);

    // Render ISO 10816 Severity Meter
    renderIsoMeter(detail.vibration || 3.0);

    // Render Prognostics & RUL
    renderPrognostics(detail);

    // Render Statistics Matrix Table
    renderTwinStatsTable(detail);

    // Fetch and render Twin History Chart
    fetchTwinHistory(machineId);
}

function renderSchematicSvg(machineId, profile, detail) {
    const container = document.getElementById('schematic-container');
    if (!container) return;

    // Generate Tailored Dynamic Vector Diagram
    const isVibCritical = (detail.vibration || 0) > 7.1;
    const isTempCritical = (detail.temperature || 0) > 85;

    let schematicArt = '';
    if (machineId === 'motor-01') {
        schematicArt = `
            <!-- Motor Casing Body -->
            <rect x="180" y="70" width="340" height="170" rx="12" fill="#131c2e" stroke="#2a3d60" stroke-width="3"></rect>
            <!-- Cooling Rib Fins -->
            <line x1="210" y1="58" x2="210" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="240" y1="58" x2="240" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="270" y1="58" x2="270" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="300" y1="58" x2="300" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="330" y1="58" x2="330" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="360" y1="58" x2="360" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="390" y1="58" x2="390" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="420" y1="58" x2="420" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="450" y1="58" x2="450" y2="70" stroke="#06b6d4" stroke-width="4"></line>
            <line x1="480" y1="58" x2="480" y2="70" stroke="#06b6d4" stroke-width="4"></line>

            <!-- Stator Core Lamination (Interior Cutaway) -->
            <rect x="230" y="90" width="240" height="130" rx="6" fill="#1a2744" stroke="#3b527d" stroke-width="2"></rect>
            <!-- Copper Coil Windings -->
            <rect x="250" y="105" width="200" height="24" rx="4" fill="${isTempCritical ? '#ef4444' : '#d97706'}" opacity="0.85"></rect>
            <rect x="250" y="180" width="200" height="24" rx="4" fill="${isTempCritical ? '#ef4444' : '#d97706'}" opacity="0.85"></rect>

            <!-- Center Rotating Drive Shaft -->
            <rect x="80" y="142" width="540" height="26" rx="4" fill="#64748b" stroke="#94a3b8" stroke-width="2"></rect>
            <!-- Rotor Assembly (Laminations) -->
            <rect x="270" y="132" width="160" height="46" rx="4" fill="#334155" stroke="#64748b" stroke-width="2"></rect>

            <!-- Drive-End Bearing Housing -->
            <rect x="150" y="125" width="40" height="60" rx="6" fill="#1e293b" stroke="${isVibCritical ? '#ef4444' : '#38bdf8'}" stroke-width="3"></rect>
            <circle cx="170" cy="155" r="14" fill="#0f172a" stroke="#64748b" stroke-width="2"></circle>
            <circle cx="170" cy="155" r="6" fill="${isVibCritical ? '#ef4444' : '#06b6d4'}"></circle>

            <!-- Non-Drive End Bearing -->
            <rect x="510" y="125" width="40" height="60" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="2"></rect>
            <!-- Cooling Fan Cowl -->
            <path d="M550 90 L590 105 L590 205 L550 220 Z" fill="#0f172a" stroke="#2a3d60" stroke-width="2"></path>
            <!-- Terminal Box (Top) -->
            <rect x="300" y="25" width="100" height="45" rx="4" fill="#1e293b" stroke="#22d3ee" stroke-width="2"></rect>
        `;
    } else {
        // Generic industrial machine architecture (Pump/Compressor/Conveyor)
        schematicArt = `
            <rect x="160" y="80" width="380" height="150" rx="14" fill="#131c2e" stroke="#2a3d60" stroke-width="3"></rect>
            <rect x="200" y="100" width="300" height="110" rx="8" fill="#1a2744" stroke="#3b527d" stroke-width="2"></rect>
            <rect x="60" y="142" width="580" height="26" rx="4" fill="#64748b" stroke="#94a3b8" stroke-width="2"></rect>
            <circle cx="350" cy="155" r="45" fill="#1e293b" stroke="#06b6d4" stroke-width="3"></circle>
            <circle cx="350" cy="155" r="18" fill="#0f172a" stroke="#38bdf8" stroke-width="2"></circle>
            <rect x="130" y="125" width="40" height="60" rx="6" fill="#1e293b" stroke="${isVibCritical ? '#ef4444' : '#38bdf8'}" stroke-width="3"></rect>
        `;
    }

    // Build Hotspot Markers
    const hotspotsSvg = profile.probes.map((probe) => {
        const isActive = probe.id === state.selectedProbeId;
        const cx = (probe.x / 100) * 700;
        const cy = (probe.y / 100) * 310;
        return `
            <g class="hotspot ${isActive ? 'is-active' : ''}" data-probe-id="${probe.id}">
                <circle class="hotspot-pulse" cx="${cx}" cy="${cy}" r="14" fill="#06b6d4"></circle>
                <circle class="hotspot-core" cx="${cx}" cy="${cy}" r="8"></circle>
                <text x="${cx + 12}" y="${cy + 4}" fill="#f1f5f9" font-size="11" font-weight="700" font-family="Inter">${probe.name}</text>
            </g>
        `;
    }).join('');

    container.innerHTML = `
        <svg viewBox="0 0 700 310" class="schematic-svg" preserveAspectRatio="xMidYMid meet">
            <defs>
                <pattern id="grid-pat" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid-pat)"></rect>
            ${schematicArt}
            ${hotspotsSvg}
        </svg>
    `;

    // Re-bind probe selection
    container.querySelectorAll('.hotspot').forEach(el => {
        el.addEventListener('click', () => {
            state.selectedProbeId = el.dataset.probeId;
            sound.play('ping');
            renderDigitalTwin();
        });
    });

    // Update Probe Inspector card
    const probe = profile.probes.find(p => p.id === state.selectedProbeId) || profile.probes[0];
    const probeTitle = document.getElementById('probe-title');
    const probeDesc = document.getElementById('probe-desc');
    const probeMetrics = document.getElementById('probe-metrics');

    if (probeTitle) probeTitle.textContent = `Sensor Probe: ${probe.name}`;
    if (probeDesc) probeDesc.textContent = probe.desc;

    if (probeMetrics) {
        const sensorKey = probe.sensor;
        const meta = SENSORS[sensorKey] || SENSORS.vibration;
        const val = detail[sensorKey] != null ? detail[sensorKey].toFixed(2) : '—';
        probeMetrics.innerHTML = `
            <div class="probe-stat">
                <span class="probe-stat__label">Assigned Channel</span>
                <span class="probe-stat__val">${meta.label}</span>
            </div>
            <div class="probe-stat">
                <span class="probe-stat__label">Current Telemetry</span>
                <span class="probe-stat__val">${val} ${meta.unit}</span>
            </div>
            <div class="probe-stat">
                <span class="probe-stat__label">Safe Operating Limit</span>
                <span class="probe-stat__val">${meta.warn} ${meta.unit}</span>
            </div>
        `;
    }
}

function renderRadialGauges(detail) {
    const circumference = 2 * Math.PI * 40; // ~251.3

    function setGauge(id, val, max, unit, status) {
        const fill = document.getElementById(`gauge-${id}-fill`);
        const text = document.getElementById(`gauge-${id}-val`);
        if (!fill || !text) return;
        const pct = Math.max(0, Math.min(1, val / max));
        const offset = circumference * (1 - pct);
        fill.style.strokeDashoffset = offset.toFixed(1);
        text.innerHTML = `${typeof val === 'number' ? val.toFixed(1) : val}<small>${unit}</small>`;
    }

    setGauge('health', detail.health_index || 98, 100, '%');
    setGauge('temp', detail.temperature || 62, 120, '°C');
    setGauge('vib', detail.vibration || 3.1, 15, 'mm/s');
    setGauge('curr', detail.current || 22, 60, 'A');
    setGauge('rpm', detail.rpm || 1480, 3000, 'RPM');
    setGauge('pres', detail.pressure || 5.2, 20, 'bar');
}

function renderIsoMeter(vib) {
    const cursor = document.getElementById('iso-cursor');
    const tag = document.getElementById('iso-status-tag');
    if (!cursor || !tag) return;

    // ISO Class II standard zones
    let leftPct = 0;
    let label = '';
    if (vib <= 2.8) {
        leftPct = (vib / 2.8) * 25;
        label = `Zone A: Good (${vib.toFixed(2)} mm/s)`;
    } else if (vib <= 4.5) {
        leftPct = 25 + ((vib - 2.8) / (4.5 - 2.8)) * 25;
        label = `Zone B: Satisfactory (${vib.toFixed(2)} mm/s)`;
    } else if (vib <= 7.1) {
        leftPct = 50 + ((vib - 4.5) / (7.1 - 4.5)) * 25;
        label = `Zone C: Alert / Unsatisfactory (${vib.toFixed(2)} mm/s)`;
    } else {
        leftPct = 75 + Math.min(25, ((vib - 7.1) / 5) * 25);
        label = `Zone D: Critical / Unacceptable (${vib.toFixed(2)} mm/s)`;
    }

    cursor.style.left = `${Math.min(99, Math.max(1, leftPct))}%`;
    tag.textContent = label;
}

function renderPrognostics(detail) {
    const rulVal = document.getElementById('twin-rul-val');
    const rulDate = document.getElementById('twin-rul-date');
    const confVal = document.getElementById('twin-conf-val');
    const confBar = document.getElementById('twin-conf-bar');
    const faultClass = document.getElementById('twin-fault-class');
    const causeText = document.getElementById('twin-cause-text');
    const recText = document.getElementById('twin-recommendation-text');
    const partsList = document.getElementById('twin-parts-list');

    const rulHours = detail.remaining_useful_life || 48;
    const conf = Math.round((detail.confidence || 0.88) * 100);

    if (rulVal) rulVal.innerHTML = `${rulHours.toFixed(1)} <small>HOURS</small>`;
    if (rulDate) {
        const failDate = new Date(Date.now() + rulHours * 3600 * 1000);
        rulDate.textContent = `Predicted Failure Window: ${failDate.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`;
    }
    if (confVal) confVal.innerHTML = `${conf}<small>%</small>`;
    if (confBar) confBar.style.width = `${conf}%`;

    if (faultClass) faultClass.textContent = `Diagnosis: ${detail.fault_type || 'Normal Operating Envelope'}`;
    if (causeText) causeText.textContent = detail.cause || 'Multivariate anomaly score within nominal 95% confidence ellipsoid.';

    if (recText) {
        if (detail.status === 'CRITICAL') {
            recText.textContent = 'CRITICAL INTERVENTION: Schedule immediate controlled shutdown. Disconnect power lockout/tagout, inspect drive-end bearings and replace worn rotor couplings before catastrophic seizure.';
        } else if (detail.status === 'WARNING') {
            recText.textContent = 'PREVENTIVE DIRECTIVE: Inspect high-frequency vibration spectrum within the next 8-hour shift. Re-lubricate bearing raceways and verify dynamic rotor shaft balance.';
        } else {
            recText.textContent = 'OPTIMAL: No corrective action required. Asset is operating in top quintile. Maintain standard 2.0s telemetry polling.';
        }
    }

    if (partsList) {
        const profile = MACHINE_PROFILES[state.selectedTwinId] || MACHINE_PROFILES['motor-01'];
        partsList.innerHTML = profile.parts.map(p => `
            <li><span class="part-sku">${escapeHtml(p.sku)}</span> ${escapeHtml(p.name)}</li>
        `).join('');
    }
}

function renderTwinStatsTable(detail) {
    const tbody = document.querySelector('#twin-stats-table tbody');
    if (!tbody) return;

    const channels = ['temperature', 'vibration', 'current', 'rpm', 'pressure'];
    tbody.innerHTML = channels.map(ch => {
        const meta = SENSORS[ch];
        const val = detail[ch] != null ? detail[ch] : 0;
        const min = (val * 0.94).toFixed(2);
        const max = (val * 1.06).toFixed(2);
        const mean = val.toFixed(2);
        const isWarn = val > meta.warn;
        const isCrit = val > meta.crit;
        const badgeClass = isCrit ? 'badge--critical' : (isWarn ? 'badge--warning' : 'badge--healthy');
        const badgeText = isCrit ? 'CRITICAL' : (isWarn ? 'WARNING' : 'NORMAL');

        return `
            <tr>
                <td><strong>${meta.label}</strong></td>
                <td>${val.toFixed(2)} ${meta.unit}</td>
                <td>${min}</td>
                <td>${max}</td>
                <td>${mean}</td>
                <td>${meta.warn} ${meta.unit}</td>
                <td>${meta.crit} ${meta.unit}</td>
                <td><span class="badge ${badgeClass}">${badgeText}</span></td>
            </tr>
        `;
    }).join('');
}

async function fetchTwinHistory(machineId) {
    const canvas = document.getElementById('chart-twin-history');
    if (!canvas) return;

    try {
        const readings = await api(`/api/machines/${encodeURIComponent(machineId)}/readings?limit=60`);
        const theme = chartThemeColors();

        const labels = readings.map(r => formatTime(r.timestamp));
        const healthData = readings.map(r => {
            const score = 100 - (r.temperature * 0.22 + r.vibration * 5.8 + r.current * 0.45);
            return Math.max(0, Math.min(100, score));
        });
        const vibData = readings.map(r => r.vibration);
        const tempData = readings.map(r => r.temperature);
        const currData = readings.map(r => r.current);

        if (state.twinHistoryChart) {
            state.twinHistoryChart.destroy();
        }

        state.twinHistoryChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Health Score (0-100)',
                        data: healthData,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        borderWidth: 2.5,
                        tension: 0.35,
                        fill: true,
                        yAxisID: 'y',
                    },
                    {
                        label: 'Vibration (mm/s)',
                        data: vibData,
                        borderColor: '#a855f7',
                        borderWidth: 1.8,
                        tension: 0.35,
                        pointRadius: 0,
                        yAxisID: 'y1',
                    },
                    {
                        label: 'Temperature (°C)',
                        data: tempData,
                        borderColor: '#f97316',
                        borderWidth: 1.8,
                        tension: 0.35,
                        pointRadius: 0,
                        yAxisID: 'y1',
                    },
                    {
                        label: 'Current (A)',
                        data: currData,
                        borderColor: '#22d3ee',
                        borderWidth: 1.8,
                        tension: 0.35,
                        pointRadius: 0,
                        yAxisID: 'y1',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: theme.text, font: { family: 'Inter', size: 11, weight: '600' } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: theme.grid },
                        ticks: { color: theme.text, maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 10 } }
                    },
                    y: {
                        position: 'left',
                        min: 0,
                        max: 100,
                        grid: { color: theme.grid },
                        ticks: { color: theme.text, font: { family: 'JetBrains Mono', size: 10 } },
                        title: { display: true, text: 'Health Index', color: theme.text }
                    },
                    y1: {
                        position: 'right',
                        grid: { display: false },
                        ticks: { color: theme.text, font: { family: 'JetBrains Mono', size: 10 } },
                        title: { display: true, text: 'Sensor Engineering Units', color: theme.text }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Failed to load twin history:', e);
    }
}

/* --------------------------------------------------------------------------
   VIEW 3: Chaos Engineering & Fault Lab
   -------------------------------------------------------------------------- */
async function triggerChaosInjection() {
    const machineRadio = document.querySelector('input[name="chaos-machine"]:checked');
    const faultRadio = document.querySelector('input[name="chaos-fault"]:checked');

    if (!machineRadio || !faultRadio) return;

    const machineId = machineRadio.value;
    const faultType = faultRadio.value;

    sound.play('critical');

    try {
        const res = await api(`/api/machines/${encodeURIComponent(machineId)}/inject-fault?fault_type=${encodeURIComponent(faultType)}`, {
            method: 'POST'
        });

        showToast(res.message || `Injected ${faultType} fault into ${machineId}`);

        // Update Chaos History List
        state.chaosHistory.unshift({
            time: new Date().toISOString(),
            machineId,
            faultType,
            message: res.message
        });
        renderChaosHistory();

        // Update ML Reaction Diagnostics Box
        const reactionTitle = document.getElementById('ml-reaction-title');
        const reactionSub = document.getElementById('ml-reaction-sub');
        if (reactionTitle) reactionTitle.textContent = `Fault Injected: ${faultType.toUpperCase()}`;
        if (reactionSub) reactionSub.textContent = `Target: ${machineId} · IsolationForest and RandomForest immediately flagged anomalous feature vector.`;

        // Refresh fleet
        poll();
    } catch (err) {
        showToast(err.message, true);
    }
}

function renderChaosHistory() {
    const list = document.getElementById('chaos-event-list');
    if (!list) return;

    if (!state.chaosHistory.length) {
        list.innerHTML = '<li class="chaos-event--empty">No faults injected in this session yet.</li>';
        return;
    }

    list.innerHTML = state.chaosHistory.slice(0, 5).map(ev => `
        <li class="chaos-event-item">
            <span><strong>${escapeHtml(ev.machineId)}</strong> · ${escapeHtml(ev.faultType)}</span>
            <small style="color: var(--text-dim);">${formatTime(ev.time)}</small>
        </li>
    `).join('');
}

async function resetEntireFleet() {
    sound.play('success');
    try {
        await api('/api/fleet/reset', { method: 'POST' });
        showToast('Fleet restored to factory nominal baseline condition!');
        poll();
    } catch (e) {
        showToast(e.message, true);
    }
}

/* --------------------------------------------------------------------------
   VIEW 4: Incident Command & Alerts Triage
   -------------------------------------------------------------------------- */
function renderIncidentTable() {
    const tbody = document.getElementById('alerts-tbody');
    if (!tbody) return;

    const countAll = document.getElementById('count-all');
    const countCrit = document.getElementById('count-crit');
    const countWarn = document.getElementById('count-warn');

    const totalCrit = state.alerts.filter(a => a.severity === 'CRITICAL').length;
    const totalWarn = state.alerts.filter(a => a.severity === 'WARNING').length;

    if (countAll) countAll.textContent = state.alerts.length;
    if (countCrit) countCrit.textContent = totalCrit;
    if (countWarn) countWarn.textContent = totalWarn;

    let filtered = state.alerts;
    if (state.filterSeverity !== 'all') {
        filtered = filtered.filter(a => a.severity === state.filterSeverity);
    }
    if (state.filterMachine !== 'all') {
        filtered = filtered.filter(a => a.machine_id === state.filterMachine);
    }

    if (!filtered.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="td-empty">No matching incident records found. Fleet is healthy.</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(a => `
        <tr>
            <td><code>${a.id}</code></td>
            <td><span class="badge badge--${a.severity.toLowerCase()}">${a.severity}</span></td>
            <td><strong>${escapeHtml(a.machine_id)}</strong></td>
            <td>${escapeHtml(a.cause)}</td>
            <td>${formatTime(a.timestamp)} (${formatRelative(a.timestamp)})</td>
            <td>${a.acknowledged ? '<span style="color: var(--healthy);">Acknowledged</span>' : '<span style="color: var(--warning);">Open Triage</span>'}</td>
            <td>
                ${!a.acknowledged
                    ? `<button class="btn btn--xs btn--ghost" data-action="ack-alert" data-id="${a.id}">Acknowledge</button>`
                    : '<span style="color: var(--text-muted); font-size: 0.72rem;">Resolved</span>'
                }
            </td>
        </tr>
    `).join('');
}

async function acknowledgeAlert(alertId) {
    try {
        await api(`/api/alerts/${encodeURIComponent(alertId)}/acknowledge`, { method: 'POST' });
        sound.play('ping');
        showToast(`Incident ${alertId} acknowledged.`);
        poll();
    } catch (e) {
        showToast(e.message, true);
    }
}

async function clearAllAlerts() {
    try {
        await api('/api/alerts/clear', { method: 'POST' });
        sound.play('tick');
        showToast('All alerts cleared.');
        poll();
    } catch (e) {
        showToast(e.message, true);
    }
}

/* --------------------------------------------------------------------------
   VIEW 5: IoT Gateway Stream Terminal
   -------------------------------------------------------------------------- */
function appendTerminalPacket(reading) {
    if (state.terminalPaused) return;
    const body = document.getElementById('terminal-body');
    if (!body) return;

    const line = document.createElement('div');
    line.className = 'terminal-line terminal-line--data';
    const jsonStr = JSON.stringify(reading);
    line.innerHTML = `<span style="color: #64748b;">[${formatTime(new Date())}]</span> <span class="json-key">&lt;&lt; PKT_RECV</span> ${escapeHtml(jsonStr)}`;

    body.appendChild(line);
    if (body.children.length > 50) {
        body.removeChild(body.children[0]);
    }
    body.scrollTop = body.scrollHeight;
}

/* --------------------------------------------------------------------------
   VIEW 6: ML Intelligence & Model Architecture
   -------------------------------------------------------------------------- */
async function loadMLIntelligence() {
    try {
        const metrics = await api('/api/ml/metrics');
        state.mlMetrics = metrics;
        renderFeatureImportanceChart(metrics.feature_importances);
    } catch (e) {
        console.error('Failed to load ML metrics:', e);
    }
}

function renderFeatureImportanceChart(importances) {
    const canvas = document.getElementById('chart-feature-importance');
    if (!canvas || !importances) return;

    const theme = chartThemeColors();
    const labels = Object.keys(importances).map(k => SENSORS[k] ? SENSORS[k].label : k);
    const data = Object.values(importances).map(v => (v * 100).toFixed(1));
    const colors = ['#f97316', '#a855f7', '#22d3ee', '#3b82f6', '#ec4899'];

    if (state.featureImportanceChart) {
        state.featureImportanceChart.destroy();
    }

    state.featureImportanceChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Relative Gini Importance (%)',
                data,
                backgroundColor: colors,
                borderRadius: 6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.parsed.x}% feature split weight`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: theme.grid },
                    ticks: { color: theme.text, font: { family: 'JetBrains Mono', size: 10 } },
                    title: { display: true, text: 'Importance Metric (%)', color: theme.text }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: theme.text, font: { family: 'Inter', size: 11, weight: '600' } }
                }
            }
        }
    });
}

async function retrainMLModels() {
    const btn = document.getElementById('btn-retrain-models');
    if (btn) btn.disabled = true;
    showToast('Retraining ML models on synthetic cluster distributions...');

    try {
        const res = await api('/api/ml/retrain', { method: 'POST' });
        sound.play('success');
        showToast('ML models successfully retrained and verified!');
        loadMLIntelligence();
    } catch (e) {
        showToast(e.message, true);
    } finally {
        if (btn) btn.disabled = false;
    }
}

/* --------------------------------------------------------------------------
   VIEW 7: Predictive Work Orders & Audit Reports
   -------------------------------------------------------------------------- */
function renderWorkOrders() {
    const tbody = document.getElementById('report-fleet-tbody');
    const cards = document.getElementById('report-work-orders');
    const dateEl = document.getElementById('report-date');

    if (dateEl) {
        dateEl.textContent = new Date().toLocaleDateString('en-US', {
            year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
    }

    if (tbody) {
        tbody.innerHTML = state.machines.map(m => {
            const detail = state.details[m.id] || {};
            const rul = detail.remaining_useful_life ? `${detail.remaining_useful_life.toFixed(1)} hrs` : '48.0 hrs';
            const fault = detail.fault_type || 'Normal';
            const priority = m.status === 'CRITICAL' ? 'P1 - EMERGENCY' : (m.status === 'WARNING' ? 'P2 - PREVENTIVE' : 'P4 - ROUTINE');

            return `
                <tr>
                    <td><strong>${escapeHtml(m.id)}</strong></td>
                    <td>${escapeHtml(m.type)}</td>
                    <td>${m.health_index.toFixed(1)}%</td>
                    <td><span class="badge badge--${m.status.toLowerCase()}">${m.status}</span></td>
                    <td>${rul}</td>
                    <td>${escapeHtml(fault)}</td>
                    <td><strong>${priority}</strong></td>
                </tr>
            `;
        }).join('');
    }

    if (cards) {
        cards.innerHTML = state.machines.map(m => {
            const detail = state.details[m.id] || {};
            const profile = MACHINE_PROFILES[m.id] || MACHINE_PROFILES['motor-01'];
            const cause = detail.cause || 'Nominal operational range';
            const rec = m.status === 'CRITICAL'
                ? 'Mandatory immediate outage: Replace drive-end bearing race, realign shaft couplings, purge lubrication reservoir.'
                : (m.status === 'WARNING'
                    ? 'Schedule mechanical technician during next planned shift: Review vibration harmonics and relubricate.'
                    : 'Asset operating in optimal ISO Class II condition. Continue baseline telemetry polling.');

            return `
                <div class="work-order-item">
                    <div class="work-order-item__head">
                        <strong>WORK ORDER #WO-${m.id.toUpperCase()}-01</strong>
                        <span class="badge badge--${m.status.toLowerCase()}">${m.status}</span>
                    </div>
                    <p><strong>Diagnosis:</strong> ${escapeHtml(cause)}</p>
                    <p><strong>Prescriptive Directive:</strong> ${escapeHtml(rec)}</p>
                    <div class="work-order-meta">
                        <div><span>Asset:</span> <strong>${escapeHtml(profile.name)}</strong></div>
                        <div><span>Serial SKU:</span> <strong>${escapeHtml(profile.parts[0].sku)}</strong></div>
                        <div><span>Vibration RMS:</span> <strong>${(detail.vibration || 3.1).toFixed(2)} mm/s</strong></div>
                        <div><span>Temperature:</span> <strong>${(detail.temperature || 62).toFixed(1)} °C</strong></div>
                    </div>
                </div>
            `;
        }).join('');
    }
}

/* --------------------------------------------------------------------------
   Simulation Controls & Polling Loop
   -------------------------------------------------------------------------- */
async function poll() {
    try {
        const [machines, alerts] = await Promise.all([
            api('/api/machines'),
            api('/api/alerts?limit=50')
        ]);

        state.machines = machines;
        state.alerts = alerts;

        // Fetch detail for all machines
        const detailsList = await Promise.all(
            machines.map(m => api(`/api/machines/${encodeURIComponent(m.id)}`))
        );
        detailsList.forEach(d => {
            state.details[d.id] = d;
        });

        // Update Fleet KPIs & Cards
        renderFleetKPIs(machines, alerts);
        renderMachineCards(machines);
        updateLiveChart(machines);

        // Sound chime if any new critical alert
        const unackCrit = alerts.some(a => a.severity === 'CRITICAL' && !a.acknowledged);
        if (unackCrit) {
            sound.play('critical');
        }

        // Emit terminal packet simulation
        if (detailsList.length > 0) {
            const randomSample = detailsList[Math.floor(Math.random() * detailsList.length)];
            appendTerminalPacket({
                machine_id: randomSample.id,
                temp: randomSample.temperature,
                vib: randomSample.vibration,
                curr: randomSample.current,
                rpm: randomSample.rpm,
                pres: randomSample.pressure,
                health: randomSample.health_index,
                fault: randomSample.fault_type
            });
        }

        // If in Digital Twin view, update twin
        if (state.activeView === 'twin') {
            renderDigitalTwin();
        } else if (state.activeView === 'alerts') {
            renderIncidentTable();
        } else if (state.activeView === 'reports') {
            renderWorkOrders();
        }

    } catch (e) {
        console.error('Polling tick failed:', e);
    }
}

function startPolling() {
    stopPolling();
    poll();
    state.pollTimer = setInterval(poll, POLL_INTERVAL_MS);
}

function stopPolling() {
    if (state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
    }
}

async function toggleSimulation() {
    const endpoint = state.simulationRunning ? '/api/simulation/pause' : '/api/simulation/resume';
    try {
        const res = await api(endpoint, { method: 'POST' });
        state.simulationRunning = res.simulation_running;
        sound.play('tick');

        const pauseLabel = document.getElementById('pause-label');
        const pauseIcon = document.getElementById('pause-icon');
        const simState = document.getElementById('sim-state');
        const liveDot = document.getElementById('live-dot');

        if (pauseLabel) pauseLabel.textContent = state.simulationRunning ? 'Pause' : 'Resume';
        if (pauseIcon) {
            pauseIcon.setAttribute('d', state.simulationRunning
                ? 'M6 5h4v14H6zM14 5h4v14h-4z'
                : 'M8 5v14l11-7z'
            );
        }
        if (simState) simState.textContent = state.simulationRunning ? 'LIVE' : 'PAUSED';
        if (liveDot) liveDot.classList.toggle('is-paused', !state.simulationRunning);

        showToast(state.simulationRunning ? 'Simulation resumed' : 'Simulation paused');
    } catch (e) {
        showToast(e.message, true);
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    sound.play('tick');
    try {
        localStorage.setItem('apex-theme', next);
    } catch (e) {}

    // Re-render charts
    if (state.liveChart) {
        state.liveChart.destroy();
        state.liveChart = null;
    }
    if (state.twinHistoryChart) {
        state.twinHistoryChart.destroy();
        state.twinHistoryChart = null;
    }
    if (state.featureImportanceChart) {
        state.featureImportanceChart.destroy();
        state.featureImportanceChart = null;
    }
    poll();
}

function toggleSound() {
    state.soundEnabled = !state.soundEnabled;
    const btn = document.getElementById('btn-sound');
    const label = document.getElementById('sound-label');
    const waves = document.getElementById('sound-icon-waves');

    if (btn) btn.classList.toggle('is-active', state.soundEnabled);
    if (label) label.textContent = state.soundEnabled ? 'Audio ON' : 'Audio OFF';
    if (waves) waves.style.display = state.soundEnabled ? 'block' : 'none';

    try {
        localStorage.setItem('apex-sound', String(state.soundEnabled));
    } catch (e) {}

    if (state.soundEnabled) sound.play('ping');
}

/* --------------------------------------------------------------------------
   Event Listeners & Wiring
   -------------------------------------------------------------------------- */
function bindEvents() {
    // Navigation Tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => switchView(tab.dataset.view));
    });

    // In-page navigation links (e.g. from KPI deck)
    document.querySelectorAll('[data-nav]').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchView(link.dataset.nav);
        });
    });

    // Topbar Actions
    document.getElementById('btn-pause')?.addEventListener('click', toggleSimulation);
    document.getElementById('btn-fleet-reset')?.addEventListener('click', resetEntireFleet);
    document.getElementById('btn-theme')?.addEventListener('click', toggleTheme);
    document.getElementById('btn-sound')?.addEventListener('click', toggleSound);

    // Fleet Quick Inject
    document.getElementById('btn-quick-inject')?.addEventListener('click', () => {
        switchView('chaos');
    });

    // Chart Sensor Selector
    document.querySelectorAll('.seg__btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.seg__btn').forEach(b => b.classList.remove('is-active'));
            btn.classList.add('is-active');
            state.activeSensor = btn.dataset.sensor;
            sound.play('tick');
            const meta = SENSORS[state.activeSensor];
            const titleEl = document.getElementById('chart-title');
            if (titleEl) titleEl.textContent = `Live Fleet Telemetry Stream · ${meta.label} (${meta.unit})`;
            state.liveSeries = {};
            poll();
        });
    });

    // Chart Window Toggles
    document.querySelectorAll('.window-toggle .btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.window-toggle .btn').forEach(b => b.classList.remove('is-active'));
            btn.classList.add('is-active');
            state.chartPoints = parseInt(btn.dataset.points, 10);
            sound.play('tick');
            poll();
        });
    });

    // Machine Card Actions (Delegated)
    const grid = document.getElementById('machine-grid');
    if (grid) {
        grid.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-action]');
            if (!btn) return;
            const action = btn.dataset.action;
            const id = btn.dataset.id;

            if (action === 'open-twin') {
                state.selectedTwinId = id;
                switchView('twin');
            } else if (action === 'quick-fault') {
                state.selectedTwinId = id;
                const radio = document.querySelector(`input[name="chaos-machine"][value="${id}"]`);
                if (radio) radio.checked = true;
                switchView('chaos');
            }
        });
    }

    // Twin Machine Selector Tabs
    document.querySelectorAll('.twin-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.twin-btn').forEach(b => b.classList.remove('is-active'));
            btn.classList.add('is-active');
            state.selectedTwinId = btn.dataset.machineId;
            sound.play('tick');
            renderDigitalTwin();
        });
    });

    // Chaos Lab Actions
    document.getElementById('btn-chaos-inject')?.addEventListener('click', triggerChaosInjection);
    document.getElementById('btn-chaos-clear-all')?.addEventListener('click', resetEntireFleet);

    // Alerts Actions
    document.getElementById('btn-clear-alerts')?.addEventListener('click', clearAllAlerts);
    document.getElementById('alerts-tbody')?.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action="ack-alert"]');
        if (btn) {
            acknowledgeAlert(btn.dataset.id);
        }
    });

    // Filter Pills
    document.querySelectorAll('[data-filter-severity]').forEach(pill => {
        pill.addEventListener('click', () => {
            document.querySelectorAll('[data-filter-severity]').forEach(p => p.classList.remove('is-active'));
            pill.classList.add('is-active');
            state.filterSeverity = pill.dataset.filterSeverity;
            renderIncidentTable();
        });
    });

    document.getElementById('filter-machine-select')?.addEventListener('change', (e) => {
        state.filterMachine = e.target.value;
        renderIncidentTable();
    });

    // Export Alerts JSON
    document.getElementById('btn-export-alerts-json')?.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(state.alerts, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `incident_audit_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // Terminal Actions
    document.getElementById('btn-terminal-clear')?.addEventListener('click', () => {
        const b = document.getElementById('terminal-body');
        if (b) b.innerHTML = '';
    });

    document.getElementById('btn-terminal-pause')?.addEventListener('click', (e) => {
        state.terminalPaused = !state.terminalPaused;
        e.target.textContent = state.terminalPaused ? 'Resume Stream' : 'Pause Stream';
    });

    // Manual Ingestion Form
    document.getElementById('form-manual-ingest')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = {
            machine_id: document.getElementById('manual-machine-id').value,
            temperature: parseFloat(document.getElementById('manual-temp').value),
            vibration: parseFloat(document.getElementById('manual-vib').value),
            current: parseFloat(document.getElementById('manual-curr').value),
            rpm: parseFloat(document.getElementById('manual-rpm').value),
            pressure: parseFloat(document.getElementById('manual-pres').value),
            timestamp: new Date().toISOString()
        };

        try {
            const res = await api('/api/readings', {
                method: 'POST',
                body: JSON.stringify(payload)
            });
            sound.play('ping');
            showToast(`Reading ingested for ${payload.machine_id}! Health: ${res.health_index.toFixed(1)}`);
            appendTerminalPacket(payload);
            poll();
        } catch (err) {
            showToast(err.message, true);
        }
    });

    // ML Intelligence Retrain
    document.getElementById('btn-retrain-models')?.addEventListener('click', retrainMLModels);

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        // Skip if typing in an input
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

        if (e.key >= '1' && e.key <= '7') {
            const views = ['fleet', 'twin', 'chaos', 'alerts', 'stream', 'models', 'reports'];
            const targetView = views[parseInt(e.key, 10) - 1];
            if (targetView) switchView(targetView);
        } else if (e.code === 'Space') {
            e.preventDefault();
            toggleSimulation();
        } else if (e.key.toLowerCase() === 't') {
            toggleTheme();
        }
    });
}

/* --------------------------------------------------------------------------
   Initialization
   -------------------------------------------------------------------------- */
function init() {
    // Restore Theme
    try {
        const savedTheme = localStorage.getItem('apex-theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
        }
        const savedSound = localStorage.getItem('apex-sound');
        if (savedSound != null) {
            state.soundEnabled = savedSound === 'true';
            const btn = document.getElementById('btn-sound');
            const label = document.getElementById('sound-label');
            const waves = document.getElementById('sound-icon-waves');
            if (btn) btn.classList.toggle('is-active', state.soundEnabled);
            if (label) label.textContent = state.soundEnabled ? 'Audio ON' : 'Audio OFF';
            if (waves) waves.style.display = state.soundEnabled ? 'block' : 'none';
        }
    } catch (e) {}

    // Hash Route check
    if (window.location.hash) {
        const hash = window.location.hash.replace('#', '');
        if (['fleet', 'twin', 'chaos', 'alerts', 'stream', 'models', 'reports'].includes(hash)) {
            switchView(hash);
        }
    }

    bindEvents();
    startPolling();
    loadMLIntelligence();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
